"""Run the actual six-joint X-VLA policy in full-contact MuJoCo, never hardware."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from examples.box_dataset_pilot import apply_camera_perturbations
from examples.box_pickup import Pickup, camera, load_recording, load_scene, options, save_run, scene_model, validate
from examples.box_training import FPS, SIZE, render_camera_frames
from sourccey.build import ROOT
from sourccey.rendering import create_renderer
from sourccey.sandbox import copy_display_state
from sourccey.xvla_eval import JOINT_NAMES, JointMapping, LiftTracker, PolicyWorker
from sourccey.box_task import PICKUP_TASK, PLACE_TASK, PlacementTracker, placement_observation


MODEL = ROOT/"xvla_sourccey_box_sim_b4/checkpoints/020000/pretrained_model"
SOURCE = ROOT/"artifacts/box_xvla_training_handoff_5090/lerobot-vulcan"
PYTHON = ROOT.parents[1]/"VulcanLerobot/lerobot-vulcan/.venv/Scripts/python.exe"
PILOT = ROOT/"artifacts/box_xvla_pilot"
OUTPUT = ROOT/"artifacts/box_policy_eval"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_pair(config):
    full, measured = scene_model(config)
    display, _ = scene_model(config, display_only=True)
    if config.get("domain_randomization", {}).get("camera_perturbations"):
        apply_camera_perturbations(full, config)
        apply_camera_perturbations(display, config)
    return full, display, measured


def evaluate(args, shared_policy=None):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene_path = args.scene or PILOT/args.episode/"scene.json"
    config = read_json(scene_path)
    if "object" not in config:
        config = load_scene(scene_path)
    validate(config)
    (output/"scene.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    mapping_path = ROOT/"artifacts/robot_joint_mapping.json"
    metadata = read_json(scene_path.parent/"episode.json") if (scene_path.parent/"episode.json").exists() else {}
    if metadata.get("mapping_sha256") and metadata["mapping_sha256"] != hashlib.sha256(mapping_path.read_bytes()).hexdigest():
        raise ValueError("Joint mapping changed since these demonstrations were generated")
    if metadata.get("names") and metadata["names"] != [name+".pos" for name in JOINT_NAMES]:
        raise ValueError("Demonstration channel order differs from the policy adapter")
    mapping = JointMapping(mapping_path)
    full, display, measured = model_pair(config)
    run = Pickup(config, model=full, measured=measured)
    run.phase = "learned-policy"
    run.frames = [(float(run.d.time), run.d.qpos.copy(), run.phase)]
    data = mujoco.MjData(display)
    tracker = LiftTracker(config["success"], measured["tabletop_z_m"])
    task = config.get("dataset_pilot", {}).get("instruction", PICKUP_TASK)
    if task not in (PICKUP_TASK, PLACE_TASK) or (task == PLACE_TASK) != ("placement" in config):
        raise ValueError("Scene task and placement contract disagree")
    placement = (PlacementTracker(config["placement"], measured["near_edge_x_m"], measured["tabletop_z_m"])
                 if task == PLACE_TASK else None)
    def completed():
        return placement.stable_placement is not None if placement is not None else tracker.stable_hold is not None
    self_contacts, floor_contacts = {}, {}
    force = np.zeros(6)
    threshold = config.get("workspace", {}).get("thresholds", {}).get("maximum_self_contact_force_n", .1)

    def observe(r):
        tracker.observe(float(r.d.time), r.bottom(), r.d.xpos[r.box], r.fingers())
        if placement is not None:
            placement.observe(float(r.d.time), **placement_observation(r), lifted=tracker.stable_hold is not None)
        for index, contact in enumerate(r.d.contact):
            ga, gb = int(contact.geom1), int(contact.geom2)
            ba, bb = int(full.geom_bodyid[ga]), int(full.geom_bodyid[gb])
            if r.box in (ba, bb) or contact.efc_address < 0:
                continue
            if ba == 0 or bb == 0:
                world, robot = (ga, gb) if ba == 0 else (gb, ga)
                if full.geom(world).name != "floor" or full.geom(robot).name.startswith("contact_"):
                    continue
                target = floor_contacts
            else:
                target = self_contacts
            mujoco.mj_contactForce(full, r.d, index, force)
            if force[0] > threshold:
                pair = " / ".join(sorted((full.geom(ga).name, full.geom(gb).name)))
                target[pair] = max(target.get(pair, 0.), float(force[0]))

    run.on_step = observe
    qp = [full.joint(name).qposadr[0] for name in JOINT_NAMES]
    joint_limits = np.array([full.joint(name).range for name in JOINT_NAMES])
    records, latency = [], []
    clipped = np.zeros(6, dtype=int)
    started = time.perf_counter()
    error = None
    ready = {}
    try:
        if shared_policy is None:
            print("Loading the trained X-VLA checkpoint (simulation only)...", flush=True)
        context = (nullcontext(shared_policy) if shared_policy is not None else
                   PolicyWorker(args.policy_python, args.checkpoint, args.lerobot_source, output,
                                args.device, args.seed, task=task))
        with context as policy:
            if policy.task != task:
                raise ValueError("Shared policy worker task differs from the scene")
            ready = policy.ready
            policy.reset(args.seed, output)
            if shared_policy is None:
                print(f"Loaded on {ready['gpu'] or ready['device']}; {ready['load_wall_s']:.1f}s", flush=True)
            if not 1 <= args.horizon <= ready["chunk_size"]:
                raise ValueError("Execution horizon must fit the predicted chunk")
            with create_renderer(display, width=SIZE[0], height=SIZE[1], backend="filament") as renderer:
                seconds = args.seconds if args.seconds is not None else (36. if placement is not None else 24.)
                total_frames = round(seconds*FPS)
                while len(records) < total_frames and not completed():
                    copy_display_state(run.d, display, data)
                    images = render_camera_frames(display, data, renderer, config, len(records))
                    state = mapping.state(run.d.qpos[qp])
                    actions, duration = policy.predict(state, images)
                    latency.append(duration)
                    for action in actions[:min(args.horizon, total_frames-len(records))]:
                        radians, limited = mapping.command(action, joint_limits)
                        clipped += limited
                        records.append((float(run.d.time), mapping.state(run.d.qpos[qp]), action.copy(), radians))
                        run.sim.set_joints(dict(zip(JOINT_NAMES, radians)))
                        run.advance(1/FPS)
                        if completed():
                            break
                    if len(latency) == 1 or len(latency) % 10 == 0 or tracker.stable_hold:
                        print(f"Sim {run.d.time:5.2f}s | inference {duration:.3f}s | "
                              f"max lift {tracker.maximum_clearance*1000:.1f}mm | "
                              f"hold {tracker.longest_contact_hold_s:.2f}s", flush=True)
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
        print(error, flush=True)
    final = (float(run.d.time), run.d.qpos.copy(), "complete")
    if run.frames[-1][0] == final[0]:
        run.frames[-1] = final
    else:
        run.frames.append(final)
    checks = dict(lifted_50mm=tracker.maximum_clearance >= tracker.clearance,
        stable_two_finger_hold=tracker.stable_hold is not None,
        no_robot_table_contact=run.table_contact_steps == 0,
        no_robot_self_contact=not self_contacts, no_unintended_floor_contact=not floor_contacts,
        base_stationary=run.max_base_drift <= config["success"]["maximum_base_translation_mm"]/1000,
        elevator_stationary=run.max_elevator_drift <= config["success"]["maximum_elevator_drift_mm"]/1000,
        physics_healthy=not run.sim.state()["warnings"], inference_healthy=error is None)
    checks = {k:bool(v) for k,v in checks.items()}
    if placement is not None:
        checks['released_and_placed'] = placement.stable_placement is not None
    report = dict(format="sourccey-xvla-evaluation-v1", task=task, simulation_only=True,
        passed=all(checks.values()), checks=checks, scene=config, layout=measured,
        checkpoint=str(args.checkpoint.resolve()), source_scene=str(scene_path.resolve()),
        source_split=config.get("split", "unspecified"), policy_seed=args.seed,
        simulated_duration_s=float(run.d.time), wall_s=time.perf_counter()-started,
        action_hz=FPS, execution_horizon=args.horizon, policy_calls=len(latency),
        inference=ready, latency_s=dict(median=float(np.median(latency)) if latency else None,
                                       maximum=max(latency, default=None)),
        timing="offline synchronous closed-loop: physics pauses during inference; not a real-time benchmark",
        maximum_bottom_clearance_mm=tracker.maximum_clearance*1000,
        longest_contact_hold_s=tracker.longest_contact_hold_s, successful_hold=tracker.stable_hold,
        clipped_action_samples_by_joint=dict(zip(JOINT_NAMES, clipped.tolist())),
        robot_table_contact_steps=run.table_contact_steps,
        robot_table_contact_geoms=sorted(run.table_contact_geoms),
        maximum_table_penetration_mm=run.max_table_penetration*1000,
        self_contact_peak_force_n=self_contacts, unintended_floor_peak_force_n=floor_contacts,
        box_final_center_m=run.d.xpos[run.box].tolist(), warnings=run.sim.state()["warnings"],
        learner_features=["observation.state"]+["observation.images."+name for name in images] if records else [],
        teacher_control=False, grasp_assistance=False, error=error)
    if placement is not None:
        report['placement'] = dict(passed=placement.stable_placement is not None,
            stable_placement=placement.stable_placement, final=placement.last, destination=config['placement'])
    report["evaluation_source_sha256"] = {
        name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ("examples/box_policy_eval.py", "sourccey/xvla_eval.py", "integrations/xvla_worker.py", "sourccey/box_task.py")}
    save_run(run, report, output, export_model=False)
    if records:
        np.savez_compressed(output/"actions.npz", timestamp=[r[0] for r in records],
            state=[r[1] for r in records], action=[r[2] for r in records], command_rad=[r[3] for r in records])
    # A useful inspection image even when the learned attempt misses the box.
    copy_display_state(run.d, display, data)
    from PIL import Image
    with create_renderer(display, width=960, height=720, backend="filament") as renderer:
        renderer.update_scene(data, camera=camera(), scene_option=options())
        Image.fromarray(renderer.render()).save(output/"final.png")
    print(json.dumps({k:report[k] for k in ("passed", "maximum_bottom_clearance_mm", "successful_hold", "latency_s", "error")}, indent=2), flush=True)
    print(f"Saved learned-policy rollout: {output}", flush=True)
    return report


def review(output, seconds=0):
    """Replay only the saved policy-driven physics, never regenerate a teacher."""
    from mujoco.studio import launch_passive, viewer_protocol
    from sourccey.studio_review import NativePresentation, RecordedExperiment, ExperimentPanel
    output = Path(output)
    config = read_json(output/"scene.json")
    recording, report = load_recording(output, config)
    if report.get("format") != "sourccey-xvla-evaluation-v1":
        raise ValueError("Require an actual learned-policy evaluation recording")
    model, _ = scene_model(config, display_only=True)
    if config.get("domain_randomization", {}).get("camera_perturbations"):
        apply_camera_perturbations(model, config)
    data = mujoco.MjData(model)
    data.qpos[:] = recording["qpos"][0]
    mujoco.mj_forward(model, data)
    plugin = RecordedExperiment(model, data, recording, report)
    viewer = viewer_protocol.ViewerConfig(title="Sourccey | X-VLA learned pickup evaluation",
        width=1440, height=960, gfx="opengl", open_browser=False)
    started = time.perf_counter()
    with launch_passive.launch(viewer, plugins=[plugin, ExperimentPanel(), NativePresentation()]) as handle:
        handle.set_model(model, data=data)
        while handle.is_running():
            handle.sync(model, data)
            if seconds and time.perf_counter()-started >= seconds:
                break


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODEL)
    parser.add_argument("--policy-python", type=Path, default=PYTHON)
    parser.add_argument("--lerobot-source", type=Path, default=SOURCE)
    parser.add_argument("--episode", default="location_01_01_attempt_0000")
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--collection", type=Path, help="Continuous collection root; evaluate its saved fresh poses without LeRobot export")
    parser.add_argument("--dataset", type=Path, help="LeRobot dataset for a legacy saved-scene sweep")
    parser.add_argument("--pilot", type=Path, default=PILOT, help="Source scenes for a legacy dataset sweep")
    parser.add_argument("--final-test", action="store_true", help="Explicitly unlock test scenes after freezing model and runtime settings")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seconds", type=float, help="Trial limit; defaults to 24 s for pickup, 36 s for pick and place")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--view-seconds", type=float, default=0)
    parser.add_argument("--replay", action="store_true", help="View the saved AI attempt without loading the model")
    parser.add_argument("--sweep", choices=("validation", "train", "test"),
                        help="Evaluate every saved scene in this split; final test remains reserved")
    parser.add_argument("--limit", type=int, help="Run only the first N selected sweep scenes")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if (args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0)) or args.horizon < 1:
        parser.error("Require positive duration and execution horizon")
    if not np.isfinite(args.view_seconds) or args.view_seconds < 0:
        parser.error("View duration must be finite and nonnegative")
    if args.sweep:
        if args.sweep == 'test' and (not args.collection or not args.final_test):
            parser.error('Test evaluation requires --collection and explicit --final-test')
        if args.scene or args.view or args.replay:
            parser.error("Sweep saves all attempts; replay an individual output directory afterward")
        if args.limit is not None and args.limit < 1:
            parser.error("Sweep limit must be positive")
        from examples.box_policy_sweep import sweep
        report = sweep(args)
        if report["error"]:
            raise SystemExit(1)
        return
    if args.collection or args.dataset or args.final_test:
        parser.error('--collection, --dataset and --final-test require --sweep')
    if args.limit is not None:
        parser.error("--limit requires --sweep")
    report = evaluate(args) if not args.replay else None
    if args.view or args.replay:
        review(args.output, args.view_seconds)
    if report and report["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
