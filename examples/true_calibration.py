"""Simulation-only True Calibration, using the installed robot's controllers.

Run: python -m examples.true_calibration
See docs/true_calibration.md for fidelity limits and the gripper probe.
"""
import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace
import time

import mujoco
import numpy as np

from integrations.calibration_controller import controller, REFERENCE
from sourccey.build import MODEL, ROOT, ROLES
from sourccey.collision import add_torso_collision, decomposed_collision
from sourccey.simulation import Simulation


# Raw encoder -> CAD joint direction, including the mirrored right assembly.
# These are differential source conventions, not the old gripper endpoint fit.
RAW_SIGN = {"left": (1, -1, 1, -1, -1, 1),
            "right": (-1, -1, -1, -1, 1, -1)}
MOTOR_PER_JOINT = {"shoulder_lift": 1.5, "wrist_roll": 2.035}
CURRENT_EFFORT_NM_PER_UNIT = .1  # Explicit surrogate; no measured motor Kt.


def scene_model(free_gripper=False, *, self_collision=False, free_arms=False,
                collision_model="cad-hulls"):
    if collision_model not in ("cad-hulls", "decomposed"):
        raise ValueError("Unknown collision model")
    spec = mujoco.MjSpec.from_file(str(MODEL))
    spec.modelname = ("Sourccey CAD Contact Calibration (simulation only)"
                      if self_collision and collision_model == "decomposed"
                      else "Sourccey Self Collision Calibration (simulation only)" if self_collision
                      else "Sourccey True Calibration (simulation only)")
    if self_collision and collision_model == "decomposed":
        decomposed_collision(spec)
    if self_collision:
        spec.memory = 64 * 1024 * 1024
        add_torso_collision(spec)
        # Existing robot hulls use bit 2; the environment uses bit 1. Include
        # robot/robot contacts without turning duplicate visual meshes into geoms.
        for geom in spec.geoms:
            if geom.contype & 2:
                geom.conaffinity |= 2
        # Finger/hand contact crosses a parent joint. The usual parent filter
        # would silently omit this key part of the requested experiment.
        spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    spec.add_equality(name="calibration_fixed_base", type=mujoco.mjtEq.mjEQ_WELD,
                      name1="base_link", name2="world", objtype=mujoco.mjtObj.mjOBJ_BODY,
                      solref=[.002, 1], solimp=[.999, .9999, .001, .5, 2])
    model = spec.compile()
    if self_collision:
        # Pre-existing CAD intersections can generate huge static forces. Scale
        # their arrows for inspection without altering any physical parameters.
        model.vis.map.force = .00001
    # Servo goals must be allowed past stops to generate a load reading.
    # Physical joint limits are preserved, except in the explicit gripper probe.
    for name in ("linear_actuator", *(f"{s}_{r}" for s in RAW_SIGN for r in ROLES)):
        model.actuator_ctrllimited[model.actuator(name).id] = False
    if free_gripper:
        for side in RAW_SIGN:
            model.jnt_limited[model.joint(f"{side}_gripper").id] = False
    if free_arms:
        for side in RAW_SIGN:
            for role in ROLES:
                model.jnt_limited[model.joint(f"{side}_{role}").id] = False
    return model


def contact_snapshot(model, data, joint=None):
    """Active contact pairs, optionally restricted to loads across one joint."""
    root = model.jnt_bodyid[joint] if joint is not None else None

    def downstream(body):
        while body and body != root:
            body = model.body_parentid[body]
        return body == root

    pairs = []
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.efc_address < 0:
            continue
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if joint is not None and downstream(b1) == downstream(b2):
            continue
        mujoco.mj_contactForce(model, data, index, force)
        pairs.append({"geom1": model.geom(g1).name, "geom2": model.geom(g2).name,
                      "body1": model.body(b1).name, "body2": model.body(b2).name,
                      "distance_mm": float(contact.dist * 1000),
                      "normal_force_n": float(force[0]),
                      "robot_self_contact": b1 != 0 and b2 != 0})
    return sorted(pairs, key=lambda p: p["normal_force_n"], reverse=True)


def calibration_physics_step(sim):
    """Split the same implicit dynamics to avoid redundant collision passes.

    Observers refresh forward dynamics when they actually read loads or render.
    Gravity/Coriolis feed-forward is still updated at every physics timestep.
    """
    model, data = sim.model, sim.data
    data.qfrc_applied[:] = 0
    mujoco.mj_step1(model, data)
    for name, aid in sim.actuators.items():
        if name not in ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"):
            model.actuator_biasprm[aid, 0] = data.qfrc_bias[model.jnt_dofadr[sim.joints[name]]]
    mujoco.mj_step2(model, data)


class Clock:
    def __init__(self, sim):
        self.sim = sim

    def monotonic(self):
        return float(self.sim.data.time)


class SearchLog:
    """Observe the pinned controller's decisions without changing its logic."""
    def __init__(self, run, side):
        self.run, self.side = run, side

    def info(self, message, *args):
        message = message % args if args else message
        start = re.search(r"Testing (positive|negative) direction for (\w+)", message)
        exhausted = re.search(r"Reached search range limit \((\d+)\) for (\w+) (positive|negative)", message)
        if start:
            direction, role = start.groups()
            self.run.searches.append({"arm": self.side, "joint": role, "direction": direction,
                                      "started_s": self.run.clock.monotonic(), "outcome": "running"})
        if exhausted:
            _, role, direction = exhausted.groups()
            self.run.finish_search(self.side, role, "search_range_exhausted")

    def warning(self, message, *args):
        logging.getLogger(__name__).warning(message, *args)

    def error(self, message, *args):
        logging.getLogger(__name__).error(message, *args)


class ArmBus:
    """Only the three raw bus operations used by mechanical-limit detection."""
    def __init__(self, run, side):
        self.run, self.sim, self.side = run, run.sim, side
        self.motors = dict.fromkeys(ROLES)
        self.start = {r: self.angle(r) for r in ROLES}
        # _initialize_calibration makes the present pose half-turn, then moves
        # only the shoulder encoder's reference to 3800 (left) / 296 (right).
        self.zero = dict.fromkeys(ROLES, 2048)
        self.zero["shoulder_lift"] = 3800 if side == "left" else 296
        self.scale = {}
        for role, sign in zip(ROLES, RAW_SIGN[side]):
            gear = 2.076 if role == "wrist_roll" and side == "right" else MOTOR_PER_JOINT.get(role, 1)
            self.scale[role] = sign * 2 * np.pi / (4096 * gear)
        self.goal = self.zero.copy()
        self.active = "waiting"
        self.current = 0.

    def angle(self, role):
        return float(self.sim.data.qpos[self.sim.model.joint(f"{self.side}_{role}").qposadr[0]])

    def disable_torque(self):
        # The hardware transitions consume no simulation time. Existing holding
        # commands stay latched while the controller disables/re-enables torque.
        pass

    def enable_torque(self):
        pass

    def sync_read(self, register, normalize=False):
        return {role: self.read(register, role, normalize=normalize) for role in ROLES}

    def read(self, register, role, normalize=False):
        if normalize:
            raise ValueError("True Calibration uses raw encoder values")
        model, data = self.sim.model, self.sim.data
        if register == "Present_Position":
            return round(self.zero[role] + (self.angle(role) - self.start[role]) / self.scale[role])
        if register != "Present_Current":
            raise ValueError(register)
        jid = model.joint(f"{self.side}_{role}").id
        dof = model.jnt_dofadr[jid]
        # Constraint reaction is a load surrogate, excluding the idealized
        # gravity feed-forward. It is deliberately not labelled measured mA.
        effort = abs(float(data.qfrc_constraint[dof]))
        self.current = effort / CURRENT_EFFORT_NM_PER_UNIT
        threshold = 375 if role == "shoulder_lift" else 75
        if self.current > threshold:
            at_joint_limit = any(
                data.efc_type[k] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
                and data.efc_id[k] == jid and abs(data.efc_force[k]) > 1e-5
                for k in range(data.nefc))
            cause = "model_joint_limit" if at_joint_limit else "contact_reaction"
            self.run.finish_search(self.side, role, cause)
            self.run.events.append({"time_s": float(data.time), "arm": self.side,
                                    "joint": role, "angle_deg": float(np.rad2deg(self.angle(role))),
                                    "raw_position": self.read("Present_Position", role),
                                    "current_surrogate_units": self.current,
                                    "cause": cause,
                                    "contacts": contact_snapshot(model, data, jid)})
        return self.current

    def write(self, register, role, value, normalize=False):
        if normalize or register != "Goal_Position":
            raise ValueError("Only raw Goal_Position writes are supported")
        self.active = role
        self.goal[role] = value
        aid = self.sim.model.actuator(f"{self.side}_{role}").id
        self.sim.data.ctrl[aid] = self.start[role] + (value - self.zero[role]) * self.scale[role]


class Lift:
    """Native 10-bit sensor and direct drive adapter; no GPIO or saved files."""
    def __init__(self, run):
        self.run, self.sim = run, run.sim
        self.motor_invert = self.invert = False
        self.is_connected = True
        self.motor = "lift"
        self.driver = self.sensor = self
        self.calibration_min, self.calibration_max = 0, 1023
        self.command = 0.

    def set_velocity(self, motor, value, **kwargs):
        self.command = value

    def stop(self):
        self.command = 0.
        aid = self.sim.model.actuator("linear_actuator").id
        self.sim.data.ctrl[aid] = self.sim.data.qpos[self.sim.model.joint("linear_actuator").qposadr[0]]

    def stop_position_controller(self):
        pass

    def read_raw(self):
        joint = self.sim.model.joint("linear_actuator")
        q = self.sim.data.qpos[joint.qposadr[0]]
        # Synthetic ADC spans the model's existing lift endpoints.
        value = round(1023 * (q - joint.range[0]) / np.ptp(joint.range))
        return SimpleNamespace(raw=int(np.clip(value, 0, 1023)))

    def set_calibration(self, *, raw_min, raw_max, invert):
        self.calibration_min, self.calibration_max, self.invert = raw_min, raw_max, invert

    def _save_calibration(self):
        pass

    def step(self):
        model, data = self.sim.model, self.sim.data
        aid = model.actuator("linear_actuator").id
        bounds = model.joint("linear_actuator").range
        # Assumed 20 mm/s drive. Real lift speed is not measured here.
        data.ctrl[aid] = np.clip(data.ctrl[aid] + self.command * .02 * model.opt.timestep,
                                 bounds[0] - .02, bounds[1] + .02)


class CalibrationRun:
    def __init__(self, arms="both", free_gripper=False, include_lift=True, *,
                 self_collision=False, free_arms=False, collision_model="cad-hulls"):
        if arms not in ("both", "left", "right"):
            raise ValueError("arms must be both, left, or right")
        self.sim = Simulation(model=scene_model(free_gripper, self_collision=self_collision,
                                               free_arms=free_arms, collision_model=collision_model), traction=False)
        self.clock = Clock(self.sim)
        self.free_gripper, self.include_lift = free_gripper, include_lift
        self.self_collision, self.free_arms = self_collision, free_arms
        self.collision_model = collision_model
        self.initial_contacts = contact_snapshot(self.sim.model, self.sim.data)
        self.events, self.results, self.jobs, self.searches = [], {}, [], []
        self.phase, self.complete = "lift" if include_lift else "arms", False
        self.arm_names = ("left", "right") if arms == "both" else (arms,)
        self.buses = {}
        self.lift = Lift(self)
        if include_lift:
            cls = controller("lift.py.txt", self.clock)["SourcceyZCalibrator"]
            self.jobs.append([0., "lift", cls(self.lift).auto_calibrate(full_reset=True)])
        else:
            self.start_arms()

    def arm_motion(self, side):
        # Initialize references when the arm starts, matching the host's stagger.
        bus = self.buses[side] = ArmBus(self, side)
        cls = controller("arm_motion.py.txt", self.clock, logger=SearchLog(self, side))["SourcceyFollowerCalibrator"]
        calibrator = cls()
        calibrator.robot = SimpleNamespace(bus=bus, config=SimpleNamespace(
            orientation=side, max_current_calibration_threshold=75))
        result = yield from calibrator._detect_mechanical_limits(reverse=side == "right")
        bus.active = "complete"
        return result

    def finish_search(self, side, role, outcome):
        for search in reversed(self.searches):
            if search["arm"] == side and search["joint"] == role and search["outcome"] == "running":
                search.update(outcome=outcome, finished_s=self.clock.monotonic(),
                              angle_deg=float(np.rad2deg(self.buses[side].angle(role))))
                return

    def start_arms(self):
        self.phase = "arms"
        for index, side in enumerate(self.arm_names):
            self.jobs.append([self.clock.monotonic() + index * 3., side, self.arm_motion(side)])

    def step(self):
        if self.complete:
            return
        if any(self.clock.monotonic() + 1e-9 >= job[0] for job in self.jobs):
            mujoco.mj_forward(self.sim.model, self.sim.data)
        for job in list(self.jobs):
            if self.clock.monotonic() + 1e-9 < job[0]:
                continue
            try:
                delay = next(job[2])
                job[0] = self.clock.monotonic() + float(delay)
            except StopIteration as result:
                self.results[job[1]] = asdict(result.value) if job[1] == "lift" else result.value
                self.jobs.remove(job)
        if not self.jobs:
            if self.phase == "lift":
                self.start_arms()
            else:
                self.complete, self.phase = True, "complete"
                return
        self.lift.step()
        calibration_physics_step(self.sim)
        if self.sim.data.time > 240:
            raise RuntimeError("Calibration exceeded the 240-second scene deadline")
        if not np.all(np.isfinite(self.sim.data.qpos)) or any(w.number for w in self.sim.data.warning):
            raise RuntimeError("MuJoCo reported unstable simulation")

    def report(self):
        mujoco.mj_forward(self.sim.model, self.sim.data)
        return {"complete": self.complete, "duration_s": float(self.sim.data.time),
                "controller": "robot True Calibration / full_reset=True motion methods",
                "hardware_commands_sent": False, "gripper_limit_removed": self.free_gripper,
                "self_collision_enabled": self.self_collision, "arm_angle_limits_removed": self.free_arms,
                "collision_model": self.collision_model,
                "main_body_collision": self.self_collision,
                "bearing_clearance_applied": self.self_collision and self.collision_model == "decomposed",
                "initial_contacts": self.initial_contacts,
                "lift_included": self.include_lift, "detected_raw_ranges": self.results,
                "stop_events": self.events,
                "search_outcomes": self.searches,
                "limitations": ["Current is a constraint-effort surrogate, not measured motor current.",
                                ("CAD self-contact is enabled, including parent/child links; remaining approximation errors can cause false stops."
                                 if self.self_collision else "Arm self-collision remains disabled in the supplied CAD hull model."),
                                ("All twelve arm angle limits are disabled; lift limits and controller search caps remain."
                                 if self.free_arms else "Inherited joint stops are model limits, not discovered hand contact."),
                                ("Torso and head use solid CAD outer envelopes; openings and access to the internal cavity are not modeled."
                                 if self.self_collision else "The torso is visual-only in the baseline comparison."),
                                "Right-arm direction references are provisional; dynamics and lift speed are unmeasured.",
                                "No hardware or production calibration files are written."],
                "reference": json.loads((REFERENCE / "provenance.json").read_text())}


def write_report(run, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.report(), indent=2) + "\n", encoding="utf-8")


def force_arrow_scale(model, data, max_length_m=.08):
    """Limit displayed arrows, retaining the unscaled forces in reports/physics."""
    force, peak = np.zeros(6), 0.
    for index in range(data.ncon):
        if data.contact[index].efc_address >= 0:
            mujoco.mj_contactForce(model, data, index, force)
            peak = max(peak, float(np.linalg.norm(force[:3])))
    return min(.00001, max_length_m * model.stat.meanmass / max(peak, 1e-9))


def playback_speed(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Playback speed must be positive and finite")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--speed", type=playback_speed, default=3.,
                        help="Recorded playback speed (default: 3; does not change the physics)")
    parser.add_argument("--live", action="store_true",
                        help="Run interactive physics instead of preparing/reusing a recording; CPU limited")
    parser.add_argument("--arms", choices=("both", "left", "right"), default="both")
    parser.add_argument("--skip-lift", action="store_true")
    parser.add_argument("--free-gripper", action="store_true",
                        help="Diagnostic: remove inherited gripper angle stops in this scene only")
    parser.add_argument("--self-collision", action="store_true",
                        help="Enable existing robot collision hulls against each other, including parent/child links")
    parser.add_argument("--free-arms", action="store_true",
                        help="Remove all twelve arm joint-angle limits in this scene only")
    parser.add_argument("--collision-model", choices=("cad-hulls", "decomposed"), default="decomposed",
                        help="Self-collision geometry: cached CAD decomposition (default) or original whole-part hulls")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/true_calibration/report.json")
    args = parser.parse_args()
    run = CalibrationRun(args.arms, args.free_gripper, not args.skip_lift,
                         self_collision=args.self_collision, free_arms=args.free_arms,
                         collision_model=args.collision_model)
    if args.headless:
        while not run.complete:
            run.step()
        write_report(run, args.report)
        print(json.dumps(run.report(), indent=2))
        return
    recording = None
    if not args.live:
        from examples.calibration_playback import CACHE, physics_signature, record_run, load_recording
        config = dict(arms=args.arms, free_gripper=args.free_gripper, include_lift=not args.skip_lift,
                      self_collision=args.self_collision, free_arms=args.free_arms,
                      collision_model=args.collision_model)
        signature = physics_signature(config)
        path = CACHE / (signature + ".npz")
        if not path.exists():
            print("Preparing accurate calibration playback; this computation is cached for later replays.", flush=True)
            record_run(run, path, signature)
        recording = load_recording(path, signature)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(recording["report"], indent=2) + "\n", encoding="utf-8")
    play(run, args, recording)


def play(run, args, recording=None):
    from mujoco import viewer

    paused, restart, show_hulls, show_contacts = [False], [False], [False], [False]

    def key_callback(key):
        if key == ord("P"):
            paused[0] = not paused[0]
        elif key == ord("R"):
            restart[0] = True
        elif key == ord("G") and recording is None:
            args.free_gripper = not args.free_gripper
            restart[0] = True
        elif key == ord("J") and recording is None:
            args.free_arms = not args.free_arms
            restart[0] = True
        elif key == ord("V"):
            show_hulls[0] = not show_hulls[0]
        elif key == ord("F"):
            show_contacts[0] = not show_contacts[0]

    with viewer.launch_passive(run.sim.model, run.sim.data, key_callback=key_callback,
                               show_left_ui=False, show_right_ui=False) as window:
        window.cam.lookat[:] = [0, 0, .83]
        window.cam.distance, window.cam.azimuth, window.cam.elevation = 2.0, 135, -16
        window.opt.geomgroup[3] = 0
        saved = False
        playback_position = 0.
        playback_wall = 0.
        playback_complete = False
        last = time.perf_counter()
        while window.is_running():
            now = time.perf_counter()
            if restart[0] and recording is not None:
                playback_position, playback_complete, restart[0] = 0., False, False
                playback_wall, saved = 0., False
                last = now
            elif restart[0]:
                # Preserve the viewer's model/data identities across restarts.
                new_run = CalibrationRun(args.arms, args.free_gripper, not args.skip_lift,
                                         self_collision=args.self_collision, free_arms=args.free_arms,
                                         collision_model=args.collision_model)
                run.sim.reset()
                run.sim.model.jnt_limited[:] = new_run.sim.model.jnt_limited
                mujoco.mj_copyData(run.sim.data, run.sim.model, new_run.sim.data)
                new_run.sim = run.sim
                new_run.clock.sim = run.sim
                new_run.lift.sim = run.sim
                run, saved, restart[0] = new_run, False, False
            if recording is not None:
                from examples.calibration_playback import frame_at
                if not paused[0] and not playback_complete:
                    playback_position += (now - last) * args.speed
                    playback_wall += now - last
                frame_index, playback_complete = frame_at(recording, run.sim, playback_position)
                if playback_complete and not saved:
                    duration = float(recording["time"][-1])
                    print(f"Playback finished: {duration:.2f} simulated seconds in {playback_wall:.2f} wall seconds "
                          f"({duration / max(playback_wall, 1e-9):.2f}x).", flush=True)
                    saved = True
            elif not paused[0] and not run.complete:
                for _ in range(min(75, max(1, round((now - last) * args.speed / run.sim.model.opt.timestep)))):
                    run.step()
            last = now
            if recording is None or show_contacts[0]:
                mujoco.mj_forward(run.sim.model, run.sim.data)
            if recording is None and run.complete and not saved:
                write_report(run, args.report)
                print(f"True Calibration finished; report: {args.report}", flush=True)
                saved = True
            phase = str(recording["phase"][frame_index]) if recording is not None else run.phase
            complete = playback_complete if recording is not None else run.complete
            lines = [f"Time: {run.sim.data.time:6.1f} s    Phase: {phase}",
                     f"Playback: {'paused' if paused[0] else 'complete' if complete else 'running'}  ({args.speed:g}x {'recorded' if recording is not None else 'live target'})",
                     f"Arm angle limits: {'removed' if args.free_arms else 'gripper only removed' if args.free_gripper else 'enabled'}",
                     f"Self-collision: {args.collision_model if args.self_collision else 'disabled'}"]
            if args.self_collision:
                lines.append("Torso/head collision: enabled")
            for side in run.arm_names:
                bus = run.buses.get(side)
                active = str(recording[side][frame_index]) if recording is not None else bus.active if bus else 'waiting'
                lines.append(f"{side:5}: {active}")
            events = (recording["report"]["stop_events"][:int(recording["event_count"][frame_index])]
                      if recording is not None else run.events)
            lines += [f"Detected stops: {len(events)}"]
            if args.self_collision:
                lines += [f"Startup contact points: {len(run.initial_contacts)}"]
            if events:
                e = events[-1]
                lines += [f"Last: {e['arm']} {e['joint']}", f"{e['angle_deg']:.1f} degrees - {e['cause']}"]
            window.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 "True Calibration (simulation only)\n" + "\n".join(lines), ""),
                (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 ("P: pause/resume   R: replay   (G/J edits require --live)\n" if recording is not None
                  else "P: pause/resume   R: run again   J: toggle arm limits   G: toggle gripper limit\n")
                 +
                 "V: collision geometry   F: contact points/forces\n"
                 + ("Current is simulated. Existing hull overlaps can produce false contact stops."
                    if args.self_collision else "Current is simulated. Robot self-contact is disabled."), "")])
            window.opt.geomgroup[3] = show_hulls[0]
            window.opt.geomgroup[2] = not show_hulls[0]
            window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = show_hulls[0]
            window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = show_contacts[0]
            window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = show_contacts[0]
            if show_contacts[0]:
                run.sim.model.vis.map.force = force_arrow_scale(run.sim.model, run.sim.data)
            window.sync()
            time.sleep(.01)
    if recording is None and not saved:
        write_report(run, args.report)


if __name__ == "__main__":
    main()
