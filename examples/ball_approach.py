"""Offline, open-gripper ball approach. No hardware connection or grasp assist.

Run: python -m examples.ball_approach --scene examples/ball_approach.example.json
See docs/ball_approach.md for coordinate and measurement conventions.
"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

from sourccey.build import MODEL, ROLES
from sourccey.simulation import Simulation


def vector(value, size=3):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Expected {size} finite numbers")
    return result


def positive(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Expected a positive finite number")
    return value


def frame(sim, config):
    rotation = sim.robot_rotation.copy()
    origin = sim.data.xpos[sim.base_id] + rotation @ (vector(config["datum_offset_mm"]) / 1000)
    return origin, rotation


def scene_model(config):
    side = config["side"]
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    if not config["datum_description"].strip():
        raise ValueError("Describe the repeatable base datum")
    radius = positive(config["ball_diameter_mm"]) / 2000
    center = vector(config["ball_center_mm"]) / 1000
    top = float(config["tabletop_height_mm"]) / 1000
    if not np.isfinite(top) or abs(center[2] - top - radius) > 0.001:
        raise ValueError("Ball center Z must equal tabletop height + ball radius (within 1 mm)")
    source = Simulation()
    origin, rotation = frame(source, config)
    spec = mujoco.MjSpec.from_file(str(MODEL))
    # Only the support fixtures use constraints. The ball is a free body.
    spec.add_equality(name="benchmark_fixed_base", type=mujoco.mjtEq.mjEQ_WELD,
                      name1="base_link", name2="world", objtype=mujoco.mjtObj.mjOBJ_BODY,
                      solref=[.002, 1], solimp=[.999, .9999, .001, .5, 2])
    elevator = float(config["elevator_m"])
    bounds = source.model.joint("linear_actuator").range
    if not np.isfinite(elevator) or not bounds[0] <= elevator <= bounds[1]:
        raise ValueError("elevator_m outside model limits")
    spec.add_equality(name="benchmark_fixed_elevator", type=mujoco.mjtEq.mjEQ_JOINT,
                      name1="linear_actuator", data=[elevator] + [0] * 10,
                      objtype=mujoco.mjtObj.mjOBJ_JOINT,
                      solref=[.002, 1], solimp=[.999, .9999, .001, .5, 2])
    half = vector(config["table_half_size_mm"], 2) / 1000
    if np.any(half <= radius):
        raise ValueError("Table half sizes must exceed the ball radius")
    spec.worldbody.add_geom(name="benchmark_table", type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=origin + rotation @ [center[0], center[1], top - .02],
                            size=[*half, .02], contype=1, conaffinity=3,
                            rgba=[.45, .30, .18, 1])
    ball = spec.worldbody.add_body(name="benchmark_ball", pos=origin + rotation @ center,
                                  mass=.02, ipos=[0, 0, 0],
                                  inertia=[.4 * .02 * radius**2] * 3)
    ball.add_freejoint(name="benchmark_ball_free")
    ball.add_geom(name="benchmark_ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  size=[radius], mass=.02, contype=3, conaffinity=3,
                  rgba=[1, .4, .1, 1])
    suffix = "1" if side == "left" else "2"
    # Reference landmarks inspired by dual_grasp's pad centers. These are
    # configurable measurement landmarks, not a claim of calibrated CAD tips.
    for label, body in (("fixed", f"Gripper_Base_v1_{suffix}"),
                        ("moving", f"Gripper_Finger_v1_{suffix}")):
        spec.body(body).add_site(name=f"benchmark_{label}",
                                pos=vector(config[f"{label}_landmark_mm"]) / 1000,
                                size=[.003], rgba=[.1, 1, .3, 1])
    if config.get("contact_pads", False):
        # Same simplified finger pads as dual_grasp; replace only hand hulls.
        for body in (f"Gripper_Base_v1_{suffix}", f"Gripper_Finger_v1_{suffix}"):
            for index in (0, 1):
                geom = spec.geom(f"collision_{body}_{index}")
                geom.contype = geom.conaffinity = 0
        for label, body, depth in (("fixed", f"Gripper_Base_v1_{suffix}", .004),
                                   ("moving", f"Gripper_Finger_v1_{suffix}", .006)):
            pos = ([.08, 0, -.008] if side == "left" else [0, -.08, -.008]) if label == "fixed" else (
                [.04, 0, -.004] if side == "left" else [0, -.04, -.004])
            spec.body(body).add_geom(
                name=f"benchmark_{label}_pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos, size=([.04, .014, depth] if side == "left" else [.014, .04, depth]),
                contype=2, conaffinity=1, condim=4, friction=[2.5, .03, .002],
                solref=[.008, 1], rgba=[.9, .9, .75, .5])
    model = spec.compile()
    for role, limits in config.get("joint_motion_limits_rad", {}).items():
        if role not in ROLES:
            raise ValueError(f"Unexpected motion-limit joint {role}")
        bounds = vector(limits, 2)
        joint = model.joint(f"{side}_{role}")
        if bounds[0] >= bounds[1] or bounds[0] < joint.range[0] or bounds[1] > joint.range[1]:
            raise ValueError(f"Invalid reduced motion limits for {role}")
        joint.range[:] = bounds
    key = model.key("startup").id
    adr = model.joint("benchmark_ball_free").qposadr[0]
    model.key_qpos[key, adr:adr + 7] = model.qpos0[adr:adr + 7]
    model.key_qpos[key, model.joint("linear_actuator").qposadr[0]] = elevator
    model.key_ctrl[key, model.actuator("linear_actuator").id] = elevator
    for role, angle in config["arm_pose_rad"].items():
        if role not in ROLES[:5]:
            raise ValueError(f"Unexpected arm joint {role}")
        joint = model.joint(f"{side}_{role}")
        if not np.isfinite(angle) or not joint.range[0] <= angle <= joint.range[1]:
            raise ValueError(f"Invalid angle for {role}")
        model.key_qpos[key, joint.qposadr[0]] = angle
        model.key_ctrl[key, model.actuator(f"{side}_{role}").id] = angle
    # Open is 60 degrees in the current simulation API; hardware endpoints
    # still require independent verification.
    joint = model.joint(f"{side}_gripper")
    model.key_qpos[key, joint.qposadr[0]] = joint.range[1]
    model.key_ctrl[key, model.actuator(f"{side}_gripper").id] = joint.range[1]
    for role in config.get("joint_motion_limits_rad", {}):
        joint = model.joint(f"{side}_{role}")
        if not joint.range[0] <= model.key_qpos[key, joint.qposadr[0]] <= joint.range[1]:
            raise ValueError(f"Startup pose outside reduced motion limits for {role}")
    return model


def grasp_point(model, data):
    return np.mean([data.site_xpos[model.site(f"benchmark_{label}").id]
                    for label in ("fixed", "moving")], axis=0)


class TableContactMonitor:
    """Inspect every physics step, excluding the ball's intended support contact."""

    def __init__(self, model):
        self.table = model.geom("benchmark_table").id
        self.ball_body = model.body("benchmark_ball").id
        self.pads = [model.geom(f"benchmark_{label}_pad").id for label in ("fixed", "moving")]
        self.contact_steps = 0
        self.contact_geoms = set()
        self.max_penetration = 0.
        self.min_pad_clearance = .1

    def observe(self, sim):
        m, d = sim.model, sim.data
        touched = False
        for contact in d.contact:
            if self.table not in (contact.geom1, contact.geom2):
                continue
            other = int(contact.geom2 if contact.geom1 == self.table else contact.geom1)
            if m.geom_bodyid[other] in (0, self.ball_body):
                continue
            touched = True
            self.contact_geoms.add(m.geom(other).name)
            self.max_penetration = max(self.max_penetration, -float(contact.dist))
        self.contact_steps += int(touched)
        for pad in self.pads:
            distance = mujoco.mj_geomDistance(m, d, self.table, pad, .1, None)
            self.min_pad_clearance = min(self.min_pad_clearance, distance)

    def report(self):
        return {"robot_table_contact_steps": self.contact_steps,
                "contact_geoms": sorted(self.contact_geoms),
                "max_penetration_mm": self.max_penetration * 1000,
                "minimum_pad_clearance_mm": self.min_pad_clearance * 1000}


def run(config, measurement=None, on_step=None, speed=1.0):
    speed = positive(speed)
    if speed > 1:
        raise ValueError("speed must be at most 1 (use 0.5 for half-speed)")
    if config.get("pickup", False) and not config.get("contact_pads", False):
        raise ValueError("The concept pickup requires contact_pads")
    sim = Simulation(model=scene_model(config), traction=False)
    m, d = sim.model, sim.data
    monitor = TableContactMonitor(m) if config.get("contact_pads", False) else None

    def observe(simulation):
        if monitor is not None:
            monitor.observe(simulation)
        if on_step is not None:
            on_step(simulation)

    origin, rotation = frame(sim, config)
    local_mm = lambda world: (rotation.T @ (world - origin) * 1000).tolist()
    target_mm = vector(config["ball_center_mm"]) + vector(config["approach_offset_mm"])
    target = origin + rotation @ (target_mm / 1000)
    names = [f"{config['side']}_{role}" for role in ROLES[:3]]
    qp = np.array([m.joint(name).qposadr[0] for name in names])
    bounds = np.array([m.joint(name).range for name in names])
    scratch = mujoco.MjData(m)
    scratch.qpos[:] = d.qpos

    def residual(angles):
        scratch.qpos[qp] = angles
        mujoco.mj_forward(m, scratch)
        return grasp_point(m, scratch) - target

    # Seed the bent-elbow branch; a straight-down start otherwise traps this
    # position-only solve against the opposite elbow limit.
    guess = d.qpos[qp].copy()
    guess[1:] = [1.0 if config["side"] == "left" else -1.0, -1.8]
    solution = least_squares(residual, np.clip(guess, bounds[:, 0], bounds[:, 1]), bounds=bounds.T,
                             ftol=1e-10, xtol=1e-10, gtol=1e-10)
    initial_base = d.qpos[:7].copy()
    initial_ball = d.xpos[m.body("benchmark_ball").id].copy()
    start = d.qpos[qp].copy()
    max_base_translation = max_base_rotation = max_elevator = 0.

    def advance():
        nonlocal max_base_translation, max_base_rotation, max_elevator
        sim.step()
        observe(sim)
        max_base_translation = max(max_base_translation, float(np.linalg.norm(d.qpos[:3] - initial_base[:3])))
        dot = abs(float(d.qpos[3:7] @ initial_base[3:7]))
        max_base_rotation = max(max_base_rotation, float(2 * np.arccos(np.clip(dot, 0, 1))))
        max_elevator = max(max_elevator, abs(float(d.qpos[m.joint("linear_actuator").qposadr[0]]) - config["elevator_m"]))

    def move_joints(destination):
        source = d.qpos[qp].copy()
        for fraction in np.linspace(0, 1, int(np.ceil(120 / speed)) + 1)[1:]:
            smooth = fraction * fraction * (3 - 2 * fraction)
            sim.set_joints(dict(zip(names, source + smooth * (destination - source))))
            for _ in range(8):
                advance()

    # Raise outside the table before bending and swinging forward. Folding
    # while hanging down sweeps the hand into the chassis.
    observe(sim)  # Include the exact parked state for replay preflight.
    for _ in range(300):
        advance()
    outward = start.copy()
    outward[0] = .94
    move_joints(outward)
    raised = outward.copy()
    staging_lift = positive(config.get("staging_lift_rad", 2.0))
    raised[1] = staging_lift if config["side"] == "left" else -staging_lift
    move_joints(raised)
    tucked = raised.copy()
    tucked[2] = -1.8
    move_joints(tucked)
    forward = tucked.copy()
    forward[0] = solution.x[0]
    move_joints(forward)
    move_joints(solution.x)
    for _ in range(300):
        advance()
    actual = grasp_point(m, d)
    ball_end = d.xpos[m.body("benchmark_ball").id].copy()
    error = float(np.linalg.norm(actual - target) * 1000)
    ball_motion = float(np.linalg.norm(ball_end - initial_ball) * 1000)
    passed = (error < positive(config["approach_tolerance_mm"])
              and np.linalg.norm(solution.fun) < .002 and max_base_translation < .001
              and max_base_rotation < .001 and max_elevator < .001
              and ball_motion < 2 and not sim.state()["warnings"]
              and (monitor is None or monitor.contact_steps == 0))
    report = {
        "scene": config, "speed": speed,
        "mode": "simulation only, open approach; no ball attachment",
        "frame": "+X forward, +Y robot left, +Z up; origin at described datum",
        "grasp_point": "midpoint of fixed and moving landmarks, in their respective body frames",
        "intended_grasp_point_mm": target_mm.tolist(),
        "simulated_grasp_point_mm": local_mm(actual),
        "approach_error_mm": error, "ik_error_mm": float(np.linalg.norm(solution.fun) * 1000),
        "simulated_arm_rad": {role: float(d.qpos[m.joint(f"{config['side']}_{role}").qposadr[0]]) for role in ROLES},
        "max_base_translation_mm": max_base_translation * 1000,
        "max_base_rotation_rad": max_base_rotation, "max_elevator_drift_mm": max_elevator * 1000,
        "ball_displacement_mm": ball_motion, "warnings": sim.state()["warnings"],
        "simulation_passed": bool(passed), "real_comparison": None,
        "benchmark_complete": False,
        "limitations": ["Example dimensions are not physical measurements" ] if config.get("illustrative", False) else [],
    }
    if measurement is not None:
        # Evaluate FK at the independently mapped, actually observed real pose.
        # Never infer elbow/wrist scales or silently substitute commanded angles.
        if measurement["side"] != config["side"] or measurement["datum_description"] != config["datum_description"]:
            raise ValueError("Measurement side/datum differs from scene")
        if not np.allclose(vector(measurement["ball_center_mm"]), config["ball_center_mm"], atol=.01, rtol=0):
            raise ValueError("Measurement must use the same marked ball location")
        for field in ("measured_at", "measurement_method", "joint_mapping_notes"):
            if not measurement[field].strip():
                raise ValueError(f"Measurement requires {field}")
        scratch.qpos[:] = d.qpos
        for role in ROLES:
            vector([measurement["hardware_positions"][f"{config['side']}_{role}.pos"]], 1)
            angle = float(measurement["mujoco_joint_rad"][role])
            joint = m.joint(f"{config['side']}_{role}")
            if not np.isfinite(angle) or not joint.range[0] <= angle <= joint.range[1]:
                raise ValueError(f"Measured mapping outside limits: {role}")
            scratch.qpos[joint.qposadr[0]] = angle
        mujoco.mj_forward(m, scratch)
        predicted = np.asarray(local_mm(grasp_point(m, scratch)))
        real = vector(measurement["grasp_point_mm"])
        delta = real - predicted
        report["real_comparison"] = {
            "measurement": measurement, "simulated_at_observed_pose_mm": predicted.tolist(),
            "measured_real_grasp_point_mm": real.tolist(), "real_minus_sim_mm": delta.tolist(),
            "position_error_mm": float(np.linalg.norm(delta)),
            "mapping_verified": measurement.get("mapping_verified") is True,
        }
        report["benchmark_complete"] = bool(passed and not config.get("illustrative", False)
                                              and measurement.get("mapping_verified") is True)
    if config.get("pickup", False):
        report["pickup"] = pickup(sim, config, origin, rotation, observe, speed=speed)
        report["pickup"]["passed"] &= monitor.contact_steps == 0
    if monitor is not None:
        report["table_clearance"] = monitor.report()
        report["benchmark_complete"] &= monitor.contact_steps == 0
    report["simulated_seconds"] = float(d.time)
    return report


def pickup(sim, config, origin, rotation, on_step=None, speed=1.0):
    """Concept-only contact test, independent of the open-approach comparison."""
    m, d = sim.model, sim.data
    side = config["side"]
    names = [f"{side}_{role}" for role in ROLES[:5]]
    qp = np.array([m.joint(name).qposadr[0] for name in names])
    bounds = np.array([m.joint(name).range for name in names])
    scratch = mujoco.MjData(m)
    palm = m.body("Gripper_Base_v1_1" if side == "left" else "Gripper_Base_v1_2").id
    table = m.geom("benchmark_table").id
    pads = [m.geom(f"benchmark_{label}_pad").id for label in ("fixed", "moving")]
    pinch_local = vector(config["fixed_landmark_mm"]) / 1000
    pinch_local[2] += config["ball_diameter_mm"] / 2000
    preferred_normal = vector(config.get("pinch_normal_world", [0, 1, 0]))
    length = np.linalg.norm(preferred_normal)
    if length < 1e-6:
        raise ValueError("pinch_normal_world must have a nonzero direction")
    preferred_normal /= length

    def pinch(data):
        return data.xpos[palm] + data.xmat[palm].reshape(3, 3) @ pinch_local

    def advance(count):
        for _ in range(count):
            sim.step()
            if on_step is not None:
                on_step(sim)

    def move(target):
        scratch.qpos[:] = d.qpos
        def residual(angles):
            scratch.qpos[qp] = angles
            mujoco.mj_forward(m, scratch)
            normal = scratch.xmat[palm].reshape(3, 3)[:, 2]
            clearance = [mujoco.mj_geomDistance(m, scratch, table, pad, .1, None) for pad in pads]
            return np.r_[pinch(scratch) - target, .1 * (normal - preferred_normal),
                         5 * np.maximum(.005 - np.asarray(clearance), 0)]
        solution = least_squares(residual, np.clip(d.qpos[qp], bounds[:, 0], bounds[:, 1]), bounds=bounds.T)
        start = d.qpos[qp].copy()
        for fraction in np.linspace(0, 1, int(np.ceil(120 / speed)) + 1)[1:]:
            smooth = fraction**2 * (3 - 2 * fraction)
            sim.set_joints(dict(zip(names, start + smooth * (solution.x - start))))
            advance(8)

    ball_id = m.body("benchmark_ball").id
    initial = d.xpos[ball_id].copy()
    move(initial + [0, 0, .08])
    move(initial.copy())
    advance(200)
    for closure in np.linspace(0, 100, int(np.ceil(100 / speed)) + 1)[1:]:
        sim.set_gripper(side, closure)
        advance(8)
    advance(200)
    before_lift = d.xpos[ball_id].copy()
    move(pinch(d) + [0, 0, .12])
    advance(600)
    held = d.xpos[ball_id].copy()
    obj = m.geom("benchmark_ball_geom").id
    touching = []
    for label in ("fixed", "moving"):
        pad = m.geom(f"benchmark_{label}_pad").id
        if any({int(c.geom1), int(c.geom2)} == {obj, pad} for c in d.contact):
            touching.append(label)
    lift = float(held[2] - initial[2]) * 1000
    return {"mode": "contact only, simplified pads; no object attachment or object pose commands",
            "ball_initial_mm": (rotation.T @ (initial - origin) * 1000).tolist(),
            "ball_held_mm": (rotation.T @ (held - origin) * 1000).tolist(),
            "ball_motion_before_lift_mm": float(np.linalg.norm(before_lift - initial) * 1000),
            "ball_horizontal_motion_before_lift_mm": float(np.linalg.norm(before_lift[:2] - initial[:2]) * 1000),
            "hold_seconds": 600 * m.opt.timestep,
            "lift_mm": lift, "held_pad_contacts": touching,
            "passed": bool(lift > 80 and len(touching) == 2 and not sim.state()["warnings"]),
            "warnings": sim.state()["warnings"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--measurement", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ball_approach.json"))
    parser.add_argument("--gif", type=Path, help="Save an animated simulation preview")
    parser.add_argument("--view", action="store_true", help="Watch the run in MuJoCo at real time")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Motion speed from 0 to 1, excluding 0; 0.5 halves arm/gripper ramp speed")
    parser.add_argument("--trajectory", type=Path, help="Save simulated joint commands/readbacks; not hardware commands")
    args = parser.parse_args()
    renderer = None
    frames = []
    step_index = 0
    trajectory = []
    viewer = None
    playback_start = None

    def capture(sim):
        nonlocal renderer, step_index, viewer, playback_start
        step_index += 1
        if args.view:
            if viewer is None:
                from mujoco import viewer as mujoco_viewer
                viewer = mujoco_viewer.launch_passive(sim.model, sim.data)
                viewer.cam.lookat[:] = [.40, .18, .85]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.35, 180, -18
                viewer.opt.geomgroup[3] = 0
                playback_start = time.perf_counter() - sim.data.time
            if not viewer.is_running():
                raise KeyboardInterrupt
            if step_index % 10 == 0:
                viewer.sync()
                remaining = playback_start + sim.data.time - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
        if step_index % 60:
            return
        if args.trajectory:
            names = [f"{config['side']}_{role}" for role in ROLES]
            trajectory.append({"time_s": float(sim.data.time),
                               "joint_rad": {n: float(sim.data.qpos[sim.model.joint(n).qposadr[0]]) for n in names},
                               "command_rad": {n: float(sim.data.ctrl[sim.model.actuator(n).id]) for n in names}})
        if not args.gif:
            return
        from PIL import Image
        if renderer is None:
            renderer = mujoco.Renderer(sim.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.40, .18, .85]
        camera.distance, camera.azimuth, camera.elevation = 1.35, 180, -18
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        renderer.update_scene(sim.data, camera=camera, scene_option=options)
        frames.append(Image.fromarray(renderer.render()))

    try:
        config = json.loads(args.scene.read_text(encoding="utf-8"))
        measurement = json.loads(args.measurement.read_text(encoding="utf-8")) if args.measurement else None
        report = run(config, measurement, capture if args.view or args.gif or args.trajectory else None,
                     speed=args.speed)
    except KeyboardInterrupt:
        print("Simulation stopped.")
        return
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    finally:
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()
    if args.gif and frames:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                       duration=120, loop=0)
    if args.trajectory:
        args.trajectory.parent.mkdir(parents=True, exist_ok=True)
        args.trajectory.write_text(json.dumps({"units": "MuJoCo radians; uncalibrated for hardware",
                                              "speed": args.speed,
                                              "samples": trajectory}, indent=2) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["simulation_passed"] or (report.get("pickup") and not report["pickup"]["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
