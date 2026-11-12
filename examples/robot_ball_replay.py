"""Preflight a ball pickup, then optionally replay it with live feedback.

Default operation is entirely offline. Missing joint mappings fail closed.
"""
import argparse
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

from examples.ball_approach import run, scene_model
from examples.robot_start_pose import (
    HardwareLink, KEYS, ROLES, check_feedback, execute, finite,
    plan, positions, tracking_error, validate_host_limits, validate_profile,
    validate_stall_timeout, validate_tracking_tolerance,
)

REPLAY_TRACKING_TOLERANCE = 5.
REPLAY_STALL_TIMEOUT = 15.
REPLAY_MAX_COMMAND_STEP = 1.


def joint_fit(entry):
    """Fit independently measured angles; never assume normalized means degrees."""
    if "derived_affine" in entry:
        fit = entry["derived_affine"]
        slope, offset = finite(fit["slope"]), finite(fit["offset"])
        low, high = map(finite, fit["hardware_range"])
        if abs(slope) < 1e-6 or low >= high:
            raise ValueError("Invalid derived joint conversion")
        return {"slope": slope, "offset": offset, "hardware_range": [low, high],
                "angle_range": sorted([slope*low+offset, slope*high+offset])}
    if not entry.get("measurement_method", "").strip():
        raise ValueError("measurement_method is required")
    samples = entry.get("samples", [])
    if len(samples) < 2:
        raise ValueError("needs two measured hardware/angle pairs spanning the motion")
    hardware = np.array([finite(s["hardware_normalized"]) for s in samples])
    angles = np.array([finite(s["mujoco_rad"]) for s in samples])
    if np.ptp(hardware) < 1 or np.ptp(angles) < math.radians(5):
        raise ValueError("reference span is too small")
    slope, offset = np.linalg.lstsq(np.c_[hardware, np.ones(len(samples))], angles, rcond=None)[0]
    if abs(slope) < 1e-6 or np.max(np.abs(slope * hardware + offset - angles)) > .05:
        raise ValueError("references do not describe a consistent signed linear mapping")
    return {"slope": float(slope), "offset": float(offset),
            "hardware_range": [float(hardware.min()), float(hardware.max())],
            "angle_range": [float(angles.min()), float(angles.max())]}


def motor_calibration_summary(snapshot, host):
    """Import existing motor data without pretending it specifies CAD gearing."""
    if snapshot["host"] != host:
        raise ValueError("Motor calibration snapshot belongs to a different robot")
    result = {}
    for side in ("left", "right"):
        for role in ROLES:
            motor = snapshot["arms"][side][role]
            low, high = finite(motor["range_min"]), finite(motor["range_max"])
            if low >= high:
                raise ValueError(f"Invalid motor calibration: {side}_{role}")
            result[f"{side}_{role}"] = {
                **motor, "encoder_ticks_per_normalized_unit_magnitude":
                (high-low) / (100 if role == "gripper" else 200),
                "cad_direction_and_gearing_defined_in_this_file": False,
            }
    return result


def collect(config, speed):
    frames = []
    names = [f"{config['side']}_{role}" for role in ROLES]

    def sample(sim):
        # 25 Hz plus exact initial state. Use actual simulated joint motion,
        # rather than the discontinuous internal position-servo setpoints.
        if frames and sim.data.time - frames[-1]["time_s"] < .04 - 1e-7:
            return
        frames.append({"time_s": float(sim.data.time), "qpos": sim.data.qpos.copy(),
                       "joint_rad": {n: float(sim.data.qpos[sim.model.joint(n).qposadr[0]]) for n in names}})

    result = run(config, speed=speed, on_step=sample)
    return result, frames


def preflight(config, result, frames, calibration, profile, policy, host, motion_only=False):
    errors, fits = [], {}
    validate_profile(profile, host)
    validate_host_limits({"start": profile["positions"], "target": profile["positions"]}, policy, host)
    if not calibration:
        errors.append("Replay mapping file is missing; this does not mean robot motor calibration is invalid")
    elif calibration.get("version") != 1 or calibration.get("host") != host or calibration.get("units") != "normalized":
        errors.append("Joint mapping must match this host, version 1, and normalized units")
    if not profile.get("stationary_hold_verified"):
        errors.append("Starting profile has no verified stationary hold")
    if config.get("illustrative", True) and not motion_only:
        errors.append("Scene still uses illustrative dimensions; physical table/ball placement must match the scene")
    if not result["simulation_passed"] or not result.get("pickup", {}).get("passed"):
        errors.append("Simulation approach/pickup did not pass")
    if result.get("table_clearance", {}).get("robot_table_contact_steps", 1):
        errors.append("Simulation has robot/table contact or no clearance report")
    ranges = {}
    for role in ROLES:
        name = f"{config['side']}_{role}"
        values = np.array([f["joint_rad"][name] for f in frames])
        ranges[name] = [float(values.min()), float(values.max())]
        entry = calibration.get("joints", {}).get(name)
        if entry is None:
            if calibration:
                errors.append(f"{name}: no joint conversion supplied")
            continue
        if "derived_affine" in entry:
            status = entry.get("verification_status")
            source_confirmed = (status == "source_reference_confirmed"
                                and bool(entry.get("verification_evidence", "").strip()))
            if status != "hardware_verified" and not source_confirmed:
                errors.append(f"{name}: imported Unity conversion needs a supervised direction/travel check")
        try:
            fit = joint_fit(entry)
            fits[name] = fit
            low, high = fit["angle_range"]
            if values.min() < low - .002 or values.max() > high + .002:
                errors.append(f"{name}: trajectory {values.min():.3f}..{values.max():.3f} rad exceeds mapping range {low:.3f}..{high:.3f}; no extrapolation")
            hw = (values - fit["offset"]) / fit["slope"]
            key = name + ".pos"
            # Include both ends of the slow initialization, including gripper opening.
            check_values = np.r_[hw, profile["positions"][key]]
            bounds = fit["hardware_range"]
            if check_values.min() < bounds[0] - .1 or check_values.max() > bounds[1] + .1:
                errors.append(f"{name}: requested normalized range {check_values.min():.2f}..{check_values.max():.2f} exceeds mapping range {bounds[0]:.2f}..{bounds[1]:.2f}")
            # Simulated servo settling can overshoot an endpoint by a tiny
            # fraction of one readback unit. Conversion clips only this small
            # allowance; large travel violations still fail preflight.
            if check_values.min() < (0 if role == "gripper" else -100) - .05 or check_values.max() > 100 + .05:
                errors.append(f"{name}: requested normalized positions exceed servo limits")
            if role != "gripper":
                anchor = fit["slope"] * profile["positions"][key] + fit["offset"]
                if abs(anchor - values[0]) > .03:
                    errors.append(f"{name}: mapping disagrees with verified down pose by {math.degrees(abs(anchor-values[0])):.1f} degrees")
            if role == "shoulder_lift":
                limit = policy["limits"][key]
                if hw.min() < limit["min"] - 1e-8 or hw.max() > limit["max"] + 1e-8:
                    errors.append(f"{name}: host guard would replace a trajectory target")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"{name}: {exc}")
    return {"ready": not errors, "errors": errors, "simulated_joint_ranges_rad": ranges,
            "joint_fits": fits, "simulation": result, "hardware_commands_sent": False,
            "hardware_mode": "motion-only test; no physical grasp validation" if motion_only else "physical ball pickup",
            "mapping_verification": {name: entry.get("verification_status", "measured angle references")
                                     for name, entry in calibration.get("joints", {}).items()
                                     if name.startswith(config["side"] + "_")}}


def prepare_curve(frames, fits, profile, speed):
    """Build nominal path timing; live feedback controls actual playback pace."""
    speed = finite(speed)
    if not 0 < speed <= 2:
        raise ValueError("speed must be greater than 0 and at most 2")
    start = np.array([profile["positions"][k] for k in KEYS])
    rows = []
    for frame in frames:
        row = start.copy()
        for name, angle in frame["joint_rad"].items():
            fit = fits[name]
            row[KEYS.index(name + ".pos")] = np.clip(
                (angle - fit["offset"]) / fit["slope"], 0 if name.endswith("_gripper") else -100, 100)
        rows.append(row)
    # A separate two-second normal-speed gripper opening occurs while parked.
    # The pickup itself uses the exact normal-speed simulation timestamps.
    times = np.r_[0., .25, 2.25 + np.array([f["time_s"] for f in frames])] / speed
    values = np.vstack([start, start, rows])
    return PchipInterpolator(times, values, axis=0), 1.


def curve_target(curve, elapsed):
    target = {}
    for key, raw in zip(KEYS, curve(elapsed)):
        value = finite(raw)
        low = 0 if key.endswith("gripper.pos") else -100
        if value < low - 1e-7 or value > 100 + 1e-7:
            raise ValueError(f"Replay curve outside normalized range: {key}={value}")
        target[key] = min(100., max(float(low), value))
    return positions(target)


def go_to_start(link, profile, policy, host, speed, log, *,
                tracking_tolerance=REPLAY_TRACKING_TOLERANCE, max_delta=30,
                stall_timeout=REPLAY_STALL_TIMEOUT):
    """Hold the live pose before ramping; replay does not capture calibration."""
    tracking_tolerance = validate_tracking_tolerance(tracking_tolerance)
    stall_timeout = validate_stall_timeout(stall_timeout)
    observed = link.read()
    check_feedback(observed, observed["positions"], profile["elevator"])
    # A retry can begin with an open gripper after the preparation stage.
    # Motion-only callers can also permit a full return from a partial replay.
    motion = plan(profile, observed, speed, max_delta=max_delta, max_gripper_delta=100)
    validate_host_limits(motion, policy, host)
    held = motion["start"].copy()
    sent = False
    try:
        # The first command is the freshest measured pose, never the saved
        # start or a median from an unpowered arm's observation window.
        # Logging also checks that the viewer is still open before sending.
        log({"phase": "hold-current-start", "observed": observed, "commanded": held})
        link.send(held)
        sent = True
        deadline, settled = time.monotonic() + stall_timeout, None
        while True:
            observed = link.read()
            now = time.monotonic()
            check_feedback(observed, held, profile["elevator"])
            error = replay_tracking_error(observed, held, tracking_tolerance)
            log({"phase": "hold-current", "observed": observed,
                 "commanded": held, "tracking": error})
            settled = (now if settled is None else settled) if error["within_tolerance"] else None
            if settled is not None and now - settled >= 1:
                break
            if now >= deadline:
                raise TimeoutError(
                    f"Current-pose hold did not settle at {error['joint']}: "
                    f"{error['error_units']:.3f} units; replay has not started")
            link.send(held)
            time.sleep(1 / 30)
    except BaseException:
        if sent:
            try:
                link.send(held)
            except Exception:
                pass
        raise
    log({"phase": "go-to-start-plan", "motion": motion})
    return execute(link, motion, profile["elevator"], log,
                   tracking_tolerance=tracking_tolerance, stall_timeout=stall_timeout)


def replay_tracking_error(observed, reference, tolerance=REPLAY_TRACKING_TOLERANCE):
    """Check the largest joint error against the configured tracking window."""
    tolerance = validate_tracking_tolerance(tolerance)
    error = tracking_error(observed, reference)
    return {**error, "tolerance_units": tolerance,
            "within_tolerance": error["error_units"] <= tolerance}


def advance_curve(curve, elapsed, last, observed, dt, tolerance):
    """Advance along the same joint path only as far as live feedback allows.

    Bound the proposed target, not just the previous target's tracking error.
    Check intervening knots so a fast out-and-back cannot skip an excursion.
    All joints share one path time; no independent joint clipping cuts corners.
    """
    current = positions(observed["positions"])
    horizon = min(float(curve.x[-1]), elapsed + max(0., dt))
    step_limit = min(REPLAY_MAX_COMMAND_STEP, tolerance / 2)

    def permitted(target):
        return (max(abs(target[key] - current[key]) for key in KEYS) <= tolerance
                and max(abs(target[key] - last[key]) for key in KEYS) <= step_limit)

    if not permitted(last):
        return elapsed, last, True
    lower, target = elapsed, last
    knots = curve.x[(curve.x > elapsed) & (curve.x < horizon)]
    for upper in (*knots, horizon):
        candidate = curve_target(curve, upper)
        if permitted(candidate):
            lower, target = float(upper), candidate
            continue
        # PCHIP preserves component monotonicity between adjacent knots.
        # Locate the last feasible point before the first limiting boundary.
        for _ in range(24):
            middle = (lower + upper) / 2
            candidate = curve_target(curve, middle)
            if permitted(candidate):
                lower, target = middle, candidate
            else:
                upper = middle
        return lower, target, True
    return horizon, target, False


def replay(link, curve, elevator, log, show=None, *, tracking_tolerance=REPLAY_TRACKING_TOLERANCE,
           stall_timeout=REPLAY_STALL_TIMEOUT):
    tracking_tolerance = validate_tracking_tolerance(tracking_tolerance)
    stall_timeout = validate_stall_timeout(stall_timeout)
    elapsed, previous = 0., time.monotonic()
    last = curve_target(curve, 0)
    started = progress_at = previous
    progress_target = last.copy()
    sent = False
    settled = None
    deadline = None
    try:
        while True:
            observed = link.read()
            now = time.monotonic()
            check_feedback(observed, last, elevator)
            dt, previous = min(max(now - previous, 0), 1 / 30), now
            next_elapsed, target, limited = advance_curve(
                curve, elapsed, last, observed, dt, tracking_tolerance)
            command_step = max(abs(target[key] - last[key]) for key in KEYS)
            # A slow joint may take arbitrarily longer than the nominal path.
            # Abort only after no meaningful path progress, not while it moves.
            if (not limited or max(abs(target[key] - progress_target[key]) for key in KEYS)
                    >= min(.1, tracking_tolerance / 10)):
                progress_at, progress_target = now, target.copy()
            error = replay_tracking_error(observed, target, tracking_tolerance)
            if limited and now - progress_at > stall_timeout:
                raise TimeoutError(
                    f"Replay made no progress for {stall_timeout:g} s at {error['joint']}: "
                    f"observed {error['observed']:.3f}, commanded {error['commanded']:.3f}, "
                    f"error {error['error_units']:.3f} units (tolerance {error['tolerance_units']:.3f})")
            elapsed, last = next_elapsed, target
            if show is not None:
                show(elapsed, observed)
            link.send(last)
            sent = True
            log({"phase": "replay", "trajectory_s": elapsed, "wall_elapsed_s": now - started,
                 "observed": observed, "command_step_units": command_step,
                 "commanded": last, "tracking": error,
                 "waiting_for_tracking": limited})
            if elapsed >= curve.x[-1]:
                deadline = deadline or now + max(8., stall_timeout)
                settled = (settled or now) if error["within_tolerance"] else None
                if settled is not None and now - settled >= 1:
                    return
                if now > deadline:
                    raise TimeoutError("Final pose did not settle")
            time.sleep(1 / 30)
    finally:
        if sent:
            # Hold the last command on completion, window close, or error.
            # This does not disable torque or implement an emergency stop.
            try:
                link.send(last)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("examples/ball_approach.example.json"))
    parser.add_argument("--mapping", type=Path, default=Path("artifacts/robot_joint_mapping.json"))
    parser.add_argument("--motor-calibration", type=Path,
                        default=Path("artifacts/robot_diagnostics/active_motor_calibration.json"))
    parser.add_argument("--profile", type=Path, default=Path("artifacts/robot_start_pose_verified.json"))
    parser.add_argument("--host-limits", type=Path, default=Path("artifacts/robot_host_limits.json"))
    parser.add_argument("--robot-ip", default="192.168.1.79")
    parser.add_argument("--speed", type=float, default=.5,
                        help="Requested playback multiplier, greater than 0 and at most 2; feedback may slow it")
    parser.add_argument("--tracking-tolerance", type=float, default=REPLAY_TRACKING_TOLERANCE,
                        help="Permitted arm/gripper lag in normalized units (default: 5; must be below 10)")
    parser.add_argument("--stall-timeout", type=float, default=REPLAY_STALL_TIMEOUT,
                        help="Seconds without meaningful progress before aborting (default: 15)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/robot_ball_preflight.json"))
    parser.add_argument("--trace", type=Path, default=Path("artifacts/robot_ball_replay.jsonl"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--motion-only", action="store_true",
                        help="Test the arm path in clear space without claiming a physical grasp; permits illustrative scene dimensions")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    try:
        args.tracking_tolerance = validate_tracking_tolerance(args.tracking_tolerance)
        args.stall_timeout = validate_stall_timeout(args.stall_timeout)
        if not 0 < finite(args.speed) <= 2:
            raise ValueError("speed must be greater than 0 and at most 2")
        config, profile, policy = load(args.scene), load(args.profile), load(args.host_limits)
        calibration = load(args.mapping) if args.mapping.exists() else {}
        candidate = Path("artifacts/robot_joint_mapping.unity_candidate.json")
        mapping_source = args.mapping if args.mapping.exists() else None
        if not calibration and args.mapping == Path("artifacts/robot_joint_mapping.json") and candidate.exists():
            calibration, mapping_source = load(candidate), candidate
        # Record once at normal simulation speed, then scale its timestamps.
        # Running physics faster than real time is offline computation, not
        # the motion duration. --speed 0.5 must double the recorded duration.
        computation_started = time.perf_counter()
        result, frames = collect(config, 1.)
        report = preflight(config, result, frames, calibration, profile, policy, args.robot_ip,
                           motion_only=args.motion_only)
        report["mapping_source"] = str(mapping_source) if mapping_source else None
        if args.motor_calibration.exists():
            snapshot = load(args.motor_calibration)
            report["saved_motor_calibration"] = motor_calibration_summary(snapshot, args.robot_ip)
            report["motor_calibration_source_sha256"] = snapshot["source_files"]
        if report["ready"]:
            curve, dilation = prepare_curve(frames, report["joint_fits"], profile, args.speed)
            report.update(duration_s=float(curve.x[-1]), additional_time_dilation=dilation,
                          nominal_duration_s=float(curve.x[-1]), duration_is_nominal=True,
                          speed=args.speed, normal_pickup_duration_s=float(frames[-1]["time_s"]),
                          pickup_duration_s=float(frames[-1]["time_s"] / args.speed),
                          parked_gripper_preparation_s=2.25 / args.speed,
                          tracking_tolerance_units=args.tracking_tolerance,
                          timing_mode="adaptive path following; speed is a ceiling, feedback can extend duration",
                          maximum_command_step_units=REPLAY_MAX_COMMAND_STEP,
                          no_progress_timeout_s=args.stall_timeout,
                          return_to_start_duration_s="depends on current pose; separate from pickup timing")
        report["offline_computation_wall_s"] = time.perf_counter() - computation_started
        report["phase"] = "offline-preflight"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in (
            "simulation", "joint_fits", "saved_motor_calibration", "motor_calibration_source_sha256")}, indent=2))
        if not args.execute:
            return 0 if report["ready"] else 2
        if not report["ready"]:
            raise ValueError("Preflight failed; no robot connection or movement commands made")
        if not args.normalized:
            raise ValueError("--normalized is required to confirm the host command units")
        # Open the viewer before any motion; closing it aborts the operation.
        from mujoco import viewer
        model = scene_model(config)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("startup").id)
        fits = report["joint_fits"]
        with viewer.launch_passive(model, data) as window:
            window.cam.lookat[:] = [.4, .18, .85]
            window.cam.distance, window.cam.azimuth, window.cam.elevation = 1.35, 180, -18

            def show(elapsed, observed):
                if not window.is_running():
                    raise KeyboardInterrupt("MuJoCo window closed")
                # Display measured joint angles. The ball remains a scene marker;
                # no claim that its real pose is observed by this bridge.
                for name, fit in fits.items():
                    data.qpos[model.joint(name).qposadr[0]] = fit["slope"] * observed["positions"][name + ".pos"] + fit["offset"]
                mujoco.mj_forward(model, data)
                window.sync()

            args.trace.parent.mkdir(parents=True, exist_ok=True)
            with args.trace.open("w", encoding="utf-8") as trace:
                last_progress_print = 0.

                def log(record):
                    nonlocal last_progress_print
                    trace.write(json.dumps(record) + "\n")
                    trace.flush()
                    message = {
                        "hold-current-start": "Holding the live robot pose...",
                        "go-to-start-plan": "Returning to the saved starting pose...",
                        "replay-start": "Following the recorded MuJoCo path at the real arm's pace...",
                    }.get(record["phase"])
                    if message:
                        print(message, flush=True)
                    if (record["phase"] == "replay"
                            and record["wall_elapsed_s"] - last_progress_print >= 2):
                        last_progress_print = record["wall_elapsed_s"]
                        progress = record["trajectory_s"] / float(curve.x[-1])
                        suffix = "; slowing for the real arm" if record["waiting_for_tracking"] else ""
                        print(f"Path {progress:.0%} complete{suffix}", flush=True)
                    if "observed" in record:
                        show(0, record["observed"])
                link = HardwareLink(args.robot_ip)
                try:
                    go_to_start(link, profile, policy, args.robot_ip, args.speed, log,
                                tracking_tolerance=args.tracking_tolerance,
                                stall_timeout=args.stall_timeout,
                                max_delta=200 if args.motion_only else 30)
                    log({"phase": "replay-start"})
                    replay(link, curve, profile["elevator"], log,
                           tracking_tolerance=args.tracking_tolerance, stall_timeout=args.stall_timeout)
                    log({"phase": "complete"})
                    print("Replay complete; holding final pose. Trace:", args.trace)
                finally:
                    link.close()
    except (ValueError, KeyError, OSError, TimeoutError, KeyboardInterrupt) as exc:
        print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
