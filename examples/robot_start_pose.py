"""Lower real arms using recorded down captures, or return to a saved start.

No MuJoCo-to-hardware conversion or grasp replay. Install requirements-robot.txt.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
import time

from examples.calibrate_arm_directions import save

ROLES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
KEYS = tuple(f"{side}_{role}.pos" for side in ("left", "right") for role in ROLES)


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite hardware value")
    return value


def positions(values):
    result = {key: finite(values[key]) for key in KEYS}
    for key, value in result.items():
        if not (0 if key.endswith("gripper.pos") else -100) <= value <= 100:
            raise ValueError(f"{key} outside normalized hardware range")
    return result


def validate_profile(profile, host):
    if profile["version"] != 1 or profile["host"] != host:
        raise ValueError("Start profile version/host mismatch; capture a profile for this host")
    if profile["units"] != "normalized" or profile["pose"] != "operator-positioned arms down":
        raise ValueError("Expected an operator-captured normalized arms-down profile")
    positions(profile["positions"])
    finite(profile["elevator"])


def arms_down_profile(checkpoint, host, observed):
    """Reuse measured hardware targets; never guess radians-to-servo mappings."""
    if checkpoint["version"] != 1 or checkpoint["source"]["host"] != host:
        raise ValueError("Arm-direction captures belong to a different host/version")
    targets = positions(observed["positions"])
    for side in ("left", "right"):
        pose = checkpoint["poses"][f"{side}_down"]
        shoulder = pose["mujoco_shoulder_rad"]
        expected_lift = -math.pi/2 if side == "left" else math.pi/2
        if abs(finite(shoulder["shoulder_lift"]) - expected_lift) > .05:
            raise ValueError(f"{side}_down is not a downward shoulder reference")
        for role in ROLES[:-1]:
            targets[f"{side}_{role}.pos"] = pose["hardware_positions"][role]
    return {"version": 1, "host": host, "units": "normalized",
            "pose": "operator-positioned arms down", "positions": positions(targets),
            "elevator": finite(observed["elevator"]),
            "source": "recorded left_down/right_down hardware positions; grippers preserved"}


def plan(profile, observed, speed, max_delta=30, *, max_gripper_delta=None):
    speed = finite(speed)
    if not 0 < speed <= 2:
        raise ValueError("speed must be greater than 0 and at most 2")
    start, target = positions(observed["positions"]), positions(profile["positions"])
    if abs(finite(observed["elevator"]) - finite(profile["elevator"])) > 2:
        raise ValueError("Elevator differs from captured start; position it manually first")
    distances = {key: abs(target[key] - start[key]) for key in KEYS}
    for key, distance in distances.items():
        limit = (max_gripper_delta if max_gripper_delta is not None and key.endswith("gripper.pos")
                 else max_delta)
        if distance > limit:
            raise ValueError(
                f"Start pose is over {limit} normalized units away at {key}; teleoperate closer first")
    # Gripper travel still contributes to the ramp's velocity/acceleration limits.
    delta = max(distances.values())
    # Quintic blend: peak derivative 1.875, peak second derivative ~5.774.
    # These command limits are engineering defaults, not measured motor ratings.
    duration = max(5., 1.875 * delta / 10., math.sqrt(5.774 * delta / 20.)) / speed
    return {"start": start, "target": target, "duration_s": duration,
            "speed": speed, "maximum_delta_units": delta,
            "command_velocity_limit_units_s": 10 * speed,
            "command_acceleration_limit_units_s2": 20 * speed**2}


def interpolate(motion, elapsed):
    t = max(0., min(1., elapsed / motion["duration_s"]))
    blend = t**3 * (10 - 15 * t + 6 * t*t)
    return {key: motion["start"][key] + blend * (motion["target"][key] - motion["start"][key])
            for key in KEYS}


def validate_host_limits(motion, policy, host):
    """Reject positions the host would replace before the servo receives them."""
    if policy["version"] != 1 or policy["host"] != host or policy["units"] != "normalized":
        raise ValueError("Host-limit snapshot does not match this robot/units")
    for side in ("left", "right"):
        joint = side + "_shoulder_lift.pos"
        limit = policy["limits"][joint]
        low, high = finite(limit["min"]), finite(limit["max"])
        if low >= high:
            raise ValueError(f"Invalid host limits for {joint}")
        for phase in ("start", "target"):
            value = finite(motion[phase][joint])
            if not low <= value <= high:
                raise ValueError(
                    f"Host safety guard rejects {phase} {joint}={value:.3f}; "
                    f"permitted range is {low:.3f} to {high:.3f}. "
                    "No movement commands sent. The down reference and host safety range "
                    "must be reconciled; do not disable the guard or retry at higher speed."
                )


def tracking_error(observed, reference):
    current = positions(observed["positions"])
    joint = max(KEYS, key=lambda key: abs(current[key] - reference[key]))
    return {"joint": joint, "observed": current[joint], "commanded": reference[joint],
            "error_units": abs(current[joint] - reference[joint])}


def error_description(error):
    return (f"{error['joint']}: observed {error['observed']:.3f}, "
            f"commanded {error['commanded']:.3f}, error {error['error_units']:.3f} units")


def check_feedback(observed, reference, elevator):
    error = tracking_error(observed, reference)
    if error["error_units"] > 10:
        raise ValueError(f"Arm tracking error exceeds 10 normalized units; stopping the ramp. {error_description(error)}")
    if abs(finite(observed["elevator"]) - elevator) > 2:
        raise ValueError("Elevator moved during go-to-start")
    if any(abs(finite(value)) > .02 for value in observed["base_velocity"]):
        raise ValueError("Base is moving; stop it before go-to-start")


class HardwareLink:
    def __init__(self, host, observation_port=5556, command_port=5555):
        import zmq
        # The wire schema has no LeRobot/torch dependency. Keep simulation and
        # replay in one environment instead of importing the full robot plugin.
        from integrations import sourccey_pb2
        self.zmq, self.pb = zmq, sourccey_pb2
        self.context = zmq.Context()
        self.observations = self.context.socket(zmq.PULL)
        self.observations.setsockopt(zmq.CONFLATE, 1)
        self.observations.setsockopt(zmq.LINGER, 0)
        self.observations.connect(f"tcp://{host}:{observation_port}")
        self.command_endpoint = f"tcp://{host}:{command_port}"
        self.commands = None
        self.first_read = True

    def read(self):
        timeout = 5000 if self.first_read else 500
        if not self.observations.poll(timeout):
            raise TimeoutError(f"No fresh robot observation within {timeout} ms")
        self.first_read = False
        state = self.pb.SourcceyRobotState.FromString(self.observations.recv())
        required = ("left_arm_joints", "right_arm_joints", "base_position", "base_velocity")
        if not all(state.HasField(field) for field in required):
            raise ValueError("Observation is missing arm/base/elevator state")
        raw = {f"{side}_{role}.pos": getattr(getattr(state, f"{side}_arm_joints"), role)
               for side in ("left", "right") for role in ROLES}
        return {"positions": positions(raw), "elevator": finite(state.base_position.z_pos),
                "base_velocity": [finite(getattr(state.base_velocity, field))
                                  for field in ("x_vel", "y_vel", "theta_vel", "z_vel")]}

    def send(self, target):
        # Created only after --execute; capture and dry-run never open a command socket.
        if self.commands is None:
            self.commands = self.context.socket(self.zmq.PUSH)
            self.commands.setsockopt(self.zmq.CONFLATE, 1)
            self.commands.setsockopt(self.zmq.LINGER, 0)
            self.commands.connect(self.command_endpoint)
        action = self.pb.SourcceyRobotAction()
        for side in ("left", "right"):
            arm = getattr(action, f"{side}_arm_target_joints")
            for role in ROLES:
                setattr(arm, role, target[f"{side}_{role}.pos"])
        action.base_target_velocity.SetInParent()  # Explicit zero base and Z velocities.
        # Omit base_target_position: host preserves the existing elevator target.
        if not self.commands.poll(250, self.zmq.POLLOUT):
            raise TimeoutError("Robot command connection is not writable")
        self.commands.send(action.SerializeToString(), flags=self.zmq.NOBLOCK)

    def close(self):
        self.observations.close(0)
        if self.commands is not None:
            self.commands.close(0)
        self.context.term()


def capture(link, host):
    samples = [link.read() for _ in range(10)]
    for key in KEYS:
        values = [s["positions"][key] for s in samples]
        if max(values) - min(values) > 1:
            raise ValueError(f"{key} moved during capture; hold the pose steady")
    elevator = median(s["elevator"] for s in samples)
    for sample in samples:
        check_feedback(sample, samples[0]["positions"], elevator)
    return {"version": 1, "host": host, "units": "normalized",
            "pose": "operator-positioned arms down", "captured_at": datetime.now(timezone.utc).isoformat(),
            "positions": {key: median(s["positions"][key] for s in samples) for key in KEYS},
            "elevator": elevator}


def validate_tracking_tolerance(value):
    value = finite(value)
    if not 0 < value < 10:
        raise ValueError("tracking tolerance must be greater than 0 and below the 10-unit abort limit")
    return value


def validate_stall_timeout(value):
    value = finite(value)
    if value <= 0:
        raise ValueError("stall timeout must be greater than 0")
    return value


def execute(link, motion, elevator, on_sample=None, *, tracking_tolerance=2., stall_timeout=4.):
    """Rate-limited targets; no catch-up jump after a slow loop."""
    tracking_tolerance = validate_tracking_tolerance(tracking_tolerance)
    stall_timeout = validate_stall_timeout(stall_timeout)
    elapsed = 0.
    last_target = motion["start"].copy()
    sent_any = False
    previous = time.monotonic()
    started = previous
    waiting_since = None
    try:
        while elapsed < motion["duration_s"]:
            observed = link.read()
            now = time.monotonic()
            error = tracking_error(observed, last_target)
            error["tolerance_units"] = tracking_tolerance
            if on_sample is not None:
                on_sample({"phase": "ramp", "wall_elapsed_s": now - started,
                           "ramp_elapsed_s": elapsed, "observed": observed,
                           "commanded": last_target.copy(), "tracking": error,
                           "waiting_for_tracking": error["error_units"] > tracking_tolerance})
            check_feedback(observed, last_target, elevator)
            dt = min(now - previous, 1 / 30)
            previous = now
            if error["error_units"] > tracking_tolerance:
                # Hold this small waypoint until feedback catches up instead
                # of allowing a slow/stalled joint to accumulate a large lead.
                waiting_since = waiting_since if waiting_since is not None else now
                if now - waiting_since > stall_timeout:
                    raise TimeoutError(f"Joint did not catch up within {stall_timeout:g} s; stopping the ramp. {error_description(error)}")
                if sent_any:
                    link.send(last_target)
                time.sleep(1 / 30)
                continue
            waiting_since = None
            elapsed = min(motion["duration_s"], elapsed + dt)
            target = interpolate(motion, elapsed)
            link.send(target)
            sent_any = True
            last_target = target
            time.sleep(1 / 30)
        deadline, settled_since = time.monotonic() + max(8., stall_timeout), None
        while time.monotonic() < deadline:
            observed = link.read()
            if on_sample is not None:
                on_sample({"phase": "settle", "wall_elapsed_s": time.monotonic() - started,
                           "ramp_elapsed_s": elapsed, "observed": observed,
                           "commanded": last_target.copy(),
                           "tracking": tracking_error(observed, last_target)})
            check_feedback(observed, last_target, elevator)
            error = max(abs(observed["positions"][k] - motion["target"][k]) for k in KEYS)
            if error <= tracking_tolerance:
                settled_since = settled_since or time.monotonic()
                if time.monotonic() - settled_since >= 1:
                    return {"start_reached": True, "max_error_units": error}
            else:
                settled_since = None
            link.send(last_target)
            time.sleep(1 / 30)
        raise TimeoutError(f"Start pose did not settle within {tracking_tolerance:g} units; do not begin grasp replay. "
                           + error_description(tracking_error(observed, motion["target"])))
    except BaseException as exc:
        # Hold the last sent target, rather than jump to a possibly stale readback.
        # This is not an emergency stop or a torque release.
        if sent_any:
            try:
                link.send(last_target)
            except Exception:
                pass
        if on_sample is not None:
            on_sample({"phase": "aborted", "error": str(exc),
                       "ramp_elapsed_s": elapsed, "last_commanded": last_target,
                       "sent_any": sent_any})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("hold-current", "lower-arms", "capture", "go-to-start"))
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--profile", type=Path, default=Path("artifacts/robot_start_pose.json"))
    parser.add_argument("--calibration", type=Path, default=Path("artifacts/arm_directions.json"),
                        help="Existing down-pose captures for lower-arms")
    parser.add_argument("--speed", type=float, default=.5)
    parser.add_argument("--trace", type=Path, default=Path("artifacts/robot_start_pose_trace.jsonl"),
                        help="Feedback/command trace written during --execute")
    parser.add_argument("--host-limits", type=Path, default=Path("artifacts/robot_host_limits.json"),
                        help="SSH-inspected host safety limits; required for --execute")
    parser.add_argument("--execute", action="store_true", help="Enable physical go-to-start motion")
    parser.add_argument("--normalized", action="store_true",
                        help="Confirm host uses normalized arm units, not degree mode")
    args = parser.parse_args()
    if not args.normalized:
        parser.error("Confirm the active host uses normalized joint units with --normalized")
    if args.mode == "capture" and args.execute:
        parser.error("capture reads observations only; omit --execute")
    if not math.isfinite(args.speed) or not 0 < args.speed <= 1:
        parser.error("speed must be greater than 0 and at most 1")
    if args.execute and not args.host_limits.is_file():
        parser.error("--execute requires a host safety-limit snapshot; inspect the host first")
    link = None
    try:
        profile = None
        if args.mode == "go-to-start":
            profile = json.loads(args.profile.read_text(encoding="utf-8"))
            validate_profile(profile, args.robot_ip)
        link = HardwareLink(args.robot_ip)
        if args.mode == "capture":
            profile = capture(link, args.robot_ip)
            save(args.profile, profile)
            print(f"Saved real start pose: {args.profile}")
        else:
            if args.mode == "hold-current":
                capture(link, args.robot_ip)  # Require a stable, stationary observation window first.
            observed = link.read()
            if args.mode == "lower-arms":
                checkpoint = json.loads(args.calibration.read_text(encoding="utf-8"))
                profile = arms_down_profile(checkpoint, args.robot_ip, observed)
            elif args.mode == "hold-current":
                profile = {"positions": observed["positions"].copy(), "elevator": observed["elevator"]}
            check_feedback(observed, observed["positions"], profile["elevator"])
            # The named down-pose capture supports a full-range slow return;
            # arbitrary user start profiles retain the nearby-pose restriction.
            motion = plan(profile, observed, args.speed,
                          max_delta=200 if args.mode == "lower-arms" else 30)
            if args.mode == "hold-current":
                assert motion["start"] == motion["target"]
                motion["duration_s"] = 2.
                motion["hold_current_only"] = True
            print(json.dumps(motion, indent=2))
            if args.host_limits.is_file():
                policy = json.loads(args.host_limits.read_text(encoding="utf-8"))
                validate_host_limits(motion, policy, args.robot_ip)
            else:
                print("Host safety limits have not been validated. Movement remains disabled.")
            if args.execute:
                args.trace.parent.mkdir(parents=True, exist_ok=True)
                print(f"Feedback trace: {args.trace}")
                with args.trace.open("w", encoding="utf-8") as stream:
                    def record(sample):
                        stream.write(json.dumps(sample, allow_nan=False) + "\n")
                        stream.flush()
                    record({"phase": "plan", "mode": args.mode, "motion": motion})
                    result = execute(link, motion, observed["elevator"], on_sample=record)
                    record({"phase": "complete", **result})
                print(json.dumps(result, indent=2))
            else:
                print("Dry run only. Add --execute to move the real robot to the starting position.")
    except KeyboardInterrupt:
        parser.exit(1, "Stopped ramp; last target held. Use the hardware stop if needed.\n")
    except (ValueError, KeyError, OSError, TimeoutError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")
    finally:
        if link is not None:
            link.close()


if __name__ == "__main__":
    main()
