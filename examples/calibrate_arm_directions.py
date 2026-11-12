"""Record cardinal directions from Sourccey and replay the shoulder mapping."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile

SIDES = ("left", "right")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
DIRECTIONS = {
    "forward": ((1, 0, 0), "horizontal toward the robot's front"),
    "down": ((0, 0, -1), "vertically down, keeping shoulder pan at the forward setting"),
    "up": ((0, 0, 1), "vertically up, keeping shoulder pan at the forward setting"),
    "left": ((0, 1, 0), "horizontal toward the robot's left"),
    "right": ((0, -1, 0), "horizontal toward the robot's right"),
    "forward_out": ((1, 1, 0), "horizontal halfway between forward and outward"),
    "back": ((-1, 0, 0), "horizontal toward the robot's back"),
}


def save(path, session):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", delete=False, mode="w", encoding="utf-8"
        ) as stream:
            temporary = Path(stream.name)
            json.dump(session, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def references(sim):
    """Match the shoulder-to-elbow vector in the simulator's robot frame."""
    import mujoco
    import numpy as np

    model, data = sim.model, sim.data
    rotation = sim.robot_rotation.copy()
    result = {}
    for side in SIDES:
        pan, lift, elbow = [model.joint(f"{side}_{joint}").id for joint in JOINTS[:3]]
        addresses = model.jnt_qposadr[[pan, lift]]
        limits = model.jnt_range[[pan, lift]]
        for name, (direction, _) in DIRECTIONS.items():
            if name == "forward_out":
                direction = (math.sqrt(0.5), math.sqrt(0.5) * (1 if side == "left" else -1), 0)
            bounds = limits.copy()
            vertical = name in ("up", "down")
            if vertical:
                bounds[0] = result[f"{side}_forward"][0]["shoulder_pan"]
            best_score = -math.inf
            for samples in (25, 9, 9):
                pan_grid, lift_grid = [
                    np.unique(np.linspace(low, high, samples)) for low, high in bounds
                ]
                for pan_angle in pan_grid:
                    for lift_angle in lift_grid:
                        data.qpos[addresses] = angles = (pan_angle, lift_angle)
                        mujoco.mj_kinematics(model, data)
                        vector = rotation.T @ (data.xanchor[elbow] - data.xanchor[lift])
                        score = float(vector @ direction / np.linalg.norm(vector))
                        if score > best_score:
                            best_score, best = score, angles
                step = (bounds[:, 1] - bounds[:, 0]) / (samples - 1)
                bounds = np.column_stack((np.maximum(limits[:, 0], best - step),
                                          np.minimum(limits[:, 1], best + step)))
            error = math.degrees(math.acos(max(-1.0, min(1.0, best_score))))
            result[f"{side}_{name}"] = (dict(zip(JOINTS[:2], map(float, best))), error)
        sim.reset()
    return result


def read_pose(socket, message_type, side, max_spread):
    import zmq

    # Discard the queued packet so a pause at the prompt cannot reuse old data.
    try:
        socket.recv(zmq.DONTWAIT)
    except zmq.Again:
        pass
    samples = []
    for _ in range(5):
        if not socket.poll(2000):
            raise ValueError("No new observation from the Sourccey host.")
        state = message_type.FromString(socket.recv())
        field = f"{side}_arm_joints"
        if not state.HasField(field):
            raise ValueError(f"Observation did not include the {side} arm.")
        arm = getattr(state, field)
        sample = [float(getattr(arm, joint)) for joint in JOINTS]
        if not all(math.isfinite(value) for value in sample):
            raise ValueError("Observation contained a non-finite joint value.")
        samples.append(sample)
    columns = list(zip(*samples))
    for joint, values in zip(JOINTS[:2], columns[:2]):
        if max(values) - min(values) > max_spread:
            raise ValueError(f"{joint} moved during capture. Hold the pose and try again.")
    return dict(zip(JOINTS, map(median, columns)))


def capture(args, session, targets):
    import zmq
    from google.protobuf.message import DecodeError
    from lerobot_robot_sourccey.robots.protobuf.generated.sourccey_pb2 import SourcceyRobotState

    print("Position the arm with your usual controls. Keep the base and elevator fixed.")
    print("Aim the upper arm (shoulder to elbow). Directions are from the robot's perspective.")
    print("Skip poses that lack clearance or are outside the real arm's range.")
    print("During capture, the host splits observation packets between this script and teleop.")
    poses = session["poses"]
    print(f"Resuming with {len(poses)} saved poses.")
    with zmq.Context() as context:
        for side in SIDES:
            for name, (_, instruction) in DIRECTIONS.items():
                key = f"{side}_{name}"
                if key in poses:
                    continue
                target, error = targets[key]
                tolerance = 10 if name in ("up", "down") else 5
                if error > tolerance:
                    print(f"{key}: model misses this direction by {error:.1f} degrees. Skipping.")
                    continue
                if name in ("up", "down"):
                    print(f"The model's closest pose is {error:.1f} degrees off vertical.")
                while True:
                    answer = input(
                        f"\n{len(poses)} saved. Point the {side} upper arm {instruction}.\n"
                        "Enter to save, s to skip, q to finish: "
                    ).strip().lower()
                    if answer == "q":
                        return
                    if answer == "s":
                        break
                    if answer:
                        continue
                    try:
                        with context.socket(zmq.PULL) as socket:
                            socket.setsockopt(zmq.LINGER, 0)
                            socket.setsockopt(zmq.CONFLATE, 1)
                            socket.connect(f"tcp://{args.robot_ip}:{args.observation_port}")
                            readings = read_pose(socket, SourcceyRobotState, side, args.max_spread)
                    except (ValueError, DecodeError) as exc:
                        print(f"Pose not saved. {exc}")
                        continue
                    poses[key] = {
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                        "hardware_positions": readings,
                        "mujoco_shoulder_rad": target,
                        "reference_error_deg": error,
                    }
                    session["shoulder_fits"] = fit_shoulders(poses)
                    save(args.checkpoint, session)
                    print(f"Saved {key} to {args.checkpoint}")
                    break


def fit_shoulders(poses):
    import numpy as np

    fits = {}
    for side in SIDES:
        for joint in JOINTS[:2]:
            pairs = [
                (pose["hardware_positions"][joint], pose["mujoco_shoulder_rad"][joint])
                for key, pose in poses.items()
                if key.startswith(side + "_")
                and (joint != "shoulder_pan" or key not in (f"{side}_up", f"{side}_down"))
            ]
            if len(pairs) < 2:
                continue
            hardware, radians = np.asarray(pairs).T
            if np.ptp(hardware) < 1e-6 or np.ptp(radians) < math.radians(5):
                continue
            design = np.column_stack((hardware, np.ones_like(hardware)))
            slope, offset = np.linalg.lstsq(design, radians, rcond=None)[0]
            error = np.max(np.abs(slope * hardware + offset - radians))
            fits[f"{side}_{joint}"] = {
                "rad_per_hardware_unit": float(slope),
                "offset_rad": float(offset),
                "hardware_range": [float(hardware.min()), float(hardware.max())],
                "samples": len(pairs),
                "max_fit_error_deg": math.degrees(error),
            }
    return fits


def preview(sim, poses, fits):
    import mujoco
    import mujoco.viewer
    import numpy as np

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        for key, pose in poses.items():
            if not viewer.is_running():
                return
            side = key.split("_", 1)[0]
            with viewer.lock():
                sim.reset()
                for joint, reference in pose["mujoco_shoulder_rad"].items():
                    name = f"{side}_{joint}"
                    fit = fits.get(name)
                    target = reference
                    if fit is not None:
                        target = fit["rad_per_hardware_unit"] * pose["hardware_positions"][joint]
                        target += fit["offset_rad"]
                    joint_id = sim.joints[name]
                    sim.data.qpos[sim.model.jnt_qposadr[joint_id]] = np.clip(
                        target, *sim.model.jnt_range[joint_id]
                    )
                mujoco.mj_forward(sim.model, sim.data)
            viewer.sync()
            if input(f"Showing {key}. Enter for next pose, q to close: ").strip().lower() == "q":
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", help="Sourccey host IP for capture")
    parser.add_argument("--observation-port", type=int, default=5556)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/arm_directions.json"))
    parser.add_argument("--preview", action="store_true", help="Replay saved poses in MuJoCo")
    parser.add_argument("--max-spread", type=float, default=2.0,
                        help="Maximum shoulder movement during capture, in host readback units")
    args = parser.parse_args()
    if not args.robot_ip and not args.preview:
        parser.error("provide --robot-ip to capture or --preview to replay")
    if not math.isfinite(args.max_spread) or args.max_spread <= 0:
        parser.error("--max-spread must be positive and finite")

    source = {"host": args.robot_ip, "port": args.observation_port}
    if args.checkpoint.exists():
        session = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if session.get("version") != 1:
            parser.error("unsupported checkpoint version")
        if args.robot_ip and session["source"] != source:
            parser.error("checkpoint belongs to a different host or port; use another --checkpoint")
    elif args.robot_ip:
        session = {"version": 1, "source": source, "poses": {}}
    else:
        parser.error(f"no checkpoint found at {args.checkpoint}")

    from sourccey.simulation import Simulation

    sim = Simulation()
    if args.robot_ip:
        try:
            capture(args, session, references(sim))
        except (KeyboardInterrupt, EOFError):
            print("\nCapture stopped. Previously saved poses are on disk.")
            return
        except ImportError as exc:
            parser.exit(1, f"Capture requires pyzmq and the Sourccey LeRobot plugin: {exc}\n")

    poses = session["poses"]
    if not poses:
        print("No poses captured.")
        return
    fits = fit_shoulders(poses)
    for joint, fit in fits.items():
        print(f"{joint}: {fit['samples']} poses, max fit error {fit['max_fit_error_deg']:.1f} degrees")
    if len(fits) < 4:
        print("Some shoulders need more distinct poses. Preview uses model references for unfitted joints.")
    if args.preview:
        preview(sim, poses, fits)


if __name__ == "__main__":
    main()
