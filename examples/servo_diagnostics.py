"""Run on the ROBOT in its existing environment, replacing the normal host command.

Uses the normal host's bus connections. Diagnostic hooks only read registers;
normal host startup/shutdown and commands from other clients are unchanged.
This file can be copied and run standalone; it imports no MuJoCo workspace code.
"""
import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import time


REGISTERS = ("Goal_Position", "Present_Position", "Torque_Enable", "Operating_Mode",
             "Min_Position_Limit", "Max_Position_Limit", "Homing_Offset",
             "Present_Current", "Present_Load", "Present_Voltage", "Present_Temperature")


def read_motor(bus, name):
    motor = bus.motors[name]
    calibration = bus.calibration.get(name)
    result = {"id": motor.id, "model": motor.model,
              "norm_mode": str(motor.norm_mode),
              "calibration": asdict(calibration) if is_dataclass(calibration) else calibration,
              "registers": {}, "read_errors": {}}
    for register in REGISTERS:
        try:
            result["registers"][register] = bus.read(register, name, normalize=False)
        except Exception as exc:
            # Older servo firmware / libraries may not expose every register.
            result["read_errors"][register] = str(exc)
    for register in ("Goal_Position", "Present_Position"):
        raw = result["registers"].get(register)
        if raw is not None:
            try:
                result[register + "_normalized"] = bus._normalize({motor.id: raw})[motor.id]
            except Exception as exc:
                result["read_errors"][register + "_normalized"] = str(exc)
    goal = result.get("Goal_Position_normalized")
    present = result.get("Present_Position_normalized")
    if goal is not None and present is not None:
        result["goal_minus_present_units"] = goal - present
    return result


def snapshot(robot):
    result = {"at": datetime.now(timezone.utc).isoformat(), "event": "servo_snapshot", "arms": {}}
    for side in ("left", "right"):
        arm = getattr(robot, side + "_arm")
        result["arms"][side] = {"calibration_file": str(getattr(arm, "calibration_fpath", "unknown")),
                                "use_degrees": arm.config.use_degrees,
                                "shoulder_lift": read_motor(arm.bus, "shoulder_lift")}
    return result


def install_hooks(robot_class, emit, interval):
    original_observation = robot_class.get_observation
    original_action = robot_class.send_action
    next_sample = [0.]

    def get_observation(robot, *args, **kwargs):
        observation = original_observation(robot, *args, **kwargs)
        now = time.monotonic()
        if now >= next_sample[0]:
            next_sample[0] = now + interval
            try:
                report = snapshot(robot)
                emit(report)
                for side, arm in report["arms"].items():
                    motor = arm["shoulder_lift"]
                    print(f"{side} lift: goal={motor.get('Goal_Position_normalized', '?')} "
                          f"present={motor.get('Present_Position_normalized', '?')} "
                          f"torque={motor['registers'].get('Torque_Enable', '?')} "
                          f"raw goal={motor['registers'].get('Goal_Position', '?')} "
                          f"raw present={motor['registers'].get('Present_Position', '?')}", flush=True)
            except Exception as exc:
                emit({"event": "diagnostic_error", "error": str(exc)})
        return observation

    def send_action(robot, action, *args, **kwargs):
        # Log existing commands; never generate, modify, or resend a command.
        emit({"event": "incoming_command", "at": datetime.now(timezone.utc).isoformat(),
              "action": dict(action)})
        return original_action(robot, action, *args, **kwargs)

    robot_class.get_observation = get_observation
    robot_class.send_action = send_action

    def restore():
        robot_class.get_observation = original_observation
        robot_class.send_action = original_action
    return restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/sourccey_servo_diagnostics.jsonl"))
    parser.add_argument("--interval", type=float, default=2.)
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error("interval must be between 1 and 60 seconds")
    host = importlib.import_module("lerobot_robot_sourccey.robots.sourccey.sourccey_host")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        def emit(value):
            stream.write(json.dumps(value, default=str, allow_nan=False) + "\n")
            stream.flush()
        restore = install_hooks(host.Sourccey, emit, args.interval)
        print(f"Diagnostics: {args.output}. No commands added; normal host lifecycle applies.", flush=True)
        try:
            host.main()
        finally:
            restore()


if __name__ == "__main__":
    main()
