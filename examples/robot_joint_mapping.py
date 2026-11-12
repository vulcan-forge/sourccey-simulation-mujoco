"""Record a measured joint-angle reference using read-only robot feedback.

This never positions the arm. The angle must be independently known in the
MuJoCo joint convention, not inferred from normalized motor readbacks.
"""
import argparse
import json
import math
from pathlib import Path

from examples.calibrate_arm_directions import save
from examples.robot_start_pose import HardwareLink, KEYS, capture, finite


def add_reference(mapping, host, joint, angle_deg, note, reading):
    if joint + ".pos" not in KEYS or not note.strip():
        raise ValueError("Choose an arm joint and describe the independent angle measurement")
    if mapping.get("version") != 1 or mapping.get("host") != host or mapping.get("units") != "normalized":
        raise ValueError("Joint reference file belongs to different host/units/version")
    sample = {"mujoco_rad": math.radians(finite(angle_deg)),
              "hardware_normalized": reading["positions"][joint + ".pos"],
              "captured_at": reading["captured_at"], "measurement_method": note,
              "elevator": reading["elevator"]}
    entry = mapping.setdefault("joints", {}).setdefault(joint, {"samples": []})
    entry["measurement_method"] = "Independently measured angles; methods recorded per sample"
    # Preserve history; inconsistent repeat measurements should fail the fit,
    # rather than silently replacing evidence with the most recent reading.
    entry["samples"].append(sample)
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.1.79")
    parser.add_argument("--joint", required=True, choices=[k[:-4] for k in KEYS])
    parser.add_argument("--angle-deg", type=float, required=True,
                        help="Independently measured angle in the MuJoCo joint convention")
    parser.add_argument("--measurement-method", required=True)
    parser.add_argument("--normalized", action="store_true", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/robot_joint_mapping.json"))
    args = parser.parse_args()
    link = None
    try:
        finite(args.angle_deg)
        mapping = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {
            "version": 1, "host": args.robot_ip, "units": "normalized", "joints": {}}
        link = HardwareLink(args.robot_ip)
        reading = capture(link, args.robot_ip)
        add_reference(mapping, args.robot_ip, args.joint, args.angle_deg, args.measurement_method, reading)
        save(args.output, mapping)
        print(f"Saved measured reference for {args.joint} to {args.output}; no commands sent")
    except (ValueError, KeyError, OSError, TimeoutError) as exc:
        parser.error(str(exc))
    finally:
        if link is not None:
            link.close()


if __name__ == "__main__":
    main()
