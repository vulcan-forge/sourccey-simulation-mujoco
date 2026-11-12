"""Import a non-executable conversion candidate from the existing Unity sources.

Uses robot-side motor spans and the verified parked anchor. Does not send
commands, change calibration, or certify the ported CAD direction conventions.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from examples.robot_start_pose import ROLES


def dictionary(source, method):
    start = source.index("private static Dictionary<string, float> " + method + "()")
    end = source.find("private static Dictionary", start + 1)
    section = source[start:end if end >= 0 else None]
    return {name: float(value) for name, value in re.findall(
        r'\{\s*"(\w+)"\s*,\s*([-\d.]+)f\s*\}', section)}


def import_candidate(unity, calibration, alignment):
    bridge_path = unity / "Assets/Scripts/UnityMotorBridge/UnityIKMotorBridge.cs"
    scene_path = unity / "Assets/Scenes/SourcceyFullControlVR.unity"
    arm_path = unity / "Assets/Resources/arm_calibration.json"
    visual_path = unity / "Assets/Scripts/Controllers/FullRobotVisualDriver.cs"
    processor_path = unity / "Assets/Scripts/UnityMotorBridge/UnityArmTeleopProcessor.cs"
    bridge, scene = bridge_path.read_text(), scene_path.read_text()
    arms = json.loads(arm_path.read_text())
    gears = dictionary(bridge, "DefaultJointScaleMap")
    if set(gears) != set(ROLES[:-1]):
        raise ValueError("Unity transmission map changed; inspect before importing")
    blocks = [b for b in re.split(r"^---", scene, flags=re.M) if "reverseShoulderPan:" in b]
    if len(blocks) != 2:
        raise ValueError("Expected exactly two arm processors in the FullControl scene")
    result = {"version": 1, "host": calibration["host"], "units": "normalized", "joints": {},
              "status": "candidate_only_not_hardware_verified",
              "anchor_policy": "Use the verified down anchor; import Unity differential signs/scales only, not old Unity offsets",
              "limitations": ["Unity-to-CAD axis correspondence must be checked on hardware",
                              "Bundled wrist-roll correction conflicts with older measurement notes",
                              "Do not mark verified merely to bypass preflight"],
              "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (bridge_path, scene_path, arm_path, visual_path, processor_path)},
              "motor_calibration_source_sha256": calibration["source_files"]}
    # Differential CAD mapping from FullRobotVisualDriver.CopyArm. The offsets
    # (-45 degrees etc.) deliberately do not replace the new parked alignment.
    cad_signs = {"left": [1, -1, 1, 1, 1], "right": [-1, 1, 1, -1, 1]}
    override_names = ["ShoulderPan", "ShoulderLift", "ElbowFlex", "WristFlex", "WristRoll"]
    for side in ("left", "right"):
        index = 0 if side == "left" else 1
        block = next(b for b in blocks if re.search(rf"^  armSide: {index}$", b, re.M))
        signs = dictionary(bridge, "Default" + side.title() + "ArmSignMap")
        for i, role in enumerate(ROLES[:-1]):
            name = side + "_" + role
            reverse = int(re.search(rf"^  reverse{override_names[i]}: ([01])$", block, re.M).group(1))
            motor = calibration["arms"][side][role]
            span = motor["range_max"] - motor["range_min"]
            scale = arms[side]["wrist_roll_scale"] if role == "wrist_roll" else 1
            # Unity bridge converts degrees using resolution - 1 (4095), then
            # applies the drive-mode sign. Keep that exact convention here.
            per_radian = (200 / span * 4095 / (2*math.pi) * gears[role] * scale
                          * signs[role] * (-1 if reverse else 1) * cad_signs[side][i]
                          * (-1 if motor["drive_mode"] else 1))
            anchor = alignment["joint_anchors"][name]
            slope = 1 / per_radian
            result["joints"][name] = {
                "verification_status": "pending",
                "method": "Unity gear/sign maps + FullControl scene overrides + CAD delta convention + new parked anchor",
                "motor_per_joint_scale": gears[role] * scale,
                "derived_affine": {"slope": slope,
                    "offset": anchor["mujoco_rad"] - slope * anchor["hardware_normalized"],
                    "hardware_range": [-100, 100]}}
        # Legacy controller defines left 100/right 0 as closed. Port the
        # simulator's own -5..60 degree convention, pending endpoint check.
        slope = math.radians(-65 if side == "left" else 65) / 100
        result["joints"][side + "_gripper"] = {
            "verification_status": "pending",
            "method": "Legacy left/right closure convention and MuJoCo -5..60 degree endpoints; physical endpoint check pending",
            "derived_affine": {"slope": slope,
                "offset": math.radians(60 if side == "left" else -5), "hardware_range": [0, 100]}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unity", type=Path, default=Path("../../Unity/Sourccey-VR-Controller"))
    parser.add_argument("--motor-calibration", type=Path, default=Path("artifacts/robot_diagnostics/active_motor_calibration.json"))
    parser.add_argument("--alignment", type=Path, default=Path("artifacts/robot_start_alignment.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/robot_joint_mapping.unity_candidate.json"))
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    result = import_candidate(args.unity, load(args.motor_calibration), load(args.alignment))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(result['joints'])} joint conversion candidates to {args.output}; not hardware verified")


if __name__ == "__main__":
    main()
