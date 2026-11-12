"""Parked startup: shoulders facing forward, both arms hanging down.

The CAD shoulder axes leave the upper arms about nine degrees off vertical;
the straight forearms hang nearly vertically with elbow and wrist at zero.
This is a simulation pose, separate from hardware calibration. SI units.
"""

DEFAULT_POSE = {
    "linear_actuator": -0.0142,
    "left_shoulder_pan": -0.6299546458333334,
    "right_shoulder_pan": -0.6299546458333334,
    "left_shoulder_lift": -1.5707963265,
    # The CAD right shoulder lift has the opposite sign from the left.
    "right_shoulder_lift": 1.5707963265,
    "left_elbow_flex": 0.0,
    "right_elbow_flex": 0.0,
    "left_wrist_flex": 0.0,
    "right_wrist_flex": 0.0,
    "left_wrist_roll": 0.0,
    "right_wrist_roll": 0.0,
    "left_gripper": -0.087266463,
    "right_gripper": -0.087266463,
    "front_left_wheel": 0.0,
    "front_right_wheel": 0.0,
    "rear_left_wheel": 0.0,
    "rear_right_wheel": 0.0,
}

# Older table and two-hand demonstrations stage their objects around this
# raised pose. Keep their scene setup independent of the robot's parked home.
RAISED_DEMO_POSE = {
    **DEFAULT_POSE,
    "left_shoulder_pan": -.785, "right_shoulder_pan": -.801,
    "left_shoulder_lift": 2.01, "right_shoulder_lift": -2.01,
    "left_elbow_flex": -2., "right_elbow_flex": -2.,
    "left_wrist_flex": .691, "right_wrist_flex": -.723,
}


def set_startup_pose(model, positions):
    """Override only a demo model's startup keyframe, leaving the source alone."""
    key = model.key("startup").id
    for name, angle in positions.items():
        model.key_qpos[key, model.joint(name).qposadr[0]] = angle
        model.key_ctrl[key, model.actuator(name).id] = angle
