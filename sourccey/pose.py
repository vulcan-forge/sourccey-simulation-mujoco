"""Startup pose matched visually to the user's MuJoCo screenshot.

The screenshot's Control panel displayed rounded actuator targets, not settled
joint positions. The right CAD lift axis needs a negative angle for the shown
raised-arm geometry. SI units throughout; wheels are parked at startup.
"""

DEFAULT_POSE = {
    "linear_actuator": -0.0142,
    "left_shoulder_pan": -0.785,
    "right_shoulder_pan": -0.801,
    "left_shoulder_lift": 2.01,
    # The CAD right shoulder lift has the opposite sign from the left.
    "right_shoulder_lift": -2.01,
    "left_elbow_flex": -2.0,
    "right_elbow_flex": -2.0,
    "left_wrist_flex": 0.691,
    "right_wrist_flex": -0.723,
    "left_wrist_roll": 0.0,
    "right_wrist_roll": 0.0,
    "left_gripper": -0.087266463,
    "right_gripper": -0.087266463,
    "front_left_wheel": 0.0,
    "front_right_wheel": 0.0,
    "rear_left_wheel": 0.0,
    "rear_right_wheel": 0.0,
}
