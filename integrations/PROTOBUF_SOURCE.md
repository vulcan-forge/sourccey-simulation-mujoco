`sourccey.proto` and generated `sourccey_pb2.py` are copied unchanged from the
local Vulcan Sourccey plugin's `robots/protobuf/` directory. The generator version
is 6.31.0. These define the same wire messages already used by robot_start_pose;
they do not contain motor drivers, calibration, or host safety policy.

Source: `VulcanLerobot/lerobot-robot-sourccey/src/lerobot_robot_sourccey/robots/protobuf/`.
The source plugin uses Apache-2.0; see `SOURCCEY_PLUGIN_LICENSE` in this directory.
Regenerate/update this pair together if the host protocol changes.
