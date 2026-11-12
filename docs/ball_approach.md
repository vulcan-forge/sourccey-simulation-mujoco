# Tabletop ball concept

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m examples.ball_approach --scene examples/ball_approach.example.json --gif artifacts/ball_approach.gif --trajectory artifacts/ball_approach_trajectory.json
```

Add `--view --speed 0.5` to watch half-speed arm and gripper motion. Speed
defaults to 1 and must be greater than 0 and at most 1. Slower speeds increase
the number of smooth command samples at the same physics timestep; they do
not merely slow the viewer clock. Movement and closure durations scale by
1/speed; settling pauses and the 1.2-second hold stay unchanged. Exported
trajectory timestamps include this scaling. Hardware replay is implemented in
`examples.robot_ball_replay`; see `ROBOT_REPLAY.md` and `GIT_BASH_COMMANDS.mdx`.
Slower simulation motion alone does not validate hardware safety.

The illustrative scene uses a 40 mm ball, a tabletop 580 mm above the datum,
and a ball center at [500, 180, 600] mm. The datum is initially the model's
base_link origin, 119.125 mm above the world floor. Axes are X forward,
Y robot left, Z up, matching Simulation.robot_rotation (not raw CAD axes).
datum_offset_mm locates a replacement datum relative to base_link in these
axes. All scene positions and real measurements use that datum. Table height
is relative to the datum, not the floor. The table is centered below the ball.

The first stage approaches 80 mm above the ball with the gripper open. The
robot begins with both arm segments hanging down, pauses briefly, raises the
active arm outward beside the table, then bends and swings forward above it.
This staging avoids folding the hand through the chassis from the parked pose.
The arm rises above the table before swinging forward. Pickup IK also penalizes
pad clearance below 5 mm, so it can adjust the front pinch orientation to
clear the tabletop. The default scene maintains about 4.8 mm of pad clearance
through the dynamic run. This is a staged path for the example scene, not a
general collision-free planner for arbitrary table dimensions.
The inactive arm stays down. The
reported grasp point is the average of two configurable landmarks attached
to the fixed and moving fingers. Landmark coordinates are in their respective
CAD body frames. These defaults come from the simplified pads in dual_grasp;
verify corresponding physical landmarks before treating this as metrology.
The original wrist-origin ee sites are unchanged.

The optional pickup stage reorients above the ball, descends, closes, lifts,
and holds for 1.2 simulated seconds. Its IK aligns the fixed pad normal with
the configured `pinch_normal_world` (default world +X for a front pinch).
The fixed finger's contact reference is x=100 mm, nearer the fingertip than
the original x=80 mm reference. This pickup keeps wrist roll within about one
degree of neutral and avoids the old hardware travel violation. Optional
`joint_motion_limits_rad` narrows the active model joints' allowed travel;
it cannot expand the model's bounds or exclude the startup pose.
It uses five arm joints and a contact target one ball radius from the fixed
pad surface. This differs from the open-finger midpoint used for the first
measurement. Set pickup=false to stop after the open approach. The ball is
always a free body; only the base and elevator use fixture constraints.
contact_pads replaces the four hand collision hulls with the existing demo's
simple pads and friction values. Contact success is a concept result, not
validated real grip performance. The ball can roll during closing; the report
records its displacement and final pad contacts. Changing ball size or
location can cause pickup failure; the CLI returns failure if either enabled
stage fails. Base/elevator maximum drift fields cover the open approach.
With contact pads enabled, the table monitor inspects every physics step across
the approach, close, lift, and hold. Any robot-table contact fails the relevant
stage, while ball-table support contact is allowed. The report includes contact
step count, contacting robot geometries, maximum penetration, and minimum pad
clearance; a successful lift alone is insufficient to pass.

The JSON report separates approach success, contact pickup success, and real
comparison completion. The GIF plays the physics at its nominal rate. The
trajectory records simulated commands and angles every 120 ms for inspection;
these values are not calibrated hardware commands.

For a real comparison, manually approach through the usual teleoperation with
the operator present, keeping base and elevator fixed. Capture all six actual
side-prefixed joint readbacks and measure the same finger landmarks in the
base datum frame. Use shoulder fits only within the captured range. Elbow and
wrist signed scales, wrist neutral, and gripper endpoints still require known
angular/open/closed references; do not extrapolate those from the shoulder fit
or substitute the parked pose for the straight-arm reference. This script
does not connect to or command the robot.

Pass --measurement path/to/measurement.json once those data are available.
The measurement JSON must contain side, datum_description, ball_center_mm,
measured_at, measurement_method, joint_mapping_notes, grasp_point_mm,
hardware_positions (all six keys such as left_shoulder_pan.pos), and
mujoco_joint_rad (all six unprefixed role keys). Normalized hardware readbacks
are logged as supplied; the script does not guess their conversion. The
externally established mapped angles drive FK for the comparison, rather
than using the approach's commanded pose. The report includes the real-minus-
sim XYZ vector and its Euclidean norm in mm. mapping_verified=true is an
operator assertion that those mappings, the open pose, and common datum have
been verified, not an automatic calibration check. Set illustrative=false
only after substituting measured physical scene dimensions. Without both,
benchmark_complete remains false.
