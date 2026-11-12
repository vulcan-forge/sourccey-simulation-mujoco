# MuJoCo to robot replay

`examples.robot_ball_replay` runs the simulation offline first. Without
`--execute` it never constructs a robot connection. The operator confirmed that
the existing VR teleoperation correctly tracks real left elbow/wrist direction
and approximate angle. Its conversion data, together with the previously captured
shoulder references and recorded gripper convention, now supplies the left-arm
mapping in `artifacts/robot_joint_mapping.json`. Entries are labeled
`source_reference_confirmed` with specific evidence, not `hardware_verified`.
The operator confirmed successful end-to-end motion-only replay on 2026-11-12.
The saved client trace ends in `complete`: 43.188 seconds of adaptive replay,
with maximum target lead below 5 units and maximum command change below 1 unit.
This confirms the tested arm path, not a physical ball grasp.

The working mapping, verified start, host limits, motor calibration snapshot,
and host correction sources are saved as selected versioned files under
`artifacts/`. They describe this particular robot at `192.168.1.79`; they are
not generic calibration for other robots. Logs and generated outputs continue
to use the ignored artifacts directory. A compact successful-run record is
saved in `artifacts/robot_diagnostics/robot_motion_replay_success.json`.

Offline preflight now passes with `--motion-only`. That explicit mode permits
illustrative scene dimensions for testing the motion in empty, clear space. It
does not certify a physical grasp or waive mapping, travel, feedback, or simulation
checks. Physical pickup mode still requires actual matching scene measurements.
The right-arm mappings remain pending; the replay holds its hardware start.

The old sideways pickup asked for about -1.634 rad wrist roll, requiring about
+188 normalized units and exceeding +100. Merely restricting its wrist failed
to lift the ball. The default scene now instead pinches from the front, with
the fixed-finger contact reference at x=100 mm rather than x=80 mm. It solves
the pickup with wrist roll within about 0.85 degrees of neutral. At half speed,
the revised simulation lifted the ball 145 mm with both pads holding, zero
robot/table contacts, and 4.77 mm minimum pad clearance. These remain simplified
contact pads and do not certify the real fingers' geometry.

The unconfirmed source import is kept separately as
`artifacts/robot_joint_mapping.unity_candidate.json`. Regenerating it does not
overwrite the mapping containing the operator/source evidence.

To regenerate the candidate from the local Unity source and robot snapshot:

```bash
cd /c/Users/Theor/Documents/WebsiteCode/VulcanSimulation/SourcceyMuJoCo
source .venv/Scripts/activate
python -m integrations.import_unity_mapping
```

The importer records source hashes, combines gear ratios, left/right sign maps,
scene overrides, full-CAD delta conventions, motor direction flags and spans,
and the new parked anchors. It deliberately imports differential conversions
rather than old Unity offsets. Source `Assets/Resources/arm_calibration.json`
uses wrist roll scales 2.035 (left) and 2.076 (right). The older external
`Documents/SourcceyCalibration/CALIBRATION-RECORD.md` records approximately 1.628
and 1.661 instead, so the deployed wrist response needs checking. The older
record also explicitly says shoulder lift was not calibrated at that time.

The actual robot calibration was inspected again over SSH and saved as
`artifacts/robot_diagnostics/active_motor_calibration.json`, with SHA256 hashes
of both source files. Preflight imports and reports its motor fields. The robot
files contain `id`, `drive_mode`, `homing_offset`, `range_min`, and `range_max`.
The host normalizes non-gripper travel to -100..100 and grippers to 0..100;
it applies the drive-mode inversion for Feetech motors in software. Neither
these files nor the inspected follower/normalization code defines the CAD
joint axis correspondence or mechanical gear ratios. For example, the actual
left elbow range is 517..3658: 15.705 encoder ticks per normalized unit. That
known motor conversion is retained; the Unity gear/sign maps supply its CAD
relationship without repeating motor endpoint calibration.

With valid references and physical scene measurements, `--execute --normalized`
opens MuJoCo, holds the freshest measured robot pose, slowly returns both arms to
`artifacts/robot_start_pose_verified.json`, slowly opens the active gripper, and
replays the simulated active-arm angles. Other arm targets remain at the saved
start; base velocity is zero and the elevator target is preserved. The MuJoCo
window displays converted **measured arm angles**. Its ball is a scene marker;
this bridge does not observe the real ball or verify a real grasp.

Replay startup does not take a new calibration capture. The former ten-reading
capture could reject small settling motion before sending any command. Startup
now checks base/elevator state, return distance, and host limits, sends the live
pose as its first target, and streams that fixed hold until feedback stays
within the configured tracking tolerance for one second. It allows up to the configured stall timeout to settle
before aborting. The return ramp and pickup retain their feedback checks. The
console reports hold, return, and replay separately. The initial JSON's
`hardware_commands_sent: false` describes the offline preflight only.

Default speed is 0.5; `--speed 2` plays at double normal speed. The bridge records the normal-speed simulation, builds a
shape-preserving cubic curve, and divides its timestamps by `speed`. Half speed
therefore doubles the pickup duration, halves velocity and quarters acceleration.
These timestamps specify the requested maximum pace. Live feedback slows path
progress whenever the real joints need more time. The normal
recording is 20.16 seconds, so half speed is 40.32 seconds and double speed is
10.08 seconds. Parked gripper preparation adds 2.25 seconds at normal speed,
4.5 seconds at half speed, or 1.125 seconds at double speed. Return to start is
a separate ramp whose time depends on the actual pose and scales with `speed`;
double speed halves its planned duration and doubles its velocity limit.
Offline simulation computes faster than real
time, and its wall time is reported separately from the recorded motion.
Host safety remains enabled. `--tracking-tolerance` defaults to 5 normalized
units for all arm and gripper joints. It applies to the initial hold, return
ramp, replay, and final settling. Replay checks each proposed target against
current feedback before sending it, keeping the target within the tracking
window and limiting each command change to 1 normalized unit. It advances a
single time coordinate along the recorded joint path; it does not clip joints
independently or skip large intervening excursions. At sharp or fast sections
the path clock slows or waits, so the arm can follow the general motion without
matching simulation timing. The reported duration is nominal; actual wall time
may be much longer. Progress is printed every two seconds.

`--stall-timeout` defaults to 15 seconds. It stops replay only when a limited
path makes no meaningful progress for that interval; continuous slow progress
resets the timer, and intentional stationary dwell does not count as a stall.
The timeout also applies to initial settling and return-to-start waiting.
Any joint error over 10 units still aborts. The configurable tracking tolerance must
be positive and below 10. The separate `robot_start_pose` command retains its
2-unit default and maximum CLI speed of 1. Pausing can stop the target immediately.
Closing the window or aborting holds the last target; it does not release torque
or act as an emergency stop. There is no general obstacle planner for the
return to start. Keep that return path clear.

The first live replay stopped at trajectory time 2.219 seconds, just before the
parked gripper opening ended. Its open command was 0.071 and its stationary
readback was 2.186, a residual of 2.116 closure units (about 33 encoder ticks).
Every other joint was within 0.17 units of its target. A subsequent replay
reached trajectory time 5.631 seconds, then stopped with the left shoulder
lift at -40.806 against a -38.669 command, a 2.137-unit residual. The default
5-unit window accommodates both observed residuals. Mapping and commanded
endpoints are unchanged.

A retry after opening can start almost 100 closure units from the saved gripper
position, and an interrupted arm motion can leave the arm far from its start.
Motion-only mode permits a full-range ramped return within normalized hardware
and host limits, so that partial replays can be retried. Physical pickup mode
retains the 30-unit return-distance limit for arm joints. Gripper travel still
contributes to the return duration under the velocity and acceleration limits.

To replay twice as fast with the wider tracking window:

```bash
python -m examples.robot_ball_replay --robot-ip 192.168.1.79 --normalized --speed 2 --tracking-tolerance 5 --motion-only --execute
```

Every executed attempt writes the full feedback/command trace to
`artifacts/robot_ball_replay.jsonl`. The next attempt replaces that file.
Offline results go to `artifacts/robot_ball_preflight.json`.

## Supply measured references

Mapping file: `artifacts/robot_joint_mapping.json`. Its shape is:

```json
{
  "version": 1,
  "host": "192.168.1.79",
  "units": "normalized",
  "joints": {
    "left_elbow_flex": {
      "measurement_method": "independently measured angles relative to the model zero",
      "samples": [
        {"hardware_normalized": 0, "mujoco_rad": 0},
        {"hardware_normalized": 50, "mujoco_rad": -1.57}
      ]
    }
  }
}
```

The numbers above show the file format only; **they are not robot calibration**.
Do not copy them into an executing configuration. All six active-arm joints
need independent angle references spanning the entire requested motion, with
signed direction and scale. A single down pose gives an anchor only. The
preflight output lists each joint's requested angular span. A third reference
is useful to verify linearity, especially for the simplified gripper model.

The read-only recorder captures feedback at an independently known angle:

```bash
cd /c/Users/Theor/Documents/WebsiteCode/VulcanSimulation/SourcceyMuJoCo
source .venv/Scripts/activate
# Example syntax only: substitute the actual measured angle and method.
python -m examples.robot_joint_mapping --robot-ip 192.168.1.79 --normalized --joint left_elbow_flex --angle-deg 0 --measurement-method "Measured against the verified parked reference"
```

This does not move the robot or measure an angle by itself. The host must be
running with both arms connected and no other command client. Repeated records
are retained. References must use the same hardware calibration, normalized
units, robot datum, and joint zero convention as the saved start.

Shoulder poses from `artifacts/arm_directions.json` are useful evidence, but their
old fitted offsets must not silently replace the newer down alignment. Replay
checks every mapped down angle against the simulation within 0.03 rad, rejects
inconsistent linear fits, and disallows extrapolation beyond measured travel.
Do not bypass these errors by widening ranges without measurements.

Copy the example scene to a physical scene file, supply actual table/ball
dimensions and datum, and set `illustrative` to false only when they match the
physical setup. Both arms start down, the elevator must match the saved height,
and simplified simulated pads do not certify real finger clearance.

The shoulder staging angle in the example is now 1.56 rad (89.4°), reduced from
2 rad (114.6°). The reduced path passed the half-speed contact pickup with zero
robot/table contacts. This alone does not certify physical motion.
