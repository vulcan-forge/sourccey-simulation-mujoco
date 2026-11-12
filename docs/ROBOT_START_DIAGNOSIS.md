# Why the recorded down pose moved on command

SSH inspection on 2026-11-12 confirmed that the kiosk's installed Sourccey
plugin differs from the desktop checkout. Its follower safety code protects
the shoulder's single-turn encoder boundary. It can replace a requested
shoulder position with a permitted endpoint before writing Goal_Position.

The existing robot-side trace for the failed attempt records:

| Joint | Client requested | Host safety output | Raw goal |
| --- | ---: | ---: | ---: |
| Left shoulder lift | -99.092506 | -89.218041 | 3719 |
| Right shoulder lift | 98.025780 | 89.167556 | 376 |

Evidence: robot file
`/home/sourccey/.local/share/ai.vulcanrobotics.studio/logs/motor-traces/shoulder-startup-20261112-121436-pid8249.jsonl`,
events `shoulder_command_prepare`, sequence 7, at approximately 12:20:49 EST.
The protected normalized intervals are left [-89.218041, 89.218041] and right
[-89.220918, 89.167556]. Both use raw limits 376..3719. These came from the
installed safety class and the robot's own calibration JSONs, not bundled
desktop calibration defaults.

The capture records where the arms physically were. It does not guarantee
that the host permits commanding those positions while energized. The observed
movement therefore has a confirmed software explanation; it is not evidence
that the MuJoCo shoulder fit is grossly wrong or that the motors cannot track.

The client now rejects a start or target outside the inspected shoulder limits
before creating any command socket. `--execute` requires the saved host-limit
snapshot (`artifacts/robot_host_limits.json`). This snapshot must be refreshed
if robot calibration, normalization mode, or the host guard changes. It is not
live negotiation with the host.

No host code, calibration, safety settings, torque settings, or running
services were changed during the SSH investigation, and no movement commands
were sent. The guard should remain enabled. The physical arms-down reference
is legitimate; the conflict is the guard demanding 376 ticks of clearance
after frame centering provides only about 174 ticks. Changing the physical
home solely to satisfy this software policy is not the intended correction.

`integrations/prepare_shoulder_guard_patch.py` prepares a separate candidate
module from the exact inspected source hash. It caps the desired guard at the
clearance already available outside calibrated travel, requires at least one
bounded command step of clearance on each side, and fails closed on invalid
or insufficiently separated encoder frames. This preserves the calibrated
endpoints instead of cropping them. Continuity tracking, ambiguity rejection,
startup recovery, and step limiting remain in place. This changes the original
four-step margin policy and requires operator-supervised hardware verification;
offline tests alone do not establish physical safety.

The candidate and its diff are under `artifacts/robot_diagnostics`. Following
operator approval, the correction was installed on disk on the robot. The old
file was preserved as `sourccey_follower_safety.py.backup-20261112-125659`
beside the installed module. The installed LF-source SHA256 is
`e0f59382f621eb8dfb3f403e0b7bd5c89b4b288342220751499ddf42a5f2b708`.

The operator restarted the host from the kiosk. PID 11357 loaded the corrected
module, but the right follower failed to finish connecting. After separate
explicit approval, this host was stopped. With the ports confirmed released,
a read-only diagnostic found all six right-arm servos responding, with torque
off and goals matching measured positions. No register writes were performed
by that diagnostic.

The kiosk continued to display a running host although its old process had
exited. A temporary SSH-launched host (PID 12072) connected both arms. Its
corrected shoulder intervals, derived from the unchanged calibration and
corrected guard, are now [-100, 100] normalized units. The client-limit snapshot
was refreshed, preserving the previous snapshot separately.

The approved `hold-current` test then sent identical start/target positions
for two seconds of target streaming plus one second of settled verification.
The client recorded 95 fresh feedback samples and zero measured change across
all twelve arm readbacks. The host recorded 93 commands per shoulder, zero
guard rewrites, and torque enabled during the hold. Raw target quantization was
one tick below the measured shoulder position; neither shoulder moved at the
reported resolution. This was a supervised, supported stationary test, not a
validation of an arbitrary moving path or grasp.

Evidence is saved in `artifacts/robot_hold_current_after_guard_fix.jsonl` and
`artifacts/robot_hold_current_summary.json`. The verified hardware reference is
`artifacts/robot_start_pose_verified.json`; the single-pose MuJoCo alignment was
updated to point to it. The temporary host was stopped afterward. The guard
correction remains installed and backed up; no automatic full-arm grasp replay
was attempted. Full grasp replay still requires the missing elbow/wrist scales
and verified gripper endpoints.

The replay bridge is now implemented in `examples/robot_ball_replay.py` and
documented in `docs/ROBOT_REPLAY.md`. Its offline preflight runs the full pickup
and rejects incomplete mappings before constructing a hardware connection.
The example's shoulder staging angle was reduced from 2.0 to 1.56 rad and
passed the half-speed pickup without robot/table contact. The physical grasp
has not been run. A fresh read-only SSH inspection confirmed that the robot's
motor calibration files contain encoder ranges, direction flags, and offsets,
but no CAD joint direction/gear-ratio mapping. Their exact contents and file
hashes are retained in `artifacts/robot_diagnostics/active_motor_calibration.json`.

## Endpoint recovery deadlock

A later double-speed motion-only attempt began with the left shoulder lift
already reporting -100. It stayed there throughout return, gripper preparation,
and shoulder pan. At replay time 2.532 seconds the lift target reached -93.011,
but feedback remained -100 for the stall interval. This is a stationary endpoint
failure, beyond the earlier small tracking residuals.

The exact saved installed host revision (`e0f59382...`) contains a reproducible
deadlock in `prepare_shoulder_command`: a raw reading outside 174..3921 starts
recovery toward the nearest endpoint, and recovery suppresses ordinary targets
until feedback is strictly back inside that interval. For example, raw 3923
forces the -100 endpoint goal (raw 3921) indefinitely if servo deadband leaves
the two-tick error unresolved. Normalized feedback can clip this to -100.
Password-authenticated SSH inspection confirmed this exact failure in the live
host trace `shoulder-startup-20261112-155352-pid5497.jsonl`: the left shoulder
settled at raw 3923 while Goal_Position remained 3921, with Torque_Enable=1.
The live follower overwrites the requested shoulder target with the recovery
goal before recording `incoming_normalized`, explaining why the host trace
shows -100 even while the client requests -93.011. The installed source hash
matched the saved `e0f59382...` revision exactly.

`integrations/prepare_shoulder_recovery_patch.py` prepares a separate candidate
from that exact host hash. It moves the recovery target one bounded step into
the safe interval on the same numeric side. The existing re-entry test then
releases normal commands when feedback enters the interval. It changes neither
calibration nor seam limits, and preserves continuity and downstream step safety.
Regression tests reproduce the old deadlock and exercise the correction at
both endpoints, both motor directions, both normalization modes, and encoder
edges. The pinned host source is a test fixture for clean-checkout regression.

Generate the candidate locally with:

```bash
python -m integrations.prepare_shoulder_recovery_patch
```

The standalone preparation command does not install the candidate or restart
the robot host. Following live source and trace verification, the correction
was installed on the robot on 2026-11-12 at 16:17 EST. The source SHA256 is
`cd19dd761b5efcbc0914cfa5d5ea9fb3c9590dad13110c6964c3516920079bb0`.
The previous module was retained beside it with suffix
`.backup-recovery-20261112-211719-329136`. No servo settings or motor commands
were written by installation. The running host must be stopped and started
from the kiosk to load the new module. Installation evidence is in
`artifacts/robot_diagnostics/shoulder_recovery_installation.json`; the source
and trace inspection is in `live_recovery_inspection.json` in the same folder.

The operator stopped and started Host from the kiosk. The new host, PID 2362,
started at 16:18:48 EST and completed connection with both arms available. The
installed module hash was rechecked against the correction, and the host-limit
snapshot provenance was refreshed without changing limits. Evidence is in
`artifacts/robot_diagnostics/shoulder_recovery_restart_verification.json`.
The host correction is active after restart; the full physical replay still
needs an operator-run attempt.

The next physical replay did lift the shoulder, then exposed a separate
desktop replay-controller error. At path time 2.688 seconds feedback was
-72.565 and the previous target was -67.967 (4.598 units of lag). The old loop
accepted that error, advanced one nominal frame, and sent -60.858: a 7.109-unit
target jump and an 11.707-unit lead over that feedback. The following read
triggered the 10-unit abort at observed -71.337. The endpoint-recovery correction
was not the cause of this new stop.

Replay now bounds each proposed command against current feedback before
sending it, advances along the same multi-joint curve by a smaller amount when
needed, and limits command changes to 1 unit. The path can take longer than its
nominal simulation timing. Its no-progress timeout is 15 seconds and resets
during continuous slow path progress. The exact failing transition is pinned
in `tests/fixtures/replay_shoulder_lag.json`. Regression checks also cover slow
motors, delayed response, stopped motors, reversals, and long stationary dwells.
An offline replay of the full actual pickup path with a 7-unit/s shoulder
completed in 59.27 emulated seconds, with proposed target lead at most 5 units
and command changes at most 1 unit. That software validation is recorded in
`artifacts/robot_diagnostics/adaptive_replay_offline_validation.json`; physical
verification of the adaptive follower remains pending.

The operator subsequently confirmed the adaptive replay worked. The saved
client trace ends in `complete`, with 1,036 replay feedback samples over
43.188 seconds, maximum proposed target lead below 5 units, and maximum command
change below 1 unit. This verifies the tested motion-only arm path; physical
grasp validation remains outside this test. The successful configuration and
compact run summary are versioned for restoration.
