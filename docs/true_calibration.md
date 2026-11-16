# True Calibration scene

```bash
python -m examples.true_calibration
```

This opens a separate MuJoCo viewer with calibration status. By default it
prepares or reuses a physics recording and plays it at **3x speed**, runs once,
and leaves the viewer open. The first recording of a detailed collision scene
can take several minutes; later replays reuse it. Press
**P** to pause/resume, **R** to run again, or **G** to toggle the inherited
gripper angle stop and restart. **J** toggles all arm angle limits and restarts.
**V** switches between the visible robot and the actual convex collision hulls;
**F** toggles contact points and force arrows. Closing the viewer ends playback.
G/J geometry editing is available in `--live` mode; recorded playback uses the
geometry and settings with which it was computed.

```bash
# Three-times-speed playback of the corrected collision experiment.
python -m examples.true_calibration --self-collision --free-arms --skip-lift --speed 3

# Interactive live physics, for changing calibration settings during a run.
python -m examples.true_calibration --self-collision --free-arms --skip-lift --live --speed 1
```

Playback speed changes only the display clock. Calibration sleeps, 2 ms physics
steps, contact handling, and resulting joint paths remain unchanged. Recordings
sample states at 50 Hz and interpolate poses for display. Cache keys include the
calibration configuration, controller/physics code, CAD sources, collision assets,
and MuJoCo version, so incompatible runs are not reused. Use `--speed 1` for
normal-speed recorded playback. Live mode is limited by physics computation and
may run slower than its requested speed.

The scene runs the **motion controllers copied from the installed robot**, rather
than a hand-authored approximation of their trajectories. The copies, upstream
hashes, and Apache license attribution are in
[`integrations/true_calibration_reference`](../integrations/true_calibration_reference).
Only sleeps become cooperative yields; the simulation clock advances the physics
while each controller waits. The reference checksums are verified on loading.

The sequence corresponds to **True Calibration / `full_reset=True`**:

1. Seek the lift bottom until its raw sensor has remained stable for three seconds.
   Back away for one second and confirm the endpoint. Repeat at the top, verify
   repeatability and sensor span, then leave the lift at the top.
2. Start the left arm, then start the right arm three seconds later.
3. On each arm, search shoulder pan, shoulder lift, elbow flex, wrist flex, wrist
   roll, and gripper in the installed motor dictionary order.
4. Use the installed 50-tick increments, 100-tick shoulder increments, 0.1-second
   settling intervals, per-joint search caps, mirrored search directions, and
   current thresholds of 75 / 375 for the shoulder.
5. Return the joint after each direction using the actual **linear** three-second
   interpolation at 10 Hz, followed by a half-second wait. The shoulder uses its
   special reset position while the other joints are searched, then returns home.

The gripper searches **only toward open**: positive raw ticks on the left,
negative on the right. Its initial pose supplies the other endpoint. It does
not independently find a closed endpoint; the scene preserves this detail.
Homing is represented as a change of raw coordinate reference, without movement:
2048 for ordinary motors and 3800 / 296 for the shoulders.

The result is written to `artifacts/true_calibration/report.json`. It includes
raw detected ranges, stop causes, joint angles, provenance, and fidelity limits.
These ranges are diagnostic output before the hardware calibrator's final
25-tick gripper extension and shoulder frame centering. The scene never imports
the hardware link, connects to the robot, or writes production calibration.

## What this comparison can establish

This is a faithful replay of **calibration decisions**, with idealized servo
dynamics. Current sensing uses joint constraint effort divided by an explicit
0.1 Nm per surrogate unit. This is not a measured torque constant or a prediction
of real motor current. Gravity compensation, controller gains and force limits
are those of the existing simulation; hardware PID registers are not equivalent.
The lift uses an assumed 20 mm/s drive and a synthetic 10-bit sensor spanning
the existing model limits. The chassis is held in place for inspection.
Both arm direction conventions come from the existing source mapping; the right
arm remains provisional. The completed scene holds its pose for inspection.

By default, existing model joint limits remain enabled. Servo target clipping is
disabled **in this separate model** so pushing beyond a stop creates a measurable
constraint load. An inherited 60-degree gripper stop is reported as
`model_joint_limit`, never as contact with the back of the hand. MuJoCo's soft
constraints can allow a small angular deflection under load.

The baseline model's arm self-contact is disabled because the original convex
hulls overlap. The self-collision variant below replaces those hulls and adds
the missing torso/head colliders. This experiment still does **not** certify a
new physical calibration: remaining small CAD interference, transmission
alignment, and unmodeled internal motor stops need validation against hardware.

## Diagnostic variants

```bash
# Test the controller without opening windows.
python -m examples.true_calibration --headless

# Compare what happens when the inherited gripper angle stop is removed.
python -m examples.true_calibration --free-gripper --skip-lift --arms left

# Run only one arm, omitting the lift stage.
python -m examples.true_calibration --arms left --skip-lift

# Enable existing self-contact and remove every arm angle stop.
python -m examples.true_calibration --self-collision --free-arms --skip-lift
```

The free-gripper probe changes only this scene's two gripper joint-limit flags.
It still uses the real search cap (33 steps of 50 ticks). Without validated
self-contact, it reaches that cap rather than inventing a hand stop. Gripper
motion uses the direct 4096-tick motor rotation, not the legacy normalized
0–100 to −5–60 degree fit under investigation. The other motors include the
existing shoulder and wrist transmission ratios.

## Self-collision experiment

`--self-collision` now uses cached convex decompositions of the CAD surfaces by
default. It also disables MuJoCo's parent/child contact filter: otherwise contact
between the moving finger and its parent hand assembly would still be omitted.
Rigidly attached pieces retain MuJoCo's same-weld-body filtering. Duplicate
visual meshes remain non-colliding. The torso and head have solid outer
collision envelopes derived from their visible CAD panels, doors, and dome
meshes. These close the body volume that the original experiment left empty.
They preserve the outer profile but do not model access through apertures or
inside the chassis. The shoulder pivots sit outside these envelopes, so they
introduce no initial arm/body intersections. The lower chassis proxy remains.

`--free-arms` disables angle limits on all twelve arm joints, including both
grippers. It leaves the lift endpoints and the real controller's finite search
caps in place. This changes only the separate scene. The primary robot model,
source URDF, and hardware mappings are untouched.

For comparison, `--collision-model cad-hulls` uses the original whole-part hulls
without trimming or excluding their intersections. In the parked pose those
hulls already produce numerous contact points, including approximately 30 mm of
penetration around the motor supports and biceps. Their forces obstruct
calibration before a physical robot would stop.

The default `--collision-model decomposed` replaces each collision mesh with
multiple convex pieces while preserving its body transform. CAD recesses and
holes remain open instead of being filled by a whole-part hull. CoACD 1.0.14
generates these pieces offline with a 1 mm real-metric concavity criterion and
fixed seed. That criterion is an approximation setting, not a certified bound
on every surface error. No contact pairs are silently excluded. Source and
generated-asset hashes are checked before loading.

### Elbow and wrist bearing correction

The saved hardware encoder spans and the previous collision recording exposed
two major outliers on the left arm (the right arm showed the same obstruction):

| Joint | Saved hardware travel | Previous simulated travel |
| --- | ---: | ---: |
| Shoulder pan | about 186 degrees | about 178 degrees |
| Shoulder lift | about 220 degrees | about 221 degrees |
| Elbow flex | about 276 degrees | about 157 degrees |
| Wrist flex | about 208 degrees | about 217 degrees |
| Wrist roll | about 103 degrees | about 0.6 degrees |
| Gripper | about 137 degrees | about 129 degrees |

These are travel spans, using the saved raw encoder ranges and source gearing,
not independently measured absolute CAD endpoints. The gripper's saved range
also includes hardware calibration postprocessing. They identify discrepancies;
they are not replacement angle limits or targets for forcing the simulation.

At the early elbow stop (about 28 degrees), the forearm/support bearing collar
contributed about 7.1 Nm of friction out of 7.9 Nm total simulated constraint
load. Wrist rotation stopped within 0.4 degrees of home, with hundreds of shaft
and mounting-face contacts. These mating interfaces already overlap in the CAD
assembly; treating them as high-friction exterior collisions locks the hinges.

The decomposed scene now subtracts narrow bearing clearances from the collision
pieces only. The elbow support has an 18 mm radius recess between -22.5 and
-19.5 mm along its joint axis (mirrored to +19.5 through +22.5 mm on the right).
The rotating hand base has a 12.5 mm radius recess between -1.5 and +2.5 mm along
the wrist-roll axis. These bounds cover the diagnosed mating interference while
leaving the surrounding walls intact. Each recess is a bounded 32-sided prism;
the remaining pieces stay separate convex colliders so MuJoCo cannot fill the
bearing hole again. No body-pair exclusions or global friction reductions are
used. Visible meshes, inertias, gearing, search steps, and current thresholds
are unchanged. The report marks `bearing_clearance_applied` for this variant.

Isolated checks cover elbow travel through both 120-degree poses and wrist roll
through both 54-degree poses on both arms, along with short servo tracking tests.
The outer bicep/forearm, hand/finger, and torso stops remain active. The remaining
folded-elbow housing stop is still approximate CAD contact; its exact endpoint
and any internal wrist transmission stops require validation against hardware.
Reaching the controller's search cap is still reported as such, never as a
discovered physical stop. Changing the collision code invalidates old recordings
automatically; the next requested playback will prepare a fresh recording.

Force arrows start hidden. **F** reveals them with an adaptive display scale
that limits the longest arrow to 8 cm, so large loads cannot obscure the robot
as in the original experiment. This scaling does not change physics or
calibration thresholds. **V** also shows the blue torso/head envelopes.

The generated assets in `models/collision` need only the normal runtime
dependencies. To regenerate them:

```bash
python -m pip install -r requirements-collision.txt
python -m integrations.build_collision_meshes
```

Geometry generation follows [CoACD's convex decomposition workflow](https://github.com/SarahWeiii/CoACD).
Each convex component becomes a separate MuJoCo mesh/geom; merging them back
into one collision mesh would recreate the original filled-hull problem.

The report now records the initial active contacts and the contact pairs
crossing the moving joint at each detected stop, including penetration depth
and simulated normal force. `search_outcomes` distinguishes a contact stop,
an angular limit, and exhaustion of the real controller's finite search cap.
Use a separate report path to preserve comparisons:

```bash
python -m examples.true_calibration --self-collision --free-arms --skip-lift --report artifacts/true_calibration/self_collision.json
```
