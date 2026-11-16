# Box pickup: test A

```bash
python -m examples.box_pickup --view --camera-feeds
# Replay the saved physics without recomputing it:
python -m examples.box_pickup --replay --view
```

`box-pickup-a.cmd` starts the first command from the repository root. Without
`--view`, the experiment runs headlessly and returns a nonzero exit code if the
task fails. The viewer plays the computed physics once and keeps the final pose
visible. **P** pauses/resumes, **R** replays, **1** restores the overview, and
**2** switches to the gripper view. Mouse orbit/zoom remain available. `--speed`
changes display speed only; it never changes the 2 ms physics timestep.
`--camera-feeds` adds both eye and both wrist views, with matching colored
field-of-view outlines. **C** toggles the tiles and **F** toggles the outlines.
Recorded playback updates camera transforms with the moving wrists.

This is a scripted baseline for test A, with one fixed initial box position and
rotation. It supplies a scene, a controller and measured success criteria for
later learning work; it does not train a policy or connect to the real robot.
This A entry point retains its fixed rotation. The separate
[test B curriculum](box_training.md) conditions the teacher's approach and
finger alignment on the settled box pose and supports small upright yaw
changes. Tests C (reachable position variation) and D (mobile reaching) are
not yet enabled.

## Dimensions and placement

The scene configuration is [`examples/box_pickup_a.json`](../examples/box_pickup_a.json).
The table values were copied from `reference_setup` in
`VulcanBenchmark/vulcan-benchmark/configs/benchmark-draft.json`; the source hash
is recorded in the configuration. They are owner-reported dimensions, not
independent precision measurements. The scene is portable and does not require
that other checkout at runtime.

| Quantity | Value |
| --- | --- |
| Table length across the robot | 1,828.8 mm |
| Table front-to-back depth | 908.05 mm |
| Top surface above the world floor | 762 mm |
| Middle chest surface to the near table edge | 104.775 mm, approximately 104.8 mm |
| Box width / depth / height | 58.8 / 38.4 / 58.4 mm |
| Complete unopened box mass | Estimated 0.12 kg |
| Initial box center from near edge | 110 mm into the tabletop |
| Initial box center left of robot centerline | 180 mm |
| Box yaw | 0 degrees, upright |

World axes are +X robot forward, +Y robot left and +Z up. The table is centered
laterally and square to the robot. Those alignment choices are assumptions:
the benchmark does not yet register the physical lateral offset or yaw. The
long edge is taken to face the robot. The CAD torso's outer surface at Y=0 and
tabletop height supplies the chest datum, approximately X=207.49 mm. The near
edge is therefore X=312.26 mm, not 104.8 mm from the base origin. Box center is
approximately [422.26, 180, 791.2] mm in the floor-based world frame. Table
height is not added to the base_link origin, which sits above the floor.

The base and elevator have explicit fixture constraints. Both arms start in
the existing parked pose; the right arm remains parked. The left arm raises
behind the near table edge, opens, approaches above and behind the grasp,
descends, aligns the fixed jaw, closes, lifts and holds. The grasp is near the
upper part of the box so that the rotating finger clears the box roof while
open. Wrist position and pad orientation are jointly optimized within existing
joint bounds. Near the wrist-flex bound, orientation and position cannot both
match an arbitrary target perfectly; measured IK residuals are retained in
the phase report rather than hidden.

## Contact physics and initial assumptions

The free box uses the owner's dimensions from
[`examples/objects/feetech_sts3250_box.json`](../examples/objects/feetech_sts3250_box.json).
No equality, attachment, object-position command, or external object force is
used to lift it. It remains free throughout the physics run. Only the base and
elevator are constrained. The separate playback viewer displays recorded poses;
those display assignments do not produce the measured pickup result.

The hand uses its decomposed CAD collision surfaces, including the diagnosed
bearing clearances, rather than the ball demo's substitute rectangular pads.
Nonadjacent robot self-collision and robot/environment collision are enabled.
MuJoCo's standard adjacent-body collision filter stays enabled, and the primary
model's joint limits remain in force. This scene does not promote the separate
calibration experiment's unlimited-joint settings into pickup control.

The supplied rubber/cardboard estimates are static friction 0.45 and dynamic
friction 0.35. MuJoCo uses a Coulomb sliding coefficient rather than separate
static/dynamic entries. This baseline uses the conservative 0.35 estimate on
box/hand contacts, with explicit separate table contact at an assumed 0.4.
The torsional and rolling entries are separate length-valued assumptions,
0.001 m and 0.0001 m. The supplied 0.05 restitution estimate determines an
initial damped-contact setting, not a certified bounce coefficient; restitution
would require a separate drop-test comparison. See
[MuJoCo contact parameters](https://mujoco.readthedocs.io/en/3.14.0/modeling.html#contact-parameters)
and [restitution](https://mujoco.readthedocs.io/en/3.14.0/modeling.html#restitution).

The initial gripper torque cap is 0.6 Nm with scene-local damping/gains. This
is a simulation setting, not a measured hardware rating. A 0.3 Nm trial grasped
the box but lost it during lifting at the estimated friction. Arm dynamics
retain the existing idealized gravity feed-forward and servo model. The box is
rigid, with centered uniform-box inertia; the enclosed servo's actual center of
mass, possible internal movement and cardboard deformation are unmeasured.
Table thickness (40 mm), legs, contact friction, CAD contact landmarks, and
physical robot registration also need comparison before real transfer. No real
calibration or hardware command mapping is changed by this scene.

## Success and artifacts

A pass requires the box's **lowest corner** to remain at least 50 mm above the
table throughout a full two-second hold. Both finger bodies must retain at
least 0.1 N normal contact throughout that interval, and box-center drift during
the hold must stay below 5 mm. Every physics step is inspected for robot/table
contact, including the legs; any such contact fails the trial. Base translation
and elevator drift must each stay below 1 mm, with no MuJoCo warnings.

Outputs are in `artifacts/box_pickup_a/`: `report.json`, `recording.npz`,
`final.png` and `grasp_detail.png`. The report includes the complete scene and
assumptions, success checks, hold measurements, finger forces, phase IK errors,
box movement before closure, and fixture drift. Recordings contain physics
states at 50 Hz and carry a signature of the scene, model, collision assets,
physics source and MuJoCo version. `--replay` rejects a mismatched recording.

The regression checks include an open-gripper negative control and verify that
a dropped sample or a short hold cannot pass based on the final frame alone:

```bash
python -m pytest tests/test_box_pickup.py -q
```
