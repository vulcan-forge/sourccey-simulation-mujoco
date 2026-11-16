# Box workspace: position first, then rotation

This implements steps 1 and 2 of the next curriculum stage (C), with the left
arm, fixed wheels and fixed elevator. It validates the complete executed
approach, grasp, lift and two-second hold. It does not collect an image training
dataset, train a model, connect to the real robot or start stage D.

**The policy selected for the later training phase is X-VLA.** Earlier ACT
examples remain historical pipeline checks; this workflow does not invoke them.
Model integration, large-scale demonstrations and policy evaluation are later
steps after reviewing this workspace map.

## Run and review

```bash
# Map the local grid, then test progressively wider rotations at valid positions.
./mujoco4.cmd -m examples.box_workspace --jobs 2 --output artifacts/box_workspace_c_new
# Continue a matching completed/interrupted run; completed trials are reused.
./mujoco4.cmd -m examples.box_workspace --resume --jobs 2 --output artifacts/box_workspace_c_new
# Open the first accepted trial in Studio, without recomputing physics.
./mujoco4.cmd -m examples.box_workspace --review --output artifacts/box_workspace_c_new
# Review a particular success or failure listed in manifest.json.
./mujoco4.cmd -m examples.box_workspace --review --output artifacts/box_workspace_c_new --trial location_01_01_yaw_p12
```

The trial name above is illustrative: only generated trials can be reviewed.
Use `workspace.png` for an overview and `manifest.json` for exact IDs, results,
per-location rotation coverage and reasons for rejection. Failed constructions
have no motion recording; other failed attempts can be reviewed in Studio.
Closing Studio releases the viewer and camera renderers.

## First completed map

The default plan was run on MuJoCo 4.0.1 on November 16, 2026. Results are in
`artifacts/box_workspace_c_validated`: **38 accepted poses out of 43 trials**,
with six of nine positions accepted at zero yaw. All accepted trials completed
the measured lift and two-second hold and passed the additional path checks.

| Depth / lateral (mm) | Accepted yaw angles (degrees) |
| --- | --- |
| 80 / 140, 180 or 220 | None; nominal pickup failed, wider angles skipped |
| 110 / 140, 180 or 220 | -20, -12, -6, 0, +6, +12, +20 |
| 140 / 140 | -20, -12, -6, 0, +6, +12, +20 |
| 140 / 180 | -12, -6, 0, +6; +12 failed, +/-20 skipped |
| 140 / 220 | -12, -6, 0, +6, +12, +20; -20 failed |

The five rejected trials failed to lift and maintain both finger contacts.
The three nominal failures also missed the grasp alignment bounds. These are
limitations of the current teacher/model combination, not proof that those
physical poses are unreachable. No continuous angular range is inferred.

Review a successful rotated pickup from that completed map:

```bash
./mujoco4.cmd -m examples.box_workspace --review --output artifacts/box_workspace_c_validated --trial location_01_01_yaw_p20
```

All 273 regression tests passed. Accepted recordings were checked against their
scene signatures, and the Studio display model was rebuilt successfully.
Generated outputs remain local and ignored by Git.

Use `--plan path/to/plan.json` to supply another grid. The default plan lives in
[`examples/box_workspace.json`](../examples/box_workspace.json). Existing output
directories are preserved. Resume requires matching source/model/engine/plan
signatures. An incomplete trial without its final assessment is preserved and
requires a fresh output directory; an assessment completed before interruption
can be recovered into the manifest. Changing the teacher or validation code
requires a fresh map.

## What is tested

The initial grid has nine positions, centered around the working A/B placement:

| Coordinate | Values | Definition |
| --- | --- | --- |
| Depth | 80, 110, 140 mm | Box center from the table's near edge |
| Lateral | 140, 180, 220 mm | Box center left of the robot's center |
| Initial yaw | 0 degrees | Upright, same orientation as A |
| Rotation stages | +/-6, +/-12, +/-20 degrees | Yaw about world +Z |

The default spans a 60 x 80 mm local patch. This is a probe, not a claim that
the whole table or even the whole patch is reachable. Exact measured box/table
dimensions, camera transforms and existing collision geometry remain in use.
Each trial starts independently from the same parked robot and a free box.

At each grid position:

1. Run the zero-degree trial through the entire teacher motion.
2. Only if accepted, test both -6 and +6 degrees.
3. Only if both pass, test -12 and +12 degrees.
4. Only if both pass, test -20 and +20 degrees.

Failure at one angle prevents higher stages at that location, but retains every
tested result, including an accepted angle whose opposite failed. Other grid
positions continue independently. Untested larger angles are unvalidated, not
classified as impossible. The C probe permits configured stages up to 45 degrees;
B keeps its existing +/-15-degree guard. No wheel motion is introduced.

Appearance, mass and friction stay fixed while testing geometry and motion.
The PBR material profile is used for review. Broader domain randomization,
random continuous pose samples and image datasets belong to the next phase.
Any future positional/angular jitter must be physically validated too.

## Acceptance criteria and diagnostics

Each trial uses the full collision model and the original A success checks:
at least 50 mm lowest-box-corner clearance, both finger contacts, two seconds of
holding, at most 5 mm hold drift, no robot/table/leg contact, stable base/elevator,
and healthy physics. There is no grasp weld or object pose assistance.

Additional C checks cover the path and setup:

- Reject active robot self-contact with normal force above 0.1 N at any physics
  step. The normal adjacent-body filtering and existing bearing clearances remain
  part of the model; this is validation of that collision approximation.
- Reject robot/floor contact above 0.1 N except the four intended wheel contacts.
- Require joint transit endpoints to track their targets within 5 degrees.
- Require grasp alignment IK error no greater than 6 mm and pad-normal alignment
  no greater than 8 degrees before closing.
- Reject box displacement above 2 mm before the closing command.
- Bound approach/descend/lift IK errors to 35 mm and record the value by phase.

The existing five-axis teacher cannot satisfy every requested Cartesian pose and
pad direction exactly. At the original successful placement, measured diagnostic
IK errors were about 27 mm at approach, 24 mm during lift, and 4.7 mm during grasp
alignment. The 35 mm transit bound explicitly allows that existing approximation;
it is not a millimeter-accurate trajectory claim. Tighter grasp checks and actual
contact/hold criteria determine whether the pickup works. These thresholds are
simulation validation choices, not measured robot tolerances.

## Outputs and the next phase

| File | Meaning |
| --- | --- |
| `manifest.json` | Plan, source signature, every completed trial and summary |
| `workspace.png` | Grid showing accepted positions and tested rotations |
| `validated_scenes.json` | Explicit whitelist of accepted, discrete scene configurations |
| `trial_id/scene.json` | Exact dimensions, box pose, fixtures and thresholds |
| `trial_id/assessment.json` | Acceptance checks, path errors, self-contact diagnostics and rejection reasons |
| `trial_id/report.json` | Full teacher physics report and C assessment |
| `trial_id/recording.npz` | Compact replay of actual simulated states |

Validation runs omit duplicate compiled `model.mjb` snapshots: each would repeat
roughly 288 MB of unchanged geometry. Studio reconstructs the model from the
signed scene, so failed motions remain inspectable without these copies. Standard
A/B recordings still export their model snapshots by default.

Each grid position has a deterministic reserved `train`, `validation` or `test`
split. All rotations at that position keep the same split. The catalog is not
an image dataset and does not contain learned-policy performance measurements.
It proves success only for its accepted discrete poses under this model and
teacher. Future collection must preserve those location groups, log new
randomization, and avoid leaking held-out locations into X-VLA training.

```bash
./mujoco4.cmd -m pytest tests/test_box_workspace.py tests/test_box_pickup.py -q
```
