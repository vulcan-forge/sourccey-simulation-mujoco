# Benchmark camera framing comparison

The literal interpretation of the supplied mount angles and 120-degree lens
DFOV did not match the recorded 320x240 benchmark feeds. The current cameras
use a provisional image-framing fit to all six local datasets under
`C:/Users/Theor/.cache/huggingface/lerobot/Benchmark`. Source datasets are read
only. Prior camera files are preserved as
`models/source/camera_poses.visual_fit_20261001.json` and
`models/source/camera_poses.specified_20261116.json`.

The saved fit and source metadata hashes are in
`models/source/camera_benchmark_fit_20261116.json`. This is an approximate
rendering alignment, not an independent physical camera calibration.

## What was compared

Four frames per dataset (initial, ten seconds, middle and final) provide
24 joint-state samples and 96 camera images. Decoded video PTS must agree
with the chosen telemetry timestamp within 50 ms. Example sheets and raw
extracted frames are in `artifacts/benchmark_camera_fit/`. Eight real/before/
after sheets are in its `final_comparisons` subdirectory.

Rendered comparisons use the measured benchmark table and chest-to-near-edge
gap. Recorded `observation.state` is converted using the existing affine joint
mapping to display both arms. The real shirt and office furniture are not
modeled; comparisons focus on optical framing, gripper silhouette and tabletop.

Historical motor calibration has not been recovered independently. The right
arm mapping remains provisional. The benchmark does not register lateral base
position, facing angle, exact joint start pose or base trajectory. These are
limits on a pixel-perfect whole-arm/table comparison even with correct cameras.

## Current approximation

The head fit preserves the supplied +/-15-degree yaw and mean 20-degree
downward pitch. SIFT correspondences in the top 85 pixels of initial eye
images are filtered by a 0.7 descriptor ratio and 2-pixel homography RANSAC.
Three black-shirt datasets supply 38 training pairs; the three white-shirt
datasets supply 52 held-out pairs. A robust rotation-only fit estimates shared
centered pinhole focal length, relative pitch and image roll.

| Parameter | Current value |
| --- | --- |
| Effective focal length at 320x240 | 283.59 pixels |
| Effective vertical FOV | 45.8705 degrees |
| Head yaw, left / right | +15 / -15 degrees |
| Head elevation, left / right | Approximately -20 / -20 degrees |
| Head image roll, left / right | Approximately +7.34 / +0.86 degrees |
| Nominal lens DFOV, retained as reference | 120 degrees |

Held-out median correspondence error is about 2.51 pixels under those
assumptions, versus 91.25 pixels for the literal 120-degree diagonal pinhole
view. These points are mostly distant room features; depth, finite stereo
baseline and distortion remain unmodeled. The fit is conditional on the
nominal head angles. It cannot establish an independently measured lens FOV
or identify whether crop, distortion, camera specifications or other model
errors explain the difference.

Wrist camera centers remain at the nominal CAD openings. Their orientation
receives a bounded camera-frame XYZ adjustment around the archived mechanical
mount. The approximate deltas are [-5.83, -1.20, +2.37] degrees on the left and
[-3.21, +8.87, +4.29] on the right. These are local corrections, not replacement
mechanical mount angles. The nominal 16.9-degree inward and 2.5-degree upward
specification stays in the source metadata.

The wrist objective uses median dark foreground contours from the six initial
frames. Both fixed palm and moving finger CAD geometry are included. Cloth,
reflections and exposure affect this heuristic; regression checks tolerate
9 pixels median error on the left and 6 on the right. Images from later poses
are also retained for visual inspection. This does not establish full hand-eye
calibration or correct historical joint mapping.

Ray checks found the old head optical centers inside the CAD eye platform.
The left center is advanced 10 mm and the right 8 mm along the fitted optical
axis. These are clearance estimates rather than surveyed optical offsets.
The rendering near plane is reduced from 14 mm to 1.4 mm for the model's 1.4 m
extent, so nearby geometry is visible. Physics geometry and limits are unchanged.
The bottom camera was not fitted.

## Reproduce the inspection

From this simulation repository, use the existing LeRobot Python for PyAV,
PyArrow and OpenCV extraction, then the simulation Python for MuJoCo renders:

```powershell
$benchmarkPython = 'C:/Users/Theor/Documents/WebsiteCode/VulcanLerobot/lerobot-vulcan/.venv/Scripts/python.exe'
& $benchmarkPython -m integrations.benchmark_camera_samples --source C:/Users/Theor/.cache/huggingface/lerobot/Benchmark --output artifacts/benchmark_camera_fit
.venv/Scripts/python.exe -m integrations.fit_benchmark_cameras --output artifacts/benchmark_camera_fit/final_comparisons
```

Add `--fit` to the second command to recompute candidate fitting reports.
That option does not alter the source camera file. The bounded silhouette
optimizer is approximate and may vary slightly across renderers. The saved
source report records the promoted values. Camera regression inputs are retained
in `tests/fixtures/benchmark_camera_observations.json`.

To inspect new simulated box instances with the active cameras:

```bash
python -m examples.box_training --count 3 --seed 20261116 --output artifacts/box_camera_benchmark_fit --view
```

Use a fresh output directory. Earlier episode videos retain their original
optics and provenance, and their playback signatures intentionally reject
changes to the camera source. No real robot or Host changes are made by this work.
