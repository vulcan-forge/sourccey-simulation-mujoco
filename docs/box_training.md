# Box pickup: rotation curriculum and demonstrations

This implements the first part of test B: one measured table/box placement,
with ten seeded upright box rotations between approximately -11 and +11.7
degrees. The stationary robot uses its left arm. The teacher computes the
approach, finger normal and clearance from the settled simulated box pose.
That privileged pose is kept for diagnosis; the exported learner inputs are
four camera images and six observed joint/gripper positions.

The teacher remains a scripted controller. The demonstrations and model
compatibility check do not establish a learned pickup policy or real-world
transfer. Wider rotations, position variation, base motion, and recovery are
subsequent curriculum stages.

## Review the generated ten instances

From the repository root, with the simulation environment active:

```bash
python -m examples.box_training --review
```

`box-training.cmd` launches the same reviewer. The ten scenes are in
`artifacts/box_pickup_b/`; `instances.png` shows all ten final grasps. Controls:

| Key | Effect |
| --- | --- |
| N / Right arrow | Next instance |
| B / Left arrow | Previous instance |
| R | Replay the current instance |
| P | Pause/resume |
| 1 / 2 | Overview/gripper view |
| F | Toggle colored camera field-of-view outlines |
| C | Toggle the four camera feed tiles |

The feeds come from `front_left`, `front_right`, `wrist_left` and `wrist_right`.
Their colors match the outlines in the main view. Wrist feeds follow the
moving hands. These four are used for this dataset; the model's additional
bottom camera is excluded. The preview refreshes at 10 Hz; recordings have
50 Hz images. Mouse orbit/zoom remain available. Switching instances opens a
new MuJoCo window with that instance's materials and lights.

Review is recorded-pose playback, using visible CAD geometry to reduce graphics
upload cost. The measured results come from the separate full-collision physics
run, including CAD hand contacts, robot self-collision, table/leg checks,
fixtures and free-box dynamics. Playback never changes the outcome.

**Physical optical calibration is still pending.** The active four cameras use
the [benchmark framing fit](benchmark_camera_fit.md): nominal head yaw and mean
pitch, fitted image roll, approximately 45.87 degrees vertical effective FOV,
and bounded wrist orientation corrections at the CAD opening centers.
Both the older visual fit and literal 120-degree lens interpretation are
archived under `models/source/camera_poses.*.json`. The four views let you
inspect their current orientation; they do not prove physical alignment.
Metadata records the source, body-relative transforms, ideal pinhole intrinsics,
resolution and optical-axis convention. Each frame also records world camera
transforms. Physical lens distortion and a multi-pose extrinsic calibration
remain unmeasured. Camera poses deliberately stay fixed in this initial batch
so orientation inspection is meaningful.

The original `artifacts/box_pickup_b` batch and LeRobot export retain their
original narrow fitted-camera images and provenance. Changing camera mounts
invalidates that batch's playback signature. Generate a new batch for corrected
camera inspection; do not mix camera configurations during training.
`artifacts/box_camera_check` also retains the intermediate literal 120-degree
interpretation. Use a fresh path for the current benchmark fit:

```bash
python -m examples.box_training --count 3 --seed 20261116 --output artifacts/box_camera_benchmark_fit --view
# Later, replay those corrected-camera instances without recomputing:
python -m examples.box_training --output artifacts/box_camera_benchmark_fit --review
```

## Generate another batch

```bash
python -m examples.box_training --count 10 --seed 20261117 --output artifacts/box_pickup_b_next --view
```

Choose a new output directory: existing episodes and failed attempts are never
overwritten. FFmpeg must be on PATH for four synchronized H.264 videos.
`--no-video` is available for physics diagnosis only; such output is not an
image training dataset. `--speed` changes review speed, not physics.

Yaw is stratified over -12 to +12 degrees; the teacher explicitly rejects
values outside the initial +/-15-degree curriculum. Object dimensions,
position, table dimensions, fixtures and camera poses remain fixed. Per episode,
the seed controls:

- Table, cardboard box, robot, hand and floor colors/patterns; light direction,
  ambient/diffuse light, exposure, white balance and pixel noise.
- Estimated mass from 115 to 125 g; hand sliding friction from 0.35 to 0.40;
  table sliding friction from 0.35 to 0.45.

These narrow dynamics ranges are assumptions around the supplied estimates,
not measured real robot variation. The original 58.8 x 38.4 x 58.4 mm box and
benchmark table dimensions are preserved. Gripper torque remains the existing
unmeasured 0.6 Nm simulation cap. See [contact assumptions](box_pickup.md).
The generated `scene.json` contains every realized parameter, enabling replay.

The first eight instances are training demonstrations; the two larger positive
yaws are held out. This is a small, deliberately harder angle split rather than
a statistical estimate of broad task performance. Seeds and scene assignments
are saved; hold-out scenes must stay out of fitting and normalization.

## Recorded data and success

Each `episode_000` through `episode_009` directory includes:

| Artifact | Content |
| --- | --- |
| `scene.json` | Dimensions, pose, seed, split and randomization settings |
| `report.json` | Physics success checks, force/clearance measurements and provenance |
| `recording.npz` | Physics playback with a source/model/config signature |
| `transitions.npz` | Synchronized state, action, next state, terminal masks and privileged diagnostic labels |
| `episode.json` | Feature units, names, calibration, action semantics and imitation eligibility |
| `videos/*.mp4` | Four 320 x 240 RGB streams at 50 Hz |
| `cameras_*.png` | Initial, middle and final four-camera mosaics |
| `final.png`, `grasp_detail.png` | External inspection views |

The scene still requires a full two-second hold, minimum 50 mm lowest-corner
clearance, bilateral finger contact, less than 5 mm hold drift, no robot/table
contact, stable fixtures and no physics warnings. Failed trials remain in the
batch with `eligible_for_imitation=false`; the exporter excludes them.

Observations are captured before integration, paired with the teacher targets
set for the following interval. Measured joint state is not substituted for the
command. The six left-arm targets use the existing source-confirmed normalized
hardware mapping, with its hash saved. Raw radians and simulator states remain
available for diagnosis. Every 20 ms corresponds to ten 2 ms physics steps.
The last row is a terminal observation with no valid next action. The current
batch has 947 observations and 946 valid action samples per episode.

## Local LeRobot export

Use an environment with LeRobot v3, PyAV and PyArrow. The existing environment
in `VulcanLerobot/lerobot-vulcan/.venv` was used successfully here; no packages
or files in that checkout are changed. In PowerShell from this repo:

```powershell
$lerobotPython = 'C:/Users/Theor/Documents/WebsiteCode/VulcanLerobot/lerobot-vulcan/.venv/Scripts/python.exe'
& $lerobotPython -m integrations.export_box_dataset --source artifacts/box_pickup_b --validate-only
# Already exported for this batch; use a new path for another export:
& $lerobotPython -m integrations.export_box_dataset --source artifacts/box_pickup_b --output artifacts/datasets/box_pickup_b_v3_new
```

The completed export is `artifacts/datasets/box_pickup_b_v3`. It contains
9,460 frames across ten successful episodes, six-dimensional observations and
actions, and four video features. The terminal row is excluded. Images use
the explicit PyAV decoder, since this machine's optional TorchCodec installation
does not load. Train/test indices and full simulation provenance are included;
normalization statistics are recomputed from the eight training episodes only.
Nothing is uploaded.

For the default seed, all ten teacher pickups passed, with minimum hold
clearance between 69.1 and 70.9 mm. The scene/data regression suite passed
19 checks, all 40 raw camera-video headers matched the observation clock,
and export decoded every nonterminal source frame. These are teacher/format
results, not a learned-policy success rate.

`act_train.json` supplies a pilot ACT configuration: 50-step action chunks,
10 actions per inference, and only training episode indices. At 50 Hz these
correspond to a one-second predicted chunk and 0.2 seconds executed before
replanning. It requests pretrained ResNet18 image weights, which would need a
download if not cached. No full model training is launched by generation or
review, and ten episodes are a pipeline pilot rather than sufficient training
coverage.

For an offline dataset/model compatibility check without pretrained downloads:

```powershell
& $lerobotPython -m integrations.check_box_learning --dataset artifacts/datasets/box_pickup_b_v3
```

This constructs a small ACT model, reads real image/action batches, and performs
one gradient update and a finite-output inference check. A successful check
writes `learning_smoke_test.json`; it does not save a deployable policy.
The check passed on this batch with a two-sample, ten-action window and all
four image inputs. It runs strictly offline and places its Hugging Face dataset
cache under `artifacts/datasets/.hf_box_cache`, avoiding global cache locks.
The local LeRobot fork eagerly imports unrelated Wall-X/Qwen code whose installed
dependencies currently fail. The checker isolates ACT imports inside its own
process, leaving that checkout untouched. Standard LeRobot training CLI startup
may require fixing that fork's dependency/import issue separately; the generated
configuration alone does not fix it.

## Before real-world learning or deployment

The next useful measurements are the four physical camera intrinsics and
multi-pose extrinsics, table-to-robot registration, real gripper opening/contact
landmarks, grip force, latency and real-box mass/friction. Then compare real
and rendered images at matched poses, expand the simulated training corpus,
train a vision policy and evaluate its closed-loop success on unseen seeds.
Real camera acquisition must run independently from the buffered motion loop
before deploying visual feedback; the existing hardware replay path is not
changed by this simulation work.

Checks:

```bash
python -m pytest tests/test_box_training.py tests/test_box_pickup.py -q
```
