# Randomized box demonstrations for X-VLA

For the portable two-RTX-5090 training handoff, eight-hour workflow and
real-robot feature contract, see [the training guide](box_xvla_training.md).

The pilot collects 100 successful simulated pickups in LeRobot v3 format. Every
frame has the exact task **`Pick up the box`**. The robot uses its left arm with
the wheels and elevator fixed; the box starts upright. There is no hardware
connection, grasp weld, object pose assistance or upload.

The source is the completed [stage C map](box_workspace.md). Each accepted
location keeps its reserved train/validation/test group. Each candidate cycles
through successful anchor rotations, adds up to 3 mm of position jitter and
1.5 degrees of yaw jitter, and stays within +/-20 degrees. Each candidate must
pass the full physics, collision, grasp, lift and two-second hold checks again.
An accepted anchor does not automatically qualify its jittered variants.

## Completed local pilot

The November 16, 2026 run is ready at
`artifacts/datasets/box_xvla_pilot_v3`, with source recordings in
`artifacts/box_xvla_pilot`.

| Result | Value |
| --- | --- |
| LeRobot format | v3.0 |
| Exact task | `Pick up the box` |
| Successful episodes / attempts | 100 / 115; 15 rejected attempts excluded |
| Unique sampled positions / yaw values | 100 / 91, around six location groups |
| Exported time steps | 94,600 at 50 Hz |
| Camera features | Four synchronized RGB streams, 320 x 240 |
| Train / validation / test episodes | 50 / 17 / 33 |
| Approximate dataset size | 717 MB |
| Minimum measured hold clearance | 61.1 mm |
| Maximum measured hold drift | 4.82 mm |

All 281 regression tests passed. The full dataset check decoded 1,200 images
(four cameras at the start, middle and end of every episode), verified exact
task labels and train-only normalization, and passed the reduced X-VLA gradient,
held-out loss and six-channel inference checks. It used LeRobot 0.6.2,
Transformers 5.3.0 and PyTorch 2.9.1. Results are saved in the dataset's
`xvla_compatibility.json`.

The source folder also includes `dataset_preview.png` and
`physics_quality.json`. The dataset is a local pilot around six accepted
position groups, not evidence of continuous table coverage or a trained policy.

## Randomization and recording

Box/table dimensions, collision geometry and parked robot pose stay fixed.
Each seed samples appearance, lighting, PBR roughness, exposure, white balance,
pixel noise, estimated box mass (115..125 g), gripper sliding friction
(.35...40) and table sliding friction (.35...45). Camera perturbations are fixed
within each episode: +/-1 mm in each mount axis, +/-0.5 degrees per local
rotation axis and +/-1% vertical FOV. Physics recording, camera calibration
metadata, video rendering and Studio review use the same realized transforms.
These are assumed variation ranges, not measured physical calibration.

The full-collision teacher records observations and upcoming commanded targets
at 50 Hz. All four 320 x 240 RGB streams share that clock. Radian positions,
velocities, camera transforms and the box pose remain diagnostic source labels.
Only these features enter the LeRobot learner data:

| Feature | Contents |
| --- | --- |
| `observation.state` | Six observed left-arm joint/gripper positions |
| `action` | Six upcoming absolute position targets in the same normalized hardware units |
| `observation.images.front_left` | Left head camera |
| `observation.images.front_right` | Right head camera |
| `observation.images.wrist_left` | Left hand camera |
| `observation.images.wrist_right` | Right hand camera |
| `task` | `Pick up the box`, resolved through LeRobot's task metadata |

Every successful episode has 947 raw observations, of which 946 have valid
actions. The terminal observation is excluded from imitation training. Failed
attempts stay in the source with rejection reasons and are excluded from export.
All rotations and position jitters around one location stay in the same split;
the +/-3 mm regions of different locations cannot overlap. Global normalization
statistics are recomputed using training episodes only.

## Generate and export

Run from the repository root. Use a fresh output directory for another pilot.

```bash
# CPU physics collection; resume an identical interrupted/completed run with --resume.
./mujoco4.cmd -m examples.box_dataset_pilot --count 100 --jobs 4 --phase physics --output artifacts/box_xvla_pilot_new
# In a second terminal, render successful episodes as physics finishes.
./mujoco4.cmd -m examples.box_pilot_render --follow --jobs 2 --output artifacts/box_xvla_pilot_new
# Or render after collection has completed, omitting --follow.
```

The collector owns the manifest during physics. The renderer writes signed
per-episode completion markers and merges only after collection finishes.
Two independent GPU processes limit graphics memory use; each owns its context.
No viewer or screen-capture session is opened. Restarting rendering reuses
completed markers. A partially written physics attempt without a final result
is preserved and requires a fresh output path.

Export with the existing LeRobot Python environment (PowerShell):

```powershell
$lerobotPython = 'C:/Users/Theor/Documents/WebsiteCode/VulcanLerobot/lerobot-vulcan/.venv/Scripts/python.exe'
& $lerobotPython -m integrations.export_box_dataset --source artifacts/box_xvla_pilot_new --output artifacts/datasets/box_xvla_pilot_v3_new --repo-id local/sourccey-box-xvla-pilot --policy xvla
& $lerobotPython -m integrations.check_box_xvla --dataset artifacts/datasets/box_xvla_pilot_v3_new
```

Once the full physics quota is complete, export can also run concurrently with
the parallel renderer by adding `--follow`. It waits for matching signed video
completion markers and records the final source-manifest hash after rendering
finishes. Ordinary export still requires an already completed image batch.

The exporter requires the small published X-VLA base configuration at
`artifacts/xvla_setup/xvla-base-config.json`, or `--xvla-base-config PATH`.
The compatibility checker uses the downloaded BART tokenizer at
`artifacts/xvla_setup/bart_tokenizer`, or `--tokenizer PATH`. These files were
downloaded from `lerobot/xvla-base` and `facebook/bart-large`; no model weights
are needed for export or the reduced compatibility check. The inspected base
configuration revision was `cdb7964e4fe842935d671bfab5a5ebe00a96648c`.

Dataset output contains standard `meta/`, `data/` Parquet and `videos/` paths,
plus `episode_splits.json`, `simulation_provenance.json` and `xvla_train.json`.
The provenance maps every exported episode back to its exact source scene,
location split, randomization seed, mapping hash and camera calibration.
Export preserves source failures and refuses incomplete image batches or
episodes that fail physics checks.

## X-VLA check and training boundary

The training configuration uses X-VLA `auto` action mode: six real action
coordinates are padded to the checkpoint's 20 model coordinates. The published
checkpoint's architecture, including its **20-coordinate proprioception width**,
is preserved. Four image features replace its original camera feature names.
State/actions use training-only mean/std normalization; the official X-VLA
processors apply image scaling, ImageNet normalization, BART tokenization and
domain ID insertion. See the [official X-VLA documentation](https://huggingface.co/docs/lerobot/en/xvla).

`check_box_xvla` validates the full model configuration without allocating its
weights, reads the beginning/middle/end of every exported episode, and checks
all camera features and exact task text. It then runs three gradient steps,
held-out validation loss and six-channel inference through the installed X-VLA
implementation using a deliberately reduced, random-initialized Florence-2
backbone and transformer. It saves `xvla_compatibility.json`.
It also recomputes joint/action mean, standard deviation, minimum and maximum
from training rows in the exported Parquet files and verifies the saved statistics.

This checks the data/model interface. It does not establish pretrained-weight
compatibility by loading weights, sufficient GPU memory for full fine-tuning,
or closed-loop pickup success. No deployable policy is saved. Full pretrained
fine-tuning and held-out simulation rollouts are the next stage after reviewing
the dataset; `xvla_train.json` is preparation for that stage, not a completed run.
The local LeRobot fork's unrelated eager policy imports are isolated inside the
checker process; no files in that checkout are changed.
PyAV random-access decoding was slow in the full Windows check; decoder and
DataLoader throughput still need benchmarking before a full training run.

Review an existing successful episode without recomputing physics:

```bash
./mujoco4.cmd -m examples.box_pilot_review --source artifacts/box_xvla_pilot
```

Use `--episode EPISODE_ID` from the source manifest to select another instance.
Generated datasets, videos, tokenizer files and diagnostic artifacts remain
local under ignored `artifacts/` directories.
