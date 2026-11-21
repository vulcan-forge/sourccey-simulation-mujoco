# Linux Codex handoff: randomized box pick and place

Read this file first on the Linux GPU computer. Continue this repository's
simulation/data/training work using one RTX 5090 initially. The user wants a
validated overnight collection followed by training and evaluation. This Git
handoff contains code and reference inputs; generated episodes, datasets,
videos, downloaded weights, environments and old checkpoints are excluded.

## Objective and task contract

Collect successful programmatic **pickup and placement** demonstrations of the
box at continuously randomized reachable positions and upright yaw angles on
the table, with randomized textures, lighting, materials and camera variation.
Train X-VLA on camera images and observed joints/actions, then evaluate new
held-out poses. Privileged object pose is available to the scripted teacher and
success evaluator only, never as a policy input or a policy fallback.

Fixed destination: box center 110 mm from the table's near edge and 180 mm left
of the robot. Starting poses must be at least 40 mm from that destination.
The broad plan is `examples/box_random_dataset.json`: depth 80..240 mm,
lateral 80..320 mm, yaw -40..40 degrees, 64 XY/yaw strata. These are proposed
bounds; the current teacher has not demonstrated success everywhere inside them.
The small smoke plan is a software check, not a useful final training dataset.

Keep existing lift, hold, transport, release, settled-placement, collision and
joint-target checks. Never attach/weld/teleport the box to produce successes,
clamp invalid training actions, substitute easy regions for difficult ones,
or count a successful scripted demonstration as a trained-model success.

## Evidence from Windows

- The historical pickup-only model scored 0/17 on the old validation scenes.
  Its old dataset covered tiny position clusters, not the requested workspace.
- The new pick-and-place pipeline passed a narrow six-episode check, four-camera
  rendering, LeRobot export, reduced-model gradient checks and trainer preflight.
  No full pretrained model has been trained on this new task.
- A broad fixed 50-attempt baseline produced 20 successes and 30 rejections.
- The same 50 frozen scenes with doubled gripper sliding friction produced
  **30 successes and 20 rejections**: 11 improved, one regressed. Both comparisons
  retain poses, appearance, mass, table friction, controls and success criteria.
- Effective grip friction was approximately 0.35..0.40 before and 0.70..0.80
  afterward. This is an uncalibrated approximation for ridged pads, not adhesion
  or measured physical calibration. Table friction was unchanged.
- Local recordings, datasets and comparison reports remain on Windows under
  ignored `artifacts/`; they are deliberately absent from this clone.

Current regular MuJoCo dynamics and SciPy IK run on the CPU. Rendering and
training use the GPU. The collectors currently allow at most four CPU workers,
and rendering one or two workers. Measure throughput before changing those
limits. A GPU physics port to MJX/Warp is separate work requiring equivalence
checks; a 5090 alone does not move the current physics onto CUDA.

## Fresh Linux setup

Use Python 3.12, Git, a working NVIDIA driver, FFmpeg and system EGL/OpenGL
libraries. Inspect `nvidia-smi`, CPU count, RAM and free SSD space first. On
Ubuntu/Debian the usual system prerequisites include `python3-venv`, `ffmpeg`,
`libegl1` and `libgl1`; choose package names for the actual distribution.
Install `uv` separately for training. Do not install desktop/robot extras.

From the repository root:

```bash
bash setup-linux.sh
bash mujoco4.sh -m integrations.linux_preflight --render
bash mujoco4.sh -m pytest tests/test_box_random_dataset.py tests/test_box_attempt_batch.py tests/test_box_friction_experiment.py tests/test_box_pilot_render.py tests/test_box_pilot_export.py tests/test_box_xvla_training.py tests/test_linux_handoff.py -q
```

`mujoco4.sh` defaults to `MUJOCO_GL=egl`, Filament and OpenGL, with explicit
environment overrides. Linux headless rendering has **not** been verified on
the target machine. Inspect the four images in `artifacts/linux_preflight/`.
Investigate failures rather than silently switching renderer midway through
a dataset. Renderer and physics provenance must be identical within a run.
Do not regenerate tracked MJCF/collision assets just to set up the environment.

`setup-linux.sh` restores the exact reference joint mapping from
`handoff/linux/robot_joint_mapping.json` to its legacy runtime path under
`artifacts/`. A legacy mapping is already tracked there, but Git can convert its
line endings across platforms. The helper restores the fingerprinted reference
bytes when only newlines differ, preserving the clone copy as a backup. It
verifies the checksum and refuses to overwrite different calibration values.
Meshes, collision caches and camera specifications already
live in tracked `models/`. They are required inputs, not generated episodes.

## First Linux collection

Run fresh small outputs; do not attempt to resume Windows recordings without
transferring their signed inputs and addressing source newline fingerprints.
First verify the narrow pipeline:

```bash
bash mujoco4.sh -m examples.box_random_dataset --plan examples/box_random_smoke.json --phase probe --attempts-per-cell 1 --jobs 2 --output artifacts/linux_smoke_support
bash mujoco4.sh -m examples.box_random_dataset --plan examples/box_random_smoke.json --phase collect --support artifacts/linux_smoke_support/manifest.json --train-per-cell 1 --validation-per-cell 1 --test-per-cell 1 --jobs 2 --output artifacts/linux_smoke_data
bash mujoco4.sh -m examples.box_pilot_render --output artifacts/linux_smoke_data --jobs 1
```

The Windows friction override currently works **only** for paired runs with
`box_attempt_batch --reference <existing baseline> --grip-friction-scale 2`.
It does not change the default random collector. Before large Linux collection,
add an explicit, validated fresh-collection friction option/plan setting using
the same effective range and provenance. Cover both probe and collection;
freeze it in the sampling/source signatures and reject mixed-profile resumes.
Do not silently fall back to baseline friction or require thousands of duplicate
baseline simulations just to create fresh ridged-pad samples. The initial smoke
commands above deliberately exercise the unchanged baseline pipeline.

Then sample the broad workspace, inspect failures and coverage, benchmark
successful episodes/hour and disk GB/episode, and set an overnight budget.
Keep train/validation/test independent and frozen; ensure each supported XY/yaw
stratum receives its quota. Save every rejection and its reason. Existing
collectors preserve partial outputs and enforce matching signatures on resume;
unrecognized partially written episodes require review. Add any needed robust
overnight interruption handling before relying on unattended recovery.

Progress for fixed-budget batches works on Linux without a GUI:

```bash
bash mujoco4.sh -m examples.box_status --watch
# For quota-based collections inspect manifest.json and coverage.json directly.
```

Run long jobs in a persistent session such as tmux, capture stdout/stderr,
leave disk headroom, and avoid changing source while collection/rendering runs.
Do not start a large run until Linux physics, camera recording/export and
throughput checks pass. See `docs/box_random_dataset.md` for acceptance details,
coverage/split logic and later learned-policy evaluation.

## Training dependencies, export and checks

```bash
bash setup-linux.sh --training
lerobot_python="$PWD/artifacts/linux_deps/lerobot-vulcan/.venv/bin/python"
"$lerobot_python" -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.ones(8, device="cuda").sum().item())'
"$lerobot_python" -m integrations.export_box_dataset --source artifacts/linux_smoke_data --output artifacts/datasets/linux_smoke_v3 --repo-id local/sourccey-linux-smoke --policy xvla
"$lerobot_python" -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("facebook/bart-large").save_pretrained("artifacts/xvla_setup/bart_tokenizer")'
"$lerobot_python" -m integrations.check_box_xvla --dataset artifacts/datasets/linux_smoke_v3
"$lerobot_python" -m integrations.prepare_box_xvla_training --dataset artifacts/datasets/linux_smoke_v3 --lerobot-repo artifacts/linux_deps/lerobot-vulcan --output artifacts/linux_smoke_training_handoff
"$lerobot_python" artifacts/linux_smoke_training_handoff/train.py --check-only --gpus 1 --output artifacts/linux_smoke_training_check
```

The dependency helper clones the original Vulcan forks at the exact revisions
in `handoff/linux/manifest.json`, with a file checksum inventory and the local
package layout required by the lockfile. Unrelated uncommitted Windows speech
and hardware teleoperation changes are excluded. No training data or dependency
source trees are vendored. GitHub access to both repositories is required;
configure Git authentication on Linux if they are private. Existing dependency
directories are checked and preserved, never reset automatically.

For fixed-budget batch automatic export, pass
`--export-python "$lerobot_python"` explicitly: the legacy default is Windows.
For evaluation pass `--policy-python "$lerobot_python"` and
`--lerobot-source "$PWD/artifacts/linux_deps/lerobot-vulcan"` explicitly.

After the Linux dataset preflight, run a short **full pretrained** one-GPU
training smoke with checkpoint save/reload. The reduced CPU check is not a
VRAM test. Explicitly use `--gpus 1` or `GPUS=1`; the legacy training launcher
defaults to two GPUs. Tune per-GPU batch size if needed. Follow
`docs/box_xvla_training.md` for the training/checkpoint contract, substituting
the new pick-and-place dataset and its fixed-destination contract for the old
pickup-only examples. A one-GPU main run can use the prepared bundle:

```bash
GPUS=1 PYTHON="$lerobot_python" bash artifacts/NEW_TRAINING_BUNDLE/run.sh artifacts/NEW_TRAINING_OUTPUT
```

The bundle's eight-hour budget includes a 20-step pretrained smoke; download
weights first if you want the budget to represent mostly training. Keep final
test scenes reserved until model/runtime settings are frozen. Evaluate actual
closed-loop pick-and-place success and report failures by position/yaw stratum.

## Files and continuation priorities

- `examples/box_random_dataset.py`, its plans and `sourccey/box_task.py`: teacher,
  sampling, task checks and quota collection.
- `examples/box_attempt_batch.py`, `box_friction_experiment.py`, `box_status.py`:
  fixed-budget comparison, recovery and progress monitoring.
- `examples/box_pilot_render.py`, `integrations/export_box_dataset.py`: signed
  replay rendering, synchronized four-camera export and train-only statistics.
- `training/box_xvla/`, `integrations/check_box_xvla.py`,
  `integrations/prepare_box_xvla_training.py`: portable training/checks.
- `examples/box_policy_eval.py`, `box_policy_sweep.py`,
  `sourccey/xvla_eval.py`, `integrations/xvla_worker.py`: learned-policy evaluation.

First verify Linux setup. Then implement fresh ridged-pad collection, assess
remaining teacher failures, benchmark and launch a balanced resumable overnight
run. Keep the large dataset and training outputs on the Linux SSD; return small
reports, selected replays and final checkpoints over SSH as needed. Do not push
generated data, credentials or model weights to this Git repository.
