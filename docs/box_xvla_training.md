# Train the Sourccey box policy on two RTX 5090s

This handoff targets **Linux, two RTX 5090 GPUs with 32 GB VRAM each, and an
approximately eight-hour session**. It fine-tunes `lerobot/xvla-base` on the
existing LeRobot v3 dataset, with the exact instruction **`Pick up the box`**.
It does not connect to a robot, publish anything, or claim a sim-trained policy
is ready for hardware.

## What to transfer

Copy the entire generated `artifacts/box_xvla_training_handoff_5090/` directory to
the training computer. It contains:

- `dataset/`: all four camera videos, Parquet data, metadata, split lists and provenance.
- `train.py` and `run.sh`: the portable launcher and eight-hour workflow.
- `robot_joint_mapping.json` and `camera_poses.json`: the calibration references
  used for this dataset, needed again when bringing the model back.
- `handoff.json`: dataset file hashes, the inspected training source hashes,
  pretrained revision, task and deployment contract.
- This README.
- `lerobot-vulcan/` and `packages/lerobot-robot-sourccey/`: small installable
  snapshots of the inspected training code and its local robot package, in the
  relative layout required by the fork's lockfile. They contain no virtualenv,
  Git credentials, caches or unrelated experiments.

The included snapshot is the **same `lerobot-vulcan` source** inspected here,
revision `7461069b79c56d616ee9445db63202a4788e6061`. Use the fork, not a generic
`pip install lerobot`. The launcher verifies the important source files against
`handoff.json`, allowing Windows/Linux line-ending differences. Source-file
hashes are authoritative if the local checkout has changes relative to its commit.
No absolute Windows path from the dataset template is used for training.

## Install and run on Linux

In the included `lerobot-vulcan` snapshot, create its environment using its
existing lockfile and extras. These commands assume `uv` is installed. Replace
`/path/to/box_xvla_training_handoff` with the transferred folder's actual path:

```bash
cd /path/to/box_xvla_training_handoff/lerobot-vulcan
uv sync --frozen --python 3.12 --extra training --extra xvla
source .venv/bin/activate
python -c 'import torch; print(torch.__version__, torch.version.cuda); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'

bash /path/to/box_xvla_training_handoff/run.sh /path/to/outputs/sourccey_box_xvla
```

Use an NVIDIA driver and CUDA-enabled PyTorch build supporting Blackwell;
PyTorch introduced Blackwell wheels with CUDA 12.8. The fork's Linux PyTorch
source is its cu128 index; do not replace it with an older cu121/cu124 wheel.
The launcher tests actual CUDA kernels on each GPU before loading the model.
See [PyTorch's Blackwell announcement](https://pytorch.org/blog/pytorch-2-7/).
TorchCodec and FFmpeg must be compatible with the fork's lockfile even though
this launcher selects PyAV for its initial decoding path.
Install system FFmpeg through the Linux distribution's package manager if it
is absent (for example, `sudo apt install ffmpeg` on Ubuntu/Debian).

Use `--frozen` with this snapshot to install the recorded lock without resolving
all of the fork's unrelated extras/platforms again. Its selected versions are
PyTorch 2.11.0/cu128, torchvision 0.26.0/cu128, Transformers 5.5.4, Accelerate
1.15.0 and Draccus 0.11.6. The local data preflight ran in the existing older
environment (PyTorch 2.9.1, Transformers 5.3.0, Accelerate 1.13.0); the Linux
locked environment and full pretrained training must pass the remote checks.
An offline install-plan check passed here; packages were not installed locally.

The workflow first checks file integrity, the actual trainer/configuration,
episode splits, training-only statistics and a real four-camera batch. It then
runs **20 full pretrained training steps on both GPUs**, including saving a
checkpoint, before starting the main run. The first run needs internet access
for the pinned base weights. The small BART tokenizer is included in `tokenizer/`.
Keep the model cache on local SSD.

The main run stops at **20,000 micro-batches or the remaining time budget**,
whichever comes first. Ten minutes are reserved for the final checkpoint; time
spent in preflight, downloading and smoke training counts toward the eight hours.
This is a soft wall-time limit: an unusually slow update, validation pass or
checkpoint can exceed the reserve. Use the scheduler's outer job limit with
additional allowance if a strict cluster timeout applies.

The launcher requests a final save collectively on all ranks, at a completed
gradient-accumulation boundary, and stops only after checkpoint writes and the
`last` link finish. It keeps the original LR horizon for a later resume.

To run just the preparation check again without loading model weights:

```bash
python /path/to/box_xvla_training_handoff/train.py \
  --output /path/to/outputs/check_only --check-only
```

## Recipe and reasons

| Setting | Initial recipe |
| --- | --- |
| Base weights | `lerobot/xvla-base`, revision `cdb7964e4fe842935d671bfab5a5ebe00a96648c` |
| Model adaptation | Full fine-tuning: vision, language encoder, action transformer and soft prompts |
| Optimizer | Fork's X-VLA AdamW preset; action/head LR `1e-4`, VLM LR `1e-5` |
| Schedule | 1,000-micro-batch warmup, cosine decay through 20,000, final LR `2.5e-6` |
| GPUs | Two processes, DDP; full model replicated on each GPU |
| Precision | FP32 model/optimizer parameters with Accelerate BF16 autocast |
| Batch | 2 per GPU, accumulation 1: effective global batch 4 |
| Input cameras | `front_left`, `front_right`, `wrist_left`, `wrist_right`, in that order |
| Images | RGB 320 x 240; checkpoint's aspect-preserving resize/padding to 224 x 224 and X-VLA ImageNet preprocessing |
| State / action | Six left-arm channels, observed state and upcoming absolute target kept distinct |
| Normalization | Training-only mean/std for state/actions; inverse transform saved for inference |
| Model widths | Keep checkpoint state width 20 and action width 20; `auto` handles six actual channels |
| Timing | Dataset actions at 50 Hz; predict 30 actions (0.6 s), initial execution horizon 10 (0.2 s) |
| Domain / language | Default domain 0, same for deployment; BART tokenizer, length 64 |
| Validation / save | Every 1,000 micro-batches; validation capped at 256 frames; final checkpoint always saved |
| Augmentation | Existing episode domain randomization retained; no extra random crops or horizontal flips |

The official [X-VLA fine-tuning guidance](https://huggingface.co/docs/lerobot/en/xvla)
recommends training all components and a tenfold smaller VLM learning rate. This
fork's optimizer already implements those parameter groups. We keep FP32
parameters and use the fork's explicit AMP runtime instead of literally casting
all trainable parameters to BF16. Four-camera memory use still needs the remote
smoke test; no full pretrained allocation has been tested on this PC.

Two 32 GB GPUs do **not** act like one 64 GB GPU under DDP. If the smoke test
runs out of memory, rerun with a fresh output directory and `BATCH_SIZE=1`:

```bash
BATCH_SIZE=1 bash /path/to/box_xvla_training_handoff/run.sh /path/to/outputs/box_batch1
```

That changes the effective batch to 2. Gradient accumulation can restore the
effective batch without increasing activation memory; e.g. batch 1 x two GPUs x
accumulation 2 = 4. Native `--steps` and its LR scheduler count micro-batches,
so double the micro-batch count to keep the same optimizer-update count when
doubling accumulation. The launcher scales logging/save intervals accordingly.
Keep a consistent recipe within a run. Resume uses saved hyperparameters.

`train.py --shard --gpus 2` is an **opt-in, remotely unverified FSDP2 path**, with
explicit X-VLA, BART and Florence transformer wrap units; it still writes full
portable safetensors checkpoints. Start with its own short smoke test. The fork
exposes activation checkpointing and compilation as unwired placeholders, so
this recipe does not enable them. See the [LeRobot distributed training guide](https://huggingface.co/docs/lerobot/en/multi_gpu_training).

Monitor the trainer's `data_s`, `prep_s`, `updt_s`, `step_s`, `mem_gb` and samples/s.
If decoding dominates, first compare 4 versus 8 workers per process and local SSD
storage. PyAV's Windows random-access performance was poor in the earlier dataset
check. TorchCodec may be faster on Linux, but validate a matching installation
before changing the decoder and preserve the chosen environment/version report.
Do not assume two GPUs halve wall time: model synchronization, CPU decoding and
storage can dominate. An eight-hour duration is a budget, not a success criterion.

## Splits, overfitting and choosing a model

The 100 successful episodes contain 94,600 action frames. Only **50 episodes /
47,300 frames** train the policy. Seventeen episodes from separate location
groups supply validation; the remaining 33 are reserved for final test. The
launcher uses the fork's normal episode splitter, then asserts the actual IDs
equal the explicit split file. Validation is never optimized, test never enters
the training loader, and numeric normalization is checked against training rows.

At global batch 4, 20,000 steps see approximately 80,000 training examples, about
1.69 passes over the 47,300 training rows. Those rows are temporally correlated:
100 demonstrations around six position groups are still a small pilot. More
GPU time does not create broader reach coverage or recovery behavior.

Use held-out loss to compare checkpoints, then evaluate the best candidates in
**closed-loop MuJoCo rollouts with policy actions**, including lift height,
two-second hold, collisions, completion time and success across held-out
positions/rotations and new appearance seeds. Low imitation loss, or playing
back the teacher's saved trajectory, does not establish policy success. The
custom box scene is not yet registered as a stock LeRobot evaluation environment;
this script computes held-out loss and does not claim to run those rollouts.
Keep the final 33 episodes/location groups untouched while selecting settings.

## What to bring back and resume

Bring back the **whole chosen checkpoint directory**, including
`pretrained_model/` (weights, config, both processor definitions, normalization
state and tokenizer artifacts) and `training_state/` (optimizer, scheduler, RNG
and resume metadata). Also retain the sibling `OUTPUT_handoff/` folder, the
training logs and this handoff's calibration references. Keep the smoke output
separate from the actual trained model. A typical final model path is:

```text
OUTPUT/checkpoints/last/pretrained_model/
```

`last` is a relative symlink on Linux; when copying to Windows, copy its target
checkpoint directory or dereference the link. The last model is not necessarily
the best model. Allow disk space for the pretrained download and multiple full
model/optimizer checkpoints; this workflow does not prune old checkpoints.

Resume an interrupted main run from its saved config, using the same GPU count
and batch settings. The checkpoint's settings/optimizer/RNG take precedence:

```bash
python /path/to/box_xvla_training_handoff/train.py \
  --output /path/to/outputs/sourccey_box_xvla --gpus 2 --hours 8 \
  --resume /path/to/outputs/sourccey_box_xvla/checkpoints/last/pretrained_model/train_config.json
```

No automatic Hub upload is enabled. The wrapper changes no files inside the
installed training checkout. Its process-local adapters isolate unrelated eager
policy imports, fetch pinned weights through the current Hub API before the
fork's strict local load, rebuild processors for our new embodiment, assert
splits/statistics, and checkpoint at the time budget. Resume restores saved
processors instead of rebuilding them from the public base model.

## The real-robot contract

The same policy can eventually consume real observations and produce meaningful
motor targets if the surrounding runtime preserves this contract:

1. Assemble six state values **by name**, in this exact order:
   `left_shoulder_pan.pos`, `left_shoulder_lift.pos`, `left_elbow_flex.pos`,
   `left_wrist_flex.pos`, `left_wrist_roll.pos`, `left_gripper.pos`.
   The real plugin exposes 17 state coordinates; do not feed that whole vector
   into this six-channel policy or append dummy channels to its training data.
2. Use the Host's normalized mode (`left_arm_use_degrees=False`). Five arm
   coordinates use [-100,100], gripper [0,100]. These are **absolute position
   targets**, not radians, velocities, increments or binary gripper labels.
   The affine radian/hardware mapping shipped here must remain consistent with
   the robot's actual calibration. A physical recalibration can change that mapping.
3. Match all four camera identities and image rotation/crop/RGB conversion to
   training. The real camera defaults are 320 x 240 at **30 FPS**; the simulation
   has a 50 Hz synchronized observation/action clock. Keep the action timing at
   50 Hz and use fresh timestamped camera frames, holding the most recent image
   between captures. Before hardware transfer, measure actual capture latency
   and frame timing and add corresponding image holds/delay to the next sim
   dataset or real adaptation set. Do not relabel 50 Hz targets as 30 Hz.
4. Apply the checkpoint's **saved preprocessor and postprocessor exactly once**.
   The postprocessor returns six normalized hardware targets. A left-arm adapter
   must explicitly preserve the parked right arm/elevator and command zero base
   velocities. The stock full-robot CLI is not yet a drop-in six-channel adapter.
5. Retain 50 Hz chunk timestamps and replan from observed state/images. Ten
   executed actions gives a nominal 5 Hz policy refresh; benchmark actual
   inference latency first. If inference exceeds 200 ms, tune the execution
   horizon or use properly aligned buffered inference; do not slow the playback
   clock invisibly. Existing buffered trajectory replay alone does not implement
   a learned-policy feedback loop. Keep joint limits, fresh-state monitoring and
   stopping behavior in the eventual deployment runtime.
6. Start from the trained parked base/elevator/right-arm configuration and the
   measured table setup. This model covers upright box pickups in nearby local
   regions; it has no wheel/elevator policy, navigation skill or general task repertoire.

The camera model is still a benchmark framing fit, **not measured physical
intrinsics/extrinsics/distortion**, and the grasp calibration is not physically
verified. Those are material sim-to-real gaps. Next, verify calibration and
collect a modest set of real successful demonstrations with this same task,
channel order, timestamps and camera processing. Fine-tune the sim checkpoint
with those demonstrations and some sim rehearsal; reserve real positions/trials
for evaluation and choose mixing/LR from that validation. A starting hypothesis
is 30-50 real demonstrations and a smaller head LR around `1e-5`, but neither is
a demonstrated requirement or guarantee. Freezing the deployment normalization
to the sim checkpoint simplifies transfer; if you refit statistics, save and
validate the new processors together with the adapted weights.

This run creates the first learned sim pickup policy. Calibration, closed-loop
evaluation, the explicit robot adapter and real-data adaptation remain the path
to reliable real-world use.
