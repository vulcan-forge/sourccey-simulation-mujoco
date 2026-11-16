# MuJoCo 4 migration

This branch targets **MuJoCo 4.0.1**, Python 3.12. The original `.venv` has
been retained as the local 3.14 baseline. Generated models, recordings, images,
videos, virtual environments and fitted models belong in ignored directories.

## Install and run

```powershell
.\setup-mujoco4.cmd
# Compute A with procedural PBR materials, then open Studio with four cameras.
.\mujoco4.cmd -m examples.box_pickup --pbr --output artifacts/mujoco4/box_a --studio
# Review without recomputing physics.
.\mujoco4.cmd -m examples.studio_review --output artifacts/mujoco4/box_a
# Classic viewer and classic camera rendering for comparison.
.\mujoco4.cmd -m examples.box_pickup --pbr --replay --view --camera-feeds --renderer classic --output artifacts/mujoco4/box_a
# Generate ten seeded B instances. Choose a fresh output directory for every batch.
.\mujoco4.cmd -m examples.box_training --count 10 --pbr --jobs 2 --output artifacts/mujoco4/box_b_video
# Review one B instance in Studio.
.\mujoco4.cmd -m examples.studio_review --output artifacts/mujoco4/box_b_video/episode_000
```

`mujoco4.cmd` selects `.venv-mujoco4` and Filament camera rendering. Normal
Python invocations default to the classic camera renderer; `--renderer` or
`SOURCCEY_RENDERER` selects explicitly. `--view` retains the classic interactive
viewer. `--studio` selects the new Studio review. Studio includes pause, restart,
seek, speed, a dockable four-camera panel, and frustum overlays. It plays saved
states, not a second physics simulation. Closing it releases its renderers.
`--hot-reload` enables Studio plugin-code reloading.

The Windows Studio launcher uses native OpenGL. A local, version-gated plugin
supplies the native swap-chain handle and missing direct-window target path in
4.0.1 (`Target not found:` upstream); the installed MuJoCo package is untouched.
`--gfx opengl_headless` is available for invisible smoke tests. `--gfx web` streams
to a local browser instead. The existing robot executable is not rebuilt by this migration.

The package split in 4 requires uninstalling 3.x before upgrading an existing
environment. `setup.cmd` and `setup.ps1` handle that; use `setup-mujoco4.cmd` here
to preserve the original environment. To reproduce the earlier baseline on a
different machine, install `requirements-mujoco3.txt` in a separate environment.

## Physics and camera compatibility

The 2 ms timestep, existing collision decomposition, bearing clearances,
measured dimensions, camera transforms and intrinsics remain the baseline.
Equality-constraint data is sized using `mjNEQDATA` rather than assuming 11
entries. The new version requires 12.

Speculative contacts are optional: add `--speculative` to A. This is recorded
in the scene configuration and invalidates older physics caches. It does not
replace the collision meshes and does not increase the timestep. Standard
discrete contacts remain the default until each task has been compared.

Filament renders camera views from the same `MjData`. In 4.0.1 only one pixel
read request is supported per frame, so four cameras use four viewports in one
horizontal atlas, one GPU submission and one readback wait. The adapter splits
the atlas into independent top-down RGB images. Collision geometry and debug
overlays are excluded from training images. Camera resolution stays 320 x 240.
Physics and visual-only display models keep identical state layouts.

`--pbr` adds seeded cardboard, wood, painted shell, gripper and metal surfaces.
B also varies roughness alongside existing colors, lights and sensor noise.
These are procedural appearance estimates, not measured textures or optics.
Light intensities remain in classic-compatible mode so Filament supplies its
fallback environment illumination. A future photometric/HDR setup must specify
both the environment light and all direct light intensities together.

Episode metadata and LeRobot export provenance now include engine version and
renderer. Existing datasets are preserved. Old physics recordings are rejected
when engine, model, scene or relevant source hashes differ; regenerate them in
a new directory. Never relabel old videos as Filament output.

## Parallel generation

`--jobs 1..4` runs independent physics workers with separate models and data.
Two workers are the recommended start because collision models consume substantial
memory. Rendering and video encoding remain in the parent process. This preserves
the teacher's Python feedback and gravity compensation; it does not replace them
with raw native batch stepping. `--no-video` runs physics diagnostics and overview
renders but does not produce an image training dataset. ffmpeg is required for
the full four-camera videos.

## Fit physics against real measurements

Install `requirements-sysid.txt` (included in the separate environment setup).
The offline fitter uses MuJoCo's `sysid.ParameterDict` and bounded optimizer with
the project's split-step arm feed-forward when explicitly requested. It supports
joint damping, actuator P/D gains, body masses (scaling inertia consistently),
and sliding friction including explicit contact pairs. It never connects to the
robot, commands motion, or replaces the active model.

First check existing replay telemetry:

```powershell
.\mujoco4.cmd -m integrations.identify_physics --audit-trace artifacts/robot_ball_replay.jsonl
```

The archived replay trace has roughly 134 ms between status samples and lacks
precise observation-acquisition and applied-command timestamps. It is unsuitable
for fitting fast servo dynamics. No real-robot calibration is claimed from it.

Prepare **two different recordings** (training and held-out validation) of the
same support arrangement, model and controller. Each NPZ uses `allow_pickle=False`
and contains:

| Array | Shape | Meaning |
| --- | --- | --- |
| `time` | N | Increasing seconds from zero, aligned with the model's physics clock |
| `qpos` | N x nq | Measured positions, radians/meters, in exact model order |
| `qvel` | N x nv | Measured/estimated velocities; initial row initializes the rollout |
| `ctrl` | N x nu | Actual applied position targets, radians/meters in actuator order |
| `observed_joint_names` | J Unicode strings | Scalar hinge/slide joints to score |

Samples observe state before applying that row's command. Commands use zero-order
hold. Use acquisition timestamps and the full applied command history when
aligning logs; interpolate state only where justified. Do not reinterpret the
low-rate replay status trace as full-rate measurements. For the 500 Hz model,
100 Hz or 250 Hz aligned samples are possible; validate excitation, measurement
quality, support constraints and omitted disturbances first. A fixed-base
identification model must represent the actual support arrangement. Equivalent
mecanum traction is not included in this arm fitter.

Every new box run saves `model.mjb`, an engine-specific snapshot of its scene.
`examples/physics_fit.example.json` illustrates bounded parameters; its numbers
are starting assumptions. Set `gravity_feedforward` to match the controller used
in the measurements. Standard robot servo control should not be assumed to have
the simulation's idealized gravity feed-forward. Force caps, servo delays and
unmeasured contact conditions can invalidate a simple P/D fit.

```powershell
.\mujoco4.cmd -m integrations.identify_physics --model artifacts/sysid/model.mjb --train artifacts/sysid/train.npz --validation artifacts/sysid/validation.npz --config examples/physics_fit.example.json --output artifacts/sysid/fit_001
```

Outputs include `report.json`, `validation.npz`, an offline `validation.html`
comparison, and `candidate.mjb`. Validation requires optimizer success, full
Jacobian rank and improved held-out error without a material per-joint regression.
The model is still a candidate. Check additional tasks before explicitly adopting
it; parameters are not automatically applied to the robot or simulator defaults.
Delay and bias fitting, cloth tasks and GPU policy training remain separate work.

## Checks

```powershell
.\mujoco4.cmd -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover both renderers, image orientation, collision visibility, seeded PBR
appearance without geometry/camera changes, the existing contact pickup and
negative controls, and recovery of a known damping value using a separate
validation recording. The latter is a synthetic workflow check, not real-robot
calibration.

Verified locally on Windows / RTX 2070 SUPER: 259 tests passed on 4.0.1;
256 passed and three 4-only checks skipped on 3.14. A and all ten seeded B
physics trials passed, with at least 69 mm hold clearance across B. A four-camera
Filament encoding check produced 50 frames per camera at 50 Hz, 320 x 240, and
native Studio was visually inspected and closed cleanly. The ten-instance
validation batch uses `--no-video`; it is not a complete image training dataset.
Local evidence is in `artifacts/mujoco4/validation_summary.json` and test logs.
The optional speculative-contact A run also passed with the same hold metrics;
it was considerably slower on this detailed convex decomposition, so it remains
disabled by default.

Official references: [4.0.1 release](https://github.com/google-deepmind/mujoco/releases/tag/4.0.1),
[Studio](https://mujoco.readthedocs.io/en/stable/studio/index.html),
[rendering](https://mujoco.readthedocs.io/en/stable/programming/rendering.html),
[system identification](https://github.com/google-deepmind/mujoco/tree/main/python/mujoco/sysid).
