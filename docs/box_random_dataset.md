# Continuous random box pick and place

The earlier pilot trained on 50 demonstrations in three tiny position clusters.
Its 0/17 validation score tested a fourth cluster. That dataset did not cover
continuous random placements throughout the useful arm workspace.

`examples.box_random_dataset` samples continuous positions and upright yaw
angles, executes an unassisted physics teacher, and retains successful full
pick-and-place demonstrations. The task is exactly:

**Pick up the box and place it at the fixed destination**

The destination is 110 mm from the table's near edge and 180 mm left of the
robot. Starting centers must be at least 40 mm away. Final yaw is unconstrained;
the box must be upright. Wheels, elevator, object dimensions and the parked
right arm retain the existing setup.

## Sampling and coverage

The main plan, `examples/box_random_dataset.json`, proposes a 160 x 240 mm
area: depth 80..240 mm, lateral 80..320 mm, yaw -40..+40 degrees. These are
exploration bounds, not a claim that this whole area is reachable.

The plan divides XY into 40 x 60 mm cells and yaw into four 20-degree ranges.
Each resulting position/orientation stratum has a separate sampling stream and
collection quota. Positions and yaw are uniform within their stratum. There is
no anchor jitter or replayed trajectory. The teacher observes the actual settled
object and solves for the new pose.

First probe every stratum. A completed probe freezes proposed collection
strata where the teacher demonstrated a successful full sequence. A zero-success
stratum remains unvalidated; it is not classified as physically unreachable.
The current teacher can fail at reachable poses. More probing or improving the
teacher can expand this support later, using a new run and plan signature.

Collection verifies every new sample again. Rejections are retained with their
scene and reasons. Each selected stratum must meet its own quota in each split;
the collector stops with `coverage_incomplete` if it cannot, instead of quietly
replacing difficult regions with easy ones. `coverage.json` lists all accepted
poses and per-stratum counts; `coverage.png` shows all proposals and outcomes.
Review both before committing to a larger training run.

All episodes retain the existing independent texture, color, PBR reflectance,
lighting, exposure, noise, camera perturbation, mass and friction randomization.

## Physics acceptance

Every accepted episode must pass the original pickup checks: 50 mm bottom
clearance, two-finger contact, a two-second hold, at most 5 mm drift, healthy
physics, stationary base/elevator, and full-path robot/table/self/floor collision
audits. Original grasp alignment and teacher path checks are retained.

The teacher then carries the box, lowers it, opens the gripper gradually, and
retreats. Transport requires both finger contacts and at least 20 mm bottom
clearance. Touchdown is allowed to transfer the load to the table. The final
placement must be within 15 mm of the destination, upright within 12 degrees,
within 3 mm of table height, supported by table contact, released from both
fingers, and stable within 3 mm for one second. No box weld or pose teleportation
is used. Privileged object pose/contact measurements stay outside policy inputs.

## Completed local validation: November 20, 2026

An early broad diagnostic sampled 16 XY proposals across depth 80..240 mm,
lateral 80..320 mm and yaw -40..+40 degrees. Five passed the full pickup,
carry, release and settled-placement checks; three starts were excluded for
being too close to the destination. This is a teacher feasibility probe, not a
model score or a dense workspace map. Its outputs are in
`artifacts/box_random_probe_smoke_02`, including `coverage.png` and a complete
four-camera render of `random_20261120_probe_cell_01_02_00000`.

The final collector uses separate XY/yaw strata. Its deliberately narrow smoke
plan passed two support probes, then **6/6 fresh full sequences**: two training,
two validation and two test episodes. All six have different continuous starting
poses and appearance seeds. The maximum accepted placement error was 11.52 mm.
This checks the pipeline and is not the broad dataset requested for training.

| Output/check | Result |
| --- | --- |
| `artifacts/box_random_smoke_data` | Six complete recorded and rendered demonstrations |
| `artifacts/datasets/box_random_smoke_v3` | LeRobot v3 export; 9,102 nonterminal frames at 50 Hz |
| Four-camera contract | Four synchronized 320 x 240 RGB videos per episode |
| `xvla_compatibility.json` in the dataset | Passed; 72 images, exact task, train-only statistics, three reduced-model gradient steps and six-channel inference |
| `artifacts/box_random_smoke_handoff` | Portable bundle with the new task and fixed-destination contract |
| `artifacts/box_random_smoke_trainer_check.log` | Actual trainer/schema/split/processor preflight passed; test episodes excluded |
| Targeted automated checks | 33 passed |

No full pretrained model was trained or evaluated on the new task. The broader
stratified workspace map and large collection have not been launched. The earlier
20,000-step pickup checkpoint and its 0/17 validation result remain separate.

## Fixed 50-attempt batch and live progress

The November 20 follow-up started a background batch at
`artifacts/box_random_attempts_50_20261120_01`. It uses the broad main plan,
four physics workers, and exactly 50 attempts. A seeded permutation selects
50 of the 64 XY/yaw strata; this budget does not cover every stratum. Starting
poses too close to the destination are resampled before admitting an attempt.
All 50 poses and their 40/5/5 train/validation/test assignments are frozen before
physics. Failed physics attempts count toward 50 and are never replaced.

From Git Bash in this project:

```bash
./box-status.cmd
```

For automatic updates every five seconds (Ctrl+C stops monitoring):

```bash
./box-status.cmd --watch
```

Watch three successful followed by three unsuccessful scripted attempts from
the latest batch, with the recorded motion and four live-rendered camera feeds:

```bash
./box-replays.cmd --successes 3 --failures 3
```

The playlist automatically advances, with Pause, Restart, Previous and Next
controls. Closing the viewer exits the playlist. Selection is random each run;
`--seed 7` repeats a selection, `--speed 0.5` slows playback, and `--list` checks
and lists the six signed recordings without opening the viewer. Replays use
saved physics and do not rerun the teacher or load a learned model. Diagnostic
recording failures without a complete action sequence are excluded.

The command shows completed attempts out of 50, successful/rejected sequences,
in-progress and queued attempts, camera recordings ready, elapsed time, and
the current physics/rendering/export stage. Early physics ETA estimates are
rough. Completed attempts stay at 50 while successful videos render and export.
The monitor checks that the runner process is alive and reports an unexpected
stop or recorded error instead of silently displaying stale progress.

`artifacts/box_attempts_latest.json` points to the latest batch. Use
`--output artifacts/specific_batch` to inspect another, or `--json` for structured
status. Each batch also has `run_state.json`, a live `manifest.json`, coverage
files, and adjacent `.stdout.log` / `.stderr.log` files.

The background runner automatically renders only successful full sequences
with two GPU workers and exports them under `artifacts/datasets/<batch>_v3`.
Success counts in each split depend on actual outcomes; missing held-out
successes must be addressed before training. It does not launch model training.

The first batch's progress updater stopped at 33/50 when the terminal recorder
re-raised an out-of-range normalized joint-target error. The remaining worker
jobs had continued: recovery verified 49 signed result files and one signed
failed replay, restoring **50/50 attempts: 20 successful, 30 rejected** without
rerunning physics. All 297 pre-existing scene/result/recording files were checked
unchanged. The successful splits are 17 train, 1 validation and 2 test. Rendering
and export were resumed in the same output directory; resumed logs use the
`.resume_01.stdout.log` and `.resume_01.stderr.log` suffixes.

Terminal recording validation errors now produce rejected per-attempt results.
Joint limits and target validation remain unchanged. `--resume` verifies the
frozen selection, physics and existing signatures, recovers completed results,
and runs only never-started attempts. It preserves before-recovery manifest/state
snapshots and a recovery history. Unrecognized partial scenes require review;
new physics cannot resume if the teacher source changed. Completed signed
recordings can still be recovered and rendered after an error-handling repair.

To run another fixed budget in the foreground with a fresh output directory:

```bash
./mujoco4.cmd -m examples.box_attempt_batch --attempts 50 --jobs 4 --seed 2026112002 --output artifacts/box_random_attempts_next --render --export
```

## Ridged-gripper friction comparison

The follow-up batch is `artifacts/box_random_attempts_50_ridged_20261120_01`.
It reruns the first batch's exact 50 frozen scenes with twice each episode's
gripper/box sliding-friction coefficient: approximately **0.70..0.80** instead
of **0.35..0.40**. This is an experimental effective-friction approximation for
ridged rubber pads, not measured calibration or a change to the ridge geometry.
Torsional/rolling friction, table friction, grip torque/force limits, poses,
camera/texture/lighting samples, mass, controller and pass criteria are retained.
Existing baseline scenes and their replay signatures are preserved.

The run uses the existing progress/replay commands, which select the latest
batch. It will render and export its successful episodes automatically. After
the physics pass, `comparison.json` and `COMPARISON.md` in the new run directory
show improvements, regressions, unchanged successes, unchanged failures, and
each pair's failed checks. The status command also shows the completed paired
comparison. Baseline success was 20/50. These are teacher outcomes, not model
evaluation results.

To repeat this comparison in a fresh directory:

```bash
./mujoco4.cmd -m examples.box_attempt_batch --attempts 50 --jobs 4 --reference artifacts/box_random_attempts_50_20261120_01 --grip-friction-scale 2 --output artifacts/box_random_ridged_next --render --export
```

Only the new experiment receives the friction override. The original datasets
and physical estimates are retained for comparison. New paired datasets record
the contact experiment in their simulation provenance.

## Small pipeline check

Use the deliberately small smoke plan to check data flow cheaply. Its narrow
region is only a software/physics check and must not become the final dataset.
Use fresh output paths if these already exist:

```powershell
.\mujoco4.cmd -m examples.box_random_dataset --plan examples/box_random_smoke.json --phase probe --attempts-per-cell 1 --jobs 2 --output artifacts/box_random_smoke_support
.\mujoco4.cmd -m examples.box_random_dataset --plan examples/box_random_smoke.json --phase collect --support artifacts/box_random_smoke_support/manifest.json --train-per-cell 1 --validation-per-cell 1 --test-per-cell 1 --jobs 2 --output artifacts/box_random_smoke_data
.\mujoco4.cmd -m examples.box_pilot_render --output artifacts/box_random_smoke_data --jobs 2
```

## Broader collection

The initial larger probe should test several samples in every position/yaw
stratum. Inspect the resulting coverage and teacher acceptance rates first:

```powershell
.\mujoco4.cmd -m examples.box_random_dataset --phase probe --attempts-per-cell 12 --jobs 4 --output artifacts/box_random_workspace
```

After reviewing coverage, collect new independent episodes throughout the
frozen support. The following requests 100 training, 10 validation and 20 test
successes **per accepted position/yaw stratum**. Total episode count depends
on the successful strata; inspect that count and available disk space first.
This larger collection has not been launched by the small-check workflow.

```powershell
.\mujoco4.cmd -m examples.box_random_dataset --phase collect --support artifacts/box_random_workspace/manifest.json --seed 20261121 --jobs 4 --output artifacts/box_random_dataset
.\mujoco4.cmd -m examples.box_pilot_render --output artifacts/box_random_dataset --jobs 2
```

`--resume` accepts only identical plans, teacher/criteria source, physics,
calibration and support signatures. Existing outputs are preserved. Rendering
can follow collection using the existing `box_pilot_render --follow` workflow.

## Splits, export and training

Train, validation and test sample independent full episodes within the **same
coverage strata**. Every continuously sampled XY location has a stable exact
location ID; no exact location crosses splits. Coarse spatial cells are shared
intentionally. This tests new random poses in the learned workspace. It is
different from withholding a whole location region, which can be an additional
extrapolation experiment later. Probe recordings never train the policy.

Export and check using the existing LeRobot environment:

```powershell
$lerobotPython = 'C:/Users/Theor/Documents/WebsiteCode/VulcanLerobot/lerobot-vulcan/.venv/Scripts/python.exe'
& $lerobotPython -m integrations.export_box_dataset --source artifacts/box_random_dataset --output artifacts/datasets/box_random_v3 --repo-id local/sourccey-box-random-pick-place --policy xvla
& $lerobotPython -m integrations.check_box_xvla --dataset artifacts/datasets/box_random_v3
& $lerobotPython -m integrations.prepare_box_xvla_training --dataset artifacts/datasets/box_random_v3 --lerobot-repo C:/Users/Theor/Documents/WebsiteCode/VulcanLerobot/lerobot-vulcan --output artifacts/box_random_training_handoff
```

Use the [existing X-VLA training guide](box_xvla_training.md) for the GPU setup
and portable launcher. Its old pickup-only example task and dataset paths are
historical: use this new bundle and its `handoff.json` task/destination contract.
The launcher and checker accept both tasks. State/action normalization uses
training episodes only. No validation/test episode enters gradient training;
test episodes remain excluded from validation and checkpoint selection.

## Evaluate new poses with the learned policy

After training, evaluate the held-out scenes from the continuous collection.
This does not require exporting their videos: the evaluator renders live
observations. It uses the scene's pick-and-place instruction and runs for up to
36 simulated seconds, stopping only after a valid lift/hold **and** a stable,
released placement. It never calls the teacher or feeds object pose to the model.

```powershell
.\xvla-sim.cmd --checkpoint path/to/new/pretrained_model --sweep validation --collection artifacts/box_random_dataset --output artifacts/box_random_policy_validation
```

To generate a completely fresh validation set after training, run `--phase
collect` with the same frozen support, a new seed, `--train-per-cell 0`,
`--validation-per-cell 10`, and `--test-per-cell 0`, then evaluate that collection.
Only teacher qualification may reject candidates before evaluation. Every
selected policy failure counts; policy outcomes never select replacement poses.
This score is conditional on teacher-qualified support, not the whole table.

Freeze the model, execution horizon and other settings before explicitly opening
the reserved final test:

```powershell
.\xvla-sim.cmd --checkpoint path/to/new/pretrained_model --sweep test --final-test --collection artifacts/box_random_dataset --output artifacts/box_random_policy_test
```

Inspect results by position/yaw stratum as well as the overall fraction. One
model seed per pose and a small test set do not establish broad robustness.
Neither a successful teacher nor a low training loss establishes model success.
