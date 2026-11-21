# Evaluate the learned X-VLA box policy

This runs `xvla_sourccey_box_sim_b4/checkpoints/020000/pretrained_model` in
the original randomized, full-contact box-pickup environment. The model drives
the left arm using the instruction **Pick up the box**. It receives four live
320 x 240 RGB camera images and six observed joint positions; no teacher,
inverse kinematics, object-pose input, grasp attachment or trajectory fallback
controls the robot. The current kitchen appearance is a separate environment;
first establish performance in the training domain.

## Watch an already evaluated attempt

To watch the saved validation sweep in shuffled order:

```bash
./xvla-replays.cmd
```

Each attempt plays once, with four onboard cameras, its box pose, pickup
outcome and failed checks. After a two-second final-pose hold it advances.
Previous/Next buttons skip between attempts; Pause, Restart, playback speed
and Time allow detailed inspection. Pause also stops automatic advancement.
Closing the viewer or Quit playlist ends the entire program. Scene transitions
close the old viewer and open the next, releasing its rendering resources.

Use `--count 5 --speed 0.5` for five slower random attempts, `--seed 7` to
repeat the shuffle, `--hold 5` for a longer final hold, or `--loop` to repeat
the selected playlist. `--sweep artifacts/another_sweep` selects another
saved evaluation sweep. `--list` prints the order without opening a viewer.
This reads only saved learned-policy recordings; no checkpoint is loaded and
no physics rollout or inference is recomputed. It does not mix in teacher
demonstrations or the successful training-placement trial.

From Git Bash in this project:

```bash
./xvla-sim.cmd --replay --output artifacts/box_policy_eval_train_h30_20k
```

The initial 10-action-horizon training/validation attempts are also preserved:

```bash
./xvla-sim.cmd --replay --output artifacts/box_policy_eval_train_20k
./xvla-sim.cmd --replay --output artifacts/box_policy_eval_validation_20k
```

PowerShell uses `.\xvla-sim.cmd`. Replay does not load the neural network or
recompute actions. It displays the saved **actual policy-driven physics** with
four camera feeds, a pause button, a seek bar, playback speed and restart.
Close the viewer when finished. `--view-seconds 30` auto-closes it.

The 30-action-horizon validation attempt can be compared with the successful
training placement:

```bash
./xvla-sim.cmd --replay --output artifacts/box_policy_eval_validation_h30_20k
```

## Initial results

All four trials used the 20,000-step checkpoint, policy seed 0 and a maximum
24 seconds of simulation. Success stopped the trial after the measured hold.

| Source placement | Executed actions per prediction | Lift and hold outcome |
| --- | --- | --- |
| Training: location_01_01_attempt_0000 | 10 | Failed; no lift above the tabletop |
| Validation: location_02_00_attempt_0000 | 10 | Failed; no lift above the tabletop |
| Training: location_01_01_attempt_0000 | 30 | Passed: maximum bottom clearance 72.2 mm, two-second hold, minimum clearance during hold 67.7 mm |
| Validation: location_02_00_attempt_0000 | 30 | Failed; no lift above the tabletop |

All four trials passed the inference, physics, collision and fixed-base/elevator
checks. The successful hold had 4.89 mm drift and contact on both fingers.
These are one trial per placement/setting, not an estimated general success
rate. The default horizon is now 30 because it produced the first successful
integration result; the validation miss is retained. No final test split was
used and no checkpoint was retrained or modified.

## Run a fresh closed-loop evaluation

Use a new output directory each time:

```bash
./xvla-sim.cmd --output artifacts/my_xvla_attempt --view
```

The default is a 24-second trial on the known training scene
`location_01_01_attempt_0000`, seed 0. That is an integration check, not a
held-out success score. For a validation scene:

```bash
./xvla-sim.cmd --episode location_02_00_attempt_0000 --output artifacts/my_xvla_validation --view
```

The final test split is not used automatically. `--scene path/to/scene.json`
can select another expanded pilot scene or a normal box-pickup configuration.
`--checkpoint path/to/pretrained_model` selects another complete checkpoint;
weights, tokenizer and both saved processor pipelines must stay together.

The default execution horizon is **30 predicted actions**, at the original
**50 Hz** action clock, followed by fresh camera/state feedback after 0.6 s.
The initial 10-action (0.2 s) refresh missed the grasp in both tested scenes;
executing the full chunk succeeded on the training placement. `--horizon 10`
reproduces the shorter-horizon baseline. This is a runtime setting; the saved
checkpoint and its original `n_action_steps=10` configuration stay unchanged.
All executed targets are absolute normalized hardware
positions, converted with the original affine mapping and clipped to hardware
and simulator joint limits. Clipping counts are reported. The fixed base,
elevator and parked right arm remain at the trained configuration.

## Runtime and measurements

The MuJoCo 4 environment runs simulation/rendering. A persistent subprocess
uses the existing `VulcanLerobot/lerobot-vulcan/.venv` environment for PyTorch,
with the training handoff's source snapshot. No installed packages, checkpoint
files or real-robot Host settings are changed. Communication is local stdin
and an observation NPZ; this runner has no hardware connection.

Override `--policy-python` and `--lerobot-source` when those environments move
or when running on another computer. The model loads strictly from local
safetensors and uses its saved processors/statistics exactly once. The Linux
tokenizer path in its configuration is redirected in memory to the included
local tokenizer. The task and domain ID (0) match training.

Evaluation is **offline synchronous closed-loop simulation**. Physics waits
while inference runs, then integrates every action at its original 20 ms
interval. Saved playback runs smoothly at the recorded simulation speed; that
does not imply real-time inference performance. Load time and per-prediction
latency are reported. Initial testing on the RTX 2070 SUPER measured roughly
0.58 seconds per prediction after warm-up. Real-time deployment still needs a
separate latency-aware controller; replay speed does not establish it.

Success requires the box bottom to clear the table by at least 50 mm, both
fingers to maintain at least 0.1 N contact, and a two-second hold with at most
5 mm drift. Robot/table contact, self contact, unintended floor contact,
base/elevator drift, inference errors and MuJoCo warnings are checked. Object
pose and forces are evaluator-only measurements. Stop occurs on a qualified
hold or the trial time limit.

Each output contains:

- `report.json`: outcome, lift/hold/collision measurements, clipping, timing,
  checkpoint, scene split and source provenance.
- `recording.npz`: the actual simulated motion for viewing again.
- `actions.npz`: observed state and raw/limited policy targets at 50 Hz.
- `scene.json`, `final.png`: exact scene configuration and final view.
- `inference.log`, `latest_observation.npz`: runtime diagnostics and the last
  four-camera/state input.

These evaluation files are kept apart from teacher demonstrations and are not
automatically exported as successful LeRobot training episodes. A low training
loss does not establish that a closed-loop rollout can grasp and lift.

## Sweep the saved positions and rotations

The default single-scene command always starts from the same saved training
placement. The box starts stationary in each attempt; it is not moved to a new
pose partway through a rollout. Training data contains three location groups
(50 episodes). Validation contains one different location group (17 episodes),
and the reserved final test contains two more groups (33 episodes).

Actual training/validation cluster centers are approximately:

| Split / location | Episodes | From near edge | Left of robot |
| --- | --- | --- | --- |
| Train / location_01_01 | 17 | 110 mm | 180 mm |
| Train / location_01_02 | 17 | 110 mm | 220 mm |
| Train / location_02_02 | 16 | 140 mm | 220 mm |
| Validation / location_02_00 | 17 | 140 mm | 140 mm |

Each has roughly +/-3 mm position jitter and varied yaw. These are small
near-arm clusters, not broad coverage of the 1828.8 x 908.05 mm tabletop.

Run all 17 validation scenes with one model load:

```bash
./xvla-sim.cmd --sweep validation --output artifacts/my_validation_sweep
```

These validation scenes span yaw -20 to +20 degrees, near-edge distance
137.46–142.61 mm and left-of-robot offset 137.02–142.59 mm, with the saved
appearance, lighting, camera and physical variations. This is one location
group with small position jitter, not 17 independent table locations.
`--sweep train` checks the 50 saved training scenes across their three location
groups. Neither sweep selects final-test scenes. `--limit 3` is a quick smoke
run using the first three scenes in dataset order, not a representative sample.

Each episode resets model/processor state and the policy random seed (default
0), while retaining loaded weights. Physics, renderer and lift/contact metrics
are rebuilt for each scene. H30, the 50 Hz action clock, observation contract
and success criteria match single-scene evaluation. A missed pickup proceeds
to the next scene; an inference/runtime error stops the sweep and marks it
incomplete. Existing output directories are never overwritten.

The output includes `manifest.json` (frozen selection, exact box poses and
source hashes), `summary.json`, `SUMMARY.md` (results and replay commands),
`contact_sheet.jpg`, the shared `inference.log`, and one complete recording
directory per episode. Summary files update after each completed attempt.
The shared log is in the sweep root; per-episode observation files remain
inside their own attempt directories.

Replay an individual attempt without loading the model:

```bash
./xvla-sim.cmd --replay --output artifacts/my_validation_sweep/location_02_00_attempt_0000
```

Sweeps do not open viewers automatically. Use the replay commands from
`SUMMARY.md` to inspect the cases you want. The finite scene pass fraction is
reported separately from incomplete attempts; it does not establish broad
generalization or real-robot performance.

### Completed validation sweep: 20,000-step checkpoint

`artifacts/box_policy_sweep_validation_20k_h30` contains the complete 17-scene
validation sweep using H30, seed 0 and 24 simulated seconds per scene.
**0/17 passed**; maximum bottom clearance was 0 mm in every attempt.
Every inference, physics, collision and fixed-base/elevator check passed.
All 17 saved recordings passed scene/signature and finite-state verification.
The first scene reproduced the earlier standalone validation rollout exactly,
including every action and observed state. The 21 related automated checks
also passed. One model load served the sweep; its worker closed afterward.

This checkpoint has demonstrated a successful lift on one training placement,
but has not demonstrated pickup generalization to this held-out location.
The 17 variations share that location group, so this is not a score across 17
distinct table locations. The final test split remains reserved.

See the sweep's `SUMMARY.md` for every replay command, `contact_sheet.jpg`
for final views, and `placement_map.png` for the actual training/validation
location coverage. The map is a dataset inspection artifact, not model input.
