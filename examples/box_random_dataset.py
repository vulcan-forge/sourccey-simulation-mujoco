"""Probe continuous box poses, then collect stratified pick-and-place demonstrations.

Each candidate is checked in full contact physics. Probe results freeze the
proposed collection cells; they do not certify every pose inside a cell.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from examples.box_dataset_pilot import (candidate as pilot_candidate, file_hash,
    pilot_signature, apply_camera_perturbations)
from examples.box_pickup import load_scene, layout, save_run, scene_model
from examples.box_training import CAMERAS, FPS, STATE_NAMES, EpisodeRecorder
from examples.box_workspace import WorkspacePickup, assess, read_plan, write_json
from sourccey.box_task import PLACE_TASK, PlacementTracker, placement_observation
from sourccey.build import ROOT
from sourccey.rendering import provenance


DEFAULT_PLAN = ROOT/'examples/box_random_dataset.json'
SPLITS = ('train', 'validation', 'test')


def teacher_hash():
    return hashlib.sha256((file_hash(__file__)+file_hash(ROOT/'sourccey/box_task.py')).encode()).hexdigest()


def read_sampling_plan(path=DEFAULT_PLAN):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    if plan.get('version') != 1:
        raise ValueError('Require sampling plan version 1')
    for key in ('depth_edges_mm', 'lateral_edges_mm'):
        edges = plan[key]
        if (not 2 <= len(edges) <= 20 or not np.isfinite(edges).all()
                or any(b <= a for a, b in zip(edges, edges[1:]))):
            raise ValueError('Sampling edges must be finite and strictly increasing')
    yaw = plan['yaw_range_deg']
    if len(yaw) != 2 or not np.isfinite(yaw).all() or not -45 <= yaw[0] < yaw[1] <= 45:
        raise ValueError('Yaw range must increase within +/-45 degrees')
    if not isinstance(plan.get('yaw_bins', 1), int) or not 1 <= plan.get('yaw_bins', 1) <= 16:
        raise ValueError('Require 1..16 yaw coverage bins')
    p = plan['placement']
    for key, value in p.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'Placement value must be positive and finite: {key}')
    if p['maximum_tilt_deg'] >= 90:
        raise ValueError('Placement must finish upright')
    if not np.isfinite(plan['minimum_transport_mm']) or plan['minimum_transport_mm'] < 0:
        raise ValueError('Minimum transport distance must be finite and nonnegative')
    if not np.isfinite(plan['minimum_carry_clearance_mm']) or plan['minimum_carry_clearance_mm'] <= 0:
        raise ValueError('Require positive carry clearance')
    # Check the destination footprint, independently of any sampled start pose.
    base = load_scene()
    for angle in yaw:
        destination = copy.deepcopy(base)
        destination['box_pose'].update(center_from_near_edge_mm=p['center_from_near_edge_mm'],
            center_left_of_robot_mm=p['center_left_of_robot_mm'], yaw_deg=angle)
        layout(destination)
    return plan


def cells(plan):
    angles = np.linspace(*plan['yaw_range_deg'], plan.get('yaw_bins', 1)+1)
    return [dict(cell_id=f'cell_{i:02d}_{j:02d}_yaw_{k:02d}',
                 index=(i*(len(plan['lateral_edges_mm'])-1)+j)*(len(angles)-1)+k,
                 depth_mm=[a,b], lateral_mm=[c,d], yaw_deg=[float(e),float(f)])
            for i, (a,b) in enumerate(zip(plan['depth_edges_mm'], plan['depth_edges_mm'][1:]))
            for j, (c,d) in enumerate(zip(plan['lateral_edges_mm'], plan['lateral_edges_mm'][1:]))
            for k, (e,f) in enumerate(zip(angles, angles[1:]))]


def random_candidate(plan, cell, split, attempt, seed, probe=False):
    """Uniform XYZ-table pose sampling, independent streams for every split/cell.

    location_id identifies an exact sampled XY location. cell_id is the coarse
    coverage stratum and is intentionally shared across train/validation/test.
    """
    stream = [int(seed), 3101 if probe else 3102, cell['index'], SPLITS.index(split), attempt]
    rng = np.random.default_rng(np.random.SeedSequence(stream))
    base = load_scene()
    base.update(test='C', episode_id='random_template', workspace=dict(
        location_id='template', stage=0, thresholds=read_plan()['thresholds'],
        purpose='Continuous random pose pick and place'))
    appearance_seed = int(rng.integers(0, 2**31))
    group = dict(location_id='template', split=split, anchors=[base])
    config = pilot_candidate(group, cell['index'], attempt, appearance_seed)
    pose = config['box_pose']
    # Sample the whole cell, never a small offset from a successful anchor.
    pose.update(center_from_near_edge_mm=float(rng.uniform(*cell['depth_mm'])),
        center_left_of_robot_mm=float(rng.uniform(*cell['lateral_mm'])),
        yaw_deg=float(rng.uniform(*cell['yaw_deg'])))
    location = hashlib.sha256(json.dumps([pose['center_from_near_edge_mm'],
        pose['center_left_of_robot_mm']]).encode()).hexdigest()[:20]
    episode = f'random_{seed}_{"probe" if probe else split}_{cell["cell_id"]}_{attempt:05d}'
    config.update(episode_id=episode, placement=copy.deepcopy(plan['placement']))
    config['workspace']['location_id'] = 'pose_'+location
    config['dataset_pilot'].update(instruction=PLACE_TASK, location_group='pose_'+location,
        anchor_id=None, position_jitter_mm=None, cell_id=cell['cell_id'])
    config['random_collection'] = dict(format='continuous-table-pose-v1', cell=cell,
        sampling_seed=seed, probe=probe, teacher_sha256=teacher_hash(),
        minimum_transport_mm=plan['minimum_transport_mm'],
        minimum_carry_clearance_mm=plan['minimum_carry_clearance_mm'],
        pose_distribution='Uniform within XY cell and yaw range; teacher rejection recorded',
        split_strategy='Independent full episodes across the same spatial cells; exact XY poses never cross splits')
    return config


class PickAndPlace(WorkspacePickup):
    def __init__(self, config, model=None, measured=None):
        super().__init__(config, model, measured)
        self.placement_tracker = PlacementTracker(config['placement'],
            self.layout['near_edge_x_m'], self.layout['tabletop_z_m'])
        self.pickup_accepted = False
        self.transport_contact_lost = False
        self.transport_clearance_lost = False

    def observe(self):
        super().observe()
        if self.phase == 'transport':
            self.transport_contact_lost |= min(self.fingers().values()) < self.config['success']['minimum_finger_normal_force_n']
            self.transport_clearance_lost |= self.bottom()-self.layout['tabletop_z_m'] < self.config['random_collection']['minimum_carry_clearance_mm']/1000
        if self.phase == 'placement-settle':
            self.placement_tracker.observe(float(self.d.time),
                **placement_observation(self), lifted=self.pickup_accepted)

    def run(self, on_step=None):
        pickup_report = super().run(on_step)
        pickup_assessment = assess(self, pickup_report)
        self.pickup_accepted = pickup_assessment['accepted']
        if self.pickup_accepted:
            goal = self.placement_tracker.goal
            self.phase = 'transport'
            target = self.center_reference(self.d).copy()
            target[:2] += goal-self.d.xpos[self.box, :2]
            self.cartesian_move(target, 3.)
            self.phase = 'set-down'
            target = self.center_reference(self.d).copy()
            target[2] += self.layout['tabletop_z_m']-self.bottom()+.001
            self.cartesian_move(target, 2.)
            # Correct actual box error after the approximate five-axis transit.
            # Touch down before opening so the free box is supported by the table.
            self.phase = 'place-align'
            for _ in range(3):
                correction = np.r_[goal-self.d.xpos[self.box, :2],
                    self.layout['tabletop_z_m']-self.bottom()+.0005]
                if np.linalg.norm(correction) < .001:
                    break
                self.cartesian_move(self.center_reference(self.d)+correction, .6)
            self.phase = 'release'
            start = float(self.d.qpos[self.m.joint('left_gripper').qposadr[0]])
            opened = np.deg2rad(self.config['grasp']['open_angle_deg'])
            for fraction in np.linspace(0, 1, 61)[1:]:
                self.sim.set_joints({'left_gripper': start+(opened-start)*fraction})
                self.advance(.02)
            self.advance(.4)
            self.phase = 'retreat'
            self.cartesian_move(self.center_reference(self.d)-self.grasp_normal*.035, 1.)
            self.cartesian_move(self.center_reference(self.d)+[0, 0, .08], 1.5)
            self.phase = 'placement-settle'
            self.advance(self.config['placement']['settle_seconds']+.1)
        report = self.report()
        final_assessment = assess(self, report)
        # Recheck the full path, including transport/release/retreat collisions.
        report['pickup_passed'] = self.pickup_accepted
        report['workspace_validation'] = final_assessment
        report['placement'] = dict(passed=self.placement_tracker.stable_placement is not None,
            stable_placement=self.placement_tracker.stable_placement,
            final=self.placement_tracker.last, destination=self.config['placement'])
        report['checks'].update(full_path_valid=final_assessment['accepted'],
            maintained_grasp_during_transport=not self.transport_contact_lost,
            carried_above_table=not self.transport_clearance_lost,
            released_and_placed=report['placement']['passed'])
        report['passed'] = all(report['checks'].values())
        final = (float(self.d.time), self.d.qpos.copy(), 'complete')
        if self.frames[-1][0] == final[0]:
            self.frames[-1] = final
        else:
            self.frames.append(final)
        return report


def candidate_summary(config):
    return dict(episode_id=config['episode_id'], split=config['split'], seed=config['seed'],
        location_id=config['workspace']['location_id'], cell_id=config['dataset_pilot']['cell_id'],
        attempt=config['dataset_pilot']['attempt'], box_pose=config['box_pose'], yaw_deg=config['box_pose']['yaw_deg'],
        videos=False, pilot_signature=pilot_signature(config))


def record_candidate(config, directory):
    directory = Path(directory)
    directory.mkdir()
    write_json(directory/'scene.json', config)
    pose = config['box_pose']
    summary = candidate_summary(config)
    try:
        distance = np.linalg.norm(np.array([pose['center_from_near_edge_mm'],
            pose['center_left_of_robot_mm']])-np.array([config['placement']['center_from_near_edge_mm'],
            config['placement']['center_left_of_robot_mm']]))
        if distance < config['random_collection']['minimum_transport_mm']:
            raise ValueError('Start pose is too close to the fixed destination')
        model, measured = scene_model(config)
        apply_camera_perturbations(model, config)
        run = PickAndPlace(config, model, measured)
        recorder = EpisodeRecorder(run)
        run.before_step = recorder.capture
    except (ValueError, RuntimeError) as exc:
        summary.update(passed=False, eligible_for_imitation=False,
            error=str(exc), failed_checks=['construction_or_start_pose'])
        write_json(directory/'result.json', summary)
        return summary
    try:
        report = run.run(run.audit_contacts)
    except (ValueError, RuntimeError) as exc:
        report = run.report()
        report.update(passed=False, error=str(exc))
        report['checks']['teacher_runtime_healthy'] = False
    save_run(run, report, directory, export_model=False)
    try:
        _, metadata = recorder.finish(report, directory)
    except (ValueError, RuntimeError) as exc:
        # The last commanded target can still be invalid after capture aborts
        # the teacher. A terminal recording failure rejects this episode; it
        # must not escape and stop unrelated attempts or enter training data.
        report.update(passed=False, error=str(exc))
        report['checks']['training_recording_valid'] = False
        save_run(run, report, directory, export_model=False)
        summary.update(passed=False, eligible_for_imitation=False, frames=0,
            pickup_passed=report.get('pickup_passed', False),
            failed_checks=[key for key, value in report['checks'].items() if not value],
            error=str(exc), failure_stage='recording', recording_available=True,
            training_recording_available=False, partial_frames_recorded=len(recorder.rows))
        write_json(directory/'result.json', summary)
        return summary
    report['checks']['training_recording_valid'] = True
    save_run(run, report, directory, export_model=False)
    metadata.update(instruction=PLACE_TASK, pilot_signature=summary['pilot_signature'])
    write_json(directory/'episode.json', metadata)
    summary.update(passed=bool(report['passed']), eligible_for_imitation=bool(report['passed']),
        pickup_passed=report.get('pickup_passed', False), frames=metadata['frames'],
        failed_checks=[key for key, value in report['checks'].items() if not value],
        error=report.get('error'), placement=report.get('placement'))
    write_json(directory/'result.json', summary)
    return summary


def coverage(manifest):
    rows = manifest['episodes']
    accepted = [r for r in rows if r['eligible_for_imitation']]
    return dict(attempted=len(rows), accepted=len(accepted),
        split_episodes={s:sum(r['split'] == s for r in accepted) for s in SPLITS},
        cells={c['cell_id']:dict(attempted=sum(r['cell_id'] == c['cell_id'] for r in rows),
            pickup_passed=sum(r.get('pickup_passed', False) for r in rows if r['cell_id'] == c['cell_id']),
            accepted=sum(r['eligible_for_imitation'] for r in rows if r['cell_id'] == c['cell_id']))
            for c in manifest['plan']['cells']},
        accepted_poses=[dict(episode_id=r['episode_id'], split=r['split'],
            cell_id=r['cell_id'], **r['box_pose']) for r in accepted],
        scope='Teacher-accepted poses from the recorded proposal cells; no entire cell is certified reachable')


def coverage_image(output, manifest):
    from PIL import Image, ImageDraw
    plan = manifest['plan']['sampling']
    image = Image.new('RGB', (1000, 740), '#171d27')
    draw = ImageDraw.Draw(image)
    x0, x1 = plan['lateral_edges_mm'][0], plan['lateral_edges_mm'][-1]
    y0, y1 = plan['depth_edges_mm'][0], plan['depth_edges_mm'][-1]
    def point(depth, lateral):
        return (80+820*(lateral-x0)/(x1-x0), 620-520*(depth-y0)/(y1-y0))
    draw.text((30, 20), 'Continuous random box starts | measured teacher pick-and-place outcomes', fill='white')
    draw.text((30, 42), 'Green: accepted   Orange: rejected   Blue: fixed destination', fill='#a9bdd9')
    for lateral in plan['lateral_edges_mm']:
        a, b = point(y0, lateral), point(y1, lateral)
        draw.line([a, b], fill='#425269')
        draw.text((a[0]-10, 635), f'{lateral:g}', fill='white')
    for depth in plan['depth_edges_mm']:
        a, b = point(depth, x0), point(depth, x1)
        draw.line([a, b], fill='#425269')
        draw.text((35, a[1]-5), f'{depth:g}', fill='white')
    for row in manifest['episodes']:
        p = row['box_pose']
        x, y = point(p['center_from_near_edge_mm'], p['center_left_of_robot_mm'])
        color = '#79e3a2' if row['eligible_for_imitation'] else '#ff9e70'
        draw.ellipse((x-4, y-4, x+4, y+4), fill=color)
        angle = np.deg2rad(p['yaw_deg'])
        draw.line([(x, y), (x+12*np.sin(angle), y-12*np.cos(angle))], fill=color, width=2)
    p = plan['placement']
    x, y = point(p['center_from_near_edge_mm'], p['center_left_of_robot_mm'])
    draw.line([(x-9, y), (x+9, y)], fill='#70b5ff', width=3)
    draw.line([(x, y-9), (x, y+9)], fill='#70b5ff', width=3)
    draw.text((300, 665), 'Box center left of robot (mm)', fill='white')
    draw.text((30, 695), 'Vertical axis: box center from near table edge (mm). Lines show initial yaw; untested space is unvalidated.', fill='#a9bdd9')
    image.save(Path(output)/'coverage.png')


def run_collection(plan, output, *, seed=20261120, jobs=2, probe=False,
                   attempts_per_cell=4, quotas=None, support=None, resume=False):
    if not 1 <= jobs <= 4 or not isinstance(seed, int) or seed < 0:
        raise ValueError('Require 1..4 workers and a nonnegative integer seed')
    if attempts_per_cell < 1:
        raise ValueError('Require a positive probe attempt count')
    quotas = quotas or dict(train=100, validation=10, test=20)
    if set(quotas) != set(SPLITS) or any(not isinstance(v, int) or v < 0 for v in quotas.values()) or not sum(quotas.values()):
        raise ValueError('Require nonnegative train/validation/test quotas and at least one success')
    selected = cells(plan)
    support_hash = None
    if not probe:
        if support is None:
            raise ValueError('First probe the workspace, then freeze its successful cells with --support')
        previous = json.loads(Path(support).read_text(encoding='utf-8'))
        if (previous['status'] != 'probe_complete' or previous['plan']['sampling'] != plan
                or previous['plan']['teacher_sha256'] != teacher_hash()):
            raise ValueError('Support must be a completed probe with matching plan and teacher')
        support_hash = file_hash(support)
        good = {name for name, stats in previous['coverage']['cells'].items() if stats['accepted'] > 0}
        selected = [c for c in selected if c['cell_id'] in good]
        if not selected:
            raise ValueError('No successful pick-and-place cells; improve the teacher or probe before collecting')
    frozen = dict(sampling=plan, cells=selected, seed=seed, probe=probe, support_sha256=support_hash,
        attempts_per_cell=attempts_per_cell if probe else None, quotas=quotas if not probe else None,
        count=len(selected)*(attempts_per_cell if probe else sum(quotas.values())),
        teacher_sha256=teacher_hash(),
        source_signature=pilot_signature(random_candidate(plan, selected[0], 'train', 0, seed, probe)))
    output = Path(output)
    if output.exists():
        if not resume:
            raise FileExistsError('Choose a new output directory or --resume an identical run')
        manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        if manifest['plan'] != frozen:
            raise ValueError('Sampling plan, teacher, physics, mapping or support changed; choose a fresh output')
    else:
        output.mkdir(parents=True)
        manifest = dict(format='sourccey-box-pilot-v1', collection='continuous-pick-place-v1',
            instruction=PLACE_TASK, test='C', plan=frozen, simulation_only=True,
            physical_calibration_verified=False, policy='X-VLA', fps=FPS,
            camera_names=list(CAMERAS), state_names=list(STATE_NAMES), rendering=provenance(),
            split_strategy='Independent full episodes within every coverage cell; exact XY location IDs are disjoint',
            episodes=[], pending=[], status='probing' if probe else 'physics')
    for config in manifest['pending']:
        path = output/config['episode_id']/'result.json'
        if not path.exists():
            raise ValueError(f'Incomplete candidate preserved at {path.parent}; choose a new output')
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['pilot_signature'] != pilot_signature(config):
            raise ValueError('Recovered candidate signature mismatch')
        manifest['episodes'].append(result)
    manifest['pending'] = []
    def save():
        manifest['coverage'] = coverage(manifest)
        write_json(output/'manifest.json', manifest)
        write_json(output/'coverage.json', manifest['coverage'])
        coverage_image(output, manifest)
    save()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        while True:
            batch = []
            exhausted = []
            for cell in selected:
                for split in (('train',) if probe else SPLITS):
                    previous = [r for r in manifest['episodes'] if r['cell_id'] == cell['cell_id'] and r['split'] == split]
                    count = len(previous) if probe else sum(r['eligible_for_imitation'] for r in previous)
                    goal = attempts_per_cell if probe else quotas[split]
                    if count >= goal:
                        continue
                    attempt = max((r['attempt'] for r in previous), default=-1)+1
                    if not probe and attempt >= goal*8+20:
                        exhausted.append(dict(cell_id=cell['cell_id'], split=split, accepted=count, required=goal))
                        continue
                    batch.append(random_candidate(plan, cell, split, attempt, seed, probe))
            if exhausted:
                manifest.update(status='coverage_incomplete', exhausted=exhausted)
                save()
                raise RuntimeError('Some cells could not meet their quotas; retained failures and coverage.json explain the shortfall')
            if not batch:
                break
            manifest['pending'] = batch.copy()
            save()
            futures = {pool.submit(record_candidate, c, output/c['episode_id']):c for c in batch}
            for future in as_completed(futures):
                config = futures[future]
                result = future.result()
                manifest['episodes'].append(result)
                manifest['episodes'].sort(key=lambda r:r['episode_id'])
                manifest['pending'] = [c for c in manifest['pending'] if c['episode_id'] != config['episode_id']]
                save()
                print(f'{len(manifest["episodes"])} attempts / {manifest["coverage"]["accepted"]} successes | '
                    f'{result["episode_id"]}: '+('PASS' if result['passed'] else 'REJECT '+str(result['failed_checks'])), flush=True)
    manifest['status'] = 'probe_complete' if probe else 'physics_complete'
    save()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('probe', 'collect'), default='probe')
    parser.add_argument('--support', type=Path, help='Frozen probe manifest; required for collection')
    parser.add_argument('--attempts-per-cell', type=int, default=4)
    parser.add_argument('--train-per-cell', type=int, default=100)
    parser.add_argument('--validation-per-cell', type=int, default=10)
    parser.add_argument('--test-per-cell', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261120)
    parser.add_argument('--jobs', type=int, choices=range(1,5), default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run_collection(read_sampling_plan(args.plan), args.output, seed=args.seed, jobs=args.jobs,
        probe=args.phase == 'probe', attempts_per_cell=args.attempts_per_cell,
        quotas=dict(train=args.train_per_cell, validation=args.validation_per_cell, test=args.test_per_cell),
        support=args.support, resume=args.resume)


if __name__ == '__main__':
    main()
