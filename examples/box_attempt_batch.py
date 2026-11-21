"""Run an exact attempt budget, then render/export only successful sequences."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from examples.box_random_dataset import (DEFAULT_PLAN, SPLITS, cells, coverage,
    coverage_image, file_hash, pilot_signature, random_candidate, read_sampling_plan,
    record_candidate, teacher_hash, CAMERAS, FPS, STATE_NAMES, PLACE_TASK, ROOT, provenance, write_json)
from examples.box_random_dataset import candidate_summary
from examples.box_pickup import load_recording
from examples.box_status import process_alive


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def planned_attempts(plan, count, seed):
    """Freeze spatial/rotation coverage and episode splits before any outcomes."""
    if not 1 <= count <= 10000 or seed < 0:
        raise ValueError('Require 1..10000 attempts and a nonnegative seed')
    strata = cells(plan)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 501]))
    order = []
    while len(order) < count:
        order.extend(rng.permutation(len(strata)).tolist())
    validation = count//10
    test = count//10
    splits = np.array(['train']*(count-validation-test)+['validation']*validation+['test']*test)
    rng.shuffle(splits)
    goal = np.array([plan['placement']['center_from_near_edge_mm'], plan['placement']['center_left_of_robot_mm']])
    selected = []
    for ordinal, index in enumerate(order[:count]):
        # Resample geometric exclusions before admitting a physics attempt.
        for retry in range(1024):
            config = random_candidate(plan, strata[index], str(splits[ordinal]), ordinal*1024+retry, seed)
            pose = config['box_pose']
            xy = np.array([pose['center_from_near_edge_mm'], pose['center_left_of_robot_mm']])
            if np.linalg.norm(xy-goal) >= plan['minimum_transport_mm']:
                break
        else:
            raise ValueError(f'No eligible start found in {strata[index]["cell_id"]}; adjust the plan')
        config['random_collection'].update(batch_ordinal=ordinal,
            batch_source_sha256=file_hash(__file__), fixed_attempt_budget=count)
        selected.append(config)
    return selected


def recover_results(manifest, output):
    """Recover signed worker results, never rerun or overwrite completed scenes."""
    output = Path(output)
    configs = manifest['planned_scenes']
    if len(configs) != manifest['attempt_target'] or len({c['episode_id'] for c in configs}) != len(configs):
        raise ValueError('Frozen attempt selection is incomplete or duplicated')
    known = {r['episode_id']:r for r in manifest['episodes']}
    if len(known) != len(manifest['episodes']) or not set(known) <= {c['episode_id'] for c in configs}:
        raise ValueError('Completed result identities disagree with the frozen selection')
    results, pending, repaired = [], [], []
    for config in configs:
        name = config['episode_id']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Invalid episode identifier')
        directory = output/name
        if not directory.exists():
            if name in known:
                raise ValueError(f'Missing completed scene: {name}')
            pending.append(config)
            continue
        saved_config = json.loads((directory/'scene.json').read_text(encoding='utf-8'))
        if saved_config != config:
            raise ValueError(f'Saved scene differs from frozen selection: {name}')
        result_path = directory/'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding='utf-8'))
            expected = candidate_summary(config)
            if (any(result.get(key) != value for key, value in expected.items() if key != 'videos')
                    or bool(result['passed']) != bool(result['eligible_for_imitation'])
                    or (name in known and {k:v for k,v in known[name].items() if k != 'videos'}
                        != {k:v for k,v in result.items() if k != 'videos'})):
                raise ValueError(f'Saved result identity or signature mismatch: {name}')
            if name in known:
                result['videos'] = known[name].get('videos', False)
            if result['eligible_for_imitation']:
                _, report = load_recording(directory, config)
                episode = json.loads((directory/'episode.json').read_text(encoding='utf-8'))
                if not report['passed'] or not episode['eligible_for_imitation'] or not (directory/'transitions.npz').exists():
                    raise ValueError(f'Successful episode has incomplete training data: {name}')
        else:
            # Recover the specific recorder failure using its original signed
            # physics replay. Unknown partial scenes remain blocked for review.
            if not (directory/'recording.npz').exists() or not (directory/'report.json').exists():
                raise ValueError(f'Incomplete scene preserved for review: {name}')
            _, report = load_recording(directory, config)
            error = report.get('error', '')
            if report['passed'] or 'normalized hardware mapping' not in error:
                raise ValueError(f'Unrecognized incomplete scene preserved for review: {name}')
            result = candidate_summary(config)
            result.update(passed=False, eligible_for_imitation=False, frames=0,
                pickup_passed=report.get('pickup_passed', False), error=error,
                failed_checks=[key for key, value in report['checks'].items() if not value]
                    + ['training_recording_valid'], failure_stage='recording',
                recording_available=True, training_recording_available=False,
                recovered_from_signed_failure=True)
            repaired.append((result_path, result))
        results.append(result)
    for path, result in repaired:
        write_json(path, result)
    return sorted(results, key=lambda r:r['episode_id']), pending, len(repaired)


def validate_remaining_teacher(manifest, pending):
    # Completed signed replays can be recovered after a recorder-only repair.
    # New physics must still use the exact original teacher/recorder source.
    if pending and manifest['plan']['teacher_sha256'] != teacher_hash():
        raise ValueError('Teacher source changed and unfinished physics remains; preserve this run and use a fresh batch')


def run_batch(args):
    output = args.output.resolve()
    resume = getattr(args, 'resume', False)
    previous_state = None
    if output.exists() and not resume:
        raise FileExistsError('Choose a new output directory; previous attempts are preserved')
    if resume:
        if not (output/'manifest.json').exists():
            raise FileNotFoundError('No saved batch manifest to resume')
        previous_state = json.loads((output/'run_state.json').read_text(encoding='utf-8'))
        if process_alive(previous_state.get('pid')):
            raise ValueError('The existing batch runner is still alive; do not launch another')
    if not 1 <= args.jobs <= 4:
        raise ValueError('Require 1..4 physics workers')
    output.mkdir(parents=True, exist_ok=resume)
    state = dict(pid=os.getpid(), started_utc=previous_state['started_utc'] if previous_state else utc_now(), stage='initializing',
        attempt_target=args.attempts, output=str(output), dataset=None, error=None)
    if previous_state:
        state['resumed_utc'] = utc_now()
        state['last_failure'] = previous_state.get('error')
    def stage(name, **extra):
        state.update(stage=name, updated_utc=utc_now(), **extra)
        write_json(output/'run_state.json', state)
    stage('initializing')
    write_json(ROOT/'artifacts/box_attempts_latest.json', dict(output=str(output)))
    try:
        plan = read_sampling_plan(args.plan)
        if resume:
            manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            if (manifest.get('batch') != 'fixed-attempt-budget-v1' or manifest['attempt_target'] != args.attempts
                    or (not manifest.get('experiment') and manifest['plan']['seed'] != args.seed) or manifest['plan']['sampling'] != plan
                    or manifest['rendering'] != provenance()
                    or manifest['plan']['source_signature'] != pilot_signature(manifest['planned_scenes'][0])):
                raise ValueError('Frozen plan, physics, engine, mapping or rendering changed; do not mix runs')
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            write_json(output/f'manifest.before_recovery_{stamp}.json', manifest)
            write_json(output/f'run_state.before_recovery_{stamp}.json', previous_state)
            recovered, configs, repaired = recover_results(manifest, output)
            validate_remaining_teacher(manifest, configs)
            manifest['episodes'] = recovered
            manifest['pending'] = configs.copy()
            manifest.setdefault('recovery_history', []).append(dict(resumed_utc=utc_now(),
                batch_source_sha256=file_hash(__file__), teacher_wrapper_sha256=teacher_hash(),
                recovered_results=len(recovered), repaired_recording_failures=repaired,
                remaining_attempts=len(configs), original_physics_signature_verified=True))
            print(f'Recovered {len(recovered)}/{args.attempts} results; '
                f'{sum(r["eligible_for_imitation"] for r in recovered)} successful; '
                f'{len(configs)} attempts remaining. Original scene selection retained.', flush=True)
        else:
            reference = getattr(args, 'reference', None)
            experiment = None
            if reference:
                from examples.box_friction_experiment import paired_attempts
                configs, experiment = paired_attempts(reference, args.attempts,
                    args.grip_friction_scale if args.grip_friction_scale is not None else 2., plan)
                for config in configs:
                    config['random_collection']['batch_source_sha256'] = file_hash(__file__)
            else:
                configs = planned_attempts(plan, args.attempts, args.seed)
            manifest = dict(format='sourccey-box-pilot-v1', collection='continuous-pick-place-v1',
            batch='fixed-attempt-budget-v1', attempt_target=args.attempts, instruction=PLACE_TASK,
            simulation_only=True, physical_calibration_verified=False, policy='X-VLA', test='C',
            fps=FPS, camera_names=list(CAMERAS), state_names=list(STATE_NAMES), rendering=provenance(),
            plan=dict(sampling=plan, cells=cells(plan), seed=experiment['reference_seed'] if experiment else args.seed, count=0,
                teacher_sha256=teacher_hash(), batch_source_sha256=file_hash(__file__),
                source_signature=pilot_signature(configs[0])),
            split_strategy='Frozen independent episodes; 80/10/10 allocation before physics; exact XY poses disjoint',
            selection='Seeded permutation across XY/yaw strata; geometric destination exclusions resampled before physics; failures never replaced',
                planned_scenes=configs, episodes=[], pending=configs.copy(), status='physics')
            if experiment:
                manifest['experiment'] = experiment
                manifest['selection'] = 'Paired baseline scenes, unchanged episode identities and split assignments; failures never replaced'
        def save():
            manifest['coverage'] = coverage(manifest)
            write_json(output/'manifest.json', manifest)
            write_json(output/'coverage.json', manifest['coverage'])
            coverage_image(output, manifest)
        save()
        stage('physics')
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(record_candidate, config, output/config['episode_id']):config for config in configs}
            for future in as_completed(futures):
                result = future.result()
                manifest['episodes'].append(result)
                manifest['episodes'].sort(key=lambda row:row['episode_id'])
                manifest['pending'] = [c for c in manifest['pending'] if c['episode_id'] != result['episode_id']]
                save()
                stage('physics')
                print(f'{len(manifest["episodes"])}/{args.attempts} attempts completed | '
                    f'{manifest["coverage"]["accepted"]} successful | {result["episode_id"]}: '
                    +('PASS' if result['passed'] else 'REJECT '+str(result['failed_checks'])), flush=True)
        # Legacy rendering/export count denotes accepted recordings. The immutable
        # attempt_target remains the denominator for this exact-budget experiment.
        manifest['plan']['count'] = manifest['coverage']['accepted']
        manifest['status'] = 'physics_complete'
        save()
        if manifest.get('experiment'):
            from examples.box_friction_experiment import write_comparison
            comparison = write_comparison(output, manifest)
            print('Paired comparison: '+json.dumps(comparison['summary']), flush=True)
        accepted = manifest['coverage']['accepted']
        if args.render and accepted:
            stage('rendering')
            from examples.box_pilot_render import render_parallel
            manifest = render_parallel(output, jobs=2)
        if args.export and accepted:
            if not manifest['coverage']['split_episodes']['train']:
                raise ValueError('No successful training episodes to export; all attempt results have been preserved')
            stage('exporting')
            dataset = output.parent/'datasets'/(output.name+'_v3')
            command = [str(args.export_python), '-u', '-m', 'integrations.export_box_dataset',
                '--source', str(output), '--output', str(dataset), '--repo-id', 'local/'+output.name,
                '--policy', 'xvla']
            subprocess.run(command, cwd=ROOT, check=True,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            state['dataset'] = str(dataset)
        stage('complete')
        print(f'Finished {args.attempts} attempts; {accepted} successful. Output: {output}', flush=True)
        return manifest
    except BaseException as error:
        stage('failed', error=f'{type(error).__name__}: {error}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attempts', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2026112001)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Recover signed results and continue the same frozen batch')
    parser.add_argument('--reference', type=Path, help='Repeat a completed batch with exactly the same frozen scenes')
    parser.add_argument('--grip-friction-scale', type=float, help='Grip sliding friction multiplier for --reference; defaults to 2')
    parser.add_argument('--export-python', type=Path,
        default=ROOT.parents[1]/'VulcanLerobot/lerobot-vulcan/.venv/Scripts/python.exe')
    args = parser.parse_args()
    if args.export and not args.render:
        parser.error('--export also requires --render')
    if args.grip_friction_scale is not None and args.reference is None:
        parser.error('--grip-friction-scale requires --reference')
    run_batch(args)


if __name__ == '__main__':
    main()
