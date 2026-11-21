"""Paired grip-friction experiments using unchanged, signed baseline scenes."""
from collections import Counter
import copy
import json
import math
from pathlib import Path

from examples.box_dataset_pilot import file_hash, pilot_signature
from examples.box_random_dataset import teacher_hash
from examples.box_workspace import write_json


def scaled_scene(config, scale):
    if not math.isfinite(scale) or not 1 < scale <= 4:
        raise ValueError('Require a finite grip-friction multiplier greater than 1 and at most 4')
    updated = copy.deepcopy(config)
    old = config['grasp']['sliding_friction']
    updated['grasp']['sliding_friction'] = old*scale
    updated['contact_experiment'] = dict(profile='ridged-pad-effective-friction-v1',
        baseline_episode=config['episode_id'], baseline_grip_friction=old,
        grip_friction=old*scale, grip_friction_scale=scale,
        calibration='Uncalibrated effective Coulomb-friction approximation for ridged rubber pads',
        geometry_changed=False, table_friction_changed=False,
        source_sha256=file_hash(__file__))
    updated['random_collection']['teacher_sha256'] = teacher_hash()
    updated['domain_randomization']['physics_note'] = (
        f'Paired ridged-pad experiment: hand sliding friction {old*scale:.6g} '
        f'({scale:g}x baseline {old:.6g}); all other physical samples retained. Not measured calibration.')
    return updated


def paired_attempts(reference, count, scale, sampling_plan):
    reference = Path(reference).resolve()
    path = reference/'manifest.json'
    baseline = json.loads(path.read_text(encoding='utf-8'))
    configs = baseline.get('planned_scenes', [])
    results = {r['episode_id']:r for r in baseline['episodes']}
    if (baseline.get('batch') != 'fixed-attempt-budget-v1' or baseline['pending']
            or baseline['status'] not in ('physics_complete', 'rendering', 'complete')
            or baseline['attempt_target'] != count or len(configs) != count or len(results) != count
            or len({c['episode_id'] for c in configs}) != count
            or baseline['plan']['sampling'] != sampling_plan):
        raise ValueError('Reference must have a complete, matching fixed-budget physics batch')
    updated = []
    for config in configs:
        name = config['episode_id']
        directory = (reference/name).resolve()
        if directory.parent != reference or name not in results:
            raise ValueError('Invalid baseline episode identity')
        scene = json.loads((directory/'scene.json').read_text(encoding='utf-8'))
        if scene != config or results[name]['pilot_signature'] != pilot_signature(config):
            raise ValueError(f'Baseline scene or physics signature changed: {name}')
        updated.append(scaled_scene(config, scale))
    experiment = dict(kind='paired-grip-friction-v1', reference=str(reference),
        reference_seed=baseline['plan']['seed'], reference_manifest_sha256=file_hash(path),
        grip_friction_scale=scale,
        baseline_friction_range=[min(c['grasp']['sliding_friction'] for c in configs),
                                 max(c['grasp']['sliding_friction'] for c in configs)],
        friction_range=[min(c['grasp']['sliding_friction'] for c in updated),
                        max(c['grasp']['sliding_friction'] for c in updated)],
        baseline_successful=sum(bool(r['passed']) for r in results.values()),
        comparison='Same frozen poses, seeds, appearance, mass, table friction, controller, limits and success checks; only grip sliding friction changes',
        calibration='Experimental approximation, not measured real-gripper calibration')
    return updated, experiment


def outcome_comparison(baseline_rows, current_rows):
    before = {r['episode_id']:r for r in baseline_rows}
    after = {r['episode_id']:r for r in current_rows}
    if (len(before) != len(baseline_rows) or len(after) != len(current_rows)
            or set(before) != set(after) or not before):
        raise ValueError('Paired comparison requires the same unique completed episode identities')
    pairs = []
    totals = Counter()
    for name, previous in before.items():
        current = after[name]
        outcome = ('still_successful' if previous['passed'] and current['passed'] else
                   'improved' if current['passed'] else 'regressed' if previous['passed'] else 'still_failed')
        totals[outcome] += 1
        pairs.append(dict(episode_id=name, outcome=outcome,
            baseline_passed=bool(previous['passed']), passed=bool(current['passed']),
            baseline_failed_checks=previous.get('failed_checks', []),
            failed_checks=current.get('failed_checks', [])))
    baseline_passed = sum(bool(r['passed']) for r in before.values())
    passed = sum(bool(r['passed']) for r in after.values())
    summary = dict(attempts=len(before), baseline_successful=baseline_passed, successful=passed,
        net_success_change=passed-baseline_passed,
        **{key:totals[key] for key in ('improved','regressed','still_successful','still_failed')})
    return dict(summary=summary, pairs=pairs,
        baseline_failed_checks=dict(Counter(k for r in before.values() if not r['passed'] for k in r.get('failed_checks', []))),
        failed_checks=dict(Counter(k for r in after.values() if not r['passed'] for k in r.get('failed_checks', []))))


def write_comparison(output, manifest):
    experiment = manifest['experiment']
    reference_path = Path(experiment['reference'])/'manifest.json'
    if file_hash(reference_path) != experiment['reference_manifest_sha256']:
        raise ValueError('Baseline manifest changed since the paired experiment was frozen')
    baseline = json.loads(reference_path.read_text(encoding='utf-8'))
    report = outcome_comparison(baseline['episodes'], manifest['episodes'])
    report['experiment'] = experiment
    write_json(Path(output)/'comparison.json', report)
    s = report['summary']
    lines = ['# Paired grip-friction comparison', '',
        f"Baseline: {s['baseline_successful']}/{s['attempts']} successful. Higher friction: {s['successful']}/{s['attempts']}.", '',
        f"Improved: {s['improved']}. Regressed: {s['regressed']}. Still successful: {s['still_successful']}. Still failed: {s['still_failed']}.", '',
        experiment['comparison']+'.', '', experiment['calibration']+'.', '',
        '| Episode | Outcome | Baseline failed checks | Higher-friction failed checks |',
        '| --- | --- | --- | --- |']
    lines.extend(f"| {r['episode_id']} | {r['outcome']} | {', '.join(r['baseline_failed_checks']) or 'none'} | {', '.join(r['failed_checks']) or 'none'} |"
                 for r in report['pairs'])
    (Path(output)/'COMPARISON.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report
