import json
import os
from datetime import datetime, timezone

import numpy as np
import pytest

from examples.box_attempt_batch import planned_attempts
from examples.box_random_dataset import DEFAULT_PLAN
from examples import box_status


def test_fixed_budget_freezes_fifty_broad_poses_and_splits_before_outcomes():
    plan = json.loads(DEFAULT_PLAN.read_text())
    configs = planned_attempts(plan, 50, 2026112001)
    assert len(configs) == 50
    assert configs == planned_attempts(plan, 50, 2026112001)
    assert len({c['episode_id'] for c in configs}) == 50
    assert len({c['workspace']['location_id'] for c in configs}) == 50
    assert len({c['dataset_pilot']['cell_id'] for c in configs}) == 50
    assert {s:sum(c['split'] == s for c in configs) for s in ('train','validation','test')} == dict(train=40, validation=5, test=5)
    positions = np.array([[c['box_pose']['center_from_near_edge_mm'],
                          c['box_pose']['center_left_of_robot_mm']] for c in configs])
    assert np.all(np.linalg.norm(positions-[110,180], axis=1) >= 40)
    assert np.ptp(positions[:,0]) > 130 and np.ptp(positions[:,1]) > 200
    assert min(c['box_pose']['yaw_deg'] for c in configs) < -30
    assert max(c['box_pose']['yaw_deg'] for c in configs) > 30
    for bad in (0, -1, 10001):
        with pytest.raises(ValueError):
            planned_attempts(plan, bad, 1)


def test_status_counts_completed_not_queued_or_unfinished_attempts(tmp_path, monkeypatch):
    pointer = tmp_path/'latest.json'
    pointer.write_text(json.dumps(dict(output=str(tmp_path))))
    (tmp_path/'run_state.json').write_text(json.dumps(dict(pid=123, stage='physics', attempt_target=50,
        started_utc=datetime.now(timezone.utc).isoformat())))
    episodes = [dict(episode_id='passed', eligible_for_imitation=True, split='train', videos=False),
                dict(episode_id='failed', eligible_for_imitation=False, split='validation', videos=False)]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(attempt_target=50, episodes=episodes)))
    (tmp_path/'active').mkdir()
    (tmp_path/'active/scene.json').write_text('{}')
    monkeypatch.setattr(box_status, 'process_alive', lambda pid:True)
    status = box_status.snapshot(latest=pointer)
    assert status['completed'] == 2 and status['successful'] == 1 and status['rejected'] == 1
    assert status['active'] == 1 and status['queued'] == 47 and status['rendered'] == 0
    assert 'Attempts completed: 2/50' in box_status.format_status(status)
    monkeypatch.setattr(box_status, 'process_alive', lambda pid:False)
    stopped = box_status.snapshot(latest=pointer)
    assert stopped['stage'] == 'stopped unexpectedly'
    assert stopped['active'] == 0 and stopped['queued'] == 48


def test_status_reads_initializing_and_finished_stages(tmp_path):
    now = datetime.now(timezone.utc).isoformat()
    state = dict(pid=os.getpid(), stage='initializing', attempt_target=50, started_utc=now)
    (tmp_path/'run_state.json').write_text(json.dumps(state))
    status = box_status.snapshot(tmp_path)
    assert status['stage'] == 'initializing' and status['completed'] == 0 and status['runner_alive']
    state.update(stage='complete', dataset='dataset/path')
    (tmp_path/'run_state.json').write_text(json.dumps(state))
    assert box_status.snapshot(tmp_path)['dataset'] == 'dataset/path'


def test_recovery_keeps_finished_results_and_rejects_signed_mapping_failure(monkeypatch, tmp_path):
    from examples import box_attempt_batch as batch
    plan = json.loads(DEFAULT_PLAN.read_text())
    configs = planned_attempts(plan, 4, 123)
    results = []
    for config in configs[:2]:
        directory = tmp_path/config['episode_id']
        directory.mkdir()
        (directory/'scene.json').write_text(json.dumps(config))
        result = batch.candidate_summary(config)
        result.update(passed=False, eligible_for_imitation=False, failed_checks=['pickup'])
        (directory/'result.json').write_text(json.dumps(result))
        results.append(result)
    broken = tmp_path/configs[2]['episode_id']
    broken.mkdir()
    (broken/'scene.json').write_text(json.dumps(configs[2]))
    original = dict(passed=False, checks=dict(box_clear_of_table=False),
        error='Teacher target exceeds the existing normalized hardware mapping', scene=configs[2])
    report_bytes = json.dumps(original).encode()
    (broken/'report.json').write_bytes(report_bytes)
    (broken/'recording.npz').write_bytes(b'signed-replay-placeholder')
    monkeypatch.setattr(batch, 'load_recording', lambda *args:({}, original))
    manifest = dict(planned_scenes=configs, attempt_target=4, episodes=results[:1])
    recovered, pending, repaired = batch.recover_results(manifest, tmp_path)
    assert len(recovered) == 3 and pending == configs[3:] and repaired == 1
    assert all(not r['eligible_for_imitation'] for r in recovered)
    assert (broken/'report.json').read_bytes() == report_bytes
    assert (broken/'recording.npz').read_bytes() == b'signed-replay-placeholder'
    assert json.loads((broken/'result.json').read_text())['recovered_from_signed_failure']
    # Recovery is idempotent: no attempts are regenerated or failures promoted.
    manifest['episodes'] = recovered
    again, remaining, repaired_again = batch.recover_results(manifest, tmp_path)
    assert again == recovered and remaining == pending and repaired_again == 0


def test_recovery_does_not_accept_an_unrecognized_partial_episode(tmp_path):
    from examples import box_attempt_batch as batch
    plan = json.loads(DEFAULT_PLAN.read_text())
    configs = planned_attempts(plan, 1, 123)
    directory = tmp_path/configs[0]['episode_id']
    directory.mkdir()
    (directory/'scene.json').write_text(json.dumps(configs[0]))
    with pytest.raises(ValueError, match='preserved for review'):
        batch.recover_results(dict(planned_scenes=configs, attempt_target=1, episodes=[]), tmp_path)


def test_recovery_never_mixes_new_teacher_source_into_unfinished_physics(monkeypatch):
    from examples import box_attempt_batch as batch
    monkeypatch.setattr(batch, 'teacher_hash', lambda:'current-source')
    manifest = dict(plan=dict(teacher_sha256='original-source'))
    batch.validate_remaining_teacher(manifest, [])
    with pytest.raises(ValueError, match='Teacher source changed'):
        batch.validate_remaining_teacher(manifest, [dict(episode_id='unfinished')])
    manifest['plan']['teacher_sha256'] = 'current-source'
    batch.validate_remaining_teacher(manifest, [dict(episode_id='unfinished')])
