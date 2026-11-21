import json

import numpy as np
import pytest

from examples.box_random_dataset import DEFAULT_PLAN, cells, random_candidate, SPLITS
from sourccey.box_task import PLACE_TASK, PlacementTracker


@pytest.fixture
def plan():
    return json.loads(DEFAULT_PLAN.read_text())


def test_continuous_sampling_spans_cells_and_preserves_independent_episode_splits(plan):
    strata = cells(plan)
    configs = [random_candidate(plan, c, split, i, 231)
               for c in strata for split in SPLITS for i in range(6)]
    assert len({c['episode_id'] for c in configs}) == len(configs)
    assert len({c['workspace']['location_id'] for c in configs}) == len(configs)
    assert len({c['seed'] for c in configs}) == len(configs)
    for c in configs:
        cell = c['random_collection']['cell']
        pose = c['box_pose']
        assert cell['depth_mm'][0] <= pose['center_from_near_edge_mm'] < cell['depth_mm'][1]
        assert cell['lateral_mm'][0] <= pose['center_left_of_robot_mm'] < cell['lateral_mm'][1]
        assert -40 <= pose['yaw_deg'] <= 40
        assert cell['yaw_deg'][0] <= pose['yaw_deg'] < cell['yaw_deg'][1]
        assert c['dataset_pilot']['instruction'] == PLACE_TASK
        assert c['domain_randomization']['camera_pose_randomized']
        assert c['placement'] == plan['placement']
    depths = [c['box_pose']['center_from_near_edge_mm'] for c in configs]
    laterals = [c['box_pose']['center_left_of_robot_mm'] for c in configs]
    yaws = [c['box_pose']['yaw_deg'] for c in configs]
    assert np.ptp(depths) > 150 and np.ptp(laterals) > 225 and np.ptp(yaws) > 75
    # Probe and dataset seeds have separate streams, even with identical CLI seed.
    first = configs[0]
    assert first == random_candidate(plan, strata[0], 'train', 0, 231)
    assert first['box_pose'] != random_candidate(plan, strata[0], 'train', 0, 231, probe=True)['box_pose']


def observe(tracker, timestamp, **overrides):
    sample = dict(position=[*tracker.goal, .7912], bottom=.762, upright=1.,
                  finger_force=0., table_force=1., lifted=True)
    sample.update(overrides)
    tracker.observe(timestamp, **sample)


@pytest.mark.parametrize('bad', [dict(lifted=False), dict(finger_force=1.), dict(table_force=0.),
    dict(bottom=.80), dict(upright=0.), dict(position=[1, 1, .7912])])
def test_placement_rejects_no_pickup_held_boxes_wrong_goal_and_unstable_support(plan, bad):
    tracker = PlacementTracker(plan['placement'], .312, .762)
    for t in np.arange(0, 2, .02):
        observe(tracker, t, **bad)
    assert tracker.stable_placement is None


def test_placement_requires_full_uninterrupted_released_settle_window(plan):
    tracker = PlacementTracker(plan['placement'], .312, .762)
    for t in np.arange(0, .8, .02):
        observe(tracker, t)
    observe(tracker, .8, finger_force=1.)
    for t in np.arange(.82, 1.8, .02):
        observe(tracker, t)
    assert tracker.stable_placement is None
    observe(tracker, 1.82)
    assert tracker.stable_placement['duration_s'] == pytest.approx(1.)
    assert tracker.stable_placement['drift_mm'] == 0


def test_placement_rejects_box_sliding_within_destination_tolerance(plan):
    tracker = PlacementTracker(plan['placement'], .312, .762)
    for t in np.arange(0, 2, .02):
        observe(tracker, t, position=[tracker.goal[0]+.005*t, tracker.goal[1], .7912])
    assert tracker.stable_placement is None


def test_collection_evaluation_requires_completed_quotas_and_explicit_final_test(tmp_path):
    from examples.box_policy_sweep import select_collection
    (tmp_path/'manifest.json').write_text(json.dumps(dict(collection='continuous-pick-place-v1',
        pending=[], status='probe_complete', episodes=[])))
    with pytest.raises(ValueError, match='completed'):
        select_collection(tmp_path, 'validation')
    with pytest.raises(ValueError, match='reserved'):
        select_collection(tmp_path, 'test')
    with pytest.raises(ValueError, match='positive'):
        select_collection(tmp_path, 'validation', limit=0)


def test_terminal_mapping_error_rejects_episode_and_does_not_escape(monkeypatch, tmp_path, plan):
    from examples import box_random_dataset as collector
    config = random_candidate(plan, cells(plan)[0], 'train', 0, 231)
    config['box_pose'].update(center_from_near_edge_mm=110, center_left_of_robot_mm=130)
    report = dict(passed=False, checks=dict(physics_healthy=True, teacher_runtime_healthy=False),
        error='Teacher target exceeds the existing normalized hardware mapping')
    class FakeRun:
        def __init__(self, *args):
            pass
        def run(self, callback):
            return report
        def audit_contacts(self, run):
            pass
    class FakeRecorder:
        rows = [1, 2]
        def __init__(self, run):
            pass
        def capture(self, run):
            pass
        def finish(self, report, directory):
            raise ValueError('Teacher target exceeds the existing normalized hardware mapping')
    saved = []
    monkeypatch.setattr(collector, 'scene_model', lambda config:(None, None))
    monkeypatch.setattr(collector, 'apply_camera_perturbations', lambda *args:None)
    monkeypatch.setattr(collector, 'PickAndPlace', FakeRun)
    monkeypatch.setattr(collector, 'EpisodeRecorder', FakeRecorder)
    monkeypatch.setattr(collector, 'save_run', lambda run, report, *args, **kwargs:saved.append(report.copy()))
    result = collector.record_candidate(config, tmp_path/'bad_episode')
    assert not result['passed'] and not result['eligible_for_imitation']
    assert result['failure_stage'] == 'recording'
    assert 'training_recording_valid' in result['failed_checks']
    assert result['recording_available'] and not result['training_recording_available']
    assert not saved[-1]['passed']
    assert json.loads((tmp_path/'bad_episode/result.json').read_text()) == result
