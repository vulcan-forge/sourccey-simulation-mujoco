"""Measured geometry, continuous success criteria and unassisted contact pickup."""
import copy

import mujoco
import numpy as np
import pytest

from examples.box_pickup import Pickup, layout, load_recording, load_scene, save_run, validate


@pytest.fixture(scope='module')
def experiment():
    run = Pickup(load_scene())
    report = run.run()
    return run, report


def test_floor_datum_and_measured_box_dimensions(experiment):
    run, _ = experiment
    m = run.m
    table = m.geom('box_test_table')
    assert table.pos[2]+table.size[2] == pytest.approx(.762)
    assert 2*table.size[:2] == pytest.approx([.90805, 1.8288])
    assert run.layout['near_edge_x_m']-run.layout['chest_x_m'] == pytest.approx(.104775)
    assert m.geom('pickup_box_geom').size == pytest.approx([.0192, .0294, .0292])
    assert m.body('pickup_box').mass[0] == pytest.approx(.12)
    assert run.layout['box_center_m'][2] == pytest.approx(.762+.0584/2)


def test_only_base_and_elevator_are_constrained(experiment):
    run, _ = experiment
    m = run.m
    assert m.neq == 2
    assert m.joint('pickup_box_free').type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert set(m.eq_obj1id) == {m.body('base_link').id, m.joint('linear_actuator').id}
    assert m.geom('pickup_box_geom').friction[0] == pytest.approx(.35)
    assert m.pair_friction[m.pair('box_table_contact').id, :2] == pytest.approx([.4, .4])


def test_unassisted_pickup_and_full_hold(experiment):
    _, report = experiment
    assert report['passed'], report
    assert report['minimum_bottom_clearance_during_hold_mm'] >= 50
    assert report['measured_hold_duration_s'] == pytest.approx(2)
    assert min(report['minimum_finger_normal_force_during_hold_n'].values()) > .1
    assert report['hold_drift_mm'] < 5
    assert report['box_motion_before_close_mm'] < 1
    assert report['robot_table_contact_steps'] == 0
    assert report['max_base_drift_mm'] < 1
    assert report['max_elevator_drift_mm'] < 1
    assert not report['warnings']


def test_open_gripper_control_cannot_pass(experiment):
    previous, _ = experiment
    config = copy.deepcopy(previous.config)
    config['grasp']['closed_command_deg'] = config['grasp']['open_angle_deg']
    run = Pickup(config, model=previous.m, measured=previous.layout)
    report = run.run()
    assert not report['passed']
    assert not report['checks']['both_fingers_hold']
    assert not report['checks']['box_clear_of_table']


def test_one_dropped_hold_sample_fails_even_if_final_pose_succeeds(experiment):
    run, _ = experiment
    samples = run.hold_samples
    try:
        run.hold_samples = samples.copy()
        time_s, _, position, forces = samples[len(samples)//2]
        run.hold_samples[len(samples)//2] = (time_s, .762, position, forces)
        assert not run.report()['checks']['box_clear_of_table']
        run.hold_samples = samples[500:]
        assert not run.report()['checks']['full_hold_duration']
    finally:
        run.hold_samples = samples


def test_recording_rejects_changed_physics(experiment, tmp_path):
    run, report = experiment
    save_run(run, report, tmp_path)
    recording, restored = load_recording(tmp_path, run.config)
    assert restored['passed']
    assert np.all(np.diff(recording['time']) > 0)
    assert recording['qpos'].shape[1] == run.m.nq
    changed = copy.deepcopy(run.config)
    changed['object']['mass_kg'] = .15
    with pytest.raises(ValueError, match='changed'):
        load_recording(tmp_path, changed)


@pytest.mark.parametrize('section,key,value', [
    ('object', 'mass_kg', 0), ('grasp', 'sliding_friction', float('nan')),
    ('box_pose', 'yaw_deg', 45), ('robot_setup', 'base_fixed', False),
    ('grasp', 'open_angle_deg', 90), ('success', 'hold_seconds', -1),
])
def test_invalid_scene_is_rejected(section, key, value):
    config = load_scene()
    config[section][key] = value
    with pytest.raises(ValueError):
        validate(config)


def test_box_cannot_start_beyond_table_edge():
    config = load_scene()
    config['box_pose']['center_from_near_edge_mm'] = 2
    with pytest.raises(ValueError, match='wholly on'):
        layout(config)
