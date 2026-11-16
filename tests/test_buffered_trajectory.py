from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from integrations.buffered_trajectory import KEYS, Trajectory, Executor
from integrations.prepare_buffered_host_patch import prepare


def payload(times=(0., .3, .6), values=(0., 35., -25.)):
    rows = []
    for value in values:
        row = [0.] * len(KEYS)
        row[1] = value
        rows.append(row)
    return dict(version=1, units='normalized', keys=list(KEYS), times=list(times),
                positions=rows, elevator=100., tracking_tolerance=5., stall_timeout=15.)


def observation(current=None):
    return dict(positions=dict.fromkeys(KEYS, 0.) if current is None else current.copy(),
                elevator=100., base_velocity=[0.]*4)


def test_portable_interpolation_matches_original_scipy_path():
    rng = np.random.default_rng(7)
    p = payload()
    p['times'] = [0., .25, 2.25, 2.29, 2.33, 5., 6.]
    rows = rng.uniform(-90, 90, (7, len(KEYS)))
    rows[:, [5, 11]] = rng.uniform(0, 100, (7, 2))
    p['positions'] = rows.tolist()
    path = Trajectory(p)
    reference = PchipInterpolator(p['times'], rows, axis=0)
    for t in np.linspace(0, 6, 300):
        np.testing.assert_allclose(list(path.at(t).values()), reference(t), atol=2e-10)


@pytest.mark.parametrize('fault', ['duplicate_time', 'reversed_time', 'nan', 'out_of_range',
                                  'wrong_order', 'wrong_units', 'length', 'tolerance', 'duration'])
def test_invalid_upload_cannot_construct_an_executable_path(fault):
    p = payload()
    if fault == 'duplicate_time': p['times'][1] = 0.
    if fault == 'reversed_time': p['times'][1] = -1.
    if fault == 'nan': p['positions'][1][0] = float('nan')
    if fault == 'out_of_range': p['positions'][1][0] = 101.
    if fault == 'wrong_order': p['keys'].reverse()
    if fault == 'wrong_units': p['units'] = 'degrees'
    if fault == 'length': p['positions'][0].pop()
    if fault == 'tolerance': p['tracking_tolerance'] = 10.
    if fault == 'duration': p['times'][-1] = 601.
    with pytest.raises(ValueError):
        Trajectory(p)


def test_local_executor_completes_slow_path_without_pc_waypoint_updates():
    executor = Executor(Trajectory(payload()))
    current, goals, sent = dict.fromkeys(KEYS, 0.), dict.fromkeys(KEYS, 0.), []
    executor.start(0., observation(current))
    def send(target):
        assert max(abs(target[k]-current[k]) for k in KEYS) <= 5.000001
        assert max(abs(target[k]-goals[k]) for k in KEYS) <= 1.000001
        goals.update(target)
        sent.append(target.copy())
    for i in range(1, 5000):
        now = i/30
        for k in KEYS:
            current[k] += max(-3/30, min(3/30, goals[k]-current[k]))
        if i % 30 == 0:
            executor.heartbeat = now  # Only a lease heartbeat, no future targets.
        executor.tick(now, observation(current), send)
        if executor.state != 'running': break
    assert executor.state == 'complete', executor.error
    assert max(t[KEYS[1]] for t in sent) == pytest.approx(35., abs=1.)
    assert executor.elapsed == .6
    assert sent[-1][KEYS[1]] == pytest.approx(-25.)
    completed = executor.status(now)['wall_elapsed_s']
    assert executor.status(now+100)['wall_elapsed_s'] == completed


@pytest.mark.parametrize('fault', ['lease', 'local_gap', 'tracking', 'base', 'elevator', 'encoder'])
def test_local_fault_aborts_before_another_motor_command(fault):
    ex = Executor(Trajectory(payload()))
    ex.start(0., observation())
    obs, now = observation(), .03
    if fault == 'lease': ex.heartbeat = -5.
    if fault == 'local_gap': now = .6
    if fault == 'tracking': obs['positions'][KEYS[0]] = 11.
    if fault == 'base': obs['base_velocity'][0] = .1
    if fault == 'elevator': obs['elevator'] = 103.
    if fault == 'encoder': obs['positions'][KEYS[0]] = float('nan')
    sent = []
    ex.tick(now, obs, sent.append)
    assert ex.state == 'aborted'
    assert not sent
    ex.start(now+1, observation())
    ex.tick(now+1, observation(), sent.append)
    assert not sent  # A repeated or delayed start cannot revive an aborted job.


def test_stationary_motor_stalls_with_bounded_targets():
    ex = Executor(Trajectory(payload()))
    ex.start(0., observation())
    sent = []
    for i in range(1, 1000):
        now = i/30
        ex.heartbeat = now
        ex.tick(now, observation(), sent.append)
        if ex.state != 'running': break
    assert ex.state == 'aborted' and 'No path progress' in ex.error
    assert max(t[KEYS[1]] for t in sent) <= 5.


def test_host_patch_checks_exact_revision_and_preserves_legacy_safety():
    source = (Path(__file__).parent/'fixtures/host_before_buffered.py').read_text()
    patched = prepare(source)
    compile(patched, 'patched_host', 'exec')
    assert 'host.buffered.step()' in patched
    assert 'robot.send_action(data)' in patched
    assert 'Preempted by direct robot command' in patched
    with pytest.raises(ValueError, match='revision changed'):
        prepare(source+'\n')


def test_pc_payload_preserves_all_existing_curve_knots():
    from examples.robot_ball_replay import buffered_payload
    curve = PchipInterpolator([0., .25, 1., 2.], np.tile(np.arange(4)[:, None], (1, len(KEYS))), axis=0)
    p = buffered_payload(curve, 100., 5., 15., {'limits': {}})
    assert len(p['times']) == len(curve.x)
    path = Trajectory(p)
    for t in np.linspace(0, 2, 100):
        np.testing.assert_allclose(list(path.at(t).values()), curve(t), atol=1e-10)
