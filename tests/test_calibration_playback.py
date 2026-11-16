import argparse

import numpy as np
import pytest

from examples.calibration_playback import frame_at, load_recording, record_run
from examples.true_calibration import CalibrationRun, playback_speed


def test_recording_preserves_physics_states_and_rejects_a_different_scene(tmp_path):
    run = CalibrationRun(arms='left', include_lift=False)
    path = tmp_path / 'calibration.npz'
    record_run(run, path, 'original-scene')
    recording = load_recording(path, 'original-scene')
    assert recording['report']['complete']
    assert recording['time'][-1] == pytest.approx(recording['report']['duration_s'])
    assert recording['time'][0] == 0
    assert len(recording['time']) > 100
    # Displaying any saved pose, in any order, never integrates new physics.
    for index in (0, len(recording['time']) // 2, len(recording['time']) - 1, 0):
        frame, complete = frame_at(recording, run.sim, recording['time'][index])
        np.testing.assert_allclose(run.sim.data.qpos, recording['qpos'][index], atol=1e-14)
        np.testing.assert_array_equal(run.sim.data.qvel, recording['qvel'][index])
        assert complete == (index == len(recording['time']) - 1)
    with pytest.raises(ValueError, match='does not match'):
        load_recording(path, 'changed-geometry')


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_invalid_playback_speeds_are_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        playback_speed(value)
