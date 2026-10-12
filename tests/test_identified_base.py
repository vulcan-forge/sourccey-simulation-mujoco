"""Hardware response regressions and contact-driven simulation checks."""
import json
from pathlib import Path

import numpy as np
import pytest

from sourccey.build import WHEELS
from sourccey.control import IdentifiedBaseResponse, wheel_rates_from_velocity
from sourccey.simulation import Simulation


def test_measured_strafe_holds_are_reproduced_without_averaging_battery_conditions():
    response = IdentifiedBaseResponse()
    measurements = json.loads((Path(__file__).parents[1] / "sourccey/identified_base.json").read_text())
    holds = dict(measurements["provenance"]["y"]["holds"])
    for level in (.5, .6, .7, .8, .9, 1.):
        expected = (holds[level] - holds[-level]) / 2
        assert response.steady_velocity([0, level, 0])[1] == pytest.approx(expected)
        assert response.steady_velocity([0, -level, 0])[1] == pytest.approx(-expected)
    # Independent measured end points and directional asymmetry on the other axes.
    assert response.steady_velocity([1, 1, 1]) == pytest.approx([.3197, .2587452277, 1.018], abs=5e-5)
    assert response.steady_velocity([-1, 0, -1]) == pytest.approx([-.3028, 0, -1.043], abs=5e-5)
    np.testing.assert_array_equal(response.steady_velocity([.4, -.4, .4]), np.zeros(3))


@pytest.mark.parametrize("dt", [.002, .01, .025])
def test_delay_and_time_constant_follow_elapsed_time(dt):
    response = IdentifiedBaseResponse()
    for _ in range(round(.1/dt)):
        v = response.step([0, 1, 0], dt)
    assert v[1] == pytest.approx(0, abs=1e-12)
    for _ in range(round(.1/dt)):
        v = response.step([0, 1, 0], dt)
    assert v[1] == pytest.approx(.25874522769983815*(1-np.exp(-1)), abs=1e-12)


def test_short_pulse_and_reset_clear_delayed_commands():
    response = IdentifiedBaseResponse()
    for _ in range(5):
        response.step([0, 1, 0], .01)
    for _ in range(5):
        response.step([0, 0, 0], .01)
    assert response.velocity[1] == pytest.approx(0, abs=1e-12)
    # A pulse shorter than the delay still arrives; it must not be overwritten.
    for _ in range(5):
        response.step([0, 0, 0], .01)
    peak = response.velocity[1]
    assert peak > .09
    for _ in range(100):
        response.step([0, 0, 0], .01)
    assert response.velocity[1] < 1e-5
    response.step([0, 1, 0], .01)
    response.reset()
    for _ in range(30):
        np.testing.assert_array_equal(response.step([0, 0, 0], .01), np.zeros(3))


@pytest.mark.parametrize("command,dt", [([np.nan, 0, 0], .01), ([0, 0, 0], 0),
                                      ([0, 0, 0], -.1), ([0, 0, 0], np.inf)])
def test_invalid_updates_do_not_mutate_state(command, dt):
    response = IdentifiedBaseResponse()
    with pytest.raises(ValueError):
        response.step(command, dt)
    assert response.time == 0
    np.testing.assert_array_equal(response.step([0, 0, 0], .2), np.zeros(3))


@pytest.mark.parametrize("level", [-1., 1.])
def test_full_strafe_reaches_measured_speed_through_contacts(level):
    sim = Simulation(base_response="identified")
    sim.step(250)
    sim.set_base(left=level)
    sim.step(1500)
    start = sim.data.qpos[:2].copy()
    sim.step(500)
    velocity = sim.data.qpos[:2] - start  # one simulated second
    assert velocity[1] == pytest.approx(level * .2587452277, abs=.025)
    assert abs(velocity[0]) < .02
    assert not sim.state()["warnings"]
    sim.set_base()
    sim.step(750)
    assert np.linalg.norm(sim.data.qvel[:2]) < .01
    sim.set_base(left=1)
    sim.step(10)
    sim.reset()
    sim.step(100)
    np.testing.assert_array_equal(sim.base_response.velocity, np.zeros(3))
    assert all(sim.data.ctrl[sim.actuators[name]] == 0 for name in WHEELS)


def test_identified_mode_does_not_bypass_traction_or_teleport():
    sim = Simulation(base_response="identified", traction=False)
    sim.step(250)
    before = sim.data.qpos.copy()
    sim.set_base(left=1)
    np.testing.assert_array_equal(sim.data.qpos, before)
    sim.step(1000)
    assert np.linalg.norm(sim.data.qpos[:2]-before[:2]) < .02
    rates = [sim.data.qvel[sim.model.jnt_dofadr[sim.joints[name]]] for name in WHEELS]
    assert min(np.abs(rates)) > 4
    np.testing.assert_array_equal(np.sign(rates), np.sign(wheel_rates_from_velocity([0, 1, 0])))


def test_unknown_response_mode_is_rejected():
    with pytest.raises(ValueError, match="base_response"):
        Simulation(base_response="typo")
