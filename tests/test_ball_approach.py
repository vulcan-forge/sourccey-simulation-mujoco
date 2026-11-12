import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from examples.ball_approach import TableContactMonitor, frame, grasp_point, run, scene_model
from sourccey.build import ROLES
from sourccey.simulation import Simulation


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "examples/ball_approach.example.json").read_text())


def test_concept_approach_and_contact_lift(config):
    def measured_run(speed):
        peak = 0.

        def inspect(sim):
            nonlocal peak
            dofs = [sim.model.joint(f"left_{r}").dofadr[0] for r in ROLES[:5]]
            peak = max(peak, float(np.max(np.abs(sim.data.qvel[dofs]))))

        result = run(config, speed=speed, on_step=inspect)
        return result, peak

    report, peak_normal = measured_run(1)
    assert report["simulation_passed"]
    assert report["approach_error_mm"] < 1
    assert report["max_base_translation_mm"] < 1
    assert report["pickup"]["passed"]
    assert report["pickup"]["lift_mm"] > 80
    assert report["pickup"]["held_pad_contacts"] == ["fixed", "moving"]
    assert report["table_clearance"]["robot_table_contact_steps"] == 0
    assert report["table_clearance"]["minimum_pad_clearance_mm"] > 3
    assert report["real_comparison"] is None
    assert not report["benchmark_complete"]
    slow, peak_slow = measured_run(.5)
    assert slow["simulation_passed"] and slow["pickup"]["passed"]
    assert slow["table_clearance"]["robot_table_contact_steps"] == 0
    assert 1.8 * report["simulated_seconds"] < slow["simulated_seconds"] < 2 * report["simulated_seconds"]
    assert slow["pickup"]["hold_seconds"] == report["pickup"]["hold_seconds"]
    assert .45 < peak_slow / peak_normal < .55


@pytest.mark.parametrize("speed", [0, -1, 1.1, float("nan"), float("inf")])
def test_invalid_motion_speed_rejected(config, speed):
    with pytest.raises(ValueError):
        run(config, speed=speed)


def test_reduced_wrist_motion_limits_preserve_start_and_bound_ik(config):
    config["joint_motion_limits_rad"] = {"wrist_roll": [-.8, .9]}
    model = scene_model(config)
    assert model.joint("left_wrist_roll").range == pytest.approx([-.8, .9])
    config["joint_motion_limits_rad"] = {"wrist_roll": [.1, .9]}
    del config["arm_pose_rad"]["wrist_roll"]
    with pytest.raises(ValueError, match="Startup pose outside"):
        scene_model(config)


def test_measurement_uses_observed_pose_fk(config):
    config["pickup"] = False
    sim = Simulation(model=scene_model(config))
    origin, rotation = frame(sim, config)
    # Deliberately use a different observed pose from the approach endpoint.
    expected = rotation.T @ (grasp_point(sim.model, sim.data) - origin) * 1000
    measurement = {
        "side": "left", "datum_description": config["datum_description"],
        "ball_center_mm": config["ball_center_mm"], "measured_at": "synthetic test",
        "measurement_method": "synthetic FK plus known offset", "joint_mapping_notes": "test only",
        "grasp_point_mm": (expected + [3, 4, 0]).tolist(),
        "hardware_positions": {f"left_{r}.pos": 0 for r in ROLES},
        "mujoco_joint_rad": {r: float(sim.data.qpos[sim.model.joint(f"left_{r}").qposadr[0]]) for r in ROLES},
    }
    comparison = run(config, measurement)["real_comparison"]
    # Fixture compliance changes the world pose by micrometers.
    np.testing.assert_allclose(comparison["real_minus_sim_mm"], [3, 4, 0], atol=.03)
    assert comparison["position_error_mm"] == pytest.approx(5, abs=.03)
    invalid = copy.deepcopy(measurement)
    invalid["ball_center_mm"] = [0, 0, 0]
    with pytest.raises(ValueError, match="same marked ball"):
        run(config, invalid)


@pytest.mark.parametrize("field,value", [("ball_diameter_mm", -1),
                                               ("ball_center_mm", [500, 180, 800]),
                                               ("datum_offset_mm", [0, float('nan'), 0]),
                                               ("elevator_m", 10)])
def test_invalid_scene_rejected(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        scene_model(config)


def test_only_supports_are_constrained(config):
    model = scene_model(config)
    assert model.neq == 2
    assert model.joint("benchmark_ball_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.eq("benchmark_fixed_base").type == mujoco.mjtEq.mjEQ_WELD
    assert model.eq("benchmark_fixed_elevator").type == mujoco.mjtEq.mjEQ_JOINT


def test_table_monitor_ignores_ball_support_but_detects_finger_collision(config):
    sim = Simulation(model=scene_model(config), traction=False)
    sim.step(300)
    monitor = TableContactMonitor(sim.model)
    monitor.observe(sim)
    assert monitor.contact_steps == 0
    assert sim.data.ncon > 0  # Includes the ball resting on the table.
    table = sim.model.geom("benchmark_table").id
    pad = sim.model.geom("benchmark_fixed_pad").id
    sim.model.geom_pos[table] = sim.data.geom_xpos[pad]
    mujoco.mj_forward(sim.model, sim.data)
    monitor.observe(sim)
    assert monitor.contact_steps == 1
    assert "benchmark_fixed_pad" in monitor.contact_geoms
    assert monitor.max_penetration > 0
