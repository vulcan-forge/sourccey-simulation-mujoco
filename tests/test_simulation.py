import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from sourccey.build import SOURCE, MODEL, CAD_ROTATION, ROLES, WHEELS, ELBOW_LOWER_DEGREES, ELEVATOR_UPPER_METERS
from sourccey.control import wheel_rates
from sourccey.simulation import Simulation
from sourccey.pose import DEFAULT_POSE


@pytest.fixture(scope="module")
def sim():
    return Simulation()


def test_source_inventory_and_full_geometry(sim):
    root = ET.parse(SOURCE).getroot()
    assert len(root.findall("link")) == 132
    assert len(root.findall("joint")) == 131
    meshes = {m.get("filename") for m in root.findall(".//mesh")}
    assert len(meshes) == 171
    assert all((SOURCE.parent / p).is_file() for p in meshes)
    assert sim.model.nmesh == 171
    expected = {f"{s}_{r}" for s in ("left", "right") for r in ROLES} | set(WHEELS) | {"linear_actuator"}
    assert set(sim.joints) == expected
    assert sim.model.njnt == 18 and sim.model.nu == 17  # 17 source + floating base
    assert sim.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert sim.model.body_mass.sum() == pytest.approx(169.041941, abs=1e-5)
    elevator = sim.model.jnt_bodyid[sim.joints["linear_actuator"]]
    for side in ("left", "right"):
        body = sim.model.jnt_bodyid[sim.joints[side + "_shoulder_pan"]]
        while body not in (0, elevator):
            body = sim.model.body_parentid[body]
        assert body == elevator


def test_generated_fk_matches_native_urdf_import(sim):
    """Random poses catch axis, scale, mesh origin and joint-frame conversion errors."""
    raw = mujoco.MjModel.from_xml_path(str(SOURCE))
    rd = mujoco.MjData(raw)
    sd = mujoco.MjData(sim.model)
    rng = np.random.default_rng(11)
    for _ in range(5):
        for name, jid in sim.joints.items():
            value = rng.uniform(-1, 1) if name in WHEELS else rng.uniform(*sim.model.jnt_range[jid])
            rd.qpos[raw.joint(name).qposadr[0]] = value
            sd.qpos[sim.model.jnt_qposadr[jid]] = value
        mujoco.mj_forward(raw, rd); mujoco.mj_forward(sim.model, sd)
        for name, jid in sim.joints.items():
            rid = raw.joint(name).id
            np.testing.assert_allclose(sd.xanchor[jid], CAD_ROTATION @ rd.xanchor[rid] + sd.qpos[:3], atol=1e-8)
            np.testing.assert_allclose(sd.xaxis[jid], CAD_ROTATION @ rd.xaxis[rid], atol=1e-8)


def test_wheel_signs():
    np.testing.assert_array_equal(np.sign(wheel_rates([1, 0, 0])), [1, -1, 1, -1])
    np.testing.assert_array_equal(np.sign(wheel_rates([0, 1, 0])), [-1, -1, 1, 1])
    np.testing.assert_array_equal(np.sign(wheel_rates([0, 0, 1])), [-1, -1, -1, -1])


@pytest.mark.parametrize("side", ["left", "right"])
def test_elbow_bends_down_beyond_source_limit(sim, side):
    name = side + "_elbow_flex"
    jid = sim.joints[name]
    assert np.rad2deg(sim.model.jnt_range[jid][0]) == pytest.approx(ELBOW_LOWER_DEGREES)
    assert sim.model.actuator_ctrlrange[sim.actuators[name], 0] == pytest.approx(sim.model.jnt_range[jid, 0])
    sim.reset()
    before = sim.data.site_xpos[sim.model.site(side + "_ee").id, 2]
    sim.set_joints({name: np.deg2rad(ELBOW_LOWER_DEGREES)})
    sim.step(1000)
    actual = sim.data.qpos[sim.model.jnt_qposadr[jid]]
    assert np.rad2deg(actual) == pytest.approx(ELBOW_LOWER_DEGREES, abs=1)
    after = sim.data.site_xpos[sim.model.site(side + "_ee").id, 2]
    # The raised -2.0 rad startup pose is already near the -135 degree stop.
    assert after < before - .05
    assert not sim.state()["warnings"]


@pytest.mark.parametrize("command,axis", [([.5, 0, 0], 0), ([0, .5, 0], 1), ([0, 0, .5], 2),
                                         ([-.5, 0, 0], 0), ([0, -.5, 0], 1), ([0, 0, -.5], 2)])
def test_physical_drive(sim, command, axis):
    sim.reset(); sim.step(250)
    start = sim.data.qpos[:3].copy()
    sim.set_base(*command); sim.step(1000)
    delta = sim.data.qpos[:3] - start
    yaw = np.arctan2(sim.robot_rotation[1, 0], sim.robot_rotation[0, 0])
    if axis < 2:
        assert .50 < delta[axis] * np.sign(command[axis]) < .65
        assert abs(delta[1-axis]) < .025
        assert abs(yaw) < .025
    else:
        assert 1.3 < yaw * np.sign(command[axis]) < 1.7
        assert np.linalg.norm(delta[:2]) < .025
    rates = [sim.data.qvel[sim.model.jnt_dofadr[sim.joints[n]]] for n in WHEELS]
    np.testing.assert_array_equal(np.sign(rates), np.sign(wheel_rates(command)))
    assert abs(delta[2]) < .005
    assert not sim.state()["warnings"]
    sim.set_base(); sim.step(500)
    assert np.linalg.norm(sim.data.qvel[:2]) < .02


def test_no_traction_means_no_driving():
    s = Simulation(traction=False); s.step(250)
    p = s.data.qpos[:3].copy()
    s.set_base(.5, 0, 0); s.step(1000)
    assert np.linalg.norm(s.data.qpos[:2]-p[:2]) < .02
    assert abs(s.data.qvel[s.model.jnt_dofadr[s.joints[WHEELS[0]]]]) > 5


def test_airborne_wheels_do_not_translate_base(sim):
    sim.reset(); sim.data.qpos[2] += 3
    mujoco.mj_forward(sim.model, sim.data)
    sim.set_base(.5, .5, .5); sim.step(50)
    assert sim.data.ncon == 0
    assert np.linalg.norm(sim.data.qpos[:2]) < .002


@pytest.mark.parametrize("full", [False, True])
def test_elevator_endstops_move_both_arms(full):
    s = Simulation(full_elevator_range=full)
    joint = s.joints["linear_actuator"]
    np.testing.assert_allclose(s.model.jnt_range[joint], [-.315, ELEVATOR_UPPER_METERS])
    np.testing.assert_allclose(s.model.actuator_ctrlrange[s.actuators["linear_actuator"]], [-.315, ELEVATOR_UPPER_METERS])
    s.set_joints({"linear_actuator": 0})
    assert s.data.ctrl[s.actuators["linear_actuator"]] == pytest.approx(ELEVATOR_UPPER_METERS)
    s.set_elevator(ELEVATOR_UPPER_METERS); s.step(1000)
    initial = {side: s.mount_position(side) - s.data.xpos[s.base_id] for side in ("left", "right")}
    bottom = -.315 if full else -.3104
    for position_m in (bottom, ELEVATOR_UPPER_METERS):
        s.set_elevator(position_m); s.step(1500)
        q = s.data.qpos[s.model.jnt_qposadr[s.joints["linear_actuator"]]]
        assert abs(q - position_m) < .001
        expected = position_m - ELEVATOR_UPPER_METERS
        for side in initial:
            diff = s.mount_position(side) - s.data.xpos[s.base_id] - initial[side]
            assert abs(diff[2]-expected) < .002
        assert not s.state()["warnings"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_position_drives_limits_and_midpoints(sim, side):
    for role in ROLES:
        name = side + "_" + role
        j = sim.joints[name]
        low, high = sim.model.jnt_range[j]
        for value in (low, (low+high)/2, high):
            sim.reset(); sim.set_joints({name: value}); sim.step(1000)
            assert abs(sim.data.qpos[sim.model.jnt_qposadr[j]]-value) < .015, (name, value)
            assert not sim.state()["warnings"]
            assert np.all(np.abs(sim.data.actuator_force) <= sim.model.actuator_forcerange[:, 1] + 1e-8)


@pytest.mark.parametrize("side", ["left", "right"])
def test_ik_follows_full_chain_and_unreachable_is_bounded(sim, side):
    sim.reset(); sim.step(100)
    offset = [.02, .01, -.02] if side == "left" else [.02, -.01, -.02]
    target = sim.ee_local(side) + offset
    before = sim.data.qpos.copy()
    result = sim.solve_ik(side, target)
    assert result["converged"]
    np.testing.assert_array_equal(sim.data.qpos, before)  # Solver must not teleport live robot.
    sim.step(750)
    assert np.linalg.norm(sim.ee_local(side)-target) < .003
    result = sim.solve_ik(side, [10, 0, 0])
    assert not result["converged"]
    for role in ROLES:
        name = side + "_" + role
        assert sim.model.jnt_range[sim.joints[name], 0] <= sim.data.ctrl[sim.actuators[name]] <= sim.model.jnt_range[sim.joints[name], 1]


def test_screenshot_pose_is_startup_and_reset_keyframe(sim):
    sim.reset()
    assert sim.model.nkey == 1
    assert sim.model.key("startup").id == 0
    for name, value in DEFAULT_POSE.items():
        assert sim.data.ctrl[sim.actuators[name]] == pytest.approx(value, abs=1e-8)
        assert sim.data.qpos[sim.model.jnt_qposadr[sim.joints[name]]] == pytest.approx(
            0 if name in WHEELS else value, abs=1e-8)
    left, right = sim.ee_local("left"), sim.ee_local("right")
    np.testing.assert_allclose(left, right * [1, -1, 1], atol=.015)
    sim.step(500)
    assert not sim.state()["warnings"]
    for name, value in DEFAULT_POSE.items():
        if name not in WHEELS:
            assert sim.data.qpos[sim.model.jnt_qposadr[sim.joints[name]]] == pytest.approx(value, abs=.015)
    sim.set_joints({"left_shoulder_lift": 0})
    sim.step(200)
    sim.reset()
    assert sim.data.ctrl[sim.actuators["left_shoulder_lift"]] == DEFAULT_POSE["left_shoulder_lift"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_gripper_pads_have_closed_clearance_and_open_positive(sim, side):
    from sourccey.validate import pad_clearance
    closed = pad_clearance(sim, side, -5)
    zero = pad_clearance(sim, side, 0)
    opened = pad_clearance(sim, side, 60)
    assert .0008 < closed < .002
    assert closed < zero < opened
    assert opened > .03


def test_invalid_inputs_are_atomic(sim):
    sim.reset(); before = sim.data.ctrl.copy()
    with pytest.raises(ValueError):
        sim.set_joints({"left_elbow_flex": 0, "right_elbow_flex": np.nan})
    np.testing.assert_array_equal(sim.data.ctrl, before)
    with pytest.raises(ValueError):
        sim.set_base(np.inf)
    with pytest.raises(ValueError):
        sim.set_elevator(float("nan"))
    np.testing.assert_array_equal(sim.data.ctrl, before)
