from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from examples.true_calibration import CalibrationRun, calibration_physics_step, contact_snapshot, scene_model
from integrations.calibration_controller import controller
from sourccey.build import MODEL, ROLES
from sourccey.collision import bearing_recess
from sourccey.simulation import Simulation


class TestClock:
    __test__ = False

    def __init__(self):
        self.elapsed = 0.
        self.waits = []

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.elapsed += seconds


class ScriptBus:
    def __init__(self, reverse):
        self.motors = dict.fromkeys(ROLES)
        self.positions = dict.fromkeys(ROLES, 2048)
        self.positions['shoulder_lift'] = 296 if reverse else 3800
        self.goals = self.positions.copy()
        self.writes = []
        self.retry = True

    def disable_torque(self):
        pass

    def enable_torque(self):
        pass

    def sync_read(self, register, normalize=False):
        return self.positions.copy()

    def read(self, register, motor, normalize=False):
        if register == 'Present_Position':
            return self.positions[motor]
        # Exercise the read retry and overload branches as well as normal limits.
        if motor == 'shoulder_pan' and self.retry:
            self.retry = False
            raise RuntimeError('transient read error')
        if motor == 'gripper' and self.goals[motor] != self.positions[motor]:
            raise RuntimeError('Overload error')
        return 1001 if self.goals[motor] != self.positions[motor] else 0

    def write(self, register, motor, value, normalize=False):
        self.writes.append((register, motor, value, normalize))
        self.goals[motor] = value
        self.positions[motor] = int(np.clip(value, 1500, 2600))


@pytest.mark.parametrize('reverse', [False, True])
def test_cooperative_controller_matches_original_raw_commands_and_timing(reverse):
    traces = []
    for cooperative in (False, True):
        clock, bus = TestClock(), ScriptBus(reverse)
        cls = controller('arm_motion.py.txt', clock, cooperative)['SourcceyFollowerCalibrator']
        calibrator = cls()
        calibrator.robot = SimpleNamespace(bus=bus, config=SimpleNamespace(
            orientation='right' if reverse else 'left', max_current_calibration_threshold=75))
        result = calibrator._detect_mechanical_limits(reverse)
        if cooperative:
            try:
                while True:
                    clock.sleep(next(result))
            except StopIteration as end:
                result = end.value
        traces.append((result, bus.writes, clock.waits, clock.elapsed))
    assert traces[0] == traces[1]
    # A three-second return is 31 linear goal writes at 10 Hz.
    writes = traces[0][1]
    assert len(writes) > 300
    assert all(register == 'Goal_Position' and normalize is False for register, _, _, normalize in writes)
    assert traces[0][0]['max' if reverse else 'min']['gripper'] == 2048


def test_scene_keeps_production_joint_stops_and_probe_only_frees_grippers():
    original = mujoco.MjModel.from_xml_path(str(MODEL))
    baseline, probe = scene_model(), scene_model(free_gripper=True)
    np.testing.assert_array_equal(baseline.jnt_range, original.jnt_range)
    np.testing.assert_array_equal(baseline.jnt_limited, original.jnt_limited)
    changed = np.flatnonzero(probe.jnt_limited != original.jnt_limited)
    assert set(changed) == {original.joint(f'{s}_gripper').id for s in ('left', 'right')}
    assert all(not baseline.actuator_ctrllimited[baseline.actuator(f'{s}_gripper').id]
               for s in ('left', 'right'))


def test_full_physics_calibration_completes_with_both_arms_and_lift():
    run = CalibrationRun()
    while not run.complete:
        run.step()
    report = run.report()
    assert set(report['detected_raw_ranges']) == {'lift', 'left', 'right'}
    lift = report['detected_raw_ranges']['lift']
    assert lift['raw_max'] - lift['raw_min'] > 900
    for side in ('left', 'right'):
        assert set(report['detected_raw_ranges'][side]['min']) == set(ROLES)
        stops = [e for e in run.events if e['arm'] == side and e['joint'] == 'gripper']
        assert len(stops) == 1
        assert stops[0]['cause'] == 'model_joint_limit'
        assert stops[0]['angle_deg'] == pytest.approx(60, abs=2)
    assert not report['hardware_commands_sent']
    assert not any(w.number for w in run.sim.data.warning)
    assert len(run.searches) == 20
    assert not any(s['outcome'] == 'running' for s in run.searches)


def test_freed_gripper_reaches_search_cap_without_a_fake_hand_stop():
    run = CalibrationRun(arms='left', free_gripper=True, include_lift=False)
    while not run.complete:
        run.step()
    assert not [e for e in run.events if e['joint'] == 'gripper']
    detected = run.results['left']
    # floor(1664 / 50) raw steps; no old 65-degree normalization is used.
    assert detected['max']['gripper'] - detected['min']['gripper'] > 1600
    grip_search = next(s for s in run.searches if s['joint'] == 'gripper')
    assert grip_search['outcome'] == 'search_range_exhausted'


def test_self_collision_probe_removes_arm_stops_and_generates_real_contact_constraints():
    run = CalibrationRun(include_lift=False, self_collision=True, free_arms=True)
    model, data = run.sim.model, run.sim.data
    assert all(not model.jnt_limited[model.joint(f'{side}_{role}').id]
               for side in ('left', 'right') for role in ROLES)
    assert model.joint('linear_actuator').limited[0]
    assert model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    self_contacts = [c for c in run.initial_contacts if c['robot_self_contact']]
    assert self_contacts
    # The existing assembled CAD hulls already intersect before any command.
    assert any(c['distance_mm'] < -10 for c in self_contacts)
    assert any(c['normal_force_n'] > 1 for c in self_contacts)
    for _ in range(150):
        run.step()
    assert all(event['cause'] != 'model_joint_limit' for event in run.events)
    assert any(c['robot_self_contact'] for c in contact_snapshot(model, data))
    assert not any(w.number for w in data.warning)
    assert run.report()['self_collision_enabled']
    assert run.report()['arm_angle_limits_removed']


def test_split_calibration_dynamics_match_existing_simulation_at_joint_stop():
    normal, split = Simulation(traction=False), Simulation(traction=False)
    for sim in (normal, split):
        # Includes contact with a joint stop, not just unconstrained motion.
        sim.model.actuator_ctrllimited[sim.model.actuator('left_gripper').id] = False
        sim.data.ctrl[sim.model.actuator('left_gripper').id] = np.deg2rad(75)
        sim.set_joints({'left_elbow_flex': -1., 'left_wrist_flex': .7})
    for i in range(350):
        normal.step()
        if i % 50 == 0:
            mujoco.mj_forward(split.model, split.data)
        calibration_physics_step(split)
    np.testing.assert_allclose(normal.data.qpos, split.data.qpos, atol=1e-9)
    np.testing.assert_allclose(normal.data.qvel, split.data.qvel, atol=1e-8)


def test_decomposed_cad_removes_deep_false_overlaps_without_changing_robot_dynamics():
    raw = CalibrationRun(include_lift=False, self_collision=True, free_arms=True)
    rebuilt = CalibrationRun(include_lift=False, self_collision=True, free_arms=True,
                             collision_model='decomposed')
    raw_depth = max(-c['distance_mm'] for c in raw.initial_contacts)
    new_depth = max(-c['distance_mm'] for c in rebuilt.initial_contacts)
    assert raw_depth > 25
    assert new_depth < 2
    assert max(c['normal_force_n'] for c in rebuilt.initial_contacts) < 500
    assert raw.sim.model.njnt == rebuilt.sim.model.njnt
    assert raw.sim.model.nu == rebuilt.sim.model.nu
    np.testing.assert_allclose(raw.sim.model.body_mass, rebuilt.sim.model.body_mass)
    np.testing.assert_allclose(raw.sim.model.body_inertia, rebuilt.sim.model.body_inertia)
    np.testing.assert_allclose(raw.sim.data.qpos, rebuilt.sim.data.qpos)
    # The whole-part collision duplicates stay disabled, including in hull view.
    old_geoms = [g for g in range(rebuilt.sim.model.ngeom)
                 if rebuilt.sim.model.geom(g).name.startswith('collision_')]
    assert all(rebuilt.sim.model.geom_contype[g] == 0 and rebuilt.sim.model.geom_conaffinity[g] == 0
               for g in old_geoms)
    for _ in range(25):
        rebuilt.step()
    assert not any(w.number for w in rebuilt.sim.data.warning)


def test_torso_collision_is_clear_at_start_and_blocks_the_inward_pan_poses():
    run = CalibrationRun(include_lift=False, self_collision=True, free_arms=True,
                         collision_model='decomposed')
    model, data = run.sim.model, run.sim.data
    body_geoms = {model.geom('calibration_torso_envelope').id,
                  model.geom('calibration_head_envelope').id}
    assert not any(int(c.geom1) in body_geoms or int(c.geom2) in body_geoms
                   for c in data.contact[:data.ncon])
    for side in ('left', 'right'):
        mujoco.mj_resetDataKeyframe(model, data, model.key('startup').id)
        data.qpos[model.joint(side + '_shoulder_pan').qposadr[0]] = np.deg2rad(144)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        assert any(int(c.geom1) in body_geoms or int(c.geom2) in body_geoms
                   for c in data.contact[:data.ncon])


def test_bearing_recess_preserves_outer_solid_without_filling_the_bore():
    from itertools import product
    from scipy.spatial import ConvexHull
    box = np.array(list(product((-1., 1.), repeat=3)))
    parts = bearing_recess(box, np.zeros(3), np.array([0., 0., 1.]), (-.2, .3, .5))
    volume = sum(ConvexHull(p).volume for p in parts)
    bore_volume = 32 * .5**2 * np.tan(np.pi / 32) * .5
    assert volume == pytest.approx(8 - bore_volume, abs=1e-9)
    # The centre is outside EVERY retained convex collider. A single new hull
    # around all retained vertices would silently close the clearance again.
    assert all(np.any(ConvexHull(p).equations[:, 3] > 1e-8) for p in parts)
    for corner in box:
        assert any(np.all(ConvexHull(p).equations[:, :3] @ corner +
                             ConvexHull(p).equations[:, 3] <= 1e-8) for p in parts)


@pytest.fixture(scope='module')
def bearing_scene():
    return Simulation(model=scene_model(self_collision=True, free_arms=True,
                                         collision_model='decomposed'), traction=False)


def reset_bearing_scene(sim):
    sim.reset()
    # True Calibration holds the shoulders near this pose while testing the
    # elbow and wrist. Test joints directly, without running the calibration.
    for side in ('left', 'right'):
        name = side + '_shoulder_lift'
        sim.data.qpos[sim.model.joint(name).qposadr[0]] = 0
        sim.data.ctrl[sim.model.actuator(name).id] = 0
    mujoco.mj_forward(sim.model, sim.data)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('role,angles', [('elbow_flex', [-120, -60, 0, 60, 120]),
                                        ('wrist_roll', [-54, -30, 0, 30, 54])])
def test_mating_bearings_no_longer_block_joint_travel(bearing_scene, side, role, angles):
    sim = bearing_scene
    reset_bearing_scene(sim)
    joint = sim.model.joint(f'{side}_{role}')
    for angle in angles:
        sim.data.qpos[joint.qposadr[0]] = np.deg2rad(angle)
        sim.data.ctrl[sim.model.actuator(f'{side}_{role}').id] = np.deg2rad(angle)
        mujoco.mj_forward(sim.model, sim.data)
        assert not contact_snapshot(sim.model, sim.data, joint.id)
        assert abs(sim.data.qfrc_constraint[joint.dofadr[0]]) < .1


@pytest.mark.parametrize('side', ['left', 'right'])
def test_bearing_clearance_keeps_external_elbow_and_gripper_stops(bearing_scene, side):
    sim = bearing_scene
    for role, angle, bodies in (
            ('elbow_flex', 140, ('Bicep_Right', 'Forearm')),
            ('elbow_flex', -140, ('Bicep_Right', 'Forearm')),
            ('gripper', 135, ('Gripper_Base', 'Gripper_Finger'))):
        reset_bearing_scene(sim)
        joint = sim.model.joint(f'{side}_{role}')
        sim.data.qpos[joint.qposadr[0]] = np.deg2rad(angle)
        mujoco.mj_forward(sim.model, sim.data)
        assert any(c['body1'].startswith(bodies[0]) and c['body2'].startswith(bodies[1])
                   and c['distance_mm'] < -.1 for c in contact_snapshot(sim.model, sim.data, joint.id))


@pytest.mark.parametrize('direction', [-1, 1])
def test_cleared_joints_follow_servo_goals_without_false_current_stop(bearing_scene, direction):
    sim = bearing_scene
    reset_bearing_scene(sim)
    goals = {f'{side}_{role}': np.deg2rad(direction * mirror * angle)
             for side, mirror in (('left', 1), ('right', -1))
             for role, angle in (('elbow_flex', 90), ('wrist_roll', 40))}
    for name, goal in goals.items():
        sim.data.ctrl[sim.model.actuator(name).id] = goal
    # A short isolated servo check, not a calibration run or viewer playback.
    for _ in range(400):
        calibration_physics_step(sim)
    mujoco.mj_forward(sim.model, sim.data)
    for name, goal in goals.items():
        joint = sim.model.joint(name)
        assert sim.data.qpos[joint.qposadr[0]] == pytest.approx(goal, abs=np.deg2rad(1))
        assert abs(sim.data.qfrc_constraint[joint.dofadr[0]]) < .1
    assert not any(w.number for w in sim.data.warning)
