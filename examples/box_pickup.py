"""Test A: fixed-pose cardboard box pickup on the measured benchmark table.

Simulation only. The object remains free; only base/elevator fixtures are used.
Run `python -m examples.box_pickup --view` to compute and play the experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from sourccey.build import MODEL, ROOT, ROLES, WHEELS
from sourccey.collision import add_torso_collision, decomposed_collision, torso_envelopes
from sourccey.simulation import Simulation


DEFAULT_SCENE = ROOT / 'examples/box_pickup_a.json'
OUTPUT = ROOT / 'artifacts/box_pickup_a'


def load_scene(path=DEFAULT_SCENE):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    config['object'] = json.loads((ROOT / config['object_profile']).read_text(encoding='utf-8'))
    validate(config)
    return config


def validate(config):
    if config['version'] != 1 or config['test'] not in ('A', 'B') or config['side'] != 'left':
        raise ValueError('This scene implements fixed-position, left-arm tests A and B')
    if not config['robot_setup']['base_fixed'] or not config['robot_setup']['elevator_fixed']:
        raise ValueError('Test A requires fixed base and elevator')
    if config['robot_setup']['yaw_relative_to_table_deg'] != 0:
        raise ValueError('This first table placement requires square robot alignment')
    if config['test'] == 'A' and config['box_pose']['yaw_deg'] != 0:
        raise ValueError('This baseline implements the fixed zero-degree box orientation')
    if config['test'] == 'B' and abs(config['box_pose']['yaw_deg']) > 15:
        raise ValueError('The initial B curriculum is limited to +/-15 degrees')
    values = [config['object']['mass_kg'], *(config['object']['dimensions_mm'][k]
              for k in ('width', 'depth_front_to_back', 'height')),
              *(config['table'][k] for k in ('length_across_mm', 'depth_mm',
                'height_above_floor_mm', 'thickness_mm', 'chest_to_near_edge_mm')),
              config['grasp']['max_gripper_torque_nm'], config['grasp']['sliding_friction'],
              config['success']['hold_seconds'], config['grasp']['lift_distance_mm'],
              config['grasp']['approach_clearance_mm'], config['grasp']['pregrasp_backoff_mm']]
    if any(not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Dimensions, mass, contact settings and durations must be positive and finite')
    for value in (*config['grasp']['fixed_finger_contact_local_mm'],
                  *config['box_pose'].values()):
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError('Placement and landmark must be finite')
    e = config['object']['contact_estimates']['coefficient_of_restitution']
    if not 0 < e < 1:
        raise ValueError('Restitution estimate must be between zero and one')
    if not (0 <= config['grasp']['fixed_finger_clearance_mm'] < config['grasp']['pregrasp_backoff_mm']
            and -5 <= config['grasp']['closed_command_deg'] <= config['grasp']['open_angle_deg'] <= 60):
        raise ValueError('Invalid finger clearance or gripper command angles')
    if config['table']['thickness_mm'] >= config['table']['height_above_floor_mm']:
        raise ValueError('Table thickness must be less than its height')


def layout(config):
    """Floor-based coordinates; chest datum is a CAD envelope ray intersection."""
    source = Simulation(traction=False)
    if not source.model.joint('linear_actuator').range[0] <= config['robot_setup']['elevator_m'] <= source.model.joint('linear_actuator').range[1]:
        raise ValueError('Elevator position outside model travel')
    table = config['table']
    top = table['height_above_floor_mm'] / 1000
    points = torso_envelopes(MODEL.stat().st_mtime_ns)['torso']['vertices']
    points = points @ source.data.xmat[source.base_id].reshape(3, 3).T + source.data.xpos[source.base_id]
    equations = ConvexHull(points).equations
    upper = [-(c*top+k)/a for a, b, c, k in equations if a > 1e-8]
    chest = min(upper)
    if np.any(equations[:, :3] @ [chest, 0, top] + equations[:, 3] > 1e-7):
        raise ValueError('Tabletop height does not intersect the middle CAD chest envelope')
    edge = chest + table['chest_to_near_edge_mm'] / 1000
    dims = config['object']['dimensions_mm']
    half = np.array([dims['depth_front_to_back'], dims['width'], dims['height']]) / 2000
    pose = config['box_pose']
    center = np.array([edge + pose['center_from_near_edge_mm']/1000,
                       pose['center_left_of_robot_mm']/1000, top+half[2]])
    yaw = np.deg2rad(pose['yaw_deg'])
    lateral_extent = abs(np.sin(yaw))*half[0] + abs(np.cos(yaw))*half[1]
    depth_extent = abs(np.cos(yaw))*half[0] + abs(np.sin(yaw))*half[1]
    table_y = -config['robot_setup']['lateral_offset_from_table_center_mm']/1000
    if not (edge+depth_extent < center[0] < edge+table['depth_mm']/1000-depth_extent
            and abs(center[1]-table_y)+lateral_extent < table['length_across_mm']/2000):
        raise ValueError('Box footprint must lie wholly on the tabletop')
    return dict(chest_x_m=chest, near_edge_x_m=edge, tabletop_z_m=top,
                table_center_y_m=table_y, box_center_m=center.tolist(), box_half_size_m=half.tolist())


def scene_model(config, *, display_only=False):
    measured = layout(config)
    spec = mujoco.MjSpec.from_file(str(MODEL))
    spec.modelname = f'Sourccey Box Pickup - Test {config["test"]} (simulation only)'
    spec.memory = 96 * 1024 * 1024
    if display_only:
        # Recorded-state rendering needs the identical visible CAD, not the
        # thousands of hidden convex pieces used by the physics experiment.
        for geom in list(spec.geoms):
            if geom.name.startswith('collision_'):
                spec.delete(geom)
    else:
        decomposed_collision(spec)
        add_torso_collision(spec)
    # Standard adjacent-body filtering remains active. Nonadjacent arm/body,
    # arm/arm and robot/environment collisions are enabled.
    for geom in spec.geoms:
        if geom.contype & 2:
            geom.conaffinity |= 2
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 5
    spec.add_equality(name='box_test_fixed_base', type=mujoco.mjtEq.mjEQ_WELD,
                      name1='base_link', name2='world', objtype=mujoco.mjtObj.mjOBJ_BODY,
                      solref=[.002, 1], solimp=[.999, .9999, .001, .5, 2])
    elevator = config['robot_setup']['elevator_m']
    spec.add_equality(name='box_test_fixed_elevator', type=mujoco.mjtEq.mjEQ_JOINT,
                      name1='linear_actuator', objtype=mujoco.mjtObj.mjOBJ_JOINT,
                      data=[elevator]+[0]*10, solref=[.002, 1], solimp=[.999, .9999, .001, .5, 2])
    table = config['table']
    depth, across, thick = (table[k]/1000 for k in ('depth_mm', 'length_across_mm', 'thickness_mm'))
    edge, top, cy = measured['near_edge_x_m'], measured['tabletop_z_m'], measured['table_center_y_m']
    spec.worldbody.add_geom(name='box_test_table', type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[edge+depth/2, cy, top-thick/2], size=[depth/2, across/2, thick/2],
        contype=1, conaffinity=6, rgba=[.47, .32, .20, 1],
        friction=[table['sliding_friction'], .001, .0001])
    for i, x in enumerate((edge+.06, edge+depth-.06)):
        for j, y in enumerate((cy-across/2+.06, cy+across/2-.06)):
            spec.worldbody.add_geom(name=f'box_test_leg_{i}_{j}', type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, (top-thick)/2], size=[.02, .02, (top-thick)/2],
                contype=1, conaffinity=6, rgba=[.24, .25, .26, 1])
    half = np.asarray(measured['box_half_size_m'])
    mass = config['object']['mass_kg']
    full = 2*half
    inertia = mass/12*np.array([full[1]**2+full[2]**2, full[0]**2+full[2]**2, full[0]**2+full[1]**2])
    yaw = np.deg2rad(config['box_pose']['yaw_deg'])
    box = spec.worldbody.add_body(name='pickup_box', pos=measured['box_center_m'],
        quat=[np.cos(yaw/2), 0, 0, np.sin(yaw/2)], mass=mass, ipos=[0, 0, 0], inertia=inertia)
    box.add_freejoint(name='pickup_box_free')
    # A compliant contact damping estimate, not an exact restitution identity.
    log_e = np.log(config['object']['contact_estimates']['coefficient_of_restitution'])
    damping = -log_e/np.sqrt(np.pi**2+log_e**2)
    box.add_geom(name='pickup_box_geom', type=mujoco.mjtGeom.mjGEOM_BOX, size=half,
        mass=mass, contype=4, conaffinity=3, condim=4, priority=1,
        friction=[config['grasp']['sliding_friction'], config['grasp']['torsional_friction_m'],
                  config['grasp']['rolling_friction_m']], solref=[.006, damping],
        solimp=[.95, .99, .001, .5, 2], rgba=[.70, .49, .28, 1])
    # Explicit pair prevents gripper friction being applied to the table.
    spec.add_pair(name='box_table_contact', geomname1='pickup_box_geom', geomname2='box_test_table',
        condim=3, friction=[table['sliding_friction']]*2+[0, 0, 0], solref=[.006, damping])
    # Thin visual-only tape on top gives the box a visible orientation cue.
    box.add_geom(name='box_sealing_tape', type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, half[2]+.0001], size=[half[0], .005, .0001],
        contype=0, conaffinity=0, density=0, rgba=[.84, .71, .47, 1])
    spec.body('Gripper_Base_v1_1').add_site(name='box_fixed_contact',
        pos=np.asarray(config['grasp']['fixed_finger_contact_local_mm'])/1000,
        size=[.002], rgba=[0, 1, 0, 1], group=4)
    if 'domain_randomization' in config:
        from examples.box_training import randomize_appearance
        randomize_appearance(spec, config['domain_randomization'])
    model = spec.compile()
    key = model.key('startup').id
    adr = model.joint('pickup_box_free').qposadr[0]
    model.key_qpos[key, adr:adr+7] = model.qpos0[adr:adr+7]
    model.key_qpos[key, model.joint('linear_actuator').qposadr[0]] = elevator
    model.key_ctrl[key, model.actuator('linear_actuator').id] = elevator
    # Stay within the wrist travel used by the existing physical replay.
    model.joint('left_wrist_roll').range[:] = [-.8, .9]
    model.actuator_ctrlrange[model.actuator('left_wrist_roll').id] = [-.8, .9]
    aid = model.actuator('left_gripper').id
    model.actuator_forcerange[aid] = [-config['grasp']['max_gripper_torque_nm'], config['grasp']['max_gripper_torque_nm']]
    model.actuator_gainprm[aid, 0] = 8
    model.actuator_biasprm[aid, 1:3] = [-8, -.15]
    model.dof_damping[model.joint('left_gripper').dofadr[0]] = .05
    return model, measured


def physics_step(sim):
    """One unchanged 2 ms dynamics step, with existing idealized gravity support."""
    m, d = sim.model, sim.data
    d.qfrc_applied[:] = 0
    mujoco.mj_step1(m, d)
    for name, aid in sim.actuators.items():
        if name not in WHEELS:
            m.actuator_biasprm[aid, 0] = d.qfrc_bias[m.jnt_dofadr[sim.joints[name]]]
    # Do not let gravity feed-forward bypass the explicit gripper torque cap.
    mujoco.mj_step2(m, d)


class Pickup:
    def __init__(self, config, model=None, measured=None):
        self.config = config
        if model is None:
            model, measured = scene_model(config)
        self.layout = measured
        self.sim = Simulation(model=model, traction=False)
        self.m, self.d = self.sim.model, self.sim.data
        self.names = ['left_'+r for r in ROLES[:5]]
        self.qp = np.array([self.m.joint(n).qposadr[0] for n in self.names])
        self.bounds = np.array([self.m.joint(n).range for n in self.names])
        self.scratch = mujoco.MjData(self.m)
        self.palm = self.m.body('Gripper_Base_v1_1').id
        self.finger = self.m.body('Gripper_Finger_v1_1').id
        self.box = self.m.body('pickup_box').id
        self.box_geom = self.m.geom('pickup_box_geom').id
        self.table_geoms = {self.m.geom(i).id for i in range(self.m.ngeom)
                            if self.m.geom(i).name.startswith(('box_test_table', 'box_test_leg_'))}
        self.phase = 'settle'
        self.on_step = None
        self.frames = []
        self.phase_reports = []
        self.table_contact_steps = 0
        self.table_contact_geoms = set()
        self.max_table_penetration = 0.
        self.max_base_drift = self.max_elevator_drift = 0.
        self.initial_base = self.d.qpos[:3].copy()
        self.elevator_adr = self.m.joint('linear_actuator').qposadr[0]
        self.initial_box = self.d.xpos[self.box].copy()
        self.steps = 0
        self.hold_samples = []
        self.box_motion_before_close = None
        self.before_step = None
        yaw = np.deg2rad(config['box_pose']['yaw_deg'])
        self.grasp_normal = np.array([np.cos(yaw), np.sin(yaw), 0.])

    def fingers(self):
        forces = {'fixed': 0., 'moving': 0.}
        force = np.zeros(6)
        for contact_id, contact in enumerate(self.d.contact):
            geoms = [int(contact.geom1), int(contact.geom2)]
            if self.box_geom not in geoms:
                continue
            other = geoms[1] if geoms[0] == self.box_geom else geoms[0]
            body = self.m.geom_bodyid[other]
            if body in (self.palm, self.finger):
                mujoco.mj_contactForce(self.m, self.d, contact_id, force)
                forces['fixed' if body == self.palm else 'moving'] += max(0., float(force[0]))
        return forces

    def bottom(self):
        rotation = self.d.xmat[self.box].reshape(3, 3)
        return self.d.xpos[self.box, 2] - abs(rotation[2]) @ self.layout['box_half_size_m']

    def observe(self):
        touched = False
        for c in self.d.contact:
            a, b = int(c.geom1), int(c.geom2)
            if a in self.table_geoms or b in self.table_geoms:
                other = b if a in self.table_geoms else a
                if self.m.geom_bodyid[other] not in (0, self.box):
                    touched = True
                    self.table_contact_geoms.add(self.m.geom(other).name)
                    self.max_table_penetration = max(self.max_table_penetration, -float(c.dist))
        self.table_contact_steps += touched
        self.max_base_drift = max(self.max_base_drift, float(np.linalg.norm(self.d.qpos[:3]-self.initial_base)))
        self.max_elevator_drift = max(self.max_elevator_drift, abs(float(self.d.qpos[self.elevator_adr])-self.config['robot_setup']['elevator_m']))
        if self.phase == 'hold':
            self.hold_samples.append((float(self.d.time), self.bottom(), self.d.xpos[self.box].copy(), self.fingers()))
        if self.steps % 10 == 0:
            self.frames.append((float(self.d.time), self.d.qpos.copy(), self.phase))
        if self.on_step:
            self.on_step(self)

    def advance(self, seconds):
        for _ in range(round(seconds/self.m.opt.timestep)):
            if self.before_step is not None:
                self.before_step(self)
            physics_step(self.sim)
            self.steps += 1
            # mj_step1 refreshed transforms/contacts for the evaluated step.
            self.observe()
        mujoco.mj_forward(self.m, self.d)

    def joint_move(self, destination, seconds=2.):
        start = self.d.qpos[self.qp].copy()
        for fraction in np.linspace(0, 1, round(seconds/.02)+1)[1:]:
            smooth = fraction*fraction*(3-2*fraction)
            self.sim.set_joints(dict(zip(self.names, start+smooth*(destination-start))))
            self.advance(.02)

    def center_reference(self, data):
        # Fixed inner face + half the box depth along the pad's inward normal.
        return data.site_xpos[self.m.site('box_fixed_contact').id] + data.xmat[self.palm].reshape(3, 3)[:, 2]*self.layout['box_half_size_m'][0]

    def solve(self, target, seed=None):
        self.scratch.qpos[:] = self.d.qpos
        def residual(q):
            self.scratch.qpos[self.qp] = q
            mujoco.mj_kinematics(self.m, self.scratch)
            normal = self.scratch.xmat[self.palm].reshape(3, 3)[:, 2]
            # A front/back pinch across the 38.4 mm depth. Five arm axes
            # control XYZ and the pad normal; do not overconstrain the last roll.
            return np.r_[self.center_reference(self.scratch)-target,
                         .10*(normal-self.grasp_normal)]
        guess = self.d.qpos[self.qp] if seed is None else seed
        result = least_squares(residual, np.clip(guess, self.bounds[:, 0]+1e-7, self.bounds[:, 1]-1e-7),
            bounds=self.bounds.T, ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=150)
        self.scratch.qpos[self.qp] = result.x
        mujoco.mj_kinematics(self.m, self.scratch)
        error = float(np.linalg.norm(self.center_reference(self.scratch)-target))
        return result.x, error

    def cartesian_move(self, target, seconds=2.):
        source = self.center_reference(self.d).copy()
        seed = self.d.qpos[self.qp].copy()
        worst_error = 0.
        for fraction in np.linspace(0, 1, round(seconds/.04)+1)[1:]:
            smooth = fraction*fraction*(3-2*fraction)
            point = source+smooth*(target-source)
            seed, error = self.solve(point, seed)
            worst_error = max(worst_error, error)
            self.sim.set_joints(dict(zip(self.names, seed)))
            self.advance(.04)
        self.phase_reports.append(dict(phase=self.phase, time_s=float(self.d.time),
            box_center_m=self.d.xpos[self.box].tolist(), reference_m=self.center_reference(self.d).tolist(),
            max_ik_error_mm=1000*worst_error, fingers_n=self.fingers()))

    def run(self, on_step=None):
        self.on_step = on_step
        self.observe()
        self.advance(.6)
        self.phase = 'raise-behind-table'
        # Keep the hand behind the near edge while raising the parked arm.
        q = self.d.qpos[self.qp].copy()
        q[0] = .94
        self.joint_move(q, 1.5)
        q[1] = 1.56
        self.joint_move(q, 2.0)
        q[2] = -1.8
        self.joint_move(q, 1.5)
        self.phase = 'open-gripper'
        self.sim.set_joints({'left_gripper': np.deg2rad(self.config['grasp']['open_angle_deg'])})
        self.advance(1.2)
        open_error = abs(float(self.d.qpos[self.m.joint('left_gripper').qposadr[0]])
                         - np.deg2rad(self.config['grasp']['open_angle_deg']))
        if open_error > np.deg2rad(5):
            raise RuntimeError('The gripper did not reach its open pose; pickup was not attempted')
        # The teacher observes the settled object's pose, not a replayed path.
        # Privileged pose is used for supervision, never as a learner input.
        target = self.d.xpos[self.box].copy() + [0, 0, self.config['grasp']['height_above_box_center_mm']/1000]
        yaw = float(np.arctan2(self.d.xmat[self.box].reshape(3, 3)[1, 0], self.d.xmat[self.box].reshape(3, 3)[0, 0]))
        self.grasp_normal = np.array([np.cos(yaw), np.sin(yaw), 0.])
        behind = target-self.grasp_normal*self.config['grasp']['pregrasp_backoff_mm']/1000
        above = behind+[0, 0, self.config['grasp']['approach_clearance_mm']/1000]
        self.phase = 'approach'
        q, error = self.solve(above)
        self.joint_move(q, 2.5)
        self.advance(.3)
        self.phase_reports.append(dict(phase=self.phase, endpoint_ik_error_mm=error*1000,
            reference_m=self.center_reference(self.d).tolist(), box_center_m=self.d.xpos[self.box].tolist()))
        self.phase = 'descend'
        self.cartesian_move(behind, 2.)
        self.phase = 'align-fingers'
        self.cartesian_move(target-self.grasp_normal*self.config['grasp']['fixed_finger_clearance_mm']/1000, 1.5)
        self.advance(.3)
        self.phase = 'close'
        self.box_motion_before_close = float(np.linalg.norm(self.d.xpos[self.box]-self.initial_box))
        self.sim.set_joints({'left_gripper': np.deg2rad(self.config['grasp']['closed_command_deg'])})
        self.advance(1.5)
        self.phase_reports.append(dict(phase=self.phase, box_center_m=self.d.xpos[self.box].tolist(),
            gripper_angle_deg=float(np.rad2deg(self.d.qpos[self.m.joint('left_gripper').qposadr[0]])), fingers_n=self.fingers()))
        self.phase = 'lift'
        self.cartesian_move(self.center_reference(self.d)+[0, 0, self.config['grasp']['lift_distance_mm']/1000], 2.)
        self.phase = 'hold'
        self.hold_samples.append((float(self.d.time), self.bottom(), self.d.xpos[self.box].copy(), self.fingers()))
        self.advance(self.config['success']['hold_seconds'])
        self.hold_samples.append((float(self.d.time), self.bottom(), self.d.xpos[self.box].copy(), self.fingers()))
        final = (float(self.d.time), self.d.qpos.copy(), 'complete')
        if self.frames[-1][0] == final[0]:
            self.frames[-1] = final
        else:
            self.frames.append(final)
        return self.report()

    def report(self):
        success = self.config['success']
        minimum_bottom = min((r[1] for r in self.hold_samples), default=-1)
        hold_drift = max((np.linalg.norm(r[2]-self.hold_samples[0][2]) for r in self.hold_samples), default=1)
        minimum_forces = {label: min((r[3][label] for r in self.hold_samples), default=0) for label in ('fixed', 'moving')}
        hold_duration = self.hold_samples[-1][0]-self.hold_samples[0][0] if self.hold_samples else 0.
        warnings = self.sim.state()['warnings']
        checks = dict(box_clear_of_table=minimum_bottom-self.layout['tabletop_z_m'] >= success['minimum_bottom_clearance_mm']/1000,
            both_fingers_hold=all(f >= success['minimum_finger_normal_force_n'] for f in minimum_forces.values()),
            stable_hold=hold_drift <= success['maximum_box_drift_during_hold_mm']/1000,
            full_hold_duration=hold_duration >= success['hold_seconds']-1e-8,
            no_robot_table_contact=self.table_contact_steps == 0,
            base_stationary=self.max_base_drift <= success['maximum_base_translation_mm']/1000,
            elevator_stationary=self.max_elevator_drift <= success['maximum_elevator_drift_mm']/1000,
            physics_healthy=not warnings)
        checks = {name: bool(passed) for name, passed in checks.items()}
        return dict(test=self.config['test'], simulation_only=True, passed=bool(all(checks.values())), checks=checks,
            scene=self.config, layout=self.layout, physics_timestep_s=self.m.opt.timestep,
            simulated_duration_s=float(self.d.time), box_final_center_m=self.d.xpos[self.box].tolist(),
            minimum_bottom_clearance_during_hold_mm=(minimum_bottom-self.layout['tabletop_z_m'])*1000,
            hold_drift_mm=hold_drift*1000, minimum_finger_normal_force_during_hold_n=minimum_forces,
            measured_hold_duration_s=hold_duration,
            box_motion_before_close_mm=None if self.box_motion_before_close is None else 1000*self.box_motion_before_close,
            robot_table_contact_steps=self.table_contact_steps, robot_table_contact_geoms=sorted(self.table_contact_geoms),
            max_table_penetration_mm=self.max_table_penetration*1000,
            max_base_drift_mm=self.max_base_drift*1000, max_elevator_drift_mm=self.max_elevator_drift*1000,
            phases=self.phase_reports, warnings=warnings, grasp_assistance=False,
            assumptions=['Robot centered and square to table; reproduce or measure physical placement before transfer.',
                         'Centered uniform inertia approximates the unopened servo package.',
                         'Supplied mass/friction/restitution and table friction are estimates.',
                         'CAD finger landmarks, idealized arm servo dynamics and gripper torque are not hardware measurements.',
                         'Contact solref damping approximates low restitution; it is not a calibrated 0.05 bounce measurement.'])


def camera(detail=False):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [.43, .18, .84] if detail else [.32, .10, .70]
    cam.distance, cam.azimuth, cam.elevation = (.65, 115, -15) if detail else (1.45, 145, -22)
    return cam


def options():
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    opt.sitegroup[:] = 0
    return opt


def physics_signature(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode())
    digest.update(mujoco.__version__.encode())
    for path in (Path(__file__), MODEL, ROOT/'sourccey/collision.py', ROOT/'sourccey/simulation.py',
                 ROOT/'sourccey/pose.py', ROOT/'models/collision/manifest.json',
                 ROOT/'examples/box_training.py', ROOT/'models/source/camera_poses.json'):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_run(run, report, output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    np.savez_compressed(output/'recording.npz', time=np.asarray([r[0] for r in run.frames]),
        qpos=np.asarray([r[1] for r in run.frames]), phase=np.asarray([r[2] for r in run.frames]),
        config=np.asarray(json.dumps(run.config)), signature=np.asarray(physics_signature(run.config)))


def load_recording(output, config):
    with np.load(Path(output)/'recording.npz', allow_pickle=False) as archive:
        recording = {k: archive[k].copy() for k in archive.files}
    if 'signature' not in recording or str(recording['signature']) != physics_signature(config):
        raise ValueError('Scene, physics or model changed; compute a new recording without --replay')
    report = json.loads((Path(output)/'report.json').read_text(encoding='utf-8'))
    if json.loads(str(recording['config'])) != config or report['scene'] != config:
        raise ValueError('Recording and report scene do not match')
    return recording, report


def render(run, path, detail=False):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(run.m, height=720, width=960) as renderer:
        renderer.update_scene(run.d, camera=camera(detail), scene_option=options())
        Image.fromarray(renderer.render()).save(path)


def playback(sim, recording, report, speed=1., camera_feeds=False):
    """Display a physics recording; P pauses, R restarts, finish remains visible."""
    from mujoco import viewer
    from contextlib import ExitStack
    from examples.box_training import SIZE,mosaic,render_camera_frames,camera_frusta
    state = dict(paused=False, restart=False, camera=None,feeds=camera_feeds,frusta=camera_feeds)
    def key(keycode):
        if keycode in (ord('P'), ord('p')): state['paused'] = not state['paused']
        if keycode in (ord('R'), ord('r')): state.update(restart=True, paused=False)
        if keycode in (ord('1'), ord('2')): state['camera'] = keycode == ord('2')
        if keycode in (ord('C'), ord('c')): state['feeds'] = not state['feeds']
        if keycode in (ord('F'), ord('f')): state['frusta'] = not state['frusta']
    with ExitStack() as stack:
        renderer = stack.enter_context(mujoco.Renderer(sim.model,height=SIZE[1],width=SIZE[0]))
        window = stack.enter_context(viewer.launch_passive(sim.model, sim.data, key_callback=key))
        window.cam.lookat[:] = camera().lookat
        window.cam.distance, window.cam.azimuth, window.cam.elevation = camera().distance, camera().azimuth, camera().elevation
        window.opt.geomgroup[3] = 0
        window.opt.sitegroup[:] = 0
        position = float(recording['time'][0])
        last = time.perf_counter()
        next_camera = 0.
        while window.is_running():
            if state['camera'] is not None:
                cam = camera(state['camera'])
                window.cam.lookat[:] = cam.lookat
                window.cam.distance, window.cam.azimuth, window.cam.elevation = cam.distance, cam.azimuth, cam.elevation
                state['camera'] = None
            now = time.perf_counter()
            if state['restart']: position = 0.; state['restart'] = False
            elif not state['paused']: position = min(float(recording['time'][-1]), position+(now-last)*speed)
            last = now
            index = min(max(0, np.searchsorted(recording['time'], position, side='right')-1), len(recording['time'])-1)
            following = min(index+1, len(recording['time'])-1)
            span = recording['time'][following]-recording['time'][index]
            weight = (position-recording['time'][index])/span if span > 0 else 0.
            sim.data.qpos[:] = (1-weight)*recording['qpos'][index]+weight*recording['qpos'][following]
            mujoco.mj_normalizeQuat(sim.model, sim.data.qpos)
            mujoco.mj_kinematics(sim.model, sim.data)
            mujoco.mj_comPos(sim.model, sim.data)
            mujoco.mj_camlight(sim.model, sim.data)
            sim.data.time = position
            if now >= next_camera:
                if state['feeds']:
                    image = mosaic(render_camera_frames(sim.model,sim.data,renderer,report['scene'],index),width=240)
                    viewport = window.viewport
                    window.set_images((mujoco.MjrRect(max(0,viewport.width-image.shape[1]-10),30,image.shape[1],image.shape[0]),image))
                else:
                    window.clear_images()
                next_camera = now+.1
            with window.lock():
                if state['frusta']: camera_frusta(window.user_scn,sim.model,sim.data)
                else: window.user_scn.ngeom = 0
            window.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                'Box pickup - Test A\nFixed position and rotation\n'+str(recording['phase'][index]),
                f'{position:.1f} / {recording["time"][-1]:.1f} s\nPhysics result: {"PASS" if report["passed"] else "FAIL"}') ,
                (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 'P: pause/resume   R: replay   1/2: overview/gripper\nC: four camera feeds   F: camera frusta', 'Simulation only')])
            window.sync()
            time.sleep(.015)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--replay', action='store_true', help='Play the previously saved physics recording')
    parser.add_argument('--camera-feeds',action='store_true',help='Show four onboard camera tiles and field-of-view outlines')
    parser.add_argument('--speed', type=float, default=1., help='Display speed only; physics is unchanged')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('Display speed must be positive and finite')
    config = load_scene(args.scene)
    if args.replay:
        try:
            recording, report = load_recording(args.output, config)
        except ValueError as exc:
            parser.error(str(exc))
        model, _ = scene_model(config)
        sim = Simulation(model=model, traction=False)
    else:
        print('Preparing measured box pickup scene...', flush=True)
        run = Pickup(config)
        last_phase = None
        def progress(current):
            nonlocal last_phase
            if current.phase != last_phase:
                print(f'{current.d.time:.1f}s: {current.phase}', flush=True)
                last_phase = current.phase
        report = run.run(progress)
        save_run(run, report, args.output)
        render(run, args.output/'final.png')
        render(run, args.output/'grasp_detail.png', detail=True)
        print(json.dumps({k:v for k,v in report.items() if k not in ('scene', 'phases')}, indent=2), flush=True)
        recording = dict(time=np.asarray([r[0] for r in run.frames]), qpos=np.asarray([r[1] for r in run.frames]),
                         phase=np.asarray([r[2] for r in run.frames]))
        sim = run.sim
    if args.view:
        # Use visible CAD for display; the measured result came from full physics.
        model,_ = scene_model(config,display_only=True)
        if model.nq != sim.model.nq:
            raise RuntimeError('Display and physics joint layouts differ')
        sim = Simulation(model=model,traction=False)
        playback(sim, recording, report, args.speed,args.camera_feeds)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
