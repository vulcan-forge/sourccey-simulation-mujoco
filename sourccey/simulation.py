"""MuJoCo dynamics with a load-limited equivalent mecanum friction model."""
import numpy as np
import mujoco

from .build import MODEL, CAD_ROTATION, ROLES, WHEELS, build
from .control import HandTarget, finite_vector, gripper_degrees, wheel_rates


class Simulation:
    def __init__(self, full_elevator_range=False, traction=True, model=None):
        if not MODEL.exists():
            build()
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.full_elevator_range = full_elevator_range
        self.traction = traction
        self.base_id = self.model.body("base_link").id
        self.joints = {self.model.joint(i).name: i for i in range(1, self.model.njnt)}
        self.actuators = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        self.wheel_geoms = {self.model.geom("contact_" + name).id: (i, name) for i, name in enumerate(WHEELS)}
        self.arm_dofs = [self.model.jnt_dofadr[j] for n, j in self.joints.items() if n not in WHEELS]
        self.base_command = np.zeros(3)
        self.targets = {}
        self.reset()

    @property
    def robot_rotation(self):
        return self.data.xmat[self.base_id].reshape(3, 3) @ CAD_ROTATION.T

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("startup").id)
        self.model.actuator_biasprm[:, 0] = 0
        self.base_command[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.targets = {side: HandTarget(self.ee_local(side)) for side in ("left", "right")}

    def set_joints(self, positions):
        # Validate the entire command before mutation.
        pending = []
        for name, value in positions.items():
            if name not in self.joints or name in WHEELS:
                raise ValueError(f"Unknown position-controlled joint: {name}")
            value = finite_vector([value], 1)[0]
            j = self.joints[name]
            pending.append((self.actuators[name], np.clip(value, *self.model.jnt_range[j])))
        for aid, value in pending:
            self.data.ctrl[aid] = value

    def set_base(self, forward=0, left=0, yaw=0):
        self.base_command[:] = np.clip(finite_vector([forward, left, yaw], 3), -1, 1)
        for name, rate in zip(WHEELS, wheel_rates(self.base_command)):
            self.data.ctrl[self.actuators[name]] = rate

    def set_elevator(self, position_m):
        bottom = -.315 if self.full_elevator_range else -.3104
        position_m = finite_vector([position_m], 1)[0]
        self.set_joints({"linear_actuator": np.clip(position_m, bottom, -.0142)})

    def set_gripper(self, side, closure):
        self.set_joints({side + "_gripper": np.deg2rad(gripper_degrees(closure))})

    def mount_position(self, side, data=None):
        return (data or self.data).site_xpos[self.model.site(side + "_mount").id].copy()

    def ee_local(self, side):
        return self.robot_rotation.T @ (self.data.site_xpos[self.model.site(side + "_ee").id] - self.mount_position(side))

    def solve_ik(self, side, target_local, iterations=60, tolerance=0.002):
        """Position-only damped least squares on the primary 3 joints; bounded line search.

        Wrist flex/roll remain explicit commands. Uses scratch data; never teleports physics.
        Returns achieved error and convergence, including for unreachable targets.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        target_local = finite_vector(target_local, 3)
        m, d = self.model, self.ik_data
        d.qpos[:] = self.data.qpos
        d.qvel[:] = 0
        for name, aid in self.actuators.items():
            if name not in WHEELS:
                d.qpos[m.jnt_qposadr[self.joints[name]]] = self.data.ctrl[aid]
        mujoco.mj_kinematics(m, d)
        target = self.mount_position(side, d) + self.robot_rotation @ target_local
        names = [side + "_" + role for role in ROLES[:3]]
        ids = [self.joints[n] for n in names]
        qp = m.jnt_qposadr[ids]
        dof = m.jnt_dofadr[ids]
        bounds = m.jnt_range[ids]
        sid = m.site(side + "_ee").id
        jac = np.zeros((3, m.nv))
        for _ in range(iterations):
            mujoco.mj_forward(m, d)
            error = target - d.site_xpos[sid]
            norm = np.linalg.norm(error)
            if norm < tolerance:
                break
            mujoco.mj_jacSite(m, d, jac, None, sid)
            J = jac[:, dof]
            chase = error * min(1, .08 / max(norm, 1e-12))
            step = J.T @ np.linalg.solve(J @ J.T + .015**2 * np.eye(3), chase)
            step = np.clip(step, -.15, .15)
            previous = d.qpos[qp].copy()
            improved = False
            for scale in (1, .5, .25, .1):
                d.qpos[qp] = np.clip(previous + scale * step, bounds[:, 0], bounds[:, 1])
                mujoco.mj_kinematics(m, d)
                if np.linalg.norm(target - d.site_xpos[sid]) < norm:
                    improved = True
                    break
            if not improved:
                d.qpos[qp] = previous
                break
        mujoco.mj_kinematics(m, d)
        error = float(np.linalg.norm(target - d.site_xpos[sid]))
        self.set_joints(dict(zip(names, d.qpos[qp])))
        return {"error_m": error, "converged": error < tolerance}

    def _apply_traction(self):
        """Diagonal slip resistance at each actual wheel/floor contact, capped by mu*N.

        The roller's free direction has zero tangential resistance. A force applied at
        the wheel's contact point reacts on BOTH its axle and the floating chassis.
        No chassis pose/velocity servo is used. Only flat stationary floor supported.
        """
        m, d = self.model, self.data
        force6 = np.zeros(6)
        jp = np.zeros((3, m.nv))
        R = self.robot_rotation
        floor = m.geom("floor").id
        for contact_id in range(d.ncon):
            c = d.contact[contact_id]
            g1, g2 = int(c.geom1), int(c.geom2)
            gid = g2 if g1 == floor else g1 if g2 == floor else -1
            if gid not in self.wheel_geoms or c.efc_address < 0:
                continue
            index, _ = self.wheel_geoms[gid]
            mujoco.mj_contactForce(m, d, contact_id, force6)
            load = max(0, force6[0])
            direction = R @ np.array([1, (-1, 1, 1, -1)[index], 0])
            direction[2] = 0
            direction /= np.linalg.norm(direction)
            body = m.geom_bodyid[gid]
            mujoco.mj_jac(m, d, jp, None, c.pos, body)
            slip = float(direction @ (jp @ d.qvel))
            magnitude = np.clip(-1800 * slip, -.9 * load, .9 * load)
            mujoco.mj_applyFT(m, d, magnitude * direction, np.zeros(3), c.pos, body, d.qfrc_applied)

    def step(self, count=1):
        for _ in range(count):
            m, d = self.model, self.data
            d.qfrc_applied[:] = 0
            mujoco.mj_forward(m, d)
            # Idealized gravity/Coriolis feed-forward, inside actuator force limits.
            # Gains and caps are simulation settings, not measured hardware ratings.
            for name, aid in self.actuators.items():
                if name not in WHEELS:
                    m.actuator_biasprm[aid, 0] = d.qfrc_bias[m.jnt_dofadr[self.joints[name]]]
            mujoco.mj_forward(m, d)
            if self.traction:
                self._apply_traction()
            mujoco.mj_step(m, d)
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        return {"time": self.data.time, "base_position": self.data.qpos[:3].tolist(),
                "joints": {n: float(self.data.qpos[self.model.jnt_qposadr[j]]) for n, j in self.joints.items()},
                "warnings": {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(self.data.warning) if w.number}}
