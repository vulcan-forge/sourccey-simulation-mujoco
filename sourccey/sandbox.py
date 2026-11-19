"""Four-table manual manipulation sandbox. Simulation only; no robot transport."""
from dataclasses import dataclass
import time

import mujoco
import numpy as np

from .appearance import add_box_materials
from .build import MODEL, ROOT, ROLES, WHEELS
from .collision import add_torso_collision, decomposed_collision
from .simulation import Simulation


STATIONS = (
    ("01 / WORKBENCH", 0., None, (.13, .70, .80)),
    ("02 / SORT", np.pi/2, 1.55, (.45, .77, .56)),
    ("03 / TRANSFER", np.pi, 1.55, (.93, .60, .28)),
    ("04 / DISPATCH", -np.pi/2, 1.55, (.60, .50, .85)),
)


def _texture(spec, name, pixels, **kwargs):
    pixels = np.asarray(pixels, dtype=np.uint8)
    texture = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
        height=pixels.shape[0], width=pixels.shape[1], nchannel=3, data=pixels.ravel())
    uniform = kwargs.pop("texuniform", True)
    material = spec.add_material(name=name, rgba=[1, 1, 1, 1], roughness=.85,
                                 texuniform=uniform, **kwargs)
    material.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = texture.name
    return material


def build_scene(*, display_only=False):
    """Keep benchmark dimensions; free the base and add three separate stations."""
    from examples.box_pickup import load_scene, layout
    config = load_scene()
    measured = layout(config)
    spec = mujoco.MjSpec.from_file(str(MODEL))
    spec.modelname = "Sourccey / The Box Workshop"
    spec.memory = 128 * 1024 * 1024
    if display_only:
        for geom in list(spec.geoms):
            if geom.group == 3:
                spec.delete(geom)
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    else:
        decomposed_collision(spec)
        add_torso_collision(spec)
    for geom in spec.geoms:
        if geom.contype & 2:
            geom.conaffinity |= 2
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 5
    add_box_materials(spec, seed=18)
    palette = {"frame": ([.075, .10, .13, 1], .30, .7),
               "rubber": ([.10, .14, .17, 1], .92, 0),
               "wall": ([.24, .30, .37, 1], .95, 0)}
    for name, (color, rough, metal) in palette.items():
        spec.add_material(name="workshop_"+name, rgba=color, roughness=rough, metallic=metal)
    rng = np.random.default_rng(18)
    noise = rng.normal(0, 1.7, (256, 256, 1))
    concrete = np.clip(np.array([105, 119, 130])[None, None, :] + noise, 0, 255)
    concrete[::64, :, :] *= .83
    concrete[:, ::64, :] *= .83
    _texture(spec, "workshop_floor", concrete, texrepeat=[6, 6])
    spec.geom("floor").material = "workshop_floor"
    spec.geom("floor").rgba = [1, 1, 1, 1]

    table = config["table"]
    depth, across, top, thick = [table[k]/1000 for k in (
        "depth_mm", "length_across_mm", "height_above_floor_mm", "thickness_mm")]
    half = np.asarray(measured["box_half_size_m"])
    records = []
    for index, (title, yaw, near, accent) in enumerate(STATIONS):
        if near is None:
            near = measured["near_edge_x_m"]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        station = spec.worldbody.add_body(name=f"station_{index}",
            quat=[np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        station.add_geom(name=f"table_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[near+depth/2, 0, top-thick/2], size=[depth/2, across/2, thick/2],
            material="pbr_wood", contype=1, conaffinity=6,
            friction=[table["sliding_friction"], .001, .0001])
        # Dark metal perimeter and capped, tapered-looking legs.
        for y in (-across/2, across/2):
            station.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[near+depth/2, y, top-thick+.006], size=[depth/2, .012, .018],
                material="workshop_frame", contype=0, conaffinity=0)
        for x in (near, near+depth):
            station.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, 0, top-thick/2], size=[.008, across/2, .014],
                rgba=[*accent, 1], contype=0, conaffinity=0)
        for i, x in enumerate((near+.065, near+depth-.065)):
            for j, y in enumerate((-across/2+.065, across/2-.065)):
                station.add_geom(name=f"table_leg_{index}_{i}_{j}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[x, y, (top-thick)/2], size=[.024, .024, (top-thick)/2],
                    material="workshop_frame", contype=1, conaffinity=6)
        # Floor station outlines are ink, not invisible collision obstacles.
        for y in (-across/2-.09, across/2+.09):
            station.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[near+depth/2, y, .0007],
                size=[depth/2+.09, .008, .0005], rgba=[*accent, 1], contype=0, conaffinity=0)
        for x in (near-.09, near+depth+.09):
            station.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, 0, .0007],
                size=[.008, across/2+.09, .0005], rgba=[*accent, 1], contype=0, conaffinity=0)
        _station_label(spec, station, index, title, near, depth, across, top, accent)
        for b, (offset, lateral, angle) in enumerate(((.11, .18, 0), (.14, -.25, .23),
                                                     (.32, .39, -.35), (.40, -.10, .60))):
            xy = rotation @ [near+offset, lateral]
            angle += yaw
            mass = config["object"]["mass_kg"]
            full = 2*half
            inertia = mass/12*np.array([full[1]**2+full[2]**2,
                full[0]**2+full[2]**2, full[0]**2+full[1]**2])
            body = spec.worldbody.add_body(name=f"workshop_box_{index}_{b}",
                pos=[*xy, top+half[2]+.001], quat=[np.cos(angle/2), 0, 0, np.sin(angle/2)],
                mass=mass, ipos=[0, 0, 0], inertia=inertia)
            body.add_freejoint(name=f"workshop_box_free_{index}_{b}")
            geom_name = f"workshop_box_geom_{index}_{b}"
            body.add_geom(name=geom_name, type=mujoco.mjtGeom.mjGEOM_BOX, size=half,
                mass=config["object"]["mass_kg"], material="pbr_cardboard", contype=4,
                conaffinity=7, condim=4, priority=1, friction=[.35, .001, .0001],
                solref=[.006, .70], solimp=[.95, .99, .001, .5, 2])
            spec.add_pair(name=f"workshop_table_contact_{index}_{b}",
                geomname1=geom_name, geomname2=f"table_{index}", condim=3,
                friction=[table["sliding_friction"]]*2+[0, 0, 0], solref=[.006, .70])
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, half[2]+.00015],
                size=[half[0], .005, .00012], rgba=[.87, .72, .46, 1],
                contype=0, conaffinity=0, density=0)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-half[0]-.00015, 0, .006],
                size=[.0001, half[1]*.55, half[2]*.34], rgba=[*accent, 1],
                contype=0, conaffinity=0, density=0)
        records.append(dict(name=title, near_edge_m=near, yaw_rad=yaw,
                            center_xy_m=(rotation @ [near+depth/2, 0]).tolist()))
    # Low exhibition-room backdrop, behind the working area.
    for y in (-3.65, 3.65):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-.65, y, .45],
            size=[3.3, .035, .45], material="workshop_wall", contype=1, conaffinity=6)
    spec.stat.center = [0, 0, .6]
    spec.stat.extent = 4
    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 1000
    model = spec.compile()
    key = model.key("startup").id
    for i in range(4):
        for b in range(4):
            adr = model.joint(f"workshop_box_free_{i}_{b}").qposadr[0]
            model.key_qpos[key, adr:adr+7] = model.qpos0[adr:adr+7]
    for side in ("left", "right"):
        aid = model.actuator(side+"_gripper").id
        model.actuator_forcerange[aid] = [-.6, .6]
        model.actuator_gainprm[aid, 0] = 8
        model.actuator_biasprm[aid, 1:3] = [-8, -.15]
        model.dof_damping[model.joint(side+"_gripper").dofadr[0]] = .05
    return model, dict(stations=records, table_dimensions_m=[depth, across, top],
        box_dimensions_m=(half*2).tolist(), box_mass_kg=config["object"]["mass_kg"], box_count=16)


def _station_label(spec, station, index, title, near, depth, across, top, accent):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (1024, 128), (22, 32, 43))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/seguisb.ttf", 64)
    except OSError:
        font = ImageFont.load_default(size=64)
    draw.rectangle([0, 0, 15, 128], fill=tuple(int(c*255) for c in accent))
    draw.text((38, 22), title, font=font, fill=(230, 239, 245))
    _texture(spec, f"station_label_{index}", image, texrepeat=[1, 1], texuniform=False)
    station.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[near+depth-.06, 0, top+.0007], size=[.035, across/2-.08, .0003],
        material=f"station_label_{index}", contype=0, conaffinity=0)


@dataclass
class Controls:
    """Held-key axes; commands expire if the UI stops delivering heartbeats."""
    side: str = "left"
    drive: tuple = (0., 0., 0.)
    hand: tuple = (0., 0., 0.)
    wrist: tuple = (0., 0.)
    grip: float = 0.
    elevator: float = 0.
    joint: int = 0
    jog: float = 0.
    precision: bool = False


class Workshop:
    def __init__(self, model=None):
        if model is None:
            model, self.scene = build_scene()
        else:
            self.scene = {}
        self.sim = Simulation(model=model)
        self.controls = Controls()
        self.last_input = -np.inf
        self.paused = False
        self.ik_error = 0.
        self.last_side = "left"
        self.held = {s: self.sim.ee_local(s).copy() for s in ("left", "right")}
        self.command_time = 0.

    def reset(self):
        self.sim.reset()
        self.controls = Controls(side=self.controls.side)
        self.last_input = -np.inf
        self.command_time = 0.
        self.paused = False
        self.held = {s: self.sim.ee_local(s).copy() for s in ("left", "right")}

    def receive(self, controls, now=None):
        if controls.side not in ("left", "right") or not 0 <= controls.joint < 6:
            raise ValueError("Unknown arm or joint")
        vector = [*controls.drive, *controls.hand, *controls.wrist,
                  controls.grip, controls.elevator, controls.jog]
        if len(vector) != 11 or not np.isfinite(vector).all() or np.max(np.abs(vector)) > 1:
            raise ValueError("Input axes must be finite and in [-1,1]")
        self.controls = controls
        self.last_input = time.monotonic() if now is None else now

    def hold(self):
        self.sim.set_base()
        names = [side+"_"+role for side in ("left", "right") for role in ROLES[:5]]
        self.sim.set_joints({n: self.sim.data.qpos[self.sim.model.joint(n).qposadr[0]] for n in names})
        self.held = {s: self.sim.ee_local(s).copy() for s in ("left", "right")}
        self.controls = Controls(side=self.controls.side)

    def advance(self, seconds, *, now=None):
        if not np.isfinite(seconds) or not 0 <= seconds <= .1:
            raise ValueError("Advance by 0..100 ms at a time")
        now = time.monotonic() if now is None else now
        c = self.controls if now-self.last_input <= .25 else Controls(side=self.controls.side)
        sim, m, d = self.sim, self.sim.model, self.sim.data
        if self.paused:
            sim.set_base()
            return 0
        dt = m.opt.timestep
        count = int(round(seconds/dt))
        for _ in range(count):
            if d.time >= self.command_time:
                # 50 Hz control; the 2 ms physics clock is never sped up/skipped.
                self._command(c, .02)
                self.command_time = d.time+.02-1e-8
            # Split stepping avoids the redundant forwards in the general driver.
            d.qfrc_applied[:] = 0
            mujoco.mj_step1(m, d)
            for name, aid in sim.actuators.items():
                if name not in WHEELS:
                    m.actuator_biasprm[aid, 0] = d.qfrc_bias[m.jnt_dofadr[sim.joints[name]]]
            sim._apply_traction()
            mujoco.mj_step2(m, d)
        mujoco.mj_forward(m, d)
        return count

    def _command(self, c, dt):
        sim, m, d = self.sim, self.sim.model, self.sim.data
        speed = .25 if c.precision else 1.
        sim.set_base(*(np.asarray(c.drive)*.45*speed))
        if c.side != self.last_side:
            self.held[c.side] = sim.ee_local(c.side).copy()
            self.last_side = c.side
        side = c.side
        if c.elevator:
            sim.set_elevator(d.ctrl[sim.actuators["linear_actuator"]]+c.elevator*.06*speed*dt)
        direct = {side+"_wrist_flex": c.wrist[0], side+"_wrist_roll": c.wrist[1],
                  side+"_gripper": c.grip}
        if c.jog:
            direct[side+"_"+ROLES[c.joint]] = c.jog
        for name, value in direct.items():
            if not value:
                continue
            aid, joint = sim.actuators[name], m.joint(name)
            actual = d.qpos[joint.qposadr[0]]
            # Limit stored error when pushing against a solid object.
            target = np.clip(d.ctrl[aid], actual-.15, actual+.15)
            sim.set_joints({name: target+value*.75*speed*dt})
        if np.linalg.norm(c.hand) and not c.jog:
            actual = sim.ee_local(side)
            delta = np.asarray(c.hand, dtype=float)
            delta /= max(1., np.linalg.norm(delta))
            target = self.held[side] + delta*.12*speed*dt
            error = target-actual
            target = actual + error*min(1., .035/max(np.linalg.norm(error), 1e-9))
            elbow = side+"_elbow_flex"
            # The parked arm is exactly straight: its upward position Jacobian
            # is singular. Seed a small legal elbow bend in the IK scratch pose
            # so an upward key can shorten the arm instead of doing nothing.
            if (np.linalg.norm(target) < np.linalg.norm(actual)-1e-5
                    and abs(d.ctrl[sim.actuators[elbow]]) < .02):
                sim.set_joints({elbow: -.10})
            result = self.solve_hand(side, target)
            names = [side+"_"+role for role in ROLES[:3]]
            actual_joints = np.array([d.qpos[m.joint(n).qposadr[0]] for n in names])
            desired = np.array([d.ctrl[sim.actuators[n]] for n in names])
            delta = desired-actual_joints
            # Preserve the IK solution's coordinated joint direction. Clipping
            # each coordinate separately can turn a straight lift into a sweep.
            scale = min(1., .15/max(np.max(np.abs(delta)), 1e-9))
            sim.set_joints(dict(zip(names, actual_joints+delta*scale)))
            self.ik_error = result["error_m"]
            self.held[side] = target
        else:
            self.held[side] = sim.ee_local(side).copy()

    def solve_hand(self, side, target):
        """Position IK on scratch kinematics, never on the live physics pose."""
        from scipy.optimize import least_squares
        sim, m, scratch = self.sim, self.sim.model, self.sim.ik_data
        scratch.qpos[:] = sim.data.qpos
        names = [side+"_"+role for role in ROLES[:3]]
        qpos = np.array([m.joint(n).qposadr[0] for n in names])
        dofs = np.array([m.joint(n).dofadr[0] for n in names])
        for name, aid in sim.actuators.items():
            if name not in WHEELS:
                scratch.qpos[m.joint(name).qposadr[0]] = sim.data.ctrl[aid]
        world_target = sim.mount_position(side) + sim.robot_rotation @ target
        sid = m.site(side+"_ee").id
        jacobian = np.zeros((3, m.nv))

        def error(q):
            scratch.qpos[qpos] = q
            mujoco.mj_kinematics(m, scratch)
            mujoco.mj_comPos(m, scratch)
            return scratch.site_xpos[sid]-world_target

        def jac(q):
            error(q)
            mujoco.mj_jacSite(m, scratch, jacobian, None, sid)
            return jacobian[:, dofs].copy()

        bounds = np.array([m.joint(n).range for n in names])
        result = least_squares(error, np.clip(scratch.qpos[qpos], bounds[:, 0], bounds[:, 1]),
            jac=jac, bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=15,
            ftol=1e-5, xtol=1e-5, gtol=1e-6)
        sim.set_joints(dict(zip(names, result.x)))
        return {"error_m": float(np.linalg.norm(result.fun))}


def overview_camera():
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.25, 0, .45]
    camera.distance, camera.azimuth, camera.elevation = 5.8, 45, -38
    return camera


def visible_options():
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    return options


def copy_display_state(source, model, destination):
    """Render the exact live state without uploading 8,000 invisible hulls."""
    if source.qpos.shape != destination.qpos.shape or source.qvel.shape != destination.qvel.shape:
        raise ValueError("Physics and presentation model coordinates disagree")
    destination.qpos[:] = source.qpos
    destination.qvel[:] = source.qvel
    destination.ctrl[:] = source.ctrl
    destination.time = source.time
    mujoco.mj_forward(model, destination)
