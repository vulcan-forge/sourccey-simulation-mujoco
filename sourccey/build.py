"""Deterministically generate MJCF from the unmodified full-assembly URDF."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .pose import DEFAULT_POSE

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/source/SourcceyURDF/SourcceyMkV.urdf"
CAMERA_POSES = ROOT / "models/source/camera_poses.json"
MODEL = ROOT / "models/sourccey.xml"
ROLES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
WHEELS = ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")
# The Fusion/standalone URDFs both specify +/-90 degrees. The Mk.V CAD can
# visibly bend farther downward; this simulation-only range is not a measured
# hardware stop. Keep the copied source URDF untouched.
ELBOW_LOWER_DEGREES = -135.0
# The simulator's raised shoulder stop. The copied URDF reaches 0 m, but the
# requested operating barrier and home pose both stop slightly below it.
ELEVATOR_UPPER_METERS = -0.0142
CAD_ROTATION = Rotation.from_euler("z", np.pi / 4).as_matrix()


def fmt(values):
    return " ".join(f"{v:.12g}" for v in values)


def origin(element):
    o = element.find("origin")
    if o is None:
        return np.zeros(3), np.eye(3)
    return (np.fromstring(o.get("xyz", "0 0 0"), sep=" "),
            Rotation.from_euler("xyz", np.fromstring(o.get("rpy", "0 0 0"), sep=" ")).as_matrix())


def quat(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return fmt(q[[3, 0, 1, 2]])


def vertices(path):
    """Source assets are binary STL, in millimeters."""
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    dtype = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(raw, dtype=dtype, count=count, offset=84)["v"].reshape(-1, 3).astype(float)


def build():
    source = ET.parse(SOURCE).getroot()
    links = {e.get("name"): e for e in source.findall("link")}
    children = {name: [] for name in links}
    joints = source.findall("joint")
    for j in joints:
        children[j.find("parent").get("link")].append(j)
    # First compile exactly what was copied, to catch missing assets/compiler errors.
    imported = mujoco.MjModel.from_xml_path(str(SOURCE))
    assert imported.njnt == 17
    root = ET.Element("mujoco", model="Sourccey")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", fusestatic="false",
                  meshdir="source/SourcceyURDF", inertiafromgeom="false")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast",
                  cone="elliptic", iterations="60", gravity="0 0 -9.81")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="floor", type="2d", builtin="checker", width="512", height="512",
                  rgb1="0.19 0.22 0.27", rgb2="0.24 0.28 0.33")
    ET.SubElement(asset, "material", name="floor", texture="floor", texrepeat="10 10", reflectance="0.1")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="1 -2 3", dir="-0.3 0.5 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.1", material="floor",
                  contype="1", conaffinity="2", friction="0 0 0", condim="1")
    bodies, frames, mesh_names = {}, {}, {}
    arm_links = set()

    def add_link(name, parent, p, R, frame_p, frame_R, arm=False):
        body = ET.SubElement(parent, "body", name=name, pos=fmt(p), quat=quat(R))
        bodies[name] = body
        frames[name] = (frame_p, frame_R)
        link = links[name]
        if arm:
            arm_links.add(name)
        inertial = link.find("inertial")
        if inertial is not None:
            ip, ir = origin(inertial)
            tensor = inertial.find("inertia")
            a = {k: float(v) for k, v in tensor.attrib.items()}
            I = np.array([[a['ixx'], a['ixy'], a['ixz']], [a['ixy'], a['iyy'], a['iyz']],
                          [a['ixz'], a['iyz'], a['izz']]])
            I = ir @ I @ ir.T
            ET.SubElement(body, "inertial", pos=fmt(ip), mass=inertial.find("mass").get("value"),
                          fullinertia=fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))
        for index, vis in enumerate(link.findall("visual")):
            mesh = vis.find("geometry/mesh")
            filename = mesh.get("filename")
            if filename not in mesh_names:
                mesh_names[filename] = f"mesh_{len(mesh_names)}"
                ET.SubElement(asset, "mesh", name=mesh_names[filename], file=filename,
                              scale=mesh.get("scale", "1 1 1"))
            vp, vr = origin(vis)
            color = "0.64 0.68 0.72 1"
            if "Wheel" in name or "Motor" in name:
                color = "0.16 0.18 0.21 1"
            if "Gripper" in name:
                color = "0.25 0.48 0.65 1"
            ET.SubElement(body, "geom", name=f"visual_{name}_{index}", type="mesh", mesh=mesh_names[filename],
                          pos=fmt(vp), quat=quat(vr), contype="0", conaffinity="0", group="2", rgba=color)
            # Per-part convex hulls for arm/environment contact. Robot self-contact is disabled:
            # adjacent CAD shells overlap, and whole-link hulls would obstruct the fingers.
            if arm:
                ET.SubElement(body, "geom", name=f"collision_{name}_{index}", type="mesh", mesh=mesh_names[filename],
                              pos=fmt(vp), quat=quat(vr), contype="2", conaffinity="1", group="3",
                              friction="0.7 0.005 0.0001", rgba="0.8 0.4 0.1 0.25")
        for j in children[name]:
            jp, jr = origin(j)
            child_name = j.find("child").get("link")
            cb = add_link(child_name, body, jp, jr, frame_p + frame_R @ jp, frame_R @ jr,
                          arm or j.get("name") in ("left_shoulder_pan", "right_shoulder_pan"))
            if j.get("type") != "fixed":
                jname = j.get("name")
                attrs = dict(name=jname, type="slide" if j.get("type") == "prismatic" else "hinge",
                             axis=j.find("axis").get("xyz"), damping="0.1" if jname in WHEELS else "2",
                             armature="0.01" if jname in WHEELS else "0.04")
                if j.get("type") != "continuous":
                    limit = j.find("limit")
                    lower = np.deg2rad(ELBOW_LOWER_DEGREES) if jname.endswith("_elbow_flex") else float(limit.get("lower"))
                    upper = ELEVATOR_UPPER_METERS if jname == "linear_actuator" else float(limit.get("upper"))
                    attrs["range"] = f"{lower:.12g} {upper:.12g}"
                cb.insert(0, ET.Element("joint", attrs))
        return body

    base = add_link("base_link", world, np.zeros(3), CAD_ROTATION, np.zeros(3), np.eye(3))
    base.insert(0, ET.Element("freejoint", name="floating_base"))
    # Compute contact centers from actual wheel mesh bounds, projected onto the CAD axle.
    wheel_info = {}
    for wheel in WHEELS:
        j = next(j for j in joints if j.get("name") == wheel)
        name = j.find("child").get("link")
        bp, br = frames[name]
        axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
        descendant = next(n for n in bodies if "Mecanum_Wheel" in n and bodies[n] in list(bodies[name].iter("body")))
        wp, wr = frames[descendant]
        points = []
        for vis in links[descendant].findall("visual"):
            mesh = vis.find("geometry/mesh")
            vp, vr = origin(vis)
            verts = vertices(SOURCE.parent / mesh.get("filename")) * np.fromstring(mesh.get("scale"), sep=" ")
            points.append((verts @ vr.T + vp) @ wr.T + wp)
        points = np.concatenate(points)
        center = (points.min(0) + points.max(0)) / 2
        center = bp + axis * np.dot(center - bp, axis)
        local = br.T @ (center - bp)
        ET.SubElement(bodies[name], "geom", name=f"contact_{wheel}", type="sphere", size="0.052",
                      pos=fmt(local), contype="2", conaffinity="1", condim="1", friction="0 0 0",
                      group="3", rgba="0.9 0.5 0.1 0.25", solref="0.008 1")
        wheel_info[wheel] = {"body": name, "center_cad": center.tolist(),
                             "center_robot": (CAD_ROTATION @ center).tolist(), "radius": 0.052}
    height = max(0.052 - info["center_cad"][2] for info in wheel_info.values()) + 0.002
    base.set("pos", fmt([0, 0, height]))
    ET.SubElement(base, "geom", name="chassis_collision", type="box", pos="0 0 0.035", size="0.16 0.16 0.055",
                  contype="2", conaffinity="1", group="3", rgba="0.9 0.5 0.1 0.2")
    # Reach point is the wrist-roll origin, unaffected by roll/gripper opening.
    for side in ("left", "right"):
        j = next(j for j in joints if j.get("name") == side + "_wrist_roll")
        ET.SubElement(bodies[j.find("child").get("link")], "site", name=side + "_ee", pos="0 0 0",
                      size="0.008", rgba="0.1 0.9 0.3 1", group="4")
        j = next(j for j in joints if j.get("name") == side + "_shoulder_pan")
        ET.SubElement(bodies[j.find("parent").get("link")], "site", name=side + "_mount", size="0.004", group="4")
    # The camera calibration uses the same CAD-local body frames as this URDF.
    # Its `robot_root` is the generated base_link before the world +45 degree
    # alignment, so the base cameras inherit that alignment automatically.
    calibrated = json.loads(CAMERA_POSES.read_text(encoding="utf-8"))
    for name in ("front_left", "front_right", "bottom", "wrist_left", "wrist_right"):
        pose = calibrated[name]
        body = bodies["base_link" if pose["body"] == "robot_root" else pose["body"]]
        position, orientation = fmt(pose["pos"]), fmt(pose["quat_wxyz"])
        ET.SubElement(body, "camera", name=name, mode="fixed", pos=position,
                      quat=orientation, fovy=f"{pose['fovy_deg']:.12g}")
        ET.SubElement(body, "site", name="camera_mount_" + name, pos=position,
                      type="sphere", size="0.003", rgba="0.1 0.9 0.5 1", group="5")
    lidar = json.loads((SOURCE.parent.parent / "lidar_config.json").read_text(encoding="utf-8"))
    ET.SubElement(bodies[lidar["body"]], "site", name="lidar_scan_origin",
                  pos=fmt(lidar["position_body_m"]), quat=fmt(lidar["quaternion_body_wxyz"]),
                  type="sphere", size="0.004", rgba="1 0.2 0.1 1", group="5")
    actuators = ET.SubElement(root, "actuator")
    for j in joints:
        name = j.get("name")
        if j.get("type") == "fixed":
            continue
        if name in WHEELS:
            ET.SubElement(actuators, "velocity", name=name, joint=name, kv="8", ctrlrange="-35 35",
                          forcerange="-30 30")
        else:
            lim = j.find("limit")
            lift = name == "linear_actuator"
            lower = np.deg2rad(ELBOW_LOWER_DEGREES) if name.endswith("_elbow_flex") else float(lim.get("lower"))
            upper = ELEVATOR_UPPER_METERS if lift else float(lim.get("upper"))
            ET.SubElement(actuators, "position", name=name, joint=name, kp="30000" if lift else "500",
                          kv="3000" if lift else "35", forcerange="-3000 3000" if lift else "-100 100",
                          ctrlrange=f"{lower:.12g} {upper:.12g}")
    ET.SubElement(root, "statistic", center="0 0 0.5", extent="1.4")
    ET.indent(root)
    MODEL.parent.mkdir(exist_ok=True)
    ET.ElementTree(root).write(MODEL, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    assert {model.actuator(i).name for i in range(model.nu)} == set(DEFAULT_POSE)
    assert model.ncam == 5
    assert np.isclose(DEFAULT_POSE["linear_actuator"], ELEVATOR_UPPER_METERS)
    home = model.qpos0.copy()
    controls = np.zeros(model.nu)
    for name, value in DEFAULT_POSE.items():
        joint = model.joint(name)
        home[joint.qposadr[0]] = value if name not in WHEELS else 0.0
        controls[model.actuator(name).id] = value if name not in WHEELS else 0.0
    keyframes = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframes, "key", name="startup", qpos=fmt(home), ctrl=fmt(controls))
    ET.indent(root)
    ET.ElementTree(root).write(MODEL, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    metadata = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "mujoco_version": mujoco.__version__, "links": len(links), "source_joints": len(joints),
                "movable_joints": 17, "meshes": len(mesh_names), "mass_kg": float(model.body_mass.sum()),
                "base_height": height, "simulation_elbow_lower_degrees": ELBOW_LOWER_DEGREES,
                "simulation_elevator_upper_m": ELEVATOR_UPPER_METERS,
                "wheels": wheel_info}
    (MODEL.parent / "build_info.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "wheels"}, indent=2))
    return MODEL


if __name__ == "__main__":
    build()
