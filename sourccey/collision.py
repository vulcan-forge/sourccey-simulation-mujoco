"""Scene-local replacement of CAD hulls with cached convex decompositions."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from .build import ROOT, SOURCE
from .build import MODEL


COLLISION_ASSETS = ROOT / "models/collision"

# Millimetre-scale mating surfaces already constrained by the hinge, not
# external hard stops. These CAD interfaces overlap even at the parked pose.
# Bounds are (axial minimum, axial maximum, radius), in the joint's frame, m.
BEARING_CLEARANCES = {
    "elbow_flex": (-.0225, -.0195, .018),
    "wrist_roll": (-.0015, .0025, .0125),
}


def _convex_clip(points, normal, offset):
    """Keep the halfspace normal @ point <= offset of a convex polyhedron."""
    from scipy.spatial import ConvexHull
    distances = points @ normal - offset
    inside = distances <= 1e-10
    if inside.all():
        return points
    if not inside.any():
        return None
    hull = ConvexHull(points)
    edges = np.unique(np.sort(np.concatenate([hull.simplices[:, [0, 1]],
                                              hull.simplices[:, [1, 2]],
                                              hull.simplices[:, [2, 0]]]), axis=1), axis=0)
    edges = edges[inside[edges[:, 0]] != inside[edges[:, 1]]]
    a, b = edges[:, 0], edges[:, 1]
    intersections = points[a] + (points[b] - points[a]) * (
        distances[a] / (distances[a] - distances[b]))[:, None]
    result = np.unique(np.round(np.concatenate([points[inside], intersections]), 12), axis=0)
    if len(result) < 4 or np.linalg.matrix_rank(result - result[0], tol=1e-10) < 3:
        return None
    return result[ConvexHull(result).vertices]


def bearing_recess(points, center, axis, bounds):
    """Subtract a bounded 32-sided bearing bore, retaining convex outer pieces.

    Splitting matters: one hull around the remaining vertices would fill the
    bore again. Only collision geometry is cut; visible CAD and inertia stay put.
    """
    low, high, radius = bounds
    axis = axis / np.linalg.norm(axis)
    radial = np.cross(axis, np.eye(3)[np.argmin(abs(axis))])
    radial /= np.linalg.norm(radial)
    tangent = np.cross(axis, radial)
    angles = np.arange(32) * (2 * np.pi / 32)
    normals = np.concatenate([axis[None], -axis[None],
                              np.cos(angles)[:, None] * radial + np.sin(angles)[:, None] * tangent])
    offsets = np.r_[high, -low, np.full(32, radius)] + normals @ center
    distances = points @ normals.T - offsets
    if np.any(np.all(distances > 1e-10, axis=0)):
        return [points]
    remaining, outside = points, []
    for normal, offset in zip(normals, offsets):
        piece = _convex_clip(remaining, -normal, -offset)
        if piece is not None:
            outside.append(piece)
        remaining = _convex_clip(remaining, normal, offset)
        if remaining is None:
            break
    return outside


@lru_cache(maxsize=1)
def bearing_frames(model_revision):
    """Locate only the diagnosed bearing interfaces, including mirrored arms."""
    spec = mujoco.MjSpec.from_file(str(MODEL))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = {}
    for side, assembly in (("left", 1), ("right", 2)):
        for role, geom_name in (("elbow_flex", f"collision_Motor_3_support_v1_{assembly}_0"),
                                ("wrist_roll", f"collision_Gripper_Base_v1_{assembly}_0")):
            geom = spec.geom(geom_name)
            bid = model.body(geom.parent.name).id
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, geom.quat)
            rotation = data.xmat[bid].reshape(3, 3) @ rotation.reshape(3, 3)
            origin = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ geom.pos
            jid = model.joint(f"{side}_{role}").id
            bounds = BEARING_CLEARANCES[role]
            if side == "right" and role == "elbow_flex":
                # This hinge's axial direction is mirrored along with its arm.
                bounds = (-bounds[1], -bounds[0], bounds[2])
            result[geom_name] = ((data.xanchor[jid] - origin) @ rotation,
                                 data.xaxis[jid] @ rotation, bounds)
    return result


@lru_cache(maxsize=1)
def torso_envelopes(model_revision):
    """CAD-derived solid outer envelopes; the torso interior is not arm space.

    Separate head/body hulls preserve their actual external profiles. Shoulder
    pivots are outside these envelopes; arm hinge cavities still use the fine
    decomposition, where filling a recess would create a false joint stop.
    """
    from scipy.spatial import ConvexHull
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("startup").id)
    mujoco.mj_forward(model, data)
    base = model.body("base_link").id
    rotation = data.xmat[base].reshape(3, 3)
    groups = {"torso": [], "head": []}
    sources = {"torso": [], "head": []}
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if name.startswith(("visual_Second_Level_", "visual_Third_Level_", "visual_Bottom_Wall_",
                            "visual_Door_Assembly_", "visual_Breastplate_")):
            group = "torso"
        elif name.startswith(("visual_Dome_Front_", "visual_Dome_Left_", "visual_Dome_Right_",
                              "visual_Dome_Back_")):
            group = "head"
        else:
            continue
        mid = model.geom_dataid[gid]
        points = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        world = points @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        groups[group].append((world - data.xpos[base]) @ rotation)
        sources[group].append(name)
    result = {}
    for group, arrays in groups.items():
        points = np.unique(np.concatenate(arrays), axis=0)
        hull = ConvexHull(points)
        result[group] = {"vertices": points[hull.vertices], "sources": sources[group]}
    return result


def add_torso_collision(spec):
    envelopes = torso_envelopes(MODEL.stat().st_mtime_ns)
    for name, record in envelopes.items():
        mesh_name = f"calibration_{name}_envelope"
        spec.add_mesh(name=mesh_name, uservert=record["vertices"].ravel())
        spec.body("base_link").add_geom(name=mesh_name, type=mujoco.mjtGeom.mjGEOM_MESH,
                                       meshname=mesh_name, contype=2, conaffinity=3,
                                       group=3, rgba=[.15, .65, .9, .25], density=0,
                                       friction=[.7, .005, .0001])


def decomposed_collision(spec, directory=COLLISION_ASSETS, *, subset=None, manifest_name="manifest.json",
                         bearing_clearance=True):
    directory = Path(directory)
    manifest = json.loads((directory / manifest_name).read_text(encoding="utf-8"))
    originals = [g for g in spec.geoms if g.name.startswith("collision_")
                 and (subset is None or g.meshname in subset)]
    expected = {g.meshname for g in originals}
    if not expected <= manifest["meshes"].keys():
        raise ValueError("Collision decomposition does not cover every original arm mesh")
    # Assets are shared when multiple geoms refer to the same source mesh.
    meshes, vertices_by_mesh = {}, {}
    for name in sorted(expected):
        record = manifest["meshes"][name]
        path = directory / record["asset"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["asset_sha256"]:
            raise ValueError(f"Collision asset checksum mismatch: {path.name}")
        original = spec.mesh(name)
        # Byte-identical CAD assets have different filenames on mirrored arms.
        source = SOURCE.parent / original.file
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["source_sha256"]:
            raise ValueError(f"Collision source changed: {source.name}")
        if not np.allclose(original.scale, record["scale"]):
            raise ValueError(f"Collision mesh frame/scale changed: {name}")
        meshes[name] = []
        vertices_by_mesh[name] = []
        with np.load(path, allow_pickle=False) as archive:
            for i in range(record["parts"]):
                part_name = f"contact_{name}_{i}"
                spec.add_mesh(name=part_name, uservert=archive[f"v{i}"].ravel(),
                              userface=archive[f"f{i}"].ravel())
                meshes[name].append(part_name)
                vertices_by_mesh[name].append(archive[f"v{i}"].copy())
    clearances = bearing_frames(MODEL.stat().st_mtime_ns) if bearing_clearance else {}
    for geom in originals:
        parts = []
        for i, name in enumerate(meshes[geom.meshname]):
            if geom.name in clearances:
                for k, points in enumerate(bearing_recess(vertices_by_mesh[geom.meshname][i],
                                                         *clearances[geom.name])):
                    name = f"bearing_{geom.name}_{i}_{k}"
                    spec.add_mesh(name=name, uservert=points.ravel())
                    parts.append((f"decomposed_{geom.name}_{i}_clearance_{k}", name))
            else:
                parts.append((f"decomposed_{geom.name}_{i}", name))
        for geom_name, name in parts:
            geom.parent.add_geom(name=geom_name, type=mujoco.mjtGeom.mjGEOM_MESH,
                                 meshname=name, pos=geom.pos.copy(), quat=geom.quat.copy(),
                                 contype=geom.contype, conaffinity=geom.conaffinity,
                                 condim=geom.condim, friction=geom.friction.copy(),
                                 solref=geom.solref.copy(), solimp=geom.solimp.copy(),
                                 margin=geom.margin, gap=geom.gap, group=3,
                                 rgba=[.8, .4, .1, .25], density=0)
        geom.contype = geom.conaffinity = 0
        geom.group = 5  # Old hulls cannot reappear in the collision-only view.
    return {"generator": manifest["generator"], "version": manifest["version"],
            "source_meshes": len(expected), "convex_parts": sum(map(len, meshes.values())),
            "bearing_clearance_geoms": sorted(set(clearances) & {g.name for g in originals}),
            "parameters": manifest["parameters"]}
