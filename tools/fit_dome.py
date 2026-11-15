"""Offline sphere fit: python -m tools.fit_dome. Prints JSON; never changes assets."""
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np

from sourccey.build import SOURCE, origin, vertices

SHELL = ("Dome_Top_v1_1", "Dome_Front_v1_1", "Dome_Back_v1_1", "Dome_Left_v1_1", "Dome_Right_v1_1")


def fit_sphere(points):
    # |p-c|² = r² -> 2 p.c + (r²-|c|²) = |p|².
    solution = np.linalg.lstsq(np.c_[2*points, np.ones(len(points))], (points**2).sum(1), rcond=None)[0]
    return solution[:3], float(np.sqrt(solution[3] + solution[:3] @ solution[:3]))


def fit_dome():
    source = ET.parse(SOURCE).getroot()
    links = {link.get("name"): link for link in source.findall("link")}
    children = {name: [] for name in links}
    for joint in source.findall("joint"):
        children[joint.find("parent").get("link")].append(joint)
    frames = {}

    def visit(name, position, rotation):
        frames[name] = (position, rotation)
        for joint in children[name]:
            jp, jr = origin(joint)
            visit(joint.find("child").get("link"), position + rotation @ jp, rotation @ jr)

    visit("base_link", np.zeros(3), np.eye(3))
    points, hashes = [], {}
    for name in SHELL:
        position, rotation = frames[name]
        for visual in links[name].findall("visual"):
            mesh = visual.find("geometry/mesh")
            path = SOURCE.parent / mesh.get("filename")
            hashes[mesh.get("filename")] = hashlib.sha256(path.read_bytes()).hexdigest()
            vp, vr = origin(visual)
            verts = vertices(path) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            points.append((verts @ vr.T + vp) @ rotation.T + position)
    points = np.unique(np.concatenate(points).round(6), axis=0)
    center, radius = fit_sphere(points)
    outer = points[np.linalg.norm(points-center, axis=1) > radius]
    center, radius = fit_sphere(outer)
    return {"center_base_m": center.tolist(), "radius_m": radius,
            "fit_rms_m": float(np.sqrt(np.mean((np.linalg.norm(outer-center, axis=1)-radius)**2))),
            "method": "Two-pass least-squares sphere fit to the outer surface of the five shell meshes in base_link coordinates; excludes Dome_Plate parts. Reproduce with python -m tools.fit_dome.",
            "source_urdf_sha256_lf": hashlib.sha256(SOURCE.read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
            "source_mesh_sha256": hashes}


if __name__ == "__main__":
    print(json.dumps(fit_dome(), indent=2))
