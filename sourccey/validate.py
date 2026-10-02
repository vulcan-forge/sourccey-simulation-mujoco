"""Produce quantitative acceptance results and inspection images."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .build import SOURCE, ROLES, WHEELS, ELBOW_LOWER_DEGREES, ELEVATOR_UPPER_METERS, origin, vertices
from .simulation import Simulation
from .app import render, scene_options


def pad_clearance(sim, side, angle):
    """Conservative separating-plane clearance between the two CAD pad surfaces.

    Uses all original STL pad vertices, not convex collision approximations. Positive
    separation proves these two pads do not intersect, but says nothing about other parts.
    """
    sim.reset()
    jid = sim.joints[side + "_gripper"]
    sim.data.qpos[sim.model.jnt_qposadr[jid]] = np.deg2rad(angle)
    mujoco.mj_forward(sim.model, sim.data)
    index = 1 if side == "left" else 2
    base = sim.model.body(f"Gripper_Base_v1_{index}").id
    R = sim.data.xmat[base].reshape(3, 3)
    P = sim.data.xpos[base]
    source = ET.parse(SOURCE).getroot()
    pads = []
    for name in (f"Gripper_Base_v1_{index}", f"Gripper_Finger_v1_{index}"):
        link = source.find(f"./link[@name='{name}']")
        # In this source snapshot the second visual is the pad, first is the shell.
        vis = link.findall("visual")[1]
        mesh = vis.find("geometry/mesh")
        p, r = origin(vis)
        V = (vertices(SOURCE.parent / mesh.get("filename")) * np.fromstring(mesh.get("scale"), sep=" ")) @ r.T + p
        body = sim.model.body(name).id
        V = (V @ sim.data.xmat[body].reshape(3, 3).T + sim.data.xpos[body] - P) @ R
        pads.append(V)
    return float(pads[1][:, 2].min() - pads[0][:, 2].max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sim = Simulation()
    report = {"mujoco": mujoco.__version__, "source_mass_kg": float(sim.model.body_mass.sum()),
              "commanded_joints": sim.model.nu, "drive_mode": "contact-based equivalent mecanum friction",
              "simulation_elbow_lower_degrees": ELBOW_LOWER_DEGREES,
              "simulation_elevator_upper_m": ELEVATOR_UPPER_METERS,
              "startup_linear_actuator_m": float(sim.data.qpos[sim.model.jnt_qposadr[sim.joints["linear_actuator"]]]),
              "startup_elbows_rad": {side: float(sim.data.qpos[sim.model.jnt_qposadr[sim.joints[side + "_elbow_flex"]]])
                                      for side in ("left", "right")},
              "drive": {}, "ik": {}, "gripper_pad_clearance_m": {}}
    for name, command in (("forward", [.5, 0, 0]), ("left", [0, .5, 0]), ("yaw", [0, 0, .5])):
        sim.reset(); sim.step(250)
        start = sim.data.qpos[:3].copy()
        sim.set_base(*command); sim.step(1000)
        report["drive"][name] = {"duration_s": 2, "normalized_command": command,
                                "displacement_m": (sim.data.qpos[:3]-start).tolist(),
                                "yaw_degrees": float(np.rad2deg(np.arctan2(sim.robot_rotation[1, 0], sim.robot_rotation[0, 0]))),
                                "wheel_rates_rad_s": [float(sim.data.qvel[sim.model.jnt_dofadr[sim.joints[n]]]) for n in WHEELS],
                                "warnings": sim.state()["warnings"]}
    for side in ("left", "right"):
        sim.reset(); sim.step(100)
        offset = [.02, .01, -.02] if side == "left" else [.02, -.01, -.02]
        target = sim.ee_local(side) + offset
        result = sim.solve_ik(side, target); sim.step(750)
        result["dynamic_error_m"] = float(np.linalg.norm(sim.ee_local(side)-target))
        report["ik"][side] = result
        report["gripper_pad_clearance_m"][side] = {str(a): pad_clearance(sim, side, a) for a in (-5, 0, 60)}
    sim.reset(); sim.step(500)
    render(sim, args.output / "neutral.png")
    for closure, label in ((100, "closed"), (0, "open")):
        sim.reset()
        for side in ("left", "right"):
            sim.set_gripper(side, closure)
        sim.step(750)
        render(sim, args.output / f"grippers_{label}.png")
        from PIL import Image
        with mujoco.Renderer(sim.model, height=700, width=1000) as renderer:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = sim.data.site_xpos[sim.model.site("left_ee").id] + [.06, 0, 0]
            cam.distance, cam.azimuth, cam.elevation = .30, 90, -15
            renderer.update_scene(sim.data, camera=cam, scene_option=scene_options())
            Image.fromarray(renderer.render()).save(args.output / f"left_gripper_{label}_detail.png")
    for position_m, label in ((-.3104, "bottom"), (-.0142, "top")):
        sim.reset(); sim.set_elevator(position_m); sim.step(1500)
        render(sim, args.output / f"elevator_{label}.png")
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
