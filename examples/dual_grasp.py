"""Two real-contact grasp checks: close both hands, lift, hold, and release.

The scene adds small convex fingertip pads because the source CAD hulls overlap.
There are no welds, attachment constraints, or manually moved objects.
Git Bash: python -m examples.dual_grasp
"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from sourccey.build import MODEL
from sourccey.simulation import Simulation
from examples.sorting_shift import overview_camera
from sourccey.cameras import CAMERAS, FEED_SIZE, LABEL_HEIGHT


SIDES = ("left", "right")
GRIPPERS = {
    "left": ("Gripper_Base_v1_1", "Gripper_Finger_v1_1", [0.08, 0, 0]),
    "right": ("Gripper_Base_v1_2", "Gripper_Finger_v1_2", [0, -0.08, 0]),
}


def look_at_quat(origin, target):
    forward = np.asarray(target, dtype=float) - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return Rotation.from_matrix(np.column_stack((right, up, -forward))).as_quat()[[3, 0, 1, 2]]


def scene_model():
    # Find the hand frames in the scene-only starting pose. The production
    # robot model and its normal startup keyframe are left unchanged.
    source = mujoco.MjModel.from_xml_path(str(MODEL))
    pose = mujoco.MjData(source)
    mujoco.mj_resetDataKeyframe(source, pose, source.key("startup").id)
    for side in SIDES:
        pose.qpos[source.joint(f"{side}_wrist_flex").qposadr[0]] = 0
        pose.qpos[source.joint(f"{side}_gripper").qposadr[0]] = np.deg2rad(60)
    pose.qpos[source.joint("linear_actuator").qposadr[0]] = -0.20
    mujoco.mj_forward(source, pose)

    spec = mujoco.MjSpec.from_file(str(MODEL))
    for side in SIDES:
        base_name, finger_name, center_local = GRIPPERS[side]
        # Replace just the oversized gripper hulls with simple solid pads.
        # Other robot collision geoms remain as authored in the base model.
        for name in (base_name, finger_name):
            for i in (0, 1):
                geom = spec.geom(f"collision_{name}_{i}")
                geom.contype = 0
                geom.conaffinity = 0
        fixed = [0.08, 0, -0.008] if side == "left" else [0, -0.08, -0.008]
        moving = [0.04, 0, -0.004] if side == "left" else [0, -0.04, -0.004]
        for body, label, local in ((base_name, "fixed", fixed), (finger_name, "moving", moving)):
            half_size = ([0.04, 0.014, 0.004] if side == "left"
                         else [0.014, 0.04, 0.004])
            if label == "moving":
                half_size[2] = 0.006
            spec.body(body).add_geom(
                name=f"{side}_{label}_pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=local, size=half_size,
                contype=2, conaffinity=1, condim=4,
                friction=[2.5, 0.03, 0.002], solref=[0.008, 1],
                rgba=[0.9, 0.9, 0.75, 0.22],
            )
        # Diagnostic cameras follow the palms; the five calibrated camera
        # mounts in the production model remain untouched.
        camera_pos = ([0.08, -0.12, 0.07] if side == "left"
                      else [0.12, -0.08, 0.07])
        camera_target = np.array(center_local) + [0, 0, 0.004]
        spec.body(base_name).add_camera(
            name=f"{side}_grasp_inspection", pos=camera_pos,
            quat=look_at_quat(np.array(camera_pos), camera_target), fovy=50,
        )
        base_id = source.body(base_name).id
        rotation = pose.xmat[base_id].reshape(3, 3)
        world = pose.xpos[base_id] + rotation @ (np.array(center_local) + [0, 0, 0.001])
        bead = spec.worldbody.add_body(
            name=f"{side}_grasp_object", pos=world,
            mass=0.008, ipos=[0, 0, 0], inertia=[1e-6] * 3,
        )
        bead.add_freejoint(name=f"{side}_object_free")
        bead.add_geom(
            name=f"{side}_object_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=([0.014, 0.026, 0.006] if side == "left" else [0.026, 0.014, 0.006]),
            contype=3, conaffinity=3, condim=4,
            friction=[2.5, 0.03, 0.002], solref=[0.008, 1],
            rgba=([0.95, 0.22, 0.12, 1] if side == "left" else [0.98, 0.85, 0.10, 1]),
        )

    model = spec.compile()
    key = model.key("startup").id
    for name, target in (("linear_actuator", -0.20),
                         ("left_wrist_flex", 0), ("right_wrist_flex", 0),
                         ("left_gripper", np.deg2rad(60)),
                         ("right_gripper", np.deg2rad(60))):
        model.key_qpos[key, model.joint(name).qposadr[0]] = target
        model.key_ctrl[key, model.actuator(name).id] = target
    for side in SIDES:
        adr = model.joint(f"{side}_object_free").qposadr[0]
        model.key_qpos[key, adr:adr + 7] = model.qpos0[adr:adr + 7]
    return model


def grasp_frame(model, data, renderer, overview):
    width, height = FEED_SIZE
    frame = Image.new("RGB", (width * 4, (height + LABEL_HEIGHT) * 2), (15, 18, 20))
    order = (CAMERAS[0], CAMERAS[1], CAMERAS[2], "overview",
             CAMERAS[3], CAMERAS[4], "left_grasp_inspection", "right_grasp_inspection")
    draw = ImageDraw.Draw(frame)
    for index, name in enumerate(order):
        renderer.update_scene(data, camera=overview if name == "overview" else name)
        x, y = (index % 4) * width, (index // 4) * (height + LABEL_HEIGHT)
        frame.paste(Image.fromarray(renderer.render()), (x, y + LABEL_HEIGHT))
        draw.text((x + 6, y + 4), name.replace("_", " "), fill=(100, 240, 150))
    return frame


class GraspDashboard:
    def __init__(self, simulation):
        import tkinter as tk
        from PIL import ImageTk

        self.ImageTk = ImageTk
        self.simulation = simulation
        self.root = tk.Tk()
        self.root.title("Sourccey | Two-hand contact grasp")
        self.root.resizable(False, False)
        self.status = tk.StringVar(value="Preparing two-hand contact grasp")
        tk.Label(self.root, textvariable=self.status, font=("Segoe UI", 12, "bold"),
                 anchor="w", padx=8, pady=6).pack(fill="x")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.renderer = mujoco.Renderer(simulation.model, height=FEED_SIZE[1], width=FEED_SIZE[0])
        self.overview = overview_camera()
        self.frame = None
        self.photo = None
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def update(self, message):
        if self.closed:
            return
        self.status.set(message)
        self.frame = grasp_frame(self.simulation.model, self.simulation.data,
                                 self.renderer, self.overview)
        self.photo = self.ImageTk.PhotoImage(self.frame, master=self.root)
        self.label.configure(image=self.photo)
        self.root.update()

    def close(self):
        if not self.closed:
            self.closed = True
            self.renderer.close()
            self.root.destroy()


def pad_contacts(sim, side):
    obj = sim.model.geom(f"{side}_object_geom").id
    pads = {label: sim.model.geom(f"{side}_{label}_pad").id
            for label in ("fixed", "moving")}
    touching = set()
    for contact in sim.data.contact:
        pair = {int(contact.geom1), int(contact.geom2)}
        for label, pad in pads.items():
            if pair == {obj, pad}:
                touching.add(label)
    return touching


def run(sim, dashboard=None, playback=1.0, on_hold=None):
    start_wall = time.perf_counter()
    next_frame = 0.0
    object_ids = {side: sim.model.body(f"{side}_grasp_object").id for side in SIDES}
    palm_ids = {side: sim.model.body(GRIPPERS[side][0]).id for side in SIDES}
    initial = {side: sim.data.xpos[object_ids[side]].copy() for side in SIDES}
    contact_during_hold = {side: 0 for side in SIDES}
    max_relative_drift = {side: 0.0 for side in SIDES}
    initial_relative = {side: initial[side] - sim.data.xpos[palm_ids[side]].copy()
                        for side in SIDES}

    def advance(steps, status, inspect=False):
        nonlocal next_frame
        for _ in range(steps):
            if dashboard is not None and dashboard.closed:
                raise KeyboardInterrupt
            sim.step()
            if inspect:
                for side in SIDES:
                    if pad_contacts(sim, side) == {"fixed", "moving"}:
                        contact_during_hold[side] += 1
                    relative = sim.data.xpos[object_ids[side]] - sim.data.xpos[palm_ids[side]]
                    max_relative_drift[side] = max(
                        max_relative_drift[side], float(np.linalg.norm(relative - initial_relative[side])))
            if dashboard is not None and sim.data.time >= next_frame:
                dashboard.update(status)
                next_frame = sim.data.time + max(0.2, min(0.8, playback / 5))
            remaining = start_wall + sim.data.time / playback - time.perf_counter()
            if dashboard is not None and remaining > 0:
                time.sleep(remaining)

    advance(100, "Two objects resting in the open palms")
    for closure in np.linspace(0, 100, 61)[1:]:
        for side in SIDES:
            sim.set_gripper(side, closure)
        advance(8, "Both grippers closing around solid objects")
    advance(250, "Verifying contact on both sides of each object", inspect=True)
    closed_contacts = {side: sorted(pad_contacts(sim, side)) for side in SIDES}
    for elevator in np.linspace(-0.20, -0.0142, 101)[1:]:
        sim.set_elevator(elevator)
        advance(10, "Both hands lifting their objects through contact", inspect=True)
    advance(600, "Holding both objects above the starting height", inspect=True)
    lifted = {side: float(sim.data.xpos[object_ids[side], 2] - initial[side][2])
              for side in SIDES}
    held_contacts = {side: sorted(pad_contacts(sim, side)) for side in SIDES}
    if on_hold is not None:
        on_hold()
    for closure in np.linspace(100, 0, 61)[1:]:
        for side in SIDES:
            sim.set_gripper(side, closure)
        advance(8, "Opening both hands and releasing the objects")
    for angle in np.linspace(0, 1.2, 61)[1:]:
        sim.set_joints({"left_wrist_flex": angle, "right_wrist_flex": -angle})
        advance(8, "Tipping open hands to let both objects fall")
    advance(1000, "Objects falling freely after release")
    released = {side: float(sim.data.xpos[object_ids[side], 2] - initial[side][2])
                for side in SIDES}
    elapsed = time.perf_counter() - start_wall
    report = {
        "grasp_mode": "contact only; no weld or object pose commands",
        "closed_contacts": closed_contacts,
        "held_contacts": held_contacts,
        "hold_steps_with_both_pads_in_contact": contact_during_hold,
        "lift_m": {s: round(v, 4) for s, v in lifted.items()},
        "max_relative_drift_m": {s: round(v, 4) for s, v in max_relative_drift.items()},
        "released_height_vs_start_m": {s: round(v, 4) for s, v in released.items()},
        "simulated_seconds": round(sim.data.time, 2),
        "wall_seconds": round(elapsed, 2),
        "warnings": sim.state()["warnings"],
    }
    report["passed"] = (all(set(closed_contacts[s]) == {"fixed", "moving"} for s in SIDES)
                        and all(set(held_contacts[s]) == {"fixed", "moving"} for s in SIDES)
                        and all(contact_during_hold[s] > 500 for s in SIDES)
                        and all(lifted[s] > 0.12 for s in SIDES)
                        and all(max_relative_drift[s] < 0.03 for s in SIDES)
                        and all(released[s] < -0.65 for s in SIDES)
                        and not report["warnings"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--exit-on-complete", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be a positive finite number")
    sim = Simulation(model=scene_model())
    dashboard = None if args.headless else GraspDashboard(sim)
    try:
        def save_hold():
            if args.snapshot is None:
                return
            hold_path = args.snapshot.with_name(f"{args.snapshot.stem}_hold{args.snapshot.suffix}")
            hold_path.parent.mkdir(parents=True, exist_ok=True)
            if dashboard is None:
                with mujoco.Renderer(sim.model, height=FEED_SIZE[1], width=FEED_SIZE[0]) as renderer:
                    grasp_frame(sim.model, sim.data, renderer, overview_camera()).save(hold_path)
            else:
                dashboard.update("Both objects held by solid finger contact")
                dashboard.frame.save(hold_path)
        try:
            report = run(sim, dashboard, args.speed, on_hold=save_hold)
        except KeyboardInterrupt:
            print("Grasp test stopped.")
            return
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            if dashboard is None:
                with mujoco.Renderer(sim.model, height=FEED_SIZE[1], width=FEED_SIZE[0]) as renderer:
                    grasp_frame(sim.model, sim.data, renderer, overview_camera()).save(args.snapshot)
            else:
                dashboard.update("Contact grasp test complete")
                dashboard.frame.save(args.snapshot)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
        if dashboard is not None and not args.exit_on_complete:
            dashboard.status.set("Contact grasp test complete | close to exit")
            while not dashboard.closed:
                dashboard.root.update()
                time.sleep(0.03)
    finally:
        if dashboard is not None:
            dashboard.close()


if __name__ == "__main__":
    main()
