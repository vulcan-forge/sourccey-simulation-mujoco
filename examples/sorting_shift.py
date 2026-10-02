"""Five-camera sorting shift: two assisted pickups and a two-puck lane clear.

Git Bash: python -m examples.sorting_shift
Use --headless for a measured run without the live camera dashboard.
Playback targets 1.3x wall-clock speed, about one third of the previous demo.
"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from scipy.spatial.transform import Rotation

from sourccey.build import MODEL
from sourccey.cameras import FEED_SIZE, LABEL_HEIGHT, render_tiled
from sourccey.pose import DEFAULT_POSE
from sourccey.simulation import Simulation


SIDES = ("left", "right")
TABLE_Y = {"left": 0.5, "right": -0.5}
PICK_Y = {"left": 0.34, "right": -0.34}
DROP_Y = {"left": 0.44, "right": -0.44}


def scene_model():
    spec = mujoco.MjSpec.from_file(str(MODEL))
    for side in SIDES:
        y = TABLE_Y[side]
        spec.worldbody.add_geom(
            name=f"{side}_table", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.86, y, 0.68], size=[0.25, 0.25, 0.02],
            contype=1, conaffinity=3, rgba=[0.33, 0.25, 0.19, 1],
        )
        for i, x in enumerate((0.68, 1.04)):
            for j, leg_y in enumerate((y - 0.18, y + 0.18)):
                spec.worldbody.add_geom(
                    name=f"{side}_table_leg_{i}_{j}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[x, leg_y, 0.33], size=[0.025, 0.025, 0.33],
                    contype=1, conaffinity=3, rgba=[0.25, 0.19, 0.15, 1],
                )
        block = spec.worldbody.add_body(
            name=f"{side}_parcel", pos=[0.70, PICK_Y[side], 0.74],
            mass=0.08, ipos=[0, 0, 0], inertia=[0.0001] * 3,
        )
        block.add_freejoint(name=f"{side}_parcel_free")
        block.add_geom(
            name=f"{side}_parcel_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.04], contype=2, conaffinity=1,
            rgba=([0.9, 0.25, 0.16, 1] if side == "left" else [0.18, 0.65, 0.9, 1]),
        )
        block.add_site(name=f"{side}_parcel_grasp", size=[0.003])
        palm = "Gripper_Base_v1_1" if side == "left" else "Gripper_Base_v1_2"
        spec.body(palm).add_site(name=f"{side}_palm_grasp", size=[0.003])
        spec.add_equality(
            name=f"{side}_assisted_grasp", type=mujoco.mjtEq.mjEQ_WELD,
            name1=f"{side}_palm_grasp", name2=f"{side}_parcel_grasp",
            objtype=mujoco.mjtObj.mjOBJ_SITE, active=0,
        )
        # A colored landing zone makes the sorting objective legible from the
        # overview, while remaining non-colliding decoration.
        spec.worldbody.add_geom(
            name=f"{side}_landing_zone", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.76, DROP_Y[side], 0.702], size=[0.065, 0.065, 0.002],
            contype=0, conaffinity=0,
            rgba=([0.45, 0.09, 0.08, 1] if side == "left" else [0.06, 0.3, 0.5, 1]),
        )

    for index, (x, y) in enumerate(((0.94, -0.06), (1.14, 0.06)), 1):
        puck = spec.worldbody.add_body(
            name=f"lane_puck_{index}", pos=[x, y, 0.065],
            mass=0.06, ipos=[0, 0, 0], inertia=[0.00009] * 3,
        )
        puck.add_freejoint(name=f"lane_puck_{index}_free")
        puck.add_geom(
            name=f"lane_puck_{index}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.065], contype=3, conaffinity=3,
            friction=[0.6, 0.005, 0.0001],
            rgba=([0.88, 0.8, 0.18, 1] if index == 1 else [0.58, 0.3, 0.8, 1]),
        )
    model = spec.compile()
    key = model.key("startup").id
    for name in ("left_parcel_free", "right_parcel_free", "lane_puck_1_free", "lane_puck_2_free"):
        adr = model.joint(name).qposadr[0]
        model.key_qpos[key, adr:adr + 7] = model.qpos0[adr:adr + 7]
    return model


def overview_camera():
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.65, 0, 0.65]
    camera.distance = 2.6
    camera.azimuth = 135
    camera.elevation = -22
    return camera


def dashboard_frame(model, data, renderer, overview):
    image = render_tiled(model, data, renderer)
    renderer.update_scene(data, camera=overview)
    tile = Image.fromarray(renderer.render())
    x, y = 2 * FEED_SIZE[0], FEED_SIZE[1] + LABEL_HEIGHT
    image.paste(tile, (x, y + LABEL_HEIGHT))
    ImageDraw.Draw(image).text((x + 6, y + 4), "overview", fill=(100, 240, 150))
    return image


class Dashboard:
    def __init__(self, simulation):
        import tkinter as tk

        self.simulation = simulation
        self.root = tk.Tk()
        self.root.title("Sourccey | The sorting shift")
        self.root.resizable(False, False)
        self.status = tk.StringVar(value="Preparing the sorting shift")
        tk.Label(self.root, textvariable=self.status, font=("Segoe UI", 12, "bold"),
                 anchor="w", padx=8, pady=6).pack(fill="x")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.photo = None
        self.frame = None
        self.renderer = mujoco.Renderer(simulation.model, height=FEED_SIZE[1], width=FEED_SIZE[0])
        self.overview = overview_camera()
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def update(self, message):
        if self.closed:
            return
        self.status.set(message)
        image = dashboard_frame(self.simulation.model, self.simulation.data,
                                self.renderer, self.overview)
        self.frame = image
        self.photo = ImageTk.PhotoImage(image, master=self.root)
        self.label.configure(image=self.photo)
        self.root.update()

    def close(self):
        if not self.closed:
            self.closed = True
            self.renderer.close()
            self.root.destroy()


def attach(sim, side):
    grip = sim.model.eq(f"{side}_assisted_grasp").id
    palm_name = "Gripper_Base_v1_1" if side == "left" else "Gripper_Base_v1_2"
    palm_id = sim.model.body(palm_name).id
    parcel_id = sim.model.body(f"{side}_parcel").id
    palm_p, parcel_p = sim.data.xpos[palm_id].copy(), sim.data.xpos[parcel_id].copy()
    palm_r = Rotation.from_quat(sim.data.xquat[palm_id][[1, 2, 3, 0]])
    parcel_r = Rotation.from_quat(sim.data.xquat[parcel_id][[1, 2, 3, 0]])
    site_id = sim.model.site(f"{side}_parcel_grasp").id
    sim.model.site_pos[site_id] = parcel_r.inv().apply(palm_p - parcel_p)
    sim.model.site_quat[site_id] = (parcel_r.inv() * palm_r).as_quat()[[3, 0, 1, 2]]
    sim.data.eq_active[grip] = True
    return grip


def run(sim, dashboard=None, playback=1.3):
    start_wall = time.perf_counter()
    start_sim = sim.data.time
    dt = sim.model.opt.timestep
    next_render = 0.0
    notes = []
    max_joint_speed = 0.0
    worst_ik_error = 0.0
    worst_ik_phase = ""
    largest_target_step = 0.0
    arm_dofs = [sim.model.jnt_dofadr[sim.joints[f"{side}_{role}"]]
                for side in SIDES for role in ("shoulder_pan", "shoulder_lift", "elbow_flex")]

    def advance(steps, message):
        nonlocal next_render, max_joint_speed
        for _ in range(steps):
            if dashboard is not None and dashboard.closed:
                raise KeyboardInterrupt("Camera window closed")
            sim.step()
            max_joint_speed = max(max_joint_speed, float(np.max(np.abs(sim.data.qvel[arm_dofs]))))
            if dashboard is not None and sim.data.time >= next_render:
                dashboard.update(message)
                # About five dashboard frames per wall-clock second at the
                # requested playback rate, without over-rendering fast runs.
                next_render = sim.data.time + max(0.2, min(1.0, playback / 5))
            deadline = start_wall + (sim.data.time - start_sim) / playback
            remaining = deadline - time.perf_counter()
            if dashboard is not None and remaining > 0:
                time.sleep(remaining)

    def local(side, world):
        return sim.robot_rotation.T @ (world - sim.mount_position(side))

    def move_to(side, world, message, segments=55, steps_per_segment=8):
        nonlocal worst_ik_error, worst_ik_phase, largest_target_step
        source, destination = sim.ee_local(side), local(side, world)
        names = [f"{side}_{role}" for role in ("shoulder_pan", "shoulder_lift", "elbow_flex")]
        for fraction in np.linspace(0, 1, segments + 1)[1:]:
            old = np.array([sim.data.ctrl[sim.actuators[name]] for name in names])
            result = sim.solve_ik(side, source + fraction * (destination - source))
            new = np.array([sim.data.ctrl[sim.actuators[name]] for name in names])
            if result["error_m"] > worst_ik_error:
                worst_ik_error = result["error_m"]
                worst_ik_phase = message
            largest_target_step = max(largest_target_step, float(np.max(np.abs(new - old))))
            advance(steps_per_segment, message)
        return float(np.linalg.norm(sim.data.site_xpos[sim.model.site(f"{side}_ee").id] - world))

    initial_parcel = {side: sim.data.xpos[sim.model.body(f"{side}_parcel").id].copy() for side in SIDES}
    initial_pucks = {i: sim.data.xpos[sim.model.body(f"lane_puck_{i}").id].copy() for i in (1, 2)}

    sim.set_base(forward=0.25)
    start_x = float(sim.data.qpos[0])
    while sim.data.qpos[0] - start_x < 0.20 and sim.data.time < 4:
        advance(1, "Arriving at the two sorting tables")
    sim.set_base()
    advance(180, "Parking at the worktables")

    for side, color in (("left", "red"), ("right", "blue")):
        parcel_id = sim.model.body(f"{side}_parcel").id
        parcel_start = sim.data.xpos[parcel_id].copy()
        sim.set_gripper(side, 0)
        advance(90, f"Opening the {side} gripper for the {color} parcel")
        approach = parcel_start + [-0.035, 0, 0.105]
        move_to(side, approach + [0, 0, 0.07], f"Approaching the {color} parcel")
        approach_error = move_to(side, approach, f"Lining up the {color} parcel")
        advance(120, f"Settling at the {color} parcel")
        distance = float(np.linalg.norm(sim.data.site_xpos[sim.model.site(f"{side}_ee").id] - sim.data.xpos[parcel_id]))
        if approach_error > 0.03 or distance > 0.15:
            raise RuntimeError(f"{side} arm missed its parcel ({approach_error:.3f} m)")
        sim.set_gripper(side, 100)
        advance(100, f"Closing the {side} gripper")
        grip = attach(sim, side)
        advance(60, f"Assisted grasp on the {color} parcel")
        move_to(side, approach + [0, 0, 0.17], f"Lifting the {color} parcel")
        drop = np.array([0.76, DROP_Y[side], 0.84])
        move_to(side, drop, f"Sorting the {color} parcel")
        sim.data.eq_active[grip] = False
        sim.set_gripper(side, 0)
        advance(300, f"Releasing the {color} parcel")
        final = sim.data.xpos[parcel_id].copy()
        zone_error = float(np.linalg.norm(final[:2] - [0.76, DROP_Y[side]]))
        notes.append({"parcel": color,
                      "travel_m": round(float(np.linalg.norm(final[:2] - initial_parcel[side][:2])), 3),
                      "distance_to_zone_m": round(zone_error, 3),
                      "landed_on_table": bool(0.68 < final[2] < 0.8 and zone_error < 0.12)})
        names = [f"{side}_{role}" for role in ("shoulder_pan", "shoulder_lift", "elbow_flex")]
        current = np.array([sim.data.ctrl[sim.actuators[name]] for name in names])
        home = np.array([DEFAULT_POSE[name] for name in names])
        for fraction in np.linspace(0, 1, 81)[1:]:
            target = current + fraction * (home - current)
            old = np.array([sim.data.ctrl[sim.actuators[name]] for name in names])
            largest_target_step = max(largest_target_step, float(np.max(np.abs(target - old))))
            sim.set_joints(dict(zip(names, target)))
            advance(8, f"Retracting the {side} arm")
        sim.set_gripper(side, 100)

    sim.set_base(forward=0.55)
    while sim.data.qpos[0] - start_x < 1.04 and sim.data.time < 35:
        advance(1, "Clearing two stray pucks from the lane")
    sim.set_base()
    advance(350, "The lane is clear; inspecting the result")
    puck_travel = {str(i): round(float(np.linalg.norm(
        sim.data.xpos[sim.model.body(f"lane_puck_{i}").id][:2] - initial_pucks[i][:2])), 3)
                   for i in (1, 2)}
    elapsed = time.perf_counter() - start_wall
    simulated = sim.data.time - start_sim
    report = {
        "parcels": notes,
        "puck_travel_m": puck_travel,
        "base_travel_m": round(float(sim.data.qpos[0] - start_x), 3),
        "worst_ik_solver_error_m": round(worst_ik_error, 4),
        "worst_ik_phase": worst_ik_phase,
        "largest_joint_target_step_rad": round(largest_target_step, 4),
        "peak_joint_speed_rad_s": round(max_joint_speed, 3),
        "simulated_seconds": round(simulated, 2),
        "wall_seconds": round(elapsed, 2),
        "achieved_playback_x": round(simulated / elapsed, 2),
        "warnings": sim.state()["warnings"],
    }
    report["passed"] = (all(item["landed_on_table"] and item["travel_m"] > 0.08 for item in notes)
                        and all(distance > 0.08 for distance in puck_travel.values())
                        and report["worst_ik_solver_error_m"] < 0.02
                        and report["largest_joint_target_step_rad"] < 0.25
                        and report["peak_joint_speed_rad_s"] < 6
                        and not report["warnings"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run measurements without the camera dashboard")
    parser.add_argument("--speed", type=float, default=1.3, help="Target playback multiple (default: 1.3)")
    parser.add_argument("--snapshot", type=Path, help="Save the six-tile final view")
    parser.add_argument("--exit-on-complete", action="store_true",
                        help="Close the dashboard after the shift (useful for automation)")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be a positive finite number")
    sim = Simulation(model=scene_model())
    dashboard = None if args.headless else Dashboard(sim)
    try:
        try:
            report = run(sim, dashboard, playback=args.speed)
        except KeyboardInterrupt:
            print("Sorting shift stopped.")
            return
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            if dashboard is None:
                with mujoco.Renderer(sim.model, height=FEED_SIZE[1], width=FEED_SIZE[0]) as renderer:
                    dashboard_frame(sim.model, sim.data, renderer, overview_camera()).save(args.snapshot)
            else:
                dashboard.update("Shift complete: two sorted parcels and a clear lane")
                dashboard.frame.save(args.snapshot)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
        if dashboard is not None and not args.exit_on_complete:
            dashboard.status.set("Shift complete | close this window when finished")
            while not dashboard.closed:
                dashboard.root.update()
                time.sleep(0.03)
    finally:
        sim.set_base()
        if dashboard is not None:
            dashboard.close()


if __name__ == "__main__":
    main()
