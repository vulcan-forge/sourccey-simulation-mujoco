"""Standalone viewer and desktop controls; also supports headless runs and rendering."""
import argparse
from pathlib import Path
import time
import numpy as np
import mujoco

from .build import ROLES
from .cameras import CameraFeeds, save_snapshot
from .simulation import Simulation


def scene_options():
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    opt.sitegroup[:] = 0
    return opt


def camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, .55]
    cam.distance, cam.azimuth, cam.elevation = 1.8, 135, -15
    return cam


def render(sim, path):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(sim.model, height=900, width=1200) as renderer:
        renderer.update_scene(sim.data, camera=camera(), scene_option=scene_options())
        Image.fromarray(renderer.render()).save(path)


class Panel:
    def __init__(self, sim):
        import tkinter as tk
        from tkinter import ttk
        self.sim, self.tk = sim, tk
        self.root = tk.Tk()
        self.root.title("Sourccey | Simulation controls")
        self.root.geometry("510x850")
        self.closed = False
        self.paused = False
        self.joint_vars = {}
        self.ik_enabled = {side: False for side in ("left", "right")}
        self.ik_vars = {}
        self.camera_feeds = None
        for target in sim.targets.values():
            target.frozen = False
        self.status = tk.StringVar(value="Position drives | equivalent mecanum traction | no hardware connection")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        ttk.Label(self.root, text="Sourccey", font=("Segoe UI", 18, "bold")).pack(pady=5)
        ttk.Label(self.root, text="Drag in the viewer to orbit. Scroll to zoom.").pack()
        row = ttk.Frame(self.root); row.pack(fill="x", padx=12, pady=6)
        ttk.Button(row, text="STOP BASE", command=self.stop).pack(side="left")
        ttk.Button(row, text="Reset pose", command=self.reset).pack(side="left", padx=5)
        ttk.Button(row, text="Pause / resume", command=self.pause).pack(side="left")
        ttk.Button(row, text="Camera feeds", command=self.toggle_cameras).pack(side="left", padx=5)
        base = ttk.LabelFrame(self.root, text="Base — release a slider to stop")
        base.pack(fill="x", padx=12)
        self.base_vars = []
        for label in ("Forward", "Left / strafe", "CCW yaw"):
            var = tk.DoubleVar(value=0); self.base_vars.append(var)
            scale = self.scale(base, label, var, -1, 1, .01, self.drive)
            scale.bind("<ButtonRelease-1>", lambda e: self.stop())
        self.elevator = tk.DoubleVar(value=-.0142)
        bottom = -.315 if sim.full_elevator_range else -.3104
        self.scale(self.root, "Elevator (m)", self.elevator, bottom, -.0142, .001,
                   lambda _: sim.set_elevator(self.elevator.get()))
        book = ttk.Notebook(self.root); book.pack(fill="both", expand=True, padx=12, pady=5)
        for side in ("left", "right"):
            page = ttk.Frame(book); book.add(page, text=side.title() + " arm")
            ttk.Label(page, text="Joint targets in full-assembly URDF degrees").pack()
            for role in ROLES:
                name = side + "_" + role
                var = tk.DoubleVar(value=np.rad2deg(sim.data.ctrl[sim.actuators[name]]))
                self.joint_vars[name] = var
                limits = np.rad2deg(sim.model.jnt_range[sim.joints[name]])
                self.scale(page, role, var, *limits, .1, lambda _, n=name: self.joint_changed(n))
            ikbox = ttk.LabelFrame(page, text="End-effector target relative to moving shoulder (meters)")
            ikbox.pack(fill="x", pady=5)
            self.ik_vars[side] = []
            for axis, value in zip("XYZ", sim.ee_local(side)):
                var = tk.DoubleVar(value=float(value)); self.ik_vars[side].append(var)
                self.scale(ikbox, axis, var, -.5, .5, .002, lambda _, s=side: self.enable_ik(s))
            ttk.Button(ikbox, text="Freeze / resume target input", command=lambda s=side: self.freeze(s)).pack()
        ttk.Label(self.root, textvariable=self.status, wraplength=480).pack(padx=12, pady=8)
        self.root.bind("<Escape>", lambda e: self.stop())
        self.root.bind("<FocusOut>", self.focus_out)

    def scale(self, parent, label, var, low, high, resolution, callback):
        row = self.tk.Frame(parent); row.pack(fill="x", padx=10)
        self.tk.Label(row, text=label, width=19, anchor="w").pack(side="left")
        scale = self.tk.Scale(row, from_=low, to=high, resolution=resolution, variable=var,
                              orient="horizontal", length=270, command=callback, width=10)
        scale.pack(side="right", fill="x", expand=True)
        return scale

    def focus_out(self, event):
        # Stop on loss of panel focus; moving focus between its children is harmless.
        if self.root.focus_displayof() is None:
            self.stop()

    def drive(self, _=None):
        self.sim.set_base(*(v.get() for v in self.base_vars))

    def stop(self):
        for v in self.base_vars:
            v.set(0)
        self.sim.set_base()

    def reset(self):
        self.stop(); self.sim.reset(); self.elevator.set(-.0142)
        for side in self.ik_enabled:
            self.ik_enabled[side] = False
            self.sim.targets[side].frozen = False
            for var, value in zip(self.ik_vars[side], self.sim.ee_local(side)):
                var.set(float(value))
        self.sync_joint_sliders()

    def pause(self):
        self.stop(); self.paused = not self.paused

    def joint_changed(self, name):
        side = name.split("_")[0]
        self.ik_enabled[side] = False
        self.sim.set_joints({name: np.deg2rad(self.joint_vars[name].get())})

    def enable_ik(self, side):
        self.ik_enabled[side] = True
        if not self.sim.targets[side].frozen:
            self.sim.targets[side].local[:] = [v.get() for v in self.ik_vars[side]]

    def freeze(self, side):
        target = self.sim.targets[side]
        # Restore the active local target when resuming the panel slider.
        if target.frozen:
            for var, value in zip(self.ik_vars[side], target.local):
                var.set(float(value))
        target.frozen = not target.frozen
        self.status.set(f"{side.title()} target input {'frozen' if target.frozen else 'active'}; target follows chassis/elevator")

    def sync_joint_sliders(self):
        for name, var in self.joint_vars.items():
            var.set(float(np.rad2deg(self.sim.data.ctrl[self.sim.actuators[name]])))

    def update_ik(self):
        for side, enabled in self.ik_enabled.items():
            if enabled:
                target = self.sim.targets[side]
                if not target.frozen:
                    target.local[:] = [v.get() for v in self.ik_vars[side]]
                result = self.sim.solve_ik(side, target.local, iterations=8)
                for role in ROLES[:3]:
                    name = side + "_" + role
                    self.joint_vars[name].set(float(np.rad2deg(self.sim.data.ctrl[self.sim.actuators[name]])))
                self.status.set(f"{side.title()} reach error {result['error_m']*1000:.1f} mm | {'frozen input' if target.frozen else 'active input'}")

    def close(self):
        self.stop(); self.closed = True
        if self.camera_feeds is not None:
            self.camera_feeds.close()
        self.root.destroy()

    def toggle_cameras(self):
        if self.camera_feeds is not None and self.camera_feeds.window is not None:
            self.camera_feeds.close()
            self.camera_feeds = None
        else:
            self.show_cameras()

    def show_cameras(self):
        if self.camera_feeds is None or self.camera_feeds.window is None:
            self.camera_feeds = CameraFeeds(self.root, self.sim)

    def update_cameras(self):
        if self.camera_feeds is not None:
            self.camera_feeds.update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--camera-snapshot", type=Path,
                        help="Save a tiled image from all five simulated cameras and exit")
    parser.add_argument("--full-elevator-range", action="store_true")
    parser.add_argument("--no-traction", action="store_true", help="A/B check: wheel spin on frictionless contacts")
    parser.add_argument("--no-panel", action="store_true")
    parser.add_argument("--no-camera-feeds", action="store_true",
                        help="Do not open the five-camera preview on desktop startup")
    args = parser.parse_args()
    sim = Simulation(args.full_elevator_range, not args.no_traction)
    if args.headless or args.camera_snapshot:
        for _ in range(max(0, round(args.seconds / sim.model.opt.timestep))):
            sim.step()
        print(sim.state())
        if args.camera_snapshot:
            save_snapshot(sim.model, sim.data, args.camera_snapshot)
            print(f"Saved camera feeds to {args.camera_snapshot}")
    else:
        import mujoco.viewer
        panel = None if args.no_panel else Panel(sim)
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            if panel is not None and not args.no_camera_feeds:
                panel.show_cameras()
            viewer.cam.lookat[:] = camera().lookat
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.8, 135, -15
            viewer.opt.geomgroup[3] = 0
            viewer.opt.sitegroup[:] = 0
            last = time.perf_counter()
            accumulator = 0.
            while viewer.is_running() and (panel is None or not panel.closed):
                now = time.perf_counter()
                accumulator += min(now - last, .05)
                last = now
                if panel:
                    panel.root.update()
                    if panel.closed:
                        break
                    panel.update_ik()
                with viewer.lock():
                    while accumulator >= sim.model.opt.timestep:
                        if panel is None or not panel.paused:
                            sim.step()
                        accumulator -= sim.model.opt.timestep
                    if panel:
                        panel.update_cameras()
                viewer.sync()
                time.sleep(.005)
        if panel and not panel.closed:
            panel.close()
    if args.render:
        render(sim, args.render)


if __name__ == "__main__":
    main()
