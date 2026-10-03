"""Drive through a furnished room with a live LD19 map and visible scan fan.

Run: python -m examples.lidar_room
Close the window to stop. Use --headless for a fast measured check.
"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from sourccey.build import MODEL
from sourccey.lidar import ANGLES, CONFIG, map_image, scan
from sourccey.simulation import Simulation


FRAME_SIZE = (1200, 700)
MAP_SIZE = (384, 240)
PHASES = (
    (0.5, 'Settling in the room', (0, 0, 0)),
    (3.7, 'Approaching the first obstacle pair', (.55, 0, 0)),
    (4.6, 'Sliding left through the aisle', (0, .5, 0)),
    (7.8, 'Passing the colored crates', (.55, 0, 0)),
    (8.7, 'Sliding back toward center', (0, -.5, 0)),
    (11.2, 'Approaching the far wall', (.5, 0, 0)),
    (13.0, 'Turning to scan the room', (0, 0, .35)),
    (13.5, 'Room scan complete', (0, 0, 0)),
)


def scene_model():
    spec = mujoco.MjSpec.from_file(str(MODEL))

    def box(name, pos, half_size, color):
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=pos, size=half_size, rgba=color,
                                contype=1, conaffinity=1, group=0)

    box('room_far_wall', [4.7, 0, .6], [.06, 2.0, .6], [.42, .49, .54, 1])
    box('room_left_wall', [1.8, 2.0, .6], [2.9, .06, .6], [.39, .47, .53, 1])
    box('room_right_wall', [1.8, -2.0, .6], [2.9, .06, .6], [.39, .47, .53, 1])
    for name, pos, size, color in (
        ('room_amber_crate', [.9, .88, .34], [.18, .20, .34], [.98, .60, .17, 1]),
        ('room_blue_crate', [1.25, -.92, .32], [.20, .18, .32], [.16, .54, .9, 1]),
        ('room_green_cabinet', [2.0, 1.05, .55], [.22, .24, .55], [.24, .72, .45, 1]),
        ('room_purple_crate', [2.55, -.96, .38], [.19, .19, .38], [.62, .38, .82, 1]),
        ('room_red_cabinet', [3.35, 1.02, .51], [.23, .22, .51], [.9, .36, .31, 1]),
        ('room_yellow_crate', [3.70, -.93, .32], [.17, .18, .32], [.9, .76, .25, 1]),
    ):
        box(name, pos, size, color)
    return spec.compile()


def camera_for(data):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [float(data.qpos[0]) + .65, float(data.qpos[1]), .55]
    camera.distance = 3.45
    camera.azimuth = 145
    camera.elevation = -30
    return camera


def add_scan_fan(scene, reading):
    """Render-only rays and arc, following the real scanner frame and returns."""
    origin = reading['origin_world_m']
    rotation = reading['rotation_world']
    endpoints = []
    for index in range(0, len(ANGLES), 15):
        angle = ANGLES[index]
        distance = min(float(reading['ranges_m'][index]), 2.2)
        end = origin + rotation @ np.array([np.cos(angle), np.sin(angle), 0]) * distance
        endpoints.append(end)
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE,
                            np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.array([.05, .98, .87, .48], dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, origin, end)
        scene.ngeom += 1
    for first, second in zip(endpoints, endpoints[1:]):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE,
                            np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.array([.05, .98, .87, .23], dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1.0, first, second)
        scene.ngeom += 1
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([.007, .007, .007]), origin, np.eye(3).ravel(),
                        np.array([1, .34, .12, 1], dtype=np.float32))
    scene.ngeom += 1


def render_frame(renderer, sim, reading, phase):
    renderer.update_scene(sim.data, camera=camera_for(sim.data))
    add_scan_fan(renderer.scene, reading)
    frame = Image.fromarray(renderer.render())
    drawing = ImageDraw.Draw(frame)
    drawing.rectangle((0, 0, FRAME_SIZE[0], 52), fill=(12, 19, 26))
    drawing.text((18, 10), 'SOURCCEY | LD19 ROOM SCAN', fill=(220, 241, 246))
    drawing.text((18, 30), phase, fill=(100, 232, 200))
    map_frame = map_image(reading).resize(MAP_SIZE, Image.Resampling.BILINEAR)
    map_pos = (FRAME_SIZE[0] - MAP_SIZE[0] - 14, 62)
    frame.paste(map_frame, map_pos)
    drawing = ImageDraw.Draw(frame)
    drawing.rectangle((map_pos[0] - 2, map_pos[1] - 2,
                       map_pos[0] + MAP_SIZE[0] + 1, map_pos[1] + MAP_SIZE[1] + 1),
                      outline=(102, 238, 204), width=2)
    drawing.text((18, FRAME_SIZE[1] - 26),
                 'Teal fan: first 2.2 m of front aperture | orange dot: lower-front scanner | '
                 f'{CONFIG["beam_count"]} rays / {CONFIG["scan_rate_hz"]:g} Hz',
                 fill=(205, 231, 233))
    return frame


class Dashboard:
    def __init__(self, sim):
        import tkinter as tk

        self.root = tk.Tk()
        self.root.title('Sourccey | LD19 room scan')
        self.root.resizable(False, False)
        self.label = tk.Label(self.root)
        self.label.pack()
        self.photo = None
        self.frame = None
        self.closed = False
        self.renderer = mujoco.Renderer(sim.model, height=FRAME_SIZE[1], width=FRAME_SIZE[0])
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    def update(self, sim, reading, phase):
        if self.closed:
            return
        self.frame = render_frame(self.renderer, sim, reading, phase)
        self.photo = ImageTk.PhotoImage(self.frame, master=self.root)
        self.label.configure(image=self.photo)
        self.root.update()

    def close(self):
        if not self.closed:
            self.closed = True
            self.renderer.close()
            self.root.destroy()


def run(sim, dashboard=None, speed=1.0):
    start_wall = time.perf_counter()
    start = sim.data.qpos[:2].copy()
    previous_command = None
    next_frame = 0.
    readings = []
    last_phase = ''
    while sim.data.time < PHASES[-1][0]:
        if dashboard is not None and dashboard.closed:
            raise KeyboardInterrupt
        phase_end, phase, command = next(item for item in PHASES if sim.data.time < item[0])
        if command != previous_command:
            sim.set_base(*command)
            previous_command = command
        sim.step(min(50, max(1, int((phase_end - sim.data.time) / sim.model.opt.timestep))))
        if sim.data.time >= next_frame:
            reading = scan(sim.model, sim.data)
            readings.append(reading)
            if dashboard is not None:
                dashboard.update(sim, reading, phase)
            next_frame = sim.data.time + 1.0 / CONFIG['scan_rate_hz']
        last_phase = phase
        if dashboard is not None:
            delay = start_wall + sim.data.time / speed - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    sim.set_base()
    travel = float(np.linalg.norm(sim.data.qpos[:2] - start))
    center = int(np.argmin(np.abs(ANGLES)))
    center_ranges = [float(r['ranges_m'][center]) for r in readings]
    distinct_hits = sorted({hit for reading in readings for hit in reading['hit_names'] if hit})
    report = {'passed': (travel > 2.0 and len(readings) > 30 and
                         len(distinct_hits) >= 5 and
                         abs(center_ranges[0] - center_ranges[-1]) > 1.0 and
                         not sim.state()['warnings']),
              'travel_m': round(travel, 3), 'scan_count': len(readings),
              'beam_count': CONFIG['beam_count'], 'distinct_hits': distinct_hits,
              'center_range_start_m': round(center_ranges[0], 3),
              'center_range_end_m': round(center_ranges[-1], 3),
              'final_base_xy_m': sim.data.qpos[:2].round(3).tolist(),
              'warnings': sim.state()['warnings']}
    return report, readings[-1], last_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed (default: real time)')
    parser.add_argument('--snapshot', type=Path, help='Save the final combined scene and lidar map')
    parser.add_argument('--exit-on-complete', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive and finite')
    sim = Simulation(model=scene_model())
    dashboard = None if args.headless else Dashboard(sim)
    try:
        try:
            report, reading, phase = run(sim, dashboard, args.speed)
        except KeyboardInterrupt:
            print('LD19 room demo stopped.')
            return
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            if dashboard is None:
                with mujoco.Renderer(sim.model, height=FRAME_SIZE[1], width=FRAME_SIZE[0]) as renderer:
                    render_frame(renderer, sim, reading, phase).save(args.snapshot)
            else:
                dashboard.frame.save(args.snapshot)
        print(json.dumps(report, indent=2))
        if not report['passed']:
            raise SystemExit(1)
        if dashboard is not None and not args.exit_on_complete:
            while not dashboard.closed:
                dashboard.root.update()
                time.sleep(.03)
    finally:
        sim.set_base()
        if dashboard is not None:
            dashboard.close()


if __name__ == '__main__':
    main()
