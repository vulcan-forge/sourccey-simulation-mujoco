"""Planar, forward-facing virtual lidar mounted in the lower front slit."""
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'models/source/lidar_config.json'
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
ANGLES = np.deg2rad(np.linspace(CONFIG['angle_min_deg'], CONFIG['angle_max_deg'],
                               CONFIG['beam_count']))
GEOM_GROUPS = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


def scan(model, data):
    """Return one 2D scan. Infinite ranges mean no return within max range.

    Only environment geometry in MuJoCo group 0 is sensed. Robot visuals and
    collision proxies use groups 2/3, so the scanner does not see its own hull.
    """
    site = model.site('lidar_scan_origin').id
    origin = data.site_xpos[site].copy()
    rotation = data.site_xmat[site].reshape(3, 3)
    local = np.vstack((np.cos(ANGLES), np.sin(ANGLES), np.zeros_like(ANGLES)))
    directions = np.ascontiguousarray((rotation @ local).T.ravel())
    geom_ids = np.full(len(ANGLES), -1, dtype=np.int32)
    distances = np.full(len(ANGLES), -1., dtype=np.float64)
    mujoco.mj_multiRay(model, data, origin, directions, GEOM_GROUPS, True,
                       model.body('base_link').id, geom_ids, distances, None,
                       len(ANGLES), CONFIG['range_max_m'])
    valid = (geom_ids >= 0) & (distances >= CONFIG['range_min_m'])
    ranges = np.where(valid, distances, np.inf)
    hits = [model.geom(int(i)).name if ok else None
            for i, ok in zip(geom_ids, valid)]
    return {'angles_rad': ANGLES.copy(), 'ranges_m': ranges,
            'hit_names': hits, 'origin_world_m': origin,
            'rotation_world': rotation.copy(), 'time_s': float(data.time)}


def as_dict(result):
    """Portable JSON payload; null denotes a ray with no return."""
    ranges = result['ranges_m']
    return {'frame_id': 'lidar_scan_origin',
            'sensor_model': CONFIG['sensor_model'],
            'angle_min_rad': float(ANGLES[0]),
            'angle_max_rad': float(ANGLES[-1]),
            'angle_increment_rad': float(ANGLES[1]-ANGLES[0]),
            'scan_rate_hz': CONFIG['scan_rate_hz'],
            'scan_time_s': 1.0 / CONFIG['scan_rate_hz'],
            'hardware_scan_deg': CONFIG['hardware_scan_deg'],
            'range_min_m': CONFIG['range_min_m'],
            'range_max_m': CONFIG['range_max_m'],
            'ranges_m': [float(x) if np.isfinite(x) else None for x in ranges],
            'hit_names': result['hit_names'],
            'origin_world_m': result['origin_world_m'].tolist(),
            'time_s': result['time_s'],
            'calibration_status': CONFIG['calibration_status']}


def map_image(result, size=(800, 500)):
    """Make a robot-relative XY laser slice (+X up, +Y left)."""
    width, height = size
    image = Image.new('RGB', size, (15, 20, 27))
    draw = ImageDraw.Draw(image)
    center = (width // 2, height - 45)
    view_range = min(CONFIG['range_max_m'], 2.5)
    scale = (height - 90) / view_range
    for radius_m in np.arange(.5, view_range + .01, .5):
        r = radius_m * scale
        draw.arc((center[0]-r, center[1]-r, center[0]+r, center[1]+r),
                 180, 360, fill=(48, 59, 71), width=1)
        draw.text((center[0]+5, center[1]-r-14), f'{radius_m:g} m', fill=(108, 130, 150))
    draw.line((20, center[1], width-20, center[1]), fill=(48, 59, 71), width=1)
    points = []
    for angle, distance in zip(ANGLES, result['ranges_m']):
        if np.isfinite(distance) and distance <= view_range:
            x = center[0] - np.sin(angle)*distance*scale
            y = center[1] - np.cos(angle)*distance*scale
            points.append((x, y))
            draw.ellipse((x-2, y-2, x+2, y+2), fill=(100, 245, 190))
    draw.ellipse((center[0]-5, center[1]-5, center[0]+5, center[1]+5),
                 fill=(255, 150, 65))
    draw.text((12, 10), 'Sourccey virtual lidar | 180 degree XY slice', fill=(220, 233, 242))
    draw.text((12, 28), f'{len(points)} returns shown | view radius {view_range:g} m',
              fill=(145, 174, 190))
    draw.text((12, height-22), 'LEFT', fill=(145, 174, 190))
    draw.text((width-48, height-22), 'RIGHT', fill=(145, 174, 190))
    return image


def save_snapshot(model, data, path):
    result = scan(model, data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_dict(result), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    image_path = path.with_suffix('.png')
    map_image(result).save(image_path)
    return path, image_path


class LidarWindow:
    """Lightweight live scan map for the MuJoCo desktop panel."""
    def __init__(self, parent, simulation):
        import tkinter as tk
        from PIL import ImageTk
        self.ImageTk = ImageTk
        self.simulation = simulation
        self.window = tk.Toplevel(parent)
        self.window.title('Sourccey | Forward lidar (provisional mount)')
        self.window.resizable(False, False)
        self.label = tk.Label(self.window)
        self.label.pack()
        self.image = None
        self.next_frame = 0.
        self.window.protocol('WM_DELETE_WINDOW', self.close)

    def update(self):
        if self.window is None or time.monotonic() < self.next_frame:
            return
        self.next_frame = time.monotonic() + 1.0 / CONFIG['scan_rate_hz']
        result = scan(self.simulation.model, self.simulation.data)
        self.image = self.ImageTk.PhotoImage(map_image(result), master=self.window)
        self.label.configure(image=self.image)

    def close(self):
        if self.window is not None:
            self.window.destroy()
            self.window = None
