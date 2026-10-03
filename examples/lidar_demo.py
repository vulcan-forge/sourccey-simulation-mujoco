"""Scan three known obstacles in the robot's forward 180-degree plane.

Run: python -m examples.lidar_demo
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np

from sourccey.build import MODEL
from sourccey.lidar import ANGLES, save_snapshot, scan
from sourccey.simulation import Simulation


def make_model():
    spec = mujoco.MjSpec.from_file(str(MODEL))
    for name, position, color in (
        ('center_box', [1.0, 0.0, .30], [.95, .30, .12, 1.]),
        ('left_box', [1.15, .48, .30], [.20, .78, .36, 1.]),
        ('right_box', [1.15, -.48, .30], [.22, .48, .95, 1.]),
    ):
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=position, size=[.10, .10, .14],
                                contype=1, conaffinity=1, group=0, rgba=color)
    return spec.compile()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/lidar_demo.json'))
    args = parser.parse_args()
    sim = Simulation(model=make_model())
    result = scan(sim.model, sim.data)
    center = int(np.argmin(np.abs(ANGLES)))
    assert result['hit_names'][center] == 'center_box'
    assert np.isclose(result['ranges_m'][center], .72, atol=.015)
    assert 'left_box' in result['hit_names']
    assert 'right_box' in result['hit_names']
    data_path, image_path = save_snapshot(sim.model, sim.data, args.output)
    print(f'Saved {len(result["ranges_m"])} lidar beams to {data_path} and {image_path}')


if __name__ == '__main__':
    main()
