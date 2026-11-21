"""Shared task text and measured, unassisted box-placement success criteria."""
from collections import deque

import numpy as np


PICKUP_TASK = "Pick up the box"
PLACE_TASK = "Pick up the box and place it at the fixed destination"


def placement_observation(run):
    """Privileged evaluator/teacher measurements; never policy observations."""
    import mujoco
    table_force = 0.
    force = np.zeros(6)
    table = run.m.geom('box_test_table').id
    for index, contact in enumerate(run.d.contact):
        if {int(contact.geom1), int(contact.geom2)} == {run.box_geom, table}:
            mujoco.mj_contactForce(run.m, run.d, index, force)
            table_force += max(0., float(force[0]))
    return dict(position=run.d.xpos[run.box].copy(), bottom=run.bottom(),
        upright=float(run.d.xmat[run.box].reshape(3, 3)[2, 2]),
        finger_force=max(run.fingers().values()), table_force=table_force)


class PlacementTracker:
    def __init__(self, placement, near_edge_x, tabletop_z):
        self.config = placement
        self.goal = np.array([near_edge_x + placement['center_from_near_edge_mm']/1000,
                              placement['center_left_of_robot_mm']/1000])
        self.tabletop_z = tabletop_z
        self.samples = deque()
        self.stable_placement = None
        self.last = None

    def observe(self, timestamp, *, position, bottom, upright, finger_force,
                table_force, lifted):
        p = self.config
        error = float(np.linalg.norm(np.asarray(position)[:2]-self.goal))
        self.last = dict(position_error_mm=error*1000,
            bottom_above_table_mm=(bottom-self.tabletop_z)*1000,
            tilt_deg=float(np.rad2deg(np.arccos(np.clip(upright, -1, 1)))),
            finger_force_n=float(finger_force), table_force_n=float(table_force))
        if self.stable_placement is not None:
            return
        valid = (lifted and error <= p['position_tolerance_mm']/1000
            and abs(bottom-self.tabletop_z) <= p['table_height_tolerance_mm']/1000
            and upright >= np.cos(np.deg2rad(p['maximum_tilt_deg']))
            and finger_force <= p['maximum_released_finger_force_n']
            and table_force >= p['minimum_table_force_n'])
        if not valid:
            self.samples.clear()
            return
        self.samples.append((float(timestamp), np.asarray(position).copy(), self.last.copy()))
        while len(self.samples) > 1 and timestamp-self.samples[1][0] >= p['settle_seconds']-1e-9:
            self.samples.popleft()
        duration = timestamp-self.samples[0][0]
        drift = max(np.linalg.norm(row[1]-self.samples[0][1]) for row in self.samples)
        if duration >= p['settle_seconds']-1e-9 and drift <= p['maximum_settle_drift_mm']/1000:
            self.stable_placement = dict(start_s=self.samples[0][0], end_s=float(timestamp),
                duration_s=duration, drift_mm=float(drift*1000),
                maximum_position_error_mm=max(row[2]['position_error_mm'] for row in self.samples),
                final=self.last.copy())
