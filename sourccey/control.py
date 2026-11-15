"""Standalone MuJoCo control helpers."""
from dataclasses import dataclass
from collections import deque
import json
from pathlib import Path
import numpy as np

def finite_vector(value, size):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.all(np.isfinite(a)):
        raise ValueError(f"Expected {size} finite numbers")
    return a


def gripper_degrees(closure):
    closure = finite_vector([closure], 1)[0]
    return 60 - 65 * np.clip(closure, 0, 100) / 100


def wheel_rates(command):
    """Normalized forward/left/CCW -> CAD wheel rad/s, FL FR RL RR."""
    return wheel_rates_from_velocity(np.clip(finite_vector(command, 3), -1, 1) * [0.6, 0.6, np.pi / 2])


def wheel_rates_from_velocity(velocity):
    """Body velocity [m/s, m/s, rad/s] -> CAD wheel rad/s, FL FR RL RR."""
    v, s, w = finite_vector(velocity, 3)
    return np.array([v-s-.30*w, -(v+s+.30*w), v+s-.30*w, -(v-s+.30*w)]) / .052


class IdentifiedBaseResponse:
    """Measured isolated-axis speed tables, transport delay and first-order lag.

    This supplies wheel targets, never chassis forces or poses. Axis superposition
    for combined commands is provisional: wheel saturation/slip were not identified.
    """

    def __init__(self):
        data = json.loads(Path(__file__).with_name("identified_base.json").read_text(encoding="utf-8"))
        self.axes = [data["axes"][name] for name in ("x", "y", "theta")]
        self.reset()

    def reset(self):
        self.time = 0.0
        self.velocity = np.zeros(3)
        self._target = np.zeros(3)
        self._previous = np.zeros(3)
        self._pending = [deque() for _ in self.axes]

    def steady_velocity(self, command):
        command = np.clip(finite_vector(command, 3), -1, 1)
        return np.array([
            np.sign(u) * np.interp(abs(u), axis["table_u"], axis["v_pos" if u >= 0 else "v_neg"], left=0.0)
            for u, axis in zip(command, self.axes)
        ])

    def step(self, command, dt):
        target = self.steady_velocity(command)
        dt = float(finite_vector([dt], 1)[0])
        if dt <= 0:
            raise ValueError("dt must be positive")
        end = self.time + dt
        for i, axis in enumerate(self.axes):
            events = self._pending[i]
            if target[i] != self._previous[i]:
                events.append((self.time + axis["delay"], target[i]))
            cursor = self.time
            while events and events[0][0] <= end:
                when, value = events.popleft()
                self.velocity[i] += (self._target[i] - self.velocity[i]) * -np.expm1(-(when-cursor)/axis["tau"])
                self._target[i] = value
                cursor = when
            self.velocity[i] += (self._target[i] - self.velocity[i]) * -np.expm1(-(end-cursor)/axis["tau"])
        self._previous = target
        self.time = end
        return self.velocity.copy()


@dataclass
class HandTarget:
    """End-effector target in the moving shoulder-mount frame."""
    local: np.ndarray
    frozen: bool = True
