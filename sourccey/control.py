"""Standalone MuJoCo control helpers."""
from dataclasses import dataclass
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
    v, s, w = np.clip(finite_vector(command, 3), -1, 1) * [0.6, 0.6, np.pi / 2]
    return np.array([v-s-.30*w, -(v+s+.30*w), v+s-.30*w, -(v-s+.30*w)]) / .052


@dataclass
class HandTarget:
    """End-effector target in the moving shoulder-mount frame."""
    local: np.ndarray
    frozen: bool = True
