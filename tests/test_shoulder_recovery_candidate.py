import ast
from enum import Enum
import logging
import math
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from integrations.prepare_shoulder_recovery_patch import prepare


# Pin the actual inspected host revision so these regressions also run on a
# clean checkout without the operator's ignored diagnostics artifacts.
SOURCE = Path(__file__).resolve().parent / "fixtures/host_shoulder_safety_e0f59382.py"


class NormMode(Enum):
    RANGE_M100_100 = 1
    DEGREES = 2


def safety(source, low=174, high=3921, drive_mode=1, norm_mode=NormMode.RANGE_M100_100):
    namespace = {"Any": object, "MotorCalibration": object, "MotorNormMode": NormMode,
                 "math": math, "time": time, "logger": logging.getLogger(__name__)}
    classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
    exec(compile(ast.Module(body=classes, type_ignores=[]), "host_recovery", "exec"), namespace)
    calibration = SimpleNamespace(range_min=low, range_max=high, drive_mode=drive_mode)
    robot = SimpleNamespace(config=SimpleNamespace(orientation="test"), bus=SimpleNamespace(
        calibration={"shoulder_lift": calibration},
        motors={"shoulder_lift": SimpleNamespace(norm_mode=norm_mode)}))
    return namespace["SourcceyFollowerSafety"](robot), calibration


@pytest.fixture(scope="module")
def host_source():
    return SOURCE.read_text(encoding="utf-8")


@pytest.mark.parametrize("raw", [172, 3923])
def test_existing_recovery_reproduces_endpoint_deadlock(host_source, raw):
    guard, calibration = safety(host_source)
    for _ in range(10):
        _, target = guard.prepare_shoulder_command(raw)
        assert target is not None
        endpoint = calibration.range_min if raw < calibration.range_min else calibration.range_max
        assert target == guard._raw_to_shoulder_position(endpoint, calibration)
        # A two-tick error is inside the fake servo's eight-tick deadband.
        assert abs(raw - endpoint) < 8


@pytest.mark.parametrize("low,high", [(174, 3921), (173, 3921)])
@pytest.mark.parametrize("raw_side", ["low", "high"])
@pytest.mark.parametrize("drive_mode", [0, 1])
@pytest.mark.parametrize("norm_mode", list(NormMode))
def test_recovery_enters_safe_interval_despite_deadband(host_source, low, high, raw_side, drive_mode, norm_mode):
    guard, calibration = safety(prepare(host_source), low, high, drive_mode, norm_mode)
    raw = low - 2 if raw_side == "low" else high + 2
    feedback, recovery = guard.prepare_shoulder_command(raw)
    target_raw = guard._shoulder_startup_recovery_raw
    assert low < target_raw < high
    assert recovery == guard._raw_to_shoulder_position(target_raw, calibration)
    assert feedback == guard._raw_to_shoulder_position(raw, calibration)
    assert abs(target_raw - (low if raw_side == "low" else high)) <= guard._shoulder_step_ticks(calibration)
    assert (target_raw > raw) if raw_side == "low" else (target_raw < raw)
    # The original endpoint goal never moved this servo. The inward target
    # overcomes deadband while staying on the same numeric side of the seam.
    assert abs(target_raw - raw) > 8
    raw = target_raw + (2 if raw_side == "high" else -2)
    _, recovery = guard.prepare_shoulder_command(raw)
    assert recovery is None
    assert guard._shoulder_startup_recovery_raw is None
    assert guard._shoulder_safe_raw_range(calibration) == (low, high)


@pytest.mark.parametrize("raw", [0, 4095])
def test_recovery_from_encoder_edge_moves_inward_without_wrapping(host_source, raw):
    guard, calibration = safety(prepare(host_source))
    _, recovery = guard.prepare_shoulder_command(raw)
    target_raw = guard._shoulder_startup_recovery_raw
    assert recovery is not None and calibration.range_min < target_raw < calibration.range_max
    assert abs(target_raw - raw) < guard.SHOULDER_ENCODER_PERIOD / 2


def test_normal_targets_and_endpoint_guard_are_preserved(host_source):
    guard, calibration = safety(prepare(host_source))
    for raw in (174, 180, 2000, 3915, 3921):
        _, recovery = guard.prepare_shoulder_command(raw)
        assert recovery is None
    assert guard.clamp_shoulder_goal_away_from_seam({"shoulder_lift": -93.011}) == {"shoulder_lift": -93.011}
    assert guard.clamp_shoulder_goal_away_from_seam({"shoulder_lift": -110.}) == {"shoulder_lift": -100.}
    assert (calibration.range_min, calibration.range_max) == (174, 3921)


def test_recovery_patch_refuses_changed_or_uninspected_host(host_source):
    with pytest.raises(ValueError, match="differs"):
        prepare(host_source + "\n# different host\n")
    with pytest.raises(ValueError, match="differs"):
        prepare(prepare(host_source))
