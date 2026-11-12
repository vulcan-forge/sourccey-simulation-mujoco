import math
from types import SimpleNamespace

import pytest

from integrations.prepare_shoulder_guard_patch import METHOD, prepare


def candidate(low, high, max_step=5, normalized_span=200):
    namespace = {"MotorCalibration": object, "math": math}
    exec("class Candidate:\n" + METHOD, namespace)
    cls = namespace["Candidate"]
    cls.SHOULDER_ENCODER_PERIOD = 4096
    cls.SHOULDER_SEAM_GUARD_STEP_COUNT = 4
    cls._shoulder_step_ticks = lambda self, calibration: max(1, math.ceil((high-low) * max_step / normalized_span))
    return cls(), SimpleNamespace(range_min=low, range_max=high)


@pytest.mark.parametrize("low,high,normalized_span", [(174, 3921, 200), (173, 3921, 200),
                                                         (174, 3921, 360), (1000, 3000, 200)])
def test_calibrated_down_endpoint_is_not_replaced(low, high, normalized_span):
    safety, calibration = candidate(low, high, normalized_span=normalized_span)
    assert safety._shoulder_safe_raw_range(calibration) == (low, high)


@pytest.mark.parametrize("low,high", [(0, 3921), (174, 4095), (10, 4080), (3921, 174), (-1, 3000)])
def test_unsafe_or_ambiguous_encoder_frames_still_fail_closed(low, high):
    safety, calibration = candidate(low, high)
    with pytest.raises(RuntimeError):
        safety._shoulder_safe_raw_range(calibration)


def test_installer_refuses_uninspected_source():
    with pytest.raises(ValueError, match="differs"):
        prepare("# another host version\n")
