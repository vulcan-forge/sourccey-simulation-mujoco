import logging
import math
import time
from typing import Any

from lerobot.motors.motors_bus import MotorCalibration, MotorNormMode

logger = logging.getLogger(__name__)


class ShoulderSeamAmbiguityError(RuntimeError):
    """The single-turn shoulder reading has no safely inferable physical branch."""


class SourcceyFollowerSafety:
    """Slow startup and large joint moves to avoid abrupt position jumps."""

    SHOULDER_ENCODER_PERIOD = 4096
    # Keep endpoint goals four maximum step-safety increments from the encoder
    # seam. This scales with each robot's calibrated span instead of copying a
    # fleet-wide raw offset or limit.
    SHOULDER_SEAM_GUARD_STEP_COUNT = 4.0

    STEP_SAFETY_STARTUP_WINDOW_S = 3.0
    STEP_SAFETY_DELTA_THRESHOLDS = {
        "shoulder_pan": 120.0,
        "shoulder_lift": 120.0,
        "elbow_flex": 120.0,
        "wrist_flex": 120.0,
        "wrist_roll": 120.0,
        "gripper": 120.0,
    }
    STEP_SAFETY_MAX_STEPS = {
        "shoulder_pan": 5.0,
        "shoulder_lift": 5.0,
        "elbow_flex": 5.0,
        "wrist_flex": 4.0,
        "wrist_roll": 5.0,
        "gripper": 5.0,
    }

    def __init__(self, robot: Any):
        self.robot = robot
        self._action_stream_start_time: float | None = None
        self._step_safety_log_active = False
        self._shoulder_unwrapped_raw: int | None = None
        self._shoulder_startup_recovery_raw: int | None = None
        self._shoulder_guard_log_active = False

    def reset_shoulder_continuity(self) -> None:
        self._shoulder_unwrapped_raw = None
        self._shoulder_startup_recovery_raw = None

    def _shoulder_calibration(self) -> MotorCalibration:
        try:
            return self.robot.bus.calibration["shoulder_lift"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Shoulder seam protection requires shoulder_lift calibration") from exc

    def _shoulder_norm_mode(self) -> MotorNormMode:
        return self.robot.bus.motors["shoulder_lift"].norm_mode

    def _raw_to_shoulder_position(
        self,
        raw_position: float,
        calibration: MotorCalibration,
    ) -> float:
        span = calibration.range_max - calibration.range_min
        norm_mode = self._shoulder_norm_mode()
        if norm_mode is MotorNormMode.RANGE_M100_100:
            normalized = ((raw_position - calibration.range_min) / span) * 200.0 - 100.0
            return -normalized if calibration.drive_mode else normalized
        if norm_mode is MotorNormMode.DEGREES:
            midpoint = (calibration.range_min + calibration.range_max) / 2.0
            return (raw_position - midpoint) * 360.0 / (self.SHOULDER_ENCODER_PERIOD - 1)
        raise ValueError(f"Unsupported shoulder normalization mode: {norm_mode}")

    @staticmethod
    def _distance_to_interval(value: int, low: int, high: int) -> int:
        if value < low:
            return low - value
        if value > high:
            return value - high
        return 0

    def _shoulder_step_ticks(self, calibration: MotorCalibration) -> int:
        span = calibration.range_max - calibration.range_min
        max_step = self.STEP_SAFETY_MAX_STEPS["shoulder_lift"]
        normalized_span = (
            200.0
            if self._shoulder_norm_mode() is MotorNormMode.RANGE_M100_100
            else 360.0
        )
        return max(1, math.ceil(span * max_step / normalized_span))

    def _shoulder_safe_raw_range(
        self,
        calibration: MotorCalibration,
    ) -> tuple[int, int]:
        # Calibration frame selection already places the encoder seam outside
        # mechanical travel. Keep its endpoints usable instead of demanding
        # four steps of clearance when the calibrated gap cannot provide it.
        # Require at least one maximum software step on BOTH sides of the
        # seam. A narrower/invalid frame must be corrected, never truncated.
        low, high = calibration.range_min, calibration.range_max
        encoder_max = self.SHOULDER_ENCODER_PERIOD - 1
        if not 0 < low < high < encoder_max:
            raise RuntimeError("Shoulder calibration must stay strictly off the encoder seam")
        step_ticks = self._shoulder_step_ticks(calibration)
        available_clearance = min(low, encoder_max - high)
        if available_clearance < step_ticks:
            raise RuntimeError(
                "Shoulder calibration leaves less than one bounded command step "
                "between travel and the encoder seam; correct its encoder frame"
            )
        desired_guard = math.ceil(self.SHOULDER_SEAM_GUARD_STEP_COUNT * step_ticks)
        guard_ticks = min(desired_guard, available_clearance)
        return max(low, guard_ticks), min(high, encoder_max - guard_ticks)

    def prepare_shoulder_command(
        self,
        raw_position: int,
    ) -> tuple[float, float | None]:
        """Return continuous feedback and an optional startup-recovery goal.

        A shoulder may be moved by hand while torque is disabled. If it is then
        energized in the encoder-seam margin, accepting the first model target
        can make a single-turn position servo take the long numeric route. Hold
        the command stream at the calibrated safe endpoint on the *same raw
        side* until feedback has re-entered the safe interval. This requires no
        guessed starting pose and never asks the servo to cross 0/4095.
        """
        calibration = self._shoulder_calibration()
        raw = int(raw_position) % self.SHOULDER_ENCODER_PERIOD
        safe_raw_min, safe_raw_max = self._shoulder_safe_raw_range(calibration)

        if self._shoulder_startup_recovery_raw is None and not (
            safe_raw_min <= raw <= safe_raw_max
        ):
            # An endpoint target can deadlock recovery: servo deadband may
            # leave feedback a few ticks outside the safe interval forever.
            # Aim one bounded step INTO the interval on the same raw side.
            # Calibration, seam limits, and downstream step safety stay intact.
            recovery_step = min(
                self._shoulder_step_ticks(calibration),
                (safe_raw_max - safe_raw_min) // 2,
            )
            if recovery_step < 1:
                raise RuntimeError("Shoulder safe interval has no interior recovery target")
            self._shoulder_startup_recovery_raw = (
                safe_raw_min + recovery_step if raw < safe_raw_min
                else safe_raw_max - recovery_step
            )
            # Startup feedback in the gap is ambiguous as an absolute turn, but
            # recovery does not need to guess: it stays on this numeric side.
            self._shoulder_unwrapped_raw = raw
            logger.warning(
                "Shoulder startup recovery for %s arm: raw %d -> %d "
                "(safe range %d..%d)",
                self.robot.config.orientation,
                raw,
                self._shoulder_startup_recovery_raw,
                safe_raw_min,
                safe_raw_max,
            )

        continuous = self.continuous_shoulder_position(raw, strict=True)
        if continuous is None:
            raise RuntimeError("Strict shoulder continuity unexpectedly returned no position")

        recovery_raw = self._shoulder_startup_recovery_raw
        if recovery_raw is None:
            return continuous, None

        if safe_raw_min <= raw <= safe_raw_max:
            logger.info(
                "Shoulder startup recovery completed for %s arm at raw %d",
                self.robot.config.orientation,
                raw,
            )
            self._shoulder_startup_recovery_raw = None
            return continuous, None

        return continuous, self._raw_to_shoulder_position(recovery_raw, calibration)

    def continuous_shoulder_position(
        self,
        raw_position: int,
        *,
        strict: bool,
    ) -> float | None:
        """Unwrap single-turn shoulder feedback using physical continuity.

        Returns an unclamped logical position so the runtime step limiter sees a
        small seam crossing as continuous motion. With no history, readings in
        the center of the uncalibrated seam gap are fundamentally ambiguous;
        strict command processing fails closed instead of guessing a branch.
        """
        calibration = self._shoulder_calibration()
        raw = int(raw_position) % self.SHOULDER_ENCODER_PERIOD
        period = self.SHOULDER_ENCODER_PERIOD

        if self._shoulder_unwrapped_raw is None:
            if calibration.range_min <= raw <= calibration.range_max:
                unwrapped = raw
            else:
                candidates = (raw - period, raw, raw + period)
                ranked = sorted(
                    candidates,
                    key=lambda candidate: self._distance_to_interval(
                        candidate,
                        calibration.range_min,
                        calibration.range_max,
                    ),
                )
                best_distance = self._distance_to_interval(
                    ranked[0], calibration.range_min, calibration.range_max
                )
                next_distance = self._distance_to_interval(
                    ranked[1], calibration.range_min, calibration.range_max
                )
                if next_distance - best_distance <= self._shoulder_step_ticks(calibration):
                    if strict:
                        raise ShoulderSeamAmbiguityError(
                            "shoulder_lift started in the ambiguous encoder seam zone; "
                            "leave torque off and move it into the calibrated range"
                        )
                    return None
                unwrapped = ranked[0]
        else:
            nearest_turn = round((self._shoulder_unwrapped_raw - raw) / period)
            candidates = tuple(raw + (nearest_turn + delta) * period for delta in (-1, 0, 1))
            unwrapped = min(
                candidates,
                key=lambda candidate: abs(candidate - self._shoulder_unwrapped_raw),
            )

        self._shoulder_unwrapped_raw = unwrapped
        return self._raw_to_shoulder_position(unwrapped, calibration)

    def clamp_shoulder_goal_away_from_seam(
        self,
        goal_pos: dict[str, float],
    ) -> dict[str, float]:
        """Reserve a stopping margin around the single-turn encoder seam."""
        if "shoulder_lift" not in goal_pos:
            return goal_pos

        calibration = self._shoulder_calibration()
        safe_raw_min, safe_raw_max = self._shoulder_safe_raw_range(calibration)

        endpoint_a = self._raw_to_shoulder_position(safe_raw_min, calibration)
        endpoint_b = self._raw_to_shoulder_position(safe_raw_max, calibration)
        safe_min, safe_max = sorted((endpoint_a, endpoint_b))
        requested = float(goal_pos["shoulder_lift"])
        guarded = min(safe_max, max(safe_min, requested))

        if guarded != requested and not self._shoulder_guard_log_active:
            logger.warning(
                "Shoulder seam guard for %s arm limited target %.2f to %.2f "
                "(raw safe range %d..%d)",
                self.robot.config.orientation,
                requested,
                guarded,
                safe_raw_min,
                safe_raw_max,
            )
        self._shoulder_guard_log_active = guarded != requested
        if guarded == requested:
            return goal_pos

        guarded_goal = goal_pos.copy()
        guarded_goal["shoulder_lift"] = guarded
        return guarded_goal

    def should_use_step_safety(
        self,
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
    ) -> bool:
        """Detect large action jumps or startup transitions that should use slow stepping."""
        now = time.monotonic()
        if self._action_stream_start_time is None:
            self._action_stream_start_time = now

        startup_active = (now - self._action_stream_start_time) <= self.STEP_SAFETY_STARTUP_WINDOW_S

        large_deltas: dict[str, dict[str, float]] = {}
        for motor_name, target_pos in goal_pos.items():
            if motor_name not in present_pos:
                continue

            threshold = self.STEP_SAFETY_DELTA_THRESHOLDS.get(motor_name, 15.0)
            delta = abs(float(target_pos) - float(present_pos[motor_name]))
            if delta >= threshold:
                large_deltas[motor_name] = {
                    "delta": round(delta, 2),
                    "threshold": threshold,
                }

        should_use = startup_active or bool(large_deltas)
        if should_use and not self._step_safety_log_active:
            reasons: list[str] = []
            if startup_active:
                reasons.append(f"startup<{self.STEP_SAFETY_STARTUP_WINDOW_S}s")
            if large_deltas:
                reasons.append(f"large_delta={large_deltas}")

            logger.warning(
                "Step safety trigger for %s arm: %s",
                self.robot.config.orientation,
                ", ".join(reasons),
            )

        self._step_safety_log_active = should_use
        return should_use

    def apply_step_safety(
        self,
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
    ) -> dict[str, float]:
        """Move toward the target in small per-joint increments instead of jumping directly."""
        slowed_goal_pos: dict[str, float] = {}

        for motor_name, target_pos in goal_pos.items():
            if motor_name not in present_pos:
                slowed_goal_pos[motor_name] = target_pos
                continue

            max_step = self.STEP_SAFETY_MAX_STEPS.get(motor_name, 5.0)
            current_pos = float(present_pos[motor_name])
            delta = float(target_pos) - current_pos

            if delta > max_step:
                slowed_goal_pos[motor_name] = current_pos + max_step
            elif delta < -max_step:
                slowed_goal_pos[motor_name] = current_pos - max_step
            else:
                slowed_goal_pos[motor_name] = float(target_pos)

        return slowed_goal_pos

