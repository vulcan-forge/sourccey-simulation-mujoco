"""Prepare (never install) a correction for the inspected host shoulder guard.

The input is a local copy of the robot's safety module. No hardware imports,
network operations, servo writes, or running-host changes are performed here.
"""
import argparse
import ast
import hashlib
from pathlib import Path


INSPECTED_SHA256 = "29513fc8be2183f3e09bf8d3b11b3a8f648afd7e8d169b2465b87616b00004a1"
METHOD = '''    def _shoulder_safe_raw_range(
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
'''


def prepare(source):
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != INSPECTED_SHA256:
        raise ValueError("Input differs from inspected host module; inspect changes before patching")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SourcceyFollowerSafety")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_shoulder_safe_raw_range")
    lines = source.splitlines(keepends=True)
    patched = "".join(lines[:method.lineno - 1]) + METHOD + "".join(lines[method.end_lineno:])
    compile(patched, "candidate_sourccey_follower_safety.py", "exec")
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/robot_diagnostics/sourccey_follower_safety.py"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/robot_diagnostics/candidate_sourccey_follower_safety.py"))
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    patched = prepare(source)
    if args.output.resolve() == args.source.resolve():
        parser.error("Output must be a separate candidate file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(patched, encoding="utf-8")
    print(f"Prepared candidate only: {args.output}")
    print("No running host, calibration, or servo settings were changed.")


if __name__ == "__main__":
    main()
