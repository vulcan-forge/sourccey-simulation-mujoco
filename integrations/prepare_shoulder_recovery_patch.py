"""Prepare a correction for an endpoint recovery deadlock in the robot host.

Operates on the exact saved, previously installed host module. Preparing the
candidate never connects to hardware, writes registers, or restarts the host.
"""
import argparse
import hashlib
from pathlib import Path


INSPECTED_SHA256 = "e0f59382f621eb8dfb3f403e0b7bd5c89b4b288342220751499ddf42a5f2b708"
ORIGINAL = '''            self._shoulder_startup_recovery_raw = (
                safe_raw_min if raw < safe_raw_min else safe_raw_max
            )
'''
REPLACEMENT = '''            # An endpoint target can deadlock recovery: servo deadband may
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
'''


def prepare(source):
    if hashlib.sha256(source.encode()).hexdigest() != INSPECTED_SHA256:
        raise ValueError("Input differs from the inspected installed host module; inspect it before patching")
    if source.count(ORIGINAL) != 1:
        raise ValueError("Expected exactly one shoulder recovery assignment")
    patched = source.replace(ORIGINAL, REPLACEMENT, 1)
    compile(patched, "candidate_sourccey_follower_recovery.py", "exec")
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("artifacts/robot_diagnostics/candidate_sourccey_follower_safety.py"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/robot_diagnostics/candidate_sourccey_follower_recovery.py"))
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Output must be a separate candidate file")
    patched = prepare(args.source.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(patched)
    print(f"Prepared candidate only: {args.output}")
    print("No running host, motor calibration, or servo settings were changed.")


if __name__ == "__main__":
    main()
