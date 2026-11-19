#!/usr/bin/env bash
# Run from the installed LeRobot environment on Linux. No uploads or hardware.
set -euo pipefail
bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output="${1:-$PWD/outputs/sourccey_box_xvla}"
python_bin="${PYTHON:-python}"
gpus="${GPUS:-2}"
batch="${BATCH_SIZE:-2}"
workers="${WORKERS:-4}"
budget_seconds="${BUDGET_SECONDS:-28800}"
SECONDS=0

"$python_bin" "$bundle/train.py" --output "${output}_check" --check-only
"$python_bin" "$bundle/train.py" --output "${output}_smoke" \
  --gpus "$gpus" --batch-size "$batch" --workers "$workers" \
  --precision bf16 --steps 20 --hours 0

# Include preflight, model download and smoke training in the eight-hour budget.
remaining_hours="$($python_bin -c 'import sys; seconds=int(sys.argv[1])-int(sys.argv[2]); assert seconds>1200, "Less than 20 minutes remain; start the main run separately"; print(seconds/3600)' "$budget_seconds" "$SECONDS")"
"$python_bin" "$bundle/train.py" --output "$output" \
  --gpus "$gpus" --batch-size "$batch" --workers "$workers" \
  --precision bf16 --steps 20000 --hours "$remaining_hours"
