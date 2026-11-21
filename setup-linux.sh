#!/usr/bin/env bash
# Simulation setup is separate from the much larger training environment.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--training" ) ]]; then
  echo 'Usage: bash setup-linux.sh [--training]' >&2
  exit 2
fi
python_bin="${PYTHON:-python3.12}"
if [[ ! -f .venv-mujoco4/bin/python ]]; then
  "$python_bin" -m venv .venv-mujoco4
fi
sim_python="$root/.venv-mujoco4/bin/python"
"$sim_python" -m pip install -r requirements.txt -r requirements-robot.txt
"$sim_python" scripts/linux_handoff.py restore-inputs
if [[ "${1:-}" == "--training" ]]; then
  command -v uv >/dev/null || { echo 'Install uv before using --training.' >&2; exit 1; }
  "$sim_python" scripts/linux_handoff.py dependencies
  (cd artifacts/linux_deps/lerobot-vulcan && uv sync --frozen --python 3.12 --extra training --extra xvla)
fi
echo 'Next: bash mujoco4.sh -m integrations.linux_preflight --render'
