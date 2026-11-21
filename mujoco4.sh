#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SOURCCEY_RENDERER="${SOURCCEY_RENDERER:-filament}"
export SOURCCEY_FILAMENT_API="${SOURCCEY_FILAMENT_API:-opengl}"
exec "$root/.venv-mujoco4/bin/python" "$@"
