#!/usr/bin/env bash
# Run headless Blender 4.3 (docker image noctavia-blender:local) on files inside this project.
# The project root is mounted at /work. Usage (paths relative to the project root):
#   tools/blender.sh -b -P face/build/some_script.py -- [script args]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec docker run --rm --memory=3g --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$ROOT:/work" -w /work noctavia-blender:local "$@"
