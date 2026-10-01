#!/usr/bin/env bash
# Meshopt-compress the face for the web: 7.1 MB -> ~2.1 MB. Positions stay float because the porcelain seam
# shader and the eye shader read raw object-space positions (quantization would move them into a scaled space).
# usage: face/build/pack.sh <in.glb> <out.glb>   (needs `npx gltfpack`)
set -euo pipefail
npx --yes gltfpack@0.24 -i "$1" -o "$2" -cc -kn -km -ke -vpf -vn 12
