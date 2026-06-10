#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHITECTURES="${VOXCPM_CUDA_ARCHITECTURES:-89}"
JOBS="${VOXCPM_BUILD_JOBS:-$(python - <<'PY'
import os
print(max(1, min(os.cpu_count() or 2, 8)))
PY
)}"

cmake \
  -S "${ROOT_DIR}/VoxCPM.cpp" \
  -B "${ROOT_DIR}/VoxCPM.cpp/build-cuda" \
  -DVOXCPM_BUILD_TESTS=OFF \
  -DVOXCPM_BUILD_BENCHMARK=OFF \
  -DVOXCPM_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCHITECTURES}"

cmake \
  --build "${ROOT_DIR}/VoxCPM.cpp/build-cuda" \
  --config Release \
  --target voxcpm-server \
  -j "${JOBS}"

echo "Built ${ROOT_DIR}/VoxCPM.cpp/build-cuda/examples/voxcpm-server"
