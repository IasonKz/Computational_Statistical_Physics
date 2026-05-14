#!/bin/bash
set -euo pipefail
module purge || true
module load stack/2024-06 || module load stack/2025-06 || true
mkdir -p build
cd build
cmake ..
make -j 4
echo "Built executable:"
ls -lh heisenberg_dmi
