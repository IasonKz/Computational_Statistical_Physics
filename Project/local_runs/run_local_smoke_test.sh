#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build results/raw results/configs figures/local logs/local
cd build
cmake ..
make -j
cd ..

./build/heisenberg_dmi \
  --L 8 --T 0.8 --J 1.0 --D 1.0 --seed 1 \
  --therm 100 --samples 150 --skip 2 --proposal 0.7 \
  --init helical_z --out results/raw --save-config

./build/heisenberg_dmi \
  --L 8 --T 2.0 --J 1.0 --D 1.0 --seed 2 \
  --therm 100 --samples 150 --skip 2 --proposal 0.7 \
  --init helical_z --out results/raw

python3 scripts/collect_results.py
python3 scripts/plot_summary.py
echo "Smoke test complete."
