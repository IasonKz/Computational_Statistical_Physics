#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build results/raw results/configs_ml ml_data_L8 figures/report
cd build
cmake ..
make -j
cd ..

for row in "0.7 1" "0.8 1" "2.0 0" "2.1 0" "1.3 -1"; do
    set -- $row
    T=$1
    label=$2
    ./build/heisenberg_dmi \
      --L 8 --T "$T" --J 1.0 --D 1.0 --seed 1 \
      --therm 60 --samples 80 --skip 2 --proposal 0.7 \
      --init helical_z --out results/raw \
      --save-configs-every 4 --max-saved-configs 8 \
      --config-out "results/configs_ml/local_L8_T${T}_label${label}" \
      --ml-label "$label"
done

python3 scripts/merge_ml_metadata.py --config-root results/configs_ml
python3 ml/build_dataset.py --config-dir results/configs_ml --out-dir ml_data_L8 --L 8 --low-max 0.9 --high-min 1.8
python3 ml/train_mlp.py --data-dir ml_data_L8 --epochs 5 --batch-size 8
python3 ml/evaluate_mlp.py --data-dir ml_data_L8
echo "Local ML debug complete."
