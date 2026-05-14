# Mandatory ML workflow: 3D Heisenberg + DMI

This patch adds the supervised-learning part of the project.

The C++ simulation now saves many thermalized spin configurations during the measurement phase. The ML pipeline then trains a classifier to distinguish:

- `1`: helical ordered phase
- `0`: disordered phase
- `-1`: transition/test-only region

The logic follows the Ex04 Ising idea: train only far below and far above the transition, then predict across all temperatures and locate the point of maximal confusion where `P_helical(T)=0.5`.

## 1. Rebuild after applying the patch

```bash
mkdir -p build
cd build
cmake ..
make -j
cd ..
```

## 2. Local test of ML configuration saving

```bash
./build/heisenberg_dmi \
  --L 16 --T 0.8 --J 1.0 --D 1.0 --seed 1 \
  --therm 500 --samples 1000 --skip 5 --proposal 0.7 \
  --init helical_z \
  --out results/raw \
  --save-configs-every 10 \
  --max-saved-configs 20 \
  --config-out results/configs_ml \
  --ml-label 1
```

Check:

```bash
ls results/configs_ml
head results/configs_ml/metadata.csv
```

## 3. Euler ML dataset generation

```bash
python3 scripts/make_ml_parameter_table.py --L 16
```

This creates:

```text
parameters/ml_config_scan_D1_L16.csv
```

Submit:

```bash
mkdir -p logs/euler results/raw results/configs_ml
sbatch --array=0-<LAST>%64 scripts/submit_ml_config_array.sh
```

The script prints the correct `<LAST>`.

Default labels:

```text
T <= 0.9 -> helical label 1
T >= 1.8 -> disordered label 0
otherwise -> label -1, test only
```

## 4. Build NumPy dataset

```bash
python3 ml/build_dataset.py \
  --config-dir results/configs_ml \
  --out-dir ml_data \
  --L 16 \
  --low-max 0.9 \
  --high-min 1.8
```

Outputs:

```text
ml_data/X_train.npy
ml_data/y_train.npy
ml_data/T_train.npy
ml_data/X_val.npy
ml_data/y_val.npy
ml_data/T_val.npy
ml_data/X_test.npy
ml_data/y_test.npy
ml_data/T_test.npy
ml_data/metadata_used.csv
```

## 5. Train raw-spin MLP

```bash
python3 ml/train_mlp.py --data-dir ml_data --epochs 40 --batch-size 64
```

Outputs:

```text
ml_data/models/mlp_raw_spins.pt
ml_data/mlp_training_history.csv
figures/report/mlp_loss_curve.pdf
figures/report/mlp_accuracy_curve.pdf
```

## 6. Evaluate and plot P_helical(T)

```bash
python3 ml/evaluate_mlp.py --data-dir ml_data
```

Outputs:

```text
ml_data/mlp_predictions.csv
ml_data/mlp_P_helical_by_T.csv
figures/report/mlp_P_helical_vs_T.pdf
```

The ML transition estimate is where:

```text
P_helical(T) = 0.5
```

Compare it with heat-capacity peak, helical susceptibility-like peak, and the drop in helical order.

## 7. Feature baseline

Optional sanity baseline:

```bash
python3 ml/train_feature_baseline.py --data-dir ml_data
```

This uses `E_density`, `m_abs_density`, `helical_order`, and `q_peak`. It is useful for comparison, but the main mandatory ML result should be the raw-spin MLP.
