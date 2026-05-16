# 3D Heisenberg + DMI Monte Carlo + Supervised ML

Full project repo for the CSP term project.

## Main idea

We simulate a classical 3D Heisenberg model with Dzyaloshinskii–Moriya interaction using Metropolis Monte Carlo. We measure thermodynamic observables and helical order, then train supervised ML models to classify configurations.

## Lattice sizes

Use powers of two only:

```text
L = 8, 16, 32
L = 64 optional/heavy
```

## ML approaches

1. Feature baseline
2. Structure-factor model
3. **Main model: 3D CNN**

Recommended ML dataset:

```text
L = 8,16,32
20 training temperatures + 10 near-transition temperatures
5 seeds
50 configurations per (L,T,seed)
Total ≈ 22.5k configurations
```

## Folders

```text
src/                 C++ simulation core
local_runs/          local C++ smoke tests and small scans
euler/               Euler notes
scripts/             build, Slurm, collection, plotting scripts
notebooks/           Julia local notebook + analysis notebook
ml/                  ML scripts
machine_learning/    copy/documentation of ML scripts
data/                optional external/euler data folders
results/             raw summaries, processed summaries, saved configs
figures/             local, Julia, report figures
docs/                project_plan.html and docs
report/              final report
```

## Local smoke test

```bash
bash local_runs/run_local_smoke_test.sh
```

## Julia tiny full-computation test

```bash
jupyter lab notebooks/julia_local_small_scale_test.ipynb
```

This notebook does a full small-scale computation in Julia with `Distributed` workers and produces a colored 2D slice without arrows.

## Euler physics production

```bash
bash scripts/build_on_euler.sh
python3 scripts/make_euler_parameter_table.py --mode full
mkdir -p logs/euler results/raw results/processed
PARAM_FILE=parameters/euler_scan_D1_powers2.csv sbatch --array=0-<LAST_ID>%64 scripts/submit_euler_array.sh
python3 scripts/collect_results.py
```

## Euler ML configuration collection

```bash
python3 scripts/make_ml_euler_parameter_table.py
mkdir -p logs/euler results/raw results/configs_ml
PARAM_FILE=parameters/ml_euler_configs_D1_recommended.csv sbatch --array=0-<LAST_ID>%64 scripts/submit_ml_config_array.sh
python3 scripts/merge_ml_metadata.py --config-root results/configs_ml
```

## ML main model: 3D CNN

Build one dataset per L:

```bash
python3 ml/build_dataset.py --config-dir results/configs_ml --out-dir ml_data_L16 --L 16 --low-max 0.95 --high-min 1.80 --store-cnn
python3 ml/train_cnn3d.py --data-dir ml_data_L16 --epochs 40 --batch-size 16
python3 ml/evaluate_cnn3d.py --data-dir ml_data_L16
```

Repeat for L=8 and L=32 if desired.

Multi-L CNN plot:

```bash
python3 ml/plot_cnn_multi_L.py --dirs ml_data_L8,ml_data_L16,ml_data_L32
```

## Project plan

Open:

```text
docs/project_plan.html
```
