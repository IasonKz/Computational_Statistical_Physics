# 3D Heisenberg-DMI Monte Carlo and Supervised ML

This repository contains the code, scripts, notebooks, processed data, figures, and report for the Computational Statistical Physics term project.

## Main idea

The project studies a classical three-dimensional Heisenberg model on a cubic lattice with Dzyaloshinskii-Moriya interaction (DMI). The DMI favors finite-wave-vector helical order. The model is sampled with Metropolis Monte Carlo, and the loss of helical order with increasing temperature is analyzed using thermodynamic observables, structure-factor observables, and supervised machine-learning classifiers.

The final production pipeline uses:

- C++ for the Monte Carlo simulations,
- Python for analysis, plotting, and machine learning,
- Slurm job arrays on ETH Euler for production runs and ML dataset generation.

## Development note

Some notebooks and scripts contain local filesystem paths and exploratory plotting cells from development. These are not required for reproducing the main results in the report and can be ignored. I initially experimented with parts of the workflow in Julia, mainly to explore parallelism, but the final pipeline uses C++ for Monte Carlo simulations and Python for analysis, plotting, and machine learning.

## Lattice sizes

The main simulations use powers of two:

```text
L = 8, 16, 32
```

Larger sizes, such as `L = 64`, are possible but were not used for the final production analysis because of the higher computational cost.

## Main observables

The report focuses on:

- energy density,
- heat capacity from energy fluctuations,
- vector magnetic susceptibility,
- helical order from the finite-wave-vector spin structure factor,
- dominant helical wave vector,
- supervised ML classification probability `P_helical(T)`.

The helical phase is detected primarily through the non-zero-wave-vector structure factor, not through the uniform magnetization.

## Machine-learning approaches

Three supervised approaches were implemented:

1. Feature baseline using physical summary features.
2. Structure-factor model using Fourier/structure-factor features.
3. Main model: 3D CNN trained directly on raw spin tensors.

The final report uses the `L = 16` ML production dataset for the structure-factor model and the 3D CNN. Multi-size ML runs for `L = 8` and `L = 32` are supported by the scripts but are not part of the final reported CNN result.

## Folder overview

```text
src/                 C++ simulation core
local_runs/          Local C++ smoke tests and small scans
scripts/             Build scripts, Slurm scripts, collection scripts, plotting scripts
parameters/          CSV parameter tables for Euler and visualization runs
notebooks/           Analysis notebook and local small-scale test notebook
ml/                  Machine-learning scripts
machine_learning/    Additional/legacy copy of ML scripts and documentation
data/                Optional external or Euler data folders
results/             Raw summaries, processed summaries, and saved configurations
figures/             Local figures, exploratory figures, and final report figures
docs/                Project plan (html page with the pipeline of the project) and documentation
report/              Folder for the final report(tex + pdf)
```

Some folders may contain exploratory files or unused intermediate data. The main reproducible parts are the C++ source code, the scripts, the processed summaries, the report figures, and the final report PDF.

## Local smoke test

From the project root, run:

```bash
bash local_runs/run_local_smoke_test.sh
```

This builds/runs a small local simulation and is useful for checking that the C++ executable works.

## Local analysis notebooks

The main analysis notebook is:

```text
notebooks/analysis.ipynb
```

There may also be a local small-scale test notebook used during development. It is not required for the final report results.

## Euler physics production

A typical Euler production workflow is:

```bash
bash scripts/build_on_euler.sh
python3 scripts/make_euler_parameter_table.py --mode full
mkdir -p logs/euler results/raw results/processed
PARAM_FILE=parameters/euler_scan_D1_powers2.csv sbatch --array=0-<LAST_ID>%64 scripts/submit_euler_array.sh
python3 scripts/collect_results.py
```

Replace `<LAST_ID>` by the last row index of the parameter table.

## Euler visualization configurations

Representative full spin configurations for visualization can be generated with the visual-slice parameter table and the corresponding Slurm script. These saved configurations are used for real-space spin-angle plots, while the quantitative observables in the report are computed from the raw Monte Carlo summaries.

## Euler ML configuration collection

To generate saved configurations for the ML dataset:

```bash
python3 scripts/make_ml_euler_parameter_table.py
mkdir -p logs/euler results/raw results/configs_ml
PARAM_FILE=parameters/ml_euler_configs_D1_recommended.csv sbatch --array=0-<LAST_ID>%64 scripts/submit_ml_config_array.sh
python3 scripts/merge_ml_metadata.py --config-root results/configs_ml
```

## ML main model: 3D CNN

Build one dataset for `L = 16`:

```bash
python3 ml/build_dataset.py --config-dir results/configs_ml --out-dir ml_data_L16 --L 16 --low-max 0.95 --high-min 1.80 --store-cnn
python3 ml/train_cnn3d.py --data-dir ml_data_L16 --epochs 40 --batch-size 16
python3 ml/evaluate_cnn3d.py --data-dir ml_data_L16
```

Optional multi-size CNN plots can be produced if the corresponding datasets are available:

```bash
python3 ml/plot_cnn_multi_L.py --dirs ml_data_L8,ml_data_L16,ml_data_L32
```

## Final report

The final project report should be submitted as a single PDF file together with the code. If the PDF is not already inside the repository, place it at the project root or inside `report/` before submission.
