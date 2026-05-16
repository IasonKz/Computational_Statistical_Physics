Euler ML Data Collection Workflow
================================

Purpose
-------
Generate many thermalized spin configurations on Euler for supervised ML.

This requires the C++ ML patch to be installed. The executable must support:

    --save-configs-every
    --max-saved-configs
    --config-out
    --ml-label

Files in this package
---------------------
scripts/build_on_euler.sh
scripts/make_ml_euler_parameter_table.py
scripts/submit_ml_config_array.sh
scripts/merge_ml_metadata.py

Where to put them
-----------------
Copy or unzip them into your project root:

    project/heisenberg_dmi_ml/

so that the scripts are in:

    project/heisenberg_dmi_ml/scripts/

Step 1: Build on Euler
----------------------

    cd path/to/project/heisenberg_dmi_ml
    bash scripts/build_on_euler.sh

Step 2: Generate ML parameter table
-----------------------------------

    python3 scripts/make_ml_euler_parameter_table.py

This creates:

    parameters/ml_euler_configs_D1_L16.csv

Default setup:

    J = 1
    D = 1
    B = 0
    L = 16
    low T  = 0.6, 0.7, 0.8, 0.9      label 1, helical
    mid T  = 1.0 ... 1.7              label -1, transition/test only
    high T = 1.8, 1.9, 2.0, 2.1, 2.2 label 0, disordered
    seeds = 1,2,3,4,5
    therm = 3000
    samples = 5000
    skip = 5
    proposal = 0.7
    save_every = 10
    max_configs = 300 per run

Step 3: Submit Slurm array
--------------------------

The parameter script prints the exact command, for example:

    mkdir -p logs/euler results/raw results/configs_ml
    sbatch --array=0-84%64 scripts/submit_ml_config_array.sh

The %64 means at most 64 runs are active at the same time.
Use %32 if you want to be more conservative.

Step 4: Monitor
---------------

    squeue -u $USER

Step 5: Merge metadata
----------------------

Each job writes to its own folder:

    results/configs_ml/run_<run_id>/

After all jobs finish:

    python3 scripts/merge_ml_metadata.py --config-root results/configs_ml

This creates:

    results/configs_ml/metadata.csv

Step 6: Build the ML dataset
----------------------------

From project root:

    python3 ml/build_dataset.py \
      --config-dir results/configs_ml \
      --out-dir ml_data \
      --L 16 \
      --low-max 0.9 \
      --high-min 1.8

This should create arrays such as:

    ml_data/X_train.npy
    ml_data/y_train.npy
    ml_data/X_val.npy
    ml_data/y_val.npy
    ml_data/X_test.npy
    ml_data/T_test.npy

Step 7: Train MLP
-----------------

    python3 ml/train_mlp.py --data-dir ml_data --epochs 40 --batch-size 64

Step 8: Evaluate
----------------

    python3 ml/evaluate_mlp.py --data-dir ml_data

Expected ML output
------------------

Main plot:

    figures/report/mlp_P_helical_vs_T.pdf

Main estimate:

    Tc_ML from P_helical(T) = 0.5

Compare Tc_ML with:
    heat capacity peak
    helical susceptibility peak
    helical order drop

Notes
-----
Do not run heavy simulations on the Euler login node.
The login node is for building, preparing parameter tables, and submitting Slurm jobs.
The simulations run on compute nodes through sbatch.
