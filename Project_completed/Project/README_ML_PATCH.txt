ML patch for Heisenberg + DMI project
=====================================

Apply from project root:

    cd ~/projects/CSP_project/csp26exercises/project/heisenberg_dmi_ml
    unzip /path/to/heisenberg_dmi_ml_patch.zip

This overwrites:

    src/config.hpp
    src/main.cpp

and adds:

    ml/build_dataset.py
    ml/train_mlp.py
    ml/evaluate_mlp.py
    ml/train_feature_baseline.py
    scripts/make_ml_parameter_table.py
    scripts/submit_ml_config_array.sh
    docs/ML_WORKFLOW.md

Rebuild:

    mkdir -p build && cd build && cmake .. && make -j && cd ..

Local ML save test:

    ./build/heisenberg_dmi \
      --L 16 --T 0.8 --J 1.0 --D 1.0 --seed 1 \
      --therm 500 --samples 1000 --skip 5 --proposal 0.7 \
      --init helical_z \
      --out results/raw \
      --save-configs-every 10 \
      --max-saved-configs 20 \
      --config-out results/configs_ml \
      --ml-label 1

Then:

    head results/configs_ml/metadata.csv

Euler ML generation:

    python3 scripts/make_ml_parameter_table.py --L 16
    sbatch --array=0-<LAST>%64 scripts/submit_ml_config_array.sh

Dataset + train + evaluate:

    python3 ml/build_dataset.py --config-dir results/configs_ml --out-dir ml_data --L 16
    python3 ml/train_mlp.py --data-dir ml_data --epochs 40 --batch-size 64
    python3 ml/evaluate_mlp.py --data-dir ml_data
