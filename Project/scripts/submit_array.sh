#!/bin/bash
#SBATCH --job-name=heis_dmi
#SBATCH --output=logs/heis_dmi_%A_%a.out
#SBATCH --error=logs/heis_dmi_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1

# Usage:
#   python scripts/make_parameter_table.py
#   mkdir -p logs results/raw
#   sbatch --array=0-107 scripts/submit_array.sh
#
# On Euler, load the compiler/CMake modules appropriate for your environment before building.
# Then submit from the termproject directory.

set -euo pipefail
mkdir -p logs results/raw

PARAM_FILE="parameters/scan_L_T_seed.csv"
EXE="./build/heisenberg_dmi"

if [[ ! -x "$EXE" ]]; then
    echo "Executable $EXE not found. Build first: mkdir -p build && cd build && cmake .. && make -j"
    exit 1
fi

# Extract row with run_id == SLURM_ARRAY_TASK_ID using Python's csv parser.
eval $(python3 - <<'PY'
import csv, os, shlex
param_file = "parameters/scan_L_T_seed.csv"
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
with open(param_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row["run_id"]) == task_id:
            for k, v in row.items():
                print(f"{k}={shlex.quote(v)}")
            break
    else:
        raise SystemExit(f"No row with run_id={task_id}")
PY
)

echo "Running task ${SLURM_ARRAY_TASK_ID}: L=$L T=$T seed=$seed"

$EXE \
  --L "$L" --T "$T" --J "$J" --D "$D" \
  --Bx "$Bx" --By "$By" --Bz "$Bz" \
  --seed "$seed" \
  --therm "$therm" --samples "$samples" --skip "$skip" \
  --proposal "$proposal" --init "$init" \
  --out results/raw
