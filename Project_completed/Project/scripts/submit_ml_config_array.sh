#!/bin/bash
#SBATCH --job-name=ml_cfg_dmi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=12:00:00
#SBATCH --output=logs/euler/%x_%A_%a.out
#SBATCH --error=logs/euler/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

PARAM_FILE="${PARAM_FILE:-parameters/ml_euler_configs_D1_recommended.csv}"
EXECUTABLE="${EXECUTABLE:-build/heisenberg_dmi}"

mkdir -p results/raw results/processed results/configs_ml logs/euler

if [[ ! -f "$PARAM_FILE" ]]; then
    echo "Parameter file not found: $PARAM_FILE"
    echo "Generate it with: python3 scripts/make_ml_euler_parameter_table.py"
    exit 1
fi
if [[ ! -x "$EXECUTABLE" ]]; then
    echo "Executable not found: $EXECUTABLE"
    echo "Build first with: bash scripts/build_on_euler.sh"
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
LINE=$(awk -F',' -v id="$TASK_ID" 'NR > 1 && $1 == id {print; exit}' "$PARAM_FILE")
if [[ -z "$LINE" ]]; then
    echo "No parameter row found for run_id=$TASK_ID"
    exit 1
fi

IFS=',' read -r run_id L T J D Bx By Bz seed therm samples skip proposal init save_every max_configs label split <<< "$LINE"
init=$(echo "$init" | tr -d '\r')
split=$(echo "$split" | tr -d '\r')

CONFIG_OUT="results/configs_ml/L${L}/run_${run_id}"
mkdir -p "$CONFIG_OUT"

echo "============================================================"
echo "ML config job: run_id=$run_id L=$L T=$T seed=$seed label=$label split=$split"
echo "save_every=$save_every max_configs=$max_configs config_out=$CONFIG_OUT"
echo "node=$(hostname) job=${SLURM_JOB_ID:-unknown} task=${SLURM_ARRAY_TASK_ID:-unknown}"
echo "============================================================"

"$EXECUTABLE" \
  --L "$L" --T "$T" --J "$J" --D "$D" \
  --Bx "$Bx" --By "$By" --Bz "$Bz" \
  --seed "$seed" \
  --therm "$therm" --samples "$samples" --skip "$skip" \
  --proposal "$proposal" --init "$init" \
  --out results/raw \
  --save-configs-every "$save_every" \
  --max-saved-configs "$max_configs" \
  --config-out "$CONFIG_OUT" \
  --ml-label "$label"

echo "Finished run_id=$run_id"
