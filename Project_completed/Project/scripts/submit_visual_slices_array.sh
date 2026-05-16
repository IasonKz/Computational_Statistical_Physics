#!/bin/bash
#SBATCH --job-name=dmi_vis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --output=logs/euler/%x_%A_%a.out
#SBATCH --error=logs/euler/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

PARAM_FILE="${PARAM_FILE:-parameters/visual_slices_L32.csv}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"

LINE=$(awk -F',' -v id="$TASK_ID" 'NR > 1 && $1 == id {print; exit}' "$PARAM_FILE")

if [[ -z "$LINE" ]]; then
    echo "No row found for run_id=$TASK_ID"
    exit 1
fi

IFS=',' read -r run_id L T J D seed therm samples skip proposal init <<< "$LINE"
init=$(echo "$init" | tr -d '\r')

mkdir -p results/raw results/configs logs/euler

echo "Visualization run_id=$run_id L=$L T=$T seed=$seed"

./build/heisenberg_dmi \
  --L "$L" --T "$T" --J "$J" --D "$D" --seed "$seed" \
  --therm "$therm" --samples "$samples" --skip "$skip" \
  --proposal "$proposal" --init "$init" \
  --out results/raw \
  --save-config

echo "Done visualization run_id=$run_id"
