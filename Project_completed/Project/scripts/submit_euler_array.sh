#!/bin/bash
#SBATCH --job-name=heis_dmi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=12:00:00
#SBATCH --output=logs/euler/%x_%A_%a.out
#SBATCH --error=logs/euler/%x_%A_%a.err

set -euo pipefail

# Run this script from the project root with:
#   sbatch --array=0-N%64 scripts/submit_euler_array.sh
#
# It uses the Slurm array index SLURM_ARRAY_TASK_ID to pick one row
# from parameters/euler_scan_D1.csv.

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

PARAM_FILE="${PARAM_FILE:-parameters/euler_scan_D1.csv}"
EXECUTABLE="${EXECUTABLE:-build/heisenberg_dmi}"

mkdir -p results/raw results/processed results/configs logs/euler

if [[ ! -f "$PARAM_FILE" ]]; then
    echo "Parameter file not found: $PARAM_FILE"
    echo "Generate it with:"
    echo "  python3 scripts/make_euler_parameter_table.py --mode full"
    exit 1
fi

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "Executable not found or not executable: $EXECUTABLE"
    echo "Build it first on Euler with:"
    echo "  bash scripts/build_on_euler.sh"
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"

# Find CSV row with run_id == TASK_ID.
LINE=$(awk -F',' -v id="$TASK_ID" 'NR > 1 && $1 == id {print; exit}' "$PARAM_FILE")

if [[ -z "$LINE" ]]; then
    echo "No parameter row found for run_id=$TASK_ID in $PARAM_FILE"
    exit 1
fi

IFS=',' read -r run_id L T J D Bx By Bz seed therm samples skip proposal init <<< "$LINE"
init=$(echo "$init" | tr -d '\r')

echo "============================================================"
echo "Euler job information"
echo "Job ID:       ${SLURM_JOB_ID:-unknown}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID:-unknown}"
echo "Node:         $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Parameters:   run_id=$run_id L=$L T=$T J=$J D=$D seed=$seed"
echo "Therm/samples/skip/proposal: $therm / $samples / $skip / $proposal"
echo "============================================================"

"$EXECUTABLE" \
  --L "$L" --T "$T" --J "$J" --D "$D" \
  --Bx "$Bx" --By "$By" --Bz "$Bz" \
  --seed "$seed" \
  --therm "$therm" --samples "$samples" --skip "$skip" \
  --proposal "$proposal" --init "$init" \
  --out results/raw

echo "Finished run_id=$run_id"
