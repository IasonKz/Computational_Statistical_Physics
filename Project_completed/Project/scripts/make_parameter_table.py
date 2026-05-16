#!/usr/bin/env python3
"""
Create a parameter table for Slurm array runs.
Adjust L_values, T_values and seeds below for production scans.
"""
from pathlib import Path
import csv
import numpy as np

out = Path("parameters/scan_L_T_seed.csv")
out.parent.mkdir(parents=True, exist_ok=True)

J = 1.0
D = 0.7
B = (0.0, 0.0, 0.0)

# Starter scan. Increase these for production runs.
L_values = [8, 10, 12]
T_values = np.round(np.linspace(0.7, 1.8, 12), 6)
seeds = [1, 2, 3]

thermalization_sweeps = 500
measurement_samples = 1000
sweeps_between_measurements = 3
proposal_width = 0.35
init_mode = "helical_z"

rows = []
run_id = 0
for L in L_values:
    for T in T_values:
        for seed in seeds:
            rows.append({
                "run_id": run_id,
                "L": L,
                "T": float(T),
                "J": J,
                "D": D,
                "Bx": B[0],
                "By": B[1],
                "Bz": B[2],
                "seed": seed,
                "therm": thermalization_sweeps,
                "samples": measurement_samples,
                "skip": sweeps_between_measurements,
                "proposal": proposal_width,
                "init": init_mode,
            })
            run_id += 1

with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} runs to {out}")
print("Submit with something like:")
print(f"  sbatch --array=0-{len(rows)-1} scripts/submit_array.sh")
