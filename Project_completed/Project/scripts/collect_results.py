#!/usr/bin/env python3
"""
Collect one-line summary CSV files from results/raw into results/processed/summary.csv.
"""
from pathlib import Path
import pandas as pd

raw_dir = Path("results/raw")
out_dir = Path("results/processed")
out_dir.mkdir(parents=True, exist_ok=True)

files = sorted(raw_dir.glob("summary_*.csv"))
if not files:
    raise SystemExit(f"No summary_*.csv files found in {raw_dir}")

frames = []
for f in files:
    frames.append(pd.read_csv(f))

df = pd.concat(frames, ignore_index=True)
df = df.sort_values(["L", "T", "seed"]).reset_index(drop=True)

out = out_dir / "summary.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows to {out}")

# Also average over seeds for each L,T,J,D.
group_cols = ["L", "T", "J", "D", "Bx", "By", "Bz"]
value_cols = [c for c in df.columns if c not in group_cols and c not in ["seed", "therm_sweeps", "samples", "skip", "proposal_width"]]
summary_mean = df.groupby(group_cols, as_index=False)[value_cols].mean()
summary_std = df.groupby(group_cols, as_index=False)[value_cols].std()

mean_out = out_dir / "summary_by_L_T_mean.csv"
std_out = out_dir / "summary_by_L_T_std.csv"
summary_mean.to_csv(mean_out, index=False)
summary_std.to_csv(std_out, index=False)
print(f"Wrote seed-averaged means to {mean_out}")
print(f"Wrote seed standard deviations to {std_out}")
