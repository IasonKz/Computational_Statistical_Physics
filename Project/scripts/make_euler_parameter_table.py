#!/usr/bin/env python3
"""
Generate Euler/Slurm parameter table for physics production scans.

Default sizes are powers of two only:
    L = 8, 16, 32

Optional heavy run:
    --include-64

Recommended final model:
    J = 1, D = 1, B = 0
"""

from __future__ import annotations
import argparse, csv
from pathlib import Path

def parse_csv_numbers(text: str, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]

def make_temperature_grid(mode: str):
    if mode == "coarse":
        return [round(0.8 + 0.1*i, 10) for i in range(13)]       # 0.8...2.0
    if mode == "refined":
        return [round(1.1 + 0.05*i, 10) for i in range(13)]      # 1.1...1.6
    if mode == "full":
        return sorted(set(
            [round(0.8 + 0.1*i, 10) for i in range(13)] +
            [round(1.1 + 0.05*i, 10) for i in range(13)]
        ))
    raise ValueError(mode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfile", default="parameters/euler_scan_D1_powers2.csv")
    ap.add_argument("--mode", choices=["coarse","refined","full"], default="full")
    ap.add_argument("--Ls", default="8,16,32", help="Powers of two only by convention.")
    ap.add_argument("--include-64", action="store_true")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--D", type=float, default=1.0)
    ap.add_argument("--Bx", type=float, default=0.0)
    ap.add_argument("--By", type=float, default=0.0)
    ap.add_argument("--Bz", type=float, default=0.0)
    ap.add_argument("--therm", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--proposal", type=float, default=0.7)
    ap.add_argument("--init", default="helical_z")
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()

    Ls = parse_csv_numbers(args.Ls, int)
    if args.include_64 and 64 not in Ls:
        Ls.append(64)
    allowed = {8,16,32,64}
    bad = [L for L in Ls if L not in allowed]
    if bad:
        raise SystemExit(f"Only powers-of-two L in {sorted(allowed)} are allowed; got {bad}")

    temps = make_temperature_grid(args.mode)
    seeds = parse_csv_numbers(args.seeds, int)

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id","L","T","J","D","Bx","By","Bz","seed","therm","samples","skip","proposal","init"]

    run_id = 0
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for L in sorted(Ls):
            for T in temps:
                for seed in seeds:
                    w.writerow(dict(run_id=run_id, L=L, T=T, J=args.J, D=args.D,
                                    Bx=args.Bx, By=args.By, Bz=args.Bz, seed=seed,
                                    therm=args.therm, samples=args.samples, skip=args.skip,
                                    proposal=args.proposal, init=args.init))
                    run_id += 1

    last = run_id - 1
    print(f"Wrote {run_id} runs to {out}")
    print("Submit:")
    print("  mkdir -p logs/euler results/raw results/processed")
    print(f"  PARAM_FILE={out} sbatch --array=0-{last}%{args.concurrency} scripts/submit_euler_array.sh")

if __name__ == "__main__":
    main()
