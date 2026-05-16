#!/usr/bin/env python3
"""
Generate Euler/Slurm parameter table for ML configuration generation.

Main ML plan recommended by the project:
    L = 8, 16, 32
    20 training temperatures + 10 near-transition temperatures
    5 seeds
    50 saved configurations per (L,T,seed)
    total configs = 3 * 30 * 5 * 50 = 22500

Labels:
    low T  -> 1  helical phase
    high T -> 0  disordered phase
    mid T  -> -1 near-transition/test-only

Important:
    Raw-spin MLP and 3D CNN input dimensions depend on L. Build/train one ML model per L:
        ml_data_L8, ml_data_L16, ml_data_L32
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path


def parse_list(text, cast):
    return [cast(x.strip()) for x in text.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outfile', default='parameters/ml_euler_configs_D1_recommended.csv')
    ap.add_argument('--Ls', default='8,16,32')
    ap.add_argument('--include-64', action='store_true')
    ap.add_argument('--J', type=float, default=1.0)
    ap.add_argument('--D', type=float, default=1.0)
    ap.add_argument('--Bx', type=float, default=0.0)
    ap.add_argument('--By', type=float, default=0.0)
    ap.add_argument('--Bz', type=float, default=0.0)
    ap.add_argument('--seeds', default='1,2,3,4,5')

    # 20 training temperatures: 10 low + 10 high.
    ap.add_argument('--low-temps', default='0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95')
    ap.add_argument('--mid-temps', default='1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40,1.45,1.50')
    ap.add_argument('--high-temps', default='1.80,1.85,1.90,1.95,2.00,2.05,2.10,2.15,2.20,2.25')

    ap.add_argument('--therm', type=int, default=3000)
    ap.add_argument('--samples', type=int, default=5000)
    ap.add_argument('--skip', type=int, default=5)
    ap.add_argument('--proposal', type=float, default=0.7)
    ap.add_argument('--init', default='helical_z')

    # Recommended image/table: 50 configs per run.
    ap.add_argument('--save-every', type=int, default=10)
    ap.add_argument('--max-configs', type=int, default=50)
    ap.add_argument('--concurrency', type=int, default=64)
    args = ap.parse_args()

    Ls = parse_list(args.Ls, int)
    if args.include_64 and 64 not in Ls:
        Ls.append(64)
    allowed = {8,16,32,64}
    bad = [L for L in Ls if L not in allowed]
    if bad:
        raise SystemExit(f'Only powers-of-two L in {sorted(allowed)} are allowed; got {bad}')

    seeds = parse_list(args.seeds, int)
    jobs = []
    for T in parse_list(args.low_temps, float):
        jobs.append((T, 1, 'train_low'))
    for T in parse_list(args.mid_temps, float):
        jobs.append((T, -1, 'test_transition'))
    for T in parse_list(args.high_temps, float):
        jobs.append((T, 0, 'train_high'))

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ['run_id','L','T','J','D','Bx','By','Bz','seed','therm','samples','skip','proposal','init','save_every','max_configs','label','split']
    run_id = 0
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for L in sorted(Ls):
            for T, label, split in jobs:
                for seed in seeds:
                    w.writerow(dict(run_id=run_id, L=L, T=T, J=args.J, D=args.D,
                                    Bx=args.Bx, By=args.By, Bz=args.Bz, seed=seed,
                                    therm=args.therm, samples=args.samples, skip=args.skip,
                                    proposal=args.proposal, init=args.init,
                                    save_every=args.save_every, max_configs=args.max_configs,
                                    label=label, split=split))
                    run_id += 1
    last = run_id - 1
    total_configs = run_id * args.max_configs
    print(f'Wrote {run_id} ML config-generation runs to {out}')
    print(f'Expected saved configs: {run_id} * {args.max_configs} = {total_configs}')
    print('Submit:')
    print('  mkdir -p logs/euler results/raw results/configs_ml')
    print(f'  PARAM_FILE={out} sbatch --array=0-{last}%{args.concurrency} scripts/submit_ml_config_array.sh')
    print('After jobs finish:')
    print('  python3 scripts/merge_ml_metadata.py --config-root results/configs_ml')
    print('  # Build one dataset/model per L, e.g.:')
    print('  python3 ml/build_dataset.py --config-dir results/configs_ml --out-dir ml_data_L16 --L 16 --low-max 0.95 --high-min 1.80 --store-cnn')

if __name__ == '__main__':
    main()
