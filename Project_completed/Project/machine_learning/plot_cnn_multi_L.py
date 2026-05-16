#!/usr/bin/env python3
"""Plot CNN P_helical(T) for multiple L-specific datasets."""
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dirs', default='ml_data_L8,ml_data_L16,ml_data_L32')
    args=ap.parse_args()
    Path('figures/report').mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(7,4.6))
    for dtext in [x.strip() for x in args.dirs.split(',') if x.strip()]:
        d=Path(dtext); f=d/'cnn3d_P_helical_by_T.csv'
        if not f.exists():
            print('skip missing', f); continue
        df=pd.read_csv(f)
        m=re.search(r'L(\d+)', d.name); label=f'L={m.group(1)}' if m else d.name
        err=df.get('P_helical_stderr', pd.Series(np.zeros(len(df))))
        ax.errorbar(df['T'], df['P_helical_mean'], yerr=err, marker='o', capsize=3, label=label)
    ax.axhline(0.5, linestyle='--', linewidth=1)
    ax.set_xlabel('Temperature T'); ax.set_ylabel(r'$P_{\mathrm{helical}}$ from 3D CNN')
    ax.set_ylim(-0.05,1.05); ax.legend(); fig.tight_layout()
    fig.savefig('figures/report/cnn3d_P_helical_vs_T_multi_L.pdf')
    fig.savefig('figures/report/cnn3d_P_helical_vs_T_multi_L.png', dpi=200)
    print('saved figures/report/cnn3d_P_helical_vs_T_multi_L.pdf')

if __name__ == '__main__': main()
