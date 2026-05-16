#!/usr/bin/env python3
"""Evaluate 3D CNN and plot P_helical(T)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import torch
except ImportError as e:
    raise SystemExit('PyTorch required') from e
from models import SpinCNN3D


def crossing(T,P):
    for i in range(len(T)-1):
        if (P[i]-0.5)*(P[i+1]-0.5) < 0:
            return float(T[i] + (0.5-P[i])*(T[i+1]-T[i])/(P[i+1]-P[i]))
    return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='ml_data_L16')
    ap.add_argument('--model', default=None)
    ap.add_argument('--batch-size', type=int, default=64)
    args=ap.parse_args()
    data=Path(args.data_dir); model_path=Path(args.model) if args.model else data/'models'/'cnn3d_spins.pt'
    Path('figures/report').mkdir(parents=True, exist_ok=True)
    X=np.load(data/'X_test_cnn.npy').astype(np.float32)
    y=np.load(data/'y_test.npy').astype(np.float32)
    T=np.load(data/'T_test.npy').astype(np.float32)
    ckpt=torch.load(model_path, map_location='cpu')
    model=SpinCNN3D(channels=3, dropout=ckpt.get('dropout',0.25))
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    probs=[]
    with torch.no_grad():
        for i in range(0,len(X),args.batch_size):
            logits=model(torch.from_numpy(X[i:i+args.batch_size]))
            probs.append(torch.sigmoid(logits).numpy())
    P=np.concatenate(probs)
    pred=(P>=0.5).astype(int)
    labeled=y>=0
    if labeled.any(): print('labeled test acc', float((pred[labeled]==y[labeled]).mean()))
    df=pd.DataFrame({'T':T,'y_true':y,'P_helical':P,'pred_label':pred})
    df.to_csv(data/'cnn3d_predictions.csv', index=False)
    grouped=df.groupby('T').agg(P_helical_mean=('P_helical','mean'),P_helical_std=('P_helical','std'),count=('P_helical','size')).reset_index().sort_values('T')
    grouped['P_helical_std']=grouped['P_helical_std'].fillna(0.0)
    grouped['P_helical_stderr']=grouped['P_helical_std']/np.sqrt(grouped['count'])
    grouped.to_csv(data/'cnn3d_P_helical_by_T.csv', index=False)
    Tc=crossing(grouped['T'].to_numpy(float), grouped['P_helical_mean'].to_numpy(float))
    print('Tc_CNN3D', Tc)
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(6.6,4.4))
    ax.errorbar(grouped['T'], grouped['P_helical_mean'], yerr=grouped['P_helical_stderr'], marker='o', capsize=3, label='3D CNN')
    ax.axhline(0.5, linestyle='--', linewidth=1)
    if Tc is not None: ax.axvline(Tc, linestyle=':', label=fr'$T_c^{{CNN}}\approx {Tc:.3f}$')
    ax.set_xlabel('Temperature T'); ax.set_ylabel(r'$P_{\mathrm{helical}}$'); ax.set_ylim(-0.05,1.05); ax.legend(); fig.tight_layout()
    fig.savefig(f'figures/report/cnn3d_P_helical_vs_T_{data.name}.pdf')
    fig.savefig(f'figures/report/cnn3d_P_helical_vs_T_{data.name}.png', dpi=200)

if __name__ == '__main__': main()
