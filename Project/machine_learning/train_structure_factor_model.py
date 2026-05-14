#!/usr/bin/env python3
"""
Train a structure-factor classifier.

Uses S(q) features built by build_structure_factor_dataset.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def crossing(T, P):
    for i in range(len(T)-1):
        if (P[i]-0.5)*(P[i+1]-0.5) < 0:
            return float(T[i] + (0.5-P[i])*(T[i+1]-T[i])/(P[i+1]-P[i]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='ml_data_sf_L16')
    ap.add_argument('--model', choices=['logistic','mlp'], default='mlp')
    args = ap.parse_args()
    data = Path(args.data_dir)
    Path('figures/report').mkdir(parents=True, exist_ok=True)

    Xtr = np.load(data/'Xsf_train.npy'); ytr = np.load(data/'ysf_train.npy').astype(int)
    Xva = np.load(data/'Xsf_val.npy'); yva = np.load(data/'ysf_val.npy').astype(int)
    Xte = np.load(data/'Xsf_test.npy'); yte = np.load(data/'ysf_test.npy')
    Tte = np.load(data/'Tsf_test.npy')

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    import joblib

    if args.model == 'logistic':
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    else:
        clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=123))
    clf.fit(Xtr, ytr)
    print('train acc', accuracy_score(ytr, clf.predict(Xtr)))
    print('val acc', accuracy_score(yva, clf.predict(Xva)))
    print('val confusion')
    print(confusion_matrix(yva, clf.predict(Xva)))

    classes = list(clf.classes_)
    idx1 = classes.index(1)
    P = clf.predict_proba(Xte)[:,idx1]
    df = pd.DataFrame({'T':Tte, 'y_true':yte, 'P_helical':P})
    df.to_csv(data/'structure_factor_predictions.csv', index=False)
    grouped = df.groupby('T').agg(P_helical_mean=('P_helical','mean'), P_helical_std=('P_helical','std'), count=('P_helical','size')).reset_index().sort_values('T')
    grouped['P_helical_std'] = grouped['P_helical_std'].fillna(0.0)
    grouped['P_helical_stderr'] = grouped['P_helical_std']/np.sqrt(grouped['count'])
    grouped.to_csv(data/'structure_factor_P_helical_by_T.csv', index=False)
    Tc = crossing(grouped['T'].to_numpy(float), grouped['P_helical_mean'].to_numpy(float))
    print('Tc_structure_factor', Tc)
    joblib.dump(clf, data/'structure_factor_model.joblib')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6,4.4))
    ax.errorbar(grouped['T'], grouped['P_helical_mean'], yerr=grouped['P_helical_stderr'], marker='o', capsize=3)
    ax.axhline(0.5, linestyle='--', linewidth=1)
    if Tc is not None:
        ax.axvline(Tc, linestyle=':', label=fr'$T_c^{{SF}}\approx {Tc:.3f}$')
    ax.set_xlabel('Temperature T'); ax.set_ylabel(r'$P_{\mathrm{helical}}$')
    ax.set_title('Structure-factor model')
    ax.set_ylim(-0.05,1.05); ax.legend()
    fig.tight_layout()
    tag = data.name
    fig.savefig(f'figures/report/structure_factor_P_helical_vs_T_{tag}.pdf')
    fig.savefig(f'figures/report/structure_factor_P_helical_vs_T_{tag}.png', dpi=200)

if __name__ == '__main__':
    main()
