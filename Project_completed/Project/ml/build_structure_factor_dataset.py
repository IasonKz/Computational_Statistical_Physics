#!/usr/bin/env python3
"""
Build a structure-factor feature dataset from saved spin configurations.

For each config, compute S(q) along x,y,z for n=1..nmax.
This gives a physics-informed supervised ML model that still uses configurations.

Example:
    python3 ml/build_structure_factor_dataset.py \
      --config-dir results/configs_ml --out-dir ml_data_sf_L16 --L 16 --nmax 8 \
      --low-max 0.95 --high-min 1.80
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def path_column(columns):
    for c in ['filename','file','path','filepath']:
        if c in columns:
            return c
    raise SystemExit('metadata must contain filename/file/path/filepath column')


def label_from_row(row, low_max, high_min):
    for col in ['label','phase_label','ml_label']:
        if col in row and not pd.isna(row[col]):
            lab = int(row[col])
            if lab in (-1,0,1):
                return lab
    T = float(row['T'])
    return 1 if T <= low_max else (0 if T >= high_min else -1)


def read_config(path, L):
    arr = np.fromfile(path, dtype=np.float32)
    expected = L*L*L*3
    if arr.size != expected:
        raise ValueError(f'{path}: got {arr.size}, expected {expected}')
    return arr.reshape(L,L,L,3)


def sf_axis(spins, axis, n):
    L = spins.shape[0]
    coords = np.arange(L, dtype=np.float64)
    phase = np.exp(1j * 2*np.pi*n*coords/L)
    if axis == 0:
        weights = phase[:,None,None,None]
    elif axis == 1:
        weights = phase[None,:,None,None]
    else:
        weights = phase[None,None,:,None]
    amp = np.sum(spins * weights, axis=(0,1,2))
    N = L**3
    return float(np.sum(np.abs(amp)**2) / (N*N))


def features_from_config(spins, nmax):
    feats = []
    for axis in range(3):
        for n in range(1, nmax+1):
            feats.append(sf_axis(spins, axis, n))
    feats = np.array(feats, dtype=np.float32)
    # helpful summary features
    extra = np.array([feats.max(), feats.mean(), feats.std()], dtype=np.float32)
    return np.concatenate([feats, extra])


def split_indices(y, val_fraction, seed):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for lab in sorted(set(y.tolist())):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        nval = max(1, int(round(len(idx)*val_fraction))) if len(idx) > 1 else 0
        val.extend(idx[:nval].tolist())
        train.extend(idx[nval:].tolist())
    rng.shuffle(train); rng.shuffle(val)
    return np.array(train), np.array(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-dir', default='results/configs_ml')
    ap.add_argument('--metadata', default=None)
    ap.add_argument('--out-dir', default='ml_data_sf_L16')
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--nmax', type=int, default=8)
    ap.add_argument('--low-max', type=float, default=0.95)
    ap.add_argument('--high-min', type=float, default=1.80)
    ap.add_argument('--val-fraction', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--max-configs', type=int, default=0)
    args = ap.parse_args()

    config_dir = Path(args.config_dir)
    meta_path = Path(args.metadata) if args.metadata else config_dir/'metadata.csv'
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(meta_path)
    pc = path_column(list(meta.columns))
    if 'L' in meta.columns:
        meta = meta[meta['L'].astype(int) == args.L].copy()
    if args.max_configs and len(meta) > args.max_configs:
        meta = meta.sample(args.max_configs, random_state=args.seed).reset_index(drop=True)
    else:
        meta = meta.reset_index(drop=True)

    X, rows = [], []
    for _, row in meta.iterrows():
        rel = Path(str(row[pc]))
        p = rel if rel.is_absolute() else config_dir/rel
        if not p.exists():
            print('missing', p); continue
        try:
            spins = read_config(p, args.L)
            X.append(features_from_config(spins, args.nmax))
            rows.append(row)
        except Exception as e:
            print('skip', p, e)
    if not X:
        raise SystemExit('no configs loaded')
    X = np.stack(X).astype(np.float32)
    used = pd.DataFrame(rows).reset_index(drop=True)
    used['phase_label'] = used.apply(lambda r: label_from_row(r,args.low_max,args.high_min), axis=1)
    y = used['phase_label'].to_numpy(np.int64)
    T = used['T'].to_numpy(np.float32)
    labeled = np.where(y >= 0)[0]
    tr, va = split_indices(y[labeled], args.val_fraction, args.seed)
    train_idx, val_idx = labeled[tr], labeled[va]
    test_idx = np.arange(len(used))

    np.save(out/'Xsf_train.npy', X[train_idx]); np.save(out/'ysf_train.npy', y[train_idx].astype(np.float32)); np.save(out/'Tsf_train.npy', T[train_idx])
    np.save(out/'Xsf_val.npy', X[val_idx]); np.save(out/'ysf_val.npy', y[val_idx].astype(np.float32)); np.save(out/'Tsf_val.npy', T[val_idx])
    np.save(out/'Xsf_test.npy', X[test_idx]); np.save(out/'ysf_test.npy', y[test_idx].astype(np.float32)); np.save(out/'Tsf_test.npy', T[test_idx])
    used.to_csv(out/'metadata_used_sf.csv', index=False)
    print('wrote', out)
    print('X shape', X.shape, 'train', len(train_idx), 'val', len(val_idx), 'test', len(test_idx))

if __name__ == '__main__':
    main()
