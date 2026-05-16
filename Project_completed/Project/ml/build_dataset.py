#!/usr/bin/env python3
"""
Build supervised-learning datasets from saved Heisenberg+DMI spin configurations.

Expected input:
    results/configs_ml/metadata.csv
    results/configs_ml/<config files>.bin

Each .bin file is expected to contain raw float32 values of shape:
    (L, L, L, 3)

Label convention:
    label = 1   helical / ordered
    label = 0   disordered / paramagnetic
    label = -1  transition/test-only, excluded from training

If the metadata does not contain a valid explicit label, labels can be inferred from T:
    T <= low_max  -> 1
    T >= high_min -> 0
    otherwise     -> -1

Example:
    python3 ml/build_dataset.py \
      --config-dir results/configs_ml \
      --out-dir ml_data \
      --L 16 \
      --low-max 0.9 \
      --high-min 1.8 \
      --store-cnn
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def find_path_column(columns: list[str]) -> str:
    for c in ["filename", "file", "path", "filepath"]:
        if c in columns:
            return c
    raise ValueError(
        "metadata.csv must contain one of: filename, file, path, filepath"
    )


def infer_or_read_label(row: pd.Series, low_max: float, high_min: float) -> int:
    for col in ["label", "phase_label", "ml_label"]:
        if col in row and not pd.isna(row[col]):
            try:
                lab = int(row[col])
                if lab in (-1, 0, 1):
                    return lab
            except Exception:
                pass

    T = float(row["T"])
    if T <= low_max:
        return 1
    if T >= high_min:
        return 0
    return -1


def read_config_binary(path: Path, L: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    expected = L * L * L * 3
    if arr.size != expected:
        raise ValueError(f"{path}: got {arr.size} float32 values, expected {expected}")
    return arr.reshape(L, L, L, 3)


def stratified_train_val_split(y: np.ndarray, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []

    for lab in sorted(set(y.tolist())):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)

        if len(idx) <= 1:
            n_val = 0
        else:
            n_val = max(1, int(round(val_fraction * len(idx))))

        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-dir", default="results/configs_ml")
    p.add_argument("--metadata", default=None)
    p.add_argument("--out-dir", default="ml_data")
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--low-max", type=float, default=0.9)
    p.add_argument("--high-min", type=float, default=1.8)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-configs", type=int, default=0,
                   help="Optional global random subsample. 0 means use all.")
    p.add_argument("--max-per-temperature", type=int, default=0,
                   help="Optional cap per temperature. 0 means no cap.")
    p.add_argument("--store-cnn", action="store_true",
                   help="Also store X_*_cnn.npy with shape (N, 3, L, L, L).")
    args = p.parse_args()

    config_dir = Path(args.config_dir)
    metadata_path = Path(args.metadata) if args.metadata else config_dir / "metadata.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise SystemExit(f"metadata file not found: {metadata_path}")

    meta = pd.read_csv(metadata_path)
    path_col = find_path_column(list(meta.columns))

    if args.L is None:
        if "L" not in meta.columns:
            raise SystemExit("metadata has no L column; pass --L explicitly.")
        L_values = sorted(meta["L"].dropna().unique().tolist())
        if len(L_values) != 1:
            raise SystemExit(f"Multiple L values found: {L_values}. Pass --L.")
        L = int(L_values[0])
    else:
        L = args.L
        if "L" in meta.columns:
            meta = meta[meta["L"].astype(int) == L].copy()

    if args.max_per_temperature > 0:
        meta = (
            meta.groupby("T", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), args.max_per_temperature),
                                      random_state=args.seed))
            .reset_index(drop=True)
        )

    if args.max_configs > 0 and len(meta) > args.max_configs:
        meta = meta.sample(args.max_configs, random_state=args.seed).reset_index(drop=True)
    else:
        meta = meta.reset_index(drop=True)

    rows = []
    X_list = []

    for _, row in meta.iterrows():
        rel = Path(str(row[path_col]))
        path = rel if rel.is_absolute() else config_dir / rel

        if not path.exists():
            print(f"Missing config, skipping: {path}")
            continue

        try:
            config = read_config_binary(path, L)
        except Exception as e:
            print(f"Failed to read {path}, skipping. Error: {e}")
            continue

        X_list.append(config.reshape(-1))
        rows.append(row)

    if not X_list:
        raise SystemExit("No configurations loaded.")

    X = np.stack(X_list).astype(np.float32)
    used = pd.DataFrame(rows).reset_index(drop=True)
    used["phase_label"] = used.apply(
        lambda r: infer_or_read_label(r, args.low_max, args.high_min),
        axis=1,
    )

    y = used["phase_label"].to_numpy(dtype=np.int64)
    T = used["T"].to_numpy(dtype=np.float32)

    labeled = np.where(y >= 0)[0]
    if len(labeled) < 4:
        raise SystemExit("Not enough labeled samples for train/validation.")

    train_local, val_local = stratified_train_val_split(
        y[labeled], args.val_fraction, args.seed
    )

    train_idx = labeled[train_local]
    val_idx = labeled[val_local]
    test_idx = np.arange(len(used), dtype=int)

    np.save(out_dir / "X_train.npy", X[train_idx])
    np.save(out_dir / "y_train.npy", y[train_idx].astype(np.float32))
    np.save(out_dir / "T_train.npy", T[train_idx])

    np.save(out_dir / "X_val.npy", X[val_idx])
    np.save(out_dir / "y_val.npy", y[val_idx].astype(np.float32))
    np.save(out_dir / "T_val.npy", T[val_idx])

    np.save(out_dir / "X_test.npy", X[test_idx])
    np.save(out_dir / "y_test.npy", y[test_idx].astype(np.float32))
    np.save(out_dir / "T_test.npy", T[test_idx])

    if args.store_cnn:
        X_cnn = X.reshape((-1, L, L, L, 3)).transpose(0, 4, 1, 2, 3).astype(np.float32)
        np.save(out_dir / "X_train_cnn.npy", X_cnn[train_idx])
        np.save(out_dir / "X_val_cnn.npy", X_cnn[val_idx])
        np.save(out_dir / "X_test_cnn.npy", X_cnn[test_idx])

    used.to_csv(out_dir / "metadata_used.csv", index=False)

    print(f"Dataset written to {out_dir}")
    print(f"L = {L}")
    print(f"loaded configs = {len(used)}")
    print(f"train = {len(train_idx)}, val = {len(val_idx)}, test = {len(test_idx)}")
    print("train labels:", dict(zip(*np.unique(y[train_idx], return_counts=True))))
    print("val labels:  ", dict(zip(*np.unique(y[val_idx], return_counts=True))))
    print("test labels: ", dict(zip(*np.unique(y[test_idx], return_counts=True))))
    print("test temperatures:", sorted(np.unique(T).tolist()))


if __name__ == "__main__":
    main()
