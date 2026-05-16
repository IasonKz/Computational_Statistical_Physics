#!/usr/bin/env python3
"""
Evaluate the raw-spin MLP and plot P_helical(T).

Example:
    python3 ml/evaluate_mlp.py --data-dir ml_data
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import torch
except ImportError as e:
    raise SystemExit("PyTorch is required.") from e

from models import RawSpinMLP


def estimate_crossing(grouped: pd.DataFrame):
    grouped = grouped.sort_values("T")
    T = grouped["T"].to_numpy(float)
    P = grouped["P_helical_mean"].to_numpy(float)

    for i in range(len(T) - 1):
        if (P[i] - 0.5) == 0:
            return float(T[i])
        if (P[i] - 0.5) * (P[i + 1] - 0.5) < 0:
            return float(T[i] + (0.5 - P[i]) * (T[i + 1] - T[i]) / (P[i + 1] - P[i]))

    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="ml_data")
    p.add_argument("--model", default=None)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    data = Path(args.data_dir)
    model_path = Path(args.model) if args.model else data / "models" / "mlp_raw_spins.pt"
    Path("figures/report").mkdir(parents=True, exist_ok=True)

    X = np.load(data / "X_test.npy").astype(np.float32)
    y = np.load(data / "y_test.npy").astype(np.float32)
    T = np.load(data / "T_test.npy").astype(np.float32)

    checkpoint = torch.load(model_path, map_location="cpu")
    model = RawSpinMLP(
        input_dim=checkpoint["input_dim"],
        hidden1=checkpoint.get("hidden1", 256),
        hidden2=checkpoint.get("hidden2", 64),
        dropout=checkpoint.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probs = []
    with torch.no_grad():
        for start in range(0, len(X), args.batch_size):
            xb = torch.from_numpy(X[start:start + args.batch_size])
            logits = model(xb)
            probs.append(torch.sigmoid(logits).numpy())

    P = np.concatenate(probs)
    pred = (P >= 0.5).astype(int)

    labeled = y >= 0
    if labeled.any():
        acc = float((pred[labeled] == y[labeled]).mean())
        print(f"Labeled test accuracy: {acc:.4f}")

    df = pd.DataFrame({
        "T": T,
        "y_true": y,
        "P_helical": P,
        "pred_label": pred,
    })
    df.to_csv(data / "mlp_predictions.csv", index=False)

    grouped = (
        df.groupby("T")
        .agg(
            P_helical_mean=("P_helical", "mean"),
            P_helical_std=("P_helical", "std"),
            count=("P_helical", "size"),
        )
        .reset_index()
        .sort_values("T")
    )
    grouped["P_helical_std"] = grouped["P_helical_std"].fillna(0.0)
    grouped["P_helical_stderr"] = grouped["P_helical_std"] / np.sqrt(grouped["count"])
    grouped.to_csv(data / "mlp_P_helical_by_T.csv", index=False)

    Tc = estimate_crossing(grouped)
    print("Tc_ML from P_helical=0.5:", Tc)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        ax.errorbar(
            grouped["T"],
            grouped["P_helical_mean"],
            yerr=grouped["P_helical_stderr"],
            marker="o",
            capsize=3,
            label="ML classifier",
        )
        ax.axhline(0.5, linestyle="--", linewidth=1.0, label="P=0.5")
        if Tc is not None:
            ax.axvline(Tc, linestyle=":", linewidth=1.2,
                       label=fr"$T_c^{{ML}}\approx {Tc:.3f}$")
        ax.set_xlabel("Temperature T")
        ax.set_ylabel(r"$P_{\mathrm{helical}}$")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig("figures/report/mlp_P_helical_vs_T.pdf")
        fig.savefig("figures/report/mlp_P_helical_vs_T.png", dpi=200)
        plt.close(fig)

    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
