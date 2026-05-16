#!/usr/bin/env python3
"""
Feature-based baseline classifier.

This is NOT the main ML result, but it is useful as a sanity check.
It uses physics-inspired features from results/processed/summary.csv.

Example:
    python3 ml/train_feature_baseline.py \
      --summary results/processed/summary.csv \
      --low-max 0.9 \
      --high-min 1.8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def assign_label(T: float, low_max: float, high_min: float) -> int:
    if T <= low_max:
        return 1
    if T >= high_min:
        return 0
    return -1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/processed/summary.csv")
    p.add_argument("--low-max", type=float, default=0.9)
    p.add_argument("--high-min", type=float, default=1.8)
    p.add_argument("--out-dir", default="ml_data")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path("figures/report").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)
    df["label"] = df["T"].apply(lambda T: assign_label(float(T), args.low_max, args.high_min))

    feature_candidates = [
        "E_density_mean",
        "M_abs_mean",
        "chi_abs",
        "chi_vec",
        "Cv_per_spin",
        "helical_order_mean",
        "helical_susc_like",
        "acceptance_rate",
        "q_peak_final",
    ]
    features = [c for c in feature_candidates if c in df.columns]
    if len(features) < 2:
        raise SystemExit("Not enough feature columns found in summary file.")

    train = df[df["label"] >= 0].copy()
    test = df.copy()

    X_train = train[features].to_numpy(float)
    y_train = train["label"].to_numpy(int)
    X_test = test[features].to_numpy(float)
    T_test = test["T"].to_numpy(float)

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import accuracy_score, confusion_matrix
    except ImportError as e:
        raise SystemExit("scikit-learn required for feature baseline.") from e

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    clf.fit(X_train, y_train)

    pred_train = clf.predict(X_train)
    print("Feature baseline train accuracy:", accuracy_score(y_train, pred_train))
    print("Confusion matrix:")
    print(confusion_matrix(y_train, pred_train))

    P = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]

    out_df = test.copy()
    out_df["P_helical_feature_baseline"] = P
    out_df.to_csv(out / "feature_baseline_predictions.csv", index=False)

    grouped = (
        pd.DataFrame({"T": T_test, "P": P})
        .groupby("T")
        .agg(P_mean=("P", "mean"), P_std=("P", "std"), count=("P", "size"))
        .reset_index()
        .sort_values("T")
    )
    grouped["P_std"] = grouped["P_std"].fillna(0.0)
    grouped["P_stderr"] = grouped["P_std"] / np.sqrt(grouped["count"])
    grouped.to_csv(out / "feature_baseline_P_by_T.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.errorbar(grouped["T"], grouped["P_mean"],
                    yerr=grouped["P_stderr"], marker="o", capsize=3)
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_xlabel("Temperature T")
        ax.set_ylabel(r"$P_{\mathrm{helical}}$ feature baseline")
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig("figures/report/feature_baseline_P_helical_vs_T.pdf")
        fig.savefig("figures/report/feature_baseline_P_helical_vs_T.png", dpi=200)
        plt.close(fig)

    except Exception as e:
        print("Plotting failed:", e)


if __name__ == "__main__":
    main()
