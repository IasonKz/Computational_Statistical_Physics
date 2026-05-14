#!/usr/bin/env python3
"""
Compare ML transition estimate with physics observables.

Reads:
    ml_data/mlp_P_helical_by_T.csv
    results/processed/summary_by_L_T_mean.csv or summary.csv

Output:
    figures/report/ml_vs_physics_transition.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def crossing(T, P):
    for i in range(len(T) - 1):
        if (P[i] - 0.5) * (P[i + 1] - 0.5) < 0:
            return T[i] + (0.5 - P[i]) * (T[i + 1] - T[i]) / (P[i + 1] - P[i])
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ml", default="ml_data/mlp_P_helical_by_T.csv")
    p.add_argument("--physics", default="results/processed/summary_by_L_T_mean.csv")
    p.add_argument("--L", type=int, default=None)
    args = p.parse_args()

    Path("figures/report").mkdir(parents=True, exist_ok=True)

    ml = pd.read_csv(args.ml).sort_values("T")
    phys_path = Path(args.physics)
    if not phys_path.exists():
        phys_path = Path("results/processed/summary.csv")
    phys = pd.read_csv(phys_path)

    if args.L is not None and "L" in phys.columns:
        phys = phys[phys["L"].astype(int) == args.L]

    # If multiple L values remain, average over them for a simple overlay.
    phys = phys.groupby("T", as_index=False).mean(numeric_only=True).sort_values("T")

    T_ml = ml["T"].to_numpy(float)
    P = ml["P_helical_mean"].to_numpy(float)
    Tc_ml = crossing(T_ml, P)

    Tc_cv = None
    if "Cv_per_spin" in phys.columns and len(phys):
        Tc_cv = float(phys.loc[phys["Cv_per_spin"].idxmax(), "T"])

    Tc_hs = None
    if "helical_susc_like" in phys.columns and len(phys):
        Tc_hs = float(phys.loc[phys["helical_susc_like"].idxmax(), "T"])

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7.0, 4.6))
    ax1.errorbar(
        ml["T"], ml["P_helical_mean"],
        yerr=ml.get("P_helical_stderr", pd.Series(np.zeros(len(ml)))),
        marker="o",
        capsize=3,
        label=r"$P_{\mathrm{helical}}$ ML",
    )
    ax1.axhline(0.5, linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Temperature T")
    ax1.set_ylabel(r"$P_{\mathrm{helical}}$")
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    if "Cv_per_spin" in phys.columns:
        ax2.plot(phys["T"], phys["Cv_per_spin"], marker="s", alpha=0.65, label=r"$C_V/N$")
        ax2.set_ylabel(r"Physics observable")

    if Tc_ml is not None:
        ax1.axvline(Tc_ml, linestyle=":", linewidth=1.4, label=fr"$T_c^{{ML}}\approx {Tc_ml:.3f}$")
    if Tc_cv is not None:
        ax1.axvline(Tc_cv, linestyle="-.", linewidth=1.2, label=fr"$C_V$ peak $\approx {Tc_cv:.3f}$")
    if Tc_hs is not None:
        ax1.axvline(Tc_hs, linestyle=(0, (4, 2)), linewidth=1.2, label=fr"helical fluct. peak $\approx {Tc_hs:.3f}$")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8)

    fig.tight_layout()
    fig.savefig("figures/report/ml_vs_physics_transition.pdf")
    fig.savefig("figures/report/ml_vs_physics_transition.png", dpi=200)
    plt.close(fig)

    print("Tc_ML:", Tc_ml)
    print("Tc_Cv_peak:", Tc_cv)
    print("Tc_helical_fluct_peak:", Tc_hs)


if __name__ == "__main__":
    main()
