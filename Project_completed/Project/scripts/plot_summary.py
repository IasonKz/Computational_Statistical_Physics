#!/usr/bin/env python3
"""
Quick plotting script after collect_results.py.
It creates figures similar to the Ex01 style: magnetization, susceptibility, energy, heat capacity, helical order.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

summary = Path("results/processed/summary_by_L_T_mean.csv")
if not summary.exists():
    summary = Path("results/processed/summary.csv")
if not summary.exists():
    raise SystemExit("Run scripts/collect_results.py first.")

df = pd.read_csv(summary)
fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)

for L, g in df.groupby("L"):
    g = g.sort_values("T")
    label = f"L={int(L)}"
    plt.figure(1)
    plt.plot(g["T"], g["M_abs_mean"], marker="o", label=label)
    plt.figure(2)
    plt.plot(g["T"], g["chi_abs"], marker="o", label=label)
    plt.figure(3)
    plt.plot(g["T"], g["E_density_mean"], marker="o", label=label)
    plt.figure(4)
    plt.plot(g["T"], g["Cv_per_spin"], marker="o", label=label)
    plt.figure(5)
    plt.plot(g["T"], g["helical_order_mean"], marker="o", label=label)

plots = [
    (1, "Magnetization density", r"$T$", r"$\langle |M| \rangle/N$", "magnetization_vs_T.pdf"),
    (2, "Magnetic susceptibility", r"$T$", r"$\chi_{abs}$", "susceptibility_vs_T.pdf"),
    (3, "Energy density", r"$T$", r"$\langle E \rangle/N$", "energy_vs_T.pdf"),
    (4, "Heat capacity", r"$T$", r"$C_V/N$", "heat_capacity_vs_T.pdf"),
    (5, "Helical order from structure factor", r"$T$", r"$\max_{q\neq 0} S(q)$", "helical_order_vs_T.pdf"),
]

for num, title, xlabel, ylabel, fname in plots:
    plt.figure(num)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    path = fig_dir / fname
    plt.savefig(path)
    print(f"Saved {path}")

plt.show()
