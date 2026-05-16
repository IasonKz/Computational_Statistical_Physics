#!/usr/bin/env python3
"""
Plot a color-coded 2D spin slice from a saved CSV configuration.

No arrows are used. Hue encodes atan(Sy,Sx), brightness encodes Sz.

Usage:
    python3 scripts/plot_colored_slice.py --config results/configs/config_....csv --plane xz
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def spin_rgb(sx, sy, sz):
    phi = np.arctan2(sy, sx)
    hue = (phi/(2*np.pi)) % 1.0
    sat = np.full_like(hue, 0.95)
    val = 0.55 + 0.45*(sz+1)/2
    hsv = np.stack([hue, sat, val], axis=-1)
    return hsv_to_rgb(hsv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--plane", default="xz", choices=["xy","xz","yz"])
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.config)
    L = int(df[["x","y","z"]].max().max()) + 1
    fixed = L//2 if args.index < 0 else args.index

    if args.plane == "xz":
        sl = df[df["y"] == fixed].copy(); a,b = "x","z"
    elif args.plane == "xy":
        sl = df[df["z"] == fixed].copy(); a,b = "x","y"
    else:
        sl = df[df["x"] == fixed].copy(); a,b = "y","z"

    img = np.zeros((L,L,3))
    colors = spin_rgb(sl["Sx"].to_numpy(), sl["Sy"].to_numpy(), sl["Sz"].to_numpy())
    for (_, row), c in zip(sl.iterrows(), colors):
        aa = int(row[a]); bb = int(row[b])
        img[L-1-bb, aa, :] = c

    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{args.plane} colored slice, index={fixed}")
    fig.tight_layout()

    if args.out is None:
        Path("figures/report").mkdir(parents=True, exist_ok=True)
        out = f"figures/report/colored_slice_{Path(args.config).stem}_{args.plane}.png"
    else:
        out = args.out
    fig.savefig(out, dpi=250)
    print("Saved", out)

if __name__ == "__main__":
    main()
