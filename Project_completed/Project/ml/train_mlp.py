#!/usr/bin/env python3
"""
Train a fully connected raw-spin MLP.

This is the closest analogue of Ex04:
    input  = full spin configuration flattened to a vector
    output = P_helical

Example:
    python3 ml/train_mlp.py --data-dir ml_data --epochs 40 --batch-size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise SystemExit("PyTorch is required. Install/load torch first.") from e

from models import RawSpinMLP


def binary_accuracy(logits, y):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return (pred == y).float().mean().item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="ml_data")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden1", type=int, default=256)
    p.add_argument("--hidden2", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = Path(args.data_dir)
    model_dir = data / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    Path("figures/report").mkdir(parents=True, exist_ok=True)

    X_train = np.load(data / "X_train.npy").astype(np.float32)
    y_train = np.load(data / "y_train.npy").astype(np.float32)
    X_val = np.load(data / "X_val.npy").astype(np.float32)
    y_val = np.load(data / "y_val.npy").astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"input dimension: {X_train.shape[1]}")
    print(f"train samples: {len(X_train)}, val samples: {len(X_val)}")

    model = RawSpinMLP(
        input_dim=X_train.shape[1],
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_accs = []

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(binary_accuracy(logits.detach(), yb))

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = loss_fn(val_logits, yv).item()
            val_acc = binary_accuracy(val_logits, yv)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_acc": float(np.mean(train_accs)),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        history.append(row)

        print(
            f"epoch {epoch:03d} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.3f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_acc']:.3f}"
        )

    checkpoint = {
        "model_type": "RawSpinMLP",
        "model_state_dict": model.state_dict(),
        "input_dim": X_train.shape[1],
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "dropout": args.dropout,
    }
    torch.save(checkpoint, model_dir / "mlp_raw_spins.pt")
    pd.DataFrame(history).to_csv(data / "mlp_training_history.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        hist = pd.DataFrame(history)

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(hist["epoch"], hist["train_loss"], label="train")
        ax.plot(hist["epoch"], hist["val_loss"], label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary cross entropy")
        ax.legend()
        fig.tight_layout()
        fig.savefig("figures/report/mlp_loss_curve.pdf")
        fig.savefig("figures/report/mlp_loss_curve.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(hist["epoch"], hist["train_acc"], label="train")
        ax.plot(hist["epoch"], hist["val_acc"], label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig("figures/report/mlp_accuracy_curve.pdf")
        fig.savefig("figures/report/mlp_accuracy_curve.png", dpi=200)
        plt.close(fig)

    except Exception as e:
        print(f"Plotting failed: {e}")

    print(f"Saved model to {model_dir / 'mlp_raw_spins.pt'}")


if __name__ == "__main__":
    main()
