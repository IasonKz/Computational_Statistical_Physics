#!/usr/bin/env python3
"""
Train the main 3D CNN classifier.

This is the main ML model for the project.
Requires build_dataset.py --store-cnn.

Example:
    python3 ml/train_cnn3d.py --data-dir ml_data_L16 --epochs 40 --batch-size 16
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
    raise SystemExit('PyTorch required') from e
from models import SpinCNN3D


def acc(logits, y):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return (pred == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='ml_data_L16')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data = Path(args.data_dir); (data/'models').mkdir(parents=True, exist_ok=True)
    Path('figures/report').mkdir(parents=True, exist_ok=True)
    Xtr = np.load(data/'X_train_cnn.npy').astype(np.float32)
    ytr = np.load(data/'y_train.npy').astype(np.float32)
    Xva = np.load(data/'X_val_cnn.npy').astype(np.float32)
    yva = np.load(data/'y_val.npy').astype(np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device, 'X_train', Xtr.shape)
    model = SpinCNN3D(channels=3, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=args.batch_size, shuffle=True)
    Xv = torch.from_numpy(Xva).to(device); yv = torch.from_numpy(yva).to(device)
    hist=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]; accs=[]
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(); logits=model(xb); loss=loss_fn(logits,yb); loss.backward(); opt.step()
            losses.append(loss.item()); accs.append(acc(logits.detach(), yb))
        model.eval()
        with torch.no_grad():
            vl=model(Xv); vloss=loss_fn(vl,yv).item(); vacc=acc(vl,yv)
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'train_acc':float(np.mean(accs)),'val_loss':float(vloss),'val_acc':float(vacc)}
        hist.append(row)
        print(f"epoch {epoch:03d} train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.3f} val_loss={vloss:.4f} val_acc={vacc:.3f}")
    torch.save({'model_state_dict':model.state_dict(),'dropout':args.dropout,'model_type':'SpinCNN3D'}, data/'models'/'cnn3d_spins.pt')
    pd.DataFrame(hist).to_csv(data/'cnn3d_training_history.csv', index=False)
    import matplotlib.pyplot as plt
    h=pd.DataFrame(hist)
    fig, ax = plt.subplots(figsize=(6.2,4.2)); ax.plot(h.epoch,h.train_loss,label='train'); ax.plot(h.epoch,h.val_loss,label='val'); ax.set_xlabel('Epoch'); ax.set_ylabel('BCE'); ax.legend(); fig.tight_layout(); fig.savefig(f'figures/report/cnn3d_loss_curve_{data.name}.pdf'); fig.savefig(f'figures/report/cnn3d_loss_curve_{data.name}.png', dpi=200)
    fig, ax = plt.subplots(figsize=(6.2,4.2)); ax.plot(h.epoch,h.train_acc,label='train'); ax.plot(h.epoch,h.val_acc,label='val'); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.legend(); fig.tight_layout(); fig.savefig(f'figures/report/cnn3d_accuracy_curve_{data.name}.pdf'); fig.savefig(f'figures/report/cnn3d_accuracy_curve_{data.name}.png', dpi=200)
    print('saved model', data/'models'/'cnn3d_spins.pt')

if __name__ == '__main__':
    main()
