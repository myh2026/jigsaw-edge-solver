# -*- coding: utf-8 -*-
"""拼图题训练模板（PyTorch）。

任务：输入打乱后的 480×480 网格，输出“解格 -> 输入块”的排列。
     logits[b, s, q] = 解格 s 放输入块 q 的分数
     target[b, s]    = inverse_permutation(meta["permutation"])[s]

这个脚本是一个可跑通的最小 baseline（共享碎片 CNN + Transformer）。
把 PieceJigsawNet 换成你自己的模型，保持输入/输出约定即可。

注意：study_100 只有 2×2 和 3×3；4×4/5×5 需要你自己的训练集，
或从 exam 中按 source_id 划分训练/测试（不要按题随机划分，同一原图的裁剪会泄漏）。

用法：
    python train_jigsaw.py --grid 3 --epochs 20
    python train_jigsaw.py --grid 3 --epochs 5 --limit 40
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
except Exception:
    print("需要 PyTorch：pip install torch")
    sys.exit(1)

from loader import JigsawDataset
from metrics import adjacency_accuracy

ROOT = Path(__file__).resolve().parent


class ToTensor:
    """PIL -> [3,H,W] float，归一化到 [-1,1]。"""

    def __call__(self, img):
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr.transpose(2, 0, 1))


class PieceJigsawNet(nn.Module):
    def __init__(self, rows, cols, dim=128, layers=2, heads=4):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.n = rows * cols
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.pos = nn.Parameter(torch.zeros(1, self.n, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                           dim_feedforward=dim * 2,
                                           dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(dim, self.n)

    def forward(self, x):
        b, _, h, w = x.shape
        ph, pw = h // self.rows, w // self.cols
        # [B,3,H,W] -> [B*N,3,ph,pw]
        pieces = x.unfold(2, ph, ph).unfold(3, pw, pw)
        pieces = pieces.permute(0, 2, 3, 1, 4, 5).reshape(b * self.n, 3, ph, pw)
        feat = self.encoder(pieces).flatten(1).reshape(b, self.n, -1)
        feat = feat + self.pos
        feat = self.transformer(feat)
        return self.head(feat)  # [B,N,N]


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    tgts = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return imgs, tgts, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--grid", type=int, default=3, help="3/4/5；study 只有 2/3")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条（0=全部）")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    g = args.grid
    ds = JigsawDataset(args.dataset, split="study", target="solution",
                       rows=g, cols=g, limit=args.limit or None,
                       transform=ToTensor())
    n = len(ds)
    if n < 2:
        raise SystemExit(f"study 里 {g}x{g} 只有 {n} 条；"
                         "请用你自己的训练集，或换 --grid 3。")
    idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    val_n = max(1, int(n * args.val_frac))
    val_ds = Subset(ds, idx[:val_n])
    train_ds = Subset(ds, idx[val_n:])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, collate_fn=collate)

    model = PieceJigsawNet(g, g).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"{g}x{g}  study={n}  train={len(train_ds)}  val={len(val_ds)}  "
          f"device={args.device}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_n, correct = 0.0, 0, 0
        for imgs, tgts, _ in train_loader:
            imgs, tgts = imgs.to(args.device), tgts.to(args.device)
            logits = model(imgs)                     # [B,N,N]
            loss = F.cross_entropy(logits.reshape(-1, g * g), tgts.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            total_n += imgs.size(0)
            correct += (logits.argmax(-1) == tgts).float().mean().item() * imgs.size(0)
        train_loss = total_loss / max(1, total_n)
        train_acc = correct / max(1, total_n)

        model.eval()
        val_loss, val_n, val_correct = 0.0, 0, 0
        exact, adjs = 0.0, []
        with torch.no_grad():
            for imgs, tgts, metas in val_loader:
                imgs, tgts = imgs.to(args.device), tgts.to(args.device)
                logits = model(imgs)
                loss = F.cross_entropy(logits.reshape(-1, g * g), tgts.reshape(-1))
                val_loss += loss.item() * imgs.size(0)
                val_n += imgs.size(0)
                pred = logits.argmax(-1)
                val_correct += (pred == tgts).float().mean().item() * imgs.size(0)
                exact += (pred == tgts).all(dim=1).float().mean().item() * imgs.size(0)
                for p, meta in zip(pred.cpu().tolist(), metas):
                    scramble = meta["permutation"]
                    pred_orig = [scramble[q] for q in p]
                    adjs.append(adjacency_accuracy(pred_orig, g, g))

        print(f"epoch {epoch:02d}  train loss {train_loss:.4f} acc {train_acc:.3f}  "
              f"val loss {val_loss/max(1,val_n):.4f} acc {val_correct/max(1,val_n):.3f} "
              f"exact {exact/max(1,val_n):.3f} adj {np.mean(adjs) if adjs else 0:.3f}")
        if val_correct / max(1, val_n) > best:
            best = val_correct / max(1, val_n)
            torch.save({"state_dict": model.state_dict(), "rows": g, "cols": g},
                       out / f"jigsaw_{g}x{g}.pt")
            print(f"  saved {out / f'jigsaw_{g}x{g}.pt'}")


if __name__ == "__main__":
    main()
