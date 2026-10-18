# -*- coding: utf-8 -*-
"""严格"只用学习卷学习"的自生成训练：

数据源 = study_100/answers 的 100 张原图（学习卷图片，零考试泄漏）。
在线随机重排生成无限样本（复刻 build_dataset 的构造：切块 -> perm 重排
-> 每块 1px 自适应描边），训练各网格的 PieceJigsawNet。

原图级 80/20 划分（验证集只用没训过的图，防图泄漏）。

用法：
    python train_study_reshuffle.py --grid 3 --epochs 10
    python train_study_reshuffle.py --grid 4 --epochs 8
    python train_study_reshuffle.py --grid 5 --epochs 6
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    print("需要 PyTorch：pip install torch")
    sys.exit(1)

from train_jigsaw import PieceJigsawNet, ToTensor


def adaptive_border_scramble(img, rows, cols, perm):
    """复刻 build_dataset.build_scrambled：重排 + 1px 自适应描边。"""
    w, h = img.size
    pw, ph = w // cols, h // rows
    canvas = Image.new("RGB", (pw * cols, ph * rows), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    for idx, orig in enumerate(perm):
        r, c = divmod(idx, cols)
        x, y = c * pw, r * ph
        piece = img.crop(((orig % cols) * pw, (orig // cols) * ph,
                          (orig % cols + 1) * pw, (orig // cols + 1) * ph))
        canvas.paste(piece, (x, y))
        mean = np.asarray(piece.convert("L"), dtype=np.float32).mean()
        border = (240, 240, 240) if mean < 110 else (18, 18, 18)
        d.rectangle([x, y, x + pw - 1, y + ph - 1], outline=border, width=1)
    return canvas


class ReshuffleDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, rows, cols, length, seed, train_size=240):
        self.images = [Image.open(p).convert("RGB") for p in image_paths]
        self.train_size = train_size  # 训练分辨率（CPU 提速；评测仍用 480 原图）
        self.rows, self.cols, self.length = rows, cols, length
        self.rng = random.Random(seed)
        self.tf = ToTensor()

    def __len__(self):
        return self.length

    def __getitem__(self, _idx):
        img = self.rng.choice(self.images)
        n = self.rows * self.cols
        perm = list(range(n))
        self.rng.shuffle(perm)
        scrambled = adaptive_border_scramble(img, self.rows, self.cols, perm)
        if self.train_size:
            scrambled = scrambled.resize((self.train_size, self.train_size), Image.BILINEAR)
        inv = [0] * n
        for i, p in enumerate(perm):
            inv[p] = i  # target: 解格 s -> 输入块
        return self.tf(scrambled), torch.tensor(inv, dtype=torch.long), {"permutation": perm}


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    tgts = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return imgs, tgts, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, required=True, choices=[2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--samples-per-epoch", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=128)  # 须与 encoder 输出一致
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    g = args.grid
    answer_dir = Path("dataset/study_100/answers")
    paths = sorted(answer_dir.glob("*.jpg"))
    assert paths, "找不到 study answers"
    order = list(range(len(paths)))
    random.Random(args.seed).shuffle(order)
    val_n = max(1, len(paths) // 5)
    val_paths = [paths[i] for i in order[:val_n]]
    train_paths = [paths[i] for i in order[val_n:]]

    train_ds = ReshuffleDataset(train_paths, g, g, args.samples_per_epoch, args.seed)
    val_ds = ReshuffleDataset(val_paths, g, g, max(80, args.samples_per_epoch // 6), args.seed + 1)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = PieceJigsawNet(g, g, dim=args.dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"{g}x{g}  学习卷原图 train={len(train_paths)} val={len(val_paths)}  device={args.device}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    ckpt = out / f"jigsaw_{g}x{g}_reshuffle.pt"
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_n = correct = 0
        for imgs, tgts, _ in train_loader:
            imgs, tgts = imgs.to(args.device), tgts.to(args.device)
            logits = model(imgs)
            loss = F.cross_entropy(logits.reshape(-1, g * g), tgts.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            total_n += imgs.size(0)
            correct += (logits.argmax(-1) == tgts).float().mean().item() * imgs.size(0)

        model.eval()
        v_n = v_correct = v_exact = 0
        with torch.no_grad():
            for imgs, tgts, _ in val_loader:
                imgs, tgts = imgs.to(args.device), tgts.to(args.device)
                logits = model(imgs)
                pred = logits.argmax(-1)
                v_n += imgs.size(0)
                v_correct += (pred == tgts).float().mean().item() * imgs.size(0)
                v_exact += (pred == tgts).all(dim=1).float().sum().item()
        va, ve = v_correct / max(1, v_n), v_exact / max(1, v_n)
        print(f"epoch {epoch:02d}  train loss {total_loss/max(1,total_n):.4f} acc {correct/max(1,total_n):.3f}  "
              f"val(未见原图) acc {va:.3f} exact {ve:.3f}", flush=True)
        if va > best:
            best = va
            torch.save({"state_dict": model.state_dict(), "rows": g, "cols": g}, ckpt)
            print(f"  saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
