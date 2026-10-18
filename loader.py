# -*- coding: utf-8 -*-
"""拼图题数据集加载器（PyTorch / 纯 PIL 均可）。

用法：
    from loader import JigsawDataset, split_pieces, reconstruct, inverse_permutation

    # 默认返回“输入格 -> 原图块”的 scramble permutation
    ds = JigsawDataset("dataset", split="study")
    img, perm, meta = ds[0]

    # 训练推荐：返回“解格 -> 输入块”的 solution permutation（scramble 的逆）
    ds = JigsawDataset("dataset", split="study", target="solution",
                       rows=3, cols=3)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    HAS_TORCH = True
except Exception:
    _TorchDataset = object
    HAS_TORCH = False


def load_manifest(root):
    root = Path(root)
    with open(root / "manifest.csv", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def inverse_permutation(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


class JigsawDataset(_TorchDataset):
    """返回 (打乱后的拼图图, target, 标签 dict)。

    target="scramble"：target[i] = 输入格 i 里的原图块编号（数据集原始标签）。
    target="solution"：target[s] = 解格 s 应放哪个输入块编号（scramble 的逆，训练推荐）。

    可选 rows/cols 只取指定网格（如 4×4），limit 限制条数。
    """

    def __init__(self, root="dataset", split="study", transform=None,
                 rows=None, cols=None, limit=None, target="scramble"):
        self.root = Path(root).resolve()
        self.split = split
        self.transform = transform
        if target not in ("scramble", "solution"):
            raise ValueError("target 必须是 'scramble' 或 'solution'")
        self.target = target
        items = [r for r in load_manifest(self.root) if r["split"] == split]
        if rows is not None:
            items = [r for r in items if int(r["rows"]) == int(rows)]
        if cols is not None:
            items = [r for r in items if int(r["cols"]) == int(cols)]
        if limit:
            items = items[:limit]
        if not items:
            raise ValueError(f"split={split!r}, rows={rows}, cols={cols} 没有数据")
        self.items = items
        self.rows = items  # 兼容旧代码

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        img = Image.open(self.root / row["input"]).convert("RGB")
        meta = json.load(open(self.root / row["label"], encoding="utf-8"))
        perm = meta["permutation"]
        if self.target == "solution":
            perm = inverse_permutation(perm)
        if self.transform is not None:
            img = self.transform(img)
        if HAS_TORCH:
            perm = torch.tensor(perm, dtype=torch.long)
        return img, perm, meta


def split_pieces(image, rows, cols, trim_border=1):
    """把打乱输入图切成碎片（行优先）。

    trim_border=1 会去掉每块四周 1px 的拼缝，得到与答案完全一致的碎片。
    """
    w, h = image.size
    pw, ph = w // cols, h // rows
    out = []
    for r in range(rows):
        for c in range(cols):
            box = (c * pw + trim_border, r * ph + trim_border,
                   (c + 1) * pw - trim_border, (r + 1) * ph - trim_border)
            out.append(image.crop(box))
    return out


def reconstruct(image, permutation, rows, cols, trim_border=1):
    """按 permutation 把打乱图复原成答案图。

    这里按数据集原始语义使用：permutation[cell] = 该输入格里的原图块编号。
    如果你的模型输出“解格 -> 输入块”，先 inverse_permutation() 再传入。
    """
    w, h = image.size
    pw, ph = w // cols, h // rows
    pieces = split_pieces(image, rows, cols, trim_border)
    canvas = Image.new("RGB",
                       ((pw - 2 * trim_border) * cols,
                        (ph - 2 * trim_border) * rows),
                       (20, 20, 20))
    for idx, orig in enumerate(permutation):
        orow, ocol = divmod(orig, cols)
        canvas.paste(pieces[idx], (ocol * (pw - 2 * trim_border),
                                   orow * (ph - 2 * trim_border)))
    return canvas


def solution_permutation(meta):
    """数据集标签 -> 训练目标：解格 s 应放哪个输入块。"""
    return inverse_permutation(meta["permutation"])


if __name__ == "__main__":
    ds = JigsawDataset("dataset", split="study")
    img, perm, meta = ds[0]
    print("样本数:", len(ds))
    print("图像:", img.size, "网格:", meta["rows"], "x", meta["cols"])
    print("scramble:", perm)
    print("solution:", solution_permutation(meta))
    print("类型:", type(perm))
