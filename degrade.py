# -*- coding: utf-8 -*-
"""独立块级退化：模拟碎片独立获取/扫描的现实设置。

每块独立经历：亮度偏移 + gamma 抖动 + 高斯噪声 + 块级 JPEG 重压缩。
参数由 severity（0/1/2 = 无/轻/重）控制。真相邻块的接缝像素不再连续，
绝对像素比较（L1）的判别力被破坏——这是学习度量的用武之地。

协议合规：退化只施加在学习卷原图（训练增强）与考试卷**评测副本**
（内存中，不改盘上数据）上；退化族与参数对两种方法完全相同。
"""

from __future__ import annotations

import io
import random

import numpy as _np

import numpy as np
from PIL import Image

_PRESETS = {
    0: dict(bright=0.0, gamma=0.0, sigma=0.0, q=100),
    1: dict(bright=0.08, gamma=0.05, sigma=0.012, q=45),   # 轻
    2: dict(bright=0.16, gamma=0.10, sigma=0.025, q=26),   # 重
}


def degrade_piece(arr: np.ndarray, rng, severity: float) -> np.ndarray:
    """rng: np.random.RandomState（独立块级随机参数）。"""
    """对单块 [H,W,3] uint8 独立退化；severity 可为连续值（训练增强采样用）。"""
    if severity <= 0:
        return arr
    p = _PRESETS[min(2, max(1, int(round(severity))))]
    s = severity if severity <= 1 else 1.0 + (severity - 1)  # 线性外推
    x = arr.astype(np.float32) / 255.0
    if p["bright"] > 0:
        x = x * (1.0 + rng.uniform(-p["bright"], p["bright"]) * s)
    if p["gamma"] > 0:
        g = 1.0 + rng.uniform(-p["gamma"], p["gamma"]) * s
        x = np.clip(x, 0.0, 1.0) ** g
    if p["sigma"] > 0:
        x = x + rng.normal(0.0, p["sigma"] * s, x.shape)
    out = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    q = max(15, int(round(p["q"] - (p["q"] - 15) * (s - 1)))) if s > 1 else p["q"]
    if q < 100:
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="JPEG", quality=q, subsampling=1)
        buf.seek(0)
        out = np.asarray(Image.open(buf).convert("RGB"))
    return out


def degrade_scrambled(img: Image.Image, rows: int, cols: int, severity: float,
                      seed: int) -> Image.Image:
    """对整张打乱图的每块区域独立退化（评测副本用；不动盘上数据）。"""
    if severity <= 0:
        return img
    w, h = img.size
    pw, ph = w // cols, h // rows
    arr = np.asarray(img).copy()
    rng = np.random.RandomState(seed)
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * ph, c * pw
            block = arr[y0:y0 + ph, x0:x0 + pw]
            arr[y0:y0 + ph, x0:x0 + pw] = degrade_piece(block, rng, severity)
    return Image.fromarray(arr)
