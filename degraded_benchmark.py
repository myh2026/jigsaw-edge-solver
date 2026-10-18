# -*- coding: utf-8 -*-
"""退化基准：独立块级退化下，学习度量 vs 经典 L1 的优越性实验。

协议：退化族与参数对所有方法相同；训练增强只用学习卷原图；
每个严重度下 α 在**退化学习卷**上重校准（考试卷零调参）。
考试卷在内存中退化为副本，不改盘上数据。

用法：python degraded_benchmark.py
输出：results/degraded_results.md（表）+ 控制台打印
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from degrade import degrade_scrambled
from edge_net_solver import EdgeNetSolver, beam_solve, CKPT
from loader import JigsawDataset, split_pieces, inverse_permutation
from metrics import exact_match, piece_accuracy
from solve_demo import edge_costs as l1_edge_costs

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" if (ROOT / "dataset").exists() else ROOT.parent / "下载/25级人工智能实验班学生资料 (3)/25级人工智能实验班学生资料/拼图题数据集/dataset"


def solver_pieces(img, rows, cols):
    return [np.asarray(p) for p in split_pieces(img, rows, cols, trim_border=1)]


def run_grid(solver_net, split, rows, limit, severity, seed_base, methods=("l1", "net", "hybrid"), alpha=0.0):
    ds = JigsawDataset(str(DATASET), split=split, rows=rows, cols=rows, limit=limit)
    scores = {m: [0.0, 0] for m in methods}  # piece_sum, exact
    n_total = len(ds)
    for k in range(n_total):
        img, perm, meta = ds[k]
        if severity > 0:
            img = degrade_scrambled(img, rows, rows, severity, seed=seed_base + k)
        cols_ = rows
        pieces = solver_pieces(img, rows, cols_)
        n = rows * rows
        target = inverse_permutation(meta["permutation"])
        l1r, l1d = l1_edge_costs(pieces)
        nr, nd = (None, None)
        if "net" in methods or "hybrid" in methods:
            nr, nd = solver_net._net_costs(pieces)
        grids = {}
        if "l1" in methods:
            grids["l1"] = beam_solve(pieces, rows, cols_, l1r, l1d, beam=4)
        if "net" in methods:
            grids["net"] = beam_solve(pieces, rows, cols_, nr, nd, beam=5)
        if "hybrid" in methods:
            right = solver_net._mix(l1r, nr, alpha, n)
            down = solver_net._mix(l1d, nd, alpha, n)
            grids["hybrid"] = beam_solve(pieces, rows, cols_, right, down, beam=5)
        for m, grid in grids.items():
            scores[m][0] += piece_accuracy(grid, target)
            scores[m][1] += exact_match(grid, target)
    return {m: (s / n_total, e / n_total) for m, (s, e) in scores.items()}, n_total


def calibrate_alpha_degraded(solver_net, severity, seed_base=7000):
    """在退化学习卷上重校准 α（每严重度独立；考试卷不参与）。"""
    ds = JigsawDataset(str(DATASET), split="study")
    cache = []
    for k in range(len(ds)):
        img, perm, meta = ds[k]
        img_d = degrade_scrambled(img, meta["rows"], meta["cols"], severity, seed=seed_base + k)
        pieces = solver_pieces(img_d, meta["rows"], meta["cols"])
        cache.append((pieces, meta["rows"], meta["cols"], inverse_permutation(meta["permutation"])))
    best = (None, -1.0)
    for alpha in (0.0, 0.5, 1.0):
        exact = 0
        for pieces, r, c, target in cache:
            n = r * c
            l1r, l1d = l1_edge_costs(pieces)
            nr, nd = solver_net._net_costs(pieces)
            grid = beam_solve(pieces, r, c,
                              solver_net._mix(l1r, nr, alpha, n),
                              solver_net._mix(l1d, nd, alpha, n), beam=5)
            exact += exact_match(grid, target)
        acc = exact / len(cache)
        print("  sev=%s alpha=%.1f -> study exact %.3f" % (severity, alpha, acc), flush=True)
        if acc > best[1]:
            best = (alpha, acc)
    return best


def main():
    net = EdgeNetSolver(ckpt=str(CKPT.parent / "edge_net_robust.pt"))
    lines = ["# 退化基准：独立块级退化下的优越性实验", ""]
    lines.append("每块独立退化（亮度/γ/噪声/块级 JPEG）；退化学习卷重校准 α；考试卷 100 题/格，同题对照。")
    lines.append("")
    lines.append("| 严重度 | 网格 | L1 基线 piece/exact | EdgeNet-robust piece/exact | 混合(校准 α) piece/exact |")
    lines.append("|:---:|:---:|:---|:---|:---|")
    print(lines[0])
    for severity in (1, 2):
        alpha, study_acc = calibrate_alpha_degraded(net, severity)
        print("sev=%d best alpha=%.1f (study exact %.3f)" % (severity, alpha, study_acc))
        for g in (3, 5):
            res, n = run_grid(net, "exam", g, 100, severity, seed_base=8000 + severity * 1000 + g * 100,
                              alpha=alpha)
            row = "| {} | {}×{} | {:.3f} / {:.3f} | {:.3f} / {:.3f} | {:.3f} / {:.3f}（α={}） |".format(
                "轻" if severity == 1 else "重", g, g,
                res["l1"][0], res["l1"][1], res["net"][0], res["net"][1],
                res["hybrid"][0], res["hybrid"][1], alpha)
            lines.append(row)
            print(row)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "degraded_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved", out / "degraded_results.md")


if __name__ == "__main__":
    main()
