# -*- coding: utf-8 -*-
"""经典边匹配拼图求解器（不训练，直接“玩”）。

它不是神经网络，而是用相邻碎片的边缘颜色差做束搜索（beam search）。
把它当作“AI 接口”的可运行示例：你的模型只要输出 permutation，
就能替换这里的 solve()，其余数据加载、指标、复原可视化都不用改。

用法：
    python solve_demo.py --split study --index 50 --beam 4
    python solve_demo.py --split exam --index 0 --beam 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from loader import JigsawDataset, split_pieces
from metrics import adjacency_accuracy, exact_match, piece_accuracy
from loader import inverse_permutation

ROOT = Path(__file__).resolve().parent
FONT = "C:/Windows/Fonts/msyh.ttc"


def edge_costs(pieces):
    """返回 right_cost[i,j]（i 在 j 左边）和 down_cost[i,j]（i 在 j 上边）。"""
    n = len(pieces)
    right = np.zeros((n, n), dtype=np.float64)
    down = np.zeros((n, n), dtype=np.float64)
    edges_r = [p[:, -1, :].astype(np.float32) for p in pieces]
    edges_l = [p[:, 0, :].astype(np.float32) for p in pieces]
    edges_b = [p[-1, :, :].astype(np.float32) for p in pieces]
    edges_t = [p[0, :, :].astype(np.float32) for p in pieces]
    for i in range(n):
        for j in range(n):
            if i == j:
                right[i, j] = down[i, j] = 1e9
                continue
            right[i, j] = np.abs(edges_r[i] - edges_l[j]).mean()
            down[i, j] = np.abs(edges_b[i] - edges_t[j]).mean()
    return right, down


def solve(pieces, rows, cols, beam=4):
    """束搜索：枚举每个起始块，按行优先填格，保留累计边缘代价最小的 beam 个。"""
    n = rows * cols
    if len(pieces) != n:
        raise ValueError("pieces 数量与 rows*cols 不一致")
    right, down = edge_costs(pieces)
    best_grid, best_cost = None, float("inf")

    for start in range(n):
        # (已放置序列, 网格, 累计代价)
        beams = [([start], [start], 0.0)]
        for cell in range(1, n):
            r, c = divmod(cell, cols)
            cand = []
            for placed, grid, cost in beams:
                used = set(placed)
                for p in range(n):
                    if p in used:
                        continue
                    add = 0.0
                    if c > 0:
                        add += right[grid[cell - 1], p]
                    if r > 0:
                        add += down[grid[cell - cols], p]
                    cand.append((placed + [p], grid + [p], cost + add))
            cand.sort(key=lambda x: x[2])
            beams = cand[:beam]
            if not beams:
                break
        if beams and beams[0][2] < best_cost:
            best_cost = beams[0][2]
            best_grid = beams[0][1]
    return best_grid, best_cost


def assemble(pieces, grid, rows, cols):
    ph, pw = pieces[0].shape[:2]
    canvas = np.zeros((ph * rows, pw * cols, 3), dtype=np.uint8)
    for cell, orig in enumerate(grid):
        r, c = divmod(cell, cols)
        canvas[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw] = pieces[orig]
    return Image.fromarray(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--split", default="study", choices=["study", "exam"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--out", default="previews/solve_demo.png")
    args = ap.parse_args()

    ds = JigsawDataset(args.dataset, split=args.split)
    img, perm, meta = ds[args.index]
    rows, cols = meta["rows"], meta["cols"]
    pieces = [np.asarray(p) for p in split_pieces(img, rows, cols, trim_border=1)]

    grid, cost = solve(pieces, rows, cols, beam=args.beam)
    scramble = list(perm) if not hasattr(perm, "tolist") else perm.tolist()
    # 模型输出约定：解格 -> 输入块编号；目标就是 scramble 的逆
    target = inverse_permutation(scramble)
    pred_orig = [scramble[q] for q in grid]   # 解格 -> 原图块编号，用于相邻指标

    print(f"{meta['id']}  {rows}x{cols}  {rows*cols} 块")
    print("输入打乱 permutation（输入格 -> 原图块）:", scramble)
    print("目标（解格 -> 输入块）:", target)
    print("预测（解格 -> 输入块）:", grid)
    print(f"位置正确率: {piece_accuracy(grid, target):.3f}")
    print(f"整题完全正确: {exact_match(grid, target):.1f}")
    print(f"相邻正确率: {adjacency_accuracy(pred_orig, rows, cols):.3f}")
    print(f"边缘代价: {cost:.2f}")

    pred_img = assemble(pieces, grid, rows, cols)
    ans_img = Image.open(Path(args.dataset) / meta["answer"]).convert("RGB").crop(
        (1, 1, 1 + pred_img.width, 1 + pred_img.height))
    pad, header = 16, 44
    sheet = Image.new("RGB", (pred_img.width * 3 + pad * 4,
                              pred_img.height + header + pad + 26), (245, 247, 250))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT, 18)
    d.text((pad, 12), f"{meta['id']}  {rows}x{cols}  ·  束搜索解 {rows*cols} 块",
           fill=(30, 41, 59), font=font)
    sheet.paste(img, (pad, header))
    sheet.paste(pred_img, (pad * 2 + pred_img.width, header))
    sheet.paste(ans_img, (pad * 3 + pred_img.width * 2, header))
    for i, t in enumerate(["打乱输入", "求解器复原", "原图答案"]):
        d.text((pad + i * (pred_img.width + pad), header + pred_img.height + 4),
               t, fill=(71, 85, 105), font=font)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print("已保存:", args.out)


if __name__ == "__main__":
    main()
