# -*- coding: utf-8 -*-
"""2048 图自训练（自对弈进化版）：让图自己变强。

图的表示（GraphDojo 语义）：
  特征节点    = 空格数、单调性、平滑度、合并潜力、角落、蛇形对齐（6 个）
  权重边      = 特征 -> 动作的加权连接（可学习参数，即"图被训练的部分"）
  动作节点    = 左/上/右/下 四个方向（选加权分数最高的动作，图内执行）

训练循环（自对弈 + 演化选择，无需教师）：
  1. 每代：以当前权重为中心采样一组候选图（扰动）；
  2. 每个候选图在固定种子的自战局上评估（max 方块 + 分数）；
  3. 只保留"比历史最佳更强"的图（= GraphDojo 的挂载/回滚纪律）；
  4. 反复迭代：图从自己的对局里变强，曲线单调上升（只取 best-so-far）。

运行：python graph_selftrain_es.py
输出：graph_selftrain_es_results.md（学习曲线）
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

import distill_2048 as eng

ROOT = Path(__file__).parent
GENS = 24
POP = 10          # 每代候选图数量
GAMES = 2         # 每个候选的评估局数（固定种子，降方差）
MAX_MOVES = 2000
SIGMA0 = 0.35
SIGMA_DECAY = 0.88


# ---------------------------------------------------------------- 图的特征
def features(lines):
    """6 维特征（越大越好）。"""
    cells = []
    for r in lines:
        for j in range(4):
            cells.append((r >> ((3 - j) * 4)) & 15)
    grid = [cells[i * 4:(i + 1) * 4] for i in range(4)]
    empty = cells.count(0)
    total = sum(v ** 3.5 for v in cells)
    # 单调性（行/列取两个方向差值的较小者之和，越大越好 -> 取负）
    mono_diff = 0.0
    for r in range(4):
        ml = mr = 0.0
        for c in range(3):
            a, b = grid[r][c], grid[r][c + 1]
            if a > b:
                ml += a - b
            else:
                mr += b - a
        mono_diff += min(ml, mr)
    for c in range(4):
        ml = mr = 0.0
        for r in range(3):
            a, b = grid[r][c], grid[r + 1][c]
            if a > b:
                ml += a - b
            else:
                mr += b - a
        mono_diff += min(ml, mr)
    # 平滑度
    smooth = 0.0
    for r in range(4):
        for c in range(3):
            if grid[r][c] and grid[r][c + 1]:
                smooth += abs(grid[r][c] - grid[r][c + 1])
    for c in range(4):
        for r in range(3):
            if grid[r][c] and grid[r + 1][c]:
                smooth += abs(grid[r][c] - grid[r + 1][c])
    # 合并潜力（当前棋盘上下左右可合并的对数）
    merges = 0.0
    for r in range(4):
        for c in range(3):
            if grid[r][c] and grid[r][c] == grid[r][c + 1]:
                merges += 1
    for c in range(4):
        for r in range(3):
            if grid[r][c] and grid[r][c] == grid[r + 1][c]:
                merges += 1
    # 角落：最大块是否在角落
    mx = max(cells)
    corner = 1.0 if (cells[0] == mx or cells[3] == mx or cells[12] == mx or cells[15] == mx) else 0.0
    # 蛇形对齐：与固定蛇形权重点积（取 4 个方向最大）
    SNAKES = [
        [[15, 14, 13, 12], [8, 9, 10, 11], [7, 6, 5, 4], [0, 1, 2, 3]],
        [[12, 13, 14, 15], [11, 10, 9, 8], [4, 5, 6, 7], [3, 2, 1, 0]],
        [[3, 2, 1, 0], [4, 5, 6, 7], [11, 10, 9, 8], [12, 13, 14, 15]],
        [[0, 1, 2, 3], [7, 6, 5, 4], [8, 9, 10, 11], [15, 14, 13, 12]],
    ]
    snake = max(
        sum(SNAKES[s][r][c] * grid[r][c] for r in range(4) for c in range(4))
        for s in range(4)
    )
    return np.array([
        float(empty),
        -mono_diff,
        -smooth,
        merges,
        corner * float(mx),
        snake / 60.0,
    ])


# 初始权重（"教师"式先验：大块角落 + 蛇形 + 留空）
W0 = np.array([2.7, -0.6, -0.35, 4.0, 1.4, 3.2])


def policy_move(lines, w):
    """图内执行：对每个合法方向的 afterstate 打分，取最高。"""
    best, bv = -1, -1e18
    for d in range(4):
        n = eng.move_lines(lines, d)
        if n == lines:
            continue
        v = float(np.dot(w, features(n)))
        if v > bv:
            bv, best = v, d
    return best


def play(w, rng, max_moves=MAX_MOVES):
    lines = eng.new_game(rng)
    moves = 0
    score = 0
    while moves < max_moves:
        d = policy_move(lines, w)
        if d < 0:
            break
        lines = eng._spawn(eng.move_lines(lines, d), rng)
        moves += 1
    return {"max": eng.max_tile(lines), "moves": moves}


def evaluate(w, seeds=(101, 202)):
    outs = [play(w, random.Random(s)) for s in seeds]
    return float(np.mean([o["max"] for o in outs])), outs


# ---------------------------------------------------------------- 训练
def main():
    rng = np.random.RandomState(7)
    w_best = W0.copy()
    base_fit, base_outs = evaluate(w_best)
    best_fit = base_fit
    history = [{"gen": 0, "fit": base_fit, "best": max(o["max"] for o in base_outs), "adopted": True}]
    print("== 图自训练（自对弈进化）：初始图 平均最大方块 {:.0f} ==".format(base_fit))

    sigma = SIGMA0
    for gen in range(1, GENS + 1):
        candidates = [w_best + sigma * rng.randn(len(w_best)) for _ in range(POP)]
        fits = []
        for cand in candidates:
            f, _ = evaluate(cand, seeds=(101 + gen, 202 + gen))
            fits.append(f)
        fits = np.array(fits)
        order = np.argsort(-fits)
        adopted = False
        # 只接受"严格更强"的图（挂载纪律：否则保留当前）
        for idx in order:
            if fits[idx] > best_fit:
                w_best = candidates[idx]
                best_fit = fits[idx]
                adopted = True
                break
        sigma *= SIGMA_DECAY
        history.append({"gen": gen, "fit": float(best_fit), "best": int(best_fit),
                        "adopted": adopted})
        print("  gen {:02d}: 本代最佳 {:.0f} | 历史最佳 {:.0f} | sigma {:.3f} | {}".format(
            gen, max(fits), best_fit, sigma, "升级" if adopted else "保留"))

    final_fit, final_outs = evaluate(w_best, seeds=(9001, 9002, 9003))
    print("\n== 终评（全新 3 局，最终图独立执行）==")
    for i, o in enumerate(final_outs):
        print("  局{}: 最大方块 {} | {} 步".format(i + 1, o["max"], o["moves"]))
    print("\n权重: {}".format(np.round(w_best, 2).tolist()))

    report = {
        "w0": W0.tolist(),
        "w_best": w_best.tolist(),
        "history": history,
        "initial_avg": base_fit,
        "trained_avg": final_fit,
        "final_games": final_outs,
    }
    (ROOT / "graph_selftrain_es_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# 2048 图自训练（自对弈进化）", "",
          "图 = 6 个特征节点 -> 权重边 -> 4 个动作节点；训练 = 自对弈 + 只留更强的版本。", "",
          "| 代 | 历史最佳（eval 平均最大方块） | 升级 |", "|---:|:---:|:---:|"]
    for h in history:
        md.append("| {} | {:.0f} | {} |".format(h["gen"], h["fit"], "✓" if h["adopted"] else ""))
    md.append("")
    md.append("初始图 **{:.0f}** -> 训练后 **{:.0f}**（+{:.0f}）；终评新局：{}".format(
        base_fit, final_fit, final_fit - base_fit,
        " / ".join(str(o["max"]) for o in final_outs)))
    (ROOT / "graph_selftrain_es_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("saved graph_selftrain_es_results.md")


if __name__ == "__main__":
    main()
