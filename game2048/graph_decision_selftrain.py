# -*- coding: utf-8 -*-
"""2048 决策图自训练：图自己做决策，并从自己的决策里学习。

图 = 决策树：
  内部节点 = 决策测试（如 "空格 <= 3 ?" / "向下可合并 ?" / "最大块在角 ?"）
  叶节点   = 决策（左/上/右/下）
  运行时：从根走到叶，叶子就是这一步的决策（图在决策，不是程序在算）。

自训练循环（GraphDojo 语义）：
  冷启动     空图 -> 教师兜底（深 2 搜索）播种第一批"状态->决策"样本
  挂载       归纳出初始决策图（决策树），挂载进系统
  自对弈     图做决策（仅在无合法/未覆盖时教师兜底），记录每一步
  误差归因   用一步前瞻估值找"后悔决策"（图选的动作明显不如最优）
  结构精炼   只用后悔样本 + 少量随机样本重归纳决策树（新分支=图长出新决策）
  验证/回滚  固定种子自战：新图不比历史最佳更强 -> 回滚（挂载纪律）

导出：final_tree.json（决策图结构，可直接在浏览器 bot 中遍历决策）。

运行：python graph_decision_selftrain.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

import distill_2048 as eng

ROOT = Path(__file__).parent
SEED_GAMES = 6
EPISODES = 12
TRAIN_GAMES_PER_EP = 2
MAX_MOVES = 1500
TEACHER_DEPTH = 3
REGRET_MARGIN = 200.0
CONF_TH = 0.0  # 不按置信度兜底，只在决策非法时兜底（让图尽量接管）


# ---------------------------------------------------------------- 决策特征
def decision_features(lines):
    """8 个可解释决策变量。"""
    cells = []
    for r in lines:
        for j in range(4):
            cells.append((r >> ((3 - j) * 4)) & 15)
    grid = [cells[i * 4:(i + 1) * 4] for i in range(4)]
    empty = cells.count(0)
    # 各方向当前是否可合并
    def merges_in(dir_):
        n = 0
        if dir_ in (0, 2):
            for r in range(4):
                for c in range(3):
                    if grid[r][c] and grid[r][c] == grid[r][c + 1]:
                        n += 1
        else:
            for c in range(4):
                for r in range(3):
                    if grid[r][c] and grid[r][c] == grid[r + 1][c]:
                        n += 1
        return n
    ml = merges_in(0)
    mu = merges_in(1)
    mr = merges_in(2)
    md = merges_in(3)
    mx = max(cells)
    corner = 1 if (cells[0] == mx or cells[3] == mx or cells[12] == mx or cells[15] == mx) else 0
    # 派生策略特征：蛇形对齐 / 单调性 / 平滑度 / 合并价值
    SNAKES = [
        [[15,14,13,12],[8,9,10,11],[7,6,5,4],[0,1,2,3]],
        [[12,13,14,15],[11,10,9,8],[4,5,6,7],[3,2,1,0]],
        [[3,2,1,0],[4,5,6,7],[11,10,9,8],[12,13,14,15]],
        [[0,1,2,3],[7,6,5,4],[8,9,10,11],[15,14,13,12]],
    ]
    snake = 0.0
    for S in SNAKES:
        sc = sum(S[r][c] * grid[r][c] for r in range(4) for c in range(4))
        snake = max(snake, float(sc))
    mono = 0.0
    for r in range(4):
        ml2 = mr2 = 0.0
        for c in range(3):
            a, b = grid[r][c], grid[r][c + 1]
            if a > b: ml2 += a - b
            else: mr2 += b - a
        mono += min(ml2, mr2)
    for c in range(4):
        ml2 = mr2 = 0.0
        for r in range(3):
            a, b = grid[r][c], grid[r + 1][c]
            if a > b: ml2 += a - b
            else: mr2 += b - a
        mono += min(ml2, mr2)
    smooth = 0.0
    for r in range(4):
        for c in range(3):
            if grid[r][c] and grid[r][c + 1]:
                smooth += abs(grid[r][c] - grid[r][c + 1])
    for c in range(4):
        for r in range(3):
            if grid[r][c] and grid[r + 1][c]:
                smooth += abs(grid[r][c] - grid[r + 1][c])
    merge_val = 0.0
    for r in range(4):
        for c in range(3):
            if grid[r][c] and grid[r][c] == grid[r][c + 1]:
                merge_val += grid[r][c]
    for c in range(4):
        for r in range(3):
            if grid[r][c] and grid[r][c] == grid[r + 1][c]:
                merge_val += grid[r][c]
    return ([float(empty), float(ml > 0), float(mu > 0), float(mr > 0),
             float(md > 0), float(corner), float(mx), float(sum(1 for v in cells if v))]
            + [float(v) for v in cells]
            + [snake / 60.0, -float(mono), -float(smooth), float(merge_val)])


FEATURE_NAMES = (["empties", "merge_L", "merge_U", "merge_R", "merge_D",
                  "corner_max", "max_log", "tile_count"]
                 + ["r{}c{}".format(r, c) for r in range(4) for c in range(4)]
                 + ["snake", "mono", "smooth", "merge_val"])


def q_of_move(lines, d):
    return eng.evaluate(eng.move_lines(lines, d))


def best_action(lines, legal):
    qs = {d: q_of_move(lines, d) for d in legal}
    return max(qs, key=qs.get), qs


# ---------------------------------------------------------------- 对局
def play(tree, rng, teacher_depth=TEACHER_DEPTH, collect=None, tag=""):
    lines = eng.new_game(rng)
    moves = graph_dec = fb = 0
    samples = []
    while moves < MAX_MOVES:
        legal = eng.legal_moves(lines)
        if not legal:
            break
        f = decision_features(lines)
        act = None
        if tree is not None:
            proba = tree.predict_proba(np.array([f], dtype=np.float32))[0]
            cand = int(tree.classes_[np.argmax(proba)])
            conf = float(max(proba))
            if cand in legal and conf >= CONF_TH:
                act = cand
                graph_dec += 1
        if act is None:
            act = eng.teacher_move(lines, depth=teacher_depth)
            fb += 1
        # 后悔归因（自对弈核心）：图的选择 vs 一步前瞻最优
        if collect is not None:
            if tree is None:
                collect.append((f, act))        # 冷启动：教师全部决策作种子
            best, qs = best_action(lines, legal)
            if qs[best] - qs[act] > REGRET_MARGIN:
                collect.append((f, eng.teacher_move(lines, depth=TEACHER_DEPTH)))
            elif rng.random() < 0.25:
                collect.append((f, eng.teacher_move(lines, depth=TEACHER_DEPTH)))
        lines = eng._spawn(eng.move_lines(lines, act), rng)
        moves += 1
    return {"moves": moves, "max": eng.max_tile(lines),
            "graph": graph_dec, "fallback": fb, "samples": samples}


def fit_tree(samples, max_depth=6):
    X = np.array([s[0] for s in samples], dtype=np.float32)
    y = np.array([s[1] for s in samples])
    t = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=12, random_state=7)
    t.fit(X, y)
    return t


def validate(tree, seeds=(301, 302, 303)):
    outs = [play(tree, random.Random(s), collect=None) for s in seeds]
    return float(np.mean([o["max"] for o in outs])), outs


# ---------------------------------------------------------------- 导出决策图
def export_tree(tree):
    t = tree.tree_
    class_vals = [int(c) for c in tree.classes_]

    def node(i):
        if t.children_left[i] == -1:
            return {"action": class_vals[int(np.argmax(t.value[i]))],
                    "n": int(t.n_node_samples[i])}
        feat = FEATURE_NAMES[t.feature[i]]
        return {
            "test": "{} <= {:.2f}".format(feat, t.threshold[i]),
            "feature": feat,
            "threshold": float(t.threshold[i]),
            "left": node(t.children_left[i]),
            "right": node(t.children_right[i]),
        }

    return node(0)


def count_leaves(tree):
    return tree.get_n_leaves()


# ---------------------------------------------------------------- 主流程
def collect_states(games, depth, seed_base, cap=900):
    """从教师对局采集状态-决策对（留出集/种子集共用）。"""
    rng = random.Random(seed_base)
    out = []
    for g in range(games):
        lines = eng.new_game(rng)
        for _ in range(MAX_MOVES):
            legal = eng.legal_moves(lines)
            if not legal:
                break
            out.append((decision_features(lines), eng.teacher_move(lines, depth=depth)))
            d = eng.teacher_move(lines, depth=depth)
            lines = eng._spawn(eng.move_lines(lines, d), rng)
            if len(out) >= cap:
                return out
    return out


def agreement(tree, holdout):
    X = np.array([h[0] for h in holdout], dtype=np.float32)
    y = np.array([h[1] for h in holdout])
    pred = tree.predict(X)
    return float((pred == y).mean())


def main():
    rng = random.Random(20480901)
    print("== 留出决策集（教师深 3 对局，不参与训练）==")
    holdout = collect_states(2, TEACHER_DEPTH, 7001, cap=800)
    print("  留出状态 {} 条".format(len(holdout)))

    print("== 冷启动：图为空，教师兜底播种 ==")
    seed_samples = []
    for g in range(SEED_GAMES):
        collect = []
        r = play(None, rng, collect=collect)
        seed_samples.extend(collect)
        print("  教师局{}: {} 步 最大 {}".format(g + 1, r["moves"], r["max"]))
    print("  种子样本 {}".format(len(seed_samples)))

    tree = fit_tree(seed_samples, max_depth=6)
    best_tree = tree
    best_agree = agreement(tree, holdout)
    best_fit, _ = validate(tree, seeds=(301, 302, 303))
    print("  初始决策图：叶 {} | 决策一致性 {:.1%} | 对局验证 {:.0f}".format(
        count_leaves(tree), best_agree, best_fit))

    history = []
    pool = list(seed_samples)
    for ep in range(1, EPISODES + 1):
        collect = []
        ep_graph = ep_fb = ep_max = 0
        for _ in range(TRAIN_GAMES_PER_EP):
            r = play(tree, rng, collect=collect)
            ep_graph += r["graph"]
            ep_fb += r["fallback"]
            ep_max = max(ep_max, r["max"])
        pool.extend(collect)
        pool = pool[-30000:]

        cand = fit_tree(pool, max_depth=min(14, 6 + ep // 2))
        cand_agree = agreement(cand, holdout)
        # 挂载纪律：决策一致性提升才接受（更稳定的验收信号）
        if cand_agree > best_agree:
            tree, best_tree = cand, cand
            best_agree = cand_agree
            verdict = "挂载"
        else:
            tree = best_tree
            verdict = "回滚"
        val_fit, _ = validate(best_tree, seeds=(301, 302, 303))
        coverage = ep_graph / max(1, ep_graph + ep_fb)
        history.append({"ep": ep, "train_max": ep_max, "agree": best_agree,
                        "val": val_fit, "leaves": count_leaves(best_tree),
                        "coverage": coverage, "new_samples": len(collect),
                        "verdict": verdict})
        print("  集{:02d}: 训练局最大 {:4d} | 一致性 {:.1%} | 对局验证 {:.0f} | 叶 {:3d} | "
              "覆盖 {:5.1%} | 新样本 {:4d} | {}".format(
                  ep, ep_max, best_agree, val_fit, count_leaves(best_tree),
                  coverage, len(collect), verdict))

    final_fit, final_outs = validate(best_tree, seeds=(9001, 9002, 9003))
    graph_json = export_tree(best_tree)
    (ROOT / "final_decision_graph.json").write_text(
        json.dumps(graph_json, indent=2, ensure_ascii=False), encoding="utf-8")

    def _plain(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(str(type(o)))

    (ROOT / "graph_decision_selftrain_results.json").write_text(
        json.dumps({"history": history, "final_val": final_fit,
                    "final_games": [o["max"] for o in final_outs],
                    "agreement": best_agree,
                    "leaves": count_leaves(best_tree), "tree": graph_json},
                   indent=2, ensure_ascii=False, default=_plain), encoding="utf-8")

    md = ["# 2048 决策图自训练（图在做决策）", "",
          "验收信号：留出 800 个教师状态的**决策一致性**（稳定）；对局结果仅作参考。", "",
          "| 集 | 决策一致性 | 对局验证 | 决策叶 | 图决策覆盖 | 新样本 | 判定 |",
          "|---:|:---:|:---:|:---:|:---:|---:|:---:|"]
    for h in history:
        md.append("| {} | {:.1%} | {:.0f} | {} | {:.1%} | {} | {} |".format(
            h["ep"], h["agree"], h["val"], h["leaves"], h["coverage"],
            h["new_samples"], h["verdict"]))
    md.append("")
    md.append("最终：决策一致性 **{:.1%}**，决策叶 **{}**；终评新局：{}".format(
        best_agree, count_leaves(best_tree),
        " / ".join(str(o["max"]) for o in final_outs)))
    (ROOT / "graph_decision_selftrain_results.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")

    print("\n== 账本 ==")
    print("  决策一致性 {:.1%}，决策叶 {}，终评新局：{}".format(
        best_agree, count_leaves(best_tree), [o["max"] for o in final_outs]))
    print("  决策图已导出 final_decision_graph.json")


if __name__ == "__main__":
    main()
