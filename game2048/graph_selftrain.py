# -*- coding: utf-8 -*-
"""2048 图自训练（GraphDojo 范式）：让图从自己的失误里长大、变强。

架构对应：
  图（Graph）   = 规则表：情境键 -> 方向（成千上万个"规则节点"，可快速执行）
  教师（Teacher）= expectimax 深 3 搜索（只在图未覆盖/需要校对时出场）
  判分（Judge） = 一局结束时的最大方块（客观标准，不依赖教师）
  误差归因      = 逐决策比较 图的选择 与 教师的选择 的启发式估值，
                  图明显更差的情境 = 坏规则节点
  隔离区        = 把坏节点候选与未覆盖情境交给教师，产出规则更新（提案）
  挂载/回滚     = 局级验证：新规则表若没有优于历史最佳 -> 回滚到最佳快照
  记忆          = 规则表本身（覆盖的情境越来越多，教师调用越来越少）

关键区别（相对"统计克隆"）：图不是拟合教师整个策略，而是
"自己的对局经验 + 教师在校准点的纠错"——图随覆盖增长变强，
失败驱动的精炼只发生在它真正踩坑的情境上。

运行：python graph_selftrain.py
输出：graph_selftrain_results.md（学习曲线）+ 最终规则表统计
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

import distill_2048 as eng

ROOT = Path(__file__).parent
EPISODES = 15
GAMES_PER_EP = 1
MAX_MOVES = 2000
TEACHER_DEPTH = 3
REFINE_TOPK = 400
MARGIN = 1200.0  # 启发式差值超过它才认为"图这一步明显更差"


# ---------------------------------------------------------------- 情境与工具
def count_tiles(lines) -> int:
    n = 0
    for r in lines:
        for j in range(4):
            if (r >> ((3 - j) * 4)) & 15:
                n += 1
    return n


def cells_of(lines):
    out = []
    for r in lines:
        for j in range(4):
            out.append((r >> ((3 - j) * 4)) & 15)
    return out


def situation(lines):
    """规则键：空格数 × 可合并方向掩码 × 最大块在角 × 最大幂。
    粗粒度是刻意的——图学的是"大原则"，不是逐状态查表。"""
    e = eng.count_empty(lines)
    base = count_tiles(lines)
    mm = 0
    for d in range(4):
        n = eng.move_lines(lines, d)
        if n != lines and count_tiles(n) < base:
            mm |= 1 << d
    cells = cells_of(lines)
    mx = max(cells)
    corner = 1 if (cells[0] == mx or cells[3] == mx or cells[12] == mx or cells[15] == mx) else 0
    return (e, mm, corner, mx)


def heu(lines) -> float:
    return eng.evaluate(lines)


# ---------------------------------------------------------------- 对局（图 + 教师兜底）
def play_episode(table, rng, teacher_depth=TEACHER_DEPTH):
    lines = eng.new_game(rng)
    moves = graph_dec = teacher_dec = 0
    bad = {}       # 情境 -> 教师建议（坏规则需精炼）
    unseen = {}    # 情境 -> 教师建议（未覆盖，需挂载）
    trace = []
    while moves < MAX_MOVES:
        legal = eng.legal_moves(lines)
        if not legal:
            break
        key = situation(lines)
        ta = eng.teacher_move(lines, depth=teacher_depth)
        act = table.get(key)
        if act is not None and act in legal:
            graph_dec += 1
            v_act = heu(eng.move_lines(lines, act))
            v_ta = heu(eng.move_lines(lines, ta))
            if v_ta > v_act + MARGIN:
                bad[key] = ta           # 误差归因：图这一步明显更差
            chosen = act
        else:
            teacher_dec += 1
            unseen[key] = ta            # 学习区：新情境直接向教师学
            chosen = ta
        trace.append((key, chosen))
        lines = eng._spawn(eng.move_lines(lines, chosen), rng)
        moves += 1
    return {
        "moves": moves,
        "max": eng.max_tile(lines),
        "graph": graph_dec,
        "teacher": teacher_dec,
        "bad": bad,
        "unseen": unseen,
        "trace": trace,
    }


# ---------------------------------------------------------------- 训练循环
def main():
    rng = random.Random(20480816)
    table: dict = {}
    best_table: dict = {}
    best_score = -1
    history = []

    print("== 图自训练：{} 集，每集 {} 局（目标：曲线持续变强 + 覆盖率上升）==".format(
        EPISODES, GAMES_PER_EP))
    for ep in range(1, EPISODES + 1):
        ep_max, ep_moves, ep_graph, ep_teacher = 0, 0, 0, 0
        all_bad, all_unseen = {}, {}
        for _ in range(GAMES_PER_EP):
            r = play_episode(table, rng)
            ep_max = max(ep_max, r["max"])
            ep_moves += r["moves"]
            ep_graph += r["graph"]
            ep_teacher += r["teacher"]
            all_bad.update(r["bad"])
            all_unseen.update(r["unseen"])

        coverage = ep_graph / max(1, ep_graph + ep_teacher)
        # 局级验证：没变强就回滚（GraphDojo 的挂载纪律）
        if ep_max >= best_score:
            best_score = ep_max
            best_table = dict(table)
            verdict = "保留"
        else:
            table = dict(best_table)
            verdict = "回滚"

        # 隔离区：精炼坏规则 + 挂载新情境（上限 REFINE_TOPK）
        changes = 0
        for key, act in list(all_bad.items())[:REFINE_TOPK]:
            if table.get(key) != act:
                table[key] = act
                changes += 1
        for key, act in list(all_unseen.items())[:REFINE_TOPK]:
            if key not in table:
                table[key] = act
                changes += 1

        history.append({
            "episode": ep, "max": ep_max, "moves": ep_moves,
            "coverage": coverage, "rules": len(table), "changes": changes,
            "teacher_calls": ep_teacher, "verdict": verdict,
        })
        print("  集{:02d}: 最大 {:4d} | 图决策 {:5d} 教师 {:4d} (覆盖 {:5.1%}) | "
              "规则 {} 条 (+{}) | {}".format(
                  ep, ep_max, ep_graph, ep_teacher, coverage, len(table), changes, verdict))

    # ---- 最终评估：纯图（未覆盖仍教师兜底），新局 ----
    final_table = dict(table)
    final = []
    for g in range(3):
        r = play_episode(final_table, rng)
        final.append(r)
        print("  终局{}: 最大 {:4d} | 覆盖 {:5.1%} | 教师 {}".format(
            g + 1, r["max"], r["graph"] / max(1, r["graph"] + r["teacher"]), r["teacher"]))

    best_curve = max(h["max"] for h in history)
    first3 = np.mean([h["max"] for h in history[:3]])
    last3 = np.mean([h["max"] for h in history[-3:]])
    report = {
        "episodes": EPISODES,
        "history": history,
        "final_games": [{k: v for k, v in r.items() if k in ("max", "moves", "graph", "teacher")}
                        for r in final],
        "first3_avg": first3,
        "last3_avg": last3,
        "best_episode_max": best_curve,
        "rules": len(final_table),
    }
    (ROOT / "graph_selftrain_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# 2048 图自训练实验（GraphDojo 范式）", "",
          "| 集 | 最大方块 | 图决策 | 教师决策 | 覆盖率 | 规则数 | 变更 | 判定 |",
          "|---:|:---:|---:|---:|---:|---:|---:|:---:|"]
    for h in history:
        md.append("| {} | {} | {} | {} | {:.1%} | {} | +{} | {} |".format(
            h["episode"], h["max"], h["moves"] - h["teacher_calls"],
            h["teacher_calls"], h["coverage"], h["rules"], h["changes"], h["verdict"]))
    md.append("")
    md.append("前 3 集平均最大方块 **{:.0f}** → 后 3 集 **{:.0f}**；最佳单集 **{}**；"
              "规则表 **{}** 条；终局评估：{}".format(
                  first3, last3, best_curve, len(final_table),
                  " / ".join(str(r["max"]) for r in final)))
    (ROOT / "graph_selftrain_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n== 自训练账本 ==")
    print("  前 3 集平均最大 {:.0f} -> 后 3 集 {:.0f}（最佳单集 {}）".format(
        first3, last3, best_curve))
    print("  最终规则表 {} 条；终局最大方块: {}".format(
        len(final_table), " / ".join(str(r["max"]) for r in final)))
    print("saved graph_selftrain_results.md / .json")


if __name__ == "__main__":
    main()
