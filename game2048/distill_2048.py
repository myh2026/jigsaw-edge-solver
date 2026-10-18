# -*- coding: utf-8 -*-
"""2048 × GraphDojo 蒸馏范式：教师慢思考 -> 蒸馏快规则 -> 图接管/教师兜底。

对应 GraphDojo 的训练期架构（L: 教师兜底+归纳挂载；G: 图确定性执行）：

  教师（Teacher）  = 位板 expectimax（深 2-3，~几十 ms/步，"LLM 级"慢思考）
  蒸馏（Distill）  = 教师对局轨迹 -> 决策树规则图（学习卷）
  图接管（Graph）  = 决策树直接决策（~几十 µs/步，零教师调用）
  兜底（Fallback） = 树置信度 < 阈值时回教师（GraphDojo 的路由→教师兜底）
  学习卷           = 教师打的 3 局（状态-动作对）
  考试卷           = 全新对局（蒸馏策略/混合策略上评测）

运行：python distill_2048.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 位板引擎（教师）
ROW_L = [0] * 65536
ROW_R = [0] * 65536
for r in range(65536):
    cells = [(r >> 12) & 15, (r >> 8) & 15, (r >> 4) & 15, r & 15]
    src = [v for v in cells if v]

    def merge_left(seq):
        out, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == seq[i + 1]:
                out.append(seq[i] + 1)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    out_l = merge_left(src)
    # 右移 = 从右往左配对合并（2048 规则：靠移动方向一侧的块先合并），
    # 再左补零对齐
    out_r = list(reversed(merge_left(list(reversed(src)))))
    out_l = out_l + [0] * (4 - len(out_l))
    out_r = [0] * (4 - len(out_r)) + out_r
    ROW_L[r] = (out_l[0] << 12) | (out_l[1] << 8) | (out_l[2] << 4) | out_l[3]
    ROW_R[r] = (out_r[0] << 12) | (out_r[1] << 8) | (out_r[2] << 4) | out_r[3]


def transpose(lines):
    out = [0, 0, 0, 0]
    for i in range(4):
        v = 0
        for j in range(4):
            v |= ((lines[j] >> ((3 - i) * 4)) & 15) << ((3 - j) * 4)
        out[i] = v
    return out


def move_lines(lines, d):
    """d: 0=L,1=U,2=R,3=D（对行优先棋盘）。"""
    a, b, c, dd = lines
    if d == 0:
        return [ROW_L[a], ROW_L[b], ROW_L[c], ROW_L[dd]]
    if d == 2:
        return [ROW_R[a], ROW_R[b], ROW_R[c], ROW_R[dd]]
    t = transpose(lines)
    rr = ROW_L if d == 1 else ROW_R
    t2 = [rr[t[0]], rr[t[1]], rr[t[2]], rr[t[3]]]
    return transpose(t2)


def count_empty(lines):
    e = 0
    for r in lines:
        for j in range(4):
            if not ((r >> ((3 - j) * 4)) & 15):
                e += 1
    return e


def empty_cells(lines):
    out = []
    for i in range(4):
        for j in range(4):
            if not ((lines[i] >> ((3 - j) * 4)) & 15):
                out.append((i, j))
    return out


def set_cell(lines, r, c, v):
    out = list(lines)
    out[r] = (out[r] & ~(15 << ((3 - c) * 4))) | ((v & 15) << ((3 - c) * 4))
    return out


def _row_eval(row, acc):
    cells = [(row >> 12) & 15, (row >> 8) & 15, (row >> 4) & 15, row & 15]
    empty, merges, total, mono = acc
    ml = mr = 0.0
    for j in range(3):
        a, b = cells[j], cells[j + 1]
        if a > b:
            ml += a ** 4 - b ** 4
        else:
            mr += b ** 4 - a ** 4
    mono += min(ml, mr)
    for j in range(4):
        if not cells[j]:
            empty += 1
        else:
            total += cells[j] ** 3.5
        if j < 3 and cells[j] and cells[j] == cells[j + 1]:
            merges += cells[j] ** 4 * 2
    return [empty, merges, total, mono]


def evaluate(lines):
    acc = [0, 0, 0, 0]
    for r in lines:
        acc = _row_eval(r, acc)
    for r in transpose(lines):
        acc = _row_eval(r, acc)
    empty, merges, total, mono = acc
    return 200000.0 + 270.0 * np.log(empty + 1) + 700.0 * merges - 47.0 * mono - 11.0 * total


def expectimax(lines, depth, player):
    if depth == 0:
        return evaluate(lines)
    if player:
        best, any_ = -1e300, False
        for d in range(4):
            n = move_lines(lines, d)
            if n == lines:
                continue
            any_ = True
            v = expectimax(n, depth - 1, False)
            if v > best:
                best = v
        return best if any_ else evaluate(lines) - 1e6
    es = empty_cells(lines)
    if not es:
        return evaluate(lines)
    s = 0.0
    for (r, c) in es:
        s += 0.9 * expectimax(set_cell(lines, r, c, 1), depth - 1, True)
        s += 0.1 * expectimax(set_cell(lines, r, c, 2), depth - 1, True)
    return s / len(es)


def teacher_move(lines, depth=None):
    if depth is None:
        e = count_empty(lines)
        depth = 3 if e > 4 else 4
    best, bv = -1, -1e300
    for d in range(4):
        n = move_lines(lines, d)
        if n == lines:
            continue
        v = expectimax(n, depth - 1, False)
        if v > bv:
            bv, best = v, d
    return best


# ---------------------------------------------------------------- 对局模拟
def new_game(rng):
    lines = [0, 0, 0, 0]
    for _ in range(2):
        lines = _spawn(lines, rng)
    return lines


def _spawn(lines, rng):
    es = empty_cells(lines)
    r, c = es[rng.randrange(len(es))]
    return set_cell(lines, r, c, 1 if rng.random() < 0.9 else 2)


def max_tile(lines):
    m = 0
    for r in lines:
        for j in range(4):
            v = (r >> ((3 - j) * 4)) & 15
            m = max(m, v)
    return 1 << m if m else 0


def features(lines):
    cells = []
    for r in lines:
        for j in range(4):
            cells.append((r >> ((3 - j) * 4)) & 15)
    f = list(cells)
    f.append(sum(1 for v in cells if not v))            # 空格数
    f.append(max(cells))                                 # 最大幂
    # 四方向可合并对数
    for d in range(4):
        n = move_lines(lines, d)
        f.append(1.0 if n != lines else 0.0)
    # 最大块是否在角落
    m = max(cells)
    f.append(1.0 if cells[0] == m or cells[3] == m or cells[12] == m or cells[15] == m else 0.0)
    # 行/列单调性（4 个方向累计）
    for axis in range(2):
        asc = desc = 0.0
        for i in range(4):
            seq = [cells[i * 4 + j] for j in range(4)] if axis == 0 else [cells[j * 4 + i] for j in range(4)]
            for a, b in zip(seq, seq[1:]):
                if a <= b: asc += b - a
                else: desc += a - b
        f.append(min(asc, desc))
    return f


DIRS = [0, 1, 2, 3]  # L, U, R, D


def legal_moves(lines):
    return [d for d in DIRS if move_lines(lines, d) != lines]


def play_game(policy, rng, cap=2000, record=False):
    lines = new_game(rng)
    moves, score_states = 0, []
    teacher_calls = 0
    tree_decisions = 0
    t_teacher_ms = 0.0
    t_tree_ms = 0.0
    while moves < cap:
        legal = legal_moves(lines)
        if not legal:
            break
        d, used_teacher, tt, ttree = policy(lines, legal)
        if used_teacher:
            teacher_calls += 1
            t_teacher_ms += tt
        else:
            tree_decisions += 1
            t_tree_ms += ttree
        if record:
            score_states.append((features(lines), d))
        lines = _spawn(move_lines(lines, d), rng)
        moves += 1
    return {
        "moves": moves,
        "max_tile": max_tile(lines),
        "teacher_calls": teacher_calls,
        "tree_decisions": tree_decisions,
        "teacher_ms_total": t_teacher_ms,
        "tree_ms_total": t_tree_ms,
        "trace": score_states,
    }


# ---------------------------------------------------------------- 蒸馏（决策树 = 规则图）
def main():
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = random.Random(20261018)
    out_dir = Path(__file__).parent
    print("== 学习卷：教师（expectimax）打 3 局，录制状态-动作对 ==")
    traces = []
    study_summary = []
    for g in range(3):
        t0 = time.time()
        rec = play_game(lambda l, legal: (teacher_move(l), True, 0.0, 0.0), rng, cap=1100, record=True)
        rec.pop("trace", None)
        # 重新录制含时延
        study_summary.append(rec)
        print("  局{}: {} 步, 最大 {}, 用时 {:.0f}s".format(
            g + 1, rec["moves"], rec["max_tile"], time.time() - t0))
    # 录制轨迹（重打一遍太贵——直接在第一遍就录）：
    # 上面的 record=True 已在 play_game 内部收集，但 pop 掉了 trace；重写：
    # （为省时间，直接用一次带录制的短局 + 上面三局不带录制的成绩）
    # 正确做法：study 阶段 record=True 且保留 trace：
    # 这里重新组织——已收集在 study_summary 中的每局其实丢了 trace，
    # 因此补一次带 trace 的对局生成训练数据。

    # 简化：用 3 局教师对局（带 trace）
    traces = []
    study_summary = []
    for g in range(10):
        t0 = time.time()

        def policy(l, legal, _store=None):
            t1 = time.perf_counter()
            d = teacher_move(l)
            dt = (time.perf_counter() - t1) * 1000
            _store.append(dt)
            return d, True, dt, 0.0

        store = []
        lines = new_game(rng)
        n_moves = 0
        while n_moves < 1100:
            legal = legal_moves(lines)
            if not legal:
                break
            d, _, dt, _ = policy(lines, legal, store)
            traces.append((features(lines), d))
            lines = _spawn(move_lines(lines, d), rng)
            n_moves += 1
        study_summary.append({
            "moves": n_moves,
            "max_tile": max_tile(lines),
            "teacher_ms_avg": sum(store) / max(1, len(store)),
        })
        print("  局{}: {} 步, 最大 {}, 教师均 {:.1f}ms/步".format(
            g + 1, n_moves, max_tile(lines), study_summary[-1]["teacher_ms_avg"]))

    X = np.array([t[0] for t in traces], dtype=np.float32)
    y = np.array([t[1] for t in traces])
    print("学习卷样本: {}".format(len(X)))

    # 80/20 划分（状态级随机划分；对局内状态相关性高，作为一致性检验足够）
    idx = rng.sample(range(len(X)), len(X))
    cut = int(len(X) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    tree = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.15, max_leaf_nodes=63, random_state=7)
    tree.fit(X[tr], y[tr])
    agree = float((tree.predict(X[te]) == y[te]).mean())
    print("蒸馏规则图（梯度提升 300 轮）：留出一致性 = {:.1%}".format(agree))

    def tree_decide(lines, legal, conf_th=0.6):
        t1 = time.perf_counter()
        proba = tree.predict_proba(np.array([features(lines)], dtype=np.float32))[0]
        classes = tree.classes_
        order = np.argsort(-proba)
        d = None
        conf = 0.0
        for oi in order:
            if int(classes[oi]) in legal:
                d = int(classes[oi])
                conf = float(proba[oi])
                break
        dt = (time.perf_counter() - t1) * 1000
        if d is None or (d is not None and conf < conf_th):
            return None, conf, dt
        return d, conf, dt

    # ---- 考试卷 A：纯蒸馏（图全接管） ----
    print("\n== 考试卷 A：纯蒸馏策略（规则图全接管，零教师） ==")
    pure_results = []
    for g in range(3):
        def pure_policy(l, legal, _s=[None]):
            t1 = time.perf_counter()
            proba = tree.predict_proba(np.array([features(l)], dtype=np.float32))[0]
            classes = tree.classes_
            d = None
            for oi in np.argsort(-proba):
                if int(classes[oi]) in legal:
                    d = int(classes[oi]); break
            dt = (time.perf_counter() - t1) * 1000
            if d is None:
                d = legal[0]
            return d, False, 0.0, dt
        r = play_game(pure_policy, rng, cap=1500)
        pure_results.append(r)
        print("  局{}: {} 步, 最大方块 {}, 树均 {:.3f}ms/步".format(
            g + 1, r["moves"], r["max_tile"], r["tree_ms_total"] / max(1, r["tree_decisions"])))

    # ---- 考试卷 B：混合（图接管 + 教师兜底）——GraphDojo 主循环形态 ----
    print("\n== 考试卷 B：混合策略（置信度<0.6 -> 教师兜底） ==")
    hybrid_results = []
    for g in range(3):
        def hybrid_policy(l, legal):
            d, conf, dt = tree_decide(l, legal)
            if d is not None:
                return d, False, 0.0, dt
            t1 = time.perf_counter()
            td = teacher_move(l)
            dt2 = (time.perf_counter() - t1) * 1000
            return td, True, dt2, 0.0
        r = play_game(hybrid_policy, rng, cap=1500)
        takeover = r["tree_decisions"] / max(1, r["moves"])
        hybrid_results.append(r)
        print("  局{}: {} 步, 最大方块 {}, 图接管率 {:.0%}, 教师调用 {}".format(
            g + 1, r["moves"], r["max_tile"], takeover, r["teacher_calls"]))

    def fmt(runs, key):
        return " / ".join(str(x[key]) for x in runs)

    report = {
        "study": study_summary,
        "samples": len(X),
        "holdout_agreement": agree,
        "pure": [{k: v for k, v in r.items() if k != "trace"} for r in pure_results],
        "hybrid": [{k: v for k, v in r.items() if k != "trace"} for r in hybrid_results],
    }
    (out_dir / "distill_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    teacher_avg = np.mean([s["teacher_ms_avg"] for s in study_summary])
    tree_avg = np.mean([r["tree_ms_total"] / max(1, r["tree_decisions"]) for r in pure_results])
    print("\n== 蒸馏账本 ==")
    print("  教师均 {:.1f} ms/步  vs  蒸馏规则图均 {:.3f} ms/步（约 {:.0f} 倍）".format(
        teacher_avg, tree_avg, teacher_avg / max(tree_avg, 1e-9)))
    print("  纯蒸馏最大方块: {}".format(fmt(pure_results, "max_tile")))
    print("  混合策略最大方块: {}".format(fmt(hybrid_results, "max_tile")))
    print("saved distill_results.json")


if __name__ == "__main__":
    main()
