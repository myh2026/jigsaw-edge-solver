# -*- coding: utf-8 -*-
"""2048 × LLM 教师（DeepSeek）：让真 LLM 给决策，图从 LLM 的决策里学。

流程（GraphDojo 原味循环）：
  1. 采集一批盘面状态（来自教师深搜对局，覆盖中后期）；
  2. 批量询问 DeepSeek：每个盘面走哪个方向（L/U/R/D），temperature=0；
  3. 评估 LLM 决策质量：与深搜最优决策的一致率；
  4. 以 LLM 决策为标签训练决策图（决策树），留出一致率；
  5. 图上场打 3 局（非法决策时深搜兜底），看 LLM 教出来的图能打多远。

成本：批量 8 盘/次，约 30 次调用。

运行：DEEPSEEK_API_KEY=... python llm_teacher.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

import distill_2048 as eng
from graph_decision_selftrain import decision_features, play, export_tree, count_leaves, validate

ROOT = Path(__file__).parent
BATCH = 8
N_STATES = 240
SEARCH_DEPTH = 3

DIR_NAMES = ["L", "U", "R", "D"]        # 与 eng 的 0..3 对应（0=L,1=U,2=R,3=D）
NAME_TO_DIR = {n: i for i, n in enumerate(DIR_NAMES)}


def board_text(lines):
    rows = []
    for r in lines:
        vals = [str((r >> ((3 - j) * 4)) & 15) for j in range(4)]
        # 打印真实数值（2^log）
        vals = [str(1 << int(v)) if int(v) else "0" for v in vals]
        rows.append(" ".join(vals))
    return " / ".join(rows)


def collect_states(n, seed_base=8800):
    rng = random.Random(seed_base)
    out = []
    while len(out) < n:
        lines = eng.new_game(rng)
        for _ in range(1500):
            legal = eng.legal_moves(lines)
            if not legal:
                break
            if rng.random() < 0.45:  # 采样中后期状态
                out.append((decision_features(lines), lines, legal))
                if len(out) >= n:
                    break
            d = eng.teacher_move(lines, depth=SEARCH_DEPTH)
            lines = eng._spawn(eng.move_lines(lines, d), rng)
    return out


def ask_llm(client, states, model="deepseek-chat"):
    """一次问 BATCH 个盘面，返回方向名列表（长度与输入一致）。"""
    lines_txt = []
    for idx, (_f, lines, legal) in enumerate(states):
        legal_names = [DIR_NAMES[d] for d in legal]
        lines_txt.append("{}. board: {} | legal: {}".format(
            idx, board_text(lines), ",".join(legal_names)))
    prompt = (
        "你在玩 2048。棋盘按行给出，0 表示空格，数字是方块值。\n"
        "规则：每次选择一个方向，所有方块向该方向滑动，相同的两块合并。\n"
        "请为每个棋盘选择一个方向（只能从该棋盘的 legal 列表里选）。\n"
        "输出严格 JSON：{\"moves\": [\"L\", \"U\", ...]}，顺序与输入一致。\n\n"
        + "\n".join(lines_txt)
    )
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=60,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            moves = data.get("moves")
            if isinstance(moves, list) and len(moves) == len(states):
                return [str(m).strip().upper()[:1] for m in moves], resp
            prompt += "\n注意：moves 数量必须等于棋盘数量。"
        except Exception as exc:
            print("  LLM 调用失败（{}），重试...".format(exc))
            time.sleep(2)
    return None, None


def main():
    try:
        from openai import OpenAI
    except ImportError:
        print("需要 openai 包：pip install openai")
        return 2
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("缺少环境变量 DEEPSEEK_API_KEY")
        return 2

    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    print("== 采集 {} 个盘面状态（教师深 3 对局中采样） ==".format(N_STATES))
    states = collect_states(N_STATES)
    print("  完成")

    print("== 询问 DeepSeek（每批 {} 个盘面） ==".format(BATCH))
    llm_dir = []          # 每个状态的 LLM 方向（None=失败）
    tokens_in = tokens_out = calls = 0
    for start in range(0, len(states), BATCH):
        chunk = states[start:start + BATCH]
        names, resp = ask_llm(client, chunk)
        calls += 1
        if resp is not None:
            u = getattr(resp, "usage", None)
            tokens_in += getattr(u, "prompt_tokens", 0) or 0
            tokens_out += getattr(u, "completion_tokens", 0) or 0
        for k in range(len(chunk)):
            name = names[k] if names else "?"
            legal = chunk[k][2]
            d = NAME_TO_DIR.get(name)
            llm_dir.append(d if d in legal else None)  # 非法/未知记为 None
        if calls % 5 == 0:
            print("  已询问 {}/{} 批".format(calls, (len(states) + BATCH - 1) // BATCH))

    ok = [i for i, d in enumerate(llm_dir) if d is not None]
    print("  LLM 有效决策 {}/{}（{} 调用, tokens 入 {} / 出 {}）".format(
        len(ok), len(states), calls, tokens_in, tokens_out))

    # ---- LLM 决策质量：与深搜最优比较 ----
    opt_agree = 0
    for i in ok:
        lines = states[i][1]
        legal = states[i][2]
        best = eng.teacher_move(lines, depth=SEARCH_DEPTH)
        if llm_dir[i] == best:
            opt_agree += 1
    print("  LLM 与深搜最优决策一致率：{:.1%}".format(opt_agree / max(1, len(ok))))

    # ---- 用 LLM 决策训练决策图 ----
    samples = [(states[i][0], llm_dir[i]) for i in ok]
    X = np.array([s[0] for s in samples], dtype=np.float32)
    y = np.array([s[1] for s in samples])
    idx = np.random.RandomState(11).permutation(len(X))
    cut = int(len(idx) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=8, random_state=7)
    tree.fit(X[tr], y[tr])
    hold = float((tree.predict(X[te]) == y[te]).mean())
    print("== 决策图（由 LLM 教）== 叶 {} | 留出与 LLM 决策一致率 {:.1%}".format(
        count_leaves(tree), hold))

    # ---- 图上场（深搜仅兜底非法决策） ----
    print("== 图（LLM 教）实战 3 局 ==")
    outs = []
    for seed in (9001, 9002, 9003):
        r = play(tree, random.Random(seed))
        outs.append(r)
        print("  局{}: 最大 {:4d} | {} 步 | 图决策 {} 兜底 {}".format(
            seed, r["max"], r["moves"], r["graph"], r["fallback"]))

    # ---- 对照：LLM 直接决策比例（同一批状态上的"LLM 会怎么打"） ----
    result = {
        "states": len(states),
        "llm_valid": len(ok),
        "llm_vs_search_agree": opt_agree / max(1, len(ok)),
        "tree_holdout_agree": hold,
        "leaves": count_leaves(tree),
        "games": [{"max": r["max"], "moves": r["moves"]} for r in outs],
        "calls": calls,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
    (ROOT / "llm_teacher_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=lambda o: int(o)),
        encoding="utf-8")
    (ROOT / "final_llm_tree.json").write_text(
        json.dumps(export_tree(tree), indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# 2048 × LLM 教师（DeepSeek）", "",
          "| 指标 | 值 |", "|:---|:---|",
          "| 询问盘面 | {} |".format(len(states)),
          "| LLM 有效决策 | {} |".format(len(ok)),
          "| LLM 与深搜最优一致率 | {:.1%} |".format(opt_agree / max(1, len(ok))),
          "| 图（LLM 教）留出一致率 | {:.1%} |".format(hold),
          "| 决策叶 | {} |".format(count_leaves(tree)),
          "| 图上场最大方块 | {} |".format(" / ".join(str(r["max"]) for r in outs)),
          "| API 调用 / tokens | {} / {} 入 {} 出 |".format(calls, tokens_in, tokens_out),
          "", "结论：LLM 的 2048 决策与深搜一致率决定了它能教给图上什么；",
          "图把 LLM 的决策压缩成可执行规则后，实战表现见上表。"]
    (ROOT / "llm_teacher_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n== LLM 教师账本 ==")
    print("  LLM vs 深搜一致 {:.1%} | 图留出 {:.1%} | 图实战 {}".format(
        opt_agree / max(1, len(ok)), hold, [r["max"] for r in outs]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
