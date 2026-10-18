# -*- coding: utf-8 -*-
"""拼图题常用评测指标。

约定：permutation[i] = 输入网格第 i 格（行优先）对应的原始碎片编号。
理想解为 [0, 1, 2, ..., N-1]。
"""

from __future__ import annotations


def _to_list(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return list(x)


def piece_accuracy(pred, target):
    """位置正确率：有多少块放回了正确位置。"""
    p, t = _to_list(pred), _to_list(target)
    if len(p) != len(t):
        raise ValueError("长度不一致")
    return sum(int(a == b) for a, b in zip(p, t)) / max(1, len(t))


def exact_match(pred, target):
    """整题完全正确率。"""
    return float(_to_list(pred) == _to_list(target))


def adjacency_accuracy(pred, rows, cols):
    """相邻正确率：拼好的结果里，左右/上下邻居有多少对是原图中真正相邻的。"""
    perm = _to_list(pred)
    n = rows * cols
    if len(perm) != n or sorted(perm) != list(range(n)):
        return 0.0
    correct = total = 0
    for r in range(rows):
        for c in range(cols):
            a = perm[r * cols + c]
            ar, ac = divmod(a, cols)
            if c + 1 < cols:
                b = perm[r * cols + c + 1]
                br, bc = divmod(b, cols)
                total += 1
                correct += int(br == ar and bc == ac + 1)
            if r + 1 < rows:
                b = perm[(r + 1) * cols + c]
                br, bc = divmod(b, cols)
                total += 1
                correct += int(br == ar + 1 and bc == ac)
    return correct / max(1, total)


def evaluate(preds, targets, rows, cols):
    """preds/targets: 等长的 permutation 列表。"""
    if not preds:
        return {"piece_acc": 0.0, "exact": 0.0, "adjacency": 0.0, "n": 0}
    n = len(preds)
    return {
        "piece_acc": sum(piece_accuracy(p, t) for p, t in zip(preds, targets)) / n,
        "exact": sum(exact_match(p, t) for p, t in zip(preds, targets)) / n,
        "adjacency": sum(adjacency_accuracy(p, rows, cols) for p in preds) / n,
        "n": n,
    }


def inverse_permutation(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(_to_list(perm)):
        inv[p] = i
    return inv


if __name__ == "__main__":
    import random
    rows = cols = 3
    n = rows * cols
    perfect = list(range(n))
    rnd = perfect.copy()
    random.shuffle(rnd)
    print("perfect:", evaluate([perfect], [perfect], rows, cols))
    print("random :", evaluate([rnd], [perfect], rows, cols))
