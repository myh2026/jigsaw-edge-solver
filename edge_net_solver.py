# -*- coding: utf-8 -*-
"""边缘匹配网络求解器：严格"只用学习卷学习"的神经拼图解法。

思路：拼图复原的核心是"这两块该不该相邻"。用一个小 1D-CNN 学习
边缘条带的匹配分数（正样本=学习卷原图的真实相邻对，负样本=同图
不相邻/跨图对），推理时替换经典 L1 边缘代价，束搜索组装不变。

- 训练数据：study_100/answers 的 100 张原图（80 训练 / 20 验证，图级划分）
- 覆盖网格 3/4/5（边缘条带经 conv1d+池化，支持任意块宽）
- 用法：
    python edge_net_solver.py --train            # 训练并保存 checkpoints/edge_net.pt
    python evaluate_model.py --solver model --model edge_net_solver.py:EdgeNetSolver \
        --split exam --rows 4 --limit 100
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from loader import split_pieces
from degrade import degrade_piece
from solve_demo import edge_costs as l1_edge_costs

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "edge_net.pt"
K_COLS = 5  # 每边取边缘条带列数


class EdgeNet(nn.Module):
    """输入 [B, 3*K_COLS, L]（A 侧 K_COLS 列 + B 侧 K_COLS 列，沿边缘方向 L 像素）-> 匹配 logit。"""

    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(12 * K_COLS, 32, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Linear(128, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))

    def forward(self, x):
        f = self.body(x)  # [B,64,L]
        pooled = torch.cat([f.mean(dim=2), f.max(dim=2).values], dim=1)  # mean+max：保留峰值失配
        return self.head(pooled).squeeze(1)  # [B]


def _strip(arr, side):
    """取块 arr [H,W,3] 的边缘条带：side='r' 右侧 K 列（含边列），'l'/'b'/'t' 同理。"""
    if side == "r":
        s = arr[:, -K_COLS:, :].transpose(1, 0, 2)  # [K,W->L,3]
    elif side == "l":
        s = arr[:, :K_COLS, :].transpose(1, 0, 2)
    elif side == "b":
        s = arr[-K_COLS:, :, :]  # [K,H->L,3]
    else:
        s = arr[:K_COLS, :, :]
    # [L, K, 3] -> [3*K, L] 通道拼平（边列在最前，靠内侧在后）
    return s.transpose(1, 0, 2).reshape(s.shape[1], -1).T.astype(np.float32)


def _invariant_strip(a):
    """不变性特征：条带对比归一化（消独立亮度/对比度偏移）
    + 沿边缘方向一阶差分（弱化 gamma 类单调非线性）。"""
    a = a.astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-5)
    d = np.diff(a, axis=1)
    d = np.concatenate([d, np.zeros((d.shape[0], 1), dtype=np.float32)], axis=1)
    return np.concatenate([a, d], axis=0)  # [6K, L]（原始归一化 + 一阶差分）


def pair_input(piece_a, piece_b, direction):
    """A 在 B 左（direction='h'）或 A 在 B 上（'v'）的不变性输入。"""
    if direction == "h":
        a, b = _strip(piece_a, "r"), _strip(piece_b, "l")
    else:
        a, b = _strip(piece_a, "b"), _strip(piece_b, "t")
    return np.concatenate([_invariant_strip(a), _invariant_strip(b)], axis=0)  # [12K, ~L]


def norm(x):
    return x  # 不变性特征在 pair_input 内完成归一化；此处保持接口恒等


def image_pieces(img, rows, cols, rng=None, max_sev=0.0):
    """原图切网格块并裁掉最外 1px（与推理 split_pieces(trim=1) 同宽）。

    rng/max_sev>0 时每块独立随机退化（severity~U(0,max_sev)，含 0=干净），
    教网络对独立块级退化不变——模拟碎片独立获取的现实设置。
    """
    w, h = img.size
    pw, ph = w // cols, h // rows
    out = []
    for r in range(rows):
        for c in range(cols):
            cell = img.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph))
            arr = np.asarray(cell)[1:-1, 1:-1, :]
            if rng is not None and max_sev > 0:
                arr = degrade_piece(arr, rng, rng.uniform(0.0, max_sev))
            out.append(arr)
    return out


def build_pairs(paths, seed, augment=0.0):
    """从原图生成正/负边缘对。正=真实相邻；负=同图不相邻 + 跨图。"""
    rng = random.Random(seed)
    pos, neg = [], []
    all_imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        all_imgs.append(img)
        aug = dict(rng=np.random.RandomState(seed + 997), max_sev=augment) if augment > 0 else {}
        for g in (3, 4, 5):
            pieces = image_pieces(img, g, g, **aug)
            for r in range(g):
                for c in range(g):
                    i = r * g + c
                    if c + 1 < g:
                        pos.append(pair_input(pieces[i], pieces[i + 1], "h"))
                    if r + 1 < g:
                        pos.append(pair_input(pieces[i], pieces[i + g], "v"))
            # 难负样本：同行隔块 / 同列隔块（外观相近但不应相邻）
            p3 = image_pieces(img, 3, 3, **aug)
            for r in range(3):
                for c in range(3):
                    i = r * 3 + c
                    if c + 2 < 3:
                        neg.append(pair_input(p3[i], p3[i + 2], "h"))
                    if r + 2 < 3:
                        neg.append(pair_input(p3[i], p3[i + 6], "v"))
                    if c == 2 and r < 2:  # 行末-下行行首绕接（经典易错）
                        neg.append(pair_input(p3[i], p3[(r + 1) * 3], "h"))
            # 一般负样本：随机非相邻对
            for _ in range(12):
                i, j = rng.sample(range(9), 2)
                neg.append(pair_input(p3[i], p3[j], rng.choice(("h", "v"))))
    # 跨图负样本
    for _ in range(len(neg) // 2):
        ia, ib = rng.sample(range(len(all_imgs)), 2)
        ga = rng.choice((3, 4))
        pa = image_pieces(all_imgs[ia], ga, ga)
        pb = image_pieces(all_imgs[ib], ga, ga)
        i, j = rng.randrange(ga * ga), rng.randrange(ga * ga)
        neg.append(pair_input(pa[i], pb[j], rng.choice(("h", "v"))))
    return pos, neg


def train():
    paths = sorted((ROOT / "dataset/study_100/answers").glob("*.jpg"))
    order = list(range(len(paths)))
    random.Random(7).shuffle(order)
    val_n = max(1, len(paths) // 5)
    val_paths = [paths[i] for i in order[:val_n]]
    train_paths = [paths[i] for i in order[val_n:]]

    augment = 1.5 if args.robust else 0.0
    pos, neg = build_pairs(train_paths, seed=11, augment=augment)
    vpos, vneg = build_pairs(val_paths, seed=12, augment=augment)
    print(f"训练对: 正 {len(pos)} 负 {len(neg)} | 验证对: 正 {len(vpos)} 负 {len(vneg)}")

    def batch(pairs, labels, bs=256):
        # 按边缘长度分桶（不同网格块宽不同），桶内同长可 stack
        by_len = {}
        for i, p in enumerate(pairs):
            by_len.setdefault(p.shape[1], []).append(i)
        for L, idx in by_len.items():
            random.Random(3).shuffle(idx)
            for k in range(0, len(idx) - bs + 1, bs):
                sel = idx[k:k + bs]
                yield torch.stack([torch.from_numpy(norm(pairs[i])) for i in sel]), torch.tensor([labels[i] for i in sel], dtype=torch.float32)

    model = EdgeNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X = pos + neg
    Y = [1.0] * len(pos) + [0.0] * len(neg)
    vX = vpos + vneg
    vY = [1.0] * len(vpos) + [0.0] * len(vneg)
    for epoch in range(1, 25):
        model.train()
        tot = n = 0.0
        for xb, yb in batch(X, Y):
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(yb); n += len(yb)
        model.eval()
        correct = cnt = 0
        with torch.no_grad():
            for xb, yb in batch(vX, vY):
                pred = (model(xb) > 0).float()
                correct += (pred == yb).float().sum().item(); cnt += len(yb)
        print(f"epoch {epoch}  loss {tot/n:.4f}  验证(未见原图) acc {correct/cnt:.4f}", flush=True)
    CKPT.parent.mkdir(exist_ok=True)
    out_path = CKPT.parent / ("edge_net_robust.pt" if args.robust else "edge_net.pt")
    torch.save(model.state_dict(), out_path)
    print("saved", out_path)


# ---------------------------------------------------------------- 求解器
class EdgeNetSolver:
    """evaluate_model 的 --solver model 接口：predict(image, meta) -> 解格->输入块。"""

    def __init__(self, rows=None, cols=None, beam=5, ckpt=None):
        self.beam = beam
        self.net = EdgeNet()
        state = torch.load(ckpt or CKPT, map_location="cpu")
        self.net.load_state_dict(state)
        self.net.eval()

    @staticmethod
    def _zscore(m):
        finite = m[m < 1e8]
        mu, sd = finite.mean(), finite.std() + 1e-8
        return (m - mu) / sd

    @torch.no_grad()
    def _costs(self, pieces):
        """混合代价：alpha * z(经典L1) + (1-alpha) * z(学习网络)，alpha 在学习卷校准。"""
        import json
        alpha = 0.5
        alpha_path = CKPT.parent / "edge_net_alpha.json"
        if alpha_path.exists():
            cfg = json.loads(alpha_path.read_text(encoding="utf-8"))
            grid = int(getattr(self, "_current_rows", 0))
            alpha = cfg.get(str(grid), cfg.get("default", 0.5))
        n = len(pieces)
        l1_right, l1_down = l1_edge_costs(pieces)
        net_right, net_down = self._net_costs(pieces)
        right = self._mix(l1_right, net_right, alpha, n)
        down = self._mix(l1_down, net_down, alpha, n)
        return right, down

    @staticmethod
    def _mix(l1, net, alpha, n):
        zl, zn = EdgeNetSolver._zscore(l1), EdgeNetSolver._zscore(net)
        mixed = alpha * zl + (1.0 - alpha) * zn
        np.fill_diagonal(mixed, 1e9)
        return mixed

    @torch.no_grad()
    def _net_costs(self, pieces):
        n = len(pieces)
        right = np.zeros((n, n), dtype=np.float64)
        down = np.zeros((n, n), dtype=np.float64)
        pairs_h, pairs_v = [], []
        for i in range(n):
            for j in range(n):
                if i == j:
                    right[i, j] = down[i, j] = 1e9
                    continue
                pairs_h.append((i, j, pair_input(pieces[i], pieces[j], "h")))
                pairs_v.append((i, j, pair_input(pieces[i], pieces[j], "v")))

        def score_matrix(pairs, out):
            bs = 256
            for k in range(0, len(pairs), bs):
                chunk = pairs[k:k + bs]
                x = torch.stack([torch.from_numpy(norm(p[2])) for p in chunk])
                logits = self.net(x).numpy()
                for (i, j, _), lg in zip(chunk, logits):
                    out[i, j] = -float(lg)  # 代价 = 负匹配分

        score_matrix(pairs_h, right)
        score_matrix(pairs_v, down)
        return right, down

    def _solve(self, pieces, rows, cols):
        n = rows * cols
        right, down = self._costs(pieces)
        best_grid, best_cost = None, float("inf")
        for start in range(n):
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
                cand.sort(key=lambda t: t[2])
                beams = cand[: self.beam]
                if not beams:
                    break
            if beams and beams[0][2] < best_cost:
                best_cost = beams[0][2]
                best_grid = beams[0][1]
        return best_grid

    def predict(self, image, meta):
        rows, cols = meta["rows"], meta["cols"]
        self._current_rows = rows
        pieces = [np.asarray(p) for p in split_pieces(image, rows, cols, trim_border=1)]
        return self._solve(pieces, rows, cols)



def beam_solve(pieces, rows, cols, right, down, beam=5):
    """束搜索（接受自定义代价矩阵；与 solve_demo.solve 同构）。"""
    n = rows * cols
    best_grid, best_cost = None, float("inf")
    for start in range(n):
        beams = [([start], [start], 0.0)]
        for cell in range(1, n):
            r, c = divmod(cell, cols)
            cand = []
            for placed, grid, cost in beams:
                used = set(placed)
                for q in range(n):
                    if q in used:
                        continue
                    add = 0.0
                    if c > 0:
                        add += right[grid[cell - 1], q]
                    if r > 0:
                        add += down[grid[cell - cols], q]
                    cand.append((placed + [q], grid + [q], cost + add))
            cand.sort(key=lambda t: t[2])
            beams = cand[:beam]
            if not beams:
                break
        if beams and beams[0][2] < best_cost:
            best_cost = beams[0][2]
            best_grid = beams[0][1]
    return best_grid


def calibrate():
    """网格级校准（只用学习卷）：每个有学习样本的网格独立选 α；
    未见过的网格回退保守 α=1.0（L1 主导）。写入 edge_net_alpha.json。"""
    """学习卷上校准混合权重 alpha（只用学习卷，不碰考试卷）。"""
    from loader import JigsawDataset, inverse_permutation
    from metrics import exact_match
    import json

    solver = EdgeNetSolver()
    by_grid = {}
    ds = JigsawDataset("dataset", split="study")
    for k in range(len(ds)):
        img, perm, meta = ds[k]
        rows, cols = meta["rows"], meta["cols"]
        by_grid.setdefault(int(rows), []).append((img, rows, cols, inverse_permutation(meta["permutation"])))
    out = {"default": 1.0}
    for g, items in sorted(by_grid.items()):
        results = {}
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            exact = 0
            for img, rows, cols, target in items:
                pieces = [np.asarray(p) for p in split_pieces(img, rows, cols, trim_border=1)]
                n = rows * cols
                l1r, l1d = l1_edge_costs(pieces)
                nr, nd = solver._net_costs(pieces)
                grid = beam_solve(pieces, rows, cols,
                                  solver._mix(l1r, nr, alpha, n),
                                  solver._mix(l1d, nd, alpha, n), beam=5)
                exact += exact_match(grid, target)
            results[alpha] = exact / len(items)
        best_alpha = max(results, key=results.get)
        out[str(g)] = best_alpha
        print("grid %dx%d: " % (g, g) + " ".join(
            "a=%.2f->%.3f" % (a, r) for a, r in results.items()) + "  best %.2f" % best_alpha, flush=True)
    (CKPT.parent / "edge_net_alpha.json").write_text(json.dumps(out), encoding="utf-8")
    print("grid-level alphas:", out)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--robust", action="store_true", help="退化增强训练（保存 edge_net_robust.pt）")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    if args.train:
        train()
    if args.calibrate:
        calibrate()
    else:
        print("用 --train 训练；求解经 evaluate_model.py --solver model 调用")
