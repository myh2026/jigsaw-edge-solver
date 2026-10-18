# -*- coding: utf-8 -*-
"""在拼图题数据集上评测你的模型 / 基线。

支持三种“玩家”：
    --solver edge     经典边匹配束搜索（无需训练，3×3/4×4 可解）
    --solver random   随机排列基线
    --solver model   你自己的模型（--model my_model.py:MyModel [--checkpoint ckpt.pt]）

模型接口（二选一）：
    model.predict(image: PIL.Image, meta: dict) -> list[int]
        返回长度 N 的“解格 -> 输入块”排列；
    或 model(images: Tensor[B,3,480,480]) -> Tensor[B,N,N] logits
        第 s 行第 q 列 = 解格 s 放输入块 q 的分数。

用法：
    python evaluate_model.py --solver edge --split exam --rows 4 --limit 20
    python evaluate_model.py --solver model --model my_model.py:MyModel \
        --checkpoint checkpoints/jigsaw_4x4.pt --split exam --rows 4 --limit 100
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from loader import JigsawDataset, inverse_permutation, split_pieces
from metrics import adjacency_accuracy, exact_match, piece_accuracy

ROOT = Path(__file__).resolve().parent
FONT = "C:/Windows/Fonts/msyh.ttc"


def load_model(spec, checkpoint, device, rows, cols):
    if ":" not in spec:
        raise SystemExit("--model 格式应为 path/to/model.py:ClassName")
    path, cls_name = spec.rsplit(":", 1)
    if path.endswith(".py"):
        p = Path(path)
        mod_spec = importlib.util.spec_from_file_location(p.stem, p)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(path)
    cls = getattr(module, cls_name)
    try:
        model = cls(rows=rows, cols=cols)
    except TypeError:
        model = cls()
    if checkpoint:
        import torch
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def to_tensor(img):
    import torch
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def model_predict(model, img, meta, device, hungarian=False):
    if hasattr(model, "predict"):
        return [int(x) for x in model.predict(img, meta)]
    import torch
    with torch.no_grad():
        if True:
            x = to_tensor(img).to(device)
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits = logits[0].detach().cpu()
            if logits.dim() == 2:
                n = logits.shape[0]
                if hungarian:
                    from scipy.optimize import linear_sum_assignment
                    rows, cols = linear_sum_assignment(-logits.numpy())
                    pred = [0] * n
                    for r, c in zip(rows, cols):
                        pred[r] = int(c)
                else:
                    pred = logits.argmax(-1).tolist()
            else:
                pred = logits.tolist()
    return [int(x) for x in pred]


def edge_predict(img, meta, beam):
    from solve_demo import solve
    rows, cols = meta["rows"], meta["cols"]
    pieces = [np.asarray(p) for p in split_pieces(img, rows, cols, trim_border=1)]
    grid, _ = solve(pieces, rows, cols, beam=beam)
    return [int(x) for x in grid]


def save_triplet(img, pred, meta, path, dataset_root):
    rows, cols = meta["rows"], meta["cols"]
    pieces = split_pieces(img, rows, cols, trim_border=1)
    pw, ph = pieces[0].size
    canvas = Image.new("RGB", (pw * cols, ph * rows))
    for s, q in enumerate(pred):
        r, c = divmod(s, cols)
        canvas.paste(pieces[q], (c * pw, r * ph))
    ans = Image.open(Path(dataset_root) / meta["answer"]).convert("RGB").crop(
        (1, 1, 1 + canvas.width, 1 + canvas.height))
    pad, header = 14, 40
    sheet = Image.new("RGB", (canvas.width * 3 + pad * 4,
                              canvas.height + header + pad + 24), (245, 247, 250))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT, 16)
    d.text((pad, 10), f"{meta['id']}  {rows}x{cols}  预测 / 复原 / 答案",
           fill=(30, 41, 59), font=font)
    sheet.paste(img, (pad, header))
    sheet.paste(canvas, (pad * 2 + canvas.width, header))
    sheet.paste(ans, (pad * 3 + canvas.width * 2, header))
    for i, t in enumerate(["打乱输入", "模型复原", "原图答案"]):
        d.text((pad + i * (canvas.width + pad), header + canvas.height + 3),
               t, fill=(71, 85, 105), font=font)
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--split", default="exam", choices=["study", "exam"])
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--limit", type=int, default=20, help="0 表示全部")
    ap.add_argument("--solver", default="edge",
                    choices=["edge", "random", "model", "csv"])
    ap.add_argument("--predictions-csv", default=None,
                    help="--solver csv 时读取：id,pred（pred 为空格分隔的排列）")
    ap.add_argument("--model", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda" if _cuda() else "cpu")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--hungarian", action="store_true",
                    help="模型输出 logits 时用匈牙利算法保证合法排列")
    ap.add_argument("--save-images", type=int, default=3)
    ap.add_argument("--out", default="predictions")
    args = ap.parse_args()

    ds = JigsawDataset(args.dataset, split=args.split,
                       rows=args.rows, cols=args.cols,
                       limit=args.limit or None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = None
    pred_map = {}
    if args.solver == "model":
        if not args.model:
            raise SystemExit("--solver model 需要 --model")
        model = load_model(args.model, args.checkpoint, args.device, args.rows, args.cols)
    elif args.solver == "csv":
        if not args.predictions_csv:
            raise SystemExit("--solver csv 需要 --predictions-csv")
        with open(args.predictions_csv, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                pred_map[row["id"]] = [int(x) for x in str(row["pred"]).split()]

    rows_csv = []
    accs, exacts, adjs = [], [], []
    for i in range(len(ds)):
        img, _, meta = ds[i]
        rows, cols = meta["rows"], meta["cols"]
        n = rows * cols
        scramble = meta["permutation"]
        target = inverse_permutation(scramble)  # 解格 -> 输入块

        if args.solver == "edge":
            pred = edge_predict(img, meta, args.beam)
        elif args.solver == "random":
            pred = list(range(n))
            random.shuffle(pred)
        elif args.solver == "csv":
            if meta["id"] not in pred_map:
                raise KeyError(f"CSV 里没有 {meta['id']} 的预测")
            pred = pred_map[meta["id"]]
        else:
            pred = model_predict(model, img, meta, args.device, args.hungarian)

        if sorted(pred) != list(range(n)):
            raise ValueError(f"{meta['id']} 输出不是合法排列：{pred}")
        pred_orig = [scramble[q] for q in pred]  # 解格 -> 原图块
        a = piece_accuracy(pred, target)
        e = exact_match(pred, target)
        j = adjacency_accuracy(pred_orig, rows, cols)
        accs.append(a); exacts.append(e); adjs.append(j)
        rows_csv.append({
            "id": meta["id"], "rows": rows, "cols": cols,
            "piece_acc": round(a, 4), "exact": int(e), "adjacency": round(j, 4),
            "pred": " ".join(map(str, pred)),
            "target": " ".join(map(str, target)),
        })
        if i < args.save_images:
            save_triplet(img, pred, meta, out / f"{meta['id']}_pred.png", args.dataset)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(ds)}  累计 piece_acc={np.mean(accs):.3f}", flush=True)

    csv_path = out / "predictions.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader(); w.writerows(rows_csv)

    print(f"\n样本 {len(ds)}")
    print(f"位置正确率 piece_acc = {np.mean(accs):.4f}")
    print(f"整题完全正确 exact    = {np.mean(exacts):.4f}")
    print(f"相邻正确率 adjacency  = {np.mean(adjs):.4f}")
    print(f"预测清单：{csv_path.resolve()}")
    print(f"可视化：{out.resolve()}")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
