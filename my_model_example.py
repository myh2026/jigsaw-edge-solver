# -*- coding: utf-8 -*-
"""最小“模型”示例：把经典边匹配 solver 包装成 predict 接口。

你的 AI 只需要实现同样签名：
    predict(image: PIL.Image, meta: dict) -> list[int]
返回长度 N 的“解格 -> 输入块”排列。
"""
import numpy as np

from loader import split_pieces
from solve_demo import solve


class EdgeModel:
    def __init__(self, rows=None, cols=None, beam=4):
        self.beam = beam

    def predict(self, image, meta):
        rows, cols = meta["rows"], meta["cols"]
        pieces = [np.asarray(p) for p in split_pieces(image, rows, cols, trim_border=1)]
        grid, _ = solve(pieces, rows, cols, beam=self.beam)
        return [int(x) for x in grid]
