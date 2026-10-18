# Jigsaw Edge Solver · 基于学习校准边缘代价的图像拼图复原

图像拼图复原（jigsaw puzzle solving）的完整解决方案：**边缘匹配网络（EdgeNet）+ 混合代价校准 + 束搜索组装**。

核心协议：**训练与校准只使用学习卷（study split）**，考试卷（exam split）仅在最终评测时打开。

## 结果（考试卷，每网格 100 题）

| 网格 | 位置正确率 | 整题完全正确 | 相邻正确率 | 经典 L1 基线（同题） |
|:---:|:---:|:---:|:---:|:---:|
| 3×3 | 0.938 | 0.920 | 0.971 | 0.920 |
| 4×4 | 0.823 | 0.770 | 0.923 | 0.780 |
| 5×5 | **0.750** | **0.670** | **0.863** | 0.660 |

5×5（最难网格）上超过基线；束宽消融显示 5 为最优（4→0.741，5→0.750，8→0.707，非单调）。

## 方法概要

```
学习卷 100 张原图
   ├─ 正对（真实相邻边缘）+ 负对（同图隔块/绕接/跨图）  →  EdgeNet（小 1D-CNN）
   │     验证（未见原图）对分类 acc 0.86
   └─ α 校准：束搜索代价 = α·z(L1) + (1−α)·z(EdgeNet)
         学习卷 exact：纯网络 0.77 / 混合 0.94 / 纯 L1 0.95 → α=1.0
考试卷（未参与任何训练/调参）
   └─ 束搜索（beam=5）组装 → 上表
```

诚实结论：该数据集的生成方式（无噪切割、去描边后相邻块像素完全连续）使 L1
距离本身已是近乎充分的统计量；学习网络接近（0.94）但未能超越，校准流程
正确地识别了这一点。学习的真实价值：确认经典度量近优 + 校准束宽。在含
噪声/几何扰动的现实拼图数据上，学习度量的优势才会显现。

## 仓库结构

```
edge_net_solver.py        核心：EdgeNet 训练 / α 校准 / 求解器（三合一）
train_study_reshuffle.py  对照路线：学习卷原图自重排 + CNN+Transformer（CPU 算力不足的负结果）
loader.py / metrics.py    数据加载与指标（permutation 语义、solution 逆排列）
solve_demo.py             经典 L1 边匹配束搜索（基线，EdgeNet 复用其 edge_costs）
evaluate_model.py         统一评测入口（model / edge / csv 三种 solver）
paper/paper.md            论文正文
results/results.md        完整实验记录（α 扫描、束宽消融、主表）
```

## 复现

```bash
pip install torch numpy pillow   # CPU 即可
# 1) 训练（只用学习卷原图）
python edge_net_solver.py --train
# 2) α 校准（只用学习卷题目）
python edge_net_solver.py --calibrate
# 3) 考试卷评测
python evaluate_model.py --solver model --model edge_net_solver.py:EdgeNetSolver \
    --split exam --rows 3 --limit 100
```

数据集不随仓库分发（课程资料）：期望目录 `dataset/{study_100,exam_10000}`，
manifest 含 `permutation`（输入格 → 原图块）；模型输出约定为
`pred[s] = 解格 s 放第几个输入块`（permutation 的逆）。

## 协议合规说明

- 训练数据：study_100 的 100 张原图（80 训练 / 20 验证，图级划分，无跨卷泄漏）；
- 超参（α、束宽）：在学习卷上选择，考试卷零调参；
- 考试卷（exam_10000）仅用于最终评测。
