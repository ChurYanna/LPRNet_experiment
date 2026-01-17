"""LPRNet/CCPD 解码与评测辅助函数（中文注释版）。

这里放“实验会频繁复用”的小函数：
- 从 CCPD 原始文件名解析出车牌文本（用于数据准备/在线加载）
- CTC greedy 解码（用于评测）
- 指标：整牌准确率、字符级准确率、NED（归一化编辑距离）

这些函数会被新的训练/评测脚本调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from lpr.charset import decode_ccpd_plate_indices


@dataclass(frozen=True)
class CtcDecoded:
    """一次 CTC 解码结果。"""

    label_ids: List[int]


def parse_ccpd_filename_plate_indices(filename: str) -> List[int]:
    """从 CCPD 原始文件名解析 plate 索引序列。

    CCPD 文件名典型格式：
    <id>-<angle>-<bbox>-<points>-<plate>-<brightness>-<blur>.jpg

    其中 <plate> 形如："0_0_26_25_32_15_24"（蓝牌 7 位）
    或 "0_0_3_17_24_33_27_26"（绿牌 8 位）

    返回
    - plate 索引列表（int）

    注意
    - 只做字符串解析，不做合法性校验；校验在 decode_ccpd_plate_indices 里完成。
    """

    stem = filename
    # 只取文件名，不关心目录
    if "/" in stem:
        stem = stem.rsplit("/", 1)[-1]
    if "\\" in stem:
        stem = stem.rsplit("\\", 1)[-1]

    # 去掉后缀
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]

    parts = stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"Not a CCPD-style filename: {filename}")

    plate_part = parts[4]
    idx_strs = plate_part.split("_")
    return [int(x) for x in idx_strs]


def parse_ccpd_filename_to_plate_text(filename: str) -> str:
    """从 CCPD 原始文件名得到车牌文本（中文+字母+数字）。"""

    indices = parse_ccpd_filename_plate_indices(filename)
    return decode_ccpd_plate_indices(indices).plate_text


def ctc_greedy_decode(
    probs: np.ndarray,
    blank_id: int,
) -> CtcDecoded:
    """CTC greedy 解码（去重 + 去 blank）。

    参数
    - probs: shape = [C, T] 或 [T, C] 的概率/打分
      训练时我们通常拿到 logits: [C, T]（例如 [68, 18]）。
      这里只要求能 argmax 出每个时间步的 label。
    - blank_id: blank 的类别 id（本仓库默认是 len(CHARS)-1，对应 '-'）。

    返回
    - CtcDecoded(label_ids=...)

    说明
    - greedy 只取每个时间步最大概率的类。
    - CTC 输出需要：
      1) 去掉重复连续字符
      2) 去掉 blank
    """

    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape={probs.shape}")

    # 统一成 [T, C]
    # LPRNet 常见输出是 [C, T]（例如 [68, 18]），而评测希望按时间步解码。
    # 因此：若第 0 维明显更大，优先认为是 [C, T] 并转置。
    if probs.shape[0] > probs.shape[1]:
        probs_tc = probs.T
    else:
        probs_tc = probs

    best_ids = probs_tc.argmax(axis=1).tolist()

    out: List[int] = []
    prev: Optional[int] = None
    for cur in best_ids:
        if cur == blank_id:
            prev = cur
            continue
        if prev == cur:
            continue
        out.append(cur)
        prev = cur

    return CtcDecoded(label_ids=out)


def _levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    """编辑距离（Levenshtein distance），用于 NED 指标。"""

    # 经典 DP，长度很短（车牌一般 <= 8），开销可忽略
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev_diag = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,        # 删除
                dp[j - 1] + 1,    # 插入
                prev_diag + cost, # 替换
            )
            prev_diag = tmp
    return dp[m]


@dataclass(frozen=True)
class LprMetrics:
    exact_acc: float
    char_acc: float
    ned: float


def compute_lpr_metrics(
    preds: Iterable[Sequence[int]],
    gts: Iterable[Sequence[int]],
) -> LprMetrics:
    """计算识别指标：整牌准确率 / 字符级准确率 / NED。

    - exact_acc：预测序列与 GT 完全一致才算对
    - char_acc：按位置匹配的字符准确率（简单直观）
    - ned：Normalized Edit Distance = 1 - (edit_distance / max_len)

    说明：
    - CCPD 车牌长度多数为 7/8，因此 char_acc 用“对齐位置”足够直观。
    - 但当长度不一致时，char_acc 会偏保守；NED 能更公平地反映“错了多少”。
    """

    preds_list = [list(p) for p in preds]
    gts_list = [list(t) for t in gts]
    if len(preds_list) != len(gts_list):
        raise ValueError("preds and gts length mismatch")

    n = len(preds_list)
    if n == 0:
        return LprMetrics(exact_acc=0.0, char_acc=0.0, ned=0.0)

    exact = 0
    char_correct = 0
    char_total = 0
    ned_sum = 0.0

    for p, t in zip(preds_list, gts_list):
        if p == t:
            exact += 1

        # 字符级：按最短长度对齐
        L = min(len(p), len(t))
        for i in range(L):
            if p[i] == t[i]:
                char_correct += 1
        char_total += max(len(t), 1)

        # NED
        dist = _levenshtein(p, t)
        denom = max(len(p), len(t), 1)
        ned_sum += 1.0 - (dist / denom)

    return LprMetrics(
        exact_acc=exact / n,
        char_acc=char_correct / char_total if char_total else 0.0,
        ned=ned_sum / n,
    )
