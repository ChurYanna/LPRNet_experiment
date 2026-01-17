"""LPR（车牌识别）相关的字符表与CCPD标签解码。

本仓库的 LPRNet 输出字典沿用 [models/LPRNet.py] 中的 CHARS。
但 CCPD 原始数据集文件名中，车牌字符通常以“索引序列”的方式编码（plate 字段）。
此文件提供：
1) CCPD plate 索引 -> 车牌字符串 的解码
2) LPRNet 字符表（CHARS）与索引工具

注意：
- CCPD 的字符集合与顺序与 LPRNet 的 CHARS 不完全一致（例如 CCPD 有“警/学”等）。
- 本仓库的识别任务聚焦蓝牌/绿牌的常见字符，暂不强求覆盖 CCPD 全字符。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# CCPD 文件名中 plate 字段的编码方式（来自 CCPD 官方/社区常用约定）。
# 参考原仓库的 tools/ccpd2lpr.py（已在此项目中存在）。
CCPD_PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "警", "学", "O",
]

# 第二位字母（不含 I，通常含 O）
CCPD_ALPHABETS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "O",
]

# 后续位（字母+数字，通常含 O）
CCPD_ADS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "O",
]


@dataclass(frozen=True)
class PlateDecodeResult:
    """一次 CCPD plate 字段解码的结果。"""

    plate_text: str
    # 原始索引序列（便于调试/统计）
    plate_indices: List[int]


def decode_ccpd_plate_indices(indices: List[int]) -> PlateDecodeResult:
    """将 CCPD 文件名里的 plate 索引序列解码为车牌字符串。

    CCPD 规则：
    - 第 1 位：省份（CCPD_PROVINCES）
    - 第 2 位：字母（CCPD_ALPHABETS）
    - 第 3..：字母/数字（CCPD_ADS），蓝牌通常 5 位，绿牌可能 6 位

    参数
    - indices: 例如 [0,0,26,25,32,15,24] 或 [0,0,3,17,24,33,27,26]

    返回
    - PlateDecodeResult(plate_text=..., plate_indices=...)

    约束
    - 若索引越界，会抛出 ValueError，提示数据不符合预期。
    """

    if len(indices) < 7:
        raise ValueError(f"CCPD plate indices too short: len={len(indices)} indices={indices}")

    p0, p1, *rest = indices
    try:
        text = CCPD_PROVINCES[p0] + CCPD_ALPHABETS[p1] + "".join(CCPD_ADS[i] for i in rest)
    except Exception as exc:
        raise ValueError(f"CCPD plate indices out of range: {indices}") from exc

    return PlateDecodeResult(plate_text=text, plate_indices=indices)
