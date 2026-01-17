"""LPRNet 实验运行时工具（最小依赖，中文注释）。

目的：
- 为 LPRNet-only 的训练/评测脚本提供必要的通用函数
- 避免依赖 YOLOv5 那套庞大的 utils/* 文件，减少仓库复杂度
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def torch_load(path: str, map_location: Optional[str] = None):
    """兼容不同 PyTorch 版本的 torch.load。

    说明：
    - PyTorch>=2.6 默认 weights_only=True，可能导致旧权重加载失败
    - 这里显式尝试 weights_only=False，失败再回退
    """

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def init_seeds(seed: int = 0) -> None:
    """固定随机种子，保证实验可复现。"""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 说明：下面两项会让训练更“可复现”，但可能牺牲一点速度
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
