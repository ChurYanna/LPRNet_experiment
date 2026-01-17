"""用于 LPRNet 实验的 Dataset（中文注释版）。

目标：
- 不依赖 Windows 路径
- 支持按比例抽样（10%/30%/100%）以满足 PDF 的“数据比例对比实验”
- 支持读取“已裁剪好的车牌小图”（推荐，速度快）

数据格式（推荐）：
- 目录结构：
    <root>/train/*.jpg
    <root>/val/*.jpg
    <root>/test/*.jpg

- 文件名必须以“车牌文本”开头，例如：
    京A12345_xxx.jpg
    粤B12D345_xxx.jpg
  Dataset 会取第一个 '_' 或 '-' 前的字符串作为 label。

注意：
- 这里的 label 文本必须能在 models/LPRNet.py 的 CHARS 中找到，否则会报错。
- 为了避免同一车牌重复导致覆盖，建议保存时加后缀（如原始stem）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from models.LPRNet import CHARS


CHARS_DICT = {c: i for i, c in enumerate(CHARS)}


def _extract_plate_text_from_filename(path: str) -> str:
    name = Path(path).name
    stem = name.rsplit(".", 1)[0]
    # 兼容两种常见约定：<plate>_<anything> 或 <plate>-<anything>
    plate = stem.split("_")[0].split("-")[0]
    return plate


def _normalize_img(img_bgr: np.ndarray) -> np.ndarray:
    """与仓库原版保持一致的归一化：img -> float32, (x-127.5)/128。"""

    img = img_bgr.astype("float32")
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return img


@dataclass(frozen=True)
class LprSample:
    img: torch.Tensor
    label_flat: torch.Tensor
    length: int


class LprCroppedDataset(Dataset):
    """读取“裁剪后的车牌小图”，用于训练/验证/测试。

    参数
    - img_dir: 图片目录（单个 split）
    - img_size: (W, H) = (94, 24)
    - subset_ratio: 0~1，按比例抽样（用于 B 组实验）
    - max_samples: 限制最大样本数（便于快速 debug）
    - seed: 抽样随机种子（保证可复现）
    - augment_fn: 可选数据增强（保持输入输出不变）
    """

    def __init__(
        self,
        img_dir: str,
        img_size: Tuple[int, int] = (94, 24),
        subset_ratio: float = 1.0,
        max_samples: Optional[int] = None,
        seed: int = 0,
        augment_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.img_dir = str(img_dir)
        self.img_size = img_size
        self.augment_fn = augment_fn

        p = Path(self.img_dir)
        if not p.exists():
            raise FileNotFoundError(f"img_dir not found: {self.img_dir}")

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        all_paths = [str(x) for x in p.rglob("*") if x.suffix.lower() in exts]
        all_paths.sort()

        # 按比例抽样（可复现）
        if not (0.0 < subset_ratio <= 1.0):
            raise ValueError(f"subset_ratio must be in (0,1], got {subset_ratio}")

        if subset_ratio < 1.0:
            rng = np.random.default_rng(seed)
            k = max(1, int(len(all_paths) * subset_ratio))
            idx = rng.choice(len(all_paths), size=k, replace=False)
            idx = np.sort(idx)
            paths = [all_paths[i] for i in idx.tolist()]
        else:
            paths = all_paths

        if max_samples is not None:
            paths = paths[: int(max_samples)]

        self.img_paths = paths

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int):
        filename = self.img_paths[index]

        # 兼容中文路径：用 imdecode
        buf = np.fromfile(filename, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {filename}")

        # 统一尺寸（LPRNet 约定 W=94, H=24）
        if (img.shape[1], img.shape[0]) != self.img_size:
            img = cv2.resize(img, self.img_size)

        if self.augment_fn is not None:
            img = self.augment_fn(img)

        plate_text = _extract_plate_text_from_filename(filename)
        label_ids: List[int] = []
        for ch in plate_text:
            if ch not in CHARS_DICT:
                raise ValueError(f"Unsupported char '{ch}' in {filename} (plate='{plate_text}')")
            label_ids.append(CHARS_DICT[ch])

        img_t = torch.from_numpy(_normalize_img(img))
        label_t = torch.tensor(label_ids, dtype=torch.long)
        return img_t, label_t, len(label_ids)


def lpr_collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, int]]):
    """把不定长 label 拼成 CTC 需要的一维向量，并返回每个样本长度。"""

    imgs: List[torch.Tensor] = []
    labels: List[int] = []
    lengths: List[int] = []

    for img, label, length in batch:
        imgs.append(img)
        labels.extend(label.tolist())
        lengths.append(int(length))

    return torch.stack(imgs, 0), torch.tensor(labels, dtype=torch.long), lengths
