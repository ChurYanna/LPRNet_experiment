"""从 CCPD2019/CCPD2020 原始图片生成 LPRNet 识别训练数据（裁剪车牌小图）。

你给的数据目录：/home/aaa/learn/sight
- CCPD2019：多个子集（ccpd_base/ccpd_blur/...）
- CCPD2020：ccpd_green/train|val|test

本脚本做的事：
1) 从文件名解析车牌文字（CCPD 的 plate 索引序列）
2) 从文件名解析车牌 bbox，裁剪出车牌区域
3) resize 到 LPRNet 输入 (94, 24)
4) 保存到输出目录（默认 data/ccpd_lpr/{train,val,test}）

为什么要生成裁剪图？
- 训练速度更快（避免每个 epoch 都做 crop）
- 也更方便做“数据比例实验”（只对裁剪图做抽样即可）

注意：
- CCPD2019 里的图片很多，本脚本默认只从 ccpd_base 抽取一定数量。
- 你可以通过 --blue-max-* / --green-max-* 控制规模，从而保证“100%训练集 <= 2小时”。

运行示例：
conda run -n pytorch python tools/prepare_lpr_data.py \
  --ccpd-root /home/aaa/learn/sight \
  --out-root data/ccpd_lpr \
  --seed 0 \
  --blue-subdir ccpd_base \
  --blue-max-train 60000 --blue-max-val 5000 --blue-max-test 5000 \
  --green-max-train 30000 --green-max-val 5000 --green-max-test 5000
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 让脚本无论从哪里运行都能正确 import 本项目代码
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.lpr_decode import parse_ccpd_filename_to_plate_text


def _parse_bbox_from_ccpd_filename(filename: str) -> Tuple[int, int, int, int]:
    """从 CCPD 文件名解析 bbox（xmin, ymin, xmax, ymax）。"""

    name = Path(filename).name
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"Not a CCPD-style filename: {filename}")

    bbox_part = parts[2]  # 形如 "228&431_515&519"
    p0, p1 = bbox_part.split("_")
    x0, y0 = [int(x) for x in p0.split("&")]
    x1, y1 = [int(x) for x in p1.split("&")]
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)
    return xmin, ymin, xmax, ymax


def _safe_crop_resize(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int], out_size: Tuple[int, int]) -> np.ndarray:
    """安全裁剪并 resize，避免越界。"""

    h, w = img_bgr.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, min(xmin, w - 1))
    xmax = max(1, min(xmax, w))
    ymin = max(0, min(ymin, h - 1))
    ymax = max(1, min(ymax, h))

    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid bbox after clamp: {bbox} for image shape={img_bgr.shape}")

    crop = img_bgr[ymin:ymax, xmin:xmax]
    crop = cv2.resize(crop, out_size)
    return crop


def _read_image_bgr(path: str) -> np.ndarray:
    """读取图片（支持中文路径）。"""

    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _write_jpg(path: str, img_bgr: np.ndarray) -> None:
    """保存 jpg（支持中文路径）。"""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode jpg: {path}")
    enc.tofile(str(out_path))


def _iter_images(dir_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in dir_path.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def _sample_paths(paths: Sequence[Path], k: int, seed: int) -> List[Path]:
    if k <= 0:
        return []
    if k >= len(paths):
        return list(paths)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=k, replace=False)
    idx = np.sort(idx)
    return [paths[i] for i in idx.tolist()]


@dataclass(frozen=True)
class SplitPlan:
    train: int
    val: int
    test: int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare CCPD -> LPRNet crops")

    p.add_argument("--ccpd-root", type=str, default="/home/aaa/learn/sight", help="CCPD 根目录")
    p.add_argument("--out-root", type=str, default="data/ccpd_lpr", help="输出目录（会生成 train/val/test）")
    p.add_argument("--seed", type=int, default=0, help="随机种子（控制抽样可复现）")
    p.add_argument("--out-size", type=int, nargs=2, default=[94, 24], help="输出尺寸 W H")

    # CCPD2019（蓝牌为主）
    p.add_argument("--blue-subdir", type=str, default="ccpd_base", help="CCPD2019 使用哪个子集目录")
    p.add_argument("--blue-max-train", type=int, default=60000)
    p.add_argument("--blue-max-val", type=int, default=5000)
    p.add_argument("--blue-max-test", type=int, default=5000)

    # CCPD2020（绿牌）
    p.add_argument("--green-max-train", type=int, default=30000)
    p.add_argument("--green-max-val", type=int, default=5000)
    p.add_argument("--green-max-test", type=int, default=5000)

    p.add_argument("--dry-run", action="store_true", help="只统计/打印，不生成文件")
    return p


def main() -> None:
    args = build_parser().parse_args()

    ccpd_root = Path(args.ccpd_root)
    out_root = Path(args.out_root)
    out_w, out_h = int(args.out_size[0]), int(args.out_size[1])
    out_size = (out_w, out_h)

    # 1) 收集路径
    blue_dir = ccpd_root / "CCPD2019" / args.blue_subdir
    green_train = ccpd_root / "CCPD2020" / "ccpd_green" / "train"
    green_val = ccpd_root / "CCPD2020" / "ccpd_green" / "val"
    green_test = ccpd_root / "CCPD2020" / "ccpd_green" / "test"

    if not blue_dir.exists():
        raise FileNotFoundError(f"Blue dir not found: {blue_dir}")
    for d in (green_train, green_val, green_test):
        if not d.exists():
            raise FileNotFoundError(f"Green dir not found: {d}")

    blue_paths = _iter_images(blue_dir)
    green_train_paths = _iter_images(green_train)
    green_val_paths = _iter_images(green_val)
    green_test_paths = _iter_images(green_test)

    # 2) 对蓝牌做抽样 + 7:1:2 划分（可复现）
    # 说明：CCPD2019 很大，强烈建议先抽样后划分。
    blue_total = args.blue_max_train + args.blue_max_val + args.blue_max_test
    blue_paths = _sample_paths(blue_paths, blue_total, seed=args.seed)
    # 再划分
    b_train = blue_paths[: args.blue_max_train]
    b_val = blue_paths[args.blue_max_train : args.blue_max_train + args.blue_max_val]
    b_test = blue_paths[args.blue_max_train + args.blue_max_val :]

    # 3) 绿牌按原 split 抽样（可复现）
    g_train = _sample_paths(green_train_paths, args.green_max_train, seed=args.seed)
    g_val = _sample_paths(green_val_paths, args.green_max_val, seed=args.seed)
    g_test = _sample_paths(green_test_paths, args.green_max_test, seed=args.seed)

    print("[Info] Blue selected:", len(b_train), len(b_val), len(b_test), "from", args.blue_subdir)
    print("[Info] Green selected:", len(g_train), len(g_val), len(g_test), "from ccpd_green")

    if args.dry_run:
        # 打印几个样例，便于你确认解码是否正确
        for demo in (b_train[:2] + g_train[:2]):
            try:
                text = parse_ccpd_filename_to_plate_text(demo.name)
                bbox = _parse_bbox_from_ccpd_filename(demo.name)
                print("[DryRun]", demo.name, "->", text, "bbox", bbox)
            except Exception as e:
                print("[DryRun][Error]", demo.name, e)
        return

    # 4) 生成裁剪图
    def process_split(paths: Sequence[Path], split: str) -> None:
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        ok = 0
        bad = 0
        for p in paths:
            try:
                plate_text = parse_ccpd_filename_to_plate_text(p.name)
                bbox = _parse_bbox_from_ccpd_filename(p.name)
                img = _read_image_bgr(str(p))
                crop = _safe_crop_resize(img, bbox, out_size)

                # 防止同名覆盖：用 plate_text + 原 stem
                out_name = f"{plate_text}_{p.stem}.jpg"
                _write_jpg(str(out_dir / out_name), crop)
                ok += 1
            except Exception:
                bad += 1
                continue

        print(f"[Info] split={split} done: ok={ok} bad={bad} out={out_dir}")

    process_split(b_train + g_train, "train")
    process_split(b_val + g_val, "val")
    process_split(b_test + g_test, "test")


if __name__ == "__main__":
    main()
