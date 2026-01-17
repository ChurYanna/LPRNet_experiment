"""LPRNet 识别训练脚本（实验版，中文注释）。

本脚本专门为了课程 PDF 的实验要求设计：
- A：不同网络规模（--width-mult）
- B：不同比例训练数据（--subset-ratio）
- C：不同学习率策略（--lr-scheduler: fixed/step/cosine）

并补齐“工程实现与可复现”加分项：
- seed 固定、日志记录（CSV + TensorBoard）
- 统一评测（exact acc / char acc / NED）
- AMP 混合精度加速（默认开启，适配你的 5060Ti）

数据输入：使用 tools/prepare_lpr_data.py 生成的裁剪图目录，例如 data/ccpd_lpr。
目录结构必须包含 train/val/test。

运行示例：
# 1) baseline
conda run -n pytorch python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name baseline

# 2) 网络规模对比
conda run -n pytorch python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name w075 --width-mult 0.75
conda run -n pytorch python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name w125 --width-mult 1.25

# 3) 数据比例对比
conda run -n pytorch python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name r10 --subset-ratio 0.1
conda run -n pytorch python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name r30 --subset-ratio 0.3

# 4) 学习率策略对比
conda run -n pytorch python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name fixed --lr-scheduler fixed
conda run -n pytorch python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name step  --lr-scheduler step
conda run -n pytorch python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name cosine --lr-scheduler cosine
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 让脚本无论从哪里运行都能正确 import 本项目代码
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.LPRNet import CHARS, LPRNet
from utils.lpr_dataset_v2 import LprCroppedDataset, lpr_collate_fn
from utils.lpr_decode import ctc_greedy_decode, compute_lpr_metrics
from utils.lpr_runtime import init_seeds, torch_load


def _ctc_lengths(T: int, lengths: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """CTC 需要 input_lengths/target_lengths。LPRNet 的时间步长度 T 固定。"""

    input_lengths = tuple([T for _ in lengths])
    target_lengths = tuple([int(x) for x in lengths])
    return input_lengths, target_lengths


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()

    blank_id = len(CHARS) - 1
    preds: List[List[int]] = []
    gts: List[List[int]] = []

    for imgs, labels_flat, lengths in loader:
        imgs = imgs.to(device, non_blocking=True)

        # 还原 batch 的 GT 序列
        start = 0
        batch_gt: List[List[int]] = []
        for L in lengths:
            seq = labels_flat[start : start + L].tolist()
            batch_gt.append(seq)
            start += L

        logits = model(imgs)  # [B, C, T]
        probs = logits.detach().cpu().numpy()

        for b in range(probs.shape[0]):
            dec = ctc_greedy_decode(probs[b], blank_id=blank_id)
            preds.append(dec.label_ids)
            gts.append(batch_gt[b])

    m = compute_lpr_metrics(preds, gts)
    return {
        "exact_acc": float(m.exact_acc),
        "char_acc": float(m.char_acc),
        "ned": float(m.ned),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train LPRNet (experiments)")

    p.add_argument("--data-root", type=str, default="data/ccpd_lpr", help="裁剪后的数据根目录")
    p.add_argument("--runs", type=str, default="runs/lprnet", help="输出目录")
    p.add_argument("--name", type=str, default="exp", help="实验名称")

    # 训练设置
    # argparse 会对 help 字符串做 %-formatting，因此字面量 % 需要写成 %%
    p.add_argument("--epochs", type=int, default=60, help="默认 60 epoch，配合你筛选后的100%%数据集控制在~2小时内")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    # 模型规模（实验A）
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.5)

    # 数据比例（实验B）
    p.add_argument("--subset-ratio", type=float, default=1.0)

    # 优化与学习率（实验C）
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-scheduler", type=str, default="cosine", choices=["fixed", "step", "cosine"])
    p.add_argument("--step-size", type=int, default=20, help="StepLR 的 step_size")
    p.add_argument("--gamma", type=float, default=0.1, help="StepLR 的衰减系数")

    # 加速
    p.add_argument("--amp", action="store_true", help="开启 AMP（默认关闭该flag；建议命令行加 --amp）")
    p.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")

    # 断点/预训练
    p.add_argument("--pretrained", type=str, default="", help="可选：加载已有权重继续训练")

    return p


def main() -> None:
    args = build_parser().parse_args()

    init_seeds(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"data-root must contain train/val: {data_root}")

    run_dir = Path(args.runs) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)

    # CSV 日志（便于报告画表/画图）
    csv_path = run_dir / "metrics.csv"
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.DictWriter(csv_f, fieldnames=["epoch", "train_loss", "val_exact_acc", "val_char_acc", "val_ned", "lr", "sec"])
    csv_w.writeheader()
    csv_f.flush()

    tb = SummaryWriter(log_dir=str(run_dir / "tb"))

    # Dataset：train 支持 subset_ratio 抽样（B组实验）
    train_ds = LprCroppedDataset(str(train_dir), subset_ratio=args.subset_ratio, seed=args.seed)
    val_ds = LprCroppedDataset(str(val_dir), subset_ratio=1.0, seed=args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lpr_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lpr_collate_fn,
        drop_last=False,
    )

    # Model
    model = LPRNet(lpr_max_len=8, phase=True, class_num=len(CHARS), dropout_rate=args.dropout, width_mult=args.width_mult)
    model.to(device)

    if args.pretrained:
        state = torch_load(args.pretrained, map_location="cpu")
        model.load_state_dict(state, strict=True)

    # Loss & Optim
    # blank 使用最后一个字符（'-'）
    ctc_loss = nn.CTCLoss(blank=len(CHARS) - 1, reduction="mean", zero_infinity=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == "fixed":
        scheduler = None
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    else:
        raise ValueError(args.lr_scheduler)

    # PyTorch 新版推荐使用 torch.amp（torch.cuda.amp 会给出 FutureWarning）。
    amp_enabled = bool(args.amp) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=amp_enabled)
    except Exception:
        # 兼容旧版 PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=amp_enabled)

    best_val = -1.0

    # LPRNet 输出时间步长度固定为 18（与当前模型实现一致）
    T = 18

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        loss_sum = 0.0
        steps = 0

        for imgs, labels_flat, lengths in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels_flat = labels_flat.to(device, non_blocking=True)

            input_lengths, target_lengths = _ctc_lengths(T, lengths)

            optimizer.zero_grad(set_to_none=True)

            # AMP 可以显著提速（对 5060Ti 很有用），也更省显存
            with autocast_ctx():
                logits = model(imgs)  # [B, C, T]
                log_probs = logits.permute(2, 0, 1).log_softmax(2)  # [T, B, C]
                loss = ctc_loss(log_probs, labels_flat, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.item())
            steps += 1

        if scheduler is not None:
            scheduler.step()

        # 验证
        val_m = evaluate(model, val_loader, device)
        lr_now = float(optimizer.param_groups[0]["lr"])
        sec = float(time.time() - t0)

        train_loss = loss_sum / max(steps, 1)

        # 记录日志
        tb.add_scalar("loss/train", train_loss, epoch)
        tb.add_scalar("val/exact_acc", val_m["exact_acc"], epoch)
        tb.add_scalar("val/char_acc", val_m["char_acc"], epoch)
        tb.add_scalar("val/ned", val_m["ned"], epoch)
        tb.add_scalar("lr", lr_now, epoch)

        csv_w.writerow({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_exact_acc": f"{val_m['exact_acc']:.6f}",
            "val_char_acc": f"{val_m['char_acc']:.6f}",
            "val_ned": f"{val_m['ned']:.6f}",
            "lr": f"{lr_now:.8f}",
            "sec": f"{sec:.2f}",
        })
        csv_f.flush()

        # 保存 best
        if val_m["exact_acc"] > best_val:
            best_val = val_m["exact_acc"]
            torch.save(model.state_dict(), run_dir / "weights" / "best.pth")

        # 也存一个 last（便于中断继续）
        torch.save(model.state_dict(), run_dir / "weights" / "last.pth")

        print(
            f"Epoch {epoch:03d}/{args.epochs} | loss={train_loss:.4f} | "
            f"val_exact={val_m['exact_acc']:.4f} val_char={val_m['char_acc']:.4f} val_ned={val_m['ned']:.4f} | "
            f"lr={lr_now:.6g} | {sec:.1f}s"
        )

    tb.close()
    csv_f.close()


if __name__ == "__main__":
    main()
