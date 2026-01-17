"""统一评测脚本：对指定权重在 test 集上评估（中文注释）。

用途：
- 对比旧权重 weights/lprnet_best.pth 与你新训练的 best.pth
- 输出更全面的指标：整牌准确率 / 字符级准确率 / NED

示例：
conda run -n pytorch python tools/eval_lprnet.py --data-root data/ccpd_lpr --weights weights/lprnet_best.pth
conda run -n pytorch python tools/eval_lprnet.py --data-root data/ccpd_lpr --weights runs/lprnet/baseline/weights/best.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

# 让脚本无论从哪里运行都能正确 import 本项目代码
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.LPRNet import CHARS, LPRNet
from utils.lpr_dataset_v2 import LprCroppedDataset, lpr_collate_fn
from utils.lpr_decode import ctc_greedy_decode, compute_lpr_metrics
from utils.lpr_runtime import torch_load
from torch.utils.data import DataLoader


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Eval LPRNet")
    p.add_argument("--data-root", type=str, default="data/ccpd_lpr")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--width-mult", type=float, default=1.0, help="若评测的是缩放网络，请保持一致")
    p.add_argument("--dropout", type=float, default=0.0, help="评测时 dropout 一般设为0")
    p.add_argument(
        "--save",
        type=str,
        default="",
        help="可选：把评测结果保存为 json/csv（传入输出文件路径，如 runs/lprnet/baseline/test_metrics.json）",
    )
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_dir = Path(args.data_root) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"data-root must contain test/: {args.data_root}")

    ds = LprCroppedDataset(str(test_dir), subset_ratio=1.0, seed=0)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lpr_collate_fn,
        drop_last=False,
    )

    model = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=args.dropout, width_mult=args.width_mult)
    state = torch_load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    blank_id = len(CHARS) - 1

    preds = []
    gts = []

    for imgs, labels_flat, lengths in loader:
        imgs = imgs.to(device, non_blocking=True)

        # 还原 GT
        start = 0
        batch_gt = []
        for L in lengths:
            seq = labels_flat[start : start + L].tolist()
            batch_gt.append(seq)
            start += L

        logits = model(imgs)
        probs = logits.detach().cpu().numpy()

        for b in range(probs.shape[0]):
            dec = ctc_greedy_decode(probs[b], blank_id=blank_id)
            preds.append(dec.label_ids)
            gts.append(batch_gt[b])

    m = compute_lpr_metrics(preds, gts)
    result = {
        "exact_acc": float(m.exact_acc),
        "char_acc": float(m.char_acc),
        "ned": float(m.ned),
        "weights": str(args.weights),
    }
    print(f"[Result] exact_acc={result['exact_acc']:.6f} char_acc={result['char_acc']:.6f} ned={result['ned']:.6f}")

    if args.save.strip():
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".json":
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        elif out_path.suffix.lower() == ".csv":
            with out_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(result.keys()))
                w.writeheader()
                w.writerow(result)
        else:
            raise ValueError("--save must end with .json or .csv")


if __name__ == "__main__":
    main()
