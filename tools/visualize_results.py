"""结果可视化与汇总（LPRNet-only，课堂实验手册配套）。

功能：
- 读取 runs/lprnet/<exp>/metrics.csv
- 生成多实验对比曲线：train_loss、val_exact_acc、val_char_acc、val_ned、lr
- 生成汇总表：每个实验 best val 的 epoch/指标/耗时

注意：
- 车牌识别是序列识别任务，不适用检测任务的 mAP。
- 本仓库的核心指标：exact_acc（整牌）、char_acc（字符级）、NED。

用法示例：
python tools/visualize_results.py --runs-dir runs/lprnet --out-dir runs/lprnet/reports
python tools/visualize_results.py --runs-dir runs/lprnet --out-dir runs/lprnet/reports --experiments baseline,A_w075,A_w100,A_w125
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt


def _read_metrics_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 统一转 float，epoch 单独处理
            row: Dict[str, float] = {}
            for k, v in r.items():
                if v is None:
                    continue
                v = str(v).strip()
                if v == "":
                    continue
                if k == "epoch":
                    row[k] = float(int(float(v)))
                else:
                    row[k] = float(v)
            rows.append(row)
    return rows


def _list_experiments(runs_dir: Path) -> List[str]:
    exps: List[str] = []
    if not runs_dir.exists():
        return exps
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "metrics.csv").exists():
            exps.append(d.name)
    return exps


def _best_by_key(rows: Sequence[Dict[str, float]], key: str) -> Tuple[int, Dict[str, float]]:
    best_epoch = -1
    best_row: Dict[str, float] = {}
    best_val = None
    for r in rows:
        if key not in r:
            continue
        val = float(r[key])
        if best_val is None or val > best_val:
            best_val = val
            best_row = r
            best_epoch = int(r.get("epoch", -1))
    return best_epoch, best_row


def _plot_overlay(
    series: Dict[str, Tuple[List[int], List[float]]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(9, 5))
    for name, (xs, ys) in series.items():
        if not xs:
            continue
        plt.plot(xs, ys, label=name, linewidth=2)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_single(
    xs: Sequence[int],
    ys: Sequence[float],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """单个实验的曲线图。"""

    plt.figure(figsize=(9, 5))
    plt.plot(list(xs), list(ys), linewidth=2)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _write_summary_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize LPRNet experiment results")
    p.add_argument("--runs-dir", type=str, default="runs/lprnet", help="训练输出根目录")
    p.add_argument("--out-dir", type=str, default="runs/lprnet/reports", help="图表/汇总输出目录")
    p.add_argument(
        "--experiments",
        type=str,
        default="",
        help="要绘制的实验名（逗号分隔）。留空表示自动发现 runs-dir 下所有含 metrics.csv 的实验。",
    )
    p.add_argument(
        "--per-experiment",
        action="store_true",
        help="额外为每个实验生成单独曲线与小结（默认关闭；建议开启用于写报告）。",
    )
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_base = Path(args.out_dir)

    if args.experiments.strip():
        experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
    else:
        experiments = _list_experiments(runs_dir)

    if not experiments:
        raise SystemExit(f"No experiments found under: {runs_dir}")

    # 读取所有实验
    exp_rows: Dict[str, List[Dict[str, float]]] = {}
    for exp in experiments:
        metrics_path = runs_dir / exp / "metrics.csv"
        if not metrics_path.exists():
            print(f"[Skip] missing metrics.csv: {metrics_path}")
            continue
        exp_rows[exp] = _read_metrics_csv(metrics_path)

    if not exp_rows:
        raise SystemExit("No usable metrics.csv found.")

    # 生成一个时间戳子目录，避免覆盖
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # 指标 key（来自 train_lprnet_exp.py 的 csv 写入）
    plots = [
        ("train_loss", "Train Loss", "loss", "train_loss.png"),
        ("val_exact_acc", "Val Exact Acc", "accuracy", "val_exact_acc.png"),
        ("val_char_acc", "Val Char Acc", "accuracy", "val_char_acc.png"),
        ("val_ned", "Val NED", "ned", "val_ned.png"),
        ("lr", "Learning Rate", "lr", "lr.png"),
    ]

    for key, title, ylabel, fname in plots:
        series: Dict[str, Tuple[List[int], List[float]]] = {}
        for exp, rows in exp_rows.items():
            xs: List[int] = []
            ys: List[float] = []
            for r in rows:
                if key not in r:
                    continue
                xs.append(int(r.get("epoch", len(xs) + 1)))
                ys.append(float(r[key]))
            series[exp] = (xs, ys)
        _plot_overlay(series, title=f"{title} (compare)", ylabel=ylabel, out_path=out_dir / fname)

    # 单个实验：每个实验一套曲线 + 小结（用于单实验展示）
    if args.per_experiment:
        single_root = out_dir / "per_experiment"
        for exp, rows in exp_rows.items():
            exp_dir = single_root / exp
            exp_dir.mkdir(parents=True, exist_ok=True)

            # 逐 key 画单曲线
            for key, title, ylabel, fname in plots:
                xs: List[int] = []
                ys: List[float] = []
                for r in rows:
                    if key not in r:
                        continue
                    xs.append(int(r.get("epoch", len(xs) + 1)))
                    ys.append(float(r[key]))
                if xs:
                    _plot_single(xs, ys, title=f"{title} ({exp})", ylabel=ylabel, out_path=exp_dir / fname)

            # 写一个 best-epoch 小结（val_exact_acc 最大）
            best_epoch, best = _best_by_key(rows, key="val_exact_acc")
            md_path = exp_dir / "summary_single.md"
            with md_path.open("w", encoding="utf-8") as f:
                f.write(f"# {exp}（单实验小结）\n\n")
                f.write(f"- best_epoch: {best_epoch}\n")
                f.write(f"- best_val_exact_acc: {best.get('val_exact_acc', 0.0):.6f}\n")
                f.write(f"- best_val_char_acc: {best.get('val_char_acc', 0.0):.6f}\n")
                f.write(f"- best_val_ned: {best.get('val_ned', 0.0):.6f}\n")
                f.write(f"- lr_at_best: {best.get('lr', 0.0):.8f}\n")
                f.write(f"- sec_at_epoch: {best.get('sec', 0.0):.2f}\n")

    # 汇总表：以 val_exact_acc 选 best epoch
    summary_rows: List[Dict[str, str]] = []
    for exp, rows in exp_rows.items():
        best_epoch, best = _best_by_key(rows, key="val_exact_acc")
        if best_epoch < 0:
            continue
        summary_rows.append(
            {
                "experiment": exp,
                "best_epoch": str(best_epoch),
                "best_val_exact_acc": f"{best.get('val_exact_acc', 0.0):.6f}",
                "best_val_char_acc": f"{best.get('val_char_acc', 0.0):.6f}",
                "best_val_ned": f"{best.get('val_ned', 0.0):.6f}",
                "lr_at_best": f"{best.get('lr', 0.0):.8f}",
                "sec_at_epoch": f"{best.get('sec', 0.0):.2f}",
            }
        )

    # 稳定排序：baseline 放第一（如果存在）
    summary_rows.sort(key=lambda r: (0 if r["experiment"] == "baseline" else 1, r["experiment"]))

    _write_summary_csv(summary_rows, out_dir / "summary_val_best.csv")

    # 输出一个简短的 markdown 表格，写报告可直接粘
    md_path = out_dir / "summary_val_best.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# LPRNet 实验汇总（Val best by exact_acc）\n\n")
        f.write("| experiment | best_epoch | val_exact_acc | val_char_acc | val_ned | lr | sec |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in summary_rows:
            f.write(
                f"| {r['experiment']} | {r['best_epoch']} | {r['best_val_exact_acc']} | {r['best_val_char_acc']} | {r['best_val_ned']} | {r['lr_at_best']} | {r['sec_at_epoch']} |\n"
            )

    if args.per_experiment:
        print(f"[OK] Wrote compare + per-experiment reports to: {out_dir}")
    else:
        print(f"[OK] Wrote compare reports to: {out_dir} (tip: add --per-experiment)")


if __name__ == "__main__":
    main()
