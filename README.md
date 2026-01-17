# 课程作业：LPRNet 车牌识别（CCPD，LPRNet-only）

本仓库已精简为 **只包含 LPRNet 识别部分**，用于完成《期末大作业.pdf》要求的实验研究。

核心目标：在可控训练时间内（你筛选后的 100% 数据建议 ≤2 小时），完成并对比三组实验：
- **A：不同网络规模**：`--width-mult`（例如 0.75 / 1.0 / 1.25）
- **B：不同比例训练数据**：`--subset-ratio`（例如 0.1 / 0.3 / 1.0）
- **C：不同学习率策略**：`--lr-scheduler`（fixed / step / cosine）

输出物：
- `runs/lprnet/<name>/metrics.csv`：每个 epoch 的 loss/指标/lr/time
- `runs/lprnet/<name>/tb/`：TensorBoard 日志


## 1. 环境准备

本 README 默认你已经**进入了自己的 Python 环境**（conda / venv 均可）。

示例：

```bash
conda activate <your_env>
```

### 1.1 安装依赖

目的：安装训练/评测/数据裁剪需要的最小依赖。

```bash
pip install -r requirements.txt
```

### 1.2 检查 GPU 是否可用

目的：确认 PyTorch 能用 CUDA（后面用 `--device cuda --amp` 才会生效）。

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```


## 2. 准备数据（从 CCPD2019/2020 生成裁剪车牌小图）

你给的数据路径示例：`/home/aaa/learn/sight/CCPD2019` 与 `.../CCPD2020`。

这里的核心逻辑是：
- CCPD 图片文件名里自带 **车牌框 bbox** 和 **车牌字符标签**
- 本脚本直接解析文件名得到 bbox，然后用 OpenCV 从原图裁剪出车牌小图（再 resize 成 94×24）
- 生成的裁剪小图会被保存到 `data/ccpd_lpr/{train,val,test}`，供 LPRNet 训练/评测使用

### 2.1（推荐）先 dry-run 检查能否正确解析

目的：不生成任何图片，只检查你的 CCPD 路径与文件名解析是否正常。

```bash
python tools/prepare_lpr_data.py --ccpd-root /home/aaa/learn/sight --out-root data/ccpd_lpr --dry-run
```

### 2.2（推荐）先生成“小规模冒烟数据”（跑通流程用）

目的：快速得到一份小数据集，便于你用 GPU 先把训练/评测跑通。

```bash
python tools/prepare_lpr_data.py --ccpd-root /home/aaa/learn/sight --out-root data/ccpd_lpr \
   --seed 0 \
   --blue-subdir ccpd_base \
   --blue-max-train 4000 --blue-max-val 400 --blue-max-test 400 \
   --green-max-train 0 --green-max-val 0 --green-max-test 0
```

产物：
- `data/ccpd_lpr/train/`、`data/ccpd_lpr/val/`、`data/ccpd_lpr/test/` 下会出现裁剪后的车牌小图

### 2.3（可选）生成“正式实验数据”（写报告用）

目的：生成更大的数据集用于 A/B/C 三组对比实验。

建议做法：如果你要重新生成一套数据，先清空旧目录，避免不同规模的数据混在一起。

```bash
rm -rf data/ccpd_lpr
```

然后生成正式数据（下面是一个起点，你可以按训练耗时再调大/调小 max_*）：

```bash
python tools/prepare_lpr_data.py --ccpd-root /home/aaa/learn/sight --out-root data/ccpd_lpr \
   --seed 0 \
   --blue-subdir ccpd_base \
   --blue-max-train 60000 --blue-max-val 5000 --blue-max-test 5000 \
   --green-max-train 0 --green-max-val 0 --green-max-test 0
```

## 3. 训练（实验版）

本节按“课堂实验手册”的写法组织：每个实验只改变一个自变量，其他条件尽量保持一致，保证对比公平。

### 3.0 统一说明（所有实验通用）

**脚本**：`tools/train_lprnet_exp.py`

**输出目录**：每次训练都会生成 `runs/lprnet/<name>/`，其中：
- `runs/lprnet/<name>/weights/best.pth`：验证集整牌准确率（exact_acc）最高的权重
- `runs/lprnet/<name>/weights/last.pth`：最后一个 epoch 的权重
- `runs/lprnet/<name>/metrics.csv`：每个 epoch 的训练损失/验证指标/lr/耗时（写报告表格用）
- `runs/lprnet/<name>/tb/`：TensorBoard 曲线（写报告截图用）

**GPU 训练推荐参数**：
- `--device cuda`：启用 GPU
- `--amp`：混合精度（更快、更省显存）

### 3.1 预备实验：GPU 冒烟（保证能跑）

**目的**：确认显存、速度、训练流程和日志输出都正常。

**控制变量**：使用小数据集（第 2.2 节生成），训练轮数很少。

**运行命令**：

```bash
python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name smoke_gpu --epochs 2 --device cuda --amp
```

**你要观察什么**：
- 终端每个 epoch 会打印：loss、val_exact/val_char/val_ned、lr、耗时
- `runs/lprnet/smoke_gpu/metrics.csv` 是否生成

### 3.2 基准实验：Baseline（后续对比都以它为参照）

**目的**：固定一套默认设置，作为 A/B/C 三组对比的基准。

**控制变量**：数据集固定为同一套“正式实验数据”（第 2.3 节）；训练轮数固定。

**运行命令**：

```bash
python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name baseline --epochs 60 --device cuda --amp
```

**记录内容（写报告）**：
- 取 `runs/lprnet/baseline/metrics.csv` 中验证集最好的 `val_exact_acc`（可写“best val”）
- 用第 4 节脚本评测 test，记录 test 的 `exact_acc/char_acc/ned`

### 3.3 实验 A：不同网络规模（width_mult）

**目的**：研究模型容量变化对识别性能与训练速度的影响。

**控制变量**：数据集不变、epoch 不变、学习率策略不变（使用默认 cosine 或保持与 baseline 一致）。

**自变量**：`--width-mult`（例如 0.75 / 1.0 / 1.25）。

**运行命令（一步一个命令）**：

```bash
python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name A_w075 --width-mult 0.75 --epochs 60 --device cuda --amp
```

```bash
python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name A_w100 --width-mult 1.00 --epochs 60 --device cuda --amp
```

```bash
python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name A_w125 --width-mult 1.25 --epochs 60 --device cuda --amp
```

**报告怎么写**：
- 表格对比三次训练的 best val 与 test 指标
- 结合 `metrics.csv` 的 `sec`（每 epoch 用时）讨论“模型更大是否更慢、是否更准”

### 3.4 实验 B：不同比例训练数据（subset_ratio）

**目的**：研究训练数据量变化对泛化性能的影响。

**控制变量**：模型规模固定（建议 width_mult=1.0，即默认）；epoch 不变；学习率策略不变。

**自变量**：`--subset-ratio`（例如 0.1 / 0.3 / 1.0）。

**运行命令（一步一个命令）**：

```bash
python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name B_r10 --subset-ratio 0.1 --epochs 60 --device cuda --amp
```

```bash
python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name B_r50 --subset-ratio 0.5 --epochs 60 --device cuda --amp
```

```bash
python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name B_r100 --subset-ratio 1.0 --epochs 60 --device cuda --amp
```

**报告怎么写**：
- 画/截取三组的验证集 exact_acc 曲线，讨论“数据更多是否更稳、更高”

### 3.5 实验 C：不同学习率策略（lr_scheduler）

**目的**：研究学习率策略对收敛速度与最终性能的影响。

**控制变量**：模型规模固定（默认 1.0）；数据比例固定（建议 1.0）；epoch 不变。

**自变量**：`--lr-scheduler`（fixed / step / cosine）。

**运行命令（一步一个命令）**：

```bash
python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name C_fixed --lr-scheduler fixed --epochs 60 --device cuda --amp
```

```bash
python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name C_step --lr-scheduler step --epochs 60 --device cuda --amp
```

```bash
python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name C_cosine --lr-scheduler cosine --epochs 60 --device cuda --amp
```

**报告怎么写**：
- 对比早期 epoch 的提升速度（收敛快慢）
- 对比最终 best val 与 test 指标

## 4. 评测（与旧权重对比）

评测脚本会在 `data/ccpd_lpr/test` 上输出三项指标：
- `exact_acc`：整牌准确率（最重要）
- `char_acc`：字符级准确率
- `ned`：归一化编辑距离（越接近 1 越好）

### 4.1 评测旧权重（对照）

```bash
python tools/eval_lprnet.py --data-root data/ccpd_lpr --weights weights/lprnet_best.pth
```

### 4.2 评测你训练得到的权重

```bash
python tools/eval_lprnet.py --data-root data/ccpd_lpr --weights runs/lprnet/baseline/weights/best.pth
```

## 6. 结果可视化（曲线 + 汇总表）

本仓库提供两种可视化方式：

### 6.1 TensorBoard（看单个实验的曲线）

**目的**：查看某个实验的 loss/指标/lr 随 epoch 的变化曲线。

**命令**：

```bash
tensorboard --logdir runs/lprnet --port 6006
```

打开浏览器访问：`http://127.0.0.1:6006`。

### 6.2 自动生成对比图与汇总表（看 baseline 与 A/B/C 差异）

**目的**：把多个实验的曲线画在同一张图上，并自动生成一张“best val 指标”汇总表，便于写报告。

1) 先确保你已经跑完 baseline 与对比实验（例如 A_w075/A_w100/A_w125）。

2) 运行可视化脚本（自动发现 runs/lprnet 下的所有 metrics.csv）：

```bash
python tools/visualize_results.py --runs-dir runs/lprnet --out-dir runs/lprnet/reports
```

如果你还需要“每个实验单独的曲线图”（用于逐个实验展示/截图），加上 `--per-experiment`：

```bash
python tools/visualize_results.py --runs-dir runs/lprnet --out-dir runs/lprnet/reports --per-experiment
```

3) 输出位置：脚本会生成一个时间戳目录，例如：
- `runs/lprnet/reports/20260104_153000/train_loss.png`
- `runs/lprnet/reports/20260104_153000/val_exact_acc.png`
- `runs/lprnet/reports/20260104_153000/summary_val_best.csv`
- `runs/lprnet/reports/20260104_153000/summary_val_best.md`

如果开启了 `--per-experiment`，还会额外生成：
- `runs/lprnet/reports/20260104_153000/per_experiment/<exp>/train_loss.png` 等单实验曲线
- `runs/lprnet/reports/20260104_153000/per_experiment/<exp>/summary_single.md`（单实验小结）

4) 如果你只想对比某几个实验（指定实验名，逗号分隔）：

```bash
python tools/visualize_results.py --runs-dir runs/lprnet --out-dir runs/lprnet/reports --experiments baseline,A_w075,A_w100,A_w125
```

> 说明：车牌识别是序列识别任务，本仓库不使用检测任务的 mAP；核心指标是 exact_acc / char_acc / NED。


## 5. 你下一步怎么做（推荐顺序）

下面是“从零到写报告”的实验手册执行顺序（每一步只做一件事）：

1) dry-run 检查：

```bash
python tools/prepare_lpr_data.py --ccpd-root /home/aaa/learn/sight --out-root data/ccpd_lpr --dry-run
```

2) 生成小数据：

```bash
python tools/prepare_lpr_data.py --ccpd-root /home/aaa/learn/sight --out-root data/ccpd_lpr \
   --seed 0 --blue-subdir ccpd_base --blue-max-train 4000 --blue-max-val 400 --blue-max-test 400 \
   --green-max-train 0 --green-max-val 0 --green-max-test 0
```

3) GPU 冒烟训练（2 epoch）：

```bash
python tools/train_lprnet_exp.py --data-root data/ccpd_lpr --name smoke_gpu --epochs 2 --device cuda --amp
```

4) 评测 smoke_gpu：

```bash
python tools/eval_lprnet.py --data-root data/ccpd_lpr --weights runs/lprnet/smoke_gpu/weights/best.pth --device cuda
```

5) 生成正式数据（可选，写报告用）：

```bash
rm -rf data/ccpd_lpr
```

```bash
python tools/prepare_lpr_data.py --ccpd-root /home/aaa/learn/sight --out-root data/ccpd_lpr \
   --seed 0 --blue-subdir ccpd_base --blue-max-train 60000 --blue-max-val 5000 --blue-max-test 5000 \
   --green-max-train 0 --green-max-val 0 --green-max-test 0
```

6) baseline + A/B/C 实验：按第 3 节命令分别跑即可。

7) 画曲线：

```bash
tensorboard --logdir runs/lprnet --port 6006
```

