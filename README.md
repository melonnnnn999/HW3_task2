# HW3 Task 2: LeRobot ACT on CALVIN

本项目实现了基于 LeRobot 的 ACT（Action Chunking Transformer）策略跨环境泛化实验。实验在 CALVIN 数据集上分别训练两个策略：

- `act_calvin_a`：仅使用环境 A 的数据训练。
- `act_calvin_abc`：使用环境 A、B、C 混合数据训练。

两个模型使用相同的 ACT 架构和超参数，并在未见过的环境 D 上进行 zero-shot 动作误差评估。

## 目录结构

```text
configs/
  task2_act_calvin.yaml          # 正式实验配置
  task2_act_calvin_smoke.yaml    # 小规模连通性测试配置
scripts/
  build_calvin_scene_splits.py   # 根据 CALVIN 元数据生成 A/B/C episode 划分
  check_datasets.py              # 检查 LeRobot 数据集是否可读
  train_act_calvin.py            # 训练 ACT 策略
  evaluate_action_error.py       # 评估 action chunk L1 误差
  task2_train.sh                 # 训练入口脚本
  task2_eval.sh                  # 评估入口脚本
  plot_losses.py                 # 绘制训练曲线
  generate_report_docx_zh.py     # 生成中文 Word 报告
data/
  raw_meta/                      # CALVIN 官方元数据
  splits/                        # A/B/C episode 划分
outputs/
  task2_act_calvin/              # 训练日志与模型 checkpoint
results/
  final/                         # 正式评估结果
plots/
  task2_loss_curve.png           # 训练曲线
report/
  task2_act_calvin_report.docx   # 中文实验报告
exports/
  task2_act_weights.tar.gz       # 两个模型的最终权重压缩包
```

## 环境配置

推荐使用 Python 3.10，并安装 PyTorch、LeRobot 和常用数据处理依赖。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果服务器上已经存在可用的 Python/LeRobot 环境，也可以直接指定解释器：

```bash
export PYTHON=/path/to/venv/bin/python
```

## 数据集

本实验使用以下 LeRobot v3.0 格式 CALVIN 数据：

- 训练数据：`Traly/calvin_abc_d-lerobot`
- D 环境测试数据：`Traly/calvin-task-ABC-D-lerobot-3.0`

实验没有直接使用完整的原始 CALVIN 压缩包，而是基于 LeRobot v3.0 数据集和官方 CALVIN 元数据构建固定 episode 子集。训练阶段使用环境 A 的 512 个 episode，以及环境 A、B、C 各 512 个 episode 组成的混合训练集；验证和 D 环境 zero-shot 测试均使用 64 个 episode。所有训练曲线、评估指标和报告分析均基于该固定子集，数据文件由 LeRobot 按需读取和缓存。

正式配置文件为：

```bash
configs/task2_act_calvin.yaml
```

默认实验设置：

| 项目 | 设置 |
|---|---:|
| A-only 训练数据 | 环境 A，512 episodes |
| ABC 训练数据 | A/B/C 各 512 episodes，共 1536 episodes |
| 验证集 | 64 episodes |
| D 测试集 | 64 episodes |
| 训练步数 | 5000 |
| Batch size | 4 |
| Learning rate | 1e-5 |
| Action chunk size | 100 |

检查数据集是否可读：

```bash
${PYTHON:-python} scripts/check_datasets.py --config configs/task2_act_calvin.yaml
```

如需重新生成 A/B/C episode 划分：

```bash
${PYTHON:-python} scripts/build_calvin_scene_splits.py
```

## 训练

训练仅使用环境 A 的 ACT 策略：

```bash
${PYTHON:-python} scripts/train_act_calvin.py \
  --config configs/task2_act_calvin.yaml \
  --run-kind a
```

训练使用 A+B+C 混合数据的 ACT 策略：

```bash
${PYTHON:-python} scripts/train_act_calvin.py \
  --config configs/task2_act_calvin.yaml \
  --run-kind abc
```

训练完成后，最终模型位于：

```text
outputs/task2_act_calvin/act_calvin_a/checkpoints/last/pretrained_model/
outputs/task2_act_calvin/act_calvin_abc/checkpoints/last/pretrained_model/
```

每个最终模型目录包含：

```text
config.json
model.safetensors
```

## 评估

运行正式评估：

```bash
PYTHON=${PYTHON:-python} bash scripts/task2_eval.sh \
  configs/task2_act_calvin.yaml \
  outputs/task2_act_calvin \
  results/final
```

评估结果汇总：

```bash
cat results/final/task2_summary.md
```

正式结果：

| Run | Split | Mean L1 | Median L1 | Std L1 | Success Proxy | Samples | Episodes |
|---|---|---:|---:|---:|---:|---:|---:|
| act_calvin_a | a_val | 0.3595 | 0.2988 | 0.2734 | 0.0505 | 3845 | 64 |
| act_calvin_a | d_test | 0.8978 | 0.8921 | 0.0676 | 0.0000 | 3796 | 64 |
| act_calvin_abc | abc_val | 0.3385 | 0.2664 | 0.2479 | 0.0083 | 3723 | 64 |
| act_calvin_abc | d_test | 0.9090 | 0.9008 | 0.0738 | 0.0000 | 3796 | 64 |

## 训练曲线

将训练日志转换为 CSV：

```bash
${PYTHON:-python} - <<'PY'
import pandas as pd
from pathlib import Path

for run in ["act_calvin_a", "act_calvin_abc"]:
    src = Path(f"outputs/task2_act_calvin/{run}/train_metrics.jsonl")
    dst = src.with_suffix(".csv")
    pd.read_json(src, lines=True).to_csv(dst, index=False)
    print(dst)
PY
```

绘制 Action L1 loss 曲线：

```bash
${PYTHON:-python} scripts/plot_losses.py \
  --single-env-csv outputs/task2_act_calvin/act_calvin_a/train_metrics.csv \
  --multi-env-csv outputs/task2_act_calvin/act_calvin_abc/train_metrics.csv \
  --output plots/task2_loss_curve.png
```

## 模型权重

打包最终模型权重：

```bash
tar -czf exports/task2_act_weights.tar.gz \
  outputs/task2_act_calvin/act_calvin_a/checkpoints/last/pretrained_model \
  outputs/task2_act_calvin/act_calvin_abc/checkpoints/last/pretrained_model
```

压缩包包含两个最终 ACT 模型：

```text
act_calvin_a/checkpoints/last/pretrained_model/
act_calvin_abc/checkpoints/last/pretrained_model/
```

## 报告

生成中文 Word 报告：

```bash
${PYTHON:-python} scripts/generate_report_docx_zh.py
```

输出文件：

```text
report/task2_act_calvin_report.docx
```

提交 PDF 时，可在 Word 中补充 GitHub 仓库链接和模型权重网盘链接后导出为 PDF。
