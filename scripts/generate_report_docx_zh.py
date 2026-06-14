#!/usr/bin/env python3
"""Generate the Chinese Word report for Task 2."""

from __future__ import annotations

import generate_report_docx as base


def document_xml_zh() -> str:
    body: list[str] = []
    body.append(base.p("题目二：基于 LeRobot 的 ACT 策略跨环境泛化实验报告", "Title"))
    body.append(base.p("GitHub 仓库链接：TODO：填写公开 GitHub Repository 链接"))
    body.append(base.p("模型权重链接：TODO：填写 exports/task2_act_weights.tar.gz 的网盘下载链接"))
    body.append(base.p("模型权重压缩包：exports/task2_act_weights.tar.gz"))

    body.append(base.heading("1. 任务背景"))
    body.append(base.p(
        "本实验研究具身智能视觉-动作策略在不同环境之间的泛化能力。实验采用 LeRobot 框架中的 "
        "Action Chunking Transformer（ACT）策略，在 CALVIN 环境 A 上训练单环境模型，并在 "
        "CALVIN 环境 A、B、C 混合数据上训练多环境模型。随后将两个模型在未见过的环境 D 上进行 "
        "zero-shot 测试，通过动作误差比较两种训练方式的跨环境泛化效果。"
    ))

    body.append(base.heading("2. 数据集与划分"))
    body.append(base.p(
        "训练数据采用 CALVIN ABC-D 数据集的 LeRobot v3.0 格式。环境 A、B、C 的 episode 划分由 "
        "官方 CALVIN task_ABC_D 元数据恢复得到。环境 D 测试数据从 ABC-D 数据集中第 17870 个 "
        "episode 开始，该位置紧接 17870 个 ABC 训练 episode 之后，因此作为未见环境 D 的测试段。"
    ))
    body.append(base.p(
        "本实验没有直接使用完整的原始 CALVIN 压缩包，而是基于 LeRobot v3.0 数据集和官方 CALVIN "
        "元数据构建固定 episode 子集。训练阶段使用环境 A 的 512 个 episode，以及环境 A、B、C 各 "
        "512 个 episode 组成的混合训练集；验证和 D 环境 zero-shot 测试均使用 64 个 episode。"
        "本文中的训练曲线、评估指标和分析均基于该固定子集，相关数据文件由 LeRobot 按需读取和缓存。"
    ))
    body.append(base.table([
        ["数据划分", "选择方式", "Episode 数量", "用途"],
        ["A train", "仅环境 A", "512", "单环境策略训练"],
        ["ABC train", "A、B、C 各 512 个 episode", "1536", "多环境联合训练"],
        ["A validation", "环境 A 验证 episode", "64", "单环境模型验证"],
        ["ABC validation", "A+B+C 验证 episode", "64", "多环境模型验证"],
        ["D test", "未见环境 D episode", "64", "zero-shot 跨环境测试"],
    ]))

    body.append(base.heading("3. 方法"))
    body.append(base.p(
        "两个实验均使用相同的 LeRobot ACTPolicy 实现、相同网络结构和相同超参数。模型输入包括 "
        "top camera RGB 图像和 15 维机器人状态，输出为 7 维末端执行器动作。ACT 在每个观测处预测 "
        "一段未来动作 chunk，并使用 chunk 内动作的 L1 误差进行训练。"
    ))
    body.append(base.table([
        ["项目", "设置"],
        ["策略模型", "LeRobot ACTPolicy"],
        ["视觉输入", "observation.images.top，200 x 200 RGB"],
        ["状态输入", "observation.state，15 维"],
        ["动作输出", "action，7 维"],
        ["Action chunk size", "100"],
        ["n_action_steps", "100"],
        ["优化器", "AdamW"],
        ["学习率", "1e-5"],
        ["Weight decay", "1e-4"],
        ["Batch size", "4"],
        ["训练步数", "5000"],
        ["Checkpoint", "第 2500 和 5000 step 保存，最终模型位于 checkpoints/last"],
    ]))

    body.append(base.heading("4. 训练结果"))
    body.append(base.p(
        "两个模型均完成 5000 step 训练，并在训练过程中出现明显收敛。A-only 模型的总 loss 从 "
        "71.3845 下降到 0.4770；ABC 多环境模型的总 loss 从 74.2411 下降到 0.4539。最终记录的 "
        "Action L1 loss 分别为 0.1106 和 0.1002。"
    ))
    body.append(base.table([
        ["模型", "训练 episode 数", "Step 1 loss", "Step 5000 loss", "Step 5000 Action L1"],
        ["ACT trained on A", "512", "71.3845", "0.4770", "0.1106"],
        ["ACT trained on A+B+C", "1536", "74.2411", "0.4539", "0.1002"],
    ]))
    body.append(base.p("图 1：两个 ACT 模型训练过程中的 Action L1 loss 曲线。"))
    body.append(base.image())

    body.append(base.heading("5. 评估结果"))
    body.append(base.p(
        "评估指标为 ACT 输出 action chunk 与数据集中真实 action chunk 之间的平均 L1 误差。"
        "Success Proxy 定义为平均 L1 误差低于 0.05 的样本比例。"
    ))
    body.append(base.table([
        ["模型", "评估 split", "Mean L1", "Median L1", "Std L1", "Success Proxy", "Samples", "Episodes"],
        ["ACT trained on A", "A val", "0.3595", "0.2988", "0.2734", "0.0505", "3845", "64"],
        ["ACT trained on A", "D test", "0.8978", "0.8921", "0.0676", "0.0000", "3796", "64"],
        ["ACT trained on A+B+C", "ABC val", "0.3385", "0.2664", "0.2479", "0.0083", "3723", "64"],
        ["ACT trained on A+B+C", "D test", "0.9090", "0.9008", "0.0738", "0.0000", "3796", "64"],
    ]))

    body.append(base.heading("6. 结果分析"))
    body.append(base.p(
        "从验证集结果看，ABC 多环境模型在 ABC validation 上的 Mean L1 为 0.3385，低于 A-only "
        "模型在 A validation 上的 0.3595，说明在相同 ACT 架构和超参数下，多环境训练没有造成训练 "
        "不稳定，并且在混合来源验证数据上取得了更低的动作误差。"
    ))
    body.append(base.p(
        "在 zero-shot D 环境上，A-only 模型的 Mean L1 为 0.8978，ABC 多环境模型为 0.9090。"
        "这表明本次实验中，多环境训练没有带来明显的 D 环境泛化提升。D 环境与训练环境存在视觉分布 "
        "差异，模型从图像中提取到的状态表征可能发生偏移，使得动作预测误差显著高于验证集。"
    ))
    body.append(base.p(
        "ACT 的动作分块机制可以一次预测多个未来动作，因此在局部时间尺度上有助于保持动作序列的连续性。"
        "但是，当输入图像来自未见环境并出现视觉分布偏移时，当前观测的视觉编码误差会影响整个 action "
        "chunk 的预测。也就是说，action chunking 提供了时间维度上的平滑性和短期稳定性，但不能完全消除 "
        "跨环境视觉差异带来的表征偏差。"
    ))
    body.append(base.p(
        "综合来看，实验完成了 A-only 与 A+B+C 两种训练策略在相同 ACT 设置下的对比。结果说明，多环境 "
        "数据可以改善已见环境组合上的验证表现，但要进一步提升未见 D 环境的 zero-shot 表现，可能还需要 "
        "更强的视觉增强、域随机化或更多环境外观变化的数据覆盖。"
    ))

    body.append(base.heading("7. 可复现命令"))
    body.append(base.table([
        ["命令", "作用"],
        ["python scripts/train_act_calvin.py --config configs/task2_act_calvin.yaml --run-kind a", "训练环境 A 模型"],
        ["python scripts/train_act_calvin.py --config configs/task2_act_calvin.yaml --run-kind abc", "训练 A+B+C 多环境模型"],
        ["bash scripts/task2_eval.sh configs/task2_act_calvin.yaml outputs/task2_act_calvin results/final", "评估两个模型"],
    ]))

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
  xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    {''.join(body)}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


if __name__ == "__main__":
    base.document_xml = document_xml_zh
    base.OUT = base.ROOT / "report" / "task2_act_calvin_report.docx"
    base.write_docx()
