# 原始实验记录：EXP-009 / QH011

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / retain without promotion
- parent_experiment：EXP-008 / QH-010
- placement：`model.model.visual.merger.output`；仅视觉 prefill，纯文本和 decode token 绕过
- candidates：QH-011、DQ-011、CC-011、QH-011-no-ent；每组 7,680 参数
- smoke：4 train / 8 eval / 4 counterfactual；mean step 2.107s；冻结参数梯度张量数 0；视觉反事实正常
- TextVQA：基座/QH-011/CC-011 soft accuracy 84.1406%/84.1406%/84.0625%；QH 与基座逐样本 soft、exact-any 均完全相同
- 视觉因果：基座/QH-011 原图-空白图增益均为 84.6875pp；CC-011 为 82.5000pp
- QH vs CC：+0.0781pp，95% CI [0, 0.2344]pp；该差异来自 CC 回归，不是 QH 改善
- decision：`retain_without_promotion`；没有硬回归，但质量增量为零；不运行 no-ent，推进 QH-012 跨块 mutation
