# 原始实验记录：EXP-013 / QH015

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / reject_or_mutate
- parent_experiment：EXP-011 token-specific rank-64 内容路径 + EXP-012 prompt-only 问题锚点
- placement：visual pooler output；每图一次 8q VQC 生成 rank-64 gate，调制每个 token 自己的低秩内容特征
- candidates：QH-015、CC-015、QH-015-no-ent；每组 738,481 参数
- static：toy QH/CC/no-ent 均 2,193 参数；等参、全梯度、每图一次、缺锚点/CPU 拒绝通过；有限差分绝对误差 8.85e-9/1.26e-8/3.04e-8
- causal plan：若主质量和视觉先过门，使用同一 checkpoint 做跨样本 shuffled-question-anchor，再运行 no-ent
- data：沿用 DS-002-hard-v1 前 256 条与固定 validation/counterfactual；不因 QH-014 结果改变学习率或 evaluator
- smoke：4 train / 8 eval / 4 counterfactual；冻结主干梯度张量数 0
- TextVQA：基座/QH/CC soft accuracy 84.1406%/83.3594%/83.2813%；QH 相对基座 -0.7813pp，95% CI [-2.3438, 0]pp
- quantum attribution：QH 相对 CC +0.0781pp，95% CI [0, 0.2344]pp，P(delta>0)=0.6338；exact-any 相同，不能支持量子特异收益
- visual dependency：QH 原图-空白图增益相对基座 -0.4688pp，95% CI [-1.4063, 0]pp
- controller：`paired_quality_regressed`、`multimodal_gain_regressed`；decision `reject_or_mutate`；不运行 shuffled anchor/no-ent
