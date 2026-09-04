# 原始实验记录：EXP-012 / QH014

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / reject_or_mutate
- parent_experiment：QH-003b 视觉区域锚点；受 MQAdapter 语义锚点思想启发但不是复现
- placement：visual pooler output；训练与推理均从不含答案的 prompt token 构建冻结文本锚点，每图仅一次 8q VQC
- candidates：QH-014、CC-014、QH-014-no-ent；每组 163,889 参数
- static：等参、全部梯度、每图一次调用、缺锚点拒绝、CPU 拒绝通过；有限差分绝对误差 2.37e-8/2.65e-8/1.77e-8
- smoke：4 train / 8 eval / 4 counterfactual 成功；mean step 2.554s；冻结参数梯度张量数 0
- causal plan：主质量与视觉门先通过才运行 shuffled-question-anchor；之后才运行 no-ent，避免在失败候选上扩大筛选
- data：与 QH-013 共享同一 train-only 困难课程和固定 validation row；学习率、步数、评估样本、反事实样本完全匹配
- TextVQA：基座/QH-014/CC-014 soft accuracy 均为 84.1406%，exact-any 均为 88.2813%；主质量逐样本完全相同
- visual dependency：QH 为 83.1250pp，基座/CC 均为 84.6875pp；QH 相对两者 -1.5625pp，95% CI [-4.6875, 0]pp
- controller：`multimodal_gain_regressed`；decision `reject_or_mutate`；不运行 shuffled-question-anchor/no-ent
