# 原始实验记录：EXP-010 / QH012

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / reject_or_mutate
- parent_experiment：EXP-009 / QH-011
- placement：视觉 merger 输出；仅 visual prefill
- candidates：QH-012、DQ-012、CC-012、QH-012-no-ent；每组 7,680 参数
- topology：320 个 4q/16 振幅寄存器；每个按 (01,23,12,30) 运行四个二量子比特 Cayley 门
- static：修复首轮 einsum 轴冲突后正式通过；QH-DQ 最大输出/梯度误差 9.54e-7/1.49e-8；产品态跨 01|23 熵 QH 0.4803 bit、no-ent 0
- smoke：4 train / 8 eval / 4 counterfactual 成功；mean step 2.067s；冻结参数梯度张量数 0
- governance：首轮错误 `pass` 已撤销并保留失败原因；只有修正后 artifact 可作为控制器输入
- TextVQA：基座/QH-012/CC-012 soft accuracy 84.1406%/83.2813%/84.0625%；QH 相对基座 -0.8594pp，95% CI [-2.5000, 0]pp；相对 CC -0.7813pp，95% CI [-2.3438, 0]pp
- 视觉因果：QH 原图-空白图增益 84.5313pp，相对基座 -0.1563pp，95% CI [-0.4688, 0]pp
- controller：`multimodal_gain_regressed`、`paired_quality_regressed`、`quantum_underperformed_equal_parameter_control`
- decision：reject_or_mutate；不运行 no-ent；更高产品态纠缠没有转化为任务收益
