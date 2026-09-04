# 原始实验记录：EXP-014 / QH016

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / reject_or_mutate
- parent_experiment：EXP-013；任务层启发为 CCG-VQC 和分阶段组合泛化，但实现为独立 mutation
- placement：visual pooler output；冻结对象 token 汇总为 2x2 区域，6 个有方向关系对经问题 soft router 后回写 token-specific rank-64 内容路径
- candidates：QH-016、CC-016、QH-016-no-ent；每组 820,675 参数
- simulator：QH/no-ent 均为 CUDA FP32/complex64 精确状态向量；6q depth-2；每图 6 个关系状态批量执行；CPU 路径禁止
- static：三组各 2,901 toy 参数且完全等参；全梯度、路由单纯形、关系计数、输入拒绝门通过；有限差分绝对误差均低于 5.4e-8
- smoke：4 train / 8 eval / 4 counterfactual；冻结参数梯度张量数 0；全尺寸参数 820,675
- data：CLEVR v1.0 官方 CC BY 4.0；只抽 functional program 含 `relate` 的 train/validation，test 不读取；train-only base-error curriculum 与固定 validation 分离
- promotion：必须同时满足相对 frozen base 主质量不退化、视觉增益不退化、相对等参 CC 有可重复收益；之后才追加 no-ent 与 CoGenT OOD
- result：base/QH/CC exact 为 87.5000%/86.7188%/86.7188%；QH-base -0.7813pp，图像聚类 bootstrap 95% CI [-3.1250, 1.5625]pp；QH-CC 逐题完全相同，delta 0、CI [0,0]
- visual：QH 原图-空白图增益 43.7500pp，与 base 相同；没有触发视觉回归，但主质量失败
- decision：`reject_or_mutate`；不追加 no-ent/CoGenT
