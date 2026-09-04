# 原始实验记录：EXP-018 / QH019

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / reject_or_mutate
- parent_experiment：EXP-016 的共同经典 relation trunk；不继承已拒绝的量子残差
- candidate：把六个经典 question-relation 先验作为前三量子比特的前六个振幅，使用 identity-near RZZ 相位纠缠后接 RY 产生候选间干涉，Born 概率直接成为最终路由
- controls：CC019 使用同 12 参数经典概率混合器；no-ent 以本地 RZ 替换 RZZ；`theta=0` 是不重训 branch-off
- execution：每图一个 3q/8 振幅精确 complex64 CUDA 状态；只在 visual prefill，decode 不调用
- parameters：QH/CC/no-ent 全尺寸均 823,757；static toy 均 657
- static：三路全梯度、概率单纯形、CPU 拒绝通过；有限差分绝对误差 7.90e-9/2.39e-8/2.15e-8
- 27B smoke：4 train / 8 eval / 4 counterfactual 成功；冻结主干梯度张量数 0；共享环境平均 step 1.17s
- data：DS-005-train-image-holdout-v2；256 图，与训练/QH018 已揭盲图像重叠均为 0；在 static 和任务结果前冻结
- formal gate：seed20260829 只跑 base/QH/CC；QH 必须同时高于 base 和 CC 且视觉不回归，才追加 no-ent、theta=0、更多 seed
- formal result：base/QH/CC exact 93.7500/94.5313/94.9219%；QH-base +0.7813pp，95% CI [-1.1719, 3.1250]pp；QH-CC -0.3906pp，CI [-1.9531, 0.7813]pp
- visual：QH-base +0.7813pp、CI [-3.9063, 6.2500]pp；QH-CC -1.5625pp、CI [-6.2500, 2.3438]pp
- decision：`reject_or_mutate`；QH 只有 5 win/3 loss 相对 base，但相对 CC 为 1 win/2 loss/253 tie，未通过量子归因门，不运行 no-ent/zero/more seeds
