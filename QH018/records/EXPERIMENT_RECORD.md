# 原始实验记录：EXP-016 / QH018

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / quantum branch rejected after causal intervention
- parent_experiment：EXP-015
- candidates：QH-018 为经典 base route + 中心化量子保真度校正；CC-018 为双经典 route；QH-018-no-ent 只删除残差线路 CNOT
- residual gate：可学习标量初值 0.05，校正加在经典 route logit 上，允许训练关闭或改变方向
- data：DS-005 train pool 的 image-disjoint development holdout；256 个训练图像与 128 个评估图像重叠 0；不是正式 validation/test
- promotion：QH 同时相对 base 和 CC 通过才追加 no-ent；否则拒绝量子校正，但单独记录共同结构是否有效
- seed20260829：base/QH/CC 92.9688%/93.7500%/92.1875%；QH-base +0.7813pp、QH-CC +1.5625pp，但两个 95% CI 均跨 0
- controller：无硬失败，`retain_without_promotion`；已在结果后冻结结构并预注册另外两个初始化 seed 与 no-ent 因果控制
- three-seed QH：93.7500/94.5313/93.7500%，平均相对 base +1.0417pp，95% CI [-0.5208, 2.6042]pp；平均相对 CC +0.2604pp，CI [-1.0417, 2.0833]pp
- three-seed quality decision：`retain_without_promotion`；质量均值达到 1pp，但区间、CC 归因、类别数、资源对照和 causal controls 均未过门
- checkpoint audit：纠缠版三 seed 的 mix 均从 0.05 上移至 0.0540～0.0596；no-ent 仅在 0.0489～0.0511 波动
- branch-off：加载相同 checkpoint、不重训、仅令 mix=0；三 seed enabled-minus-disabled 为 0/0/-1.5625pp，层级均值 -0.5208pp、95% CI [-1.8229, 0]pp、P(enabled better)=0
- final decision：`reject_or_mutate`；量子残差因果门禁失败，保留共同的经典 relation trunk，不再在该 development holdout 调参
