# 原始实验记录：EXP-019 / QH020

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / no quantum advantage over matched control
- architecture：与 QH019 完全相同，改变训练因果结构而非增加 qubit/depth
- stage A：把 QH/CC mixer 全参数精确置零并冻结，训练完全相同的共享 trunk 256 步
- stage B：冻结 trunk，在相同 hard-256 上只训练 QH 或 CC 各自严格 12 个 mixer 参数 256 步
- rationale：避免 823,745 个共享参数吸收绝大部分优化信号，让量子—经典差直接来自小核；对应分阶段组合学习文献的项目内 mutation，不宣称复现
- data：DS-005-train-image-holdout-v3，256 题/256 图，与训练和前两开发切片图像重叠 0；结果前冻结
- formal gate：先 smoke；再单 seed base/QH/CC。QH 必须同时高于 base 和 CC 且视觉不回归，才运行 no-ent/zero-mixer/多 seed
- integration smoke：Stage A/B 各 4 步，参数数 823,745/12，Stage B 梯度范数均值 8.80e-4，冻结主干梯度 0；8 eval/4 counterfactual 成功
- formal result：base/QH/CC exact 为 92.1875/94.1406/94.1406%；QH-base +1.9531pp，95% CI [-0.7813, 5.0781]pp；QH-CC 0，95% CI [-1.1719, 1.1719]pp
- visual：QH 相对 base +1.5625pp；QH 相对 CC +0.7813pp，但只覆盖 128 个反事实，且主质量没有优势
- attribution caveat：QH/CC Stage A 都将 mixer 置零，但量子 `sqrt→Born→renorm` 与经典 `log→softmax` 的浮点前向不是逐位同一函数，二者 loss 从第 2 步开始分叉，故不能把最终同分解释为严格共享 trunk 下的 mixer 比较
- decision：量子收益门失败；不运行原计划的 no-ent/zero/more-seed，转入 EXP-020 的精确共享主干因果修复
