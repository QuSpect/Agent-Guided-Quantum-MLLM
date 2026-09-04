# 原始实验记录：EXP-017 / QH018_NOENT

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / retain_without_promotion
- candidate：QH-018 删除全部 CNOT，其余参数和精确 complex64 状态向量执行不变；等参数控制复用 CC018
- dequantization：product-state，可精确经典因子化；即使有效也只作为工程方案，禁止量子优势表述
- seed20260829：94.5313%，比 base +1.5625pp，图像聚类 bootstrap 95% CI [0, 3.9063]pp；比 entangled QH +0.7813pp，CI 跨 0
- three-seed：94.5313/93.7500/92.9688%；平均相对 base +0.7813pp，95% CI [-1.0417, 2.3438]pp
- attribution：相对 CC 平均 0，95% CI [-2.0833, 2.6042]pp；平均视觉增益相对 base +3.1250pp，CI [0, 6.2500]pp
- decision：`retain_without_promotion`；可作为工程结构备选，但精确可去量子化且无经典控制收益
