# 原始实验记录：EXP-023 / QH023

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / rejected as quantum-specific method
- candidate/control：QH-023 / CC-023 / DQ-023 / no-ent / gamma-zero / frozen base
- architecture：Quantum-PEFT Eq.(2) 风格的四路 12q/10q RY-CZ 正交子空间生成器，rank4，共 209 trainable；每次推理 4 次精确 CUDA statevector
- data：EditCLEVR development，test_id/test_noop/test_hard/test_cogent 使用数 0
- result：base/QH/CC exact 68.75/69.14/69.14%；量子未超过等参经典，CI 因果门失败
- dequantization：独立逐门 tensor-contraction DQ 与 QH 前向/梯度一致，限制所有“不可经典模拟”表述
- decision：当前 QH023 停止，不追加已揭盲数据上的超参搜索
