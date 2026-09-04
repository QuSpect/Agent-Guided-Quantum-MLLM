# 原始实验记录：EXP-029 / QH029

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：weight-only feasibility complete / direct ultra-low-bond route not promoted
- source：Quantum Large Language Models via Tensor Network Disentanglers, arXiv:2410.17397v2
- scope：只读取 Qwen3.8 layer7 v_proj 权重；没有读取 train/validation/test 数据
- target：1024×5120 dense projection，5,242,880 参数；测试 balanced、head-aligned、11-site qubit-like 三种张量化顺序
- balanced retained Frobenius energy：bond 1/8/32/64/128/256/512 = 0.0024%/0.1248%/1.5427%/4.8185%/14.1692%/39.1409%/64.9492%
- parameter ledger：bond-512 MPO 2,824,729 参数仍只保留 64.95% 权重能量；通用 13q/10q 复酉矩阵自由度 68,157,438，超过原 dense 约 13 倍
- interpretation：未经张量化预训练的原始 Qwen 权重没有复现论文在已压缩 SmolLM2 上的 36–696 参数残余 MPO 现象；这不排除 activation-aware 微调或从头训练 tensorized base
- decision：拒绝“把 raw Qwen v_proj 直接压成超低 bond MPO”的捷径；QH030 只有在浅局部门线路参数、action fidelity 和 GPU 时延三项同时过门时才可启动
- artifacts：artifacts/qh029-mpo-*.json；scripts/audit_qh029_mpo_disentangler_feasibility.py
