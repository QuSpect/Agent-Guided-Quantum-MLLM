# 原始实验记录：EXP-022 / QH023_PROTO

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：CUDA static + Qwen3.8-27B train-only integration smoke pass / performance not started
- failure parent：QH022 的 12 个低秩基完全固定随机，等参 CC 明显更优；QH023 改成线路学习 `U,V` 正交列空间，不再只学习固定基系数
- architecture：`ΔW = U diag(lambda) Vᵀ`，rank 4；Qwen hidden 5120 精确拆成 4096+1024（12q+10q），U/V 共四个 RY-CZ depth-2 状态向量线路
- parameter audit：12q/10q 单线路按 Quantum-PEFT Eq.(2) 为 56/46 个角；U/V 合计 204，加 lambda 4、gamma 1，总计 209
- runtime constraint：每个 prefill/decode forward 重新运行四个 GPU 状态向量线路；CPU 入口拒绝，不能把 U/V 烘焙成永久经典权重
- controls implemented：QH023、同 209 参数经典 dense-tangent + thin-QR `CC023`、同 209 参数 no-ent；DQ023 独立实现仍是性能试验前硬阻塞项
- static result：三路参数数均 209、六组梯度均有限非零、每次 forward 四次生成器调用、gamma-zero 精确恒等；QH/CC/no-ent 正交误差最大分别 7.75e-7/2.38e-7/5.96e-7，有限差分误差约 1.0e-7/1.5e-7/1.5e-7
- full-model smoke：27,356,728,769 总参数中仅 209 可训练；4×64-token 训练健康步骤和 8 个训练集前向通过；冻结参数梯度张量数 0，8 次前向触发 32 次 simulator call，正交误差 4.17e-7
- data boundary：只读 WikiText train token 作集成健康检查；validation rows=0、test rows=0；loss 只检查 finite，不作候选效果或学习率选择
- failure record：首次专用全模型脚本保留 token 为预处理 `int32`，在 CUDA cross-entropy 入口失败；修正为 `torch.long` 后原配置通过。失败发生在量子模块之外且未生成性能结果
- artifacts：`artifacts/qh023-gpu-validation.json` SHA-256 `cd469746bc44836388927ded5d1cfa38307ffc1a325c8d6685640783964aeaa8`；`artifacts/qh023-full-model-smoke.json` SHA-256 `043b1c125613cf1f87f05467a3c219788b265ecb69b0e2cc725609c7ea07dbd9`
- decision：只通过 S0/S1 工程门，不是有效方案；完成独立 DQ023、锁定 DS-007 新 OOD 和预注册 resource gate 前，禁止性能训练
