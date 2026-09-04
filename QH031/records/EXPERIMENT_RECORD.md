# 原始实验记录：EXP-031 / QH031

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / compressed replacement retained / quantum-specific promotion failed
- parent：EXP-026；literature basis：Basis Sharing（ICLR 2025）与 QASA 单 value-projection 结果
- layer surgery：联合分解 Qwen3.8-27B full-attention layers 55/59 的两个 `v_proj`；删除 10,485,760 个 dense 参数
- replacement：共享 frozen down `5120→1024`，每层独立 frozen up `1024→1024`，中间共享 10q depth-2 exact-statevector core；QH/CC 各 80 个 core 参数及 2 个 layer gamma
- parameter audit：部署替代 7,340,114 参数，全模型净减 3,145,646；仅 82 参数可训练；每次完整前向量子模拟器调用 2 次
- selection：仅在未使用 WikiText-2 train blocks 上筛 layer pair/rank；选择 layers 55/59、rank1024；train-only base/candidate NLL 2.432659/2.433209
- engineering gates：QH 与独立逐门 DQ forward/input-grad/theta-grad 最大差 7.15e-7/4.77e-7/4.77e-6；CPU 量子入口拒绝；27B smoke frozen gradients 0
- preregistration：全新 WikiText train 1,024 blocks + 全新 C4 validation 64 blocks；与此前登记 blocks 重叠 0；lock SHA-256 `7491a0f8f5cb76756840adec62c4624a6595b04295af262a0e0cd4e16955bf14`
- formal metrics：base/shared-SVD/QH031/CC031 NLL = 2.765399/2.765708/2.765405/2.765615；QH point estimate 恢复 shared-SVD 相对 base 的约 98.0% 损失，并点胜 CC 0.000210
- paired CI：shared−base +0.000308，95% CI [-0.000474,+0.001061]；QH−base +0.000006，CI [-0.000800,+0.000783]；QH active−zero -0.000302，CI [-0.000746,+0.000129]；QH−CC -0.000210，CI [-0.000640,+0.000206]
- causal controls：QH no-ent NLL 2.765451；QH−no-ent -0.000046，CI [-0.000521,+0.000416]；独立 DQ NLL 2.765468；所有量子特异区间门失败
- optimization：QH/CC 各 1,024 steps、lr3e-3；QH/CC mean student step 0.311/0.258 s；QH simulator calls 2,048、CC 0；冻结梯度均 0
- decision：保留 layers55/59 shared-basis 作为目前压缩量最大的语言质量非劣真替代；拒绝当前 82 参数核心的量子特异主张；启动结果前锁定的大样本独立确认，不以后验调参
- artifacts：`artifacts/qh031-*.json`、`artifacts/c4_qh031/*`、`src/quantum_qwen38/qh031_shared_basis_replacement.py`
- frozen-checkpoint confirmation：结果前锁定另 512 个未使用 C4 blocks，排除既往 192 blocks；训练步数 0，结构/检查点/超参数变化 0；lock SHA `13274049eff4ce7e4788d67f65a71f923d2bc468e3bd4eae79b38a78c331d02a`
- confirmation metrics：base/shared/QH/CC/no-ent NLL = 2.710647/2.710986/2.711142/2.711128/2.711136
- confirmation CI：QH active−shared-own-zero `+0.000156`，95% CI `[+0.000012,+0.000297]`，P=0.0315；QH−CC `+0.000014`，CI `[-0.000138,+0.000166]`；QH active−no-ent `+0.000006`，CI `[-0.000131,+0.000139]`
- confirmation decision：量子支路在独立扩大样本上显著劣于关闭支路，QH031 quantum core 正式停止；共享基压缩骨架继续保留
