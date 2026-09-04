# 原始实验记录：EXP-026 / QH026

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / compression retained / quantum-specific promotion failed
- parent：EXP-025
- mutation：量子核 depth 2→3；每层重新编码；ring→offset 1/2/4 蝶形纠缠；输入和读出增加无参数 RMS 归一化
- unchanged：删除 layer7 完整 v_proj、冻结 rank-512 SVD 主干、10q CUDA 精确态矢模拟、final-hidden priming、等总参数经典控制
- parameter audit：删除 5,242,880；部署替代 3,258,459；27B 全模型净减 1,984,421；QH/CC 各 112,731 trainable
- static gate：QH 与独立 DQ 前向最大差 3.23e-6；输入/角度梯度相对 L2 1.72e-6/1.59e-6；CPU 量子入口拒绝；QH/DQ/CC/no-ent 均 90 个 core 参数且全梯度非零
- data lock：全新 WikiText-2 train 1,024 blocks + 全新 C4 validation 64 blocks；与此前已登记 train/C4 blocks 重叠均为 0；test 使用 0
- lock：artifacts/qh026-butterfly-lock.json；SHA-256 aba5f54bbf9b9b5004a0d246fa7f0a8883ce8668d5dc0176253e4c928930ae7e
- smoke：QH/CC final-hidden loss 0.000943/0.000506；单步 0.923/0.757 s；QH simulator call 1、CC 0；冻结梯度均 0
- formal metrics：base/SVD/QH/CC NLL 2.644951/2.645698/2.644988/2.645103
- paired CI：QH−base +0.000037，95% CI [-0.000817,+0.000913]；QH active−zero -0.000711，CI [-0.001616,+0.000199]；QH−CC -0.000115，CI [-0.000688,+0.000461]
- gates：SVD/QH compression noninferiority 与净减参数通过；strict-base、active-zero、equal-parameter CC 三个严格门失败
- training：QH/CC 1,024 步平均 student step 0.542/0.462 s；QH 1,024 simulator calls，CC 0；冻结梯度均 0
- latency：64×256-token eval mean sequence 0.1763 s（QH）/0.1618 s（CC）/0.1726 s（base）；不声明模拟器加速
- decision：保留 depth3/reupload/butterfly 作为有利点方向和压缩非劣证据，但拒绝量子特异晋级；下一轮改变表示/目标，而不是继续同一电路堆深度
- analysis artifact：artifacts/c4_qh026/c4-qh026-paired-analysis.json
