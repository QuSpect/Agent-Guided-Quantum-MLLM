# 原始实验记录：EXP-032 / QH032

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / train-only screen failed / no formal holdout consumed
- parent：EXP-031；literature basis：QISA-A 的 amplitude encoding + sparse Pauli observable readout
- mutation：保持 layers55/59 shared-SVD 真替代不变；10q RX/RY frame + ring CNOT 生成 60 个 X/Z/XX/ZZ 期望值，门控 Pauli-vector residual
- controls：同 82 参数 classical quadratic-form CC032、no-ent、独立 tensor-axis DQ、alpha-zero own-zero
- parameter audit：20 circuit angles +60 observable weights +2 layer gamma =82 trainable；全模型净减 3,145,646
- engineering：QH/DQ forward、input/theta/alpha gradients 最大差 5.96e-8/1.19e-7/2.24e-8/2.98e-7；27B 两步 smoke simulator calls4、冻结梯度0
- data：全新 WikiText-2 train-only 256 train +64 held blocks；validation/test rows0；lock SHA `113a64757d2a382e8b0072d5c1a054ed242ea8097fb7ebecfa63e62b7d4f6ee8`
- result：base/shared/QH/CC token-NLL 2.382818/2.383211/2.383398/2.383419；QH−shared +0.000187；QH−CC -0.000021
- decision：final-hidden MSE 下降没有转成 token-NLL；不消耗 C4，固定结构只允许 QH033 改训练目标
