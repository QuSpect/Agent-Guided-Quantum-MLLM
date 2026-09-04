# 原始实验记录：EXP-024 / QH024

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / SVD compression retained / quantum residual rejected
- layer surgery：Qwen3.8-27B layer7 full-attention v_proj 从 5,242,880 参数替换为 3,258,429 参数；全模型净减 1,984,451
- replacement：frozen SVD rank512 + 10q depth2 exact-statevector residual；QH/CC trainable 均 112,701
- selection：未见 32-block screen 比较 layer7/31/63 和 rank512/640/768/896；预规则选择 layer7 rank512
- formal data：另 32 个锁定 WikiText validation blocks，train512×256；test rows 0
- metrics：base/SVD/QH/CC NLL 2.352907/2.354116/2.354767/2.354761
- paired CI：SVD−base 95% CI [-0.001122,+0.003316]，通过 +0.005 非劣；QH−zero CI [-0.002131,+0.003851]；QH−CC CI [-0.000752,+0.000767]
- decision：真删层压缩可行；局部 projection-MSE 量子残差没有因果贡献
- artifacts：artifacts/wikitext_qh024/*；artifacts/qh024-gpu-validation.json；artifacts/qh024-full-model-smoke.json
