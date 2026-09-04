# 原始实验记录：EXP-025 / QH025

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / compression retained / quantum-specific promotion failed
- parent：EXP-024
- mutation：只把蒸馏 loss 从被删 v_proj 的局部输出移到完整 decoder + final norm 的隐藏状态；结构与参数不变
- initialization：QH025/CC025 分别载入 QH024/CC024 checkpoint；hash 在 lock 固定
- train：未用 WikiText-2 train 1,024×256 blocks；AdamW lr3e-4；冻结原 27B
- eval：C4 English validation shard0 固定 prefix 2,048 rows 中锁定 64×256-token blocks
- controls：frozen dense base、SVD512 zero、同 checkpoint QH own-zero、112,701 参数 CC025
- smoke：final-hidden relative MSE 0.006959，gradient norm 0.05086，simulator calls 1，frozen grads 0
- lock：artifacts/wikitext-qh025-end-to-end-lock.json；SHA-256 dac0d3c3ffcdbff438fa1cbdbbaecf13d9789b135f763cc922ba1f8faa3d4f57
- formal metrics：base/SVD512/QH025/CC025 NLL = 2.684419/2.684513/2.683984/2.683863；QH own-zero 与 SVD512 精确相同
- paired CI：QH−base `-0.000434`，95% CI `[-0.001286,+0.000435]`；QH active−zero `-0.000529`，CI `[-0.001571,+0.000543]`；QH−CC `+0.000121`，CI `[-0.000436,+0.000662]`
- gates：SVD 与 QH 均通过 +0.005 compression noninferiority；QH strict-base、active-zero、equal-parameter CC 三个严格门均失败
- training：QH/CC 均 1,024 steps；末窗口 final-hidden loss 0.001625/0.001685；QH 精确模拟器调用 1,024 次，CC 0；冻结梯度均 0
- decision：保留净减 1,984,451 参数的真替代与 end-to-end priming 训练协议；拒绝当前 depth2 QH025 的量子特异主张，进入独立结构 mutation
- analysis artifact：artifacts/c4_qh025/qh025-paired-analysis.json
