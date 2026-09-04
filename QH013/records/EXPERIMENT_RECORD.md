# 原始实验记录：EXP-011 / QH013

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / reject_or_mutate
- parent_experiment：EXP-007 / QH-009，受 MEDQUA 稀疏量子支路启发
- placement：visual pooler output；经典内容路径覆盖全部 token，每图 novelty-top-8 进入 8q VQC
- candidates：QH-013、CC-013、QH-013-no-ent；每组 697,449 参数
- static：等参、路由计数、全梯度、CPU 拒绝通过；核心有限差分绝对误差 QH/CC/no-ent 为 2.05e-10/2.60e-8/6.12e-9
- smoke：4 train / 8 eval / 4 counterfactual 成功；mean step 2.532s；冻结参数梯度张量数 0
- data：不复用随机 256 课程；扫描 TextVQA train/0000 基座 soft<0.6 的困难样本，validation/test read=0，QH/CC 共享不可变索引
- hard curriculum：1,899 train-only 合格样本中筛出 285 条；基座 soft/exact-any 84.4813%/89.0469%；正式训练共享前 256 条
- exclusive adapter-only：QH-009 all-token/QH-013 top-8/QH-014 one-per-image P50 为 56.42/56.96/59.49ms；峰值 82.6/61.5/54.3MB。稀疏化降显存但未降当前内核时延
- classical controls：CC-013/CC-014 P50 8.10/4.56ms，说明 GPU 精确状态向量仍有显著工程代价
- TextVQA：基座/QH-013/CC-013 soft accuracy 84.1406%/83.3594%/83.2813%；QH 相对基座 -0.7813pp，95% CI [-2.3438, 0]pp
- quantum attribution：QH 相对 CC +0.0781pp，95% CI [0, 0.2344]pp，bootstrap P(delta>0)=0.6338；exact-any 相同，证据不足
- visual dependency：QH 与基座原图-空白图增益均为 84.6875pp；视觉没有回归，但也没有挽救主质量
- controller：`paired_quality_regressed`；decision `reject_or_mutate`；主质量门失败，不运行 no-ent
