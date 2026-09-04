# 原始实验记录：EXP-015 / QH017

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / reject_or_mutate
- parent_experiment：EXP-014
- placement：visual pooler output；量子态保真度是 question-to-six-relations softmax 的直接输入
- candidates：QH-017、CC-017、QH-017-no-ent；三路预期严格等参
- simulator：CUDA FP32/complex64 精确状态向量；4q depth-2；每图 1 query + 6 key；CPU 路径禁止
- data：DS-005-v1 validation 中未被 EXP-014 使用的完整 128 题补集；与先前 eval 重叠 0；索引已在任务结果前冻结
- promotion：同一补集上 QH 同时不低于 frozen base、优于等参数 CC，且图像增益不退化；通过后才运行 no-ent、多 seed 与 CoGenT
- static：三路 toy 各 3,209 参数；全部梯度、有限差分、保真度/概率范围、CPU 拒绝通过
- result：base/QH/CC exact 88.2813%/85.9375%/92.1875%；QH-base -2.3438pp；QH-CC -6.2500pp，95% CI [-10.9375, -2.3438]pp
- decision：拒绝纯量子保真度路由；保留 CC017 的 +3.9063pp 探索性结构信号用于新 mutation
