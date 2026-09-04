# 原始实验记录：EXP-036 / FFN036_AUDIT

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / train-only feasibility / no performance claim
- purpose：在设计新量子替代前，以完整删除每层 `mlp` 的方式测量 Qwen3.8-27B 各 FFN 的功能冗余；不训练、不读取 validation/test
- data lock：12 个全新 WikiText-2 train-only blocks；排除既往 6,980 个索引；SHA `2959133530373191c2ec9f1e0ff69ebc0c281975bfc40b627ef313744b7d2574`
- base：NLL 2.464133；逐层把完整 SwiGLU FFN 输出置零并与 base 做配对比较
- result：删除 layer11/27/15/21/12/13 的点估计 NLL 分别改善 0.019621/0.010754/0.006349/0.004312/0.004304/0.003869；layer11、27 的描述性 95% CI 全在 0 以下
- conservative target：layer31 是“删除后严格变差”的 64 层中损伤最小者，ΔNLL +0.002836；选择它而非删除后偶然变好的 layer11，为替代件留下可测量的恢复目标
- interpretation：Qwen3.8 FFN 冗余明显非单调，不能用“越深越可删”规则；12 个 train blocks 仅用于架构选择，不构成语言质量结论
- artifacts：`artifacts/qh036-ffn-sensitivity-lock.json`、`artifacts/qh036-ffn-sensitivity-audit.json`
