# 原始实验记录：EXP-002 / QWEN38_BASE

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete
- 模型：Qwen/Qwen3.8-27B
- 目标：锁定支持Qwen3.8的Transformers revision；4卡加载；单图推理；显存/TTFT基线
- 结果：HF revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`；18 个权重分片与 tokenizer SHA-256 精确；4 卡 BF16 加载、图文推理和文本回归通过
