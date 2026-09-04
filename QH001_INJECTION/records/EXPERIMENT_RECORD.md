# 原始实验记录：EXP-003 / QH001_INJECTION

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete
- model：Qwen3.8-27B，27,356,728,560 base parameters
- candidates：QH-001b、QH-003b、QH-006e、QH-006d
- 结果：四方案均在真实视觉特征上有非零梯度；文本路径 token 完全一致；QH-006e/d 恒等注入逐元素零回归
- artifacts：`full_model_injection_validation.json`、`full_model_qh006_validation.json`、`qh001b_full_model_smoke_train.json`
