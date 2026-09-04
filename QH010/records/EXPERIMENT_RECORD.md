# 原始实验记录：EXP-008 / QH010

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / not promoted / exactly dequantizable
- source：arXiv:2605.05914；Qwen3.8 适配与控制组为本项目实现
- placement：`language_model.layers.7.self_attn.v_proj` 输入；1,280 个独立 4×4 块
- candidates：QH-010（SO(4) Cayley）、CC-010（等参对称非幺正）、QH-010-no-ent（SU(2)⊗SU(2)）
- 参数：三者均 7,680；默认严格恒等初始化；主干冻结
- simulator：CUDA exact statevector，complex64；CPU 拒绝；训练与推理都走同一适配器
- static：三组有限、全梯度、参数相等；有限差分绝对误差均小于 3e-6
- full-model smoke：4 train / 8 eval 成功；mean step 2.068s；冻结参数梯度张量数 0
- TextVQA：QH-010 与冻结基座 soft accuracy 同为 84.1406%；视觉增益相对基座 -0.4688pp；控制器硬失败 `multimodal_gain_regressed`
- QH vs CC：+0.0781pp，95% CI [0, 0.2344]pp；CC 自身回归，不能解释为有效量子信号
- exclusive adapter-only：QH/DQ/CC P50 7.24/8.54/5.67ms；QH vs CC 1.277×；峰值 103.0/77.6/61.7MB
- block geometry：unitarity max error 2.09e-7；operator rank 4 的块 1,277/1,280，但纠缠能力中位仅 1.03e-4
- WikiText：基座/QH/CC token-NLL 2.389055/2.388876/2.388940；QH 相对基座改善 0.000179，95% CI [-0.000329, 0.000697]；相对 CC 改善 0.000064，95% CI [-0.000513, 0.000641]
- 训练：256×256-token、单 seed；QH/CC mean step 0.968/0.969s（共享负载），冻结参数梯度张量数均为 0
- decision：TextVQA 与 WikiText 均未过门；不追加 no-ent、不扩大同一筛选；推进视觉 prefill placement QH-011
