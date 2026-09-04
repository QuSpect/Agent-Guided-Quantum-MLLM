# 原始实验记录：EXP-021 / QH022

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：CUDA static + Qwen3.8-27B integration smoke pass / performance pending new data
- literature parent：QPA（arXiv:2410.09846 / ICLR 2025）用 QNN+MLP 生成 PEFT 参数，但原方案在训练后移除量子推理；Quantum-PEFT（arXiv:2503.05431）提供全秩、低参数 Pauli 参数化动机。QH022 是项目内适配，不宣称复现
- constraint mutation：为满足用户硬约束，每次模型 forward（prefill 与每个 decode step）都在 GPU 上运行一次 6q complex64 精确状态向量，生成 6 个 Z 与 6 个邻接 ZZ 系数
- placement：Qwen3.8 第 7 个 full-attention block 的 `v_proj` 输入；冻结原投影与全部主干
- adapter：12 个固定随机正交化 rank-1 基组成低秩更新，VQC 只生成 12 个组合系数；训练参数为 24 个门角加 1 个残差 gamma，共 25
- controls：CC022 使用同 24 参数的固定 Fourier 投影生成 12 系数；QH022-no-ent 只删除 CNOT；`gamma=0` 为同检查点 branch-off。三路均为 25 参数
- static gate：CUDA-only、一次 forward 一次生成器调用、三路严格等参、全部参数组梯度非零、中心有限差分误差 <1e-3、branch-off 精确恒等、CPU 量子路径拒绝
- data gate：DS-005 v1-v3 与 TextVQA 已揭盲集均不可再用于性能筛选；在新 OOD/领域分布冻结前只允许静态和 27B 集成冒烟，不报告性能晋级
- claim limit：6q 状态向量可被经典模拟；即使有效也只能主张 simulator-resident parameterization 的经验归纳偏置，不能主张量子加速
- static result：A800 上 QH/CC/no-ent 均为 25 参数且所有参数组梯度非零；有限差分绝对误差 1.77e-6/1.16e-9/1.56e-6；每次 forward 恰好一次生成器调用；CPU 拒绝；gamma-zero 最大差 0
- 27B smoke：真实注入路径 `model.language_model.layers.7.self_attn.v_proj.input`；4×64-token 训练 + 4 个验证序列推理成功；只训练 gamma 1 参数与 theta 24 参数；梯度范数均值 1.24e-5，冻结参数梯度张量数 0
- smoke resource：四卡峰值 allocated 约 11.93/14.53/14.53/14.17GB；共享环境，只作可运行性证据
- artifacts：`artifacts/qh022-gpu-validation.json`（SHA-256 `3b40b58c1f2fff7e751dbc9ef9b3da5b4c512250545ca44e25852db136eaa808`）；`artifacts/wikitext_stage_a/wikitext-qh022-s20260829-n4-l64.json`（SHA-256 `f3166e7df8e17d25925b221b7e691bb9fb0f9c893a47d7a0bdcec6c27596db1d`）
- decision：保留为新家族待测；冒烟 NLL 不用于选型。必须先冻结全新分布，再运行 base/QH/CC，双胜后才追加 no-ent/branch-off/seed
- text complement preregistration：在任何正式结果前冻结 DS-006-WikiText2-unused-v1；128×256 token，排除 QH010 的 64 个已评估块和 QH022 冒烟覆盖的 parent blocks，最终排除 68、重叠 0；锁文件 SHA-256 `24e8d3361b7a7adc43f2a55a06a51b5d07e52a1eeef01245c62e79e00aaab186`
- formal text gate：固定 train-256×256、eval-128×256、seed20260829、lr1e-4；先 base/QH022/CC022。QH token-NLL 必须同时低于两者，才允许任何 no-ent/branch-off/多模态扩展；否则停止 25 参数配置
- formal result：base/QH/CC token-NLL 为 2.43049578/2.43047961/2.43021339；QH-base NLL reduction 1.62e-5，95% CI [-3.76e-4, 4.22e-4]、P=0.5217；QH-CC reduction -2.66e-4，95% CI [-6.78e-4, 1.56e-4]、P=0.1055
- optimization：QH/CC 256 步梯度范数均值 1.50e-5/2.45e-5，冻结梯度 0；gamma 从 0.05 分别变为 0.04730/0.05780。量子支路被轻微减弱，经典支路被增强
- checkpoint audit：QH/CC 检查点均只有 2 张量、25 参数、全有限；SHA-256 `af1e2f8c37ad2144d48f30c4eceb6345a04ce3dba25f87d0d18bca8e9a33ab46` / `3fcae656a1c2a8f6f51dac95366a6525fb330817b4edf030b733fc5ad0ecc24b`
- final decision：当前固定 rank-12 基、25 参数 QH022 配置 `reject_or_mutate`；没有超过 CC，不运行 no-ent/gamma-zero/更多 seed/多模态扩展。保留“推理常驻参数生成器”家族作为未来不同 mutation，不保留当前参数化
- agent decision：20,000 次固定 seed 配对 bootstrap 的 `decision-qh-022.json` 自动复现硬失败 `quantum_underperformed_equal_parameter_control`，SHA-256 `a7959eb7ff9c8151094479e739e5da529ab064bb78ca2449f5310d0716c7480c`；queue SHA-256 `c9084f6a0d7ea6ac12873bfa39de0a5abaecb362bf462430a821fa2361e041c3`，携带数据耗尽标志、四项禁止操作和 QH023/QH024 两个需人工复核 proposal
