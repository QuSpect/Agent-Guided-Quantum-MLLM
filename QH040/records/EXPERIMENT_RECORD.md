# 原始实验记录：EXP-040 / QH040

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / conservative bound worked / quantum gates failed / no C4 consumed
- parent：EXP-039；single mutation：把逐坐标残差系数从无界alpha改为 `0.05*tanh(raw)`；同320参数、CC037冻结骨架、depth2线路、目标、lr和512步预算不变
- engineering：effective alpha静态饱和不超过0.05；QH/DQ forward/input/theta/alpha梯度最大差0/0/7.28e-12/0；QH/CC全部320参数有梯度；冻结骨架0
- infrastructure：首次静态命令在SSH banner前超时，Python未启动；原命令重连后通过，记录为基础设施事件
- data lock：新WikiText train-only 256+64 blocks；排除7,952既往索引；SHA `19a11da7f264fd53f156b6e0b4bdbc3a3b7a4d707b1d5b138222587721f543a8`
- result：base/scaffold/QH/CC NLL 2.379066/2.379743/2.379845/2.379465
- statistics：scaffold−base +0.000676，CI [-0.002812,+0.004232]；QH−scaffold +0.000102，CI [-0.000404,+0.000612]；QH−CC +0.000380，CI [-0.000119,+0.000879]
- diagnostics：QH/CC effective-alpha L2 0.1475/0.1955，max abs 0.0310/0.0345，明显低于cap；bounded CC补回0.000278，但QH仍轻微有害
- cost：QH/CC mean step 0.3471/0.3326 s；QH exact simulator calls512；冻结梯度0
- checkpoint：两份1,322,002-byte完整checkpoint；SHA见训练参数清单
- decision：trust region作为安全机制保留，QH040 Z-readout拒绝；QH041只把量子局域观测从Z换成signed X，保持其余条件和经典控制不变
- artifacts：`artifacts/qh040-*.json`、`artifacts/qh040_bounded_residual_screen/*`、`src/quantum_qwen38/qh040_bounded_quantum_residual.py`
