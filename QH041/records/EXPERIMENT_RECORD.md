# 原始实验记录：EXP-041 / QH041

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / compression scaffold improves base on this lock / quantum attribution failed / no C4 consumed
- parent：EXP-040；single mutation：只把量子局域观测从 Pauli-Z 改为 Pauli-X；冻结 CC037 高压缩骨架、`0.05*tanh(raw)` trust region、320个可训练参数、经典控制、目标、lr 与512步预算均不变
- representation audit：对 `RY(2atan(v))` 编码，零偏置局域 Z 为偶函数 `(1-v²)/(1+v²)`，而局域 X 为保留符号的奇函数 `2v/(1+v²)`；静态正负输入审计 odd-pair 最大误差 `5.96e-8`
- engineering：QH/DQ forward 与 input-grad 差0，theta-grad最大差 `3.58e-7`；全部256个量子角和64个alpha在27B smoke上有梯度；冻结参数梯度0；CUDA exact simulator、peak约15.1GB
- data lock：全新 WikiText-2 train-only 256 train +64 held blocks；排除8,272既往索引；SHA `e5a639c1edd66b18f520699ddf0585900acf94f244710757168ed0ee666d1da8`
- result：base/scaffold/QH/CC NLL 2.379894/2.376041/2.375714/2.375512
- statistics：scaffold−base -0.003853，descriptive 95% CI [-0.007160,-0.000522]；QH−scaffold -0.000327，CI [-0.000783,+0.000143]；QH−CC +0.000203，CI [-0.000319,+0.000695]；QH−base -0.004179，CI [-0.007546,-0.000888]
- diagnostics：signed-X 使量子点估计首次在该冻结支架路线上优于 own-scaffold，并在本锁上优于base；但区间未证明active effect，且等参CC仍更好。QH/CC effective-alpha L2 0.1693/0.2135，max abs 0.0336/0.0374
- cost：QH/CC mean student step 0.3513/0.3325 s；QH exact simulator calls512；冻结梯度0
- checkpoint：`qh041-screen-checkpoint.pt` 与 `cc041-screen-checkpoint.pt` 均为1,322,002 bytes，保存完整replacement、optimizer、配置和data/source hash；SHA见参数清单
- decision：保留 signed-X 作为比 local-Z 更好的量子表征证据，但预注册 equal-parameter 点估计门失败，不消费 C4；下一变异必须提升量子可表达的跨坐标交互而不增加投影参数或偷换训练预算
- artifacts：`artifacts/qh041-*.json`、`artifacts/qh041_signed_x_screen/*`、`src/quantum_qwen38/qh041_signed_x_readout_residual.py`
