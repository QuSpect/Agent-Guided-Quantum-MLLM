# 原始实验记录：EXP-042 / QH042

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / safe active residual / quantum attribution failed / no C4 consumed
- parent：EXP-041；single mutation：局域 `X_q` 观测改为邻接双体 `X_q⊗Z_(q+1)`；冻结支架、320参数、trust-region、经典控制、训练目标和512步预算保持不变
- source basis：QASA arXiv:2504.05336 官方仓库 commit `d46d421a7809bc9fb86a4ae7883c3c100b06c070` 支持“少量、单位置、高纠缠线路”的方向；其小型时间序列结果不作27B证据。本实验的XZ观测是自行进化的量子相关特征
- implementation failure：首轮数据前 DQ 静态测试发现 group 轴未并入 simulator batch，报 30 对15维不匹配；修正为 `token×group` 展平后重跑，候选结构、随机种子和数据均未改变
- engineering：修复后 exact/DQ forward/input-grad差0，theta-grad最大差 `7.15e-7`；单坐标翻转的符号误差0；全部256角与64 alpha在27B smoke有梯度；冻结梯度0；peak约15.1GB
- data lock：全新 WikiText-2 train-only 256 train +64 held blocks；排除8,592既往索引；SHA `51256fca3ed570ca1fdb9a746da5d6f1ac9d19e5b028b671c45fa634b7cd8033`
- result：base/scaffold/QH/CC NLL 2.446555/2.445939/2.445854/2.445827
- statistics：scaffold−base -0.000616，CI [-0.003542,+0.002375]；QH−scaffold -0.000085，CI [-0.000489,+0.000322]；QH−CC +0.000027，CI [-0.000467,+0.000527]；QH−base -0.000701，CI [-0.003693,+0.002268]
- cost：QH/CC mean step 0.3510/0.3346 s；QH exact simulator calls512；冻结梯度0
- checkpoint：两份1,322,002-byte完整checkpoint，含replacement、optimizer、配置和data/source hash；SHA见参数清单
- decision：双体相关读出点估计优于own-scaffold且在base+0.005内，但未胜等参CC；按预注册门拒绝晋级、不消费C4。保留为安全的量子函数族结果，不宣称量子优势
- artifacts：`artifacts/qh042-*.json`、`artifacts/qh042_xz_screen/*`、`src/quantum_qwen38/qh042_signed_xz_correlator_residual.py`
