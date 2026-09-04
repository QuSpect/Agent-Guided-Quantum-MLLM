# 原始实验记录：EXP-039 / QH039

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / compression scaffold retained / both residuals failed / no C4 consumed
- parent：EXP-037；motivation：把“压缩骨架有效”和“量子增量有效”拆开，避免随机投影与线路共同训练掩盖归因
- scaffold：加载并冻结已登记 CC037 checkpoint；`5120→64→5120`、256参数经典 grouped core 全冻结；另加256参数 QH/CC residual core和64个逐坐标alpha，只训练320参数
- branch-off：同一模块关闭 correction 后精确回到冻结 CC037 scaffold；量子/经典共享相同 backbone checkpoint SHA `4ad20b...d56a9`
- parameter audit：部署655,936，删除267,386,880，全模型净减266,730,944；trainable 320
- engineering：QH/DQ forward/input/theta/alpha梯度最大差0/0/1.82e-12/0；alpha-zero与branch-off严格一致；冻结scaffold梯度0；27B QH/CC smoke全320参数有梯度、冻结基座0
- data lock：新 WikiText-2 train-only 256 train +64 held blocks；排除7,632既往索引；SHA `411e8e046eda29117715c93c1956a4501fe44f2e689e975f47e3dcaeb1e16a03`
- training：teacher-logit KL、AdamW lr3e-3、512步（同256 blocks两遍）；QH exact simulator calls512；QH/CC mean step 0.3552/0.3455 s
- result：base/scaffold/QH/CC NLL 2.389533/2.391340/2.391972/2.391633
- statistics：scaffold−base +0.001807，CI [-0.001670,+0.005069]；QH−scaffold +0.000632，CI [-0.000188,+0.001397]；QH−CC +0.000339，CI [-0.000435,+0.001151]
- diagnosis：QH/CC final alpha norm 2.012/2.784；训练KL下降但held NLL变差，说明不受限残差幅度过拟合。量子线路有信号但方向无益
- checkpoint：两份1,321,874-byte checkpoint含完整replacement、optimizer、backbone/data/source/entrypoint/base-config hash
- decision：保留第三个独立数据锁仍仅+0.001807 NLL的高压缩scaffold；拒绝不受限QH039/CC039残差；QH040只增加对alpha的预注册trust-region上界，不改线路/目标/预算
- artifacts：`artifacts/qh039-*.json`、`artifacts/qh039_frozen_scaffold_screen/*`、`src/quantum_qwen38/qh039_frozen_scaffold_quantum_residual.py`
