# 原始实验记录：EXP-038 / QH038

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / all screen gates failed / no C4 consumed
- parent：EXP-037；single mutation：保持 layer31、rank64、655,616 参数、两级4q局部线路、训练目标与预算不变，只在两级之间把 lane-q 循环移动 q 个 group 后重新分组
- literature basis：arXiv:2604.23931v2 与 arXiv:2602.16623 的 FC-VQC inter-block connectivity；论文仅覆盖小型表格任务，作为拓扑动机而非27B效果证据
- engineering：QH/DQ forward/input-grad差0，theta-grad最大差3.58e-7；256/256 QH/CC角有梯度；CPU拒绝；全模型 QH/CC smoke 冻结梯度0、峰值约15.1GB
- infrastructure event：经典 smoke 前两次 SSH 在 banner exchange 超时，模型进程均未启动；链路恢复后原命令原条件通过，不属于训练失败
- data lock：新 WikiText-2 train-only 256 train +64 held blocks；排除7,312既往索引；SHA `f19bc8a10de3eb4ef439a971a5804530178d82661285aceb753fbf7cd6b0ffe7`
- result：base/delete-zero/QH/CC NLL 2.361530/2.367796/2.367914/2.366980
- statistics：QH−zero +0.000119，CI [-0.001799,+0.002097]；QH−CC +0.000934，CI [-0.000414,+0.002292]；QH−base +0.006384，CI [+0.003342,+0.009441]
- optimization：QH/CC各256步；mean step 0.3567/0.3388 s；QH执行512个精确局部statevector stage；冻结梯度0
- parameter/checkpoint：净减266,731,264；两个3.94MB checkpoint均保存replacement、optimizer、data/source/entrypoint/base-config hash与精度后端
- decision：active、equal-parameter CC、base+0.005 三门全失败；shifted-ring mutation reject，不重用 held blocks；回到 QH037 成功压缩骨架上测试严格可关闭的小量子残差
- artifacts：`artifacts/qh038-*.json`、`artifacts/qh038_shifted_ring_screen/*`、`src/quantum_qwen38/qh038_shifted_ring_fcvqc_ffn_replacement.py`
