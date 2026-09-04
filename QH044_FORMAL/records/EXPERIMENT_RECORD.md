# 原始实验记录：EXP-046 / QH044_FORMAL

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / active recovery confirmed / quantum-specific and entanglement gates failed
- lock：128个全新 C4 English validation blocks；排除2,576历史C4索引；SHA `0967e302d399d1a7a6ffe039210faaad84dd8a9ad981453b75752ee68cfc8cdf`；训练0、test0
- arms：base、delete-zero、QH044、CC044、QH044-no-ent、独立tensor-axis DQ；QH/DQ/no-ent各执行128次CUDA exact simulator forward
- point NLL：base/zero/QH/CC/no-ent/DQ = 2.668882/2.675818/2.673367/2.672589/2.672734/2.673367
- statistics：QH−base +0.004484，CI [+0.002360,+0.006657]；QH−zero -0.002451，CI [-0.003610,-0.001273]；QH−CC +0.000777，CI [-0.000237,+0.001772]；QH−no-ent +0.000632，CI [+0.000096,+0.001132]
- simulator equivalence：QH−DQ逐block最大NLL差0，配对差与CI均0
- gates：active causality pass；DQ pass；compression +0.005 CI、equal-parameter quantum-specific、entanglement-specific三门fail
- decision：确认该量子替代件可恢复完整删层功能，但train-only QH-over-CC不复制；当前受控RY纠缠在C4上具有显著负贡献。不能宣称量子优于经典或无损压缩；下一变异应从QH044 checkpoint删除/重构有害纠缠，而不是加深线路
- artifacts：`artifacts/qh044-c4-formal-lock.json`、`artifacts/c4_qh044_formal/*`、`scripts/c4_qh044_formal.py`
