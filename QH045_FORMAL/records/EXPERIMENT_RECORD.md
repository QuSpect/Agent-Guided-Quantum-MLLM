# 原始实验记录：EXP-048 / QH045_FORMAL

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / train-only signal failed to replicate / quantum-specific rejected
- lock：64个全新 C4 English validation blocks，排除2,738个历史 C4 索引；SHA `3e566a48296cd1051644d5dc249c8932a7fa59c2fe09d1cbea5742c4e983b169`；训练0、test0
- arms：base、共同 QH044 no-ent 起点、冻结QH045、冻结CC045、独立tensor-axis DQ；后四臂各运行64次GPU exact simulator
- infrastructure：模块上传两次、lock一次、run-all一次SSH banner超时，均发生在传输/远端进程启动前；成功后严格使用同一源码、checkpoint与锁
- point NLL：base/no-ent/QH/CC/DQ = 2.725877/2.734450/2.735166/2.734673/2.735166
- statistics：QH−base +0.009289，CI [+0.006320,+0.012545]；QH−no-ent +0.000716，CI [+0.000129,+0.001303]；QH−CC +0.000493，CI [-0.000114,+0.001081]
- simulator equivalence：QH−DQ 每个 block 最大 NLL 差0，DQ门通过
- gates：compression +0.005、active selective-entanglement、equal-parameter quantum-specific 三门失败；DQ equivalence通过
- decision：QH045不能晋级。64参数可选择纠缠在同域held上显著好于no-ent/CC，但在C4上显著劣于no-ent且点劣于CC，属于清晰的分布过拟合；后续搜索必须把架构选择纳入多域交叉验证或资源惩罚，不能沿这组selector继续调学习率/阈值
- artifacts：`artifacts/qh045-c4-formal-lock.json`、`artifacts/c4_qh045_formal/*`、`scripts/c4_qh045_formal.py`
