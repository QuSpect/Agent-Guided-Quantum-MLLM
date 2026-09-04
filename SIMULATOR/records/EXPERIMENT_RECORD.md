# 原始实验记录：EXP-001 / SIMULATOR

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete
- candidate：QH-001/QH-003/QH-006/QH-007 circuit and adapter
- 指标：forward/backward、解析/有限差分梯度、显存、BF16 可见性、CPU fallback
- 结果：原生 GPU 模拟器小线路解析误差 6.56e-7；QH-006e 与 QH-007 有限差分均通过 2% 门槛；强制 GPU-only
