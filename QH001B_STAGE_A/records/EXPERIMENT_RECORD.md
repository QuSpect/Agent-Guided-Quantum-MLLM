# 原始实验记录：EXP-004 / QH001B_STAGE_A

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / no promotion
- dataset：DS-001-v1；train 128，validation 128，counterfactual 64；seed 20260828
- candidates：base、QH-006e、QH-001b、CC-001、QH-001b+ranking、QH-007
- 结论：没有候选达到 +1pp 且 CI 下界大于 0；等参量子优势不成立；详见 EV-004/EV-005 和逐样本 JSON
- failure reason：基线 92.97% 接近天花板，随机训练样本高损失密度低
- decision：重组训练数据，不修改独立 validation
