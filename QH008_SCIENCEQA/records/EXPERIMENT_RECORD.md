# 原始实验记录：EXP-005 / QH008_SCIENCEQA

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / no promotion
- source：仅 `ScienceQA-IMG/train`；validation/test read count 固定为 0
- scan：先 2,000 条检查点，随后哈希校验断点扩展至 6,218 条
- output：`scienceqa-hard-mine-s20260828-scan*.json`
- next：所有晋级候选共享相同 wrong_indices；与 TextVQA OCR 域混合后复筛
- 全量结果：6,218 条 train 扫描得到 322 条基线错误，train accuracy 94.8215%；validation/test read count 均为 0
- answer-token 复筛：QH-008 与 CC-008 均为 90.625%，相对冻结基线均为 -2.3438pp；QH-008 相对 CC-008 为 0pp
- 反事实：QH-008 多模态增益 9.375pp，CC-008 为 7.8125pp，差值 +1.5625pp 但 95% CI 跨 0
- 决策：ScienceQA 路线保留为负结果与回归门禁，不继续在该 validation 上搜索

### EXP-005A：前 2,000 条检查点与 96 条首轮复筛

- 状态：complete / no quantum promotion
- 挖掘结果：2,000 train rows 中 104 条错误，基线 train accuracy 94.8%；记录每 100 条原子更新且支持哈希校验断点续跑
- QH-001b：validation accuracy 与基线相同；多模态增益 -3.125pp
- CC-001：validation accuracy +0.7813pp，但 McNemar p=1.0；多模态增益不变
- 决策：旧整段 loss 路线停止；全 6,218 train 扫描继续用于下一目标，不读取 test
