# 原始实验记录：EXP-045 / QH044_TRAIN

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / first train-only significant QH-over-CC screen / promoted to fresh C4
- single mutation：从登记的step-256 QH037与CC037参数和AdamW状态出发，各增加完全相同的1,000步；架构、655,616参数、lr1e-3、teacher-logit KL与序列长度不变
- data lock：500 train +64 held 全新 WikiText-2 train-only blocks；仅同数据域排除7,634历史索引；SHA `08b2e1c16e174b1aaf8212c78ce99d2aa6fcb90468054ad431a3b98337b79f47`
- fixed arms：base/zero/QH037-start/CC037-start NLL 2.395175/2.401617/2.396365/2.395733
- continued arms：QH044/CC044 NLL 2.394943/2.397257；QH−zero -0.006674，CI [-0.008983,-0.004432]；QH−CC -0.002314，CI [-0.004016,-0.000653]；QH−base -0.000232，CI [-0.004229,+0.003606]
- optimization：QH/CC mean step 0.3571/0.3538 s；QH exact simulator calls1,000，CC0；累计步数均1,256；冻结梯度0
- checkpoint：`qh044-checkpoint.pt`/`cc044-checkpoint.pt`各3,941,537 bytes，完整保存replacement、续接optimizer、累计步数、数据/源码/入口/起点hash
- decision：首次在train-only held上让QH同时显著胜zero与等参CC并点胜base；按锁定规则冻结参数，进入全新C4-128正式六臂，不把筛选结果写成最终结论
- artifacts：`artifacts/qh044-extended-distillation-lock.json`、`artifacts/qh044_extended_distillation/*`、`scripts/qh044_extended_distillation.py`
