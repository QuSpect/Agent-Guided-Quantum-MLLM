# 原始实验记录：EXP-006 / QH008_TEXTVQA

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：QH-008 replication complete / rejected；QH-009 paired screen running
- source：`facebook/textvqa`，converted parquet revision `adeddaaa993f9f09c7ba460762c03628f37aa298`
- license：CC BY 4.0
- pilot scope：固定 train/validation 的第 0000 分片；不读取 test；候选共享相同 row index
- metric：EvalAI 答案归一化 + 官方 leave-one-out VQA soft accuracy；已用命中数 0/1/2/3/4 对应 0/.3/.6/.9/1.0 的已知阶梯单测通过
- counterfactual：original / shuffled / blank 配对；报告 soft accuracy、原图-空白、positive visual rescue 和预测改变率
- candidates：base、QH-008、CC-008；仅前两者通过门槛后才运行 no-ent；QH-006e 作为 88 参数压缩极限参考
- split integrity：train-0000/validation-0000 的 question_id 与 image_id overlap 均为 0；有 48 个规范化问题文本模板重复，但对应图片不同；审计未解码图片且 test read count=0
- baseline：128 validation soft accuracy 84.1406%；反事实 64 条原图-空白图增益 84.6875pp
- QH-008/CC-008 首轮：84.2188%/83.4375%；量子相对经典 +0.7813pp，95% CI [0, 2.3438]pp，但 exact-any 只净改善 1 条、McNemar p=1.0
- 视觉因果：QH-008 与 CC-008 的原图-空白图增益均为 84.375pp；两者都未高于基线
- next：统一扩大到 train 512 / validation 256 / counterfactual 128，学习率降为 1e-4；共享负载计时不用于成本晋级
- 512 复验：QH-008/CC-008 soft accuracy 85.5469%/85.8984%，量子-经典 -0.3516pp，95% CI [-1.1719, 0.1172]pp；视觉增益差 -0.7031pp，95% CI [-2.3438, 0.2344]pp
- 同 row 冻结基线：soft accuracy 86.0156%，QH-008 -0.4688pp；冻结基线/QH-008/CC-008 视觉增益 77.4219/75.7813/76.4844pp
- decision：QH-008 reject；首轮单样本正差未复现；控制器记录三项硬失败并锁定输入 SHA-256
