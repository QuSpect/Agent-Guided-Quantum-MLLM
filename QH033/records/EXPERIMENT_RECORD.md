# 原始实验记录：EXP-033 / QH033

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / compression retained / Pauli-observable quantum family rejected
- parent：EXP-032；single mutation：从 QH032/CC032 checkpoint 出发，只把训练目标改为完整 LM logits 上的 teacher-student KL；结构、82 参数与共享压缩主干不变
- smoke：真实 logits shape `[1,255,248320]`；KL 0.0008848；theta/alpha/gamma 梯度均非零；simulator call2；冻结梯度0
- train-only lock：全新 256 train +64 held train blocks，validation/test0；SHA `de43f6b8f106e84eabfa080538d2f504ffb58bea3734d2e092993c124aad42d8`
- screen：base/shared/QH/CC NLL 2.416112/2.416891/2.416562/2.416709；QH 点胜 own-zero 0.000329、点胜 CC 0.000147，因此按预注册规则进入全新 C4
- formal lock：128 个从未使用 C4 blocks，排除既往704 blocks，训练0步；SHA `7ae24f26dafc91cbfc8bbc2ab7273415a1e60418e63013f4ab0941d1d80dba68`
- formal metrics：base/shared/QH/CC/no-ent NLL 2.721951/2.722639/2.722555/2.722484/2.722775
- paired CI：QH−base +0.000604，95% CI [+0.000147,+0.001058]；QH active−own-zero -0.000084，CI [-0.000388,+0.000220]；QH−CC +0.000071，CI [-0.000228,+0.000369]；QH−no-ent -0.000219，CI [-0.000492,+0.000058]
- gates：压缩 +0.005 noninferiority 与净减参数通过；strict-base、active-zero、equal-parameter CC、entanglement causal 四门失败
- decision：输出 KL 修正了训练筛选方向，但未形成可复现量子贡献，且 CC 正式点估计更好；停止 Pauli-observable family，保留 shared-SVD compression
- artifacts：`artifacts/qh033-*.json`、`artifacts/c4_qh033/*`
