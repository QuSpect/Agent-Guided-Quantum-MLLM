# 原始实验记录：EXP-044 / CC037_C4

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / restoration confirmed / strict +0.005 noninferiority not confirmed / compression-only claim
- purpose：固定 EXP-037 的 CC037 checkpoint，训练0步，在独立 C4 上只确认“删掉完整layer31 SwiGLU后，655,616参数替代件是否可接受”；本实验不作量子优势主张
- data lock：256个全新 C4 English validation blocks，排除2,273个历史C4索引；锁 SHA `54b8b689bfc63bf0422eb1299bf106b2e5b5133ff38ee69d846e641f3180f49f`；索引列表 SHA `a1f3b1c74d309e660005d2e6a9ac5c67244bccf254ff7a0b2f2c0f84791a2087`
- implementation/infrastructure failures：两次SSH banner超时均未启动进程；首次base evaluator在第一个block因C4 token为Int32而CUDA CE要求Int64，在返回任何loss前失败。唯一修复为输入/标签cast `torch.long`；重签入口hash后256索引列表SHA完全不变
- arms：原始base、layer31完整delete-zero、固定CC037 scaffold；所有臂训练0步，checkpoint未更新，test读取0
- parameter result：删除267,386,880，部署655,616，净减266,731,264，约占27B基座0.975%
- point result：base/delete-zero/scaffold NLL 2.642186/2.649870/2.646708
- statistics：scaffold−base +0.004522，paired bootstrap 95% CI [+0.003075,+0.005963]；scaffold−delete-zero -0.003162，CI [-0.003824,-0.002500]
- decision：替代件显著恢复完全删层造成的约41.1% NLL损失，但预注册+0.005非劣界要求CI上界<0.005，本次上界0.005963，严格门失败。可描述为“大幅减参、轻微质量下降的候选”，不可描述为无损或已确认非劣
- artifacts：`artifacts/qh037-compression-c4-confirmation-lock.json`、`artifacts/c4_qh037_compression_confirmation/*`、`scripts/c4_qh037_compression_confirmation.py`
