# 原始实验记录：EXP-034 / QH034

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / train-only screen failed / no C4 consumed
- parent：EXP-031 compression backbone；literature basis：FuRA、HiP-LoRA、Spectral Surgery、Quantum-PEFT
- mutation：10q depth2 RY/controlled-RY 线路从固定均匀态生成 1024 个 Born 概率谱缩放；乘 rank 后严格正值、均值1；零角严格等于 shared-SVD
- controls：同 80 核参数 Walsh-softmax 谱生成器 CC034、branch-off own-zero、no-ent、独立 tensor-axis DQ；每层另1 gamma，总计82
- implementation failure record：首轮小型 static 夹具 rank16 却要求32个独立 Walsh modes，被显式容量检查在性能前拒绝；只把夹具扩大到 rank64 后原结构重跑
- static：四路零角恒等最大误差0；QH/DQ forward/input-grad/theta-grad 最大误差 2.15e-6/2.86e-6/1.34e-5；正值均值1约束和 CPU 拒绝通过
- 27B smoke：两步 simulator calls4；80/80 theta gradients nonzero；冻结梯度0；peak VRAM15.3GB；输出 KL smoke=0.000843
- parameter audit：删除10,485,760；部署7,340,114；净减3,145,646；仅82 trainable
- lock：新 WikiText train-only 256 train +64 held blocks；排除5,882既往索引、重叠0；SHA `4f27389cbfd3cb53e17661aa9473c6729e0d267e3c722ae5c35f17c7f196e3ba`
- result：base/shared/QH/CC NLL 2.327135/2.328073/2.328024/2.327847；QH−shared -0.000048，descriptive CI [-0.000445,+0.000343]；QH−CC +0.000177，CI [-0.000265,+0.000608]
- optimization：QH/CC mean step 0.314/0.255 s；QH simulator calls512；QH/CC final theta norm0.0559/0.1253；冻结梯度均0
- decision：谱坐标干预点保留，但量子 Born generator 未胜等参经典 Walsh control；按预注册门不消耗新 C4，停止当前核
- artifacts：`artifacts/qh034-*.json`、`artifacts/qh034_spectral_screen/*`、`src/quantum_qwen38/qh034_spectral_modulation_replacement.py`
