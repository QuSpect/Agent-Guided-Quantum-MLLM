# 原始实验记录：EXP-037 / QH037

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / high-compression replacement passed screen / quantum-specific gate failed / no C4 consumed
- sources：QKAN/QVAF arXiv:2509.14026v2 与官方 `Jim137/qkan` 实现只作结构启发；其单 qubit DARUAN 被作者明确称为 quantum-inspired/classically simulable，本实验另行实现真正多 qubit 纠缠 exact-statevector core
- target：删除 Qwen3.8 layer31 完整 SwiGLU `gate_proj/up_proj/down_proj`，共 267,386,880 参数
- replacement：`5120→64→5120` 无 bias 低秩通道；64 维分成 16 组，每组独立 4q depth-2 data-reuploading RY + controlled-RY ring；256 core angles；总部署/可训练参数 655,616
- compression：全模型净减 266,731,264 参数，约占 27B 基座 0.975%；替代件仅为原 FFN 参数的约 0.245%
- fairness：CC037 使用完全相同 down/up 投影和同 256 核参数的 grouped sine/product 非线性；branch-off 严格等于删层；另有 no-ent 与独立 tensor-axis DQ
- implementation failure record：首轮 DQ 静态 backward 因 controlled-gate 对 view 做 inplace 写入被 PyTorch 拒绝；在任何数据/训练前改为纯 stack 非 inplace 张量更新，候选结构未变
- engineering：QH/DQ forward/input-grad/theta-grad最大差 0/0/4.77e-7；32/32 测试角和 256/256 全模型角均有梯度；CPU 拒绝；两步 QH/CC smoke 冻结梯度均0、peak约15.1GB
- data lock：新 WikiText-2 train-only 256 train +64 held blocks；排除6,992既往索引、重叠0；SHA `0d61130ef84c5e53e56b59a06639717cb2dcf8f83259501422f7a5007f90f4c9`
- objective：teacher-logit KL；AdamW lr1e-3；QH/CC各256步；27B其余参数冻结；训练和推理均用 CUDA FP32/complex64 exact simulator
- result：base/delete-zero/QH/CC NLL 2.414987/2.415725/2.411890/2.411118
- statistics：QH−zero -0.003835，descriptive 95% CI [-0.005955,-0.001704]；QH−base -0.003097，CI [-0.006273,+0.000063]；QH−CC +0.000772，CI [-0.000737,+0.002212]
- cost：QH/CC mean student step 0.3519/0.3362 s；QH simulator calls256，CC0；冻结梯度均0
- checkpoint：`artifacts/qh037_grouped_qvaf_screen/qh037-screen-checkpoint.pt` 与 `cc037-screen-checkpoint.pt`，均包含 replacement state、optimizer、config、data lock SHA 和源码入口
- decision：保留完整 FFN 高压缩替代骨架；量子臂显著优于直接删层但未胜等参经典点估计，按预注册门不进 C4；下一代只改变量子函数族/初始化，不重用本 held blocks 调参
- artifacts：`artifacts/qh037-*.json`、`artifacts/qh037_grouped_qvaf_screen/*`、`src/quantum_qwen38/qh037_grouped_qvaf_ffn_replacement.py`
