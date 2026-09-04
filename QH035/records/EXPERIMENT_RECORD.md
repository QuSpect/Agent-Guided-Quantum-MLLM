# 原始实验记录：EXP-035 / QH035

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / compression viable / quantum-specific screen failed / no C4 consumed
- target：Qwen3.8-27B layers 11/12/13/15 的 `post_attention_layernorm`；先按 Qwen3_5 实际定义把存储权重还原成 effective scale `1 + weight`
- replacement：删除 4×5,120 个独立尺度，部署 1×5,120 frozen shared effective scale；每层输出叠加由 13q depth-2 exact-statevector 产生的逐 token 残差
- parameter audit：原层20,480参数；部署5,120 frozen +104 circuit angles +4 gammas =5,228；全模型净减15,252；QH/CC均108 trainable
- weight-only selection：最佳 post-attention k=4 cluster 为 layers 11/12/15/13；对 effective scale 的 RMS relative sharing error=1.2129%，略高于预设1%近无损门，按用户允许轻微质量下降只进入风险受控 train-only screen
- implementation correction：首轮审计曾把存储权重当成完整尺度；读取 Qwen3_5 源码确认 forward 使用 `1 + weight` 后，在任何数据评价前修正并重跑；候选层选择未依赖性能集
- engineering：零支路 identity 最大误差1.19e-7；QH/DQ forward/input-grad/theta-grad最大误差1.19e-6/9.54e-7/3.81e-6；27B两步 simulator calls8、冻结梯度0、104/104角有梯度、峰值约21.6GB
- lock：全新 WikiText-2 train-only 256 train +64 held blocks；排除6,138既往索引、重叠0；SHA `f60de4e21bc2f79a19ac3bc053dd0f13a61c2df7b000915cad0ba4fd0c324d4d`
- result：base/sharednorm/QH/CC NLL 2.418541/2.419198/2.419173/2.419122；QH−shared -0.000025，descriptive CI [-0.000513,+0.000476]；QH−CC +0.000051，CI [-0.000462,+0.000576]
- cost：QH/CC mean student step 0.680/0.525 s；QH simulator calls1,024；所有非候选参数冻结梯度0
- decision：共享 RMSNorm 以极小 NLL 代价净减15,252参数，可作为保守压缩积木；当前量子残差未胜同108参数经典控制且不确定区间宽，停止并不消耗 C4
- artifacts：`artifacts/qh035-*.json`、`artifacts/qh035_norm_screen/*`、`src/quantum_qwen38/qh035_shared_rmsnorm_replacement.py`
