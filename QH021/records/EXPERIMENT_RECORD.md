# 原始实验记录：EXP-020 / QH021

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 状态：complete / coherent-amplitude mixer lineage rejected
- parent_experiment：EXP-019
- intervention：从 QH020-QH checkpoint 只加载全部非 mixer 可训练张量，作为 QH021/CC021 唯一公共 trunk；两边各自 mixer 精确清零，冻结 trunk，并以完全相同 hard-256 顺序、步数和学习率只训练 12 个参数
- source checkpoint：`clevr-relations-qh020-staged-hard-third-holdout-s20260829-n256-adapter.pt`；运行时记录 SHA-256、载入张量数和参数数
- data：复用 DS-005-train-image-holdout-v3；该切片已经被 QH020 揭盲，所以仅用于“给定共同 trunk 时量子 mixer 是否优于经典 mixer”的因果审计，不作为新泛化、晋级或数据集确认
- gate：先运行 QH021/CC021 配对；QH 必须严格高于 CC 且视觉增益不回归，才追加同检查点 zero-mixer branch-off 或 no-ent；相同或更差则停止该谱系
- simulator：QH021 仍为 3q CUDA FP32/complex64 精确状态向量；CC021 为严格等参经典控制；CPU 量子路径继续禁止
- integrity：两路记录的源 checkpoint SHA-256 均为 `7fb31525667a85273d5b5fd877dec0c4bdf9d9e1e14a986cae5b672505830374`；均载入 16 个共同张量、823,745 个参数并将 mixer 精确清零；训练时都只有 12 个参数可训练，冻结参数梯度张量数为 0
- result：QH021/CC021 exact 为 93.7500/94.1406%；QH-CC -0.3906pp，20,000 次 image-cluster bootstrap 95% CI [-1.1719, 0]pp，`P(delta>0)=0`；逐题 0 win/1 loss/255 tie
- visual：两路原图-空白图增益均为 54.6875pp；配对差 0，95% CI [-2.3438, 2.3438]pp
- optimization diagnostic：QH/CC mixer 梯度范数均值 0.03091/3.59e-7。量子小核确实收到并改变信号，但唯一跨答案边界的变化方向不利；经典控制接近零梯度也不构成量子收益
- decision：预注册双门失败，停止相干振幅 mixer 谱系；不追加 no-ent、zero-mixer 或更多 seed，避免在已揭盲开发集上二次选择
- artifacts：`clevr-relations-{qh021,cc021}-exact-shared-trunk-third-holdout-s20260829-n256.json`；`paired-qh021-vs-cc021-exact-shared-trunk.json`
