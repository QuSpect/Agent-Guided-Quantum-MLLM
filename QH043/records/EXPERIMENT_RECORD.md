# 原始实验记录：EXP-043 / QH043

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / beats equal-parameter CC point estimate / active causal gate failed / no C4 consumed
- parent：EXP-041；single mutation：第二层数据重上传角从 `1×2atan(v)` 改为 `2×2atan(v)`，显式开放二次谐波；线路深度、256门角、local-X读出、64 alpha、冻结支架、CC与512步预算均不变
- theory basis：QNN Fourier分析表明可达频率由编码生成元与重复上传决定（arXiv:2008.08605；Communications Physics 2026 `s42005-026-02680-x`）；QHA arXiv:2606.11673只在受控高阶任务支持该动机，不作为自然语言收益证据
- engineering：exact/DQ forward/input-grad差0，theta-grad最大差 `7.15e-7`；全部256门角和64 alpha在27B smoke有梯度；冻结梯度0；GPU exact simulator
- data lock：全新 WikiText-2 train-only 256 train +64 held blocks；排除8,912既往索引；SHA `5e2ae79d3d67eaf3d3ca64f338778413df6e5f567e654cea644f894e5816f2fc`
- result：base/scaffold/QH/CC NLL 2.344812/2.346786/2.346825/2.347074
- statistics：scaffold−base +0.001974，CI [-0.001801,+0.005822]；QH−scaffold +0.000039，CI [-0.000432,+0.000519]；QH−CC -0.000249，CI [-0.000694,+0.000203]；QH−base +0.002013，CI [-0.001685,+0.005779]
- diagnostics/cost：QH/CC mean step 0.3523/0.3377 s；QH simulator calls512；effective-alpha L2 0.1523/0.2085
- checkpoint：QH/CC两份1,322,002-byte完整checkpoint；SHA见参数清单
- decision：QH点胜CC但未胜own-scaffold，且两个区间跨0；不能把“控制更差”写成量子贡献。拒绝晋级、不消费C4；保留谐波编码为有区分度但本锁无正向active effect的结果
- artifacts：`artifacts/qh043-*.json`、`artifacts/qh043_harmonic_screen/*`、`src/quantum_qwen38/qh043_harmonic_x_residual.py`、`scripts/qh043_harmonic_workflow.py`
