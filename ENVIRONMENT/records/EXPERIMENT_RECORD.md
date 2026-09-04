# 原始实验记录：EXP-000 / ENVIRONMENT

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- 日期：2026-08-28
- 状态：complete
- 主机：4×A800-SXM4-80GB，NV8互联
- 软件：Python 3.12.3；PyTorch 2.8.0+cu128；Transformers 5.16.0.dev0；PEFT 0.20；原生 PyTorch CUDA 状态向量模拟器
- 存储：30GB系统盘；约1.1TB高速数据盘
- 结果：资源满足直接Qwen3.8路线；需建立独立环境
- 结果补充：公钥免密已验证；HF mirror、AutoDL network_turbo、ModelScope 均实测并记录
