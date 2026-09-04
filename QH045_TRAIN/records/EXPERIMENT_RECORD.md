# 原始实验记录：EXP-047 / QH045_TRAIN

下文保留实验注册表的技术字段与原始措辞，供复核，不作为面向普通读者的总结。

- status：complete / train-only double-significant screen / promoted to fresh C4
- parent：EXP-046；正式 C4 已确认 QH044 固定 ring 纠缠显著有害，因此不再加深线路，改为从完全关闭纠缠的共同起点学习哪些边应重新打开
- source basis：QAS（arXiv:2010.10217）与 QuantumNAS（arXiv:2107.10845）的结构搜索/门剪枝动机；本实现为面向 Qwen3.8 FFN 的项目内变异，不宣称复现论文任务
- shared start：QH/CC 都加载同一个冻结 QH044 checkpoint；64 个跨 depth 共享 selector 精确置零时，两臂严格等于同一 no-ent simulator scaffold；down/up、原256线路角和27B基座全部冻结
- intervention：QH 用64个 selector 缩放已训练 controlled-RY 角；CC 用相同64个 selector、相同冻结角，在同一个 no-ent 模拟器输出上施加邻接乘积交互；两臂都只训练64参数且每步都运行一次GPU exact simulator
- parameter：删除267,386,880；部署655,680；净减266,731,200；每臂仅64参数可训练
- data lock：全新 WikiText-2 train-only 384 train +64 held，和 QH043/QH044 锁显式重叠0；SHA `ad56ea4f876147cc76a3e7bcb3dc02101dd769c0facf7df92925635c320f3f00`
- infrastructure：CC首次SSH banner超时发生在远端进程启动前；同一锁原样重试，不计为训练运行
- result：base/no-ent/QH/CC NLL = 2.321290/2.318746/2.317988/2.318617
- statistics：QH−no-ent -0.000758，95% CI [-0.001491,-0.000074]；QH−CC -0.000629，CI [-0.001223,-0.000025]；QH−base -0.003301，CI [-0.006877,+0.000338]
- diagnostics：QH/CC 512次simulator call、冻结梯度0；gate L2 4.590/4.859；|gate|<0.05 为2/5个，说明学习到的是密集重加权而非强稀疏
- checkpoint：两份1,317,190-byte完整训练状态；QH SHA `752cd1...59a7d`，CC SHA `b3d8fb...3293f`
- decision：train-only 同时显著胜共同起点与等参控制，按锁定规则冻结 checkpoint 并进入全新 C4；此阶段禁止量子成功主张
