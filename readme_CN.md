<p align="center">
  <img src="assets/quspect-logo.jpg" alt="量观知元（QuSpect Technology Co., Ltd.）Logo" width="240">
</p>

# Agent 引导的多模态大模型量子替代与参数高效适配

Agent-Guided Discovery of Quantum Replacements and Parameter-Efficient Adaptation Methods for Multimodal Large Language Models

**简体中文** | [English](README.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg)](https://pytorch.org/)

本仓库发布一套面向多模态大语言模型的量子替代与参数高效适配实验代码。项目采用人工监督的 Agent 工作流，通过代码检查、训练集样例分析、候选定义、实现演化和实验复核，系统探索可插入多模态大模型的参数化量子电路及量子—经典混合模块。

当前开源版本以 **Qwen3.8-27B** 为案例，包含**独立实验单元**，覆盖 候选配置以及环境检查、冻结基线、消融、审计、训练和正式复验。候选方案涉及视觉残差、问题条件路由、关系混合、量子生成低秩参数、值投影替代、共享归一化和完整 FFN 替代等方向。

本项目由 **量观知元（QuSpect Technology Co., Ltd., Beijing, China）** 发布和维护。



## 项目特点

- **Agent 引导的候选搜索**：围绕插入位置、量子编码、电路结构、可训练参数、读出方式和训练协议持续生成并修订候选。
- **覆盖多种模型接口**：包含视觉侧残差与路由、参数高效适配、注意力值投影替代、归一化替代和完整 FFN 替代。
- **实验单元独立分发**：每个目录保存自己的入口、依赖、阶段计划、实验记录、证据文件和许可证，可在新目录中单独使用。
- **保留完整实验轨迹**：成功、失败、无纠缠消融、删除基线、经典对照和冻结检查点复验均保留。
- **可审计的可移植改写**：`manifest.json` 同时记录原始文件与分发文件的 SHA-256，并标记因路径可移植化产生的改写。
- **固定基座与数据边界**：案例实验固定 Qwen3.8-27B revision；训练集挖掘、冻结评价和公开测试集的用途按各实验记录区分。

## 研究范围

本仓库中的量子模块主要采用 CUDA 上的 FP32/complex64 exact-statevector 实现。实验考察的结构包括：

1. 视觉 token 或视觉表征上的量子残差、锚点、FiLM 和关系路由；
2. 由量子电路生成或调制的低秩参数与正交子空间；
3. `v_proj` 等线性层的低参数替代；
4. 跨层共享基、谱调制和共享 RMSNorm；
5. 完整 SwiGLU FFN 的压缩替代及其量子残差演化；
6. 无纠缠、关闭量子分支、删除层和经典模块等对照或因果审计。

实验涉及 ScienceQA、TextVQA、CLEVR、EditCLEVR、WikiText 和 C4。各数据集仅在相关实验单元中使用，具体 revision、split、样本预算和预处理方式以对应目录中的 `README.md`、`stage_plan.json` 与 `records/EXPERIMENT_RECORD.md` 为准。

## 仓库结构

```text
.
├─ ENVIRONMENT/                    # 环境与基础设施审计
├─ SIMULATOR/                      # GPU exact-statevector 微基准
├─ QWEN38_BASE/                    # 冻结基座加载与推理基线
├─ QH*/                            # 量子—经典混合候选、消融和复验
├─ FFN036_AUDIT/                   # 完整 FFN 删除敏感性审计
├─ CC037_C4/                       # 经典高压缩替代件的独立 C4 复验
├─ LICENSE                         # 本项目原创内容的 MIT License
├─ THIRD_PARTY_NOTICES.md          # 基座、数据集和依赖的第三方条款
└─ readme_CN.md                    # 中文项目说明
```

每个实验目录通常包含：

```text
<EXPERIMENT>/
├─ code/src/quantum_qwen38/        # 候选模块与模拟器实现
├─ code/scripts/                   # 预处理、训练、推理和评价入口
├─ artifacts/                      # 已分发的锁文件与实验输出（如有）
├─ records/EXPERIMENT_RECORD.md    # 原始实验状态、协议、结果与限制
├─ manifest.json                   # 来源、哈希、入口和证据清单
├─ stage_plan.json                 # 本单元实际达到阶段的执行计划
├─ run_reached_stage.py            # 可移植阶段运行器
├─ prepare_scheme_weights.py       # 方案权重准备工具
├─ verify_parameters.py            # 参数清单与哈希检查
├─ download_base_model.py          # 冻结基座下载工具
├─ download_data.py                # 对应数据下载与预处理入口
├─ commands.sh / commands.ps1      # Linux 与 Windows 命令清单
└─ README.md                       # 单元级复现说明
```

各实验单元保留代码快照是有意设计：后续候选可能修改同名实现，独立快照可以避免复现实验时被较新的实现覆盖。

## 快速开始

### 1. 选择实验单元

如果希望先检查环境或模拟器，可以从以下目录开始：

- [`ENVIRONMENT`](ENVIRONMENT/README.md)：环境与基础设施检查；
- [`SIMULATOR`](SIMULATOR/README.md)：GPU exact-statevector 实现与微基准；
- [`QWEN38_BASE`](QWEN38_BASE/README.md)：Qwen3.8-27B 冻结基座加载与推理自检。

其他候选的入口和实际达到阶段见下方[实验单元索引](#实验单元索引)。

### 2. 创建 Python 环境

历史实验环境为 4×A800-SXM4-80GB、Python 3.12、PyTorch 2.8/CUDA 12.8 和 Transformers 5.16 开发线。量子训练与推理入口要求 NVIDIA CUDA GPU，并拒绝静默切换到 CPU。

Linux shell：

```bash
cd QH014  # 替换为目标实验目录
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
Set-Location QH014  # 替换为目标实验目录
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. 下载冻结基座

基座权重不随本仓库分发。每个实验单元都提供相同接口的下载脚本：

```bash
python download_base_model.py --output ./models/Qwen3.8-27B
```

登记的上游仓库为 [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)，固定 revision 为：

```text
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

### 4. 准备方案权重

GitHub 代码包不重复提交大权重。需要 checkpoint 的实验应先从对应参数仓库取得完整的 `models/<EXPERIMENT>/` 目录。将参数包中的 `manifest.json` 复制到实验目录并命名为 `scheme_manifest.json`，然后运行：

```bash
python prepare_scheme_weights.py
python verify_parameters.py
```

没有 checkpoint 的环境、静态检查或纯审计单元会在自己的 README 中明确说明可跳过此步骤。参数仓库尚未公开的实验只能复核代码、阶段计划和已分发证据，不能据此重建未提供的训练权重。

### 5. 下载数据并运行已达到阶段

```bash
python download_data.py --root ./datasets
python run_reached_stage.py
```

`run_reached_stage.py` 读取本目录的 `stage_plan.json`，只执行该实验登记过的预处理、训练、冻结评价或分析步骤。也可以审阅并逐条执行 `commands.sh` 或 `commands.ps1`。

## 实验单元索引

### 基础设施、基线与早期验证

| 实验单元 | 内容 |
|---|---|
| [`ENVIRONMENT`](ENVIRONMENT/README.md) | 远端环境与基础设施审计 |
| [`SIMULATOR`](SIMULATOR/README.md) | GPU exact-statevector 模拟器微基准 |
| [`QWEN38_BASE`](QWEN38_BASE/README.md) | Qwen3.8-27B 加载与冻结基线 |
| [`QH001_INJECTION`](QH001_INJECTION/README.md) | 真实模型量子注入与一步优化检查 |
| [`QH001B_STAGE_A`](QH001B_STAGE_A/README.md) | ScienceQA 随机 128 条 Stage-A 实验 |
| [`QH008_SCIENCEQA`](QH008_SCIENCEQA/README.md) | ScienceQA train-only 难例挖掘 |
| [`QH008_TEXTVQA`](QH008_TEXTVQA/README.md) | TextVQA OCR 能力域检查 |

### 视觉适配、量子残差与关系路由

| 实验单元 | 内容 |
|---|---|
| [`QH009`](QH009/README.md) | 相关观测量门控 |
| [`QH010`](QH010/README.md) | Cayley 二量子比特适配器 |
| [`QH011`](QH011/README.md) | 视觉 prefill Cayley 适配器 |
| [`QH012`](QH012/README.md) | 四量子比特视觉 brickwork |
| [`QH013`](QH013/README.md) | 稀疏路由量子相关门控 |
| [`QH014`](QH014/README.md) | 问题条件单图量子锚点 |
| [`QH015`](QH015/README.md) | 问题条件量子 FiLM |
| [`QH016`](QH016/README.md) | 问题路由空间关系模块 |
| [`QH017`](QH017/README.md) | 量子态保真度关系路由 |
| [`QH018`](QH018/README.md) | 保守量子残差路由 |
| [`QH018_NOENT`](QH018_NOENT/README.md) | QH018 无纠缠乘积态多 seed 消融 |
| [`QH019`](QH019/README.md) | 六关系相干振幅混合 |
| [`QH020`](QH020/README.md) | 共享 trunk 后隔离训练的 12 参数 mixer |
| [`QH021`](QH021/README.md) | 精确共享主干下的 12 参数 mixer 因果审计 |

### 参数生成、低秩适配与层替代

| 实验单元 | 内容 |
|---|---|
| [`QH022`](QH022/README.md) | 推理常驻的量子 PEFT 系数生成器 |
| [`QH023_PROTO`](QH023_PROTO/README.md) | Pauli/Stiefel 正交子空间静态原型 |
| [`QH023`](QH023/README.md) | Pauli/Stiefel 方案的 EditCLEVR 全开发集正式评价 |
| [`QH024`](QH024/README.md) | 完整 `v_proj` 真替代 |
| [`QH025`](QH025/README.md) | end-to-end final-hidden priming |
| [`QH026`](QH026/README.md) | butterfly data-reupload true-layer replacement |
| [`QH029`](QH029/README.md) | raw-Qwen MPO/disentangler 可行性检查 |
| [`QH031`](QH031/README.md) | 跨层共享基真替代 |
| [`QH032`](QH032/README.md) | Pauli 可观测量非线性读出 |
| [`QH033`](QH033/README.md) | teacher-logit KL 第二阶段 |
| [`QH034`](QH034/README.md) | 共享奇异坐标量子谱调制 |
| [`QH035`](QH035/README.md) | 共享 RMSNorm 与运行时量子残差 |

### 完整 FFN 替代、量子残差演化与正式复验

| 实验单元 | 内容 |
|---|---|
| [`FFN036_AUDIT`](FFN036_AUDIT/README.md) | 64 层完整 FFN 删除敏感性审计 |
| [`QH037`](QH037/README.md) | 分组纠缠 QVAF 完整 FFN 替代 |
| [`QH038`](QH038/README.md) | shifted-ring 跨组 FC-VQC 完整 FFN 替代 |
| [`QH039`](QH039/README.md) | 冻结高压缩骨架上的可关闭量子残差 |
| [`QH040`](QH040/README.md) | 有界 trust-region 量子残差 |
| [`QH041`](QH041/README.md) | signed-X 读出有界量子残差 |
| [`QH042`](QH042/README.md) | signed-XZ 邻接相关观测残差 |
| [`QH043`](QH043/README.md) | 二次谐波 signed-X 数据重上传 |
| [`CC037_C4`](CC037_C4/README.md) | 经典高压缩 FFN 替代件的独立 C4 确认 |
| [`QH044_TRAIN`](QH044_TRAIN/README.md) | 从 QH037/CC037 优化器状态进行等预算续训 |
| [`QH044_FORMAL`](QH044_FORMAL/README.md) | QH044 冻结检查点 C4-128 正式因果评价 |
| [`QH045_TRAIN`](QH045_TRAIN/README.md) | 同起点 64 参数可选择纠缠训练 |
| [`QH045_FORMAL`](QH045_FORMAL/README.md) | QH045 冻结检查点 C4-64 正式评价 |

## 如何阅读实验结果

每个目录中的 `records/EXPERIMENT_RECORD.md` 是判断实验状态和结论边界的首要入口。阅读时应同时检查：

1. `status` 与 `purpose`：区分完整实验、部分实验、静态检查、失败或停止；
2. 数据锁与样本预算：确认训练集挖掘、验证和测试的边界；
3. `stage_plan.json`：确认分发入口实际执行的阶段；
4. `manifest.json`：核对源码、入口、证据和可移植改写的 SHA-256；
5. `artifacts/`：核对逐样本输出、统计分析或锁文件（若分发）；
6. checkpoint 清单：确认结果使用的参数确实存在并通过哈希验证。

不同实验的样本量、seed、训练预算和统计规则并不完全相同，因此不应仅凭跨目录的单个点估计对方案进行排序。正式比较应使用相同接口、冻结基座、数据划分、优化预算和评价协议。



## 许可证与第三方材料

本项目原创代码、文档以及发布者有权许可的新增参数采用 [MIT License](LICENSE)。MIT 许可允许学术研究和商业使用，但分发副本或实质性部分时须保留版权与许可声明。

Qwen3.8-27B 基座、数据集、软件依赖以及可能涉及第三方权利的派生参数仍遵守各自条款。本项目的 MIT 声明不会替第三方材料重新授权。完整说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 引用

如果本项目对你的研究有帮助，请在成果中标注本仓库及 **量观知元（QuSpect Technology Co., Ltd.）**。正式论文或技术报告公开后，本节将补充对应的 BibTeX 条目。
