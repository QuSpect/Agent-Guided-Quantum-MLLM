# QH008_TEXTVQA：从零复现实验代码

本目录对应 EXP-006（TextVQA OCR 第二能力域）。代码副本可在任意新目录使用；模型、数据、缓存和输出均采用相对路径，不需要原服务器。模型结构、数据划分、seed、训练预算和评价定义保持历史登记值，只有路径与缓存位置被改为可移植写法。

## 文件布局

```text
./
├─ models/Qwen3.8-27B/          # 官方冻结基座
├─ models/QH008_TEXTVQA/      # 从对应 Hugging Face 参数包取得
├─ datasets/                    # 下载和预处理结果
├─ artifacts/                   # 训练、推理与评价输出
├─ code/src/quantum_qwen38/     # 候选与模拟器实现
└─ code/scripts/                # 原实验入口的可移植副本
```

## 0. 硬件与环境

- NVIDIA CUDA GPU；历史环境为 4×A800-SXM4-80GB。
- Python 3.12、PyTorch 2.8/CUDA 12.8、Transformers 5.16 开发线。
- 量子训练与推理使用 CUDA FP32/complex64 exact-statevector；CPU fallback 被拒绝。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 1. 下载官方冻结基座

官方模型：[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。

```bash
python download_base_model.py --output ./models/Qwen3.8-27B
```

固定 revision：`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。可选镜像：

```bash
python download_base_model.py --output ./models/Qwen3.8-27B --endpoint https://hf-mirror.com
```

## 2. 放置方案权重

从对应 `QH008_TEXTVQA` Hugging Face 参数包取得整个 `models/QH008_TEXTVQA/`，保持相同相对层级：

```text
./models/QH008_TEXTVQA/checkpoints/...
```

本 GitHub 代码包不重复提交大权重，也不捏造尚未发布的公网方案 URL。发布到组织账号后，可把实际仓库 ID 代入：

```bash
hf download YOUR_ORG/QH008_TEXTVQA --include "models/QH008_TEXTVQA/**" --local-dir .
```

然后把对应 Hugging Face 包的 `manifest.json` 复制为项目根目录的 `scheme_manifest.json`，运行 `python verify_parameters.py`。无 checkpoint 的环境/审计单元跳过此步。

## 3. 下载登记数据并预处理

```bash
python download_data.py --root ./datasets
```

数据源、revision、许可和 split 以原数据注册表为准。公开 test 不参与搜索。若在 AutoDL 环境使用学术加速，仅在下载命令前临时 `source /etc/network_turbo`，下载后立即 `unset http_proxy https_proxy`。

## 4. 运行已达到阶段

```bash
python prepare_scheme_weights.py
python run_reached_stage.py
```

登记入口：`run_textvqa_stage_a.py`。失败、停止或纯审计方案只复现实际达到的 static/smoke/训练/冻结评价阶段；不会把未运行的自由生成伪装为已支持。成功或部分成功方案沿用登记的完整模型前向和真实数据冻结推理。

`run_reached_stage.py` 读取 `stage_plan.json`，依次执行下列展开命令：

```bash
python code/scripts/run_textvqa_stage_a.py --candidate base
python code/scripts/run_textvqa_stage_a.py --candidate qh008
python code/scripts/run_textvqa_stage_a.py --candidate cc008
python code/scripts/run_textvqa_stage_a.py --candidate qh008_noent
python code/scripts/analyze_textvqa_stage_a.py
```

一键顺序另存为 `commands.sh` 与 `commands.ps1`。

## 5. 基座推理自检

```bash
python base_inference.py --model ./models/Qwen3.8-27B --prompt "给出一句测试回答。"
```

该命令只检查基座下载与 Transformers 环境；方案成绩必须由第 4 步入口产生。

## 历史实现

| 主要源码 |
| --- |
| `code/src/quantum_qwen38/quantum_gated_low_rank.py` |
| `code/src/quantum_qwen38/vqa_metric.py` |

| 原实验入口 |
| --- |
| `code/scripts/analyze_textvqa_stage_a.py` |
| `code/scripts/audit_textvqa_pilot_overlap.py` |
| `code/scripts/mine_textvqa_hard_train.py` |
| `code/scripts/run_textvqa_stage_a.py` |
| `code/scripts/test_qh008_gpu.py` |

`manifest.json` 同时保存原始哈希和路径可移植副本哈希。性能事实与档位见 `../../reports/QH008_TEXTVQA.md`。

## 许可证

本项目原创内容与有权许可的新增参数采用 [MIT License](LICENSE)，允许学术研究和商业使用，并须保留版权及许可声明。基座模型、数据集、依赖和涉及第三方权利的派生参数仍遵守各自条款，见 [许可范围与第三方材料](THIRD_PARTY_NOTICES.md)。
