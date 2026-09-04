# 许可范围与第三方材料

## 本项目内容

本目录的原创实验代码、文档，以及发布者有权许可的新增适配器/替代模块参数，按 [MIT License](LICENSE) 提供。参数是否存在、实际完成阶段和运行限制以本方案 README 与参数清单为准。没有训练权重的检查单元不会因添加许可证而变成可用模型。

标准 MIT 允许学术研究和商业使用，不含“仅限学术”“禁止商业使用”或强制引用论文的附加条件。分发副本或实质性部分时，须保留版权声明和许可文本；内容按现状提供，不保证实验结果或特定用途的适用性。

本许可只覆盖发布者拥有许可权的内容，不替第三方重新授权。由基座权重分解、提取或修改得到的参数，如涉及原权利人的权利，仍须同时遵守基座条款；本项目的 MIT 声明不豁免这些义务。

## 基座模型

冻结基座由使用者另行从 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 下载，不包含在本参数包中。使用或再次分发基座及其衍生内容，应检查实际下载版本中的 LICENSE、NOTICE 和模型卡。这里不把基座重新声明为 MIT，也不以项目名称推定其具体许可证。

## 数据集

各方案只使用其中相关数据；具体下载版本、划分和处理步骤见本包下载脚本、运行说明及实验记录。

| 来源 | 原始说明与许可入口 |
|---|---|
| WikiText | [Salesforce/wikitext 数据卡](https://huggingface.co/datasets/Salesforce/wikitext) |
| C4 | [allenai/c4 数据卡](https://huggingface.co/datasets/allenai/c4) |
| TextVQA | [facebook/textvqa 数据卡](https://huggingface.co/datasets/facebook/textvqa) |
| ScienceQA | [原项目](https://github.com/lupantech/ScienceQA)；[本实验下载来源](https://huggingface.co/datasets/lmms-lab/ScienceQA) |
| CLEVR | [原项目](https://cs.stanford.edu/people/jcjohns/clevr/) |
| EditCLEVR | [本实验下载来源](https://huggingface.co/datasets/torux/EditCLEVR) |

下载、预处理或划分数据，不会把原数据、图片或文本的许可变成 MIT。再次分发样本时，应保留原来源、署名和所需许可，并检查底层材料可能附带的条件。来源清单不表示本项目拥有这些数据的版权。

## 软件依赖

`requirements.txt` 中的 PyTorch、Transformers、Accelerate、Hugging Face Hub、Datasets、Safetensors、Pillow、NumPy、tqdm，以及 GPU/CUDA 运行组件，各自遵循实际安装版本的许可证。它们不因本项目使用 MIT 而被重新授权。若将依赖一起打包，请保留相应 LICENSE/NOTICE；不能只复制本项目这一份 LICENSE。

## 规范参考

许可正文采用 [Open Source Initiative 的 MIT 标准文本](https://opensource.org/license/mit)。模型卡使用 `license: mit` 标记本项目许可，写法参照 [Hugging Face 许可元数据文档](https://huggingface.co/docs/hub/repositories-licenses)；该标记不能覆盖上述第三方权利。
