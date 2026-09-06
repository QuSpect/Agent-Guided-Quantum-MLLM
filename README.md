<p align="center">
  <img src="assets/quspect-logo.jpg" alt="QuSpect Technology Co., Ltd. logo" width="240">
</p>

# Agent-Guided Quantum Replacement and Parameter-Efficient Adaptation for Multimodal Large Models

Agent-Guided Discovery of Quantum Replacements and Parameter-Efficient Adaptation Methods for Multimodal Large Language Models

[简体中文](readme_CN.md) | **English**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg)](https://pytorch.org/)

This repository releases experimental code for quantum replacements and parameter-efficient adaptation methods for multimodal large language models. The project uses a human-supervised Agent workflow to systematically explore parameterized quantum circuits and quantum-classical hybrid modules that can be inserted into multimodal large language models through source-code inspection, training-set example analysis, candidate specification, implementation evolution, and experimental review.

The current open-source release uses **Qwen3.8-27B** as a case study and comprises **independent experimental units** spanning candidate configurations, environment checks, frozen baselines, ablations, audits, training runs, and formal re-evaluations. The candidates cover visual residuals, question-conditioned routing, relation mixing, quantum-generated low-rank parameters, value-projection replacements, shared normalization, and complete FFN replacements.

This project is published and maintained by **QuSpect Technology Co., Ltd. (量观知元), Beijing, China**.

## Project Highlights

- **Agent-guided candidate search**: Candidates are continually generated and revised with respect to insertion sites, quantum encoding, circuit architecture, trainable parameters, readout methods, and training protocols.
- **Multiple model interfaces**: The repository covers visual-side residuals and routing, parameter-efficient adaptation, attention value-projection replacements, normalization replacements, and complete FFN replacements.
- **Independently distributed experimental units**: Each directory retains its own entry points, dependencies, stage plan, experimental records, evidence files, and license, and can be used independently in a new directory.
- **Complete experimental trajectory**: Successful runs, failed runs, no-entanglement ablations, deletion baselines, classical controls, and frozen-checkpoint re-evaluations are all retained.
- **Auditable portable rewrites**: Each `manifest.json` records the SHA-256 values of both the original and distributed files and marks rewrites introduced for path portability.
- **Fixed base model and data boundaries**: The case-study experiments fix the Qwen3.8-27B revision; the uses of training-set mining, frozen evaluation, and public test sets are distinguished in the corresponding experimental records.

## Research Scope

The quantum modules in this repository are implemented primarily with FP32/complex64 exact-statevector simulation on CUDA. The investigated structures include:

1. quantum residuals, anchors, FiLM, and relation routing over visual tokens or visual representations;
2. low-rank parameters and orthogonal subspaces generated or modulated by quantum circuits;
3. low-parameter replacements for linear layers such as `v_proj`;
4. cross-layer shared bases, spectral modulation, and shared RMSNorm;
5. compressed replacements for complete SwiGLU FFNs and the evolution of their quantum residuals;
6. controls or causal audits involving no entanglement, disabled quantum branches, layer deletion, and classical modules.

The experiments use ScienceQA, TextVQA, CLEVR, EditCLEVR, WikiText, and C4. Each dataset is used only in the relevant experimental units. The exact revision, split, sample budget, and preprocessing procedure are specified in the corresponding directory's `README.md`, `stage_plan.json`, and `records/EXPERIMENT_RECORD.md`.

## Repository Structure

```text
.
├─ ENVIRONMENT/                    # Environment and infrastructure audit
├─ SIMULATOR/                      # GPU exact-statevector microbenchmarks
├─ QWEN38_BASE/                    # Frozen-base loading and inference baseline
├─ QH*/                            # Quantum-classical hybrid candidates, ablations, and re-evaluations
├─ FFN036_AUDIT/                   # Complete-FFN deletion sensitivity audit
├─ CC037_C4/                       # Independent C4 re-evaluation of a classical high-compression replacement
├─ LICENSE                         # MIT License for original project content
├─ THIRD_PARTY_NOTICES.md          # Third-party terms for the base model, datasets, and dependencies
└─ readme_CN.md                    # Chinese project overview
```

Each experimental directory generally contains:

```text
<EXPERIMENT>/
├─ code/src/quantum_qwen38/        # Candidate modules and simulator implementations
├─ code/scripts/                   # Preprocessing, training, inference, and evaluation entry points
├─ artifacts/                      # Distributed lock files and experimental outputs, if available
├─ records/EXPERIMENT_RECORD.md    # Original status, protocol, results, and limitations
├─ manifest.json                   # Provenance, hashes, entry points, and evidence inventory
├─ stage_plan.json                 # Execution plan for the stages actually reached by this unit
├─ run_reached_stage.py            # Portable stage runner
├─ prepare_scheme_weights.py       # Scheme-weight preparation utility
├─ verify_parameters.py            # Parameter inventory and hash verification
├─ download_base_model.py          # Frozen-base download utility
├─ download_data.py                # Dataset download and preprocessing entry point
├─ commands.sh / commands.ps1      # Linux and Windows command lists
└─ README.md                       # Unit-level reproduction instructions
```

The code snapshots retained in individual experimental units are intentional. Later candidates may modify implementations with the same filename; independent snapshots prevent newer implementations from replacing the versions used in earlier experiments.

## Quick Start

### 1. Select an Experimental Unit

To begin with an environment or simulator check, use one of the following directories:

- [`ENVIRONMENT`](ENVIRONMENT/README.md): environment and infrastructure checks;
- [`SIMULATOR`](SIMULATOR/README.md): GPU exact-statevector implementation and microbenchmarks;
- [`QWEN38_BASE`](QWEN38_BASE/README.md): Qwen3.8-27B frozen-base loading and inference self-check.

For the entry points and reached stages of the remaining candidates, see the [Experimental Unit Index](#experimental-unit-index) below.

### 2. Create a Python Environment

The historical experimental environment used 4×A800-SXM4-80GB GPUs, Python 3.12, PyTorch 2.8/CUDA 12.8, and the Transformers 5.16 development line. The quantum training and inference entry points require an NVIDIA CUDA GPU and reject silent fallback to the CPU.

Linux shell:

```bash
cd QH014  # Replace with the target experimental directory
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
Set-Location QH014  # Replace with the target experimental directory
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Download the Frozen Base Model

The base-model weights are not distributed with this repository. Every experimental unit provides a download script with the same interface:

```bash
python download_base_model.py --output ./models/Qwen3.8-27B
```

The registered upstream repository is [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), fixed at the following revision:

```text
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

### 4. Prepare Scheme Weights

The GitHub code package does not duplicate large weight files. Experiments that require a checkpoint must first obtain the complete `models/<EXPERIMENT>/` directory from the corresponding parameter repository. Copy the parameter package's `manifest.json` into the experimental directory as `scheme_manifest.json`, then run:

```bash
python prepare_scheme_weights.py
python verify_parameters.py
```

Environment, static-check, or audit-only units without checkpoints state explicitly in their own README that this step may be skipped. When a parameter repository has not yet been released, only the code, stage plan, and distributed evidence can be reviewed; the unavailable trained weights cannot be reconstructed from those materials.

### 5. Download Data and Run the Reached Stage

```bash
python download_data.py --root ./datasets
python run_reached_stage.py
```

`run_reached_stage.py` reads the local `stage_plan.json` and executes only the preprocessing, training, frozen-evaluation, or analysis stages registered for that experiment. The commands in `commands.sh` or `commands.ps1` may also be reviewed and executed individually.

## Experimental Unit Index

### Infrastructure, Baselines, and Early Validation

| Experimental Unit | Description |
|---|---|
| [`ENVIRONMENT`](ENVIRONMENT/README.md) | Remote environment and infrastructure audit |
| [`SIMULATOR`](SIMULATOR/README.md) | GPU exact-statevector simulator microbenchmarks |
| [`QWEN38_BASE`](QWEN38_BASE/README.md) | Qwen3.8-27B loading and frozen baseline |
| [`QH001_INJECTION`](QH001_INJECTION/README.md) | Quantum injection into the actual model and a one-step optimization check |
| [`QH001B_STAGE_A`](QH001B_STAGE_A/README.md) | Stage-A experiment on 128 randomly selected ScienceQA examples |
| [`QH008_SCIENCEQA`](QH008_SCIENCEQA/README.md) | ScienceQA train-only hard-example mining |
| [`QH008_TEXTVQA`](QH008_TEXTVQA/README.md) | TextVQA OCR capability-domain check |

### Visual Adaptation, Quantum Residuals, and Relation Routing

| Experimental Unit | Description |
|---|---|
| [`QH009`](QH009/README.md) | Correlation-observable gating |
| [`QH010`](QH010/README.md) | Cayley two-qubit adapter |
| [`QH011`](QH011/README.md) | Visual-prefill Cayley adapter |
| [`QH012`](QH012/README.md) | Four-qubit visual brickwork |
| [`QH013`](QH013/README.md) | Sparsely routed quantum-correlation gating |
| [`QH014`](QH014/README.md) | Question-conditioned single-image quantum anchor |
| [`QH015`](QH015/README.md) | Question-conditioned quantum FiLM |
| [`QH016`](QH016/README.md) | Question-routed spatial-relation module |
| [`QH017`](QH017/README.md) | Quantum-state-fidelity relation routing |
| [`QH018`](QH018/README.md) | Conservative quantum-residual routing |
| [`QH018_NOENT`](QH018_NOENT/README.md) | Multi-seed no-entanglement product-state ablation for QH018 |
| [`QH019`](QH019/README.md) | Coherent amplitude mixing over six relations |
| [`QH020`](QH020/README.md) | A 12-parameter mixer trained separately after a shared trunk |
| [`QH021`](QH021/README.md) | Causal audit of the 12-parameter mixer under an exactly shared trunk |

### Parameter Generation, Low-Rank Adaptation, and Layer Replacement

| Experimental Unit | Description |
|---|---|
| [`QH022`](QH022/README.md) | Inference-resident quantum PEFT coefficient generator |
| [`QH023_PROTO`](QH023_PROTO/README.md) | Static Pauli/Stiefel orthogonal-subspace prototype |
| [`QH023`](QH023/README.md) | Formal full-development-set evaluation of the Pauli/Stiefel scheme on EditCLEVR |
| [`QH024`](QH024/README.md) | True replacement of a complete `v_proj` |
| [`QH025`](QH025/README.md) | End-to-end final-hidden priming |
| [`QH026`](QH026/README.md) | Butterfly data-reupload true-layer replacement |
| [`QH029`](QH029/README.md) | Raw-Qwen MPO/disentangler feasibility check |
| [`QH031`](QH031/README.md) | Cross-layer shared-basis true replacement |
| [`QH032`](QH032/README.md) | Nonlinear Pauli-observable readout |
| [`QH033`](QH033/README.md) | Second stage of teacher-logit KL training |
| [`QH034`](QH034/README.md) | Quantum spectral modulation in shared singular coordinates |
| [`QH035`](QH035/README.md) | Shared RMSNorm with a runtime quantum residual |

### Complete FFN Replacement, Quantum-Residual Evolution, and Formal Re-evaluation

| Experimental Unit | Description |
|---|---|
| [`FFN036_AUDIT`](FFN036_AUDIT/README.md) | Complete-FFN deletion sensitivity audit across 64 layers |
| [`QH037`](QH037/README.md) | Complete FFN replacement with grouped-entanglement QVAF |
| [`QH038`](QH038/README.md) | Complete FFN replacement with shifted-ring cross-group FC-VQC |
| [`QH039`](QH039/README.md) | Switchable quantum residual on a frozen high-compression scaffold |
| [`QH040`](QH040/README.md) | Bounded trust-region quantum residual |
| [`QH041`](QH041/README.md) | Bounded quantum residual with signed-X readout |
| [`QH042`](QH042/README.md) | Quantum residual with signed-XZ nearest-neighbor correlation observables |
| [`QH043`](QH043/README.md) | Second-harmonic signed-X data re-uploading |
| [`CC037_C4`](CC037_C4/README.md) | Independent C4 confirmation of a classical high-compression FFN replacement |
| [`QH044_TRAIN`](QH044_TRAIN/README.md) | Equal-budget continued training from the QH037/CC037 optimizer states |
| [`QH044_FORMAL`](QH044_FORMAL/README.md) | Formal causal evaluation of the frozen QH044 checkpoint on C4-128 |
| [`QH045_TRAIN`](QH045_TRAIN/README.md) | Selective-entanglement training with 64 parameters from the same starting point |
| [`QH045_FORMAL`](QH045_FORMAL/README.md) | Formal evaluation of the frozen QH045 checkpoint on C4-64 |

## How to Read the Experimental Results

The primary source for determining an experiment's status and the boundaries of its conclusions is `records/EXPERIMENT_RECORD.md` in the corresponding directory. When reading a record, check the following together:

1. `status` and `purpose`: distinguish complete experiments, partial experiments, static checks, failures, and stopped runs;
2. data locks and sample budgets: verify the boundaries among training-set mining, validation, and testing;
3. `stage_plan.json`: confirm the stages actually executed by the distributed entry point;
4. `manifest.json`: verify the SHA-256 values of source files, entry points, evidence, and portable rewrites;
5. `artifacts/`: inspect per-example outputs, statistical analyses, or lock files when distributed;
6. checkpoint inventory: confirm that the parameters used for a result exist and pass hash verification.

Sample sizes, seeds, training budgets, and statistical rules are not identical across experiments. Schemes therefore should not be ranked solely by individual point estimates taken from different directories. Formal comparisons should use aligned interfaces, frozen base models, data splits, optimization budgets, and evaluation protocols.

## License and Third-Party Materials

Original code and documentation in this project, together with newly introduced parameters that the publisher has the right to license, are released under the [MIT License](LICENSE). The MIT License permits academic and commercial use, but copies or substantial portions must retain the copyright and license notices.

The Qwen3.8-27B base model, datasets, software dependencies, and derived parameters that may involve third-party rights remain subject to their respective terms. This project's MIT declaration does not relicense third-party materials. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the complete statement.

## Citation

If you use this code or build upon our work in your research, please cite the following preprint:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22301614.svg)](https://doi.org/10.5281/zenodo.22301614)

```bibtex
@misc{quspect2026agentguided,
  author       = {{QuSpect Technology Co., Ltd., Beijing, China}},
  title        = {{Agent-Guided Discovery of Quantum Replacements and Parameter-Efficient Adaptation Methods for Multimodal Large Language Models}},
  year         = {2026},
  month        = sep,
  howpublished = {Zenodo},
  doi          = {10.5281/zenodo.22301614},
  url          = {https://doi.org/10.5281/zenodo.22301614},
  note         = {Preprint, version v1}
}
```

## Citation

If this project contributes to your research, please acknowledge this repository and **QuSpect Technology Co., Ltd. (量观知元)** in the resulting work. A BibTeX entry will be added here after the formal paper or technical report is made public.
