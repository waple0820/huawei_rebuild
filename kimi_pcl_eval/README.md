# Kimi-K2 1T Evaluation Toolkit

Automated vLLM inference validation, OpenCompass text benchmark evaluation, and VLMEvalKit multimodal benchmark evaluation for the **Kimi-K2 1T** model (DeepSeek-V3 architecture, ~1000B parameters).

## Overview

This toolkit provides three capabilities:

1. **vLLM Smoke Test** — verify the model can generate text and understand images end-to-end.
2. **OpenCompass Text Evaluation** — run 6 core text benchmarks covering knowledge, code, math, Chinese, safety, and reasoning.
3. **VLMEvalKit Multimodal Evaluation** — run 3 multimodal benchmarks covering image understanding and video comprehension.

### Text Benchmarks (OpenCompass)

| Benchmark | Domain | Description |
|-----------|--------|-------------|
| MMLU | English Knowledge | Multitask accuracy across 57 subjects |
| C-Eval | Chinese Knowledge | Chinese multi-subject examination |
| HumanEval | Code | Python function completion |
| GSM8K | Math | Grade-school math word problems |
| TruthfulQA | Safety | Resistance to generating false answers |
| HellaSwag | Reasoning | Commonsense sentence completion |

### Multimodal Benchmarks (VLMEvalKit)

| Benchmark | Type | Description |
|-----------|------|-------------|
| MMBench | Image (basic) | Visual language model comprehensive evaluation |
| MMMU | Image (hard) | University-level multi-discipline visual reasoning |
| Video-MME | Video | Long/short video comprehensive understanding |

## Prerequisites

- **Python** >= 3.10
- **CUDA** >= 12.1 with compatible NVIDIA drivers
- **Hardware**: Multi-node GPU cluster (e.g. 4x8 A800-80G or similar). A single 8-GPU node is **not** sufficient for the full 1T model — tensor parallelism across 32+ GPUs is typically required.
- The model checkpoint in HuggingFace `.safetensors` format, including custom model code (`modeling_deepseek.py`, `configuration_deepseek.py`, etc.) in the model directory.

## Installation

```bash
pip install -r requirements.txt
```

> **Note**: `torch`, `transformers`, `safetensors`, `mmengine` etc. are transitive dependencies of `vllm` / `opencompass` and will be installed automatically. Install the CUDA-matched PyTorch wheel first if your cluster requires a specific build.

## Dataset Preparation

Text evaluation datasets are pre-downloaded in the `data/` directory and are read **locally** — no internet access is required at evaluation time.

```
data/
├── mmlu/         # MMLU
├── ceval/        # C-Eval
├── humaneval/    # HumanEval
├── gsm8k/        # GSM8K
├── truthfulqa/   # TruthfulQA
├── hellaswag/    # HellaSwag
└── ...           # additional benchmarks included in the core package
```

If starting from scratch on a networked machine, run `bash download_datasets.sh` to populate the `data/` directory, then copy it to the evaluation node.

OpenCompass reads datasets from `./data` by default (symlinked into the OpenCompass package at launch time by `run_eval.sh`).

## Quick Start: vLLM Smoke Test

### Text Generation

```bash
python test_vllm_generation.py --model_path /path/to/model --tp_size 8

# Or use the convenience wrapper
bash run_vllm_test.sh --model_path /path/to/model --tp_size 32
```

### Multimodal (Image) Generation

```bash
# Prepare a test image, then:
python test_vllm_multimodal.py \
    --model_path /path/to/model \
    --tp_size 32 \
    --image_path test_image.jpg
```

### Smoke Test Options

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--model_path` | `MODEL_PATH` | `/llm_workspace_1P/wf/ckpt/iter_0000900_test/` | HuggingFace model directory |
| `--tp_size` | `TP_SIZE` | `8` | Tensor-parallel size |
| `--max_tokens` | — | `256` | Max tokens to generate |
| `--temperature` | — | `0.7` (text) / `0.2` (multimodal) | Sampling temperature |
| `--quantization` | — | `None` | Quantization method (`awq` / `gptq` / `fp8`) |
| `--image_path` | — | `test_image.jpg` | Test image (multimodal only) |

## Text Evaluation: OpenCompass Benchmarks

Run all 6 text benchmarks:

```bash
bash run_eval.sh --model_path /path/to/model --tp_size 32

# Custom data and output directories
bash run_eval.sh \
    --model_path /path/to/model \
    --tp_size 32 \
    --data_dir /data/opencompass_data \
    --work_dir /data/eval_results
```

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--model_path` | `MODEL_PATH` | `/llm_workspace_1P/wf/ckpt/iter_0000900_test/` | Model directory |
| `--tp_size` | `TP_SIZE` | `8` | Tensor-parallel size |
| `--work_dir` | `WORK_DIR` | `./outputs` | Results output directory |
| `--data_dir` | `DATA_DIR` | `./data` | Local dataset directory |
| `--max_out_len` | `MAX_OUT_LEN` | `1024` | Max output tokens per sample |
| `--max_seq_len` | `MAX_SEQ_LEN` | `4096` | Max sequence length |
| `--batch_size` | `BATCH_SIZE` | `16` | Evaluation batch size |

## Multimodal Evaluation: VLMEvalKit

Evaluate on MMBench, MMMU, and Video-MME via VLMEvalKit:

```bash
bash run_vl_eval.sh --model_path /path/to/model --tp_size 32
```

The script automatically:
1. Clones VLMEvalKit (if not already present).
2. Installs the Kimi-K2 model wrapper (`vlmeval_wrapper/kimi_k2.py`) into VLMEvalKit.
3. Registers the model in VLMEvalKit's config.
4. Runs the 3 multimodal benchmarks.

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--model_path` | `MODEL_PATH` | `/llm_workspace_1P/wf/ckpt/iter_0000900_test/` | Model directory |
| `--tp_size` | `TP_SIZE` | `8` | Tensor-parallel size |
| `--work_dir` | `WORK_DIR` | `./outputs_vl` | Results output directory |
| `--vlmeval_dir` | `VLMEVAL_DIR` | `./VLMEvalKit` | VLMEvalKit repo location |

## Repository Structure

```
kimi_pcl_eval/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies (minimal)
├── .gitignore                        # Git ignore rules
├── doc.md                            # Internal design document
│
├── test_vllm_generation.py           # Text inference smoke test
├── test_vllm_multimodal.py           # Multimodal inference smoke test
│
├── configs/
│   └── eval_kimi_1t_vllm.py         # OpenCompass evaluation config
├── run_eval.sh                       # One-click text evaluation
├── run_vllm_test.sh                  # Convenience wrapper for text smoke test
│
├── vlmeval_wrapper/
│   └── kimi_k2.py                   # VLMEvalKit model wrapper for Kimi-K2
├── run_vl_eval.sh                    # One-click multimodal evaluation
│
├── download_datasets.sh              # Download datasets (run on networked machine)
└── data/                             # Pre-downloaded evaluation datasets
    ├── mmlu/
    ├── ceval/
    ├── humaneval/
    ├── gsm8k/
    ├── truthfulqa/
    ├── hellaswag/
    └── ...
```

## Troubleshooting

### OOM (Out of Memory)

The 1T model requires substantial GPU memory. Solutions:

- **Increase TP size**: Use `--tp_size 32` or higher to spread weights across more GPUs.
- **Enable quantization**: Pass `--quantization fp8` (or `awq` / `gptq`) to reduce memory footprint.
- **Reduce batch size**: Lower `--batch_size` in evaluation to reduce activation memory.

### `trust_remote_code` Errors

This model uses custom architecture code (`modeling_deepseek.py`). The toolkit already sets `trust_remote_code=True` in all loading paths. If you see related errors, ensure the model directory contains `modeling_deepseek.py` and `configuration_deepseek.py`.

### vLLM Version Compatibility

If vLLM fails to load the model, check that your vLLM version supports the DeepSeek-V3 / MoE architecture:

```bash
pip install -U vllm
```

### Dataset Not Found

Datasets are read from the local `./data` directory by default. Verify:

1. `ls data/` shows subdirectories like `mmlu/`, `ceval/`, `gsm8k/`, etc.
2. The `--data_dir` flag points to the correct location if customized.
3. `run_eval.sh` automatically symlinks the data directory into OpenCompass at launch.

### VLMEvalKit Issues

If `run_vl_eval.sh` fails to register the model:

1. Verify VLMEvalKit is cloned: `ls VLMEvalKit/`
2. Check the wrapper was copied: `ls VLMEvalKit/vlmeval/vlm/kimi_k2.py`
3. Check registration: `grep KimiK2 VLMEvalKit/vlmeval/vlm/__init__.py`

## License

This project is part of the Kimi-K2 PCL internal evaluation infrastructure.
