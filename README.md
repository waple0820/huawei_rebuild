# Kimi2-PCL

`Kimi2-PCL` 是一个基于 kimi2 模型进行修改的模型架构仓库, 模型定义代码参考 Huggingface 模型[Kimi-K2-Base](https://huggingface.co/moonshotai/Kimi-K2-Base), 权重转换脚本参考 Mindspeed-LLM 脚本[DeppSeek](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.2.0/examples/mcore/deepseek3/convert_ckpt_deepseek3.py)：

- [models](./models) 提供模型的定义与配置（Hugging Face 风格的 config / generation config + Python 实现）。
- [utils](./utils) 提供权重格式转换脚本（HF safetensors ↔ Megatron-Core/MCore checkpoint）。
- [scripts](./scripts) 提供预训练启动脚本（torchrun + Megatron/MindSpeed 生态）。

## 目录结构

```text
Kimi2-PCL/
  models/
    config.json
    generation_config.json
    configuration_model.py
    modeling_deepseek.py
  utils/
    convert_ckpt_hf2mcore.py
    convert_ckpt_mcore2hf.py
  scripts/
    pretrain_kimi2_100b_4k.sh
    pretrain_kimi2_1000b_4k.sh
```

## 模型与配置（models）

- `config.json` / `generation_config.json`：Hugging Face 模型配置与 generation 配置。
- `configuration_model.py`：定义了 `DeepseekV3Config`（`model_type = "kimi_k2"`）。
- `modeling_deepseek.py`：完整的 DeepSeek HF 实现代码（包含 RMSNorm / RoPE / MoE MLP 等核心结构）。

注意：`config.json` 中的 `auto_map` 指向 `configuration_deepseek.*` / `modeling_deepseek.*` 的类名与文件名。若你希望直接通过 `transformers.AutoModel*` + `trust_remote_code=True` 加载本地模型目录，需要保证 `auto_map` 指向的文件名与类名在 `models/` 下可被正确解析（当前仓库的文件命名与类名可能需要对齐）。

## 权重转换（utils）

本仓库提供两类转换：

- HF（`model.safetensors*` + `model.safetensors.index.json`）→ MCore/Megatron checkpoint 目录（`iter_xxxxxxx/mp_rank_*/model_optim_rng.pt`）
- MCore/Megatron checkpoint → HF safetensors 分片与 index

### HF → MCore

入口脚本：[convert_ckpt_hf2mcore.py](./utils/convert_ckpt_hf2mcore.py)

```bash
python utils/convert_ckpt_hf2mcore.py \
  --load-dir /path/to/hf_model_dir \
  --save-dir /path/to/mcore_ckpt_dir \
  --num-layers 48 \
  --target-tensor-parallel-size 2 \
  --target-pipeline-parallel-size 8 \
  --target-expert-parallel-size 64 \
  --moe-grouped-gemm \
  --moe-tp-extend-ep \
  --mla-mm-split \
  --schedules-method dualpipev \
  --noop-layers 47
```

常用参数说明：

- `--load-dir`：Hugging Face 模型目录（包含 `model.safetensors.index.json`）。
- `--save-dir`：输出 checkpoint 目录（脚本会创建 `iter_0000001/` 并写入 `latest_checkpointed_iteration.txt`）。
- `--target-*-parallel-size`：目标 TP/PP/EP 并行度。
- `--moe-grouped-gemm` / `--moe-tp-extend-ep`：MoE 权重布局相关开关。
- `--mla-mm-split`：将 MLA 的部分投影拆分为更多矩阵以匹配训练侧实现。
- `--vpp-stage` / `--schedules-method dualpipev`：VPP/dualpipev 相关配置。
- `--num-layer-list`：当 `num_layers` 不能整除 `pp_size` 时，显式给定每个 PP 的层数分配（形如 `4,4,4,4`）。
- `--noop-layers`：指定 noop layer 索引列表（逗号分隔），用于与训练侧的 “跳层/空层” 配置对齐。
- `--qlora-nf4`：输出层权重做 bitsandbytes nf4 量化（需要额外安装 `bitsandbytes`）。

### MCore → HF

入口脚本：[convert_ckpt_mcore2hf.py](./utils/convert_ckpt_mcore2hf.py)

```bash
python utils/convert_ckpt_mcore2hf.py \
  --load-dir /path/to/mcore_ckpt_dir \
  --save-dir /path/to/hf_model_dir \
  --num-layers 48 \
  --source-tensor-parallel-size 2 \
  --source-pipeline-parallel-size 8 \
  --source-expert-parallel-size 64 \
  --moe-grouped-gemm \
  --moe-tp-extend-ep \
  --mla-mm-split \
  --schedules-method dualpipev \
  --vpp-stage 3 \
  --noop-layers 47 \
  --rotary-base 50000
```

输出产物包括：

- `model-00001-of-0000XX.safetensors` 分片文件
- `model.safetensors.index.json`

## 训练脚本（scripts）

脚本：

- [pretrain_kimi2_100b_4k.sh](./scripts/pretrain_kimi2_100b_4k.sh)
- [pretrain_kimi2_1000b_4k.sh](./scripts/pretrain_kimi2_1000b_4k.sh)

两者都是示例启动脚本，核心行为是调用：

```bash
torchrun ... pretrain_gpt.py ...
```

其中 `pretrain_gpt.py` 以及相关的运行时依赖（例如 `mindspeed_llm.*`、Megatron/MCore 等）不在本仓库内，通常来自你的训练框架工程或运行环境。

脚本中常见依赖的环境变量（按你的集群/启动器而定）：

- 分布式：`LOCAL_WORLD_SIZE`、`server_count`、`RANK`、`MASTER_ADDR`、`MASTER_PORT`
- Tokenizer：`TOKENIZER_PATH`
- 数据：`DATA_PREFIXES`（以及部分脚本里使用的 `DATA_DIR`）
- Checkpoint：`CKPT_LOAD_DIR`、`CKPT_SAVE_DIR`
- 日志：`TRAIN_LOG_PATH`

## 开发与代码质量

仓库包含基础的代码质量工具配置：

- `.pre-commit-config.yaml`：pre-commit hooks
- `.flake8`：flake8 配置

如需启用：

```bash
pip install pre-commit
pre-commit install
```

## 许可证

见 [LICENSE](./LICENSE)。
