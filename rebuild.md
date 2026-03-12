# 1000B GQA 模型重构开发文档

> 本文档是实时进度跟踪文档，每个章节对应一个 commit 粒度的工作。
> 勾选框 `[x]` 表示已完成，`[ ]` 表示待完成。

---

## 零、背景与现状分析

### 0.1 核心事实

1000B 训练脚本 `scripts/pretrain_kimi2_1000b_4k.sh` 最终执行的 `torchrun` 命令为：

```bash
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $GQA_ARGS \      # <-- 使用 GQA，而非 $MLA_ARGS
    $DUALPIPE_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $DATA_ARGS \
    --save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl
```

关键：**`$MLA_ARGS` 虽然在脚本中定义了，但从未传入 `torchrun`**。1000B 模型实际使用的是纯 GQA（Grouped Query Attention）架构。

### 0.2 1000B 模型正确参数（从训练脚本提取）

| 参数来源 | 参数名 | 值 | 说明 |
|---------|--------|-----|------|
| GPT_ARGS | hidden_size | 7168 | |
| GPT_ARGS | ffn_hidden_size | 18432 | dense 层 intermediate_size |
| GPT_ARGS | num_layers | 48 | 含 1 个 noop layer |
| GPT_ARGS | noop_layers | 47 | 第 47 层为空层 |
| GPT_ARGS | vocab_size | 163840 | |
| GPT_ARGS | max_position_embeddings | 131072 | |
| GPT_ARGS | rotary_base | 50000 | |
| GPT_ARGS | norm_epsilon | 1e-6 | |
| GQA_ARGS | num_attention_heads | 112 | Q heads |
| GQA_ARGS | num_query_groups | 4 | GQA 分组数 |
| GQA_ARGS | qk_layernorm | True | Q/K 投影后加 RMSNorm |
| MOE_ARGS | num_experts | 128 | |
| MOE_ARGS | moe_ffn_hidden_size | 8192 | expert intermediate_size |
| MOE_ARGS | moe_router_topk | 2 | |
| MOE_ARGS | first_k_dense_replace | 3 | 前 3 层为 dense MLP |
| MOE_ARGS | n_shared_experts | 1 | |
| ROPE_ARGS | rope_scaling_type | yarn | |
| ROPE_ARGS | rope_scaling_factor | 32 | |
| ROPE_ARGS | original_max_position_embeddings | 4096 | |
| ROPE_ARGS | beta_fast / beta_slow | 1 / 1 | |
| ROPE_ARGS | mscale / mscale_all_dim | 1.0 / 1.0 | |
| 并行配置 | TP / PP / EP | 2 / 8 / 64 | |
| DUALPIPE_ARGS | schedules_method | dualpipev | |

### 0.3 由训练参数推导的关键维度

```
num_kv_heads   = num_attention_heads / num_query_groups = 112 / 4 = 28
head_dim       = hidden_size / num_attention_heads      = 7168 / 112 = 64
num_real_layers = num_layers - len(noop_layers)         = 48 - 1 = 47

Q 投影输出维度: num_attention_heads * head_dim      = 112 * 64 = 7168
K 投影输出维度: num_kv_heads * head_dim             = 28 * 64  = 1792
V 投影输出维度: num_kv_heads * head_dim             = 28 * 64  = 1792
O 投影:        (hidden_size, num_attention_heads * head_dim) = (7168, 7168)

RoPE 维度:     head_dim = 64  (GQA 对整个 head_dim 施加 RoPE)
softmax_scale: head_dim^(-0.5) = 64^(-0.5) = 0.125  (乘以 mscale^2)
```

### 0.4 当前仓库问题清单

| # | 组件 | 问题 | 严重程度 |
|---|------|------|---------|
| 1 | `models/config.json` | 全部为 100B/MLA 参数 (hidden_size=4096, heads=128, 含 MLA 字段) | 致命 |
| 2 | `models/configuration_model.py` | 默认值为 MLA 参数；文件名与 `auto_map` 不匹配 (`auto_map` 指向 `configuration_deepseek`) | 致命 |
| 3 | `models/modeling_deepseek.py` | Attention 为完整 MLA 实现 (q_a_proj / kv_a_proj_with_mqa / q_b_proj / kv_b_proj 等)；import 路径指向 `configuration_deepseek` (文件不存在) | 致命 |
| 4 | `utils/convert_ckpt_mcore2hf.py` | 顶部硬编码 MLA 常量；`_set_layer_attn` 只从 `models[(0,0)]` 读 `linear_qkv`（丢失 TP 分片）；输出 MLA 格式 HF 权重 | 致命 |
| 5 | `utils/convert_ckpt_hf2mcore.py` | 顶部硬编码 MLA 常量；`_set_layer_attn` 读 MLA 格式 HF 权重 (q_a_proj 等)；MLA 逻辑写入 MCore | 致命 |
| 6 | `.gitignore` | 缺少 `rebuild.md` | 低 |

### 0.5 原 rebuild.md 勘误

| 原文内容 | 修正 |
|---------|------|
| `head_dim=128` | **应为 64**。`head_dim = hidden_size / num_attention_heads = 7168 / 112 = 64` |
| `q_dim = 112 * 128`, `kv_dim = 28 * 128` | **应为 `112 * 64 = 7168`, `28 * 64 = 1792`** |
| QKV TP gather 直接 `torch.cat(shards, dim=0)` | MCore GQA 的 QKV 每个 TP rank 内部是 `[Q_rank \| K_rank \| V_rank]` 交织排列，需要先拆分再按类别拼接 |
| 只覆盖了 `mcore2hf` | **`hf2mcore` 同样需要改造** |
| 未提及 RoPE 维度变化 | MLA 对 `qk_rope_head_dim=64` 施加 RoPE；GQA 对完整 `head_dim=64` 施加 RoPE |
| 未提及 softmax_scale 变化 | MLA: `(128+64)^(-0.5)`；GQA: `64^(-0.5)` |
| 未提及文件命名不匹配 | `auto_map` 指向 `configuration_deepseek.DeepseekV3Config`，实际文件名为 `configuration_model.py` |
| 未提及 FlashAttention2 类 | `DeepseekV3FlashAttention2` 同样基于 MLA，需同步改造 |
| 未提及 qk-layernorm 保留 | 训练脚本 GQA_ARGS 包含 `--qk-layernorm`，HF 模型需支持 |

### 0.6 MCore GQA 权重布局（关键）

MCore 中 GQA 模式的 `linear_qkv.weight` 为 ColumnParallel，每个 TP rank 存储：

```
Per TP rank shape: (q_per_tp + k_per_tp + v_per_tp, hidden_size)

其中:
  q_per_tp  = (num_attention_heads / tp_size) * head_dim = 56 * 64 = 3584
  k_per_tp  = (num_kv_heads / tp_size) * head_dim        = 14 * 64 = 896
  v_per_tp  = (num_kv_heads / tp_size) * head_dim        = 14 * 64 = 896
  total_per_tp = 5376
```

每个 TP rank 内的行排列顺序（MCore 标准 GQA interleaving）：

```
对每个 KV head group (共 kv_heads_per_tp = 14 个):
  Q heads: group_factor * head_dim 行   (4 * 64 = 256)
  K head:  head_dim 行                  (64)
  V head:  head_dim 行                  (64)
  每组共: (group_factor + 2) * head_dim = 384 行

14 组 * 384 = 5376 行 ✓
```

> **重要提示**: 上述 interleaving 布局为 MCore 标准实现。具体行为可能因 MindSpeed-LLM 版本而异。
> **建议在 Commit 3 开始前先 dump 一个实际 checkpoint 的 `linear_qkv.weight` 的 shape，确认是否为 interleaved。**
> 如果为 sequential 布局（`[Q_all | K_all | V_all]`），则拆分更简单。

**拆分方式 A — Interleaved 布局**：

```python
group_factor = num_attention_heads // num_kv_heads  # 4
kv_per_tp = num_kv_heads // tp_size                 # 14

qkv_rank = weight.reshape(kv_per_tp, group_factor + 2, head_dim, hidden_size)
q_rank = qkv_rank[:, :group_factor, :, :].reshape(-1, hidden_size)
k_rank = qkv_rank[:, group_factor, :, :].reshape(-1, hidden_size)
v_rank = qkv_rank[:, group_factor + 1, :, :].reshape(-1, hidden_size)
```

**拆分方式 B — Sequential 布局**：

```python
q_per_tp = (num_attention_heads // tp_size) * head_dim
kv_per_tp = (num_kv_heads // tp_size) * head_dim
q_rank, k_rank, v_rank = torch.split(weight, [q_per_tp, kv_per_tp, kv_per_tp], dim=0)
```

其它 MCore attention 权重：

| MCore key | 并行方式 | shape (per rank) | 说明 |
|-----------|---------|-------------------|------|
| `linear_qkv.weight` | ColumnParallel, dim=0 split | (5376, 7168) | 如上 |
| `linear_proj.weight` | RowParallel, dim=1 split | (7168, 3584) | Gather 沿 dim=1 |
| `q_layernorm.weight` | 不切分 | (64,) | head_dim 维的 RMSNorm |
| `k_layernorm.weight` | 不切分 | (64,) | head_dim 维的 RMSNorm |

> 注意: MLA 模式下 layernorm key 为 `q_layernorm` / `kv_layernorm`，GQA 模式下可能为 `q_layernorm` / `k_layernorm`。
> 需用实际 checkpoint 验证 key name。

---

## 一、Commit 1: 配置文件与文件命名修正

**范围**: `models/config.json`, `models/configuration_model.py`, `models/modeling_deepseek.py` (仅 import)

### 1.1 任务清单

- [ ] 1.1.1 将 `models/configuration_model.py` 重命名为 `models/configuration_deepseek.py`（与 `auto_map` 一致）
- [ ] 1.1.2 更新 `models/modeling_deepseek.py` 第 49 行 import（当前已经是 `from .configuration_deepseek import ...`，重命名后无需改动）
- [ ] 1.1.3 重写 `models/configuration_deepseek.py` 的 `DeepseekV3Config`
- [ ] 1.1.4 重写 `models/config.json` 为 1000B GQA 参数
- [ ] 1.1.5 删除 `models/modeling_deepseek_bak.py`（过时的备份文件）

### 1.2 详细修改

#### 1.2.1 重命名文件

```bash
git mv models/configuration_model.py models/configuration_deepseek.py
rm models/modeling_deepseek_bak.py
```

#### 1.2.2 重写 `models/configuration_deepseek.py`

移除 MLA 专有字段（`q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`），新增 GQA 字段（`num_key_value_heads`, `qk_layernorm`）。

```python
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DeepseekV3Config(PretrainedConfig):
    model_type = 'kimi_k2'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        vocab_size=163840,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=8192,
        num_hidden_layers=47,
        num_nextn_predict_layers=0,
        num_attention_heads=112,
        num_key_value_heads=28,
        n_shared_experts=1,
        n_routed_experts=128,
        ep_size=1,
        routed_scaling_factor=2.827,
        topk_method='noaux_tc',
        n_group=8,
        topk_group=2,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=3,
        norm_topk_prob=True,
        scoring_func='sigmoid',
        aux_loss_alpha=0.001,
        seq_aux=True,
        qk_layernorm=True,
        hidden_act='silu',
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=163584,
        eos_token_id=163585,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=50000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.qk_layernorm = qk_layernorm
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
```

#### 1.2.3 重写 `models/config.json`

```json
{
  "architectures": ["DeepseekV3ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_deepseek.DeepseekV3Config",
    "AutoModel": "modeling_deepseek.DeepseekV3Model",
    "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM"
  },
  "aux_loss_alpha": 0.001,
  "bos_token_id": 163584,
  "eos_token_id": 163585,
  "first_k_dense_replace": 3,
  "hidden_act": "silu",
  "hidden_size": 7168,
  "initializer_range": 0.02,
  "intermediate_size": 18432,
  "max_position_embeddings": 131072,
  "model_type": "kimi_k2",
  "moe_intermediate_size": 8192,
  "moe_layer_freq": 1,
  "n_group": 8,
  "n_routed_experts": 128,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 112,
  "num_experts_per_tok": 2,
  "num_hidden_layers": 47,
  "num_key_value_heads": 28,
  "num_nextn_predict_layers": 0,
  "pretraining_tp": 1,
  "qk_layernorm": true,
  "rms_norm_eps": 1e-06,
  "rope_theta": 50000.0,
  "routed_scaling_factor": 2.827,
  "rope_scaling": {
    "beta_fast": 1.0,
    "beta_slow": 1.0,
    "factor": 32.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
  "scoring_func": "sigmoid",
  "seq_aux": true,
  "tie_word_embeddings": false,
  "topk_group": 2,
  "topk_method": "noaux_tc",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.48.3",
  "use_cache": true,
  "vocab_size": 163840
}
```

**与原 config.json 的差异对比**:

| 字段 | 原值 (100B/MLA) | 新值 (1000B/GQA) | 说明 |
|------|----------------|------------------|------|
| hidden_size | 4096 | **7168** | |
| intermediate_size | 11264 | **18432** | |
| moe_intermediate_size | 2048 | **8192** | |
| num_hidden_layers | 32 | **47** | 48 层减去 1 个 noop |
| num_attention_heads | 128 | **112** | |
| num_key_value_heads | 128 | **28** | 112 / 4 (GQA) |
| qk_layernorm | (不存在) | **true** | 新增 |
| q_lora_rank | 1536 | **(删除)** | MLA 专有 |
| kv_lora_rank | 512 | **(删除)** | MLA 专有 |
| qk_nope_head_dim | 128 | **(删除)** | MLA 专有 |
| qk_rope_head_dim | 64 | **(删除)** | MLA 专有 |
| v_head_dim | 128 | **(删除)** | MLA 专有 |

### 1.3 验证方法

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("models/", trust_remote_code=True)
assert config.hidden_size == 7168
assert config.num_attention_heads == 112
assert config.num_key_value_heads == 28
assert config.num_hidden_layers == 47
assert config.qk_layernorm == True
assert config.hidden_size // config.num_attention_heads == 64  # head_dim
print("Config validation passed")
```

---

## 二、Commit 2: HF 模型代码 MLA -> GQA

**范围**: `models/modeling_deepseek.py`

### 2.1 任务清单

- [ ] 2.1.1 改造 `DeepseekV3Attention.__init__`: 删除 MLA 投影层，替换为标准 GQA 投影层 + 可选 qk_layernorm
- [ ] 2.1.2 改造 `DeepseekV3Attention._init_rope`: RoPE dim 从 `qk_rope_head_dim` 改为 `head_dim`
- [ ] 2.1.3 改造 `DeepseekV3Attention.forward`: 使用标准 GQA 前向逻辑
- [ ] 2.1.4 改造 `DeepseekV3FlashAttention2.forward`: 同步更新 Flash Attention 路径
- [ ] 2.1.5 修改 `softmax_scale` 计算: 使用 `head_dim^(-0.5)` 替代 `q_head_dim^(-0.5)`
- [ ] 2.1.6 修改 `_shape` 方法: 使用 `head_dim` 替代 `v_head_dim`
- [ ] 2.1.7 修改 `apply_rotary_pos_emb`: 检查 RoPE 排列方式是否需要调整（MLA 使用了特殊的 reshape 排列）

### 2.2 架构对比

**MLA (当前):**

```
hidden_states
  ├─ q_a_proj (7168 -> 1536)
  │    └─ q_a_layernorm (1536)
  │         └─ q_b_proj (1536 -> 112 * (128+64) = 21504)
  │              └─ split -> q_nope (128) + q_pe (64)
  ├─ kv_a_proj_with_mqa (7168 -> 512+64 = 576)
  │    ├─ compressed_kv (512) -> kv_a_layernorm -> kv_b_proj (512 -> 112*(128+128) = 28672)
  │    │    └─ split -> k_nope (128) + v (128)
  │    └─ k_pe (64)
  ├─ RoPE(q_pe, k_pe)  dim=64
  ├─ concat -> query_states (128+64=192), key_states (128+64=192)
  ├─ attention (scale = 192^(-0.5) * mscale^2)
  └─ o_proj (112*128 -> 7168)
```

**GQA (目标):**

```
hidden_states
  ├─ q_proj (7168 -> 112 * 64 = 7168)
  │    └─ reshape (B, S, 112, 64) -> q_layernorm (可选)
  ├─ k_proj (7168 -> 28 * 64 = 1792)
  │    └─ reshape (B, S, 28, 64) -> k_layernorm (可选)
  ├─ v_proj (7168 -> 28 * 64 = 1792)
  │    └─ reshape (B, S, 28, 64)
  ├─ RoPE(q, k)  dim=64 (全 head_dim)
  ├─ repeat_kv (K, V: 28 -> 112)
  ├─ attention (scale = 64^(-0.5) * mscale^2)
  └─ o_proj (112*64 -> 7168)
```

### 2.3 详细修改

#### 2.3.1 `DeepseekV3Attention.__init__` 替换

**删除** (原第 641-674 行全部 MLA 投影层):

```python
# 删除以下所有内容
self.q_lora_rank = config.q_lora_rank
self.qk_rope_head_dim = config.qk_rope_head_dim
self.kv_lora_rank = config.kv_lora_rank
self.v_head_dim = config.v_head_dim
self.qk_nope_head_dim = config.qk_nope_head_dim
self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

if self.q_lora_rank is None:
    self.q_proj = nn.Linear(...)
else:
    self.q_a_proj = nn.Linear(...)
    self.q_a_layernorm = DeepseekV3RMSNorm(...)
    self.q_b_proj = nn.Linear(...)

self.kv_a_proj_with_mqa = nn.Linear(...)
self.kv_a_layernorm = DeepseekV3RMSNorm(...)
self.kv_b_proj = nn.Linear(...)

self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, ...)
```

**替换为**:

```python
self.num_key_value_heads = config.num_key_value_heads
self.num_key_value_groups = self.num_heads // self.num_key_value_heads
self.head_dim = config.hidden_size // self.num_heads  # = 64

self.is_causal = True

self.q_proj = nn.Linear(
    self.hidden_size,
    self.num_heads * self.head_dim,
    bias=config.attention_bias,
)
self.k_proj = nn.Linear(
    self.hidden_size,
    self.num_key_value_heads * self.head_dim,
    bias=config.attention_bias,
)
self.v_proj = nn.Linear(
    self.hidden_size,
    self.num_key_value_heads * self.head_dim,
    bias=config.attention_bias,
)
self.o_proj = nn.Linear(
    self.num_heads * self.head_dim,
    self.hidden_size,
    bias=config.attention_bias,
)

self.qk_layernorm = getattr(config, 'qk_layernorm', False)
if self.qk_layernorm:
    self.q_layernorm = DeepseekV3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.k_layernorm = DeepseekV3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
```

#### 2.3.2 `_init_rope` 维度修改

将所有 `self.qk_rope_head_dim` 替换为 `self.head_dim`：

```python
def _init_rope(self):
    if self.config.rope_scaling is None:
        self.rotary_emb = DeepseekV3RotaryEmbedding(
            self.head_dim,  # 原: self.qk_rope_head_dim
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    else:
        scaling_type = self.config.rope_scaling['type']
        scaling_factor = self.config.rope_scaling['factor']
        if scaling_type == 'yarn':
            kwargs = { ... }
            self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
                self.head_dim,  # 原: self.qk_rope_head_dim
                ...
            )
        # 其余 scaling_type 类似修改
```

#### 2.3.3 `softmax_scale` 修改

```python
# 原:
self.softmax_scale = self.q_head_dim**(-0.5)  # (128+64)^(-0.5) = 0.0722

# 改为:
self.softmax_scale = self.head_dim**(-0.5)  # 64^(-0.5) = 0.125
if self.config.rope_scaling is not None:
    mscale_all_dim = self.config.rope_scaling.get('mscale_all_dim', 0)
    scaling_factor = self.config.rope_scaling['factor']
    if mscale_all_dim:
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        self.softmax_scale = self.softmax_scale * mscale * mscale
```

#### 2.3.4 `_shape` 方法修改

```python
def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    return (tensor.view(bsz, seq_len, self.num_heads,
                        self.head_dim).transpose(1, 2).contiguous())
    # 原: self.v_head_dim -> 改为 self.head_dim
```

#### 2.3.5 `apply_rotary_pos_emb` 检查

当前实现对 q, k 做了特殊的 reshape 排列（第 364-368 行）：

```python
b, h, s, d = q.shape
q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
```

这种排列方式是为了匹配 MCore 的 RoPE 排列。需要确认 GQA 模式下 MCore 是否也使用相同排列。
如果 MCore GQA 使用标准 Llama-style RoPE（前半后半），则此 reshape 可以去掉：

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

> **需验证**: dump 一个小模型的推理结果对比，确认 RoPE 排列是否正确。

#### 2.3.6 `DeepseekV3Attention.forward` 重写

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if self.qk_layernorm:
        query_states = self.q_layernorm(query_states)
        key_states = self.k_layernorm(key_states)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value
```

#### 2.3.7 `DeepseekV3FlashAttention2.forward` 重写

与 eager forward 类似，但使用 flash_attn API。关键差异：

```python
def forward(self, hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if self.qk_layernorm:
        query_states = self.q_layernorm(query_states)
        key_states = self.k_layernorm(key_states)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # flash_attn 需要 (bsz, seq_len, num_heads, head_dim)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # dtype cast (与原代码一致)
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        target_dtype = self.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len,
        dropout=self.attention_dropout if self.training else 0.0,
        softmax_scale=self.softmax_scale,
    )

    # GQA: head_dim == v_head_dim，无需 pad/unpad
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value
```

注意事项：
- `_flash_attention_forward` 和 `_upad_input` 方法中的 `num_key_value_heads` 和 `head_dim` 引用也需对齐（原代码使用 `self.num_heads` / `self.q_head_dim`）。
- GQA 下 Q/K/V 的 head_dim 一致，不再需要 MLA 中 `q_head_dim != v_head_dim` 的 pad 逻辑，相关 `F.pad` 和后续 slice 代码全部删除。

### 2.4 不需要修改的部分

以下组件与 MLA/GQA 无关，保持不变：

- `DeepseekV3RMSNorm`
- `DeepseekV3RotaryEmbedding` 及其子类 (Linear/Dynamic/Yarn)
- `rotate_half`, `yarn_*` 工具函数
- `DeepseekV3MLP`, `MoEGate`, `DeepseekV3MoE`
- `DeepseekV3DecoderLayer` (Attention 实例化通过 `ATTENTION_CLASSES` 字典)
- `DeepseekV3Model`, `DeepseekV3ForCausalLM`, `DeepseekV3ForSequenceClassification`
- `repeat_kv` (GQA 需要此函数)

### 2.5 HF 权重 key 对照表

| 用途 | MLA key (旧) | GQA key (新) |
|------|-------------|-------------|
| Q 投影 | `self_attn.q_a_proj.weight` + `self_attn.q_a_layernorm.weight` + `self_attn.q_b_proj.weight` | `self_attn.q_proj.weight` |
| K 投影 | `self_attn.kv_a_proj_with_mqa.weight` + `self_attn.kv_a_layernorm.weight` + `self_attn.kv_b_proj.weight` (共享) | `self_attn.k_proj.weight` |
| V 投影 | (与 K 共享 kv_b_proj) | `self_attn.v_proj.weight` |
| O 投影 | `self_attn.o_proj.weight` | `self_attn.o_proj.weight` (不变) |
| Q LayerNorm | `self_attn.q_a_layernorm.weight` (dim=1536, MLA 压缩维度) | `self_attn.q_layernorm.weight` (dim=64, head_dim) |
| K LayerNorm | `self_attn.kv_a_layernorm.weight` (dim=512, MLA 压缩维度) | `self_attn.k_layernorm.weight` (dim=64, head_dim) |
| RoPE inv_freq | `self_attn.rotary_emb.inv_freq` | `self_attn.rotary_emb.inv_freq` (不变, 但 dim=64) |

### 2.6 验证方法

```python
import torch
from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained("models/", trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# 检查 layer 0 attention 结构
attn = model.model.layers[0].self_attn
assert hasattr(attn, 'q_proj') and not hasattr(attn, 'q_a_proj'), "应为 GQA 结构"
assert attn.q_proj.weight.shape == (7168, 7168), f"Q shape mismatch: {attn.q_proj.weight.shape}"
assert attn.k_proj.weight.shape == (1792, 7168), f"K shape mismatch: {attn.k_proj.weight.shape}"
assert attn.v_proj.weight.shape == (1792, 7168), f"V shape mismatch: {attn.v_proj.weight.shape}"
assert attn.o_proj.weight.shape == (7168, 7168), f"O shape mismatch: {attn.o_proj.weight.shape}"
if config.qk_layernorm:
    assert hasattr(attn, 'q_layernorm'), "缺少 q_layernorm"
    assert hasattr(attn, 'k_layernorm'), "缺少 k_layernorm"
    assert attn.q_layernorm.weight.shape == (64,), f"q_layernorm shape mismatch"

# 检查 MoE 层
assert isinstance(model.model.layers[3].mlp, type(model.model.layers[3].mlp))  # MoE from layer 3

# 前向测试
x = torch.randint(0, 163840, (1, 32))
with torch.no_grad():
    out = model(x)
assert out.logits.shape == (1, 32, 163840)
print("Model structure validation passed")
```

---

## 三、Commit 3: mcore2hf 转换脚本修复

**范围**: `utils/convert_ckpt_mcore2hf.py`

### 3.1 任务清单

- [ ] 3.1.1 删除顶部硬编码常量 `HIDDEN_SIZE` / `NUM_EXPERTS` / `NUM_ATTENTION_HEADS` / `QK_HEAD_DIM` / `QK_POS_EMB_HEAD_DIM` / `V_HEAD_DIM` / `Q_LORA_RANK`
- [ ] 3.1.2 新增命令行参数：`--hidden-size`, `--num-attention-heads`, `--num-key-value-heads`
- [ ] 3.1.3 重写 `_set_layer_attn`: 从 MLA 转换改为 GQA 转换（正确 gather TP）
- [ ] 3.1.4 移除 `mla_mm_split` 相关参数和逻辑
- [ ] 3.1.5 移除 `inv_freq` 计算和输出（GQA 的 `rotary_emb.inv_freq` 由 HF 模型动态计算）
- [ ] 3.1.6 在 `run()` 末尾自动生成 `config.json` + 拷贝模型定义文件
- [ ] 3.1.7 更新参数解析和 `MgCkptConvert.__init__`

### 3.2 详细修改

#### 3.2.1 删除硬编码常量

删除第 14-21 行:

```python
# 删除
HIDDEN_SIZE = 7168
NUM_EXPERTS = 128
FIRST_K_DENSE_REPLACE = 3
NUM_ATTENTION_HEADS = 112
QK_HEAD_DIM = 128
QK_POS_EMB_HEAD_DIM = 64
V_HEAD_DIM = 128
Q_LORA_RANK = 1536
```

#### 3.2.2 更新 `MgCkptConvert.__init__`

替换 MLA 参数为 GQA 参数：

```python
class MgCkptConvert:
    def __init__(
        self,
        mg_load_dir: str,
        hf_save_dir: str,
        num_layers: int,
        tp_size: int,
        pp_size: int,
        ep_size: int,
        first_k_dense_replace: int,
        hidden_size: int,
        num_experts: int,
        num_attention_heads: int,
        num_key_value_heads: int,       # 新增，替代 qk_head_dim / v_head_dim 等
        moe_grouped_gemm: bool,
        moe_tp_extend_ep: bool,
        schedules_method: str | None,
        vpp_stage: int | None,
        num_layer_list: str | None,
        noop_layers: str | None,
        rotary_base: float,
    ):
        # ... 基础赋值 ...
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        # 删除: self.qk_head_dim, self.v_head_dim, self.qk_pos_emb_head_dim
        # 删除: self.mla_mm_split
```

#### 3.2.3 重写 `_set_layer_attn`（核心修改）

```python
def _set_layer_attn(self, hf: dict[str, torch.Tensor],
                    models: dict[tuple[int, int], dict[str, torch.Tensor]],
                    hf_layer: int, local_idx: int) -> None:
    prefix = f'decoder.layers.{local_idx}.self_attention'
    qkv_key = f'{prefix}.linear_qkv.weight'
    proj_key = f'{prefix}.linear_proj.weight'
    q_norm_key = f'{prefix}.q_layernorm.weight'
    k_norm_key = f'{prefix}.k_layernorm.weight'

    # 1. Gather linear_proj (RowParallel: 沿 dim=1 拼接)
    o_proj_parts = [models[(tp_rank, 0)].pop(proj_key) for tp_rank in range(self.tp_size)]
    o_proj = torch.cat(o_proj_parts, dim=1)

    # 2. Gather linear_qkv (ColumnParallel: 每个 TP rank 含该 rank 的 Q/K/V)
    q_parts, k_parts, v_parts = [], [], []
    q_per_tp = (self.num_attention_heads // self.tp_size) * self.head_dim
    kv_per_tp = (self.num_key_value_heads // self.tp_size) * self.head_dim

    for tp_rank in range(self.tp_size):
        qkv_shard = models[(tp_rank, 0)].pop(qkv_key)

        # --- 选择拆分方式 (见 0.6 节) ---
        # 方式 B: Sequential 布局 [Q_rank | K_rank | V_rank]
        q_r, k_r, v_r = torch.split(qkv_shard, [q_per_tp, kv_per_tp, kv_per_tp], dim=0)

        # 方式 A: Interleaved 布局 (如果方式 B 不对，改用此方式)
        # group_factor = self.num_key_value_groups
        # kv_heads_per_tp = self.num_key_value_heads // self.tp_size
        # qkv_grouped = qkv_shard.reshape(
        #     kv_heads_per_tp, group_factor + 2, self.head_dim, self.hidden_size)
        # q_r = qkv_grouped[:, :group_factor, :, :].reshape(-1, self.hidden_size)
        # k_r = qkv_grouped[:, group_factor, :, :].reshape(-1, self.hidden_size)
        # v_r = qkv_grouped[:, group_factor + 1, :, :].reshape(-1, self.hidden_size)

        q_parts.append(q_r)
        k_parts.append(k_r)
        v_parts.append(v_r)

    full_q = torch.cat(q_parts, dim=0)  # (num_attention_heads * head_dim, hidden_size)
    full_k = torch.cat(k_parts, dim=0)  # (num_kv_heads * head_dim, hidden_size)
    full_v = torch.cat(v_parts, dim=0)  # (num_kv_heads * head_dim, hidden_size)

    # 3. q/k layernorm (不切分，从 rank 0 取)
    q_ln = models[(0, 0)].pop(q_norm_key)
    k_ln = models[(0, 0)].pop(k_norm_key)

    # 4. 写入 HF 字典
    hf[f'model.layers.{hf_layer}.self_attn.q_proj.weight'] = full_q.clone()
    hf[f'model.layers.{hf_layer}.self_attn.k_proj.weight'] = full_k.clone()
    hf[f'model.layers.{hf_layer}.self_attn.v_proj.weight'] = full_v.clone()
    hf[f'model.layers.{hf_layer}.self_attn.o_proj.weight'] = o_proj.clone()
    hf[f'model.layers.{hf_layer}.self_attn.q_layernorm.weight'] = q_ln.clone()
    hf[f'model.layers.{hf_layer}.self_attn.k_layernorm.weight'] = k_ln.clone()
```

> **重点**: 如果验证 checkpoint 发现 k layernorm 的 key 名是 `kv_layernorm.weight` 而非 `k_layernorm.weight`，
> 则改 `k_norm_key` 为 `f'{prefix}.kv_layernorm.weight'`。

#### 3.2.4 `run()` 末尾自动生成 config

在 `run()` 方法写入 `model.safetensors.index.json` 之后追加：

```python
    import shutil

config_data = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_deepseek.DeepseekV3Config",
        "AutoModel": "modeling_deepseek.DeepseekV3Model",
        "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM"
    },
    "hidden_size": self.hidden_size,
    "intermediate_size": 18432,      # 需要参数化或从 dense 层推导
    "moe_intermediate_size": 8192,   # 需要参数化或从 expert 层推导
    "num_hidden_layers": self.num_real_layers,
    "num_attention_heads": self.num_attention_heads,
    "num_key_value_heads": self.num_key_value_heads,
    "qk_layernorm": True,
    "vocab_size": 163840,
    # ... 其余固定参数 ...
    "model_type": "kimi_k2",
    "torch_dtype": "bfloat16",
}
config_path = os.path.join(self.hf_save_dir, 'config.json')
with open(config_path, 'w') as f:
    json.dump(config_data, f, indent=2)

# 拷贝 Python 模型定义文件
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(script_dir, 'models')
for fname in ['configuration_deepseek.py', 'modeling_deepseek.py']:
    src = os.path.join(models_dir, fname)
    if os.path.isfile(src):
        shutil.copy2(src, self.hf_save_dir)

logger.info(f'HF model artifacts written to: {self.hf_save_dir}')
```

#### 3.2.5 更新命令行参数

```python
def get_args():
    parser = argparse.ArgumentParser()
    # ... 保留已有的通用参数 ...

    # 新增
    parser.add_argument('--hidden-size', type=int, default=7168)
    parser.add_argument('--num-attention-heads', type=int, default=112)
    parser.add_argument('--num-key-value-heads', type=int, default=28)

    # 删除
    # --mla-mm-split  (MLA 专有)

    args = parser.parse_args()
    return args
```

更新 `main()`:

```python
def main() -> None:
    args = get_args()
    converter = MgCkptConvert(
        mg_load_dir=args.load_dir,
        hf_save_dir=args.save_dir,
        num_layers=args.num_layers,
        tp_size=args.source_tensor_parallel_size,
        pp_size=args.source_pipeline_parallel_size,
        ep_size=args.source_expert_parallel_size,
        first_k_dense_replace=args.first_k_dense_replace,
        hidden_size=args.hidden_size,            # 不再硬编码
        num_experts=128,                         # 可进一步参数化
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        moe_grouped_gemm=args.moe_grouped_gemm,
        moe_tp_extend_ep=args.moe_tp_extend_ep,
        schedules_method=args.schedules_method,
        vpp_stage=args.num_layers_per_virtual_pipeline_stage,
        num_layer_list=args.num_layer_list,
        noop_layers=args.noop_layers,
        rotary_base=args.rotary_base,
    )
    converter.run()
```

### 3.3 验证方法

```bash
python utils/convert_ckpt_mcore2hf.py \
  --load-dir /path/to/mcore_ckpt \
  --save-dir /tmp/hf_test \
  --num-layers 48 \
  --source-tensor-parallel-size 2 \
  --source-pipeline-parallel-size 8 \
  --source-expert-parallel-size 64 \
  --moe-grouped-gemm \
  --moe-tp-extend-ep \
  --schedules-method dualpipev \
  --noop-layers 47 \
  --hidden-size 7168 \
  --num-attention-heads 112 \
  --num-key-value-heads 28
```

验证输出:

```python
import safetensors.torch, json

index = json.load(open('/tmp/hf_test/model.safetensors.index.json'))
# 抽查 layer 0 attention 权重 shape
st = safetensors.torch.load_file('/tmp/hf_test/<shard_containing_layer0>')
assert st['model.layers.0.self_attn.q_proj.weight'].shape == (7168, 7168)
assert st['model.layers.0.self_attn.k_proj.weight'].shape == (1792, 7168)
assert st['model.layers.0.self_attn.v_proj.weight'].shape == (1792, 7168)
assert st['model.layers.0.self_attn.o_proj.weight'].shape == (7168, 7168)
assert st['model.layers.0.self_attn.q_layernorm.weight'].shape == (64,)
assert st['model.layers.0.self_attn.k_layernorm.weight'].shape == (64,)
print("mcore2hf output validation passed")
```

---

## 四、Commit 4: hf2mcore 转换脚本修复

**范围**: `utils/convert_ckpt_hf2mcore.py`

### 4.1 任务清单

- [ ] 4.1.1 删除顶部硬编码常量
- [ ] 4.1.2 新增命令行参数 `--hidden-size`, `--num-attention-heads`, `--num-key-value-heads`
- [ ] 4.1.3 重写 `_set_layer_attn`: 从 GQA HF 权重构造 MCore QKV
- [ ] 4.1.4 移除 `mla_mm_split` 相关参数和逻辑
- [ ] 4.1.5 更新 `CkptConvert.__init__`

### 4.2 详细修改

#### 4.2.1 删除硬编码常量

删除第 14-20 行:

```python
# 删除
HIDDEN_SIZE = 7168
NUM_EXPERTS = 128
FIRST_K_DENSE_REPLACE = 3
NUM_ATTENTION_HEADS = 112
QK_HEAD_DIM = 128
QK_POS_EMB_HEAD_DIM = 64
V_HEAD_DIM = 128
```

#### 4.2.2 更新 `CkptConvert.__init__`

替换 MLA 参数:

```python
class CkptConvert:
    def __init__(
        self,
        hf_model_path: str,
        mg_save_path: str,
        num_layers: int,
        tp_size: int,
        pp_size: int,
        ep_size: int,
        first_k_dense_replace: int,
        hidden_size: int,
        num_experts: int,
        num_attention_heads: int,
        num_key_value_heads: int,     # 新增
        moe_grouped_gemm: bool,
        moe_tp_extend_ep: bool,
        schedules_method: str | None,
        vpp_stage: int | None,
        num_layer_list: str | None,
        noop_layers: str | None,
        qlora_nf4: bool,
    ):
        # ...
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        # 删除: self.qk_head_dim, self.v_head_dim, self.qk_pos_emb_head_dim
        # 删除: self.mla_mm_split
```

#### 4.2.3 重写 `_set_layer_attn`（核心修改）

```python
def _set_layer_attn(
    self,
    hf_layer: int,
    local_layer_idx: int,
    weights: dict[str, torch.Tensor],
    mg_model: dict[int, dict[int, dict[str, torch.Tensor]]],
) -> None:
    # 1. 读取 HF GQA 权重
    q_weight = weights.pop(f'model.layers.{hf_layer}.self_attn.q_proj.weight')
    k_weight = weights.pop(f'model.layers.{hf_layer}.self_attn.k_proj.weight')
    v_weight = weights.pop(f'model.layers.{hf_layer}.self_attn.v_proj.weight')
    o_proj = weights.pop(f'model.layers.{hf_layer}.self_attn.o_proj.weight')

    q_ln = weights.pop(f'model.layers.{hf_layer}.self_attn.q_layernorm.weight')
    k_ln = weights.pop(f'model.layers.{hf_layer}.self_attn.k_layernorm.weight')

    # 可选: 弹出 rotary_emb.inv_freq (有些 HF 模型会保存, 转换不需要)
    inv_freq_key = f'model.layers.{hf_layer}.self_attn.rotary_emb.inv_freq'
    weights.pop(inv_freq_key, None)

    prefix = f'decoder.layers.{local_layer_idx}.self_attention'
    qkv_key = f'{prefix}.linear_qkv.weight'
    proj_key = f'{prefix}.linear_proj.weight'
    q_norm_key = f'{prefix}.q_layernorm.weight'
    k_norm_key = f'{prefix}.k_layernorm.weight'

    # 2. 按 TP 切分 Q, K, V
    q_tp = torch.chunk(q_weight, self.tp_size, dim=0)
    k_tp = torch.chunk(k_weight, self.tp_size, dim=0)
    v_tp = torch.chunk(v_weight, self.tp_size, dim=0)
    o_tp = torch.chunk(o_proj, self.tp_size, dim=1)

    # 3. 为每个 (ep_rank, tp_rank) 构造 MCore 权重
    for ep_rank in range(self.ep_size):
        for tp_rank in range(self.tp_size):
            # 拼接为 MCore 的 linear_qkv 格式
            # --- Sequential 布局: [Q_rank | K_rank | V_rank] ---
            qkv = torch.cat([q_tp[tp_rank], k_tp[tp_rank], v_tp[tp_rank]], dim=0)

            # --- Interleaved 布局 (如果需要): ---
            # q_r = q_tp[tp_rank].reshape(
            #     self.num_attention_heads // self.tp_size, self.head_dim, -1)
            # k_r = k_tp[tp_rank].reshape(
            #     self.num_key_value_heads // self.tp_size, self.head_dim, -1)
            # v_r = v_tp[tp_rank].reshape(
            #     self.num_key_value_heads // self.tp_size, self.head_dim, -1)
            # kv_per_tp = self.num_key_value_heads // self.tp_size
            # group_factor = self.num_key_value_groups
            # interleaved = []
            # for g in range(kv_per_tp):
            #     interleaved.append(q_r[g*group_factor:(g+1)*group_factor])
            #     interleaved.append(k_r[g:g+1])
            #     interleaved.append(v_r[g:g+1])
            # qkv = torch.cat(interleaved, dim=0).reshape(-1, self.hidden_size)

            mg_model[ep_rank][tp_rank][qkv_key] = qkv.clone()
            mg_model[ep_rank][tp_rank][proj_key] = o_tp[tp_rank].clone()
            mg_model[ep_rank][tp_rank][q_norm_key] = q_ln.clone()
            mg_model[ep_rank][tp_rank][k_norm_key] = k_ln.clone()

            self._maybe_quant_nf4(mg_model[ep_rank][tp_rank], proj_key,
                                  o_tp[tp_rank].clone())
```

#### 4.2.4 更新命令行参数和 `main()`

```python
def get_args():
    parser = argparse.ArgumentParser()
    # ... 保留已有通用参数 ...

    # 新增
    parser.add_argument('--hidden-size', type=int, default=7168)
    parser.add_argument('--num-attention-heads', type=int, default=112)
    parser.add_argument('--num-key-value-heads', type=int, default=28)

    # 删除
    # --mla-mm-split

    args, _ = parser.parse_known_args()
    return args

def main() -> None:
    args = get_args()
    converter = CkptConvert(
        hf_model_path=args.load_dir,
        mg_save_path=args.save_dir,
        num_layers=args.num_layers,
        tp_size=args.target_tensor_parallel_size,
        pp_size=args.target_pipeline_parallel_size,
        ep_size=args.target_expert_parallel_size,
        first_k_dense_replace=args.first_k_dense_replace,
        hidden_size=args.hidden_size,
        num_experts=128,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        moe_grouped_gemm=args.moe_grouped_gemm,
        moe_tp_extend_ep=args.moe_tp_extend_ep,
        schedules_method=args.schedules_method,
        vpp_stage=args.num_layers_per_virtual_pipeline_stage,
        num_layer_list=args.num_layer_list,
        noop_layers=args.noop_layers,
        qlora_nf4=args.qlora_nf4,
    )
    converter.run()
```

### 4.3 验证方法

Round-trip 测试（需要已有 HF 权重）:

```bash
# HF -> MCore
python utils/convert_ckpt_hf2mcore.py \
  --load-dir /path/to/hf_model \
  --save-dir /tmp/mcore_test \
  --num-layers 48 \
  --target-tensor-parallel-size 2 \
  --target-pipeline-parallel-size 8 \
  --target-expert-parallel-size 64 \
  --moe-grouped-gemm \
  --moe-tp-extend-ep \
  --schedules-method dualpipev \
  --noop-layers 47 \
  --hidden-size 7168 \
  --num-attention-heads 112 \
  --num-key-value-heads 28

# MCore -> HF (round-trip)
python utils/convert_ckpt_mcore2hf.py \
  --load-dir /tmp/mcore_test \
  --save-dir /tmp/hf_roundtrip \
  --num-layers 48 \
  --source-tensor-parallel-size 2 \
  --source-pipeline-parallel-size 8 \
  --source-expert-parallel-size 64 \
  --moe-grouped-gemm \
  --moe-tp-extend-ep \
  --schedules-method dualpipev \
  --noop-layers 47 \
  --hidden-size 7168 \
  --num-attention-heads 112 \
  --num-key-value-heads 28
```

验证 round-trip 一致性:

```python
import safetensors.torch, json

idx_orig = json.load(open('/path/to/hf_model/model.safetensors.index.json'))
idx_rt = json.load(open('/tmp/hf_roundtrip/model.safetensors.index.json'))

for key in idx_orig['weight_map']:
    shard_orig = safetensors.torch.load_file(f"/path/to/hf_model/{idx_orig['weight_map'][key]}")
    shard_rt = safetensors.torch.load_file(f"/tmp/hf_roundtrip/{idx_rt['weight_map'][key]}")
    if key in shard_orig and key in shard_rt:
        assert torch.allclose(shard_orig[key], shard_rt[key], atol=1e-6), f"Mismatch: {key}"
print("Round-trip validation passed")
```

---

## 五、Commit 5: 验证与收尾

### 5.1 任务清单

- [ ] 5.1.1 将 `rebuild.md` 加入 `.gitignore`
- [ ] 5.1.2 端到端推理验证 (从 MCore ckpt -> HF -> `model.generate()`)
- [ ] 5.1.3 清理未使用的导入和变量

### 5.2 `.gitignore` 追加

在文件末尾追加:

```
# 开发文档 (不入版本控制)
rebuild.md
```

### 5.3 端到端验证流程

```bash
# 1. 从训练产出的 MCore checkpoint 转换为 HF
python utils/convert_ckpt_mcore2hf.py \
  --load-dir $CKPT_SAVE_DIR \
  --save-dir ./hf_1000b \
  --num-layers 48 \
  --source-tensor-parallel-size 2 \
  --source-pipeline-parallel-size 8 \
  --source-expert-parallel-size 64 \
  --moe-grouped-gemm \
  --moe-tp-extend-ep \
  --schedules-method dualpipev \
  --noop-layers 47

# 2. 加载并推理
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    './hf_1000b',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained('$TOKENIZER_PATH')

prompt = 'The capital of France is'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"
```

### 5.4 完成后清理

- [ ] 确认 `models/modeling_deepseek_bak.py` 已删除
- [ ] 确认 `models/configuration_model.py` 已重命名为 `models/configuration_deepseek.py`
- [ ] 确认 `models/model.safetensors.index.json` 保持为空占位符（由转换脚本填充）
- [ ] 更新 `README.md` 中的权重转换命令示例（移除 `--mla-mm-split`，增加 `--num-key-value-heads`）

---

## 六、风险项与待验证事项

| # | 事项 | 状态 | 备注 |
|---|------|------|------|
| R1 | MCore GQA 的 `linear_qkv` 是 interleaved 还是 sequential 布局 | [ ] 待验证 | dump 一个实际 ckpt 的 `linear_qkv.weight.shape` 和首 10 行值 |
| R2 | GQA 模式下 K layernorm 的 MCore key 是 `k_layernorm` 还是 `kv_layernorm` | [ ] 待验证 | `print(state_dict.keys())` 确认 |
| R3 | `apply_rotary_pos_emb` 中的特殊 reshape 排列是否适用于 GQA | [ ] 待验证 | 参考 MindSpeed-LLM 的 RoPE 实现确认 |
| R4 | mcore2hf 自动生成的 config.json 中 `intermediate_size` / `moe_intermediate_size` 是否需要参数化 | [ ] 决策 | 可从 dense/expert MLP 权重 shape 反推 |
| R5 | 1000B training 的 `generation_config.json` 是否需要更新 | [ ] 检查 | 当前 `max_length=131072`, `eos_token_id=163585` 可能不需要改 |

### 快速验证 R1/R2 的脚本

```python
import torch

ckpt_path = "/path/to/mcore_ckpt/iter_XXXXXXX/mp_rank_00_000_000/model_optim_rng.pt"
state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model = state.get('model0', state.get('model'))

for k, v in sorted(model.items()):
    if 'self_attention' in k and 'layers.0.' in k:
        print(f"  {k}: {v.shape}")
```

预期输出 (GQA, TP_rank=0):

```
  decoder.layers.0.self_attention.linear_qkv.weight: torch.Size([5376, 7168])
  decoder.layers.0.self_attention.linear_proj.weight: torch.Size([7168, 3584])
  decoder.layers.0.self_attention.q_layernorm.weight: torch.Size([64])
  decoder.layers.0.self_attention.k_layernorm.weight: torch.Size([64])  # 或 kv_layernorm
```

---

## 七、进度总览

| Commit | 内容 | 状态 |
|--------|------|------|
| 1 | 配置文件与文件命名修正 | [ ] 未开始 |
| 2 | HF 模型代码 MLA -> GQA | [ ] 未开始 |
| 3 | mcore2hf 转换脚本修复 | [ ] 未开始 |
| 4 | hf2mcore 转换脚本修复 | [ ] 未开始 |
| 5 | 验证与收尾 | [ ] 未开始 |
