import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel

from .configuration_model import CustomConfig


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(orig_dtype))


class RotaryEmbedding(nn.Module):

    def __init__(self,
                 dim: int,
                 max_position_embeddings: int = 2048,
                 base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._seq_len_cached = 0
        self.register_buffer('cos_cached', torch.empty(0), persistent=False)
        self.register_buffer('sin_cached', torch.empty(0), persistent=False)

    def _update_cache(self, x: torch.Tensor, seq_len: int):
        if seq_len <= self._seq_len_cached:
            return
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(x.dtype)
        self.sin_cached = emb.sin().to(x.dtype)
        self._seq_len_cached = seq_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.shape
        self._update_cache(x, seqlen)
        cos = self.cos_cached[:seqlen, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seqlen, :].unsqueeze(0).unsqueeze(0)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def apply_rms_norm(x: torch.Tensor,
                   weight: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return x * weight


class SelfAttention(nn.Module):

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f'num_attention_heads({self.num_heads}) must be divisible by num_key_value_heads({self.num_kv_heads})'
            )
        self.head_dim_qk = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.head_dim_k = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.head_dim_v = config.v_head_dim
        self.group_factor = self.num_heads // self.num_kv_heads
        self.qk_nope = config.qk_nope_head_dim
        self.qk_rope = config.qk_rope_head_dim
        self.q_a_layernorm = nn.Parameter(torch.ones(config.q_lora_rank))
        self.kv_a_layernorm = nn.Parameter(torch.ones(config.kv_lora_rank))
        self.q_a_proj = nn.Linear(config.hidden_size,
                                  config.q_lora_rank,
                                  bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size,
                                            config.kv_lora_rank,
                                            bias=False)
        self.q_b_proj = nn.Linear(config.q_lora_rank,
                                  self.num_heads *
                                  (self.qk_nope + self.qk_rope),
                                  bias=False)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_kv_heads * (config.qk_nope_head_dim +
                                 config.qk_rope_head_dim + config.v_head_dim),
            bias=False)
        self.o_proj = nn.Linear(self.num_heads * config.v_head_dim,
                                config.hidden_size,
                                bias=False)
        self.rope = RotaryEmbedding(
            config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = apply_rms_norm(self.q_a_proj(x), self.q_a_layernorm)
        kv = apply_rms_norm(self.kv_a_proj_with_mqa(x), self.kv_a_layernorm)
        q = self.q_b_proj(q).view(bsz, seqlen, self.num_heads,
                                  self.qk_nope + self.qk_rope)
        kv = self.kv_b_proj(kv).view(
            bsz, seqlen, self.num_kv_heads,
            self.qk_nope + self.qk_rope + self.head_dim_v)
        k_nope, k_rope, v = torch.split(
            kv, [self.qk_nope, self.qk_rope, self.head_dim_v], dim=-1)
        q_nope, q_rope = torch.split(q, [self.qk_nope, self.qk_rope], dim=-1)
        cos, sin = self.rope(q_rope.reshape(bsz, seqlen, -1))
        q_rope, k_rope = apply_rope(q_rope, k_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([k_nope, k_rope], dim=-1)
        if self.group_factor > 1:
            k_full = k_full.repeat_interleave(self.group_factor, dim=2)
            v = v.repeat_interleave(self.group_factor, dim=2)
        q = q.permute(0, 2, 1, 3)
        k_full = k_full.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k_full.transpose(
            -1, -2)) / math.sqrt(self.qk_nope + self.qk_rope)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(
            bsz, seqlen, self.num_heads * self.head_dim_v)
        out = self.o_proj(ctx)
        return out


class SwiGLU(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Expert(nn.Module):

    def __init__(self, hidden_size: int, moe_intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size,
                                   moe_intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(hidden_size,
                                 moe_intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(moe_intermediate_size,
                                   hidden_size,
                                   bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MLP(nn.Module):

    def __init__(self, config: CustomConfig, dense: bool):
        super().__init__()
        self.dense = dense
        if dense:
            self.ffn = SwiGLU(config.hidden_size, config.intermediate_size)
            self.router = None
            self.shared_expert = None
            self.experts = None
            self.gate_e_score_correction_bias = None
        else:
            self.router = nn.Linear(config.hidden_size,
                                    config.n_routed_experts,
                                    bias=False)
            self.gate_e_score_correction_bias = nn.Parameter(
                torch.zeros(config.n_routed_experts))
            self.n_shared_experts = int(config.n_shared_experts or 0)
            self.shared_experts = nn.ModuleList([
                Expert(config.hidden_size, config.moe_intermediate_size)
                for _ in range(self.n_shared_experts)
            ])
            self.experts = nn.ModuleList([
                Expert(config.hidden_size, config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ])
            self.topk = min(
                int(config.num_experts_per_tok if config.
                    num_experts_per_tok is not None else 1),
                config.n_routed_experts)
            self.norm_topk_prob = bool(config.norm_topk_prob)
            self.scoring_func = config.scoring_func
            self.routed_scaling_factor = float(config.routed_scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dense:
            return self.ffn(x)
        logits = self.router(x) + self.gate_e_score_correction_bias
        if self.scoring_func == 'softmax':
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = torch.sigmoid(logits)
        topk_val, topk_idx = torch.topk(probs, k=self.topk, dim=-1)
        if self.norm_topk_prob:
            topk_weight = topk_val / (topk_val.sum(dim=-1, keepdim=True) +
                                      1e-9)
        else:
            topk_weight = topk_val
        topk_weight = topk_weight * self.routed_scaling_factor
        out = torch.zeros_like(x)
        b, s, h = x.shape
        x_flat = x.view(b * s, h)
        for k in range(self.topk):
            idx = topk_idx[..., k].view(b * s)
            expert_out = torch.zeros_like(x_flat)
            for e in range(len(self.experts)):
                mask = (idx == e).nonzero(as_tuple=False).squeeze(-1)
                if mask.numel() == 0:
                    continue
                xo = x_flat.index_select(0, mask)
                yo = self.experts[e](xo)
                expert_out.index_copy_(0, mask, yo)
            out = out + expert_out.view(b, s, h) * topk_weight[..., k:k + 1]
        if len(self.shared_experts) > 0:
            shared_sum = torch.zeros_like(x)
            for shared_expert in self.shared_experts:
                shared_sum = shared_sum + shared_expert(x)
            out = out + shared_sum
        return out


class DecoderLayer(nn.Module):

    def __init__(self, config: CustomConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.self_attn = SelfAttention(config)
        first_k_dense_replace = int(config.first_k_dense_replace)
        moe_layer_freq = max(1, int(config.moe_layer_freq))
        use_moe = layer_idx >= first_k_dense_replace and (
            (layer_idx - first_k_dense_replace) % moe_layer_freq == 0)
        dense = not use_moe
        self.mlp = MLP(config, dense=dense)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states,
                                       attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CustomModel(PreTrainedModel):
    config_class = CustomConfig

    def __init__(self, config: CustomConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = None
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                output_hidden_states: Optional[bool] = False,
                output_attentions: Optional[bool] = False,
                return_dict: Optional[bool] = True):
        bsz, seqlen = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        min_value = torch.finfo(hidden_states.dtype).min
        causal_mask = torch.full((seqlen, seqlen),
                                 min_value,
                                 dtype=hidden_states.dtype,
                                 device=hidden_states.device)
        causal_mask = torch.triu(causal_mask,
                                 diagonal=1).unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            padding_mask = (1.0 - attention_mask[:, None, None, :].to(
                hidden_states.dtype)) * min_value
            extended_mask = causal_mask + padding_mask
        else:
            extended_mask = causal_mask
        all_hidden_states = [] if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, attention_mask=extended_mask)
        hidden_states = self.norm(hidden_states)
        if not return_dict:
            return (hidden_states, all_hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       hidden_states=all_hidden_states,
                                       past_key_values=None)


class CustomForCausalLM(CustomModel):

    def __init__(self, config: CustomConfig):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.lm_head,
                                       self.get_input_embeddings())

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_hidden_states: Optional[bool] = False,
                output_attentions: Optional[bool] = False,
                return_dict: Optional[bool] = True):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  output_hidden_states=output_hidden_states,
                                  output_attentions=output_attentions,
                                  return_dict=True)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1,
                                                     shift_logits.size(-1)),
                                   shift_labels.view(-1),
                                   ignore_index=-100)
        if not return_dict:
            return (logits, loss)
        return CausalLMOutputWithPast(loss=loss,
                                      logits=logits,
                                      past_key_values=None,
                                      hidden_states=outputs.hidden_states,
                                      attentions=None)
