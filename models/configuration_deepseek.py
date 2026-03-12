from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class DeepseekV3Config(PretrainedConfig):
    model_type = 'kimi_k2'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        vocab_size=163840,
        hidden_size=4096,
        intermediate_size=11264,
        moe_intermediate_size=2048,
        num_hidden_layers=32,
        num_nextn_predict_layers=0,
        num_attention_heads=128,
        num_key_value_heads=128,
        num_query_groups=None,
        group_query_attention=False,
        n_shared_experts=1,
        n_routed_experts=128,
        ep_size=1,
        routed_scaling_factor=2.827,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        qk_layernorm=True,
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
        moe_router_enable_expert_bias=False,
        moe_router_dtype='fp32',
        hidden_act='silu',
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        mla_fa_without_pad=False,
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
        self.num_query_groups = num_query_groups
        self.group_query_attention = group_query_attention
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_layernorm = qk_layernorm
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
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.moe_router_dtype = moe_router_dtype
        if group_query_attention and num_query_groups:
            num_key_value_heads = num_attention_heads // num_query_groups
        elif num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mla_fa_without_pad = mla_fa_without_pad
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
