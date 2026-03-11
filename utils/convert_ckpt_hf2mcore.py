import argparse
import json
import logging as logger
import os
from collections import defaultdict
from typing import Optional

import safetensors.torch
import torch

logger.basicConfig(format='')
logger.getLogger().setLevel(logger.INFO)

HIDDEN_SIZE = 7168
NUM_EXPERTS = 128
FIRST_K_DENSE_REPLACE = 3
NUM_ATTENTION_HEADS = 112
QK_HEAD_DIM = 128
QK_POS_EMB_HEAD_DIM = 64
V_HEAD_DIM = 128


def _parse_int_list(value: Optional[str]) -> Optional[list[int]]:
    if value is None or value == '':
        return None
    return list(map(int, value.split(',')))


def _ensure_iter_path(save_dir: str) -> str:
    iter_dir = os.path.join(save_dir, 'iter_0000001')
    os.makedirs(iter_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'latest_checkpointed_iteration.txt'),
              'w') as f:
        f.write('1')
    return iter_dir


def _mp_prefix(tp_rank: int, pp_rank: int, ep_rank: int, tp: int, pp: int,
               ep: int) -> str:
    if ep == 1 and pp == 1:
        return f'mp_rank_{tp_rank:02}'
    if ep == 1:
        return f'mp_rank_{tp_rank:02}_{pp_rank:03}'
    if pp == 1:
        return f'mp_rank_{tp_rank:02}_{ep_rank:03}'
    return f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}'


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
        qk_head_dim: int,
        v_head_dim: int,
        qk_pos_emb_head_dim: int,
        moe_grouped_gemm: bool,
        moe_tp_extend_ep: bool,
        mla_mm_split: bool,
        schedules_method: str | None,
        vpp_stage: int | None,
        num_layer_list: str | None,
        noop_layers: str | None,
        qlora_nf4: bool,
    ):
        self.hf_model_path = hf_model_path
        self.mg_save_path = mg_save_path
        self.num_layers = num_layers
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.first_k_dense_replace = first_k_dense_replace
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_attention_heads = num_attention_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.qk_pos_emb_head_dim = qk_pos_emb_head_dim
        self.moe_grouped_gemm = moe_grouped_gemm
        self.moe_tp_extend_ep = moe_tp_extend_ep
        self.mla_mm_split = mla_mm_split
        self.schedules_method = schedules_method
        self.dualpipe = schedules_method == 'dualpipev'
        self.vpp_stage = vpp_stage
        self.num_layer_list = num_layer_list
        self.noop_layers = noop_layers
        self.qlora_nf4 = qlora_nf4

        self.noop_layers_list = sorted(_parse_int_list(noop_layers) or [])
        if self.dualpipe:
            if vpp_stage is not None:
                raise ValueError('dualpipev 与 vpp-stage 不兼容')
            self.vpp_size = 2
            layers_each_pp = self.num_layers // self.pp_size
            if layers_each_pp % 2 != 0:
                raise ValueError('dualpipev 需要每个 PP 的层数为偶数')
            self.vpp_stage = layers_each_pp // 2
        elif vpp_stage is not None:
            self.vpp_size = (self.num_layers // self.pp_size) // vpp_stage
        else:
            self.vpp_size = None

        self._validate()
        self.iter_path = _ensure_iter_path(self.mg_save_path)

        if self.vpp_stage is None:
            self.pprank_layer_idxs: dict[int, list[int]] = defaultdict(list)
            self._build_pprank_layer_map()
        else:
            self.vpprank_layer_idxs: dict[int,
                                          dict[int,
                                               list[int]]] = defaultdict(dict)
            self._build_vpprank_layer_map()

    def _validate(self) -> None:
        if not os.path.isdir(self.hf_model_path):
            raise FileNotFoundError(self.hf_model_path)
        if self.num_layers <= 0:
            raise ValueError('num_layers 必须 > 0')
        if self.pp_size <= 0 or self.tp_size <= 0 or self.ep_size <= 0:
            raise ValueError('并行度必须 > 0')
        if self.num_layers % self.pp_size != 0 and self.num_layer_list is None:
            raise ValueError('num_layers 必须能整除 pp_size，或显式给定 num-layer-list')
        if self.first_k_dense_replace < 0 or self.first_k_dense_replace > self.num_layers:
            raise ValueError('first-k-dense-replace 非法')
        if self.num_experts % self.ep_size != 0:
            raise ValueError('num_experts 必须能整除 ep_size')
        if self.dualpipe and self.tp_size > 1 and not self.moe_tp_extend_ep:
            raise ValueError('dualpipev 下 tp>1 需要开启 moe-tp-extend-ep')
        if self.num_layer_list is not None and self.vpp_stage is not None:
            raise ValueError('num-layer-list 与 vpp/dualpipev 不可同时配置')

    def _read_weight_map(self) -> dict[str, str]:
        index_path = os.path.join(self.hf_model_path,
                                  'model.safetensors.index.json')
        with open(index_path) as f:
            return json.load(f)['weight_map']

    def _get_layer_files_map(self) -> dict[object, set[str]]:
        weight_map = self._read_weight_map()
        layer_files_map: dict[object, set[str]] = defaultdict(set)
        for k, v in weight_map.items():
            if k.startswith('model.layers.'):
                layer_id = int(k.split('model.layers.')[1].split('.')[0])
                layer_files_map[layer_id].add(v)
            else:
                layer_files_map[k].add(v)
        return layer_files_map

    def _build_pprank_layer_map(self) -> None:
        layers_each_pp = [self.num_layers // self.pp_size] * self.pp_size
        if self.num_layer_list is not None:
            layer_list = list(map(int, self.num_layer_list.split(',')))
            if len(layer_list) != self.pp_size or sum(
                    layer_list) != self.num_layers:
                raise ValueError('num-layer-list 非法')
            layers_each_pp = layer_list

        if self.noop_layers_list and self.num_layer_list is None:
            base = self.num_layers // self.pp_size
            for layer in self.noop_layers_list:
                pp_rank = layer // base
                layers_each_pp[pp_rank] -= 1

        num_noop = len(self.noop_layers_list)
        num_real_layers = self.num_layers - num_noop
        real_layers = list(range(num_real_layers))
        for pp_rank in range(self.pp_size):
            self.pprank_layer_idxs[pp_rank] = [
                real_layers.pop(0) for _ in range(layers_each_pp[pp_rank])
            ]

    def _build_vpprank_layer_map(self) -> None:
        if not self.dualpipe:
            layers_each_vpp = [[self.vpp_stage] * self.vpp_size
                               for _ in range(self.pp_size)]
            if self.noop_layers_list:
                for layer in self.noop_layers_list:
                    vpp_idx = layer // self.vpp_stage // self.pp_size
                    pp_idx = (
                        layer %
                        (self.pp_size * self.vpp_stage)) // self.vpp_stage
                    layers_each_vpp[pp_idx][vpp_idx] -= 1

            num_noop = len(self.noop_layers_list)
            num_real_layers = self.num_layers - num_noop
            real_layers = list(range(num_real_layers))
            for vpp_rank in range(self.vpp_size):
                for pp_rank in range(self.pp_size):
                    self.vpprank_layer_idxs[pp_rank][vpp_rank] = [
                        real_layers.pop(0)
                        for _ in range(layers_each_vpp[pp_rank][vpp_rank])
                    ]
            return

        noop_list = self.noop_layers_list
        min_noop = noop_list[0] if noop_list else None

        layers_each_pp = self.num_layers // self.pp_size
        layer_pop_num = layers_each_pp // 2
        all_layers = list(range(self.num_layers))
        dualpipe_layers: list[int] = []
        while all_layers:
            dualpipe_layers.extend(all_layers[:layer_pop_num])
            dualpipe_layers.extend(all_layers[-layer_pop_num:])
            all_layers = all_layers[layer_pop_num:-layer_pop_num]

        pp_rank = 0
        vpp_rank = 0
        each_pp_layer = self.num_layers // self.pp_size
        for idx, layer in enumerate(dualpipe_layers):
            if vpp_rank not in self.vpprank_layer_idxs[pp_rank]:
                self.vpprank_layer_idxs[pp_rank][vpp_rank] = []

            if not noop_list:
                self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer)
            else:
                if layer in noop_list:
                    if (idx + 1) % self.vpp_stage == 0:
                        vpp_rank += 1
                    if (idx + 1) % each_pp_layer == 0:
                        pp_rank += 1
                        vpp_rank = 0
                    continue
                if layer < min_noop:
                    self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer)
                else:
                    before = 0
                    for n in noop_list:
                        if n < layer:
                            before += 1
                        else:
                            break
                    self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer -
                                                                      before)

            if (idx + 1) % self.vpp_stage == 0:
                vpp_rank += 1
            if (idx + 1) % each_pp_layer == 0:
                pp_rank += 1
                vpp_rank = 0

    def _load_safetensors(self, filename: str) -> dict[str, torch.Tensor]:
        path = os.path.join(self.hf_model_path, filename)
        return safetensors.torch.load_file(path)

    def _load_matched_hf_weights(
            self, pp_rank: int,
            vpp_rank: int | None) -> dict[str, torch.Tensor]:
        layer_files_map = self._get_layer_files_map()
        if vpp_rank is None:
            layer_list = self.pprank_layer_idxs[pp_rank]
        else:
            layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]

        st_files: list[str] = []
        for layer in layer_list:
            st_files.extend(list(layer_files_map[layer]))

        if pp_rank == 0:
            st_files.extend(list(layer_files_map['model.embed_tokens.weight']))
            if self.dualpipe:
                st_files.extend(list(layer_files_map['lm_head.weight']))
                st_files.extend(list(layer_files_map['model.norm.weight']))

        if pp_rank == self.pp_size - 1 and not self.dualpipe:
            st_files.extend(list(layer_files_map['model.norm.weight']))
            st_files.extend(list(layer_files_map['lm_head.weight']))

        st_files = sorted(set(st_files))
        all_weights: dict[str, torch.Tensor] = {}
        for fn in st_files:
            all_weights.update(self._load_safetensors(fn))
        return all_weights

    def _maybe_quant_nf4(self, state: dict[str, torch.Tensor], key: str,
                         weight: torch.Tensor) -> None:
        if not self.qlora_nf4:
            return
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception as e:
            raise RuntimeError('启用 --qlora-nf4 需要 bitsandbytes') from e
        quantweight = bnb.nn.Params4bit(weight,
                                        requires_grad=False,
                                        quant_type='nf4').to('cpu')
        state[key] = quantweight.data
        for k, v in quantweight.quant_state.as_dict(packed=True).items():
            state[f'{key}.{k}'] = v.detach().cpu()

    def _set_preprocess(
            self, weights: dict[str, torch.Tensor],
            mg_model: dict[int, dict[int, dict[str, torch.Tensor]]]) -> None:
        emb = weights.pop('model.embed_tokens.weight')
        emb_tp = torch.chunk(emb, self.tp_size, dim=0)
        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][
                    'embedding.word_embeddings.weight'] = emb_tp[
                        tp_rank].clone()

    def _set_postprocess(
            self, weights: dict[str, torch.Tensor],
            mg_model: dict[int, dict[int, dict[str, torch.Tensor]]]) -> None:
        final_norm = weights.pop('model.norm.weight')
        lm_head = weights.pop('lm_head.weight')
        lm_head_tp = torch.chunk(lm_head, self.tp_size, dim=0)
        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][
                    'decoder.final_layernorm.weight'] = final_norm.clone()
                mg_model[ep_rank][tp_rank]['output_layer.weight'] = lm_head_tp[
                    tp_rank].clone()
                self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                      'output_layer.weight',
                                      lm_head_tp[tp_rank].clone())

    def _set_layer_norm(
        self,
        hf_layer: int,
        local_layer_idx: int,
        weights: dict[str, torch.Tensor],
        mg_model: dict[int, dict[int, dict[str, torch.Tensor]]],
    ) -> None:
        in_norm = weights.pop(
            f'model.layers.{hf_layer}.input_layernorm.weight')
        post_norm = weights.pop(
            f'model.layers.{hf_layer}.post_attention_layernorm.weight')
        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][
                    f'decoder.layers.{local_layer_idx}.input_layernorm.weight'] = in_norm.clone(
                    )
                mg_model[ep_rank][tp_rank][
                    f'decoder.layers.{local_layer_idx}.pre_mlp_layernorm.weight'] = post_norm.clone(
                    )

    def _set_layer_attn(
        self,
        hf_layer: int,
        local_layer_idx: int,
        weights: dict[str, torch.Tensor],
        mg_model: dict[int, dict[int, dict[str, torch.Tensor]]],
    ) -> None:
        q_a = weights.pop(f'model.layers.{hf_layer}.self_attn.q_a_proj.weight')
        kv_a = weights.pop(
            f'model.layers.{hf_layer}.self_attn.kv_a_proj_with_mqa.weight')
        qkv_weight = torch.cat(
            [
                q_a.reshape((-1, self.hidden_size)),
                kv_a.reshape((-1, self.hidden_size))
            ],
            dim=0,
        )
        o_proj = weights.pop(
            f'model.layers.{hf_layer}.self_attn.o_proj.weight')
        q_ln = weights.pop(
            f'model.layers.{hf_layer}.self_attn.q_a_layernorm.weight')
        kv_ln = weights.pop(
            f'model.layers.{hf_layer}.self_attn.kv_a_layernorm.weight')
        q_b_proj = weights.pop(
            f'model.layers.{hf_layer}.self_attn.q_b_proj.weight')
        kv_b_proj = weights.pop(
            f'model.layers.{hf_layer}.self_attn.kv_b_proj.weight')

        prefix = f'decoder.layers.{local_layer_idx}.self_attention'
        qkv_key = f'{prefix}.linear_qkv.weight'
        proj_key = f'{prefix}.linear_proj.weight'
        q_norm_key = f'{prefix}.q_layernorm.weight'
        kv_norm_key = f'{prefix}.kv_layernorm.weight'

        if self.mla_mm_split:
            qk_nope_key = f'{prefix}.linear_qk_nope.weight'
            qk_rope_key = f'{prefix}.linear_qk_rope.weight'
            kv_nope_key = f'{prefix}.linear_kv_nope.weight'
            linear_v_key = f'{prefix}.linear_v.weight'

            q_b_proj = q_b_proj.reshape(
                self.num_attention_heads,
                (self.qk_head_dim + self.qk_pos_emb_head_dim), -1)
            kv_b_proj = kv_b_proj.reshape(self.num_attention_heads,
                                          (self.qk_head_dim + self.v_head_dim),
                                          -1)

            qk_nope = q_b_proj[:, :self.qk_head_dim].reshape(
                -1, self.hidden_size)
            qk_rope = q_b_proj[:, self.qk_head_dim:].reshape(
                -1, self.hidden_size)
            kv_nope = kv_b_proj[:, :self.qk_head_dim].reshape(
                -1, self.hidden_size)
            linear_v = kv_b_proj[:, self.qk_head_dim:].reshape(
                -1, self.hidden_size)

            qk_nope_tp = torch.chunk(qk_nope, self.tp_size, dim=0)
            qk_rope_tp = torch.chunk(qk_rope, self.tp_size, dim=0)
            kv_nope_tp = torch.chunk(kv_nope, self.tp_size, dim=0)
            linear_v_tp = torch.chunk(linear_v, self.tp_size, dim=0)
        else:
            q_up_key = f'{prefix}.linear_q_up_proj.weight'
            kv_up_key = f'{prefix}.linear_kv_up_proj.weight'
            q_b_tp = torch.chunk(q_b_proj, self.tp_size, dim=0)
            kv_b_tp = torch.chunk(kv_b_proj, self.tp_size, dim=0)

        o_proj_tp = torch.chunk(o_proj, self.tp_size, dim=1)
        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][qkv_key] = qkv_weight.clone()
                mg_model[ep_rank][tp_rank][proj_key] = o_proj_tp[
                    tp_rank].clone()
                mg_model[ep_rank][tp_rank][q_norm_key] = q_ln.clone()
                mg_model[ep_rank][tp_rank][kv_norm_key] = kv_ln.clone()

                if self.mla_mm_split:
                    mg_model[ep_rank][tp_rank][qk_nope_key] = qk_nope_tp[
                        tp_rank].clone()
                    mg_model[ep_rank][tp_rank][qk_rope_key] = qk_rope_tp[
                        tp_rank].clone()
                    mg_model[ep_rank][tp_rank][kv_nope_key] = kv_nope_tp[
                        tp_rank].clone()
                    mg_model[ep_rank][tp_rank][linear_v_key] = linear_v_tp[
                        tp_rank].clone()
                else:
                    mg_model[ep_rank][tp_rank][q_up_key] = q_b_tp[
                        tp_rank].clone()
                    mg_model[ep_rank][tp_rank][kv_up_key] = kv_b_tp[
                        tp_rank].clone()

                self._maybe_quant_nf4(mg_model[ep_rank][tp_rank], proj_key,
                                      o_proj_tp[tp_rank].clone())

    def _set_layer_mlp(
        self,
        hf_layer: int,
        local_layer_idx: int,
        weights: dict[str, torch.Tensor],
        mg_model: dict[int, dict[int, dict[str, torch.Tensor]]],
    ) -> None:
        prefix = f'decoder.layers.{local_layer_idx}.mlp'

        if hf_layer < self.first_k_dense_replace:
            gate = weights.pop(f'model.layers.{hf_layer}.mlp.gate_proj.weight')
            up = weights.pop(f'model.layers.{hf_layer}.mlp.up_proj.weight')
            down = weights.pop(f'model.layers.{hf_layer}.mlp.down_proj.weight')

            fc1 = torch.cat([gate, up], dim=0)
            fc1_tp = torch.chunk(fc1, self.tp_size, dim=0)
            fc2_tp = torch.chunk(down, self.tp_size, dim=1)
            for ep_rank in range(self.ep_size):
                for tp_rank in range(self.tp_size):
                    mg_model[ep_rank][tp_rank][
                        f'{prefix}.linear_fc1.weight'] = fc1_tp[tp_rank].clone(
                        )
                    mg_model[ep_rank][tp_rank][
                        f'{prefix}.linear_fc2.weight'] = fc2_tp[tp_rank].clone(
                        )
                    self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                          f'{prefix}.linear_fc1.weight',
                                          fc1_tp[tp_rank].clone())
                    self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                          f'{prefix}.linear_fc2.weight',
                                          fc2_tp[tp_rank].clone())
            return

        router_w = weights.pop(
            f'model.layers.{hf_layer}.mlp.gate.weight')[:self.num_experts, :]
        router_b = weights.pop(
            f'model.layers.{hf_layer}.mlp.gate.e_score_correction_bias'
        )[:self.num_experts]

        shared_gate = weights.pop(
            f'model.layers.{hf_layer}.mlp.shared_experts.gate_proj.weight')
        shared_up = weights.pop(
            f'model.layers.{hf_layer}.mlp.shared_experts.up_proj.weight')
        shared_down = weights.pop(
            f'model.layers.{hf_layer}.mlp.shared_experts.down_proj.weight')

        shared_fc1 = torch.cat([shared_gate, shared_up], dim=0)
        shared_fc1_tp = torch.chunk(shared_fc1, self.tp_size, dim=0)
        shared_fc2_tp = torch.chunk(shared_down, self.tp_size, dim=1)

        experts_linear_fc1_list: list[torch.Tensor] = []
        experts_linear_fc2_list: list[torch.Tensor] = []
        expert_tp_size = self.tp_size if self.moe_tp_extend_ep else 1
        for expert in range(self.num_experts):
            gate = weights.pop(
                f'model.layers.{hf_layer}.mlp.experts.{expert}.gate_proj.weight'
            )
            up = weights.pop(
                f'model.layers.{hf_layer}.mlp.experts.{expert}.up_proj.weight')
            down = weights.pop(
                f'model.layers.{hf_layer}.mlp.experts.{expert}.down_proj.weight'
            )

            gate_chunks = torch.chunk(gate, expert_tp_size, dim=0)
            up_chunks = torch.chunk(up, expert_tp_size, dim=0)
            fc1 = torch.cat(
                [x for pair in zip(gate_chunks, up_chunks) for x in pair],
                dim=0)
            experts_linear_fc1_list.append(fc1.t())
            experts_linear_fc2_list.append(down.t())

        router_key = f'{prefix}.router.weight'
        router_bias_key = f'{prefix}.router.expert_bias'
        shared_fc1_key = f'{prefix}.shared_experts.linear_fc1.weight'
        shared_fc2_key = f'{prefix}.shared_experts.linear_fc2.weight'
        experts_weight1_key = f'{prefix}.experts.weight1'
        experts_weight2_key = f'{prefix}.experts.weight2'

        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][router_key] = router_w.clone()
                mg_model[ep_rank][tp_rank][router_bias_key] = router_b.clone()
                mg_model[ep_rank][tp_rank][shared_fc1_key] = shared_fc1_tp[
                    tp_rank].clone()
                mg_model[ep_rank][tp_rank][shared_fc2_key] = shared_fc2_tp[
                    tp_rank].clone()
                self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                      shared_fc1_key,
                                      shared_fc1_tp[tp_rank].clone())
                self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                      shared_fc2_key,
                                      shared_fc2_tp[tp_rank].clone())

        if self.moe_grouped_gemm:
            gemm_fc1 = torch.cat(experts_linear_fc1_list).view(
                self.hidden_size, -1)
            gemm_fc2 = torch.cat(experts_linear_fc2_list).view(
                -1, self.hidden_size)
            if self.moe_tp_extend_ep:
                gemm_fc1_ep = torch.chunk(
                    gemm_fc1.view(self.num_experts, self.hidden_size, -1),
                    self.ep_size * self.tp_size,
                    dim=0,
                )
                gemm_fc2_ep = torch.chunk(
                    gemm_fc2.view(self.num_experts, -1, self.hidden_size),
                    self.ep_size * self.tp_size,
                    dim=0,
                )
                for ep_rank in range(self.ep_size):
                    for tp_rank in range(self.tp_size):
                        idx = ep_rank * self.tp_size + tp_rank
                        w1 = gemm_fc1_ep[idx].reshape(self.hidden_size,
                                                      -1).clone()
                        w2 = gemm_fc2_ep[idx].reshape(
                            -1, self.hidden_size).clone()
                        mg_model[ep_rank][tp_rank][experts_weight1_key] = w1
                        mg_model[ep_rank][tp_rank][experts_weight2_key] = w2
                        self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                              experts_weight1_key, w1)
                        self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                              experts_weight2_key, w2)
            else:
                gemm_fc1_ep = torch.chunk(
                    gemm_fc1.view(self.num_experts, self.hidden_size, -1),
                    self.ep_size,
                    dim=0,
                )
                gemm_fc2_ep = torch.chunk(
                    gemm_fc2.view(self.num_experts, -1, self.hidden_size),
                    self.ep_size,
                    dim=0,
                )
                for ep_rank in range(self.ep_size):
                    fc1_tp = torch.chunk(gemm_fc1_ep[ep_rank],
                                         self.tp_size,
                                         dim=2)
                    fc2_tp = torch.chunk(gemm_fc2_ep[ep_rank],
                                         self.tp_size,
                                         dim=1)
                    for tp_rank in range(self.tp_size):
                        w1 = fc1_tp[tp_rank].reshape(self.hidden_size,
                                                     -1).clone()
                        w2 = fc2_tp[tp_rank].reshape(-1,
                                                     self.hidden_size).clone()
                        mg_model[ep_rank][tp_rank][experts_weight1_key] = w1
                        mg_model[ep_rank][tp_rank][experts_weight2_key] = w2
                        self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                              experts_weight1_key, w1)
                        self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                              experts_weight2_key, w2)
        else:
            num_local_experts = self.num_experts // self.ep_size
            for ep_rank in range(self.ep_size):
                for local_idx in range(num_local_experts):
                    global_idx = local_idx + ep_rank * num_local_experts
                    local_fc1 = experts_linear_fc1_list[global_idx].t()
                    local_fc2 = experts_linear_fc2_list[global_idx].t()
                    local_fc1_tp = torch.chunk(local_fc1, self.tp_size, dim=0)
                    local_fc2_tp = torch.chunk(local_fc2, self.tp_size, dim=1)
                    local_prefix = f'{prefix}.experts.local_experts.{local_idx}'
                    for tp_rank in range(self.tp_size):
                        mg_model[ep_rank][tp_rank][
                            f'{local_prefix}.linear_fc1.weight'] = local_fc1_tp[
                                tp_rank].clone()
                        mg_model[ep_rank][tp_rank][
                            f'{local_prefix}.linear_fc2.weight'] = local_fc2_tp[
                                tp_rank].clone()
                        self._maybe_quant_nf4(
                            mg_model[ep_rank][tp_rank],
                            f'{local_prefix}.linear_fc1.weight',
                            local_fc1_tp[tp_rank].clone())
                        self._maybe_quant_nf4(
                            mg_model[ep_rank][tp_rank],
                            f'{local_prefix}.linear_fc2.weight',
                            local_fc2_tp[tp_rank].clone())

    def _save_pp_rank(
        self,
        pp_rank: int,
        mg_model: dict[int, dict[int, dict[str, torch.Tensor]]]
        | dict[int, dict[int, dict[int, dict[str, torch.Tensor]]]],
        vpp: bool,
    ) -> None:
        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                prefix = _mp_prefix(tp_rank, pp_rank, ep_rank, self.tp_size,
                                    self.pp_size, self.ep_size)
                outdir = os.path.join(self.iter_path, prefix)
                os.makedirs(outdir, exist_ok=True)
                outpath = os.path.join(outdir, 'model_optim_rng.pt')
                if vpp:
                    payload = {
                        'model0': mg_model[0][ep_rank][tp_rank],
                        'model1': mg_model[1][ep_rank][tp_rank],
                        'checkpoint_version': 3.0,
                        'iteration': 1,
                    }
                else:
                    payload = {
                        'model': mg_model[ep_rank][tp_rank],
                        'checkpoint_version': 3.0,
                        'iteration': 1,
                    }
                torch.save(payload,
                           outpath,
                           pickle_protocol=4,
                           _use_new_zipfile_serialization=True)

    def run(self) -> None:
        if self.vpp_stage is None:
            for pp_rank in range(self.pp_size):
                logger.info('pp_rank=%s/%s', pp_rank, self.pp_size)
                mg_model: dict[int, dict[int, dict[
                    str,
                    torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
                weights = self._load_matched_hf_weights(pp_rank, None)
                if pp_rank == 0:
                    self._set_preprocess(weights, mg_model)
                for local_layer_idx, hf_layer in enumerate(
                        self.pprank_layer_idxs[pp_rank]):
                    self._set_layer_norm(hf_layer, local_layer_idx, weights,
                                         mg_model)
                    self._set_layer_attn(hf_layer, local_layer_idx, weights,
                                         mg_model)
                    self._set_layer_mlp(hf_layer, local_layer_idx, weights,
                                        mg_model)
                if pp_rank == self.pp_size - 1 and not self.dualpipe:
                    self._set_postprocess(weights, mg_model)
                self._save_pp_rank(pp_rank, mg_model, vpp=False)
            return

        for pp_rank in range(self.pp_size):
            logger.info(
                'pp_rank=%s/%s (vpp=%s stage=%s)',
                pp_rank,
                self.pp_size,
                self.vpp_size,
                self.vpp_stage,
            )
            mg_model: dict[int, dict[int, dict[int, dict[
                str, torch.Tensor]]]] = defaultdict(
                    lambda: defaultdict(lambda: defaultdict(dict)))

            for vpp_rank in range(self.vpp_size):
                weights = self._load_matched_hf_weights(pp_rank, vpp_rank)
                if pp_rank == 0 and vpp_rank == 0:
                    self._set_preprocess(weights, mg_model[vpp_rank])
                if self.dualpipe and pp_rank == 0 and vpp_rank == 1:
                    self._set_postprocess(weights, mg_model[vpp_rank])

                layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]
                for local_layer_idx, hf_layer in enumerate(layer_list):
                    self._set_layer_norm(hf_layer, local_layer_idx, weights,
                                         mg_model[vpp_rank])
                    self._set_layer_attn(hf_layer, local_layer_idx, weights,
                                         mg_model[vpp_rank])
                    self._set_layer_mlp(hf_layer, local_layer_idx, weights,
                                        mg_model[vpp_rank])

                if (not self.dualpipe) and (pp_rank == self.pp_size - 1) and (
                        vpp_rank == self.vpp_size - 1):
                    self._set_postprocess(weights, mg_model[vpp_rank])

            self._save_pp_rank(pp_rank, mg_model, vpp=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir',
                        type=str,
                        required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir',
                        type=str,
                        required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument(
        '--target-tensor-parallel-size',
        type=int,
        default=1,
        help='Target tensor model parallel size, defaults to 1.')
    parser.add_argument(
        '--target-pipeline-parallel-size',
        type=int,
        default=1,
        help='Target pipeline model parallel size, defaults to 1.')
    parser.add_argument(
        '--target-expert-parallel-size',
        type=int,
        default=1,
        help='Target expert model parallel size, defaults to 1.')
    parser.add_argument('--num-layers-per-virtual-pipeline-stage',
                        '--vpp-stage',
                        dest='num_layers_per_virtual_pipeline_stage',
                        type=int,
                        default=None,
                        help='Number of layers per virtual pipeline stage')
    parser.add_argument('--moe-grouped-gemm',
                        action='store_true',
                        help='Use moe grouped gemm.')
    parser.add_argument('--noop-layers',
                        type=str,
                        default='47',
                        help='Specity the noop layers.')
    parser.add_argument('--mtp-num-layers',
                        type=int,
                        default=0,
                        help='Multi-Token prediction layer num')
    parser.add_argument(
        '--num-layer-list',
        type=str,
        help='a list of number of layers, separated by comma; e.g., 4,4,4,4')
    parser.add_argument('--num-layers',
                        type=int,
                        default=48,
                        help='Number of transformer layers.')
    parser.add_argument('--first-k-dense-replace',
                        type=int,
                        default=3,
                        help='Customizing the number of dense layers.')
    parser.add_argument(
        '--moe-tp-extend-ep',
        action='store_true',
        help=
        'use tp group to extend experts parallism instead of sharding weight tensor of experts in tp group',
    )
    parser.add_argument('--mla-mm-split',
                        action='store_true',
                        default=False,
                        help='Split 2 up-proj matmul into 4 in MLA')
    parser.add_argument(
        '--schedules-method',
        type=str,
        default=None,
        choices=['dualpipev'],
        help='An innovative bidirectional pipeline parallelism algorithm.')
    parser.add_argument('--qlora-nf4',
                        action='store_true',
                        help='use bitsandbytes nf4 to quantize model.')

    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = get_args()
    logger.info('Arguments: %s', args)
    converter = CkptConvert(
        hf_model_path=args.load_dir,
        mg_save_path=args.save_dir,
        num_layers=args.num_layers,
        tp_size=args.target_tensor_parallel_size,
        pp_size=args.target_pipeline_parallel_size,
        ep_size=args.target_expert_parallel_size,
        first_k_dense_replace=args.first_k_dense_replace,
        hidden_size=HIDDEN_SIZE,
        num_experts=NUM_EXPERTS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        qk_head_dim=QK_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        qk_pos_emb_head_dim=QK_POS_EMB_HEAD_DIM,
        moe_grouped_gemm=args.moe_grouped_gemm,
        moe_tp_extend_ep=args.moe_tp_extend_ep,
        mla_mm_split=args.mla_mm_split,
        schedules_method=args.schedules_method,
        vpp_stage=args.num_layers_per_virtual_pipeline_stage,
        num_layer_list=args.num_layer_list,
        noop_layers=args.noop_layers,
        qlora_nf4=args.qlora_nf4,
    )
    converter.run()


if __name__ == '__main__':
    main()
