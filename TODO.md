# 1000B GQA 重构进度

> 详细设计与代码示例见 [rebuild.md](rebuild.md)

## 当前阶段: MLA -> GQA 改造

---

### Task 1: 配置文件与文件命名 → [rebuild.md 第一章](rebuild.md#一commit-1-配置文件与文件命名修正)

- [ ] 重写 `config.json` (hidden_size=7168, heads=112, kv_heads=28, layers=47, 删 MLA 字段)
- [ ] 重写 `DeepseekV3Config` (新增 `qk_layernorm`, 删 `q_lora_rank` 等)

### Task 2: HF 模型代码 → [rebuild.md 第二章](rebuild.md#二commit-2-hf-模型代码-mla---gqa)

- [x] `DeepseekV3Attention.__init__`: q/k/v/o_proj + 可选 qk_layernorm
- [x] `_init_rope`: dim 改为 head_dim=64
- [x] `softmax_scale`: 64^(-0.5) 替代 192^(-0.5)
- [x] `forward`: 标准 GQA (q_proj/k_proj/v_proj → RoPE → repeat_kv → attn → o_proj)
- [x] `DeepseekV3FlashAttention2.forward`: 同步改造
- [x] `apply_rotary_pos_emb`: 移除 MLA 特殊 reshape

### Task 3: mcore → HF 转换 → [rebuild.md 第三章](rebuild.md#三commit-3-mcore2hf-转换脚本修复)

- [x] 删除硬编码常量，新增 CLI `--num-key-value-heads`
- [x] `_set_layer_attn`: 正确 TP gather + QKV 拆分为 q_proj/k_proj/v_proj
- [x] 移除 `mla_mm_split` / `inv_freq` 逻辑
- [ ] `run()` 末尾自动生成 config.json + 拷贝模型文件

### Task 4: HF → mcore 转换 → [rebuild.md 第四章](rebuild.md#四commit-4-hf2mcore-转换脚本修复)

- [x] 删除硬编码常量，新增 CLI `--num-key-value-heads`
- [x] `_set_layer_attn`: 从 q_proj/k_proj/v_proj 拼接为 MCore linear_qkv
- [x] 移除 `mla_mm_split` 逻辑

### Task 5: 验证与收尾 → [rebuild.md 第五章](rebuild.md#五commit-5-验证与收尾)

- [ ] 端到端: mcore ckpt → HF → model.generate()
- [ ] round-trip: HF → mcore → HF 一致性
- [ ] 清理 + 更新 README

---

## 待验证风险项 → [rebuild.md 第六章](rebuild.md#六风险项与待验证事项)

- [ ] **R1**: MCore QKV 布局是 interleaved 还是 sequential
- [ ] **R2**: K layernorm key 是 `k_layernorm` 还是 `kv_layernorm`
- [x] **R3**: RoPE reshape 排列是否适用于 GQA (已移除 MLA 特殊 reshape)
