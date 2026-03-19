"""OpenCompass evaluation config — Kimi-K2 1T via vLLM backend.

All tunables are read from environment variables so the file never needs
manual editing.  Set them before launching OpenCompass:

    export MODEL_PATH=/llm_workspace_1P/wf/ckpt/iter_0000900_test/
    export TP_SIZE=8
    opencompass configs/eval_kimi_1t_vllm.py
"""

import os

from mmengine.config import read_base
from opencompass.models import VLLM

# ---------------------------------------------------------------------------
# Configurable parameters (override via environment variables)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/llm_workspace_1P/wf/ckpt/iter_0000900_test/"
)
TP_SIZE = int(os.environ.get("TP_SIZE", "8"))
MAX_OUT_LEN = int(os.environ.get("MAX_OUT_LEN", "1024"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "4096"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))

# ---------------------------------------------------------------------------
# Datasets — 6 core benchmarks
# ---------------------------------------------------------------------------
with read_base():
    from .datasets.mmlu.mmlu_gen import mmlu_datasets
    from .datasets.ceval.ceval_gen import ceval_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from .datasets.humaneval.humaneval_gen import humaneval_datasets
    from .datasets.hellaswag.hellaswag_gen import hellaswag_datasets
    from .datasets.truthfulqa.truthfulqa_gen import truthfulqa_datasets

datasets = sum(
    (v for k, v in locals().items() if k.endswith("_datasets")), []
)

# ---------------------------------------------------------------------------
# Model — local vLLM inference
# ---------------------------------------------------------------------------
models = [
    dict(
        type=VLLM,
        abbr="kimi-k2-1T-base",
        path=MODEL_PATH,
        model_kwargs=dict(
            tensor_parallel_size=TP_SIZE,
            trust_remote_code=True,
            dtype="bfloat16",
        ),
        max_out_len=MAX_OUT_LEN,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        generation_kwargs=dict(temperature=0.0),
        run_cfg=dict(num_gpus=TP_SIZE, num_procs=1),
    )
]
