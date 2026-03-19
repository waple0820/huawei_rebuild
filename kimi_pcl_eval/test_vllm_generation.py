#!/usr/bin/env python3
"""Kimi-K2 1T vLLM inference smoke test.

Loads the model via vLLM, runs a handful of prompts, and prints the
generated text.  A non-zero exit code means the model failed to load or
generate — useful as a quick sanity check before kicking off full
OpenCompass evaluations.
"""

import argparse
import os
import sys
import time

from vllm import LLM, SamplingParams

DEFAULT_MODEL_PATH = "/llm_workspace_1P/wf/ckpt/iter_0000900_test/"
DEFAULT_TP_SIZE = 8

TEST_PROMPTS = [
    "你好，请简单介绍一下你自己。",
    "Please write a Python function to compute the Fibonacci sequence.",
    "解释一下什么是Transformer模型，以及它为什么在自然语言处理中如此重要。",
    "Write a brief summary of the theory of relativity.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kimi-K2 1T vLLM inference smoke test"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Path to the HuggingFace model directory "
        "(env: MODEL_PATH, default: %(default)s)",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=int(os.environ.get("TP_SIZE", DEFAULT_TP_SIZE)),
        help="Tensor-parallel size "
        "(env: TP_SIZE, default: %(default)s)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (default: %(default)s)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[None, "awq", "gptq", "fp8"],
        help="Quantization method — useful when GPU memory is limited "
        "(default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print("=" * 60)
    print("Kimi-K2 1T  —  vLLM Inference Smoke Test")
    print("=" * 60)
    print(f"  Model path   : {args.model_path}")
    print(f"  TP size      : {args.tp_size}")
    print(f"  Max tokens   : {args.max_tokens}")
    print(f"  Temperature  : {args.temperature}")
    print(f"  Top-p        : {args.top_p}")
    print(f"  Quantization : {args.quantization or 'None'}")
    print("=" * 60)

    print(f"\n>>> Loading model from {args.model_path} (TP={args.tp_size}) ...")
    t0 = time.time()

    llm_kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization

    llm = LLM(**llm_kwargs)
    load_time = time.time() - t0
    print(f">>> Model loaded in {load_time:.1f}s\n")

    print(">>> Generating responses ...\n")
    t0 = time.time()
    outputs = llm.generate(TEST_PROMPTS, sampling_params)
    gen_time = time.time() - t0

    for idx, output in enumerate(outputs):
        print(f"{'─' * 60}")
        print(f"  [{idx + 1}/{len(outputs)}] Prompt:")
        print(f"    {output.prompt}")
        print("  Response:")
        for line in output.outputs[0].text.strip().splitlines():
            print(f"    {line}")
    print(f"{'─' * 60}")

    print(f"\n>>> Generation finished in {gen_time:.1f}s")
    print(">>> Smoke test PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n>>> Smoke test FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
