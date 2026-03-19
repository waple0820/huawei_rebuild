#!/usr/bin/env python3
"""Kimi-K2 1T vLLM multimodal inference smoke test.

Loads the model via vLLM, feeds it an image with a text prompt, and
prints the generated description.  Proves the model can handle
multimodal (image) inputs end-to-end.
"""

import argparse
import os
import sys
import time

from PIL import Image
from vllm import LLM, SamplingParams

DEFAULT_MODEL_PATH = "/llm_workspace_1P/wf/ckpt/iter_0000900_test/"
DEFAULT_TP_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kimi-K2 1T vLLM multimodal inference smoke test"
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
        "--image_path",
        type=str,
        default="test_image.jpg",
        help="Path to a test image file (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>\n请详细描述一下这张图片里有什么？",
        help="Multimodal prompt with <image> placeholder",
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
        default=0.2,
        help="Sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[None, "awq", "gptq", "fp8"],
        help="Quantization method (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Kimi-K2 1T  —  Multimodal vLLM Inference Smoke Test")
    print("=" * 60)
    print(f"  Model path   : {args.model_path}")
    print(f"  TP size      : {args.tp_size}")
    print(f"  Image        : {args.image_path}")
    print(f"  Max tokens   : {args.max_tokens}")
    print(f"  Temperature  : {args.temperature}")
    print(f"  Quantization : {args.quantization or 'None'}")
    print("=" * 60)

    if not os.path.isfile(args.image_path):
        print(f"ERROR: image file not found: {args.image_path}", file=sys.stderr)
        print("  Please provide a test image via --image_path.", file=sys.stderr)
        sys.exit(1)

    image = Image.open(args.image_path)
    print(f"\n>>> Loaded image: {image.size[0]}x{image.size[1]} {image.mode}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f">>> Loading model from {args.model_path} (TP={args.tp_size}) ...")
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

    print(">>> Running multimodal inference ...")
    t0 = time.time()
    outputs = llm.generate(
        {"prompt": args.prompt, "multi_modal_data": {"image": image}},
        sampling_params=sampling_params,
    )
    gen_time = time.time() - t0

    for output in outputs:
        print(f"{'─' * 60}")
        print("  Prompt:")
        print(f"    {args.prompt}")
        print("  Response:")
        for line in output.outputs[0].text.strip().splitlines():
            print(f"    {line}")
    print(f"{'─' * 60}")

    print(f"\n>>> Generation finished in {gen_time:.1f}s")
    print(">>> Multimodal smoke test PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n>>> Multimodal smoke test FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
