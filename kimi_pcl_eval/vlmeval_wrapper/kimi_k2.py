"""VLMEvalKit model wrapper for Kimi-K2 1T (vLLM backend).

Usage — copy this file into the VLMEvalKit source tree:

    cp kimi_k2.py  VLMEvalKit/vlmeval/vlm/kimi_k2.py

Then register it in VLMEvalKit/vlmeval/vlm/__init__.py:

    from .kimi_k2 import KimiK2

And add an entry in VLMEvalKit/vlmeval/config.py  supported_VLM:

    'Kimi-K2-1T': partial(KimiK2, model_path='/path/to/model', tp_size=32),
"""

import os
import warnings

from PIL import Image
from vllm import LLM, SamplingParams

from .base import BaseModel


class KimiK2(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path=None,
        tp_size=None,
        max_new_tokens=1024,
        temperature=0.0,
        quantization=None,
        **kwargs,
    ):
        super().__init__()

        model_path = model_path or os.environ.get(
            "MODEL_PATH", "/llm_workspace_1P/wf/ckpt/iter_0000900_test/"
        )
        tp_size = tp_size or int(os.environ.get("TP_SIZE", "8"))

        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        if quantization:
            llm_kwargs["quantization"] = quantization

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        self.model_path = model_path

        default_kwargs = dict(max_new_tokens=max_new_tokens, temperature=temperature)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f"KimiK2 generation config: {self.kwargs}")

    # ------------------------------------------------------------------
    # Required by BaseModel
    # ------------------------------------------------------------------

    def generate_inner(self, message, dataset=None):
        """Generate a response from a list of message dicts.

        Each dict has keys ``type`` ('text' | 'image' | 'video') and
        ``value`` (str content or file path).
        """
        prompt_parts = []
        images = []
        video = None

        for item in message:
            if item["type"] == "text":
                prompt_parts.append(item["value"])
            elif item["type"] == "image":
                images.append(Image.open(item["value"]).convert("RGB"))
                prompt_parts.append("<image>")
            elif item["type"] == "video":
                video = item["value"]
                prompt_parts.append("<video>")

        prompt = "\n".join(prompt_parts)

        mm_data = {}
        if images:
            mm_data["image"] = images if len(images) > 1 else images[0]
        if video is not None:
            mm_data["video"] = video

        inputs = {"prompt": prompt}
        if mm_data:
            inputs["multi_modal_data"] = mm_data

        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text.strip()
