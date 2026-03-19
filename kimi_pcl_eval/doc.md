【内部研发文档】Kimi-K2 1T 模型 vLLM 部署与 OpenCompass 评测基建指南
0. 任务背景与核心目标
我们目前已经成功将一个 1T 参数量的自研模型（底层架构类 DeepSeek-V3）的权重转换为了 HuggingFace 标准的 .safetensors 格式。
当前你的核心任务是开发一套自动化测试与评测的脚手架代码。当你拿到具备 vLLM 和 OpenCompass 环境的机器/镜像后，需要能够一键执行以下三个目标：

成功加载模型并跑通 vLLM 的基础推理验证。

集成 OpenCompass 评测框架。

完成 6 个核心基础 Benchmark 的自动化评测。

1. 核心前置说明（重点关注）
模型路径：/llm_workspace_1P/wf/ckpt/iter_0000900_test/（请在代码中将其提取为可通过 argparse 或环境变量配置的参数）。

架构特殊性：该模型使用了自定义的模型代码（modeling_deepseek.py 等）。因此，在任何使用 HuggingFace API 或 vLLM 加载的地方，必须开启 trust_remote_code=True。

硬件限制（OOM 预警）：这是一个 1000B 的巨型模型。单机 8 卡（哪怕是 8x96G）是绝对放不下的。测试时必须配置为多机多卡分布式推理，设置张量并行参数 tensor_parallel_size（例如 32），或者在代码里预留好量化参数的位置。

2. 任务一：构建 vLLM 极简测试脚本
需求：编写一个名为 test_vllm_generation.py 的脚本，只做一件事：加载模型，输入几个简单的 Prompt，打印输出。证明模型能正常吐字。

核心指导代码：

Python
import argparse
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="HF 模型文件夹路径")
    parser.add_argument("--tp_size", type=int, default=8, help="张量并行大小 (Tensor Parallel)")
    args = parser.parse_args()

    # 1. 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.95, 
        max_tokens=256
    )

    print(f"正在加载模型: {args.model_path}, TP={args.tp_size} ...")
    
    # 2. 初始化 vLLM
    # 关键坑位：必须开启 trust_remote_code=True，dtype 保持 bfloat16
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        dtype="bfloat16"
    )

    prompts = [
        "你好，请简单介绍一下你自己。",
        "Please write a Python function to compute the Fibonacci sequence."
    ]

    print("开始生成...")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print("-" * 40)
        print(f"Prompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")

if __name__ == "__main__":
    main()
3. 任务二：OpenCompass 评测基建
需求：使用上海人工智能实验室开源的 OpenCompass 框架，配置并跑通 6 个核心数据集。

3.1 选定的 6 个基础 Benchmark 集合
请在配置中拉取以下 6 个数据集（涵盖知识、代码、数学、中文、安全、推理）：

MMLU (mmlu): 英文综合知识百科能力。

C-Eval (ceval): 中文综合学科能力。

HumanEval (humaneval): 基础代码编写能力。

GSM8K (gsm8k): 小学数学逻辑推理。

TruthfulQA (truthfulqa): 模型抗幻觉与安全性。

HellaSwag (hellaswag): 常识与上下文推理。

3.2 核心指导：如何让 OpenCompass 挂载本地 vLLM 模型
不要使用原始的 HuggingFace 推理后端，会极其缓慢。OpenCompass 原生支持 vLLM 加速。
请在代码仓库中创建一个 OpenCompass 的专属配置文件（例如 configs/eval_kimi_1t_vllm.py）：

配置骨架参考：

Python
from mmengine.config import read_base
from opencompass.models import VLLM

# 1. 导入你需要的 6 个数据集 (OpenCompass 自带的配置)
with read_base():
    from .datasets.mmlu.mmlu_gen import mmlu_datasets
    from .datasets.ceval.ceval_gen import ceval_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from .datasets.humaneval.humaneval_gen import humaneval_datasets
    from .datasets.hellaswag.hellaswag_gen import hellaswag_datasets
    from .datasets.truthfulqa.truthfulqa_gen import truthfulqa_datasets

# 汇总 datasets
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# 2. 配置本地 vLLM 模型参数
models = [
    dict(
        type=VLLM,
        abbr='kimi-k2-1T-base',
        path='/llm_workspace_1P/wf/ckpt/iter_0000900_test/', # 替换为实际路径
        model_kwargs=dict(
            tensor_parallel_size=8, # 注意根据实际机器集群修改
            trust_remote_code=True, # 必须为 True
            dtype='bfloat16'
        ),
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=16,
        generation_kwargs=dict(temperature=0.0), # 评测通常采用贪心解码
        run_cfg=dict(num_gpus=8, num_procs=1),
    )
]
3.3 运行脚本要求
请编写一个 run_eval.sh 的外层调用脚本，里面封装好拉起 OpenCompass 的命令。例如：

Bash
#!/bin/bash
# 安装必要依赖 (按需取消注释)
# pip install opencompass vllm human_eval

# 启动评测
python run.py configs/eval_kimi_1t_vllm.py --mode all
4. 交付验收标准
提交一份包含 test_vllm_generation.py、configs/eval_kimi_1t_vllm.py 和 run_eval.sh 的完整 Git 仓库代码。

仓库内附带简明的 README.md，说明依赖的安装命令（requirements.txt）。

代码中对于模型路径、卡数（TP Size）必须做到解耦，方便后期在不同集群上修改。