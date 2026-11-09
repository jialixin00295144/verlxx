# Model Evaluation

Last updated: 07/14/2026.

The steps are the same for different models. This section lists them using only Qwen3-30B as an example.

We evaluate the model using AISBench, which supports evaluation across multiple inference backends such as vllm/sglang.

## 1. Installation method

~~~bash
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
~~~~~~~~~~~~~~~~


## 2. Download the evaluation dataset

~~~bash
cd path/to/benchmark/ais_bench/datasets
wget https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
unzip math.zip
rm math.zip
~~~~~~~~~~~

## 3. Weight Conversion

Currently, verl supports using mbridge to directly save model weights in the HF format, which you can use without conversion.

If the model weights are not in the HF format, convert them to the HF format first, and then perform the evaluation.

This refers to the native verl [conversion method](../../../../../docs/advance/checkpoint.rst)

## 4.vllm inference evaluation

**Starting the vllm_server service**

Start the inference server using the following command. You need to modify the model and tensor-parallel-size parameters.

model: the path to the Hugging Face model that stores the converted post-training weights;

tensor-parallel-size: The number of tensor parallelism replicas. It is recommended that TP remains consistent with the infer configuration during training;

data-parallel-size: The number of data parallel replicas. Keep DP consistent with the infer configuration during training. The default value is 1;

port: You can set this to any idle port;

~~~bash
vllm serve /path/to/Qwen3-30B/ \
       --served-model-name auto \
       --gpu-memory-utilization 0.9 \
       --max-num-seqs 24 \
       --max-model-len 22528 \
       --max-num-batched-tokens 22528 \
       --enforce-eager \
       --trust-remote-code \
       --distributed_executor_backend=mp \
       --tensor-parallel-size 8 \
       --data-parallel-size 1 \
       --generation-config vllm \
       --port 8080
~~~

**Modify the AISBench inference configuration to start the vllm_client evaluation**

Open the inference configuration file benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py

The host_port must match the port on the server. Modify max_seq_len and max_out_len based on the model configuration.
~~~bash
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="",
        model="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",
        host_port = 8080,
        max_out_len = 512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
~~~

Open another window for evaluation, and run the evaluation command:
~~~bash
    ais_bench --models vllm_api_stream_chat --datasets math500_gen_0_shot_cot_chat_prompt
~~~
## 5. SGLang inference evaluation
For the evaluation procedure, refer to [Ascend SGLang Best Practices](../../model_support/examples/ascend_sglang_best_practices.rst).