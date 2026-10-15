# Guide to Migrating Models to the NPU

Last updated: 05/14/2026

This article provides developers with comprehensive practical experience for migrating from GPU to NPU or independently adapting models on NPU. It covers the entire workflow, including preliminary preparation, component integration, accuracy alignment, performance optimization, and long-running evaluation.

## 1. Preliminary Preparation

Set up a basic runtime environment that supports NPU execution to ensure that models load correctly and datasets are read smoothly. This provides the foundation for subsequent migration, debugging, and business pipeline validation.


### 1.1 Software and hardware environment and dependency configuration

Refer to the official documentation [Ascend Installation Guide](../../get_start/install_guidance.rst). If the versions of the inference engines vllm and vllm_ascend, and the training engines Megatron, MindSpeed, and transformers that the model depends on differ from those in the tutorial, **the versions actually adapted by the model take precedence**.

### 1.2 Model Weights

BF16 is the **default mixed precision training data type** for training backends such as FSDP and Megatron in the VeRL framework. The Ascend NPU environment consistently uses **BF16** as the baseline precision format, and weights must be aligned and dequantized to BF16. Currently, A2 and A3 products **do not support FP8 precision training** and only support BF16 precision. Subsequent versions of the Ascend 950PR and 950DT series products will provide FP8 low-precision training capability.

### 1.3 Data Preparation

For preprocessing the dataset into parquet format, refer to [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html): (1) to ensure it contains the necessary fields required for computing reinforcement learning rewards; (2) to achieve faster reading speed.

## 2. Joint debugging of all components

The VeRL framework uses a decoupled architecture that separates the inference engine, training engine, and weight synchronization bridge (Checkpoint Engine). This design achieves deep separation of computation and data, providing a flexible foundation for migrating and adapting models to the Ascend NPU. When you migrate and adapt models to the NPU, prioritize the separate adaptation and verification of the inference engine, training engine, and Megatron-Bridge components. After the components run stably, advance the connection and debugging of the entire VeRL network link. For specific feature support of different VeRL inference and training backends on the Ascend NPU, refer to the [Ascend Backend Feature Guide](../../feature_support/ascend_backend_features.md).

### 2.1 Inference Engine Adaptation

The VeRL inference engine uses a layered architecture design. Through abstract interfaces and the factory pattern, it flexibly supports multiple mainstream inference backends such as vllm and sglang. During the migration and adaptation from GPU to NPU, follow these steps to adapt the inference engine:

Before running the full VeRL pipeline on the NPU, we recommend referring to the official model deployment tutorials for [vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/main/docs/source/tutorials/models) and [sglang](https://github.com/sgl-project/sglang/tree/main/docs/docs/basic_usage). Prioritize getting the **single-instance inference pipeline** working. Fully verify **basic inference features** such as model loading and initialization, proper Tokenizer loading, single-turn / batch generation, stop word termination, and long-context inference. After the underlying inference engine is stable and usable, integrate it into the VeRL training workflow.

### 2.2 Training Engine Selection and Adaptation

The main VeRL code abstracts the training engine into the `Engine` class. It decouples scheduling logic from the underlying training implementation through a standardized interface layer. This architecture supports the flexible, plug-and-play integration of multiple training backends, such as FSDP and Megatron. It significantly reduces migration and adaptation costs without modifying the core VeRL algorithms and scheduling logic.

The NPU has completed automatic device detection through the `is_npu_available` interface and automatically applied the corresponding NPU device adaptation patch. Currently, you only need to configure model_engine=fsdp/megatron to switch the training backend to FSDP or Megatron in a single step. The system automatically loads the NPU adaptation logic for the corresponding backend, without requiring additional code modifications. In VeRL, Ascend has adapted and optimized Megatron. For specific feature configuration, refer to the [verl-MindSpeed feature documentation](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/verl.md).

### 2.3 Megatron-Bridge adaptation

Megatron-Bridge performs bidirectional conversion between the HuggingFace weights required by the inference engine and the mcore weights required by Megatron-Core in the VeRL framework. To enable this feature, use the following configuration:

```
actor_rollout_ref.actor.megatron.use_mbridge=True
```

Megatron-Bridge has natively adapted a large number of mainstream model architectures in the community. For the supported list, refer to [supported model](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/README.md). When performing model migration and adaptation in the Ascend NPU environment, you can complete the basic configuration based on existing community capabilities. However, some special model structures and scenarios still require additional customized adaptation.

This guide uses the DSA (DeepSeek Sparse Attention) sparse attention structure as an example to introduce the method for customized adaptation. Ascend MindSpeed supports DSA capabilities based on absorption matrices. This feature requires splitting the original `linear_kv_up_proj` operator in Megatron into two independent operators: `linear_k_up_proj` and `linear_v_up_proj`. The weights required for the split must be converted from the HuggingFace format `self_attn.kv_b_proj.weight`. However, the aforementioned native PR does not support this operator splitting logic.

Therefore, you need to manually modify and adapt the related weight conversion logic to ensure that the absorption matrix can be properly loaded and take effect. Only when the absorption matrix is available can you properly enable the [sparse\_flash\_attention](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_attention) and [lightning\_indexer](https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer) fused operators. By introducing these two fused operators, you can significantly reduce the frequency of memory accesses, optimize memory usage, and improve computing performance. This ultimately improves the running efficiency and reduces the resource overhead of the large model training and inference pipelines.

### 2.4 Enabling End-to-End Network Functionality

Complete the inference engine adaptation verification and training engine adaptation development. Refer to [Training Configuration Parameters and Metrics Description](parameter_and_metrics.md) to configure the relevant parameters for the inference engine and training engine based on actual business requirements. Complete the VeRL end-to-end functionality integration to ensure the stable operation of the entire process.

## 3. Precision Alignment

The troubleshooting pipeline for precision issues in large language model (LLM) reinforcement learning is complex and involves many factors. Various precision issues typically originate from **training phase, inference phase, and training-inference consistency** issues. **Precision alignment** is critical for ensuring that the training process is reproducible and issues can be debugged.

For precision alignment during the training and inference phases, refer to the official documentation: [Precision Alignment Guide](../precision_analysis/precision_alignment.md). Therefore, this section does not reiterate the basic phase alignment process. Instead, it **focuses on the training-inference consistency scenario** and uses the msprobe precision tool to carry out precision alignment implementation practices and problem localization and troubleshooting.

### 3.1 Precision Monitoring Configuration

After the entire network runs successfully, enable the precision monitoring parameter by setting `actor_rollout_ref.rollout.calculate_log_probs=True`. During training, focus on observing the following key metrics to determine the training-inference consistency and the model training stability:

* **Training and inference consistency reference metrics**:
  * `training/rollout_probs_diff_mean` (mean of rollout probability differences): When the model converges normally, we recommend keeping this metric within 0.01. If the value consistently exceeds 0.01 or deviates significantly from the GPU baseline, you can determine that a training and inference precision anomaly exists.
  * `training/rollout_probs_diff_max` (maximum of rollout probability differences)
  * `training/rollout_actor_probs_pearson_corr` (Pearson correlation coefficient between rollout and actor probabilities)
* **Model training stability metrics**:
  * `actor/grad_norm`: Check whether it shows an overall downward trend to determine whether the model training converges normally.

In addition, configure the parameter `trainer.rollout_data_dir=./rollout_dump/` to save intermediate rollout results during training. You can manually check the exported rollout data to verify whether model replies meet expectations and whether outputs contain garbled text or repeated answers. This further confirms that the inference engine adaptation is correct.

### 3.2 Collecting Precision Data

If training/rollout_probs_diff_mean exceeds the reasonable threshold of 0.01 or deviates significantly from the GPU baseline, use the [msprobe](../../../en/dev_guide/precision_analysis/precision_debugger.md) precision tool to collect data and locate the root cause.

### 3.3 Troubleshooting and Alignment Practices for Training and Inference Discrepancies

After completing data collection, read the `construction.json` file first to perform module-level data comparison. Ensure that the input data of `layer.0.input_layernorm` is completely consistent. Then verify module by module and layer by layer to locate where the training and inference outputs first become inconsistent.

For large models, minor numerical differences accumulate and amplify layer by layer, causing significant differences between training and rollout results. In some cases, the same token can have an output probability of 0 during training and 1 during rollout. Therefore, align every difference point to be exactly equal as much as possible.

After you locate the difference node, determining the adaptation and modification scheme is also a key challenge. Various open-source communities in the industry provide multiple different implementations for related modules. To ensure the correctness of the model implementation logic, refer to authoritative source code and technical reports from multiple sources to comprehensively determine the final alignment scheme.

#### 3.3.1 Common training and inference inconsistencies

In large language model (LLM) reinforcement learning practice, you can categorize the typical root causes of training-inference inconsistency into the following five types:

1. **Inconsistent framework implementations**: This is caused by different implementation logic in training and inference frameworks. Sometimes it is "semantically correct" (for example, different operator splitting methods but mathematically equivalent), and sometimes it is "semantically incorrect" (for example, missing a scaling factor or having an extra operation). You must strictly verify this using the source code and technical reports.
2. **Precision type differences**: For example, training uses BF16 throughout, while inference implicitly upgrades precision to FP32 for calculation and then downgrades precision in sensitive operators such as normalization, causing truncation errors.
3. **Inconsistent hyperparameters**: For example, the hardcoded `eps` value in the LayerNorm module is not standardized.
4. **Parallel strategies**: Tensor parallelism during training versus continuous batching during inference causes differences in the floating-point accumulation order.
5. **Randomness control**: Dropout and sampling strategies have implementation deviations between the training and inference phases.

This section lists typical real-world cases of training and inference inconsistencies encountered during the GLM-5 model migration and adaptation process.

#### 3.3.2 Case 1: Inconsistent framework implementation of the FFN activation function

Compare sequentially from top to bottom, and trace the output inconsistency to the MLP activation function at the first layer.

The inference side is already using the NPU-optimized `npu_swiglu` fused operator, but the training side still executes the native GLU small operator implementation.

* **Root cause**: Although the configuration to enable `swiglu` has been added to the VeRL parameters, Megatron-Bridge does not explicitly configure `provider.bias_activation_fusion=True` in the NPU adaptation PR. As a result, the code does not enter the NPU fused operator branch.
* **Fix**: Add configuration items in Megatron-Bridge so that the training side correctly calls the fused operators:
  ```
  +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
  ```

#### 3.3.3 Case 2: Inconsistency between the precision of indexer_k_norm and hyperparameters

During the strict alignment process, an inconsistency was found between the precision type and the hyperparameter at `indexer_k_norm`:

* **Precision difference**: On the inference side, LayerNorm involves upcasting to fp32 in the operation `F.layer_norm( x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)`, while the training side Megatron implementation uses BF16. The minor difference becomes non-negligible after accumulating across multiple layers.
* **Fix**: Unify the training side code to add the upcast and downcast operations.
* **Hyperparameter difference**: vllm on the GLM5 inference side inherits the DeepSeek-V3.2 logic, where the EPS value of `k_norm` is hardcoded to `1e-6`; however, the training engine and the official technical report uniformly use `1e-5`.
* **Fix**: Modify the inference side EPS to `1e-5` to align with the training side.

```
self.k_norm=LayerNorm(self.head_dim,eps=1e-6 -> 1e-5)
```

#### 3.3.4 Case 3: Missing and redundant logic in the lightning_indexer module

Troubleshooting revealed that the lightning_indexer module has inconsistent implementations on the training and inference sides, specifically:

* **Missing (omitted on the inference side)**: The inference side is missing the scaling logic for `weights`. The standard implementations in the Megatron training side, slime, and transformers all include this scaling. Therefore, it is added on the inference side to align the forward pass:

```
weights, _ = self.weights_proj(x)
+weights = weights * (self.n_head**-0.5) * (self.head_dim**-0.5)
```

* **Redundancy (unnecessary and incorrect implementation on the training side)**: The Megatron implementation on the training side includes an extra `rotate_activation` (Hadamard transform). Extensive research confirms that this operation is specifically for quantization scenarios and is an incorrect implementation in the BF16 format. Following [Transformer PR#45017](https://github.com/huggingface/transformers/pull/45017), this redundant logic has been removed from the training side.

```
class DSAIndexer(MegatronModule):
    def forward_with_scores(
-		q = rotate_activation(q)
-		k = rotate_activation(k)
```

### 3.4 General Routing Stability Solution for MoE Large Models

In a typical reinforcement learning (RL) training process, an efficient inference engine such as vLLM samples data, and then the sampled data is sent to a training framework such as Megatron to optimize model training.

For conventional dense models, implementation and environment differences between inference and training frameworks only cause minor numerical deviations. However, this issue is drastically amplified for **large-scale MoE models**. The root cause lies in the MoE dynamic routing mechanism: even minor differences in framework implementation and runtime environment can route the same input token to completely different expert combinations, resulting in entirely different computation paths.

This inconsistency in routing decisions may cause unstable RL training for MoE models. It makes the "experience" obtained from the inference phase completely different for the training phase. As a result, the optimization signal is distorted, ultimately leading to catastrophic consequences.

To solve this common problem, the industry introduced the **Routing Replay** mechanism. Its core idea is to lock the expert routing paths at specific stages, shielding routing decisions from minor perturbations and thereby ensuring the stability of model training. Currently, the mainstream includes two variants: R2 and R3:

* **(1) Vanilla Routing Replay (R2)**: (This corresponds to `actor_rollout_ref.actor.megatron.router_replay.mode="R2"`, and for VeOmni, it is `actor_rollout_ref.actor.veomni.router_replay.mode="R2"`)

  * **Mechanism**: During the gradient update phase, it reproduces the expert paths computed by the training engine during the previous sampling phase.
  * **Purpose**: It mainly mitigates the impact of **policy staleness** on routing. As the policy updates, the routing computed during the current forward pass may be inconsistent with the routing used when generating the old data. R2 maintains the continuity of the optimization signal by replaying the old routing.
* **(2) Rollout Routing Replay (R3)**: (Corresponds to `actor_rollout_ref.actor.megatron.router_replay.mode="R3"`)

  * **Mechanism**: It captures the routing distribution of the inference engine during sequence generation and directly replays it to the training engine.
  * **Function**: It simultaneously resolves the two issues of **training-inference bias** and **policy staleness**. It ensures that the expert paths used for loss calculation during training strictly match the expert paths used during actual inference generation.

Therefore, the Routing Replay mechanism effectively bridges the routing gap between inference and training frameworks. This applies to both R2, which focuses on mitigating outdated strategies, and R3, which achieves full-link alignment. In the training-inference consistency alignment of **large-scale MoE models**, this mechanism has become a core method to ensure precision alignment and training stability. Currently, mainstream large language models such as DeepSeek-V3.2, GLM-5, and MiMo-V2 have all adopted the Routing Replay technology in R3 mode.

Therefore, for large MoE models, we generally recommend using the more thoroughly aligned R3 mode in actual configurations:

```
actor_rollout_ref.actor.megatron.router_replay.mode="R3" \
actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
```

For the VeOmni backend, use `actor_rollout_ref.actor.veomni.router_replay.mode="R3"` instead. The top-level `actor.router_replay` is removed and no longer takes effect.

## 4. Performance Optimization

When optimizing training performance for large model RL (reinforcement learning) on the Ascend NPU, refer to the official documentation for basic configuration tuning first: [perf_tuning.rst](https://github.com/verl-project/verl/blob/04833f01/docs/perf/perf_tuning.rst). For more efficient optimization, follow the standardized workflow of **data collection → bottleneck identification → configuration tuning → iterative verification**. This workflow significantly improves the throughput of core stages such as Rollout, Reward, and Update, while effectively reducing resource idling and load imbalance. For specific operations on performance analysis and tuning, refer to the following official guidelines:

1. [Ascend Performance Analysis Guide](../performance/ascend_performance_analysis_guide.md)
2. [Profiling Collection Guide](../performance/ascend_profiling.rst)

### 4.1 Inference Performance Optimization

The rollout phase is the core inference stage of large model RL training. Its inference time accounts for the vast majority of the entire training process. The following are common performance optimization techniques for this phase:

1. **Enable graph mode**: Graph mode compiles and optimizes the entire computation graph in advance. It can achieve deep optimizations such as operator fusion, memory reuse, and constant folding, significantly improving execution efficiency.
2. **Accelerate operator dispatch through CPU binding**: CPU binding can improve operator dispatch efficiency. Since vllm-ascend v0.18.0rc1, this capability is enabled by default on ARM architecture Ascend servers.
3. **Configure the HCCL communication algorithm to AIV mode**: Set the environment variable `HCCL_OP_EXPANSION_MODE` to `AIV` mode. This specifies that the orchestration and expansion logic of the communication algorithm runs on the Vector Core computing unit on the Device side.
4. **Enable asynchronous scheduling**: This eliminates the gap between two consecutive execute_model executions of the Worker. It allows the Worker to directly obtain the completed SchedulerOutput for model inference without blocking to wait for scheduling.

The corresponding configuration parameters are as follows:

# Enable graph mode
actor_rollout_ref.rollout.enforce_eager=False
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[2, 4, 8, 16, 24, 32]"
# CPU core binding
++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=True
# Enable asynchronous scheduling
++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=True

### 4.2 Training Performance Optimization

The Update phase of large language model (LLM) reinforcement learning (RL) training features significant variations in sequence length and high device memory consumption. In addition to basic operator fusion, you must combine sequence parallelism with device memory and computation trade-off strategies to break the bottleneck. For enabling common training performance optimization features, refer to the [MindSpeed-verl documentation](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/verl.md). Core optimization methods include:

1. ​**Operator fusion**​: Enable fused operators such as RoPE, SwiGLU, RMSNorm, and DSA. Operator fusion reduces computing overhead and device memory usage, improving training efficiency.
2. ​**Remove padding**​: In RL training, response lengths vary, and traditional padding strategies cause significant invalid computation. Enabling Remove padding packs multiple short sequences to fill tensors, significantly improving the utilization of NPU compute units (MFU).

## 5. Evaluation and Verification

After training is complete, evaluate and verify the target dataset to ensure that the business performance of the migrated model meets the standard. The evaluation steps are the same for different models. The following uses GLM-5 as an example to detail the evaluation process. This process uses the AISBenchmark tool and supports evaluating multiple inference backends such as vllm/sglang.

The evaluation uses the math dataset aime2025 and the graduate-level professional science dataset gpqa to verify that scores increase in the target direction and that catastrophic knowledge forgetting does not occur in unrelated directions.

### 5.1 Installing aisbench

```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
```

### 5.2 Download the evaluation dataset

```shell
# On the Linux server, in the root directory of the tool
cd path/to/benchmark/ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip
unzip aime2025.zip
rm aime2025.zip
```

### 5.3 Modifying AISBench Configuration Code to Enable vllm/sglang Inference Evaluation

Open the benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py file. This is the inference evaluation configuration file. We recommend keeping the output length `max_out_len` consistent with the training `max_response_len`.

```shell
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/path/to/GLM-5", # Change to the GLM-5 model path
        model="GLM-5",
	    stream=True,
        request_rate = 0,
	    use_timestamp=False,
        max_seq_len=2048,
        retry = 2,
	    api_key="",
        host_ip = "localhost", # IP address of the inference service
        host_port = 12890 , # Port of the inference service
        max_out_len = 8192,  # Maximum output token length
        batch_size=48, # Maximum concurrency for inference
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            seed = 1234,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

### 5.4 Starting the Inference Server on Multiple Machines

Refer to the [vllm_ascend GLM5 guide](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.md#multi-node-deployment) to start the dual-node A3 inference service. Keep `host_port` consistent with the configuration in the previous section, and set `max_model_len` to the sum of `max_prompt_length` and `max_response` used during training.

### 5.5 Start the vllm evaluation task

Run the following command to start the online inference evaluation task, call the deployed vLLM inference backend, and load the corresponding model configuration to complete the automated evaluation:

```
ais_bench --models vllm_api_stream_chat --datasets aime2025_gen_0_shot_chat_prompt
```

After the model is trained, the core capability metrics achieve stable improvement. The evaluation scores on the AIME2025 mathematical reasoning dataset increase steadily, and the model also achieves continuous score gains on the GPQA graduate-level professional science dataset. There are no knowledge degradation or catastrophic forgetting issues, and the training optimization results meet expectations.

| Evaluation dataset | GLM5-base | 10step | 15step | 40step | 50step |
| ---------- | --------- | ------ | ------ | ------ | ------ |
| aime2025   | 47.5      | 49.17  | 49.17  | 48.33  | 52.5   |
| gpqa       | 64.65     | 68.81  | 68.43  | 69.07  | 71.21  |

## 6. Summary

This document fully covers the end-to-end practice of migrating large models from GPUs to Ascend NPUs or independently adapting them on NPUs. The practice consists of five key phases: environment setup, component integration, precision alignment, performance optimization, and evaluation verification. It provides developers with actionable and reusable operation guides and solutions to problems.

In the preparation phase, focus on controlling environment dependency versions, model weight precision, and dataset format to lay the foundation for subsequent adaptation. For component integration, follow the principle of verifying single components first before integrating the entire network. Prioritize ensuring the stable adaptation of the inference engine, training engine, and weight conversion tools. Complete customized modifications for special model structures. Precision alignment is the core of migration adaptation. Monitor the training and inference consistency metrics closely. Resolve common differences in framework implementation and precision types through module-by-module troubleshooting. For MoE models, enable the Routing Replay mechanism to ensure training stability. Performance optimization must follow a standardized process. Focus on the core phases of inference and training. Improve efficiency and reduce resource consumption through methods such as graph mode and operator fusion. Finally, verify through standardized evaluation to ensure the model achieves expected business performance after migration without knowledge degradation.

Overall, following the process in this guide effectively reduces NPU migration and adaptation costs. It helps you avoid common pitfalls and run large models stably and efficiently on Ascend NPUs.
