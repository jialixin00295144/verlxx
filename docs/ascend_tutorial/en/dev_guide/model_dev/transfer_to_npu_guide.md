# Guide to Migrating Models to the NPU

Last updated: 05/14/2026

This article provides developers with complete practical experience for migrating from GPU to NPU or independently adapting models on NPU. It covers the full workflow of preliminary preparation, component integration, precision alignment, performance optimization, and long-running evaluation.

## 1. Preparation

Set up a basic runtime environment that supports NPU operation, ensuring normal model loading and smooth dataset reading, to serve as the foundation for subsequent migration debugging and business execution.


### 1.1 Software and Hardware Environment and Dependency Configuration

Refer to the official documentation [Ascend Installation Guide](../../get_start/install_guidance.rst). If the versions of the inference engines vllm and vllm_ascend, and the training engines Megatron, MindSpeed, and transformers that the model depends on differ from those in the tutorial, **use the versions actually adapted for the model**.

### 1.2 Model Weights

BF16 is the **default mixed precision training data type** for training backends such as FSDP and Megatron in the VeRL framework. The Ascend NPU environment uniformly uses **BF16** as the baseline precision format, and weights must be aligned and dequantized to BF16. Currently, A2 and A3 products **do not support FP8 precision training** and only support BF16 precision. Subsequent versions of the Ascend 950PR&950DT products will provide FP8 low-precision training capabilities.

### 1.3 Data Preparation

Preprocess the dataset into parquet format by referring to [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html): (1) ensure it contains the necessary fields required for computing reinforcement learning rewards; (2) the reading speed is faster.

## 2. Integration and debugging of all components

The VeRL framework adopts a decoupled architecture design for the inference engine, training engine, and weight synchronization bridge (Checkpoint Engine). This design achieves a deep separation of computation and data, providing a flexible extension foundation for migrating and adapting models to the Ascend NPU. When migrating and adapting models to the NPU, first complete the separate adaptation and verification of the inference engine, training engine, and Megatron-Bridge components. After these components run stably, proceed to connect and debug the entire VeRL network link. For specific feature support of different VeRL inference and training backends on the Ascend NPU, refer to the [Ascend Backend Features Guide](../../feature_support/ascend_backend_features.md).

### 2.1 Inference Engine Adaptation

The VeRL inference engine adopts a layered architecture design and provides flexible support for multiple mainstream inference backends such as vllm and sglang through abstract interfaces and the factory pattern. During the migration and adaptation from GPU to NPU, follow the procedure below to adapt the inference engine:

Before running the full VeRL pipeline on the NPU, refer to the official model deployment tutorials for [vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/main/docs/source/tutorials/models) and [sglang](https://github.com/sgl-project/sglang/tree/main/docs/docs/basic_usage). Prioritize getting the **single-instance inference pipeline** working. Fully verify **basic inference functions** such as model loading and initialization, normal Tokenizer loading, single-turn and batch generation, stop word termination, and long-context inference. After the underlying inference engine is stable and available, integrate it into the VeRL training workflow.

### 2.2 Training Engine Selection and Adaptation

The VeRL mainline code abstracts the training engine into the `Engine` class. It decouples the scheduling logic from the underlying training implementation through a standardized interface layer. This architecture supports the flexible, plug-and-play integration of multiple training backends such as FSDP and Megatron. It significantly reduces migration and adaptation costs without modifying the core algorithms and scheduling logic of VeRL.

The NPU has automatically detected the device through the `is_npu_available` interface and applied the corresponding NPU device adaptation patch. You only need to configure model_engine=fsdp/megatron to switch the training backend to FSDP or Megatron with one click. The system automatically loads the NPU adaptation logic for the corresponding backend, requiring no additional code modifications. VeRL provides adaptation and optimization for Megatron on the Ascend NPU. For specific feature configurations, refer to the [verl-MindSpeed feature document](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/verl.md).

### 2.3 Megatron-Bridge Adaptation

Megatron-Bridge is primarily used in the VeRL framework to perform bidirectional conversion between the HuggingFace weights required by the inference engine and the mcore weights required by Megatron-Core. You can enable this feature through the following configuration:

```
actor_rollout_ref.actor.megatron.use_mbridge=True
```

Megatron-Bridge natively supports a large number of mainstream model structures in the community. For the supported list, refer to [supported model](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/README.md). When performing model migration and adaptation in an Ascend NPU environment, you can complete the basic configuration based on existing community capabilities. However, some special model structures and scenarios still require additional customized adaptation.

Using the DSA (DeepSeek Sparse Attention) sparse attention structure as an example, this section introduces the method for customized adaptation. Ascend MindSpeed supports the DSA capability based on the absorption matrix. This feature requires splitting the original `linear_kv_up_proj` operator in Megatron into two independent operators: `linear_k_up_proj` and `linear_v_up_proj`. The weights required for splitting must be generated by converting the HuggingFace format `self_attn.kv_b_proj.weight`. However, the preceding native PR does not support this operator splitting logic.

Therefore, you must manually modify and adapt the related weight conversion logic to ensure that the absorption matrix can be properly loaded and take effect. Only when the absorption matrix is available can you enable the [sparse\_flash\_attention](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_attention) and [lightning\_indexer](https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer) fusion operators. By introducing these two fusion operators, you can significantly reduce memory access frequency and optimize memory usage. This also improves compute performance, ultimately enhancing the running efficiency of large model training and inference pipelines and reducing resource overhead.

### 2.4 Full Network Function Integration

Complete the inference engine adaptation verification and training engine adaptation development. Refer to [Training Configuration Parameters and Metrics Description](parameter_and_metrics.md) to configure the relevant parameters for the inference engine and training engine based on actual business requirements. Complete the VeRL full-network function integration to ensure stable operation of the entire process.

## 3. Precision Alignment

The localization pipeline for precision issues in large language model reinforcement learning is complex and involves numerous factors. Various precision issues are typically introduced by problems in the **training phase, inference phase, and training-inference consistency**. **Precision alignment** is critical for ensuring the reproducibility of the training process and the debuggability of issues.

For precision alignment during the training and inference phases, refer to the official documentation: [Precision Alignment Guide](../precision_analysis/precision_alignment.md). Therefore, this section does not reiterate the basic alignment process. Instead, it **focuses on training and inference consistency scenarios** to implement precision alignment and troubleshoot issues using the msprobe precision tool.

### 3.1 Precision Monitoring Configuration

After the entire network runs successfully, enable the precision monitoring parameter by setting `actor_rollout_ref.rollout.calculate_log_probs=True`. During training, closely monitor the following key metrics to determine the training-inference consistency and model training stability:

* **Training and inference consistency reference metrics**:
  * `training/rollout_probs_diff_mean` (mean rollout probability difference): When the model converges normally, keep this metric within 0.01. If the value consistently exceeds 0.01 or deviates significantly from the GPU baseline, you can determine that a training and inference precision anomaly exists.
  * `training/rollout_probs_diff_max` (maximum rollout probability difference)
  * `training/rollout_actor_probs_pearson_corr` (Pearson correlation coefficient between rollout and actor probabilities)
* **Model training stability metrics**:
  * `actor/grad_norm`: Check whether it shows an overall downward trend to determine whether the model training is converging normally.

In addition, the configuration parameter `trainer.rollout_data_dir=./rollout_dump/` saves intermediate Rollout results during training. Manually check the exported Rollout data to verify whether the model responses meet expectations. Ensure the output does not contain garbled text or repeated answers. This further confirms that the inference engine adaptation is correct.

### 3.2 Collecting precision data

When training/rollout_probs_diff_mean exceeds the reasonable threshold of 0.01 or significantly deviates from the GPU baseline, use the [msprobe](../../../en/dev_guide/precision_analysis/precision_debugger.md) precision tool to collect data for root cause analysis.

### 3.3 Troubleshooting and Alignment Practices for Training and Inference Differences

After data collection is complete, read the `construction.json` file first to perform module-level data comparison. First ensure that the input data of `layer.0.input_layernorm` is completely consistent, and then verify module by module and layer by layer to locate where the training and inference outputs first become inconsistent.

For large models, minor numerical differences accumulate and amplify layer by layer. This causes significant differences between the training and inference (rollout) results. The output probability of the same token during training and inference can even be 0 and 1, respectively. Therefore, align every difference point to be exactly equal as much as possible.

After you locate the difference nodes, determining the adaptation and modification scheme is also a key challenge. Various open-source communities in the industry have multiple different implementations for related modules. To ensure the correctness of the model implementation logic, refer to authoritative source code and technical reports from multiple sources, and comprehensively determine the final alignment scheme.

#### 3.3.1 Common training and inference inconsistencies

In large language model reinforcement learning practice, you can summarize the typical root causes of training-inference inconsistency into the following five categories:

1. **Inconsistent framework implementations**: Different implementation logic between the training and inference frameworks causes this issue. Sometimes the inconsistency is "semantically correct" (for example, different operator splitting methods that are mathematically equivalent). Sometimes it is "semantically incorrect" (for example, a missing scaling factor or an extra operation). You need to strictly verify this by combining the source code and technical reports.
2. **Precision type differences**: For example, the training side uses BF16 throughout. The inference side implicitly upcasts to FP32 for computation in sensitive operators such as normalization and then downcasts, causing truncation errors.
3. **Inconsistent hyperparameters**: For example, the hardcoded `eps` value in the LayerNorm module is not unified.
4. **Parallelism strategies**: Tensor parallelism during training versus continuous batching during inference causes differences in floating-point accumulation order.
5. **Randomness control**: Implementation deviations exist in Dropout and sampling strategies between the training and inference stages.

The following lists typical real-world cases of training and inference inconsistencies encountered during the migration and adaptation of the GLM-5 model.

#### 3.3.2 Case 1: Inconsistent framework implementation of the FFN activation function

Compare sequentially from top to bottom, and identify that the output is inconsistent at the MLP activation function of the first layer.

The inference side uses the NPU-optimized `npu_swiglu` fused operator, but the training side still executes the native small GLU operator implementation.

* **Root cause**: Although the `swiglu` enable configuration was added to the VeRL parameters, Megatron-Bridge did not explicitly set `provider.bias_activation_fusion=True` in the NPU adaptation PR. As a result, the code did not enter the NPU fused operator branch.
* **Fix**: Add configuration items in Megatron-Bridge so that the training side correctly invokes the fused operator:
  ```
  +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
  ```

#### 3.3.3 Case 2: Inconsistency between the precision and hyperparameters of indexer_k_norm

During the strict alignment process, an inconsistency between the precision type and hyperparameters is found at `indexer_k_norm`:

* **Precision difference**: On the inference side, the LayerNorm operation increases precision to fp32 using `F.layer_norm( x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)`. On the training side, Megatron implements this using BF16. Minor differences accumulate across multiple layers and cannot be ignored.
* **Fix solution**: Update the training side code to add the precision increase and decrease operations.
* **Hyperparameter difference**: The GLM5 inference side in vllm inherits the DeepSeek-V3.2 logic, where the EPS value of `k_norm` is hardcoded to `1e-6`. The training engine and the official technical report both use `1e-5`.
* **Fix solution**: Change the inference side EPS to `1e-5` to align with the training side.

```
self.k_norm=LayerNorm(self.head_dim,eps=1e-6 -> 1e-5)
```

#### 3.3.4 Case 3: Missing and redundant logic in the lightning_indexer module

Troubleshooting revealed that the implementation of the lightning_indexer module is inconsistent between the training and inference sides, specifically as follows:

* **Missing (omitted on the inference side)**: The inference side is missing the scaling logic for `weights`. The standard implementations in Megatron training, slime, and transformers all include this scaling. Therefore, add it on the inference side to align the forward pass:

```
weights, _ = self.weights_proj(x)
+weights = weights * (self.n_head**-0.5) * (self.head_dim**-0.5)
```

* **Redundancy (unnecessary and incorrect implementation on the training side)**: The Megatron implementation on the training side includes an extra `rotate_activation` (Hadamard transform). Extensive research confirms that this operation is specifically for quantization scenarios and is incorrectly implemented for the BF16 format. Remove this redundant logic from the training side based on [Transformer PR#45017](https://github.com/huggingface/transformers/pull/45017).

```
class DSAIndexer(MegatronModule):
    def forward_with_scores(
-		q = rotate_activation(q)
-		k = rotate_activation(k)
```

### 3.4 General routing stability solution for large MoE models

In a typical reinforcement learning (RL) training workflow, an efficient inference engine such as vLLM performs sample sampling, and the sampled data is then sent to a training framework such as Megatron for model training optimization.

For regular dense models, implementation and environment differences between inference and training frameworks only cause minor numerical deviations. However, for **large-scale MoE models**, this problem is drastically amplified. The root cause is the MoE dynamic routing mechanism. Minor differences in framework implementation and the running environment can route the same input token to completely different expert combinations. This leads to entirely different computation paths.

This inconsistency in routing decisions can cause the RL training of MoE models to become unstable. It makes the experience obtained during the inference phase completely different for the training phase. This distorts the optimization signals and ultimately leads to catastrophic consequences.

To address this common problem, the industry has introduced the **Routing Replay** mechanism. The core idea is to lock the expert routing paths at specific stages. This prevents minor perturbations from interfering with routing decisions, thereby ensuring stable model training. Currently, there are two mainstream variants: R2 and R3:

* **(1) Vanilla Routing Replay (R2)**: (This corresponds to `actor_rollout_ref.actor.megatron.router_replay.mode="R2"`, while for VeOmni, it is `actor_rollout_ref.actor.veomni.router_replay.mode="R2"`)

  * **Mechanism**: During the gradient update phase, reproduces the expert paths calculated by the training engine in the previous sampling phase.
  * **Purpose**: Mainly mitigates the impact of **policy staleness** on routing. As the policy updates, the routing calculated during the current forward pass may be inconsistent with the routing used when generating old data. R2 maintains the continuity of the optimization signal by replaying the old routing.
* **(2) Rollout Routing Replay (R3)**: (Corresponds to `actor_rollout_ref.actor.megatron.router_replay.mode="R3"`)

  * **Mechanism**: Captures the routing distribution of the inference engine during sequence generation and directly replays it in the training engine.
  * **Effect**: Simultaneously resolves the two issues of **training-inference bias** and **policy staleness**. It ensures that the expert path used to calculate the loss during training exactly matches the expert path used during actual inference generation.

Therefore, whether mitigating strategy staleness in R2 or achieving full-link alignment in R3, the Routing Replay mechanism effectively bridges the routing gap between inference and training frameworks. In the training-inference consistency alignment of **large-scale MoE models**, this mechanism is a core method for ensuring precision alignment and training stability. Currently, mainstream large models such as DeepSeek-V3.2, GLM-5, and MiMo-V2 adopt the Routing Replay technology in R3 mode.

Therefore, for large-scale MoE models, the more thoroughly aligned R3 mode is generally recommended in actual configurations:

```
actor_rollout_ref.actor.megatron.router_replay.mode="R3" \
actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
```

For the VeOmni backend, use `actor_rollout_ref.actor.veomni.router_replay.mode="R3"` instead. The top-level `actor.router_replay` is removed and no longer takes effect.

## 4. Performance optimization

When optimizing training performance for large language model (LLM) reinforcement learning (RL) on the Ascend NPU, refer to the official documentation for basic configuration tuning: [perf_tuning.rst](https://github.com/verl-project/verl/blob/04833f01/docs/perf/perf_tuning.rst). For more efficient optimization, follow the standardized workflow of **data collection → bottleneck identification → configuration tuning → iterative validation**. This workflow significantly improves the throughput of core stages such as Rollout, Reward, and Update, and effectively reduces resource idling and load imbalance. For specific performance analysis and tuning operations, refer to the following official guidelines:

1. [Ascend Performance Analysis Guide](../performance/ascend_performance_analysis_guide.md)
2. [Profiling Collection Guide](../performance/ascend_profiling.rst)

### 4.1 Inference Performance Optimization

The rollout phase is the core inference stage of large language model (LLM) reinforcement learning (RL) training. Its inference time consumption accounts for the majority of the entire training process. The following are common performance optimization methods for this phase:

1. **Enable the graph mode feature**: Graph mode compiles and optimizes the entire computation graph in advance. It achieves deep optimizations such as operator fusion, memory reuse, and constant folding, significantly improving execution efficiency.
2. **Accelerate operator dispatch by binding CPU cores**: Binding CPU cores improves operator dispatch efficiency. Since vllm-ascend v0.18.0rc1, this capability is enabled by default on ARM architecture Ascend servers.
3. **Configure the HCCL communication algorithm to AIV mode**: Set the environment variable `HCCL_OP_EXPANSION_MODE` to `AIV` mode. This specifies that the orchestration and expansion logic of the communication algorithm runs on the Vector Core computing unit on the device side.
4. **Enable asynchronous scheduling**: This eliminates the gap between two consecutive `execute_model` executions by the worker. The worker can directly obtain the completed SchedulerOutput for model inference, without blocking to wait for scheduling.

The corresponding configuration parameters are as follows:

```
# Graph mode enabled
actor_rollout_ref.rollout.enforce_eager=False
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[2, 4, 8, 16, 24, 32]"
# CPU core binding
++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=True
# Asynchronous scheduling enabled
++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=True
```

### 4.2 Training Performance Optimization

The Update phase of large language model (LLM) reinforcement learning (RL) training features large variance in sequence lengths and high device memory consumption. In addition to basic operator fusion, you need to combine sequence parallelism with device memory and computation trade-off strategies to break through bottlenecks. For common training performance optimization features, refer to the [MindSpeed-verl documentation](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/verl.md) to enable them. Core optimization methods include:

1. ​**Operator fusion**​: Enable fused operators such as RoPE, SwiGLU, RMSNorm, and DSA. Operator fusion reduces computing overhead and device memory usage, improving training efficiency.
2. ​**Remove padding**​: In RL training, response lengths vary, and traditional padding strategies cause significant invalid computation. Enabling Remove padding packs multiple short sequences to fill the Tensor, greatly improving the utilization of NPU compute units (MFU).

## 5. Evaluation and Verification

After training is complete, evaluate and validate the target dataset to ensure that the business performance after model migration meets the requirements. The evaluation steps are the same for different models. The following uses GLM-5 as an example to describe the evaluation process in detail (using the AISBenchmark tool, which supports the evaluation of multiple inference backends such as vllm/sglang).

The evaluation uses the mathematical dataset aime2025 and the graduate-level professional science dataset gpqa. This verifies that scores increase in the target direction and that catastrophic forgetting does not occur in irrelevant directions.

### 5.1 Installing aisbench

```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
```

### 5.2 Download the Evaluation Dataset

```shell
# On the Linux server, in the root directory of the tool
cd path/to/benchmark/ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip
unzip aime2025.zip
rm aime2025.zip
```

### 5.3 Modifying AISBench Configuration Code to Enable vllm/sglang Inference Evaluation

Open the benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py file. This is the inference evaluation configuration file. Keep the output length `max_out_len` consistent with the training `max_response_len`.

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

Refer to the [vllm_ascend GLM5 guide](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.md) to launch the dual-node A3 inference service. Keep `host_port` consistent with the configuration in the previous section, and set `max_model_len` to the sum of `max_prompt_length` and `max_response` used during training.

### 5.5 Start the vLLM evaluation task

Run the following command to start the online inference evaluation task. This command calls the deployed vLLM inference backend and loads the corresponding model configuration to complete the automated evaluation:

```
ais_bench --models vllm_api_stream_chat --datasets aime2025_gen_0_shot_chat_prompt
```

After you train the model, its core capability metrics achieve stable improvement. Evaluation scores on the AIME2025 mathematical reasoning dataset steadily increase. The model also achieves continuous score gains on the GPQA graduate-level professional science dataset. The model shows no knowledge degradation or catastrophic forgetting, and the training optimization effect meets expectations.

| Evaluation dataset | GLM5-base | 10step | 15step | 40step | 50step |
| ---------- | --------- | ------ | ------ | ------ | ------ |
| aime2025   | 47.5      | 49.17  | 49.17  | 48.33  | 52.5   |
| gpqa       | 64.65     | 68.81  | 68.43  | 69.07  | 71.21  |

## 6. Summary

This document fully covers the end-to-end practice of migrating large models from GPUs to Ascend NPUs or independently adapting them on NPUs. It consists of five key phases: environment setup, component integration, precision alignment, performance optimization, and evaluation verification. It provides developers with actionable and reusable operation guides and problem solutions.

In the preparation phase, focus on controlling environment dependency versions, model weight precision, and dataset format to lay the foundation for subsequent adaptation. During component integration, follow the principle of verifying single components first before integrating the entire network. Prioritize ensuring the stable adaptation of the inference engine, training engine, and weight conversion tool. Complete customized modifications for special model structures. Precision alignment is the core of migration adaptation. Focus on monitoring training-inference consistency metrics. Resolve common differences such as framework implementation and precision types through module-by-module troubleshooting. For MoE models, enable the Routing Replay mechanism to ensure training stability. Performance optimization must follow a standardized process. Focus on the core phases of inference and training. Improve efficiency and reduce resource consumption through graph mode, operator fusion, and other methods. Finally, verify through standardized evaluation to ensure that the model meets business performance requirements after migration and has no knowledge degradation.

Overall, following the process in this guide effectively reduces NPU migration and adaptation costs, avoids common pitfalls, and enables large models to run stably and efficiently on Ascend NPU.
