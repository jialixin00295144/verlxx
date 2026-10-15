# NPU Advanced Features Guide

> This document introduces the advanced features and optimization capabilities of the Ascend NPU in the verl ecosystem to serve as a reference for developers.
>
> Last updated: 05/13/2026.

---

## Table of Contents

- [NPU Advanced Features Guide](#npu-高级特性指南)
  - [Table of Contents](#目录)
  - [1. Inference Backend Advanced Features](#1-推理后端高级特性)
    - [1.1 vLLM Inference Backend](#11-vllm-推理后端)
    - [1.2 SGLang Inference Backend](#12-sglang-推理后端)
      - [Advanced Parameter Configuration](#高级参数配置)
  - [2. Training Backend Advanced Features](#2-训练后端高级特性)
    - [2.1 FSDP Training Backend](#21-fsdp-训练后端)
    - [2.2 Megatron Training Backend](#22-megatron-训练后端)
      - [MindSpeed Monkey Patch Framework Principles](#mindspeed-monkey-patch-框架原理)
      - [Megatron Advanced Parameter Configuration](#megatron-高级参数配置)
        - [Memory and Computation Optimization](#内存与计算优化)
        - [Fusion Operator Acceleration](#融合算子加速)
        - [Pipeline Parallelism Optimization](#流水线并行优化)
        - [Weight Management](#权重管理)
  - [3. Performance Optimization Features](#3-性能优化特性)
    - [3.1 Memory Optimization](#31-内存优化)
    - [3.2 Computation Acceleration](#32-计算加速)
    - [3.3 Parallelism Strategies](#33-并行策略)
  - [4. Mixture of Experts (MoE) Features](#4-混合专家模型-moe-特性)
    - [vLLM/SGLang Inference MoE Support](#vllmsglang-推理-moe-支持)
    - [Megatron Training MoE Support](#megatron-训练-moe-支持)
  - [5. Limitations and Precautions](#5-限制与注意事项)
  - [Appendix: Parameter Quick Reference](#附录参数速查表)
    - [Inference Backend Parameter Quick Reference](#推理后端参数速查)
    - [Training Backend Parameter Quick Reference](#训练后端参数速查)

---

## 1. Inference Backend Advanced Features

Currently, verl supports two mainstream inference backends: vLLM and SGLang. Both run on Ascend NPUs. The following lists the advanced feature parameters supported by each backend.

### 1.1 vLLM inference backend

Ascend supports the vLLM inference backend through the **vllm-ascend plugin**. This plugin follows the [RFC](https://github.com/vllm-project/vllm/issues/11162) and provides a pluggable interface to decouple the Ascend NPU from vLLM.

---

### 1.2 SGLang Inference Backend

Ascend NPU supports related features through continuous development and maintenance of the SGLang community, involving the following core components:

| Component | Description |
|:---|:---|
| [sgl_kernel_npu](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md) | A collection of optimized inference kernels for Ascend NPU, including attention mechanisms, normalization, activation functions, LoRA adapters, and so on |
| [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md) | The Ascend implementation of DeepEP, providing highly optimized expert parallelism (EP) communication kernels for MoE models |

#### Advanced Parameter Configuration

| SGLang parameter | Corresponding verl general parameter | Description |
|:---|:---|:---|
| `attention_backend` | `actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend` | **Attention backend selection** — Set to `ascend` on the NPU to invoke the Ascend optimized kernel |
| `quantization` | `actor_rollout_ref.rollout.quantization` | **Quantization support** — Supports quantized model loading and inference |


> For more SGLang NPU feature parameters, refer to the [sglang community NPU feature support documentation](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu_support_features).

---

## 2. Advanced training backend features

### 2.1 FSDP training backend

Ascend provides FSDP support capabilities through `torch_npu`.

### 2.2 Megatron training backend

Megatron is a model parallelism training framework introduced by NVIDIA. To run on an NPU, you must also install **MindSpeed** to provide underlying support. MindSpeed uses the **Monkey Patch** technology to transparently replace key Megatron components, achieving NPU adaptation.

#### MindSpeed Monkey Patch Framework Principles

**Trigger entry point:**
```python
from mindspeed.megatron_adaptor import repatch
```

**Call chain:**
```
repatch
├── Execute the megatron_adaptor.py module import
├── Import the features_manager module
├── Execute mindspeed/features_manager/__init__.py
├── Trigger the @AutoExecuteFunction decorator
├── patch_features() executes automatically
└── Perform the apply_features_pre_patches and apply_features_patches operations
```

**Core components:**

| Component | Responsibility |
|:---|:---|
| The `Patch` class | Implements dynamic replacement of functions or classes, and supports stacking multiple layers of decorators. |
| `parse_path()` | Imports and creates modules dynamically. |
| `MindSpeedPatchesManager` | Manages all patch registrations as a global singleton. |
| `MindSpeedFeature` | Serves as the base class for features, integrating the patch system through inheritance. |

#### Advanced Megatron parameter configuration

##### Memory and Compute Optimization

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs` | **Pipeline output deallocation** — Releases output data after tensors are sent to the next PP stage, reducing the device memory peak. The default value is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity` | **Recomputation granularity control** — Optional values are `full`, `selective`, and `none`. `full` recomputes the entire Transformer layer, and `selective` recomputes only the core attention part. The default value is `none`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method` | **Recomputation method** — Requires `recompute_granularity=full`. Optional values are `uniform` and `block`. The default value is `None`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers` | **Number of recomputation layers** — Requires `recompute_granularity=full`. A larger value reduces device memory usage but increases computation cost. The value must be divisible by the number of model layers in the current process. |

##### Fused Operator Acceleration

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn` | **Flash Attention** — Specifies whether to use Flash Attention to accelerate attention computation. The default value is `true`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb` | **Fused Rotary Position Embedding** — Uses fused operators to accelerate RoPE computation. The default value is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu` | **Fused SwiGLU** — Uses fused operators to accelerate the SwiGLU activation function. The default value is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm` | **Persistent LayerNorm** — Uses a persistent strategy to optimize LayerNorm. The default value is `False`. |

##### Pipeline Parallelism Optimization

| verl Parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split` | **Loss layer pipeline split** — Treats the loss layer as a standard Transformer layer for pipeline splitting, defaults to `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split` | **Embedding layer pipeline split** — Treats the input embedding layer as a standard Transformer layer for pipeline splitting, defaults to `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage` | **First stage layer count** — Specifies the number of layers in the first pipeline stage, defaults to `none` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage` | **Last stage layer count** — Specifies the number of layers in the last pipeline stage, defaults to `none` |

##### Weight Management

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.use_mbridge` | **Megatron-Bridge weight conversion** — currently must remain `True` |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | **Distributed checkpoint** — saves/loads weights using a distributed format, defaults to `False` |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | **Distributed weight path** — distributed checkpoint loading path, defaults to `null` |

---

## 3. Performance Optimization Features

### 3.1 Memory Optimization

| Feature | Inference/Training | Description |
|:---|:---|:---|
| KV Cache Dynamic Release (`free_cache_engine`) | Inference (vLLM) | Automatically offloads the KV Cache after the generation phase; enabled by default |
| Memory Saver Mode (`enable_memory_saver`) | Inference (SGLang) | Supports dynamic device memory release and restoration; verl defaults to `True` |
| Parameter CPU Offload (`param_offload`) | Training (FSDP/Megatron) | Offloads model weights to the CPU |
| Optimizer CPU Offload (`optimizer_offload`) | Training (FSDP/Megatron) | Offloads optimizer states to the CPU |
| Chunked Entropy Computation (`entropy_from_logits_with_chunking`) | Training (FSDP) | Computes entropy in chunks to reduce peak device memory |
| Entropy Computation Chunk Size (`entropy_from_logits_chunk_size`) | Training (FSDP) | The chunk size for entropy computation |
| Entropy Computation Recomputation (`entropy_checkpointing`) | Training (FSDP) | Enables recomputation for entropy computation |
| Pipeline Output Deallocation (`deallocate_pipeline_outputs`) | Training (Megatron) | Releases passed tensors in pipeline parallelism scenarios |
| Activation Recomputation (`recompute_granularity`) | Training (Megatron) | Supports three levels of granularity control: full, selective, and none |

### 3.2 Compute Acceleration

| Feature | Inference/Training | Description |
|:---|:---|:---|
| Chunked prefill (`enable_chunked_prefill`) | Inference (vLLM) | Splits large prefill into chunks and processes them with the decoding batch |
| Prefix caching (`enable_prefix_caching`) | Inference (vLLM) | Automatically caches shared prefixes to reduce redundant computation |
| Flash Attention | Training (Megatron) | Uses Flash Attention to accelerate attention computation; enabled by default |
| Fused rotary position embedding (`use_fused_rotary_pos_emb`) | Training (Megatron) | Uses fused operators to accelerate RoPE |
| Fused SwiGLU (`use_fused_swiglu`) | Training (Megatron) | Uses fused operators to accelerate the SwiGLU activation function |
| Persistent LayerNorm (`persist_layer_norm`) | Training (Megatron) | Optimizes the LayerNorm execution strategy |
| Group GEMM (`moe_grouped_gemm`) | Training (Megatron) | Optimizes Group GEMM for MoE scenarios |

### 3.3 Parallelism Strategy

| Parallelism type | vLLM | SGLang | FSDP | Megatron | Description |
|:---|:---|:---|:---|:---|:---|
| Data parallelism (DP) | ✅ | ✅ | ✅ | ✅ | Data dimension parallelism |
| Tensor parallelism (TP) | ✅ | ✅ | — | ✅ | Intra-layer tensor splitting |
| Pipeline parallelism (PP) | — | — | — | ✅ | Inter-layer pipeline splitting |
| Expert parallelism (EP) | ✅ | ✅ | — | ✅ | MoE expert dimension parallelism |
| Sequence parallelism (SP/Ulysses) | ✅ | ✅ | ✅ | ✅ | Splits the sequence dimension to support long sequences |
| Context parallelism (CP) | ✅ | — | — | ✅ | Context parallel processing |

---

## 4. Mixture of Experts (MoE) features

### vLLM/SGLang Inference MoE Support

- **Expert Parallelism (EP)** — Configure the `ep_size` parameter to assign different experts to different NPU devices.
- SGLang provides highly optimized EP communication kernels using [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md).

### Megatron Training MoE Support

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | Expert parallelism (EP) size. The default is `1`. |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` | TP extended EP size. The default is `null`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm` | **Group GEMM** — Uses Group GEMM to optimize expert computation in MoE scenarios. The default is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype` | **Routing data type** — The data type for routing and the weighted average of expert outputs. You can select `fp32` or `fp64`. The default is `fp32`, which improves stability in multi-expert scenarios. |

---

## 5. Limitations and Precautions

1. **Differences between FSDP1 and FSDP2**
   - `forward_prefetch` and `use_orig_params` only apply to FSDP1.
   - FSDP2 is the default recommended version. For API support, refer to the [Ascend PyTorch version description](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-7-1/torch-distributed-fsdp.md).

2. **Recompute parameter dependencies**
   - The `recompute_method` parameter takes effect only when `recompute_granularity='full'`.
   - The `recompute_num_layers` parameter takes effect only when `recompute_granularity='full'`.
   - When `recompute_method='uniform'`, the `recompute_num_layers` parameter specifies the Transformer layers in each recompute unit and must be divisible by the current process model layers.

4. **SGLang NPU-specific configuration**
   - `attention_backend` must be set to `ascend` to call the optimized Ascend kernel
   - `enable_memory_saver` is enabled by default in verl, requiring no additional configuration

---

## Appendix: Parameter Quick Reference

### Inference Backend Parameters Quick Reference

| Parameter category | vLLM parameter | SGLang parameter | verl general parameter |
|:---|:---|:---|:---|
| Model path | `model_path` | `model_path` | `actor_rollout_ref.model.path` |
| Device memory control | `gpu_memory_utilization` | `mem_fraction_static` | `actor_rollout_ref.rollout.gpu_memory_utilization` |
| Graph mode | `enforce_eager` | `disable_cuda_graph` | `actor_rollout_ref.rollout.enforce_eager` |
| Quantization | `quantization` | `quantization` | `actor_rollout_ref.rollout.quantization` |
| Maximum sequence length | `max_model_len` | — | `actor_rollout_ref.rollout.max_model_len` |
| Maximum concurrency | `max_num_seqs` | `max_running_requests` | `actor_rollout_ref.rollout.max_num_seqs` |
| Tokenizer | `skip_tokenizer_init` | `skip_tokenizer_init` | `actor_rollout_ref.rollout.skip_tokenizer_init` |
| Remote code | `trust_remote_code` | `trust_remote_code` | `actor_rollout_ref.model.trust_remote_code` |
| TP parallelism | `tp_size` | `tp_size` | `actor_rollout_ref.rollout.tensor_model_parallel_size` |
| DP parallelism | `dp_size` | `dp_size` | `actor_rollout_ref.rollout.data_parallel_size` |
| EP parallelism | `ep_size` | `ep_size` | `actor_rollout_ref.rollout.expert_parallel_size` |

### Training Backend Parameter Quick Reference

| Parameter category | FSDP parameter | Megatron parameter |
|:---|:---|:---|
| Parameter offload | `fsdp_config.param_offload` | `megatron.param_offload` |
| Optimizer offload | `fsdp_config.optimizer_offload` | `megatron.optimizer_offload` |
| Sequence parallelism | `ulysses_sequence_parallel_size` | `context_parallel_size` |
| Flash Attention | — | `override_transformer_config.use_flash_attn` |
| Recomputation granularity | — | `override_transformer_config.recompute_granularity` |
| Distributed checkpoint | — | `use_dist_checkpointing` |
