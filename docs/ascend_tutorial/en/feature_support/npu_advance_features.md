---
myst:
  heading_anchors: 6
---

# NPU Advanced Features Guide

> This document introduces the advanced features and optimization capabilities of the Ascend NPU in the verl ecosystem for developer reference.
>
Last updated: 05/13/2026.

---

## Contents

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
        - [Fused Operator Acceleration](#融合算子加速)
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

## 1. Inference backend advanced features

Currently, verl supports two mainstream inference backends: vLLM and SGLang. Both can run on Ascend NPUs. The following lists the advanced feature parameters supported by each backend.

### 1.1 vLLM Inference Backend

Ascend supports the vLLM inference backend through the **vllm-ascend plugin**. This plugin follows the [RFC](https://github.com/vllm-project/vllm/issues/11162) and provides a pluggable interface to decouple the Ascend NPU from vLLM.

---

### 1.2 SGLang inference backend

Ascend supports related features through continuous development and maintenance in the SGLang community, involving the following core components:

| Component | Description |
|:---|:---|
| [sgl_kernel_npu](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md) | A collection of optimized inference kernels for Ascend NPUs, including attention mechanisms, normalization, activation functions, LoRA adapters, and so on |
| [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md) | The Ascend implementation of DeepEP, providing highly optimized expert parallelism (EP) communication kernels for MoE models |

#### Advanced Parameter Configuration

| SGLang parameter | Corresponding verl general parameter | Description |
|:---|:---|:---|
| `attention_backend` | `actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend` | **Attention backend selection** — On the NPU, set this to `ascend` to invoke the Ascend optimized kernel |
| `quantization` | `actor_rollout_ref.rollout.engine_kwargs.sglang.quantization` | **Quantization support** — Supports loading and inference of quantized models |


> For more SGLang NPU feature parameters, refer to the [sglang community NPU feature support documentation](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu_support_features).

---

## 2. Advanced Features of the Training Backend

### 2.1 FSDP training backend

Ascend provides FSDP support capabilities through `torch_npu`.

### 2.2 Megatron training backend

Megatron is a model parallelism training framework introduced by NVIDIA. To run Megatron on an NPU, you need to additionally install **MindSpeed** to provide underlying support. MindSpeed uses the **Monkey Patch** technology to seamlessly replace key Megatron components to achieve NPU adaptation.

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
├── The @AutoExecuteFunction decorator is triggered
├── patch_features() is automatically executed
└── Perform the apply_features_pre_patches and apply_features_patches operations
```

**Core components:**

| Component | Responsibility |
|:---|:---|
| `Patch` class | Implements dynamic replacement of functions and classes, and supports multi-layer decorator stacking |
| `parse_path()` | Dynamically imports and creates modules |
| `MindSpeedPatchesManager` | A global singleton that manages all patch registrations |
| `MindSpeedFeature` | The base feature class; each feature integrates the patch system through inheritance |

#### Megatron Advanced Parameter Configuration

##### Memory and Computation Optimization

| verl Parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs` | **Pipeline output deallocation** — Releases output data after the tensor is sent to the next PP stage, reducing the device memory peak. Default: `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity` | **Recomputation granularity control** — Options: `full` / `selective` / `none`. `full` recomputes the entire Transformer layer, and `selective` recomputes only the core attention part. Default: `none` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method` | **Recomputation method** — Requires `recompute_granularity=full`. Options: `uniform` / `block`. Default: `None` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers` | **Number of recomputation layers** — Requires `recompute_granularity=full`. A larger value reduces device memory usage but increases the computation cost. The value must be divisible by the number of model layers in the current process. |

##### Fused Operator Acceleration

| verl Parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn` | **Flash Attention** — Specifies whether to use Flash Attention to accelerate attention computation. The default value is `true`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb` | **Fused Rotary Position Embedding** — Uses a fused operator to accelerate RoPE computation. The default value is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu` | **Fused SwiGLU** — Uses a fused operator to accelerate the SwiGLU activation function. The default value is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm` | **Persistent LayerNorm** — Uses a persistent strategy to optimize LayerNorm. The default value is `False`. |

##### Pipeline Parallelism Optimization

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split` | **Loss layer pipeline split** — Treats the loss layer as a standard Transformer layer in pipeline splitting. Default: `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split` | **Embedding layer pipeline split** — Treats the input embedding layer as a standard Transformer layer in pipeline splitting. Default: `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage` | **First stage layer count** — Specifies the number of layers in the first pipeline stage. Default: `none` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage` | **Last stage layer count** — Specifies the number of layers in the last pipeline stage. Default: `none` |

##### Weight Management

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.use_mbridge` | **Megatron-Bridge weight conversion** — Currently, keep this set to `True` |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | **Distributed checkpoint** — Saves and loads weights in a distributed format. Default: `False` |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | **Distributed weight path** — Load path for the distributed checkpoint. Default: `null` |

---

## 3. Performance Optimization Features

### 3.1 Memory Optimization

| Feature | Inference/Training | Description |
|:---|:---|:---|
| KV Cache dynamic release (`free_cache_engine`) | Inference (vLLM) | Automatically unloads the KV Cache after the generation phase; enabled by default |
| Memory saver mode (`enable_memory_saver`) | Inference (SGLang) | Supports dynamic device memory release and restoration; verl defaults to `True` |
| Parameter CPU offloading (`param_offload`) | Training (FSDP/Megatron) | Offloads model weights to the CPU |
| Optimizer CPU offloading (`optimizer_offload`) | Training (FSDP/Megatron) | Offloads optimizer states to the CPU |
| Chunked entropy computation (`entropy_from_logits_with_chunking`) | Training (FSDP) | Computes entropy in chunks to reduce peak device memory |
| Entropy computation chunk size (`entropy_from_logits_chunk_size`) | Training (FSDP) | Chunk size for entropy computation |
| Entropy computation recomputation (`entropy_checkpointing`) | Training (FSDP) | Enables recomputation for entropy computation |
| Pipeline output deallocation (`deallocate_pipeline_outputs`) | Training (Megatron) | Releases transferred tensors in pipeline parallelism scenarios |
| Activation recomputation (`recompute_granularity`) | Training (Megatron) | Supports three granularity levels: full, selective, and none |

### 3.2 Compute Acceleration

| Feature | Inference/Training | Description |
|:---|:---|:---|
| Chunked prefill (`enable_chunked_prefill`) | Inference (vLLM) | Chunks large prefill and processes it with decode batches |
| Prefix caching (`enable_prefix_caching`) | Inference (vLLM) | Automatically caches shared prefixes to reduce redundant computation |
| Flash Attention | Training (Megatron) | Accelerates attention computation using Flash Attention; enabled by default |
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
| Sequence parallelism (SP/Ulysses) | ✅ | ✅ | ✅ | ✅ | Sequence dimension splitting, supporting long sequences |
| Context parallelism (CP) | ✅ | — | — | ✅ | Context parallel processing |

---

## 4. Mixture of Experts (MoE) Model Features

### vLLM/SGLang Inference MoE Support

- **Expert parallelism (EP)** — Configured using the `ep_size` parameter to assign different experts to different NPU devices
- SGLang provides highly optimized EP communication kernels through [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md)

### Megatron Training MoE Support

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | Expert parallelism (EP) size. The default is `1`. |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` | TP-extended EP size. The default is `null`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm` | **Group GEMM** — Uses Group GEMM to optimize expert computation in MoE scenarios. The default is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype` | **Router data type** — The data type for routing and weighted average of expert outputs. Options are `fp32`/`fp64`. The default is `fp32`, which improves stability in multi-expert scenarios. |

---

## 5. Limitations and Precautions

1. **FSDP1 vs FSDP2 Differences**
   - `forward_prefetch` and `use_orig_params` apply only to FSDP1
   - FSDP2 is the recommended default version. For API support details, refer to the [Ascend PyTorch Release Notes](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-7-1/torch-distributed-fsdp.md)

2. **Recomputation parameter dependencies**
   - `recompute_method` requires `recompute_granularity='full'` to take effect
   - `recompute_num_layers` requires `recompute_granularity='full'` to take effect
   - When `recompute_method='uniform'`, `recompute_num_layers` indicates the number of Transformer layers in each recomputation unit, and must be divisible by the number of model layers in the current process

4. **SGLang NPU-specific configuration**
   - `attention_backend` must be set to `ascend` to call the Ascend optimized kernel
   - `enable_memory_saver` is enabled by default in verl, so no additional configuration is required

---

## Appendix: Parameter Quick Reference

### Inference Backend Parameter Quick Reference

| Parameter category | vLLM parameters | SGLang parameters | verl general parameters |
|:---|:---|:---|:---|
| Model path | `model_path` | `model_path` | `actor_rollout_ref.model.path` |
| Device memory control | `gpu_memory_utilization` | `mem_fraction_static` | `actor_rollout_ref.rollout.gpu_memory_utilization` |
| Graph mode | `enforce_eager` | `disable_cuda_graph` | `actor_rollout_ref.rollout.enforce_eager` |
| Quantization | `quantization` | `quantization` | `actor_rollout_ref.rollout.engine_kwargs.{vllm,sglang}.quantization` |
| Maximum sequence length | `max_model_len` | — | `actor_rollout_ref.rollout.max_model_len` |
| Maximum concurrency | `max_num_seqs` | `max_running_requests` | `actor_rollout_ref.rollout.max_num_seqs` |
| Tokenizer | `skip_tokenizer_init` | `skip_tokenizer_init` | `actor_rollout_ref.rollout.skip_tokenizer_init` |
| Remote code | `trust_remote_code` | `trust_remote_code` | `actor_rollout_ref.model.trust_remote_code` |
| TP parallelism | `tp_size` | `tp_size` | `actor_rollout_ref.rollout.tensor_model_parallel_size` |
| DP parallelism | `dp_size` | `dp_size` | `actor_rollout_ref.rollout.data_parallel_size` |
| EP parallelism | `ep_size` | `ep_size` | `actor_rollout_ref.rollout.expert_parallel_size` |

### Training Backend Parameters Quick Reference

| Parameter category | FSDP parameter | Megatron parameter |
|:---|:---|:---|
| Parameter offload | `fsdp_config.param_offload` | `megatron.param_offload` |
| Optimizer offload | `fsdp_config.optimizer_offload` | `megatron.optimizer_offload` |
| Sequence parallelism | `ulysses_sequence_parallel_size` | `context_parallel_size` |
| Flash Attention | — | `override_transformer_config.use_flash_attn` |
| Recomputation granularity | — | `override_transformer_config.recompute_granularity` |
| Distributed checkpoint | — | `use_dist_checkpointing` |
