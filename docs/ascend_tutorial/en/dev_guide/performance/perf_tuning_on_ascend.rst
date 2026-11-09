Ascend Performance Tuning Guide
====================================

Last updated:  01/29/2026.

Author:  `Xiaobo Hu <https://github.com/tardis-key>`_, `Haozhe Li <https://github.com/ZLiao097>`_

The performance tuning methods described in `Perf Tuning <https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst>`_ are also applicable to Ascend devices. This document focuses on Ascend-specific tuning methods, including operator fusion optimization, specific hardware configurations, and Ascend affinity features.

Fused Operators
--------------------------

List of common fused operators
**********************************

The optimization principle of fused operators is to use mathematically equivalent substitution to fuse the computation of multiple operators into a single operator. This reduces redundant computation and the number of dispatches, thereby improving performance. Several typical NPU fused operators are listed below. Currently, all of them have been replaced in npu_patch.py for the Qwen2 and Qwen3 series models.

For all fused operators currently used in verl, refer to `npu_patch.py <https://github.com/verl-project/verl/blob/main/verl/models/transformers/npu_patch.py>`_

Matrix Computation-Communication operator fusion (MC2) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MC2 is a collective term for a series of computation-communication fusion operators in CANN. These operators fuse originally sequential communication and computation operations together and optimize performance through internal splitting and pipeline parallel execution.

In vllm-ascend, you can specify environment variables:

.. code-block:: sh

    export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1

Enable `torch_npu.npu_mm_all_reduce_base` in the `RowParallelLinear` during the forward computation to merge the separate `matmul` and `allreduce` into a fused operator.

`RotaryMul&RotaryMulGrad <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_rotary_mul.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu interface:  ``torch_npu.npu_rotary_mul(x, r1, r2)``

Parameter description:

- x: q, k, and shape require 4-dimensional inputs, typically ``[B, N, S, D]``, ``[B, S, N, D]``, or ``[S, B, N, D]``.

- r1: cos value. The input shape must be 4-dimensional, typically ``[1, 1, S, D]`` , ``[1, S, 1, D]`` , or ``[S, 1, 1, D]`` .

- r2: sin value. The shape requires the input to be 4-dimensional, generally ``[1, 1, S, D]``, ``[1, S, 1, D]``, or ``[S, 1, 1, D]``.

`RmsNorm&RmsNormGrad <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/%EF%BC%88beta%EF%BC%89torch_npu-npu_rms_norm.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu interface:  ``torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor)`` 
Parameter description:

- self: Tensor type. The shape supports 1 to 8 dimensions.

- gamma: Tensor type, typically a weight, and the shape must match the last few dimensions of self.

- epsilon: Float data type used to prevent division by zero errors.

Output description:

- The first output is a Tensor, which is the final output y of the computation formula.

- The second output is a Tensor, the intermediate result rstd of rms_norm, used for backward computation.

`Swiglu <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/%EF%BC%88beta%EF%BC%89torch_npu-npu_swiglu.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu API:  ``torch_npu.npu_swiglu(Tensor self, int dim=-1) -> (Tensor)``

Parameter description:

- self: Tensor type, with the shape supporting 1 to 8 dimensions.

- dim: Int type, default is -1.

Output description:

- The output is a Tensor, which is the final output y of the calculation formula.

`GroupMatMul <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function prototype:

.. code:: python

    npu_grouped_matmul(
        x, 
        weight, 
        *, 
        bias=None, 
        scale=None, 
        offset=None, 
        antiquant_scale=None, 
        antiquant_offset=None, 
        per_token_scale=None, 
        group_list=None, 
        activation_input=None, 
        activation_quant_scale=None, 
        activation_quant_offset=None, 
        split_item=0, group_type=None, 
        group_list_type=0, 
        act_type=0, 
        output_dtype=None, 
        tuning_config=None
    ) -> List[Tensor]

For detailed usage, see the title document link.

Using fused operators in the FSDP backend
*****************************************

The ``verl/models/transformers/npu_patch.py`` file replaces available fused operators through patches, so you can use them by default without performing any other operations.

Using fused operators in the Megatron backend
************************************************

The fused operators of Megatron are integrated in MindSpeed. To enable them, add specific parameters:

1. **Flash Attention (must be enabled)**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True

2. **RotaryMul**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True

3. **RMSNorm**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True

4. **GroupMatMul**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True

5. **Swiglu**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True

6. **Permute/Unpermute**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.fused_permute_unpermute=True

7. **MC2**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_mc2=True

General Ascend Configuration
----------------------------

`Operator dispatch <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/ENV/docs/zh/environment_variable_reference/TASK_QUEUE_ENABLE.md>`_
************************************************************************************************************************************************************************************************************

You can configure the task_queue operator dispatch queue optimization level through ``TASK_QUEUE_ENABLE``. The default is Level 1 optimization. This configuration reduces the host dispatch time and alleviates the issue of excessive overall free time caused by dispatch.

.. image :: https://github.com/verl-project/verl-data/blob/main/images/ascend/perf_tuning_task_queue.png
    :width: 500px

Level 0: Dispatch pipeline optimization is not enabled.

Level 1 : Split the operator dispatch task into two stages. Place part of the tasks (mainly the aclnn operator calls) on the newly added secondary pipeline. The primary and secondary pipelines pass tasks through an operator queue and run in parallel. This partial overlapping reduces the overall dispatch time and improves end-to-end performance.

Level 2: Based on the Level 1 optimization, this level further balances the task load between the first-level and second-level pipelines. It mainly migrates workspace-related tasks to the second-level pipeline, achieving better masking effects and greater performance gains. This configuration takes effect only in binary scenarios. The recommended configuration value is Level 2 optimization.

`Communication Algorithm Orchestration Expansion <https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0096.html>`_
************************************************************************************************************************************************************************************************************
Use the environment variable ``HCCL_OP_EXPANSION_MODE=AIV`` to configure the orchestration expansion location of the communication algorithm. The following values are supported:

- **AI_CPU:** Indicates that the orchestration and expansion location of the communication algorithm is on the AI CPU on the Device side. The Device side automatically selects the corresponding scheduler based on the hardware model.

- **AIV:** Indicates that the communication algorithm is orchestrated and expanded on the Vector Core on the Device side, and is also executed on the Vector Core.

- **HOST:** Indicates that the communication algorithm is orchestrated and expanded on the host-side CPU, and the device side automatically selects the corresponding scheduler based on the hardware model.

- **HOST_TS:** Indicates that the communication algorithm is orchestrated and expanded on the host-side CPU. The host dispatches tasks to the Task Scheduler on the device, and the Task Scheduler schedules and executes the tasks.

Inference phase tuning
--------------------------

Chunked Prefill in V1
***************************

The current version of VLLM enables VLLM V1 by default. Use the following configuration to enable Chunked Prefill:

.. code-block:: sh

    actor_rollout_ref.rollout.enable_chunked_prefill=True

For the principles, refer to the `official VLLM documentation <https://docs.vllm.ai/en/v0.4.2/models/performance.html>`_.

Graph Mode
***************************

Similar to CUDA, the NPU enables **ACL Graph** using the following configuration:

.. code-block:: sh

    actor_rollout_ref.rollout.enforce_eager=False

.. note::
    ACL Graph and ``taskqueue Level 2`` have conflicting mechanisms, and **they cannot be enabled simultaneously**.

Training phase tuning
--------------------------

FSDP
**********************************

.. csv-table::
   :header: "FSDP", "Description"
   :widths: 30, 60

   "/","Shard optimizer only (Zero-1)"
   SHARD_GRAD_OP,Shard gradients and optimizer (Zero-2)
   "HYBRID_SHARD","Shard weights, gradients, and optimizer (Zero-3)"
   "2D device_mesh+HYBRID_SHARD","Also known as HSDP (FSDP+DDP). For example, device_mesh=[2,8]. Every 8 ranks form an FSDP group where FSDP sharding is performed. There are two groups in total, and DDP is performed between them, synchronizing gradients using allreduce."
   "2D device_mesh+HYBRID_SHARD_ZERO2","Zero2 version of HSDP"
   NO_SHARD,DDP

FSDP does not support Zero-1. VeRL determines the device mesh value based on the number of devices and ``actor_rollout_ref.actor.fsdp_config.fsdp_size``, and uses Zero-3 for sharding by default. If the model is small (less than 7B is recommended), set ``actor_rollout_ref.actor.fsdp_config.reshard_after_forward`` to ``True`` to use Zero-2 on FSDP/FSDP2 to optimize performance.

Megatron
**********************************

For large models, using Megatron as the training backend enables more flexible performance tuning.

When DP parallelism device memory cannot accommodate the model, enable TP first to split the model weights; if the model is still too large, enable PP for further splitting. If the sequence is too long and causes excessive activations, enable CP and SP for optimization. In MoE models, you can additionally enable EP to control expert splitting; if the experts are too small, to avoid splitting weights too finely, enable ETP to avoid TP splitting in the MoE part, and distribute multiple complete experts across DP and TP.

TP, PP, EP, and ETP are used in the same way as Megatron. To enable CP and SP on the NPU, use the following method:

- SP: ``Sequence Parallel`` further improves computing efficiency based on Tensor Parallel. It is a parallel computing method that splits the sequence dimension of input data. On the NPU, SP is invoked through MindSpeed:
  ::

      actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True

- CP: ``Context Parallel`` is a method that processes neural network activations in parallel across multiple GPUs/NPUs by partitioning the input tensor along the sequence dimension. On NPUs, you invoke CP using MindSpeed (both parameters must be added simultaneously):
  ::

      actor_rollout_ref.actor.megatron.context_parallel_size
      actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size

Megatron-distributed optimizer
**********************************

When training large models, you typically need to shard the optimizer across each device in a DP domain to save device memory. To enable the distributed optimizer on the NPU using the Megatron backend:

::

    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
