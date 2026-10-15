Ascend Performance Tuning Guide
====================================

Last updated:  01/29/2026.

Author:  `Xiaobo Hu <https://github.com/tardis-key>`_, `Haozhe Li <https://github.com/ZLiao097>`_

The performance tuning methods described in `Perf Tuning <https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst>`_ also apply to Ascend devices. This document highlights tuning methods specific to Ascend, including fused operator optimization, specific hardware configurations, and Ascend affinity features.

Fusion Operators
--------------------------

List of common fused operators
**********************************

The optimization principle of fused operators is to fuse the computation of multiple operators into a single operator through mathematically equivalent replacements. This reduces redundant computation and the number of dispatches, thereby improving performance. Several typical NPU fused operators are listed below. The npu_patch.py file has already completed these replacements for the Qwen2 and Qwen3 series models.

For the fully fused operators currently used in verl, refer to `npu_patch.py <https://github.com/verl-project/verl/blob/main/verl/models/transformers/npu_patch.py>`_

Matrix Computation-Communication operator fusion (MC2) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MC2 is a general term for a series of computation-communication fusion operators in CANN. These operators fuse originally sequential communication and computation operations together, optimizing performance through internal partitioning and pipeline parallel execution.

In vllm-ascend, you can specify environment variables:

.. code-block:: sh

    export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1

Enable ``torch_npu.npu_mm_all_reduce_base`` in the ``RowParallelLinear`` forward computation to merge the separate ``matmul`` and ``allreduce`` operations into a single fused operator.

`RotaryMul&RotaryMulGrad <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_rotary_mul.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu API:  ``torch_npu.npu_rotary_mul(x, r1, r2)``

Parameter description:

- x: q, k. The shape requires the input to be 4-dimensional, typically ``[B, N, S, D]``, ``[B, S, N, D]``, or ``[S, B, N, D]``.

- r1: cos value. The input shape must be 4-dimensional, typically ``[1, 1, S, D]``, ``[1, S, 1, D]``, or ``[S, 1, 1, D]``.

- r2: sin value. The input shape must be 4-dimensional, generally ``[1, 1, S, D]``, ``[1, S, 1, D]``, or ``[S, 1, 1, D]``.

`RmsNorm&RmsNormGrad <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/%EF%BC%88beta%EF%BC%89torch_npu-npu_rms_norm.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu interface:  ``torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor)`` 
Parameter description:

- self: Tensor type. The shape supports 1 to 8 dimensions.

- gamma: Tensor type, usually a weight, and the shape must match the last few dimensions of self.

- epsilon: Float data type, used to prevent division by zero errors.

Output description:

- The first output is a Tensor, which is the final output y of the calculation formula.

- The second output is a Tensor, the intermediate result rstd of rms_norm, used for backward computation.

`Swiglu <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/%EF%BC%88beta%EF%BC%89torch_npu-npu_swiglu.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The torch_npu interface is:  ``torch_npu.npu_swiglu(Tensor self, int dim=-1) -> (Tensor)``

Parameter description:

- self: Tensor type, supports shapes with 1-8 dimensions.

- dim: Int type, defaults to -1.

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

For detailed usage, see the document link in the title.

Using fused operators in the FSDP backend
*******************************************

In the ``verl/models/transformers/npu_patch.py`` file, available fused operators are replaced using patches. You do not need to perform other operations to use them by default.

Using fused operators in the Megatron backend
*********************************************

The fused operators of Megatron are integrated in MindSpeed, and you must add specific parameters to enable them:

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

Ascend General Configuration
----------------------------

`Operator dispatch <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/ENV/docs/zh/environment_variable_reference/TASK_QUEUE_ENABLE.md>`_
************************************************************************************************************************************************************************************************************

You can configure the optimization level of the task_queue operator dispatch queue using ``TASK_QUEUE_ENABLE``. The default is Level 1 optimization. This configuration reduces the host dispatch time and can alleviate the issue of excessive overall free time caused by dispatching.

.. image :: https://github.com/verl-project/verl-data/blob/main/images/ascend/perf_tuning_task_queue.png
    :width: 500px

Level 0: Dispatch pipeline optimization is not enabled.

Level 1 : Divides the operator dispatch task into two segments. One part of the task (mainly the aclnn operator calls) is placed on the newly added second-level pipeline. The first-level and second-level pipelines pass tasks through an operator queue and run in parallel. Through partial overlapping, this approach reduces the overall dispatch time and improves end-to-end performance.

Level 2 : Building on Level 1 optimization, this level further balances the task load between the first and second level pipelines. It mainly migrates workspace-related tasks to the second level pipeline. This provides better masking effects and greater performance gains. This configuration takes effect only in binary scenarios. The recommended configuration value is Level 2 optimization.

`Communication algorithm orchestration expansion <https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0096.html>`_
************************************************************************************************************************************************************************************************************
Use the environment variable ``HCCL_OP_EXPANSION_MODE=AIV`` to configure the orchestration expansion location of the communication algorithm. The following values are supported:

- **AI_CPU:** Indicates that the orchestration and expansion location of the communication algorithm is on the AI CPU on the device side. The device side automatically selects the corresponding scheduler based on the hardware model.

- **AIV:** indicates that the communication algorithm is orchestrated and expanded on the Vector Core on the Device side, and execution also occurs on the Vector Core.

- **HOST:** Indicates that the orchestration and expansion of the communication algorithm is performed on the Host CPU, and the Device side automatically selects the corresponding scheduler based on the hardware model.

- **HOST_TS:** Indicates that the orchestration and expansion of the communication algorithm takes place on the host-side CPU. The host dispatches tasks to the Task Scheduler on the device, and the Task Scheduler on the device schedules and executes the tasks.

Inference Phase Tuning
--------------------------

Chunked Prefill in V1
***************************

The current version of VLLM enables VLLM V1 by default. Use the following configuration to enable Chunked Prefill:

.. code-block:: sh

    actor_rollout_ref.rollout.enable_chunked_prefill=True

For the principles, refer to the `VLLM official documentation <https://docs.vllm.ai/en/v0.4.2/models/performance.html>`_.

Graph Mode
***************************

Similar to CUDA, the NPU enables **ACL Graph** through the following configuration:

.. code-block:: sh

    actor_rollout_ref.rollout.enforce_eager=False

.. note::
    ACL Graph and ``taskqueue Level 2`` conflict in principle; **the two cannot be enabled simultaneously**.

Training Phase Tuning
--------------------------

FSDP
**********************************

.. csv-table::
   :header: "FSDP", "Description"
   :widths: 30, 60

   "/","Shard optimizer only (Zero-1)"
   SHARD_GRAD_OP,Shard gradients and optimizer (Zero-2)
   "HYBRID_SHARD","Shard weights, gradients, and optimizer (Zero-3)"
   "2D device_mesh+HYBRID_SHARD","Also known as HSDP (FSDP+DDP). For example, if device_mesh=[2,8], every 8 ranks form an FSDP group. FSDP sharding is performed within each group. There are two groups in total, and DDP is performed between the two groups. Gradients are synchronized through allreduce."
   "2D device_mesh+HYBRID_SHARD_ZERO2","The Zero2 version of HSDP"
   NO_SHARD,DDP

FSDP does not support Zero-1. VeRL determines the device mesh value based on the number of devices and ``actor_rollout_ref.actor.fsdp_config.fsdp_size``, and uses Zero-3 for sharding by default. If the model is small (less than 7B is recommended), you can use Zero-2 on FSDP/FSDP2 to optimize performance by setting the parameter ``actor_rollout_ref.actor.fsdp_config.reshard_after_forward`` to ``True``.

Megatron
**********************************

When the model is large, using Megatron as the training backend enables more flexible performance tuning.

When the device memory under DP parallelism cannot accommodate the model, enable TP first to split the model weights. If the model is still too large, enable PP to further split it. If the sequence is too long and causes excessively large activations, you can enable CP and SP for optimization. In MoE models, you can additionally enable EP to control the splitting of experts. If the experts are too small, to avoid splitting the weights too finely, you can enable ETP to avoid TP splitting in the MoE part. This allows you to distribute multiple complete experts across DP and TP instead.

TP, PP, EP, ETP, and Megatron are used in the same way. To enable CP and SP on the NPU:

- SP: ``Sequence Parallel`` further improves computing efficiency based on Tensor Parallelism. It is a parallel computing method that splits the sequence dimension of the input data. On the NPU, invoke SP using MindSpeed:
  ::

      actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True

- CP: ``Context Parallel`` is a method for processing neural network activations in parallel across multiple GPUs/NPUs. It achieves this by partitioning the input tensor along the sequence dimension. On NPUs, you invoke CP using MindSpeed (both parameters must be added simultaneously):
  ::

      actor_rollout_ref.actor.megatron.context_parallel_size
      actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size

Megatron-distributed optimizer
**********************************

When working with larger models, the optimizer typically needs to be sharded to each device within a DP domain to save device memory. To enable the distributed optimizer on the NPU with the Megatron backend:

::

    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
