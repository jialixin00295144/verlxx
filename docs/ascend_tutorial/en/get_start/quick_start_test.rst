Ascend Quick Start Guide
========================

**Last updated:** 2026/07/14.

Key updates
-----------

- 2026/06/30: Added coverage for four common training and inference backend combinations, helping you quickly select an appropriate startup script during the quickstart phase.
- 2026/05/13: Separated the quick start and installation guidance.
- 2025/12/11: Existing verl scenarios now support automatic identification of NPU device types. When you run GPU scripts on Ascend, you generally do not need to explicitly set the ``trainer.device=npu`` parameter. You can still prioritize new features by setting ``trainer.device``, gradually adapting to the automatic identification capability.


Contents
--------

- `Hardware Support <#硬件支持>`_
- `Quick Start <#快速开始>`_
   - `Environment Preparation <#环境准备>`_ 
   - `Weight Preparation <#权重准备>`_
   - `Data Preparation <#数据准备>`_
   - `How to Run <#运行方式>`_
- `Appendix <#附录>`_
   - `SGLang Backend Enablement Notes <#SGLang-后端使能说明>`_
   - `Converting vLLM Backend Scripts to SGLang <#vLLM-后端脚本转换为-SGLang>`_

Hardware Support
----------------

- Atlas 200T A2 Box16
- Atlas 900 A2 PoD
- Atlas 800T A3



Quick Start
---------------------------------

This guide targets the Ascend NPU environment and provides a minimal GRPO training verification process using GSM8K and Qwen3-0.6B, covering four training and inference backend combinations to help you get started quickly.

Verification scope:

- Environment entry: whether the verl entry is available;
- Data reading: whether the GSM8K dataset is correctly parsed;
- Component initialization: whether the actor, rollout, and reference workers can be initialized;
- Model inference: whether the vLLM-Ascend/sglang rollout can generate;
- End-to-end pipeline: whether the training process can complete the first step.

Configuration tips:

- Default configuration: The sample scripts use the Qwen3-0.6B model and the GSM8K dataset.
- Hardware adaptation: Because each A3 card contains 2 dies, set the ``n_gpus_per_node`` parameter to 16 to run the sample (each A2 card contains 1 die)

Environment preparation
~~~~~~~~~~~~~~~~~~~~~~~

Before running the scripts in this document, ensure that you have installed the verl Ascend environment. For environment installation details, see `Ascend Installation Guide <./install_guidance.rst>`_.

Weight Preparation
~~~~~~~~~~~~~~~~~~

1. Download the Qwen3-0.6B model weights from Hugging Face on your own.

2. The default path for reading weights in the script is ``~/models/Qwen/Qwen3-0.6B``. We recommend that you place the weights in this path. If the path is different, modify MODEL_PATH in the script to point to the local path.

Data preparation
~~~~~~~~~~~~~~~~~

1. Download the original GSM8K dataset from Hugging Face yourself.

2. Run the following command (adjust the command based on your actual dataset path)

   .. code-block:: bash

      python3 examples/data_preprocess/gsm8k.py --local_dataset_path /download/path/hf_data/gsm8k/

   Generate the following files:

      .. code-block:: text

         ~/data/gsm8k/train.parquet
         ~/data/gsm8k/test.parquet

How to run
~~~~~~~~~~~~~~~

1. Enter the project directory: ``cd /your/path/verl``.

2. Enable the CANN environment: Run the following command. (If you have customized the CANN path, adjust the command based on the actual path.)

   .. code-block:: bash

      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      source /usr/local/Ascend/nnal/atb/set_env.sh

3. Run the script: Four common training and inference backend combinations are currently provided. Refer to the following table to select the corresponding script:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 60

   * - Combination
     - Training backend
     - Rollout backend
     - Run method
   * - vLLM + FSDP2
     - FSDP2
     - vLLM-Ascend
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_fsdp2_vllm_ascend.sh
   * - vLLM + Megatron
     - Megatron
     - vLLM-Ascend
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_megatron_vllm_ascend.sh
   * - SGLang + FSDP2
     - FSDP2
     - SGLang
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_fsdp2_sglang_ascend.sh
   * - SGLang + Megatron
     - Megatron
     - SGLang
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_megatron_sglang_ascend.sh

For detailed descriptions of the parameters in the script, see `Training Configuration Parameters and Metrics <../dev_guide/model_dev/parameter_and_metrics.md>`_.

For details on starting multi-node tasks, see `Multi-Machine Task Startup Practice Guide <../model_support/examples/multi-machine_task_startup_practice.rst>`_.

Appendix
-------------------------------------------

SGLang Backend Enablement Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

verl currently parses common inference parameters. For details, see the ``ServerArgs`` initialization parameters in `async_sglang_server.py <../../../../verl/workers/rollout/sglang_rollout/async_sglang_server.py>`_.

You can pass all other `SGLang arguments <https://github.com/sgl-project/sglang/blob/v0.5.10/docs/advanced_features/server_arguments.md>`_ using ``engine_kwargs``.


Converting vLLM backend scripts to SGLang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to manually convert the vLLM backend inference script to SGLang, add or modify the following parameters.

.. code-block:: bash

   # Required
   actor_rollout_ref.rollout.name=sglang \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend" \

   # Optional
   # Enable inference EP. For detailed usage, refer to:
   # https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.deepep_mode="auto" \
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.moe_a2a_backend="deepep" \

   # Must be set to True for MoE models with multiple DP (default is False)
   +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=False \