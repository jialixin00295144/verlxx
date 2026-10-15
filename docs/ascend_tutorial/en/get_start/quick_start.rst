Ascend Quick Start Guide
=========================

**Last updated:** 2026/07/14.

Key updates
-----------

- 2026/06/30: Added coverage for four common training and inference backend combinations to help users quickly select the appropriate startup script during the quickstart phase.
- 2026/05/13: Separated the quick start and install guidance.
- 2025/12/11: The verl scenario currently supports automatic identification of NPU device types. When GPU scripts run on Ascend, you no longer need to explicitly set the ``trainer.device=npu`` parameter in principle. New features can still take priority by setting ``trainer.device``, gradually adapting to the automatic identification capability.


Contents
--------

- `Hardware Support <#硬件支持>`_
- `Quick Start <#快速开始>`_
   - `Environment Preparation <#环境准备>`_ 
   - `Weight Preparation <#权重准备>`_
   - `Data Preparation <#数据准备>`_
   - `Running Methods <#运行方式>`_
- `Appendix <#附录>`_
   - `SGLang Backend Enablement Instructions <#SGLang-后端使能说明>`_
   - `Converting vLLM Backend Scripts to SGLang <#vLLM-后端脚本转换为-SGLang>`_

Hardware Support
----------------

- Atlas 200T A2 Box16
- Atlas 900 A2 PoD
- Atlas 800T A3



Quick Start
---------------------------------

This guide targets the Ascend NPU environment. It provides a minimal GRPO training verification process using GSM8K and Qwen3-0.6B. The process covers four training and inference backend combinations to help you get started quickly.

Verification scope:

- Environment entry: whether the verl entry is available;
- Data reading: whether the GSM8K dataset is correctly parsed;
- Component initialization: whether the actor, rollout, and reference workers can be initialized;
- Model inference: whether the vLLM-Ascend/sglang rollout can generate;
- End-to-end link: whether the training process can complete the first step.

Configuration tips:

- Default configuration: The example scripts use the Qwen3-0.6B model and the GSM8K dataset.
- Hardware adaptation: Because each A3 card contains 2 dies, set the ``n_gpus_per_node`` parameter to 16 to run the examples (each A2 card contains 1 die).

Environment Preparation
~~~~~~~~~~~~~~~~~~~~~~~

Before running the scripts in this document, ensure that you have completed the verl Ascend environment installation. For environment installation details, refer to the `Ascend Installation Guide <./install_guidance.rst>`_.

Preparing Weights
~~~~~~~~~~~~~~~~~

1. Download the Qwen3-0.6B model weights from Hugging Face yourself.

2. The default path for reading weights in the script is ``~/models/Qwen/Qwen3-0.6B``. We recommend placing the weights in this path. If the path is different, modify MODEL_PATH in the script to point to the local path.

Data preparation
~~~~~~~~~~~~~~~~

1. Download the original GSM8K dataset from Hugging Face on your own.

2. Run the following commands (adjust the commands according to your actual dataset path)

   .. code-block:: bash

      python3 examples/data_preprocess/gsm8k.py --local_dataset_path /download/path/hf_data/gsm8k/

   The following files are generated:

      .. code-block:: text

         ~/data/gsm8k/train.parquet
         ~/data/gsm8k/test.parquet

How to run
~~~~~~~~~~~~~~~

1. Go to the project directory: ``cd /your/path/verl``.

2. Enable the CANN environment: Run the following command. (If you customized the CANN path, adjust the command based on the actual path.)

   .. code-block:: bash

      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      source /usr/local/Ascend/nnal/atb/set_env.sh

3. Run the script: Currently, four common training and inference backend combinations are available. Refer to the following table to select the corresponding script:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 60

   * - Combination
     - Training backend
     - Rollout backend
     - Execution method
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

For detailed descriptions of the parameters in the script, refer to `Training Configuration Parameters and Metrics <../dev_guide/model_dev/parameter_and_metrics.md>`_.

For details about launching multi-node tasks, see `Multi-Machine Task Startup Practice Guide <../model_support/examples/multi-machine_task_startup_practice.rst>`_.

Appendix
-------------------------------------------

Enabling the SGLang Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, verl parses common inference parameters. For details, see the ``ServerArgs`` initialization parameters in `async_sglang_server.py <../../../../verl/workers/rollout/sglang_rollout/async_sglang_server.py>`_.

You can pass other `SGLang arguments <https://github.com/sgl-project/sglang/blob/v0.5.10/docs/advanced_features/server_arguments.md>`_ using ``engine_kwargs``.


Converting vLLM backend scripts to SGLang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to manually convert the vLLM backend inference script to SGLang, add or modify the following parameters.

.. code-block:: bash

   # Required
   actor_rollout_ref.rollout.name=sglang \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend" \

   # Optional
   # Enable inference EP. For details, see:
   # https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.deepep_mode="auto" \
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.moe_a2a_backend="deepep" \

   # Must be set to True for MoE models with multiple DP (default is False)
   +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=False \