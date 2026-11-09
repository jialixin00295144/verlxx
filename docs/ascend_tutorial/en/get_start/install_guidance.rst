Ascend Installation Guide
=========================

Last updated: 11/06/2026.

.. contents:: Contents
   :local:
   :depth: 2

Key updates
-----------

-  2026/11/03: This branch adjusts the target versions of vLLM / vLLM-Ascend to ``0.30.0``,
   torch_npu to ``2.10.0.post7``, transformers to ``5.14.1``, xgrammar to ``>=0.2.1,<1.0.0``,
   triton-ascend to ``3.6.0.post2``; Megatron-LM to ``core_v0.18.0``; MindSpeed remains at ``core_r0.18.0``.
-  2026/08/29: Updated MindSpeed / Megatron-LM from ``core_r0.16.0`` to
   ``core_r0.18.0``\ . Deprecated mbridge and switched to Megatron-Bridge, with Megatron-Bridge version ``0.5.0``\ .
   torch_npu updated to ``2.10.0.post4``\ .
-  2026/08/03: Updated vLLM / vLLM-Ascend from ``0.18.0`` to ``0.23.0``\ , and the corresponding
   base environment versions for vLLM were adjusted to torch ``2.10.0``\ , torch_npu ``2.10.0.post2``\ .
-  2026/05/13: vLLM has been updated per `PR
   #6291 <https://github.com/verl-project/verl/pull/6291>`__\ , updating vLLM /
   vLLM-Ascend from ``0.13.0`` to ``0.18.0``\ , and the corresponding
   base environment versions for vLLM were adjusted to torch ``2.9.0``\ , torch_npu
   ``2.9.0.post2``\ .
-  2025/12/11: verl existing scenarios now support automatic detection of NPU device types. In principle, GPU
   scripts no longer need to explicitly set
   ``trainer.device=npu``\ when running on Ascend; new features can still set ``trainer.device``
   to prioritize specifying the device type.

..

   [Note] To automatically recognize the NPU device type, the environment where the program runs must contain the
   ``torch_npu`` package. If the environment does not contain ``torch_npu``\ , you must explicitly specify
   ``trainer.device=npu``\ .

Hardware support
----------------

Atlas 200T A2 Box16

Atlas 900 A2 PoD

Atlas 800T A3

..

   [Description] For software installation on Ascend 950PR&950DT products, refer to :doc:`Installation Guide (Ascend 950PR&950DT products) <install_guidance_950>`.


Framework Backend Support
-------------------------

The NPU currently supports the deployment of the following common training and inference backends. You can directly obtain the published images using our :doc:`Ascend Image Guide <dockerfile_build_guidance>` or perform a custom installation according to the following sections.

.. list-table::
   :header-rows: 1

   * - Inference engine
     - Training engine
   * - vLLM
     - FSDP/FSDP2/Megatron
   * - SGLang
     - FSDP/FSDP2/Megatron

Deployment Guide
----------------

1. Obtaining, building, and using Docker images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can obtain the image from `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_, or build it from a Dockerfile. For related instructions, refer to
:doc:`Ascend Image Description <dockerfile_build_guidance>`\ .


2. Custom Installation-vLLM + FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Key Version Support and Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============= ======================================= ===================
Dependency          Version                                     Description
============= ======================================= ===================
``
============= ======================================= ===================

.. note::

   MegatronAdaptor uses ``26.2.0_core_r0.18.0``.
   MindSpeed-Bridge uses ``26.2.0_v0.5.0``.
   TransformerEngineNPU uses ``26.2.0_release_v2.17``.
   MindSpeed-Ops uses ``26.2.0``.

This table aligns torch, torch_npu, torchvision, torchaudio, triton-ascend, and transformers according to the `vLLM-Ascend 0.30.0 installation dependencies <https://github.com/vllm-project/vllm-ascend/blob/releases/v0.30.0/requirements.txt>`_; xgrammar uses the `vLLM 0.30.0 dependency range <https://github.com/vllm-project/vllm/blob/v0.30.0/requirements/common.txt>`_.
   Use the Triton implementation provided by triton-ascend; you do not need to install the triton package separately.
   Currently, the two upstreams have a FastAPI constraint conflict: vLLM requires ``>=0.133.0,<0.137.0``, and vLLM-Ascend requires ``<0.124.0``. No version satisfies both constraints simultaneously. Resolve this conflict first, then use ``pip check`` to verify the final environment and validate runtime compatibility.
   vLLM and SGLang share ``requirements-npu.txt``; vLLM installs ``triton-ascend==3.6.0.post2`` after all dependencies are installed, while SGLang keeps its original dependency version.


Pre-installation preparation (HDK & CANN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CANN is a heterogeneous computing architecture for NPUs. The following are the A3 installation instructions for the ARM platform. Follow these instructions to download and install HDK and CANN. Alternatively, download and install them from the `CANN community <https://www.hiascend.com/cann/download>`_ based on your system hardware model.

.. code:: bash

   # Configure the user group
   sudo groupadd HwHiAiUser
   sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
   # Install dependencies and configure the repository
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r) 
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # Install the NPU driver
   sudo yum install -y Atlas-A3-hdk-npu-driver-26.2.rc1
   # Install the Toolkit. You can specify --install-path to customize the path
   sudo yum install -y Ascend-cann-toolkit-9.2.1
   sudo yum install -y Ascend-cann-A3-ops-9.2.1
   # Verify after installation
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"

Source code installation
^^^^^^^^^^^^^^^^^^^^^^^^^

We provide a one-click deployment `installation script <https://github.com/verl-project/verl/blob/main/scripts/install_vllm_mcore_npu.sh>`__ based on conda. The script installs the environment step by step. If you encounter an installation error, check the cause based on the error message of the current step. Alternatively, leave us a message by opening an issue, and we will resolve it as soon as possible.

.. code:: bash

   # Note: When installing on an x86 platform, configure an additional source for pip using the following command:
   # pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
   # Enable the CANN environment. If you customized the CANN path, modify the following enable commands based on your custom path.
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   conda create -n verl-vllm-npu python=3.12 -y
   conda activate verl-vllm-npu
   git clone --recursive https://github.com/verl-project/verl.git
   bash verl/scripts/install_vllm_mcore_npu.sh
   # If you only need to use the FSDP backend
   # USE_MEGATRON=0 bash verl/scripts/install_vllm_mcore_npu.sh

Log Filtering
^^^^^^^^^^^^^^^^^^^^^^^^
When using transformers, if a large number of deprecated alias warnings appear, you can add an environment variable to filter redundant logs.

.. code:: bash

   export TRANSFORMERS_VERBOSITY=error

3. Custom installation-SGLang + FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key version support and dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============= ======================================= ===================
Dependency          Version                                     Description
============= ======================================= ===================
HDK           ``25.5.0``                              NPU hardware driver and firmware
CANN          ``>=8.5.0``                             CANN software that helps developers develop and run AI services on Ascend software and hardware platforms
Python        ``>=3.12``\ , ``3.12`` recommended
torch         ``2.8.0``                               PyTorch deep learning framework base package
torch_npu     ``2.8.0.post2``                         NPU PyTorch adaptation plugin
SGLang        ``v0.5.10``                             High-performance LLM inference engine
triton        ``3.5.0``                               Triton, used for writing custom operators
triton-ascend ``3.2.2``                               NPU Triton adaptation. For the installation command, refer to the script `installation script <https://github.com/verl-project/verl/blob/main/scripts/install_sglang_mcore_npu.sh>`__
transformers  ``5.10.4``                               Hugging Face large model library, providing model architectures and pre-trained weights
Megatron-LM   ``core_r0.18.0``                        Large-scale distributed training framework
MindSpeed     ``core_r0.18.0``                        Megatron-LM adaptation and optimization component on Ascend NPU
============= ======================================= ===================

The Ascend Megatron dependency combination is based on ``install_sglang_mcore_npu.sh``: Megatron-LM, MindSpeed, and MegatronAdaptor use ``core_r0.18.0``, and Megatron-Bridge uses ``v0.5.0``; MindSpeed-Bridge uses the default repository branch, and currently does not use a fixed tag or commit. The SGLang installation script and the current CI image build file use the same combination.


Pre-installation preparation (HDK & CANN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CANN is a heterogeneous computing architecture on the NPU. The following are the installation commands for A3 on the ARM platform. Follow these commands to download and install HDK and CANN. Alternatively, download and install them from the `CANN community <https://www.hiascend.com/cann/download?versionId=680&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_ based on your system hardware model.

.. code:: bash

   #Configure the user group
   sudo groupadd HwHiAiUser
   sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
   # Install dependencies and configure the source
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r) 
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # Install the NPU driver
   sudo yum install -y Atlas-A3-hdk-npu-driver-25.5.0
   # Install the Toolkit. You can specify --install-path to customize the path
   sudo yum install -y Ascend-cann-toolkit-8.5.0
   sudo yum install -y Ascend-cann-A3-ops-8.5.0
   # Verify after installation
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"

Source Code Installation
^^^^^^^^^^^^^^^^^^^^^^^^

We provide a one-click deployment `installation script <https://github.com/verl-project/verl/blob/main/scripts/install_sglang_mcore_npu.sh>`__ using conda. The script installs the environment step by step. If you encounter an installation error during the process, check the cause based on the error message of the current step. Alternatively, leave us a message through an issue, and we will resolve it as soon as possible.

.. code:: bash

   # Note: When installing on an x86 platform, configure an additional source for pip. The command is as follows:
   # pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
   # Enable the CANN environment. If you have customized the CANN path, modify the following command to match your custom path.
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   conda create -n verl-sgl-npu python=3.12 -y
   conda activate verl-sgl-npu
   git clone --recursive https://github.com/verl-project/verl.git
   bash verl/scripts/install_sglang_mcore_npu.sh
   # If you only need to use the FSDP backend
   # USE_MEGATRON=0 bash verl/scripts/install_sglang_mcore_npu.sh

Precautions for using SGLang
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To support the SGLang backend on the current NPU, you must add the following environment variables:

.. code:: bash

   # Support multiple processes on a single NPU device
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

   # Work around the issue where Ray cannot identify device availability through the is_npu_available interface on the device side
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

   # Define based on the current device and the required number of devices
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   # in A3
   # export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

   # Required when enabling inference EP
   export SGLANG_DEEPEP_BF16_DISPATCH=1

Appendendix
----------------

Description of ecosystem libraries not currently supported by Ascend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In verl, Ascend currently does not support the following ecosystem libraries:

+------------------+--------------------------------------------------------------------------+
| Software         | Description                                                              |
+==================+==========================================================================+
| ``flash_attn``   | Flash attention acceleration is not supported through a                  |
|                  | standalone ``flash_attn`` package. It is supported through transformers.  |
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
