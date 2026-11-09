Ascend Installation Guide (Ascend 950PR&950DT products)
=======================================================

Last updated: 11/06/2026.


.. contents:: Contents
   :local:
   :depth: 2



Framework Backend Support
-------------------------

Currently, the NPU supports the deployment of the following common training and inference backends on Ascend 950PR&950DT products. You can directly obtain the released images using our :doc:`Ascend image instructions <dockerfile_build_guidance>`, or perform a custom installation using the following sections.

.. list-table::
   :header-rows: 1

   * - Inference engine
     - Training engine
   * - vLLM
     - FSDP/FSDP2/Megatron


Deployment Guide
----------------

1. Getting, building, and using Docker images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can obtain the relevant images from `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_, or build them from a DockerFile. For related instructions, refer to
:doc:`Ascend Image Description <dockerfile_build_guidance>`\ .


2. Custom installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key Version Support and Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
============= ================================================= ===================
Dependency          Version                                               Description
============= ================================================= ===================
CANN          ``9.2.1``                                         CANN software that helps developers develop and run AI applications on Ascend software and hardware platforms
Python        ``3
============= ================================================= ===================

.. note::

   MegatronAdaptor uses ``26.2.0_core_r0.18.0``.
   MindSpeed-Bridge uses ``26.2.0_v0.5.0``.
   TransformerEngineNPU uses ``26.2.0_release_v2.17``.
   MindSpeed-Ops uses ``26.2.0``.

The running dependencies in this table align with the `vLLM-Ascend 0.30.0 installation dependencies <https://github.com/vllm-project/vllm-ascend/blob/releases/v0.30.0/requirements.txt>`_; xgrammar uses the `vLLM 0.30.0 dependency range <https://github.com/vllm-project/vllm/blob/v0.30.0/requirements/common.txt>`_.
   Use the Triton implementation provided by triton-ascend; you do not need to install the triton package separately.
   Currently, a FastAPI constraint conflict exists between the two upstream repositories: vLLM requires ``>=0.133.0,<0.137.0``, and vLLM-Ascend requires ``<0.124.0``,
   No version satisfies both requirements. Resolve this conflict first. Then use ``pip check`` to check the final environment and verify running compatibility.


You also need to install the following Bridge components for the Megatron training backend:

.. list-table::
   :header-rows: 1

   * - Dependency
     - Version
     - Description
   * - Megatron-Bridge
     - ``v0.5.0``
     - Model construction and Hugging Face weight conversion
   * - MindSpeed-Bridge
     - ``26.2.0_v0.5.0``
     - Ascend adaptation of Megatron-Bridge


Preparations Before Installation (CANN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CANN is a heterogeneous computing architecture on the NPU. The following are the installation instructions for the 950PR&DT products on the ARM platform. Follow these instructions to download and install HDK and CANN. Alternatively, download and install them from the `CANN community <https://www.hiascend.com/cann/download>`_ based on your system hardware model.

.. code:: bash


   # Install dependencies and configure the source
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r) 
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # Install the Toolkit. You can specify --install-path to customize the path.
   sudo yum install Ascend-cann-toolkit-9.2.1
   sudo yum install Ascend-cann-950-ops-9.2.1
   # Verify after installation
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"


Source code installation
^^^^^^^^^^^^^^^^^^^^^^^^

vLLM inference backend support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    # Install vLLM
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout v0.30.0
    VLLM_TARGET_DEVICE=empty pip install -v -e .
    cd ..

    # Install vllm-ascend
    # Before installation, source the CANN environment: source /usr/local/Ascend/cann/set_env.sh
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    git checkout releases/v0.30.0
    pip install -v -e . --no-build-isolation --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
    cd ..


Megatron training backend support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    # Download MindSpeed, switch to the specified commit ID, and download Megatron-related dependencies
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed && git checkout 26.2.0_core_r0.18.0 && cd ..
    git clone --depth 1 --branch core_v0.18.2 https://github.com/NVIDIA/Megatron-LM.git
    git clone --depth 1 --branch 26.2.0_core_r0.18.0 https://gitcode.com/ascend/MegatronAdaptor.git
    git clone --depth 1 --branch 26.2.0_release_v2.17 https://gitcode.com/ascend/TransformerEngineNPU.git
    git clone --depth 1 --branch 26.2.0 https://gitcode.com/ascend/MindSpeed-Ops.git
    git clone --depth 1 --branch 26.2.0_v0.5.0 https://gitcode.com/ascend/MindSpeed-Bridge.git
    git clone --depth 1 --branch v0.5.0 https://github.com/NVIDIA-NeMo/Megatron-Bridge.git

    # Install Megatron and Ascend adaptation components
    pip install -e Megatron-LM
    pip install -e MegatronAdaptor
    pip install -e TransformerEngineNPU --no-build-isolation
    pip install -e MindSpeed
    pip install -e MindSpeed-Ops --no-build-isolation --no-deps
    pip install -r MindSpeed-Bridge/requirements.txt
    pip install -e MindSpeed-Bridge --no-deps
    pip install -e Megatron-Bridge --no-build-isolation --no-deps

    # Restore inference dependency versions to avoid overwriting during training dependency installation
    pip install torch_npu==2.10.0.post7 transformers==5.14.1 "xgrammar>=0.2.1,<1.0.0"

Installing verl dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    git clone https://github.com/verl-project/verl.git
    cd verl
    pip install -e .
    pip install -r requirements-npu.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
    # Restore the Triton version required by vLLM
    pip install triton-ascend==3.6.0.post2 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn


Log Filtering
^^^^^^^^^^^^^^^^^^^^^^^^
When using transformers, if a large number of alias deprecation warnings appear, add an environment variable to filter redundant logs.

.. code:: bash

   export TRANSFORMERS_VERBOSITY=error
