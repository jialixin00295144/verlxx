Ascend Installation Guide (Ascend 950PR & 950DT products)
=========================================================

Last updated: 09/22/2026.


Directory
---------

- `Framework Backend Support <#framework-backend-support>`_
- `Deployment Guide <#deployment-guide>`_
   - `Obtaining, Building, and Using Docker Images <#1-obtaining-building-and-using-docker-images>`_
   - `Custom Installation <#2-custom-installation>`_



Framework Backend Support
-------------------------

Currently, the NPU supports the deployment of the following common training and inference backends on Ascend 950PR&950DT series products. You can directly obtain the released images using our `Ascend Image Guide <dockerfile_build_guidance.rst>`__, or perform a custom installation according to the following text.

.. list-table::
   :header-rows: 1

   * - Inference engine
     - Training engine
   * - vLLM
     - FSDP/FSDP2/Megatron


Deployment Guide
----------------

1. Obtaining, building, and using Docker images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can obtain the image from `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_, or build it from a Dockerfile. For related instructions, refer to
`Ascend Image Instructions <dockerfile_build_guidance.rst>`__\ .


2. Custom installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key version support and dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
============= ================================================= ===================
Dependency    Version                                            Description                                               
============= ================================================= ===================
CANN          ``9.1.0``                                         CANN software for developing and running AI applications on Ascend software and hardware platforms 
Python        ``3.11``                                          Python version                                             
torch         ``2.10.0``                                        PyTorch deep learning framework base package               
torch_npu     ``2.10.0.post4``                                  NPU PyTorch adaptation plugin                              
triton        ``3.5.0``                                         Triton, used to write custom operators                     
triton-ascend ``3.2.2``                                         NPU Triton adaptation                                      
transformers  ``5.10.4``                                        Hugging Face large model library that provides model architectures and pre-trained weights 
vLLM          ``0.23.0``                                        High-performance LLM inference and serving engine          
vLLM-Ascend   ``0.23.0``                                        NPU vLLM backend adaptation                                
Megatron-LM   ``core_r0.12.0``                                  Large-scale distributed training framework                 
MindSpeed     ``0c6c0ceaa523a96032dee1539a52032155e6404e``      Adaptation and optimization component for Megatron-LM on the Ascend NPU 
============= ================================================= ===================


For the Megatron training backend, you also need to install the following Bridge components:

.. list-table::
   :header-rows: 1

   * - Dependency
     - Version
     - Description
   * - Megatron-Bridge
     - ``v0.5.0``
     - Model construction and Hugging Face weight conversion
   * - MindSpeed-Bridge
     - Repository default branch (no fixed tag/commit)
     - Ascend adaptation for Megatron-Bridge


Pre-installation preparation (CANN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CANN is a heterogeneous computing architecture on the NPU. The following are the installation instructions for the arm platform A3. Refer to the following instructions to download and install HDK and CANN,
or download and install from the `CANN community <https://www.hiascend.com/cann/download?versionId=723&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_ based on your system hardware model.

.. code:: bash


   # Install dependencies and configure the source
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r) 
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # Install the Toolkit. You can specify --install-path to customize the path.
   sudo yum install Ascend-cann-toolkit-9.1.0
   sudo yum install Ascend-cann-950-ops-9.1.0
   # Verify the installation
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"


Source code installation
^^^^^^^^^^^^^^^^^^^^^^^^

vLLM inference backend support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    # Install vllm
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout v0.23.0
    VLLM_TARGET_DEVICE=empty pip install -v -e .
    cd ..

    # Install vllm-ascend
    # Before installation, source the CANN environment: source /usr/local/Ascend/cann/set_env.sh
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    git checkout releases/v0.23.0
    pip install -v -e . --no-build-isolation --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
    cd ..


Megatron Training Backend Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    # MindSpeed
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed
    git checkout 0c6c0ceaa523a96032dee1539a52032155e6404e
    pip install -e .
    cd ..

    # Megatron
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.12.0
    pip install -e .
    cd ..

    # Configure environment variables
    export PYTHONPATH=$PYTHONPATH:your_path/Megatron-LM
    export PYTHONPATH=$PYTHONPATH:your_path/MindSpeed

    # Install Megatron-Bridge
    git clone --depth 1 --branch v0.5.0 https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
    pip install -e Megatron-Bridge --no-build-isolation --no-deps

    # Install MindSpeed-Bridge (repository default branch)
    git clone --depth 1 https://gitcode.com/ascend/MindSpeed-Bridge.git
    pip install -e MindSpeed-Bridge --no-deps

    # Install transformers
    pip install transformers==5.10.4

Installing verl Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    git clone https://github.com/verl-project/verl.git
    cd verl
    pip install -e .
    pip install -r requirements-npu.txt


Log filtering
^^^^^^^^^^^^^^^^^^^^^^^^
After you upgrade transformers to version 5.10.4, the system may display a large number of deprecated alias warnings. You can add an environment variable to filter redundant logs.

.. code:: bash

   export TRANSFORMERS_VERBOSITY=error
