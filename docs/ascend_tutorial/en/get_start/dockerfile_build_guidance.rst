Ascend Image Description
===================================

Last updated: 11/09/2026.


Obtaining Images and Public Image Addresses
----------------------------------------------

Ascend hosts the daily-built A2/A3/950 images in `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_, built using the `Dockerfile <https://github.com/verl-project/verl/tree/main/docker/ascend>`_.

Latest build image name format: latest-{inference backend}-{applicable product information}-{operating system}-{other fields}

verl release version image name format: {verl release version number}-{CANN version}-{TorchNPU version}[-{applicable product information}-{operating system}]-{Python version}[-{inference backend}-{other fields}]

The vLLM image for verl v0.9.1 uses CANN 9.1.0, Python 3.12, and vLLM 0.23.0. It supports ``linux/amd64`` and ``linux/arm64``. The corresponding image names are as follows:

* A2: ``quay.io/ascend/verl:v0.9.1-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12-vllm``
* A3: ``quay.io/ascend/verl:v0.9.1-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12-vllm``
* Ascend 950PR&950DT products: ``quay.io/ascend/verl:v0.9.1-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12-vllm``



Image hardware support
-----------------------------------

Atlas 200T A2 Box16

Atlas 900 A2 PoD

Atlas 800T A3

Ascend 950PR&950DT products


Currently, the Megatron build environment for A2/A3/950 vLLM is based on ``scripts/install_vllm_mcore_npu.sh``:
Megatron-LM uses ``core_v0.18.2``, MindSpeed uses ``26.2.0_core_r0.18.0``,
MegatronAdaptor uses ``26.2.0_core_r0.18.0``,
Megatron-Bridge uses ``v0.5.0``; MindSpeed-Bridge uses ``26.2.0_v0.5.0``,
TransformerEngineNPU uses ``26.2.0_release_v2.17``; MindSpeed-Ops uses ``26.2.0``.

The following table lists the target versions of the latest Dockerfile on this branch; the images need to be rebuilt and verified.
The ``_v0.9.1`` build entry and the corresponding image tags retain the original dependencies and install the verl ``release/v0.9.1`` source code; the existing published images and historical release versions remain unchanged.

The current A2/A3/950 vLLM latest Dockerfile uses ``requirements-npu.txt`` from the cloned repository and installs ``triton-ascend==3.6.0.post2`` after installing verl.
For example, you can run the build in the repository root directory: ``docker build -f docker/ascend/Dockerfile.ascend_9.2.1_a2 .``

Current target component versions for the vLLM backend Dockerfile
-----------------------------------------------------------------

================== ==================== ======================================
Component               A2/A3 version            Ascend 950PR&950DT products
================== ==================== ======================================
Base image       Ubuntu 22.04         Ubuntu 22.04
Python             3.12                 3.12
CANN               9.2.1                9.2.1
torch              2.10.0               2.10.0
torch_npu          2.10.0.post7         2.10.0.post7
torchvision        0.25.0               0.25.0
torchaudio         2.10.0               2.10.0
transformers       5.14.1               5.14.1
xgrammar           >=0.2.1,<1.0.0       >=0.2.1,<1.0.0
vLLM               0.30.0               0.30.0
vLLM-ascend        0.30.0               0.30.0
Megatron-LM        core_v0.18.2         core_v0.18.2
MindSpeed          26.2.0_core_r0.18.0  26.2.0_core_r0.18.0
Megatron-Bridge    0.5.0                0.5.0
triton-ascend      3.6.0.post2          3.6.0.post2
================== ==================== ======================================


The A2/A3 SGLang image uses CANN ``8.5.0``, Python ``3.11``, torch ``2.8.0``,
torch_npu ``2.8.0.post2``, SGLang ``v0.5.10``, and sgl-kernel-npu ``2026.02.01``.
The training components of this SGLang image use the original Dockerfile and are not included in this upgrade; transformers uses ``5.10.4``.




.. _ascend-dockerfile-list:

Dockerfile image build script list
----------------------------------

**General image**

============== ==================== ============== ==============================================================
Device type         CANN base image version     Inference backend        Reference file
============== ==================== ============== ==============================================================
A2              9.2.1                  vLLM            `Dockerfile.ascend_9.2.1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.2.1_a2>`_
A3              9.2.1                  vLLM            `Dockerfile.ascend_9.2.1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.2.1_a3>`_
950PR&950DT     9.2.1                  vLLM            `Dockerfile.ascend_9.2.1_a5 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.2.1_a5>`_
A2              9.1.0                  vLLM            `Dockerfile.ascend_9.1.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a2>`_
A3              9.1.0                  vLLM            `Dockerfile.ascend_9.1.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a3>`_
950PR&950DT     9.1.0                  vLLM            `Dockerfile.ascend_9.1.0_a5 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a5>`_
A2              8.5.0                  vLLM            `Dockerfile.ascend_8.5.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a2>`_
A3              8.5.0                  vLLM            `Dockerfile.ascend_8.5.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a3>`_
A2              8.5.0                  SGLang          `Dockerfile.ascend.sglang_8.5.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a2>`_
A3              8.5.0                  SGLang          `Dockerfile.ascend.sglang_8.5.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a3>`_
A2              8.3.RC1                vLLM            `Dockerfile.ascend_8.3.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a2>`_
A3              8.3.RC1                vLLM            `Dockerfile.ascend_8.3.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a3>`_
A2              8.3.RC1                SGLang          `Dockerfile.ascend.sglang_8.3.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a2>`_
A3              8.3.RC1                SGLang          `Dockerfile.ascend.sglang_8.3.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a3>`_
A2              8.2.RC1                vLLM            `Dockerfile.ascend_8.2.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a2>`_
A3              8.2.RC1                vLLM            `Dockerfile.ascend_8.2.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a3>`_
============== ==================== ============== ==============================================================


**verl release version image**

============== ==================== ============== ============== ==============================================================
Device type         CANN base image version     Inference backend        verl version       Reference file
============== ==================== ============== ============== ==============================================================
A2              9.1.0                vLLM          release/v0.9.1 `Dockerfile.ascend_9.1.0_a2_v0.9.1 <https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a2_v0.9.1>`_
A3              9.1.0                vLLM          release/v0.9.1 `Dockerfile.ascend_9.1.0_a3_v0.9.1 <https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a3_v0.9.1>`_
950PR&950DT     9.1.0                vLLM          release/v0.9.1 `Dockerfile.ascend_9.1.0_a5_v0.9.1 <https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a5_v0.9.1>`_
A2              9.0.0                vLLM          release/v0.8.0 `Dockerfile.ascend_9.0.0_a2_v0.8.0 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2_v0.8.0>`_     
A3              9.0.0                vLLM          release/v0.8.0 `Dockerfile.ascend_9.0.0_a3_v0.8.0 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3_v0.8.0>`_ 
A2              8.5.0                vLLM          release/v0.7.1 `Dockerfile.ascend_8.5.0_a2_v0.7.1 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a2_v0.7.1>`_     
A3              8.5.0                vLLM          release/v0.7.1 `Dockerfile.ascend_8.5.0_a3_v0.7.1 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a3_v0.7.1>`_ 
============== ==================== ============== ============== ==============================================================


**Custom Model Image**

============== ==================== ============== ============== ==============================================================
Device type         CANN base image version     Inference backend        Model           Reference file
============== ==================== ============== ============== ==============================================================
A2              8.5.2                vLLM          Qwen3.5        `Dockerfile.ascend_8.5.2_a2_qwen3-5 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.2_a2_qwen3-5>`_   
A3              8.5.2                vLLM          Qwen3.5        `Dockerfile.ascend_8.5.2_a3_qwen3-5 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.2_a3_qwen3-5>`_ 
============== ==================== ============== ============== ==============================================================



**Description:**

* In the image with the ``vLLM`` inference backend, vLLM, vLLM-ascend, MindSpeed, Megatron-LM, and verl are installed from source code. The source code is located in the root directory ``/`` of the image.
* In the image with the ``SGLang`` inference backend, SGLang, MindSpeed, and verl are installed from source code. The source code is located in the root directory ``/`` of the image.


Image build command example
---------------------------

.. code:: bash

   # Build the latest vLLM image from the repository root directory
   cd {verl-root-path}
   docker build -f docker/ascend/Dockerfile.ascend_9.2.1_a2 -t verl-ascend:9.2.1-a2 .

   # Release and SGLang images retain the original build context
   cd {verl-root-path}/docker/ascend
   # vLLM (verl release/v0.9.1)
   docker build -f Dockerfile.ascend_9.1.0_a2_v0.9.1 -t verl-ascend:v0.9.1-9.1.0-a2 .
   # SGLang
   docker build -f Dockerfile.ascend.sglang_8.5.0_a2 -t verl-ascend-sglang:8.5.0-a2 .

   # Query local images after build
   docker images

**Description:**

* For example, for the vLLM image, ``Dockerfile.ascend_9.2.1_a2`` is the Dockerfile filename. In ``verl-ascend:9.2.1-a2``, verl-ascend is the custom image name, and 9.2.1-a2 is the custom image tag.
* When building the verl v0.9.1 image, use the Dockerfile with the ``_v0.9.1`` suffix. For A3 and 950PR&950DT, use the corresponding ``a3`` and ``a5`` files in the list, respectively.

Container startup command template
-----------------------------------

.. code:: bash

   docker run -dit \
       --ipc=host \
       --network host \
       --name {your_docker_name} \
       --privileged \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
       -v /usr/local/sbin:/usr/local/sbin \
       -v /usr/sbin:/usr/sbin \
       -v /home:/home \
       -v /data:/data \
       {image_name}:{tag} \
       /bin/bash

**Description:**

* If you need to mount other local paths to the container, add ``-v <host machine path>:<container path>`` manually.
* We recommend replacing ``{your_docker_name}`` with a meaningful container name.
* The ``--privileged`` parameter grants extended permissions to the container; evaluate whether it is necessary based on your security requirements.
* Replace ``{image_name}:{tag}`` with the image name and tag used when building the container.

Starting the container
----------------------

.. code:: bash

   docker start {your_docker_name}

Enter a running container
-------------------------

.. code:: bash

   docker exec -it {your_docker_name} bash


Disclaimer
--------------------
The Ascend-related Dockerfiles and images provided in verl are reference samples that you can use for trial purposes. If you want to use them in a production environment, contact us through official channels. Thank you.
