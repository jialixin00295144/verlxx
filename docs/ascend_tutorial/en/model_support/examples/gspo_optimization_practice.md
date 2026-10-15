# NPU Qwen3-32B GSPO Optimization Practice

Last updated: 07/03/2026.

To ensure a good experience, switch the verl repository to the main branch with commit ID 9d05508f5e3bd8ecb70cf94ab10dc087b57a716d. Note: The main branch may cause patch issues due to iterative refactoring. If you need a stable version, switch to `release/v0.8.0`.

The training script for this practice is [run_qwen3_32b_fsdp.sh](../../../../../examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh) (`examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh`).

## Operator Adaptation

By elevating the optimization granularity from **token-level** to **sequence-level**, GSPO avoids the **sharp variance increase** that GRPO encounters, which causes training instability. This increases training stability and also improves the convergence speed to a certain extent.

To successfully use the GSPO algorithm in the verl repository, perform the following required configurations

```bash
# Core algorithm configuration  
algorithm.adv_estimator=grpo \                    # Use the GRPO advantage estimator  
algorithm.use_kl_in_reward=False \                # Do not add KL penalty to the reward  
# GSPO policy loss mode  
actor_rollout_ref.actor.policy_loss.loss_mode=gspo \ # Enable GSPO policy loss
# Minimal clipping range (GSPO feature)  
actor_rollout_ref.actor.clip_ratio_low=0.0003 \   # Lower clipping bound, recommended value in the paper  
actor_rollout_ref.actor.clip_ratio_high=0.0004 \  # Upper clipping bound, recommended value in the paper  
# KL configuration (GSPO does not use KL loss)  
actor_rollout_ref.actor.use_kl_loss=False \       # Disable KL loss  
actor_rollout_ref.actor.kl_loss_coef=0.0 \        # Set the KL loss coefficient to 0  
# Sequence-level loss aggregation mode (GSPO core)  
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \ # Sequence-level average, recommended by the GSPO paper  
# Batch configuration  
actor_rollout_ref.rollout.n=16 \                  # Generate 16 responses for each prompt (group sampling)
```

Generally, select `verl.trainer.main_ppo` as the entry function. For a complete runnable sample, see [run_qwen3_32b_fsdp.sh](../../../../../examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh).

## Base Environment

Currently, Atlas 800T A3 and Atlas 900 A3 SuperPoD are supported. You need 4 Atlas 800T A3 servers to complete this best practice.

### Install the basic environment

| software      | version                                                    |
| ------------- | ---------------------------------------------------------- |
| Python        | 3.11                                                       |
| CANN          | ==9.0.0.B160 (CANN900B160)                                 |
| torch         | ==2.9.0                                                    |
| torch_npu     | ==2.9.0                                                    |
| triton_ascend | ==3.2.1                                                    |
| verl          | main                                                       |
| vllm          | v0.18.0                                                    |
| vllm-ascend   | v0.18.0                                                    |
| transformers  | 5.3.0                                                      |


```bash
cd verl
git checkout main
# Specify the corresponding recipe version
git submodule update --init --recursive recipe
```

### Obtaining Weights

Download the corresponding model weights from the Hugging Face library: [Qwen/Qwen3-32B · Hugging Face](https://huggingface.co/Qwen/Qwen3-32B)

### Dataset Preparation

```bash
# Download the math-17k dataset
git clone https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k

# Download the AIME_2024 test dataset
git clone https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
```

### Installing jemalloc

To ensure that Ray processes properly reclaim memory, install and enable the jemalloc library for memory management.

#### Ubuntu operating system

Install jemalloc using the operating system source (Note: requires Ubuntu version >= 20.04):

```shell
sudo apt install libjemalloc2
```

Before starting the task, execute the following command to import jemalloc using environment variables. First, confirm that the file exists by running **find /usr -name libjemalloc.so.2**:

```shell
# arm64 architecture
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
# x86_64 architecture
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

#### OpenEuler Operating System

Run the following command to install jemalloc from the operating system repository.

```shell
yum install jemalloc
```

If the preceding method does not work, you can install jemalloc by compiling from the source code. Go to the jemalloc official website to download the latest stable version. The official website address is: https://github.com/jemalloc/jemalloc/releases/

```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```

Before starting the task, run the following command to import jemalloc through environment variables:

```shell
#Set the environment variable based on the actual installation path. For example, if the installation path is /usr/local/lib/libjemalloc.so.2, you can set the environment variable using the following command. (Run find /usr -name libjemalloc.so.2 to check whether the file exists.)
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
```

### Multi-node task launch

To launch the multi-node task provided in this practice, use the following script.

```bash
pkill -9 python
ray stop --force
rm -rf /tmp/ray

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export TASK_QUEUE_ENABLE=1
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_ASYNC_ERROR_HANDLING=0
export CPU_AFFINITY_CONF=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2

# Change to the path of the use case you need to run
DEFAULT_SH="./run_*.sh"
echo "Use $DEFAULT_SH"

ulimit -n 32768
mkdir logs

NNODES=4
NPUS_PER_NODE=16
# Change to the IP address of the master node
MASTER_ADDR="IP FOR MASTER NODE"
# Change to the communication network interface of the current node
SOCKET_IFNAME="Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
# Get the current IP address
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # Start the master node
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # Check whether device_count equals NNODES
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          bash $DEFAULT_SH
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # The worker node attempts to register with the master node until it succeeds
  while true; do
      # Attempt to connect to the Ray cluster
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # Check whether the connection is successful
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 600
```

DEFAULT_SH: Set it to the path of the configuration sh file used for training. In this case, set it to [run_qwen3_32b_fsdp.sh](../../../../../examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh) (that is, `examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh`).

NNODES and NPUS_PER_NODE: Modify these to use the number of nodes and the number of NPUs per node. In this case, they are 4 and 16, respectively.

MASTER_ADDR: Change it to the IP address of the master node. That is, the MASTER_ADDR on all nodes must be the same.

SOCKET_IFNAME, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME: Change to the corresponding communication network interface. You can obtain the communication network interface using the following command:

```bash
ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1
```

## Performance Tuning

The optimization starts from four aspects: training, inference, scheduling, and others.

### Training

#### Dynamic bsz

```bash
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
```

**This optimization point mainly adjusts the two parameters above. However, note that setting these two parameters too large causes OOM.**

**Mainly adjust** `actor_ppo_max_token_len`. Increasing it reduces the training time. Adjusting `infer_ppo_max_token_len` does not provide obvious benefits, so you can leave it unchanged.

**The functions of these two parameters are as follows:**

**These two parameters control the maximum number of tokens that each GPU processes in dynamic batch size mode.**

- **`actor_ppo_max_token_len`**: The maximum number of tokens each GPU can process during the PPO update (forward and backward propagation) for the Actor model.
- **`infer_ppo_max_token_len`**: The maximum number of tokens each GPU can process when calculating log probabilities during the inference phase (Reference policy and Rollout).

### Inference

#### ACLgraph+FULL_DECODE_ONLY

Optimizations in inference operator dispatch yield an average performance gain of around `15%~20%`.

First, look at enabling **ACLgraph** separately, as follows:

```bash
# Enable ACLgraph+FULL_DECODE_ONLY (Note: When you set this parameter to False, you must set TASK_QUEUE_ENABLE to 1; otherwise, an error occurs)
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes='[8,16,32,64,128]' \
actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode='FULL_DECODE_ONLY'
```

After `FULL_DECODE_ONLY` is successfully enabled, the following output appears:

![FULL_DECODE_ONLY result](https://github.com/wucong25/verl-data/blob/main/ascend_acl_graph.png)

**`cudagraph_capture_sizes` Parameter Configuration Guide**

The value set in `cudagraph_capture_sizes` corresponds to the batch size. This batch size is not the batch size corresponding to the DP domain in the configuration. Instead, it is the batch size relative to vllm, and the unit is **token**.

For reference, the default generated algorithm is as follows.

![cudagraph_capture_sizes](https://github.com/wucong25/verl-data/blob/main/ascend_set_cudagraph_sizes.png)

##### Switching the Inference Backend

Usage: `export VLLM_ATTENTION_BACKEND=XFORMERS`

![VLLM_ATTENTION_BACKEND](https://github.com/wucong25/verl-data/blob/main/ascend_vllm_attn_backend.png)

Note: Some backends are not supported in older versions of vllm-ascend.

##### Enable vllm v1 version

Usage: `export VLLM_USE_V1=1`

You can keep it enabled permanently, as it generally yields positive benefits.

### Scheduling

#### AIV

To enable it, set `export HCCL_OP_EXPANSION_MODE="AIV"`.

The HCCL_OP_EXPANSION_MODE environment variable configures the orchestration expansion location for communication algorithms and supports the following values:

- AI_CPU: Indicates that the communication algorithm is orchestrated and expanded in the AI CPU compute unit on the Device side.
- AIV: Indicates that the communication algorithm is orchestrated and expanded in the Vector Core compute unit on the Device side.
- HOST: Indicates that the communication algorithm is orchestrated and expanded on the Host-side CPU. The Device side automatically selects the corresponding scheduler based on the hardware model.
- HOST_TS: Indicates that the communication algorithm is orchestrated and expanded on the Host-side CPU. The Host dispatches tasks to the Task Scheduler on the Device, and the Task Scheduler on the Device schedules and executes the tasks.

The following describes two expansion mechanisms.

##### Host deployment

<img src="https://github.com/wucong25/verl-data/blob/main/ascend_task_queue1.png" alt="image-20260113194257095" style="zoom:50%;" />

- The software stack runs on the host CPU, and the communication algorithm expands into individual tasks.
- Each task calls the runtime interface and is dispatched to the rtsqueue on the device.
- STARS sequentially fetches tasks from the rtsqueue.
- STARS calls the SDMA and RDMA engines respectively based on the task type.
  **Single operator bottleneck**: The hostbound submission for each task takes 2 to 5 microseconds. A communication operator contains hundreds of tasks. In a single-operator scenario, tasks are not cached on the device. Instead, each task is executed immediately after it is submitted.

##### AICPU Mechanism Details

<img src="https://github.com/wucong25/verl-data/blob/main/ascend_task_queue3.png" alt="image-20260113194333218" style="zoom:50%;" />

- The host side does not issue tasks one by one. It treats communication operators as individual kernels and places them on the communication operator kernel queue.
- STARS schedules the kernels on the kernel queue stream and sends the kernels to the AiCPU for execution.
- The AICPU calls the function (kernel) and uses a thread to execute the kernel function. Inside the function, it expands the communication tasks, places the tasks on the rtsqueue, and STARS calls them.
- This reduces the interaction between the host and the AICPU from hundreds of times to a single time.
- The AICPU submits the tasks and performs partial merging of the submissions.

#### TASK_QUEUE_ENABLE

**Usage:** `export TASK_QUEUE_ENABLE=2`

TASK_QUEUE_ENABLE, dispatch optimization, set to 1 in graph mode (that is, when graph mode is enabled, this must be set to 1), and set to 2 in non-graph mode

Schematic diagram:

![ascend task queue](https://github.com/wucong25/verl-data/blob/main/ascend_task_queue2.png)

##### Core Binding Optimization

**Usage:** `export CPU_AFFINITY_CONF=1`

For detailed configuration principles, refer to https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/ENV/docs/zh/environment_variable_reference/CPU_AFFINITY_CONF.md

### Other

The following content summarizes the tuning configurations for several global environment variables. These parameters often bring positive gains in both the training and inference phases. Currently, no sufficiently fine-grained ablation experiments exist to strictly distinguish their respective contribution proportions to training or inference. Therefore, this section consolidates them for subsequent continuous monitoring and further breakdown analysis.

#### Enable jemalloc

Usage (note that you must install the jemalloc library first): `export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2`

**Installation and usage guide:**[MindSpeed-RL/docs/install_guide.md · Ascend/MindSpeed-RL - AtomGit | GitCode](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/install_guide.md#高性能内存库-jemalloc-安装)

#### Multi-stream multiplexing

Memory optimizations are available.

To enable this feature: `export MULTI_STREAM_MEMORY_REUSE=1`

Principle introduction: https://www.hiascend.com/document/detail/zh/Pytorch/latest/devguide/fwfeatures/docs/zh/framework_feature_guide_pytorch/multistream_memory_reuse.md

#### VLLM_ASCEND_ENABLE_FLASHCOMM

Usage: `export VLLM_ASCEND_ENABLE_FLASHCOMM=1`

Enable the FLASHCOMM high-speed communication optimization technology specific to the Ascend NPU.

Address: https://vllm-ascend.readthedocs.io/zh-cn/latest/user_guide/release_notes.html

#### VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE

Usage: `export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1`

Enable the Ascend NPU dense computation optimization for large model inference.

Address: https://vllm-ascend.readthedocs.io/zh-cn/latest/user_guide/release_notes.html

#### VLLM_ASCEND_ENABLE_PREFETCH_MLP

Usage: `export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1`

Enable the weight prefetch mechanism for MLP layers

<img src="https://github.com/wucong25/verl-data/blob/main/ascend_prefetch.png" alt="image-20251124173132677" style="zoom:50%;" />

### verl framework parameter settings

The following are memory-related configuration switches (note that the optimizations here cause varying degrees of throughput degradation).

```bash
# Gradient Checkpointing
# Effect: Saves device memory by recomputing activations, trading computation for memory. It does not save intermediate activations during forward propagation and recomputes them during backward propagation. This significantly reduces device memory usage and allows for a larger batch size.
actor_rollout_ref.model.enable_gradient_checkpointing=True \

# Parameter Offload
# Effect: Offloads model parameters to CPU memory and loads them back to the GPU during training.
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.ref.fsdp_config.param_offload=True \

# Optimizer Offload
# Effect: Offloads optimizer states (such as Adam momentum) to the CPU. Optimizer states usually occupy a large amount of device memory (for Adam, each parameter requires an additional 8 bytes). Offloading can save device memory.
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \

# Free Cache Engine
# Effect: Releases the KV cache and weights of the inference engine during the training phase. This is the core optimization of 3D-HybridEngine, allowing inference and training to alternate on the same GPU, significantly reducing device memory requirements.
actor_rollout_ref.rollout.free_cache_engine=True \

#  Entropy Computation Optimization
# entropy_checkpointing: Enables recomputation for entropy calculation during training to reduce the device memory peak.
# entropy_from_logits_with_chunking: Processes the logits tensor in chunks (such as 2048 tokens per group) to avoid loading the entire [bsz*seq_len, vocab] tensor at once.
actor_rollout_ref.actor.entropy_checkpointing=True \
actor_rollout_ref.ref.entropy_checkpointing=True \
actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \

# Inference Engine Device Memory Configuration
# gpu_memory_utilization: Controls the proportion of GPU device memory used by vLLM (0.90 = 90%).
# enforce_eager=False: Enables CUDA graphs to accelerate inference, but this occupies additional device memory.
actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
actor_rollout_ref.rollout.enforce_eager=False \
```

## NPU tuning reference articles

For environment variables: [Environment Variable List - TorchNPU - Ascend Community](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/ENV/docs/zh/environment_variable_reference/env_variable_list.md)

Community performance tuning tutorial: [Model Performance Tuning - TorchNPU - Ascend Community](https://www.hiascend.com/document/detail/zh/ModelZoo/traditional_model_train/PyTorch/docs/zh/performance_tuning/performance_overview.md)
