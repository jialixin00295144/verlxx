Multi-Machine Task Startup Operations Guide
===========================================

Last updated: 07/28/2026.

Introduction
----------------------------------

In large-scale model training scenarios, a single machine often cannot meet the computing power requirements, so multi-machine collaborative training is required. verl implements distributed scheduling using the Ray framework. You must correctly start the Ray cluster on multiple nodes and configure the Ascend NPU-related environment variables to successfully launch multi-machine training tasks.

This article helps you understand the following:

1. Prerequisites
2. Multi-machine task startup

Prerequisites
-----------------------------------

1. Environment and network configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before multi-node training, ensure that all nodes meet the following conditions:

- Each node has completed environment setup according to the :doc:`Ascend Installation Guide <../../get_start/install_guidance>`, and key components such as verl, Ray, PyTorch, torch-npu, and CANN have consistent versions
- The training network segments between nodes are interconnected, and the nodes can access Ray ports, Dashboard ports, and the HCCL port range configured later. The ``ping`` command only verifies basic connectivity. If the cluster has a firewall enabled, confirm that TCP ports are not blocked
- The training script paths and model/data/checkpoint paths on each node are consistent (using a shared file system such as NFS is recommended)
- Each node has the NPU driver and CANN software stack installed, and ``npu-smi info`` can properly identify the devices
- Keep the system time of each node synchronized as much as possible to avoid timeline confusion during log and task troubleshooting

2.Obtain the communication network card
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multi-node communication depends on correct network interface configuration. On each node, first check the available network interfaces and their IPv4 addresses:

.. code-block:: bash

  ip -o -4 addr show scope global | awk '{print $2, $4}'

Select the network interface card for multi-machine training communication, and record the network interface card name for each node. If you have determined the primary node IP address, you can also run the following command on each node to view the network interface card used to access the primary node:

.. code-block:: bash

  MASTER_ADDR="IP FOR MASTER NODE"
  ip route get "$MASTER_ADDR" | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") {print $(i + 1); exit}}'

Use this network interface name for the ``HCCL_SOCKET_IFNAME`` and ``GLOO_SOCKET_IFNAME`` in the subsequent configuration and for the ``SOCKET_IFNAME`` in the startup script.

3. Confirm node roles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A multi-node cluster contains one **Master node** and several **Worker nodes**:

- **Master node**: Starts the Ray Head service, handles cluster scheduling, and triggers the training task after all worker nodes join.
- **Worker node**: Registers with the master node, and waits for task scheduling after joining the Ray cluster.

Select one of the nodes as the master node, and record its IP address.

Multi-Machine Task Startup
-----------------------------------

1

Configure the following environment variables on **all nodes**:

.. code-block:: bash

  # Ray log deduplication and detailed error output
  export RAY_DEDUP_LOGS=0
  export HYDRA_FULL_ERROR=1

  # Ascend NPU dispatch optimization: set to 1 for graph mode, set to 2 for non-graph mode
  export TASK_QUEUE_ENABLE=1

  # HCCL communication timeout configuration (unit: seconds), increase appropriately based on the model scale
  export HCCL_ASYNC_ERROR_HANDLING=0
  export HCCL_EXEC_TIMEOUT=3600
  export HCCL_CONNECT_TIMEOUT=3600

  # HCCL port range configuration to avoid port conflicts
  export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
  export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

  # NPU visible device configuration
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

  # Communication network interface configuration. Replace with the actual network interface name of the current node.
  export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"

  # File descriptor limit
  ulimit -n 32768

  # Optional configuration
  # Disable Hugging Face asynchronous weight loading to avoid high host memory peaks during the model loading phase in some environments.
  export HF_DEACTIVATE_ASYNC_LOAD=1

2. Write a multi-machine startup script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can run the following script on all nodes. The script automatically determines the primary or secondary node role based on the current node IP:

.. code-block:: bash

  # Clean up any Ray processes that may remain from the previous training
  pkill -9 python
  ray stop --force
  rm -rf /tmp/ray

  # ====== Configuration that users need to modify ======
  # Training script path
  DEFAULT_SH="./run_*.sh"
  echo "Use $DEFAULT_SH"

  # Number of nodes and NPUs per node
  NNODES=2
  NPUS_PER_NODE=16

  # Master node IP
  MASTER_ADDR="IP FOR MASTER NODE"

  # Communication network interface card of the current node
  SOCKET_IFNAME="Your SOCKET IFNAME"
  # ====== End of configuration ======

  # Get the current node IP
  CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

  if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    # ====== Master node ======
    ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / $NPUS_PER_NODE))

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
    # ====== Worker node ======
    while true; do
        ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

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

**Description of script configuration parameters:**

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``DEFAULT_SH``
     - Path to the configuration shell file used for training, such as ``run_qwen3moe-30b_grpo_megatron_vllm_npu.sh``
   * - ``NNODES``
     - Number of nodes participating in training
   * - ``NPUS_PER_NODE``
     - Number of NPUs per node, such as 16 for Atlas 800T A3
   * - ``MASTER_ADDR``
     - IP address of the master node. This parameter must be the same across all nodes.
   * - ``SOCKET_IFNAME``
     - Name of the communication network interface card on the current node. This may differ across nodes.

3.Start training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the preceding script as ``ray_start.sh`` and run it on **all nodes**:

.. code-block:: bash

  bash ray_start.sh

Recommended execution order:

1. First, start the script on the **master node** and wait for the Ray Head service to be ready.
2. Then, start the script on each **worker node**. The worker nodes automatically register with the master node.
3. After the master node detects that all nodes have joined, it automatically triggers the training task.

4. Monitor training status
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After training starts, you can monitor it using the following methods:

**Ray Dashboard**

Open ``http://<MASTER_ADDR>:8260`` in a browser to view the Ray cluster status, resource usage, and task execution status.

**Command line viewing**

.. code-block:: bash

  ray status

**Training Logs**

The training log output location depends on the training script that ``DEFAULT_SH`` points to. If a log file is configured in the training script, you can use the following command to view it in real time:

.. code-block:: bash

  tail -f <TRAINING_LOG_PATH>
