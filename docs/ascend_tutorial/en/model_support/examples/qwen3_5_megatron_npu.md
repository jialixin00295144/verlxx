# Qwen3.5 Megatron NPU User Guide

Last updated: 11/03/2026.

This guide provides instructions for running the Qwen3.5-35B-A3B and Qwen3.5-122B-A10B GRPO samples on Ascend NPUs using verl + Megatron + vLLM.

## Version Requirements

Use the dependency combination in the [vLLM NPU installation script](../../../../../scripts/install_vllm_mcore_npu.sh):

| Component | Version |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch / torch_npu | 2.10.0 / 2.10.0.post7 |
| vLLM / vLLM-Ascend | 0.30.0 / 0.30.0 |
| triton-ascend | 3.6.0.post2 |
| transformers | 5.14.1 |
| xgrammar | `>=0.2.1,<1.0.0` |
| Megatron-LM | `core_v0.18.0` |
| MindSpeed / MegatronAdaptor | `core_r0.18.0` |
| Megatron-Bridge | `v0.5.0` |
| MindSpeed-Bridge | Repository default branch (tag/commit not pinned) |

The installation script installs both TransformerEngineNPU and MindSpeed-Ops. You can install them by following the
[Ascend Installation Guide](../../get_start/install_guidance.rst), or use an A3 vLLM image that matches this combination.

## Models and Scripts

| model             | HF model | script |
|-------------------| --- | --- |
| Qwen3.5-35B-A3B   | `Qwen/Qwen3.5-35B-A3B` | `examples/grpo_trainer/run_qwen3_5_35b_megatron.sh` |
| Qwen3.5-122B-A10B | `Qwen/Qwen3.5-122B-A10B` | `examples/grpo_trainer/run_qwen3_5_122b_a10b_megatron.sh` |
| Qwen3.5-397B-A17B | `Qwen/Qwen3.5-397B-A17B` | `examples/grpo_trainer/run_qwen3_5_397b_megatron.sh` |

## Hardware and parallel configuration

The sample script uses the following NPU configuration by default. You can override it using environment variables with the same name:

| model | nnodes | devices per node | TP | PP | CP | EP | ETP | GEN_DP | GEN_TP | GEN_EP |
| --- |--------| --- | --- | --- | --- |----| --- |---|----|----|
| Qwen3.5-35B-A3B | 1 | 16 | 2 | 2 | 1 | 8  | 1 | 1 | 8 | 1 |
| Qwen3.5-122B-A10B | 4 | 16 | 2 | 4 | 1 | 16 | 1 | 1 | 16 | 1 |
| Qwen3.5-397B-A17B | 16 | 16 | 2 | 4 | 1 | 64 | 1 | 16 | 16 | 256 |

## Data and Model Preparation

The script uses the Geo3K dataset by default and downloads it to `$HOME/data/geo3k`:

```bash
hf download tyzhu/geo3k --repo-type dataset --local-dir $HOME/data/geo3k
```

You can use a Hugging Face model name for the model weights, or download them to a local path in advance:

```bash
hf download Qwen/Qwen3.5-35B-A3B --local-dir /path/to/Qwen3.5-35B-A3B
hf download Qwen/Qwen3.5-122B-A10B --local-dir /path/to/Qwen3.5-122B-A10B
hf download Qwen/Qwen3.5-397B-A17B --local-dir /path/to/Qwen3.5-397B-A17B
```

## Start training

Start the Ray cluster before training. For general multi-node instructions, refer to [Multinode Training](../../../../start/multinode.rst). For Ascend multi-node script examples, refer to [Ascend SGLang Best Practices](ascend_sglang_best_practices.rst).

The minimal startup method is as follows. For single-machine tasks, you only need to run the head node command. For multi-machine tasks, run the worker node command on the other nodes. Keep `MASTER_ADDR` consistent across all nodes, and set `CURRENT_IP` to the IP address of the current node.

```bash
MASTER_ADDR=<head-node-ip>
CURRENT_IP=<current-node-ip>
NPUS_PER_NODE=16

# head node
ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

# worker nodes, only needed for multi-node jobs
ray start --address="$MASTER_ADDR:6766" --node-ip-address=$CURRENT_IP --resources='{"NPU": '$NPUS_PER_NODE'}'

ray status
```

After you confirm that the number of NPU resources meets expectations using `ray status`, run the training script on the master node. By default, Qwen3.5-35B-A3B requires 16 NPU resources, and Qwen3.5-122B-A10B requires 64 NPU resources.

### Qwen3.5-35B-A3B

```bash
export DEVICE=npu
export HF_MODEL_PATH=/path/to/Qwen3.5-35B-A3B

bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
```

If you need to override the data path:

```bash
DEVICE=npu \
HF_MODEL_PATH=/path/to/Qwen3.5-35B-A3B \
train_path=/path/to/train.parquet \
test_path=/path/to/test.parquet \
bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
```

### Qwen3.5-122B-A10B

```bash
export DEVICE=npu
export HF_MODEL_PATH=/path/to/Qwen3.5-122B-A10B

bash examples/grpo_trainer/run_qwen3_5_122b_a10b_megatron.sh
```

If you need to override the data, save path, or parallel configuration:

```bash
DEVICE=npu \
HF_MODEL_PATH=/path/to/Qwen3.5-122B-A10B \
train_files=/path/to/train.parquet \
test_files=/path/to/test.parquet \
save_path=/path/to/checkpoints \
NDEVICES_PER_NODE=16 \
nnodes=4 \
bash examples/grpo_trainer/run_qwen3_5_122b_a10b_megatron.sh
```

### Qwen3.5-397B-A17B

```bash
export DEVICE=npu
export HF_MODEL_PATH=/path/to/Qwen3.5-397B-A17B

bash examples/grpo_trainer/run_qwen3_5_397b_megatron.sh
```

If you need to override the data, save path, or parallel configuration:

```bash
DEVICE=npu \
HF_MODEL_PATH=/path/to/Qwen3.5-397B-A17B \
train_files=/path/to/train.parquet \
test_files=/path/to/test.parquet \
save_path=/path/to/checkpoints \
NDEVICES_PER_NODE=16 \
nnodes=16 \
bash examples/grpo_trainer/run_qwen3_5_397b_megatron.sh
```

## Precautions

- The script automatically recognizes the NPU environment using `torch_npu`; to specify it manually, set `DEVICE=npu`.
- The Qwen3.5 Gated Delta Net does not currently use packed sequence, so the script maintains `use_remove_padding=False` and `use_dynamic_bsz=False`.
- The NPU branch sets Ascend adaptation parameters such as `use_flash_attn=True` and `moe_token_dispatcher_type=alltoall`.
