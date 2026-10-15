# Precision Alignment Guide

When you perform reinforcement learning (RL) training in the VeRL framework, **precision alignment** is a key step to ensure a reproducible and debuggable training process.

This document summarizes the methods for precision alignment between NPUs and GPUs on VeRL for your reference.

Last updated: 05/09/2026.

## 1. Environment and Weight Alignment

### 1.1 Dependency version alignment

You must strictly align the versions of VeRL and transformers; otherwise, the precision results will be directly affected.

If you cannot achieve strong alignment for other key dependencies (torch, megatron, vllm), you must prioritize keeping them consistent or similar.

### 1.2 Model weight alignment

Check whether the model weights and the config.json file are completely consistent.


## 2. Input data alignment

Add the following configuration to the verl training startup script:

```bash
data.shuffle=False
data.validation_shuffle=False
```


## 3. Configuration Alignment

When aligning precision between the NPU and the GPU, check whether the configurations are completely aligned. This includes:
1. Directly compare the configurations written in the script.
2. Save logs during execution, collect the configurations from the log output for comparison, and check whether the default parameter configurations are consistent. Ensure that key parameters are aligned.


## 4. Ensuring Determinism

### 4.1 Fixing the Random Seed

Install `msprobe` in the environment:

```bash
pip install mindstudio-probe
```

Add a deterministic function at the beginning of the worker file:

```python
from msprobe.pytorch import seed_all
seed_all(mode=True)
```

### 4.2 Fixed communication environment variables

In multiple-device communication scenarios:

- Under HCCL communication (default scenario):

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export HCCL_DETERMINISTIC="true"
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

- In LCCL communication (enabled by exporting HCCL_OP_EXPANSION_MODE="AIV"):

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export LCCL_DETERMINISTIC=1
  -  export ATB_LLM_LCOC_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

In a single-device environment without communication:

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0



## 5. Verify training accuracy

### 5.1 Training stubbing

**Stubbing** means retaining the input and output data of the current stage to facilitate comparative analysis of the results. When troubleshooting precision issues, perform stubbing to help locate the problem. A common stubbing method is to directly dump the data from the rollout stage.

**Step 1: Generate baseline data in a GPU environment**

First, run the GPU script once with the following configurations enabled:

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```
You can save the inference results for each step as a JSONL file.

**Step 2: Reproduce and verify in the NPU environment**

Enable the following parameters on the NPU, reuse the sequence generated in the previous step, and run end-to-end:

```bash
skip.rollout.enable=True \
skip.rollout.dump_dir=/path/to/rollout_dump \
```

**Step 3: Compare metrics**

With the same stubbed inference results as inputs, a consistent training configuration, and fixed randomness, compare the rewards/pg_loss/grad_norm values between the NPU and the GPU to check for differences.


## 6. Verifying inference precision

### 6.1 resharding

Before inference officially starts, vllm performs a **dummy run**. It evaluates device memory usage during inference by inferring a single token, and then allocates device memory. You can specify the load_format parameter during LLM initialization in vLLM. This determines whether the weights for the dummy run are randomly initialized (dummy) or real weights (safetensors). In VeRL, you specify this parameter using the **actor_rollout_ref.rollout.load_format** parameter.

When garbled output occurs during inference, if the engine initialization method is **load_format=dummy**, the sharding likely has issues. Even if the text generation becomes normal after switching to safetensors, the sharding still has problems, and you need to compare the forward pass.


### 6.2  Inference result alignment

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```

Save the inference results of each step as a jsonl file. You can directly open the jsonl file to quickly check whether the inference results of the entire network are garbled. This helps localize inference precision issues.


Before dumping inference data, if reproducing inference precision issues consumes significant resources, you can first try to reduce the reproduction cost. Reduce the reproduction scale and the data that needs to be dumped and compared. In multi-batch and long-sequence scenarios, you can attempt to reproduce the issue by sending single-batch requests and reducing the sequence length.


## 7. Dump comparison

[Precision debugging tool](../../../en/dev_guide/precision_analysis/precision_debugger.md). After you locate the stage where the problem occurs, you can use the msprobe tool to dump data for detailed localization.

During inference or training, the model may produce unexpected outputs, generate abnormal content, or encounter numerical instability issues such as NaN/Inf. To locate the root cause, perform fine-grained monitoring of the model execution path. Collect intermediate features, weights, activation values, and the inputs and outputs of key layers. Record context information such as prompts, tensor dtypes, and hardware configurations. By capturing these core tensors and metadata, you can systematically trace the source of precision degradation or numerical errors.




