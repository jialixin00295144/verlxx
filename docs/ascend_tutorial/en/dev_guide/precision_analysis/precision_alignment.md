# Precision Alignment Guide

When you perform reinforcement learning (RL) training in the VeRL framework, **precision alignment** is critical for ensuring a reproducible and debuggable training process.

This document summarizes the methods for performing precision alignment between NPU and GPU on VeRL for reference.

Last updated: 05/09/2026.

## 1. Environment and Weight Alignment

### 1.1 Dependency version alignment

The versions of VeRL and transformers require strict alignment; otherwise, they directly affect the precision results.

For other key dependencies (torch, megatron, vllm), if strong alignment cannot be achieved, prioritize keeping them consistent or similar.

### 1.2 Model Weight Alignment

Check whether the model weights and the config.json file are completely consistent.


## 2. Input Data Alignment

Add the following configuration to the verl training startup script:

```bash
data.shuffle=False
data.validation_shuffle=False
```


## 3. Configuration Alignment

When performing precision alignment between the NPU and GPU, check whether the configurations are fully aligned. This includes:
1. Directly compare the configurations written by the script.
2. Save logs during execution, collect the configurations from the log output for comparison, compare whether the default parameter configurations are consistent, and ensure that key parameters are aligned.


## 4. Ensuring Determinism

### 4.1 Fixing random seeds

Install `msprobe` in the environment:

```bash
pip install mindstudio-probe
```

Add the deterministic function at the beginning of the worker file:

```python
from msprobe.pytorch import seed_all
seed_all(mode=True)
```

### 4.2 Fixed Communication Environment Variables

In multi-device communication scenarios:

- In HCCL communication (default scenario):

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export HCCL_DETERMINISTIC="true"
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

- In LCCL communication mode (enabled by exporting HCCL_OP_EXPANSION_MODE="AIV"):

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export LCCL_DETERMINISTIC=1
  -  export ATB_LLM_LCOC_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

In a single-device scenario without communication:

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0



## 5. Verifying training accuracy

### 5.1 Training instrumentation

**Stubbing** refers to retaining the input and output data of the current stage to facilitate comparative analysis based on the results. When troubleshooting precision issues, perform stubbing to assist in problem localization. A common stubbing method is to directly dump the data from the rollout phase.

**Step 1: Generate baseline data in a GPU environment**

First, run the GPU script once with the following configuration enabled:

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```
This saves the inference results for each step as a jsonl file.

**Step 2: Reproduce and verify in the NPU environment**

Enable the following parameters on the NPU, reuse the sequence generated in the previous step, and run end-to-end:

```bash
skip.rollout.enable=True \
skip.rollout.dump_dir=/path/to/rollout_dump \
```

**Step 3: Compare metrics**

With identical stubbed inference results, consistent training configuration, and fixed randomness, compare the rewards/pg_loss/grad_norm values between the NPU and the GPU for differences.


## 6. Verifying inference precision

### 6.1 resharding

Before inference officially starts, vLLM performs a **dummy run** to evaluate device memory usage by inferring one token, and then allocates the device memory. During LLM initialization in vLLM, you can specify the load_format parameter to determine whether the dummy run uses randomly initialized weights (dummy) or real weights (safetensors). In VeRL, you specify this parameter using the **actor_rollout_ref.rollout.load_format** parameter.

When garbled inference output occurs, if the engine initialization method is **load_format=dummy**, then sharding is highly likely to have issues. Even if the generated text becomes normal after switching to safetensors, the sharding still has issues, and you need to compare the forward pass.


### 6.2  Inference Result Alignment

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```

Save the inference results of each step as a JSONL file. You can directly open the JSONL file to quickly check whether the full-network inference results are garbled. Use this to isolate inference precision issues.


Before dumping inference data, if reproducing the inference precision issue consumes too many resources, you can first try to reduce the reproduction cost by minimizing the reproduction scale and the data to be dumped and compared. In multi-batch and long-sequence scenarios, you can try to reproduce the issue by sending single-batch requests and reducing the sequence length.


## 7. Dump comparison

For the [precision debugging tool](../../../en/dev_guide/precision_analysis/precision_debugger.md), after you identify the stage where the problem occurs, you can use the msprobe tool to dump data for detailed analysis.

During inference or training, the model may produce unexpected outputs, generate anomalies, or even encounter numerical instability issues such as NaN/Inf. To locate the root cause, perform fine-grained monitoring on the model execution path. Collect intermediate features, weights, activations, and the inputs and outputs of key layers. Record context information such as prompts, tensor dtypes, and hardware configurations. By capturing these core tensors and metadata, you can systematically trace the source of precision degradation or numerical errors.




