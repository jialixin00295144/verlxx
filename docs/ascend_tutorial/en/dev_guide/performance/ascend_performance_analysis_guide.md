# Ascend Performance Analysis Guide

Last updated: 02/24/2026.

## Background

Following the release of DeepSeek-R1, reinforcement learning (RL) training for large language models has attracted widespread attention. In the Ascend NPU environment, the verl framework has accumulated extensive performance tuning experience. This guide systematically summarizes a methodology that includes performance data collection and analysis. It aims to help developers use the MindStudio toolchain more efficiently and achieve performance optimization in reinforcement learning scenarios.

### Overview of the Reinforcement Learning Computation Flow

1. **Rollout**: The policy (actor) model performs inference to generate responses (response sequences) based on the input prompt sequence.
2. **ref logprob**: The reference model calculates the ref logprob for KL divergence calculation based on the prompt and the generated response.
3. **logprob**: The actor model calculates the logprob for importance sampling based on the prompt and the generated response.
4. **reward**: The reward model evaluates the reward value R_N based on the prompt and the generated response.
5. **update**: The system calculates the optimization function and policy gradient based on the obtained R_N, ref logprob, and logprob to update the actor model.

![rl_data_stream](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/rl_data_stream.png)

## Enabling the profiling tool

### Enabling method

For the enablement and configuration tutorial, refer to the [Profiling Collection Guide](./ascend_profiling.rst).

## Performance Analysis Methodology

### Overall Performance Overview Analysis

#### 1. Analysis of long-running tasks and resource bubbles

- **Operation**: Use MindStudio Insight to load profiling data, automatically identify different computation stages, and locate long-running tasks and NPU resource bubbles through the pipeline graph on the RL tab
- **Value**: Quickly understand the time consumption proportion of different stages
- **Demonstration**:

![Bubble_analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Bubble_analysis.png)

#### 2. Load balancing analysis

- **Operation**: View MSTX trace data directly through MindStudio Insight to observe the load balancing across different DP ranks during the Rollout phase
- **Value**: Quickly identify load imbalance issues
- **Result display:**

![Load_Balancing_Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Load_Balancing_Analysis.gif)

#### 3. Overall Cluster Performance Analysis

- **Action**: Use the rl_analysis feature of MSTT to generate a cluster Timeline thumbnail and observe the overall duration of each stage.
- **Value**: Gain a macro-level understanding of cluster performance bottlenecks.
- **Operation guide**: [rl_analysis documentation](https://gitcode.com/Ascend/mstt/blob/pre-research/profiler/msprof_analyze/docs/features/rl_analysis.md)
- **Result display**:

![Cluster%20Performance%20Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Cluster%20Performance%20Analysis.png)

### Fine-Grained Analysis

#### Performance Analysis

- **Operation**: You can load profiling data using the Windows or Linux version of MindStudio Insight.
- **Value**: MindStudio Insight supports analyzing task scheduling efficiency, operator execution performance, computing resource utilization, collective communication performance, and so on. Its Timeline view provides task decomposition and overlap analysis functions (**a core feature unique to MindStudio, not available in NV and other competing products, and an essential tool for AI tuning**), and supports interactive analysis using a mouse.
- **Effect display**:

![performance%20analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/performance%20analysis.png)

#### Memory Analysis

##### **Analyzing system memory changes using Profiling combined with call stacks**

- **Operation**: Enable the call stack and memory view functions when you collect data.
- **Value**: Observe the memory allocation and release of the framework and CANN. You can use the call stack to trace the front-end Python code.
- **Effect display**: Analyze memory changes using the call stack. The effect is as follows:

![in-memory%20analytics](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/in-memory%20analytics.gif)

##### **Performing in-depth memory analysis using the msleaks tool**

- **Procedure**: Refer to the [msleaks tool user guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/devaids/msMemScope/docs/zh/user_guide/memory_analysis.md).
- **Value**: You can view the total framework memory allocation line chart and memory block diagram, map them directly to call stacks, and analyze framework memory usage in depth.
- **Result display**:

![msleaks](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/msleaks.gif)

## Performance Analysis Cases

To perform detailed performance analysis, enable **level1** profiling; otherwise, key operator information will be missing.

### 1.host bound diagnosis

Host bound refers to the phenomenon where the total CPU task volume exceeds the NPU, causing bubbles in NPU execution. You can determine this by examining the Host2Device synchronization lines. If the lines are skewed, it indicates that the set signal arrives earlier than the wait signal. The NPU executes as soon as it is ready, which is also device bound:

![host_bound_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_1.png)

If it is confirmed to be host bound, you can enable CPU-side profiling to find the dispatch time of each operator. Note that you need to find the accumulated value of all CPU time rather than a single layer, because the first call takes a long time. For example, GmmSwigluQuant in the following figure takes 1 ms for the first CPU call, and only 200 μs for each subsequent call.

![host_bound_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_2.png)

At this point, some operators handle the majority of the workload while others drag down performance, and the latter outnumber the former. Prioritize **identifying the top operators where the host execution time exceeds the device execution time, as these operators are the bottlenecks**. Assign these operators to the operator team for focused analysis.

### 2. Networking Rationality Analysis

Sometimes, the model topology is not configured in the most efficient way. This is very easy to identify in profiling. The following section introduces the analysis approach and provides examples.

Generally, the major hot operators in large language models (LLMs) are the matrix multiplication computations in Attention and FFN. Together, they may account for over 70% of the computation time during prefill and over 50% during decode. If the overall time proportion does not meet expectations, unfamiliar operators appear in the profiling results, or there are too many concatenation operators, analyze the model architecture to determine whether the operators are used incorrectly. Concatenation operators in particular are worth analyzing one by one.

Concatenation operators such as slice/split/concat and conversion operators such as transpose/cast often exist because the preceding and following operators are not directly compatible. If the preceding operator can directly perform tail processing on the output, it can often save the startup overhead of an operator and a redundant read/write operation. However, such a change may not conform to the basic design principles of operators.

Consider a positive example. A Matmul operation has an output shape of [m, n0 + n1]. Two slice operations follow this Matmul. Both take the [m, n0 + n1] tensor as input, and their outputs are [m, n0] and [m, n1] respectively. The first optimization approach is to replace the two slice operations with a single split operation. This roughly halves the execution time and allows the device memory of the [m, n0 + n1] tensor to be released earlier. A further optimization approach is to split the matrix multiplication weights from [k, n0 + n1] into [k, n0] and [k, n1], and divide the original matrix multiplication task into two. This completely eliminates the slice/split operation. The prerequisite is that the combined execution time of the two tasks does not degrade significantly compared to the original, and that the core partitioning strategy works correctly.

![network_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_1.png)

Consider a counterexample: Rmsnorm(fp16)+Cast(fp16->fp32)+Matmul(fp32). Although the input and output of Rmsnorm are both fp16, it uses fp32 for internal calculations to ensure the precision of accumulation operations. If you fuse the Cast into Rmsnorm, Rmsnorm can omit a final fp32->fp16 cast because it already uses fp32 internally. Together with the removed Cast, this saves two casts in total and avoids one precision loss. Although this seems to benefit both precision and performance, an Rmsnorm that takes fp16 as input and outputs fp32 violates the principle. The principle requires that the core input and output must be of the same data type. Unless you frequently find such a structure in open-source models to prove its universality, the operator team does not allow creating such an operator.

![network_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_2.png)

### 3. Operator Performance Initial Diagnosis

Use `"./ASCEND_PROFILER_OUTPUT/operator_details.csv"` to analyze and determine whether the operator has performance issues.

The Profiling tool collects the average busy time of these pipelines on different cores (xxx_time). It divides this value by the complete kernel duration of the slowest core (task_duration) to obtain the pipeline utilization (xxx_ratio). Although these pipelines have dependencies on each other, and data transfer pipelines compete for bandwidth, operators can overlap each other if they are properly designed. Therefore, you can preliminarily conclude that **when the execution duration of an operator reaches a certain level, the operator should form a bound on one of the pipelines**. That is, the utilization should reach a certain level. Based on experience, when the single operator duration reaches 50 μs, you can consider the operator to be on the bound pipeline, achieving 80%+ occupancy.

Using the following figure as an example, the first row is an FA operator, and the second row is a Matmul operator. The FA operator achieves a utilization rate of 88.1% on the vec pipeline. The Matmul operator achieves a utilization rate of 89.8% on the mac pipeline. Their performance can be considered acceptable.

![Operator%20performance](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Operator%20performance.png)

### 4. Affinity shape adjustment

For a model, hyperparameters are beyond your control. However, you can control factors such as concurrency, weight format, and partitioning strategy to accommodate operators and maximize their performance. This section discusses adjustment directions worth trying on the model side, focusing on operator data movement efficiency and load balancing.

#### 4.1 Data movement efficiency-friendly shapes

mte2 is a pipeline whose efficiency is severely affected by the shape. To ensure maximum data transfer efficiency for mte2, ensure that at least one of the following two conditions is met:

**(1) The transferred matrix uses nz as the format (optimal)
(2) The last axis of the transferred matrix is aligned to 512B and is not an integer multiple of 16KB (near-optimal)**

For weight matrices, during the inference phase, especially decode, we typically satisfy (1), and during the training phase, we typically satisfy (2). **If we cannot achieve (1), we must accommodate (2)**. Typical approaches include:

1. If condition (1) is not met, and the leading axis of the matrix is affinitive while the trailing axis is not affinitive, transpose it.
2. Adjust the TP sharding strategy to avoid non-affinitive trailing axes.

#### 4.2 Load balancing affinity shape

When the operator shape is small, operator semantics may limit core utilization, or even if all cores are enabled, the load balancing may be poor. This section mainly analyzes small shapes in the decode phase.

First, determine the number of cores on the current NPU card. If you are not sure, check the profiling results. If the results show numbers like 20 or 40, the card has 20 cores. Otherwise, the card has 24 cores. Here, the 24 cores actually represent a group consisting of one cube and two vectors. You can consider the cube as the master core and the two vectors as slave cores. If an operator is a pure vector operator, the concept of a group no longer applies. The 40 or 48 vector cores act as master cores and independently fetch logical tasks.

For vector operators in large language models (LLMs), one common core partitioning strategy is to split along the highest dimension, that is, the batch dimension. This strategy is common in operators such as norm and dynamic quantization operators that perform reduction operations on lower dimensions (also called the tail axis). Another strategy is to flatten the entire operator, allowing for very fine-grained splitting, as seen in elementwise operators. For the first type, you can focus on load balancing on the model side. For example, if you use a batch size of 48 but the hardware has 40 vector cores, those 40 cores will loop twice. In the second loop, most cores will have nothing to do, so this batch size can be considered unfriendly. If you set the batch size to 64 or 80, the performance is expected to be lossless. In the same scenario, if the card has 48 cores, this batch size can be considered very friendly.

For cube operators, a common core partitioning strategy is to split M and N using the base block (the K axis is the accumulation axis, and partitioning it introduces determinism issues). The most common block sizes are baseM=128 and baseN=256. In the decode phase, the latency can basically be considered as all spent on moving weights. This is because the M dimension of the activation is extremely small, and the M direction is likely split into only one block, so the right matrix only needs to be moved once. Therefore, within the range of M≤128, you can increase M as much as possible, which is basically lossless for performance. If M is greater than 128, consider (128, 256] as the next performance tier.
In addition to M, the task of splitting the N axis also affects operator affinity. For example, the MLA preprocessing in deepseekR1 uses the same activation (shape [batch_size, 7168]) to perform matrix multiplication with two weights (shapes [7168, 1536] and [7168, 576]). When the batch_size is not large, even if baseN is reduced to 128, the N axis cannot use all the cores. Therefore, the latency of each of these two matrix multiplications is approximately equal to the latency of the matrix multiplication with their weight N axes concatenated (shape [7168, 2112]). If you only consider model competitiveness, we prefer to merge these two weights; otherwise, the bandwidth utilization of the two small matrix multiplications will be very poor.

For the Attention operator, the common core partitioning strategies are q_seqlen, batch_size, and kv_headnum. In the incremental phase, q_seqlen is merged based on the MTP and GQA multiples. However, it usually does not exceed 128, so a second task cannot be partitioned. Therefore, the parallelism is basically batch_size * kv_headnum.

In summary, you can use shape information and operator categories to identify whether an operator has a load balancing problem. Therefore, you can predict the split strategy selection and the batch strategy for maximum throughput.
