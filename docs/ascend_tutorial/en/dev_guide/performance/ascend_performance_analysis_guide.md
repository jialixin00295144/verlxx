# Ascend performance analysis guide

Last updated: 02/24/2026.

## Background

With the release of DeepSeek-R1, large language model reinforcement learning (RL) training has attracted widespread attention. In the Ascend NPU environment, the verl framework has gained extensive experience in performance tuning. This document systematically summarizes a methodology that includes performance data collection and analysis. It aims to help developers use the MindStudio toolchain more efficiently to achieve performance optimization in reinforcement learning scenarios.

### Overview of the reinforcement learning computation flow

1. **Rollout**: The policy (actor) model generates responses (response sequences) through inference based on the input prompt sequence.
2. **ref logprob**: Based on the prompt and the generated response, the reference model computes the ref logprob for KL divergence computation.
3. **logprob**: Based on the prompt and the generated response, the actor model computes the logprob for importance sampling.
4. **reward**: Based on the prompt and the generated response, the reward model evaluates the reward value R_N.
5. **update**: Based on the computed R_N, ref logprob, and logprob, the system computes the optimization function and policy gradient to update the actor model.

![rl_data_stream](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/rl_data_stream.png)

## Enabling the profiling tool

### Enabling Method

For enabling and configuration tutorials, refer to the [Profiling Collection Guide](./ascend_profiling.rst).

## Performance Analysis Methodology

### Overall Performance Overview Analysis

#### 1. Long-running task and resource bubble analysis

- **Operation**: Use MindStudio Insight to load profiling data, automatically identify different computation stages, and locate long-running tasks and NPU resource bubbles through the pipeline graph in the RL tab.
- **Value**: Quickly understand the time consumption proportion of different stages.
- **Effect demonstration**:

![Bubble_analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Bubble_analysis.png)

#### 2. Load Balancing Analysis

- **Operation**: View MSTX trace data directly through MindStudio Insight to observe the load balancing status across different DP Ranks during the Rollout phase
- **Value**: Quickly identify load imbalance issues
- **Effect display:**

![Load_Balancing_Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Load_Balancing_Analysis.gif)

#### 3. Overall cluster performance analysis

- **Operation**: Use the rl_analysis feature of MSTT to generate a cluster Timeline thumbnail and observe the overall duration of each stage
- **Value**: Gain a macro-level understanding of cluster performance bottlenecks
- **Operation guide**: [rl_analysis documentation](https://gitcode.com/Ascend/mstt/blob/pre-research/profiler/msprof_analyze/docs/features/rl_analysis.md)
- **Effect display**:

![Cluster%20Performance%20Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Cluster%20Performance%20Analysis.png)

### Fine-grained analysis

#### Performance Analysis

- **Operation**: You can load profiling data using the Windows or Linux version of MindStudio Insight.
- **Value**: MindStudio Insight supports analyzing task scheduling efficiency, operator execution performance, computing resource utilization, collective communication performance, and so on. Its Timeline view provides task decomposition and overlap analysis capabilities (**a core feature unique to MindStudio, not available in NV and other competing products, and an essential tool for AI tuning**), and supports interactive analysis using a mouse.
- **Result display**:

![performance%20analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/performance%20analysis.png)

#### Memory Analysis

##### **Analyzing system memory changes by combining Profiling with call stacks**

- **Operation**: Enable the call stack and memory view functions when collecting data.
- **Value**: Observe the memory allocation and release of the framework and CANN. You can use the call stack to trace back to the frontend Python code.
- **Effect display**: Analyze memory changes using the call stack. The result is as follows:

![in-memory%20analytics](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/in-memory%20analytics.gif)

##### **Using the msleaks tool for in-depth memory analysis**

- **Procedure**: Refer to the [msleaks tool usage guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/devaids/msMemScope/docs/zh/user_guide/memory_analysis.md).
- **Benefits**: You can view the line chart of total framework memory allocations and memory block diagrams. These map directly to call stacks, enabling deep analysis of framework memory usage.
- **Demonstration**:

![msleaks](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/msleaks.gif)

## Performance Analysis Cases

To perform detailed performance analysis, enable **level1** profiling. Otherwise, key information about operators will be missing.

### 1.host bound diagnosis

"Host bound" refers to the phenomenon where the total CPU workload exceeds the NPU workload, causing bubbles in NPU execution. You can determine this by checking the Host2Device synchronization lines. If the lines are skewed, it indicates that the set signal arrives earlier than the wait signal. The NPU executes as soon as it is ready. This is also device bound:

![host_bound_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_1.png)

If it is confirmed as host bound, you can enable CPU-side profiling to find the dispatch time of each operator. Note that you need to find the accumulated value of all CPU time rather than a single layer, because the first call takes a long time. For example, for GmmSwigluQuant in the following figure, the first call on the CPU takes 1 ms, and each subsequent call takes only 200 μs.

![host_bound_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_2.png)

At this point, some operators carry a heavy load, while others hold back performance, and the latter outnumber the former. Prioritize **identifying the top operators where the host execution time is greater than the device execution time, as these operators are holding back performance**. You can assign them to the operator team for focused analysis.

### 2. Networking Rationality Analysis

Sometimes, the model topology is not structured in the most efficient way. This is easy to identify in profiling. The following sections introduce the analysis approach and provide examples.

Typically, the major hot operators in large language models (LLMs) are the matrix multiplication operations in Attention and FFN. Together, they can account for over 70% of the computation time during prefill and over 50% during decode. If the overall time proportion does not meet expectations, new operators appear in the profiling results, or there are too many concatenation operators, analyze the model architecture. Determine whether the operators are used incorrectly. Concatenation operators in particular are worth analyzing one by one.

Concatenation operators like slice/split/concat and conversion operators like transpose/cast often exist because the preceding and following operators do not match directly. If the preceding operator directly performs post-processing on the output, it often saves the startup overhead of an operator and a redundant read/write. However, this change does not necessarily conform to the basic design principles of operators.

Here is a positive example. For a Matmul operation with an output shape of [m, n0 + n1], we connect two slice operations after it. Both take the [m, n0 + n1] tensor as input, producing outputs of [m, n0] and [m, n1] respectively. The first optimization is to replace the two slice operations with a single split operation. This roughly halves the execution time, and the device memory for [m, n0 + n1] can be released earlier. A further optimization is to split the matrix multiplication weight from [k, n0 + n1] into [k, n0] and [k, n1], dividing the original matrix multiplication task into two. This is viable provided that the combined execution time of the two does not degrade significantly compared to the original, and the core partitioning strategy works correctly. This completely eliminates the slice/split operation.

![network_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_1.png)

Consider a counterexample: Rmsnorm(fp16)+Cast(fp16->fp32)+Matmul(fp32). Although both the input and output of Rmsnorm are fp16, it uses fp32 for internal calculations to ensure the precision of accumulation operations. If you fuse the Cast into Rmsnorm, Rmsnorm, which already uses fp32 for internal calculations, can omit a final fp32->fp16 cast. Together with the eliminated Cast, this saves two casts in total and avoids one precision loss. Although this seems to achieve both precision and performance, an Rmsnorm with fp16 input and fp32 output violates the principle (the core input and output must be the same data type). Unless you frequently find such structures in open-source models to prove their universality, the operator team does not permit creating such an operator.

![network_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_2.png)

### 3. Preliminary diagnosis of operator performance

Use `"./ASCEND_PROFILER_OUTPUT/operator_details.csv"` to analyze and determine whether the operator has performance issues.

The Profiling tool calculates the average busy time (xxx_time) of these pipelines on different cores. It divides this time by the complete kernel duration (task_duration) of the slowest core to obtain the pipeline utilization (xxx_ratio). Although these pipelines depend on each other and data movement pipelines compete for bandwidth, they can overlap if the operator is designed properly. Therefore, you can preliminarily conclude that **when an operator's execution duration reaches a certain level, it should form a bound on a specific pipeline**. That is, the utilization should reach a certain high level. Empirically, when a single operator takes 50 μs or longer, it should be on the bound pipeline, achieving 80% or higher utilization.

Take the following figure as an example. The first row is an FA operator, and the second row is a Matmul operator. The FA operator achieves 88.1% utilization on the vec pipeline, and the Matmul operator achieves 89.8% utilization on the mac pipeline. Their performance can be considered acceptable.

![Operator%20performance](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Operator%20performance.png)

### 4. Affinity shape adjustment

For a model, hyperparameters are beyond your control. However, you can control factors such as concurrency, weight format, and sharding strategy to accommodate operators and achieve maximum performance. This section discusses adjustment directions worth trying on the model side, focusing on operator transfer efficiency and load balancing.

#### 4.1 Data movement efficiency-friendly shapes

mte2 is a pipeline whose efficiency is severely affected by shape. To ensure that mte2 achieves maximum transfer efficiency, ensure that at least one of the following two conditions is met:

**(1) The matrix being moved uses nz as the format (Optimal)
(2) The tail axis of the matrix being moved is 512B-aligned and is not a multiple of 16KB (Near-optimal)**

For weight matrices, during the inference phase, especially decode, we typically satisfy (1), and during the training phase, we typically satisfy (2). **If we cannot achieve (1), we must accommodate (2)**. Typical approaches include:

1. If condition (1) is not met, and the leading axis of the matrix is affinitive while the trailing axis is not, transpose it.
2. Adjust the TP sharding strategy to avoid non-affinitive trailing axes.

#### 4.2 Load-balancing-friendly shape

When the operator shape is small, operator semantics may prevent all cores from being utilized. Even if all cores are enabled, load balancing can be poor. This section mainly analyzes small shapes in the decode phase.

First, determine the number of cores on the current NPU card. If you are unsure, check the profiling results. If the results show numbers like 20 or 40, the card has 20 cores. Otherwise, it has 24 cores. The 24 cores actually represent a group consisting of one cube and two vectors. You can consider the cube as the primary core and the two vectors as secondary cores. If an operator is a pure vector operator, the concept of a group no longer applies. The 40 or 48 vector cores act as primary cores and independently fetch logical tasks.

For vector operators in a large language model (LLM), one common core partitioning strategy partitions along the highest dimension, that is, the batch dimension. This strategy is common in operators such as norm and dynamic quantization that perform reduction operations on low dimensions (also known as the tail axis). Another strategy is to flatten the entire operator, allowing it to be split very finely. Elementwise operators use this strategy. For the first type, you can focus on the load balancing issue on the model side. For example, if you use a batch size of 48, but the hardware has 40 vector cores, these 40 cores will loop twice. In the second loop, most cores will have nothing to do. This batch size can be considered unfriendly. If you set the batch size to 64 or 80, the performance can be expected to be lossless. In the same situation, if the card has 48 cores, you can consider this a very friendly batch size.

For cube operators, a common core splitting strategy is to split the M and N axes using the base block. The K axis is the accumulation axis, and splitting it introduces determinism issues. The most common block sizes are baseM=128 and baseN=256. During the decode phase, the process primarily spends time moving weights. This occurs because the M dimension of the activation is extremely small, and the M direction likely contains only one block. As a result, the system moves the right matrix only once. You can freely increase M within the range of M≤128 without performance loss. If M is greater than 128, consider (128, 256] as the next performance tier.

In addition to M, the N axis splitting task also affects operator affinity. For example, the MLA preprocessing in deepseekR1 uses the same activation (with the shape [batch_size, 7168]) to perform matrix multiplication with two weights (with the shapes [7168, 1536] and [7168, 576]). When the batch_size is small, the N axis cannot use all cores even if baseN is reduced to 128. Therefore, the time consumption of each matrix multiplication approximately equals the time consumption of a single matrix multiplication (with the shape [7168, 2112]) formed by concatenating their weight N axes. If you only consider model competitiveness, it is preferable to merge these two weights. Otherwise, the bandwidth utilization of both small matrix multiplications will be very poor.

The common core partitioning strategies for the Attention operator are q_seqlen, batch_size, and kv_headnum. During the incremental phase, q_seqlen is merged based on the MTP and GQA multiples, but it usually does not exceed 128. As a result, it cannot be split into a second task, so the parallelism is essentially batch_size * kv_headnum.

In summary, based on the shape information and operator category, you can identify whether the operator has a load balancing problem. You can then predict the sharding strategy and the batch strategy for maximum throughput.
