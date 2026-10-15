# Ascend Tutorial
## Introduction

Ascend fully supports the use and development of verl. This document comprehensively describes how to use verl on Huawei Ascend chip NPUs.

Last updated: 05/14/2026.

## Directory structure

```
zh/
├── get_start/                     # Quick start guide
├── feature_support/               # Feature support description
├── model_support/                 # Model support description
├── dev_guide/                     # Development guide
├── faq/                           # Frequently asked questions
└── contribution_guide/            # Community contribution guide
```
## Latest news
- [verl-ascend-recipe repository created](https://github.com/verl-project/verl-ascend-recipe) - Added Ascend recipe
- [verl on Ascend 2026Q2 roadmap](https://github.com/verl-project/verl/issues/5526) - 2026Q2 roadmap released

## Quick start
- [Ascend image description](./get_start/dockerfile_build_guidance.rst) - Build and use a Docker image for the Ascend environment  
- [Ascend installation guide](./get_start/install_guidance.rst) - Customize the verl installation on an Ascend NPU                                                    
- [Ascend quick start guide](./get_start/quick_start.rst) - Get started quickly with running verl on an Ascend NPU

## Supported Features

- [Training Configuration Parameters and Metrics](./dev_guide/model_dev/parameter_and_metrics.md) - List of supported verl framework features/parameters
- [NPU Advanced Features Guide](./feature_support/npu_advance_features.md) - Description of common NPU-related features and environment variables

## Model Support

- [NPU model and algorithm support](./model_support/model_and_algorithm_support.md) - List of supported models and algorithms
- [Best practice examples](./model_support/examples) - Examples of best practices and model deployment


## Developer Guide

- [Model Development](./dev_guide/model_dev) 
    - [Model Migration to NPU Guide](./dev_guide/model_dev/transfer_to_npu_guide.md) - Model migration guide
    - [Training Configuration Parameters and Metrics](./dev_guide/model_dev/parameter_and_metrics.md) - Training parameters and metrics
    - [Model Evaluation](./dev_guide/model_dev/evaluation.md) - Model evaluation guide
- [Precision Debugging](./dev_guide/precision_analysis) 
    - [Precision Alignment Guide](./dev_guide/precision_analysis/precision_alignment.md) - Precision alignment guide
    - [Precision Debugger](../en/dev_guide/precision_analysis/precision_debugger.md) - Precision issue troubleshooting tool
- [Performance Tuning](./dev_guide/performance) 
    - [Ascend Performance Analysis Guide](./dev_guide/performance/ascend_performance_analysis_guide.md) - Performance analysis guide
    - [Ascend Performance Tuning Guide](./dev_guide/performance/perf_tuning_on_ascend.rst) - Performance tuning guide
    - [Profiling Collection Guide](./dev_guide/performance/ascend_profiling.rst) - Profiling tool usage guide


## Support and Feedback

If you encounter any issues during use, you can seek help through the following methods:

1. View the [NPU FAQ](./faq/faq.rst)
2. Submit an issue in GitHub Issues
3. Contact Ascend technical support

## Contribution guidelines
- [verl community contribution](../../contributing) - verl community contribution guidelines
- [NPU-CI addition guide](./contribution_guide/ascend_ci_guide.rst) - Ascend environment CI configuration and testing

## Related resources

- [verl official documentation](https://verl.readthedocs.io/)
- [Ascend Developer Community](https://www.hiascend.com/)
