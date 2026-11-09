# Ascend Tutorial
## Introduction

Ascend fully supports using and developing verl. This document comprehensively describes how to use verl on Huawei Ascend chip NPUs.

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
- [Ascend image guide](./get_start/dockerfile_build_guidance.rst) - Build and use Docker images for an Ascend environment  
- [Ascend installation guide](./get_start/install_guidance.rst) - Customize the verl installation on Ascend NPUs                                                    
- [Ascend quick start guide](./get_start/quick_start.rst) - Get started quickly with running verl on Ascend NPUs

## Supported Features

- [Training configuration parameters and metrics description](./dev_guide/model_dev/parameter_and_metrics.md) - List of supported verl framework features and parameters
- [NPU advanced features guide](./feature_support/npu_advance_features.md) - Description of common NPU-related features and environment variables

## Supported Models

- [NPU Model and Algorithm Support](./model_support/model_and_algorithm_support.md) - List of supported models/algorithms
- [Best Practice Examples](https://github.com/verl-project/verl/tree/main/docs/ascend_tutorial/zh/model_support/examples) - Best practices and model deployment examples


## Development Guide

- [Model development](https://github.com/verl-project/verl/tree/main/docs/ascend_tutorial/zh/dev_guide/model_dev)
    - [Model migration to NPU guide](./dev_guide/model_dev/transfer_to_npu_guide.md) - Model migration guide
    - [Training configuration parameters and metrics description](./dev_guide/model_dev/parameter_and_metrics.md) - Training parameters and metrics
    - [Model evaluation](./dev_guide/model_dev/evaluation.md) - Model evaluation guide
- [Precision debugging](https://github.com/verl-project/verl/tree/main/docs/ascend_tutorial/zh/dev_guide/precision_analysis)
    - [Precision alignment guide](./dev_guide/precision_analysis/precision_alignment.md) - Precision alignment guide
    - [Precision debugger](../en/dev_guide/precision_analysis/precision_debugger.md) - Precision troubleshooting tool
- [Performance tuning](https://github.com/verl-project/verl/tree/main/docs/ascend_tutorial/zh/dev_guide/performance)
    - [Ascend performance analysis guide](./dev_guide/performance/ascend_performance_analysis_guide.md) - Performance analysis guide
    - [Ascend performance tuning guide](./dev_guide/performance/perf_tuning_on_ascend.rst) - Performance tuning guide
    - [Profiling collection guide](./dev_guide/performance/ascend_profiling.rst) - Profiling tool usage guide


## Support and feedback

If you encounter any issues during use, you can get help through the following methods:

1. View the [NPU FAQ](./faq/faq.rst)
2. Submit an issue in GitHub Issues
3. Contact Ascend technical support

## Contribution Guide
- [verl Community Contribution](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md) - verl community contribution guide
- [NPU-CI Addition Guide](./contribution_guide/ascend_ci_guide.rst) - Ascend environment CI configuration and testing

## Related resources

- [verl official documentation](https://verl.readthedocs.io/)
- [Ascend developer community](https://www.hiascend.com/)
