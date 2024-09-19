# Advanced HPC Cluster Usage Session

Welcome to the Advanced HPC Cluster Usage Session. This Session aims to demonstrate how to parallelize code effectively on one of Europe's fastest computers at FZJ.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Agenda](#workshop-agenda)
- [Materials](#Workshop-materials)
- [Resources](#resources)

## Overview

In this workshop, you will learn:
- How to run code on the supercomputer.
- How to parallelize code using Distributed Data Parallel (DDP) .
- How Fully Sharded Data Parallel (FSDP) works.
- How to parallelize code using FSDP.
- Various sharding strategies for efficient resource utilization.
- How to run parallel code on the FZJ supercomputer.

## Prerequisites

Before attending the workshop, ensure you have:
- Access to the training project via JuDoor [training2426](https://judoor.fz-juelich.de/projects/join/training2435)
- Configured your SSH key to have access to the JUSUF machine. Otherwise, You can also use [Jupyter-JSC](https://jupyter.jsc.fz-juelich.de/)

## Installation

To get started with the examples provided in this repository, clone the repo and install the necessary dependencies.

### Clone the Repository

```sh
mkdir /p/project/training2426/$USER
cd /p/project/training2426/$USER
git clone https://github.com/sab148/nxtaim-distributed-models
```

### Install venv template

For Python:
```sh
cd nxtaim-distributed-models/DDP-FSDP-at-FZJ/
bash env/setup.sh
```

## Advanced HPC Cluster Usage Session Agenda

1. **Parallelism Strategies**
   - DDP.
   - Pipeline Parallelism.
   - ...

2. **Fully Sharded Data Parallel (FSDP)**

3. **Algorithm Overview**

4. **Model Initializations**
   - Deferred Initialization.

5. **Hands-on Coding** 

6. **Sharding Strategies**
   - Full Sharding.
   - Hybrid Sharding.
   - hands-on coding

7. **FSDP Interoperability**

8. **FSDP Limitations**

9. **Advanced Hyperparameter Optimization with Ray Tune** 

10. **Conclusion**

## Advanced HPC Cluster Usage Session Materials

The PowerPoint presentation for the first part of the session is available in this repository. We will use it to guide the session. You can find the presentation [here](https://github.com/sab148/nxtaim-distributed-models/blob/main/DDP-FSDP-at-FZJ/nxtaim_workshop_presentation.pdf) and [here](https://fz-juelich.sciebo.de/s/hc6tSV8pDCNVOaC).

During the first part of the session, we will primarily use to_distributed_training.py and run_to_distributed_training.sh files. 

## Resources
- [FSDP Paper](https://arxiv.org/pdf/2304.11277)
- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Advanced FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html)
- [PyTorch at JSC](https://sdlaml.pages.jsc.fz-juelich.de/ai/recipes/pytorch_at_jsc/)
