# Parameter-Efficient Routed Fine-Tuning: Mixture-of-Experts Demands Mixture of Adaptation Modules


[![arXiv](https://img.shields.io/badge/arXiv-2508.02587-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2508.02587)


This repository contains the implementation of **Parameter-Efficient Routed Fine-Tuning (PERFT)** for Mixture of Experts (MoE) models. PERFT incorporates routing mechanisms in adaptation modules to align with MoE's multi-expert architecture, addressing the limitation that existing PEFT strategies often fail to leverage the dynamic routing mechanism among specialized experts.

## Overview

Mixture-of-Experts (MoE) models benefit from dynamic routing mechanisms among their specialized experts, which existing Parameter-Efficient Fine-Tuning (PEFT) strategies often fail to leverage. This work investigates whether adaptation modules themselves should incorporate routing mechanisms to align with MoE's multi-expert architecture.

PERFT analyzes the dynamics between key memory vectors in experts and expert vectors in routers, demonstrating that properly routed PEFT experts can unlock a much more expressive adaptation space while maintaining MoE's efficiency and flexibility.

This project provides tools and methods for fine-tuning Mixture of Experts models efficiently with minimal parameter updates. It supports both **Mixtral-8×7B** and **OLMoE-1B-7B** architectures and includes implementations for:

- **PERFT (Parameter-Efficient Routed Fine-Tuning)**: Routed adaptation strategies with three variants:
  - **PERFT-E**: Embedded routing adapters
  - **PERFT-D**: Distributed/shared routing adapters  
  - **PERFT-S**: Shared adapters

## Repository Structure

```
PERFT/
├── PERFT/
│   ├── commonsense/          # Commonsense reasoning task
│   │   ├── finetune.py       # Standard fine-tuning script
│   │   ├── finetune_qvlora.py # QvLoRA fine-tuning
│   │   ├── finetune_full.py  # Full fine-tuning baseline
│   │   ├── evaluate.py       # Evaluation script
│   │   ├── commonsense_evaluate.py
│   │   └── *.sh              # Training job scripts
│   ├── math/                  # Math problem solving task
│   │   ├── finetune.py
│   │   ├── finetune_qvlora.py
│   │   ├── math_evaluate.py
│   │   └── *.sh              # Training job scripts
│   ├── mixtral_modification/  # Mixtral model modifications
│   │   ├── configuration_mixtral.py
│   │   └── modeling_mixtral.py
│   ├── olmoe_modification/    # OLMoE model modifications
│   │   ├── configuration_olmoe.py
│   │   └── modeling_olmoe.py
│   ├── utils.py              # Utility functions
│   └── vector-visualize.py   # Visualization tools
└── PEFT_for_MoE_ARR_25_camera_ready.pdf  # Paper
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.13+
- transformers
- peft
- datasets
- safetensors
- Other dependencies (see requirements below)

### Setup

```bash
pip install torch transformers peft datasets safetensors fire tqdm
```

For evaluation, additional dependencies may be required depending on the task.

## Quick Start

### Training

#### Standard LoRA Fine-tuning

```bash
cd PERFT/commonsense
python finetune.py \
    --base_model 'allenai/OLMoE-1B-7B-0924' \
    --data_path 'commonsense_170k.json' \
    --output_dir './checkpoints/OLMoE-lora' \
    --batch_size 16 \
    --micro_batch_size 16 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --adapter_type 'LoRA' \
    --lora_r 16 \
    --lora_alpha 32
```

#### QvLoRA Fine-tuning

```bash
python finetune_qvlora.py \
    --base_model 'allenai/OLMoE-1B-7B-0924' \
    --data_path 'math_50k.json' \
    --output_dir './checkpoints/OLMoE-qvlora' \
    --batch_size 16 \
    --micro_batch_size 16 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --adapter_type 'LoRA' \
    --lora_r 16 \
    --lora_alpha 32
```

#### PERFT with Shared Routing Adapter

```bash
python finetune.py \
    --base_model 'allenai/OLMoE-1B-7B-0924' \
    --data_path 'commonsense_170k.json' \
    --output_dir './checkpoints/OLMoE-perft' \
    --batch_size 16 \
    --micro_batch_size 16 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --shared_routing_adapter True \
    --shared_routing_adapter_num_experts 8 \
    --shared_routing_adapter_num_experts_per_tok 1 \
    --adapter_type 'Parallel_Adapter' \
    --hidden_dim 16
```

#### PERFT with Embedded Routing Adapter

```bash
python finetune.py \
    --base_model 'allenai/OLMoE-1B-7B-0924' \
    --data_path 'commonsense_170k.json' \
    --output_dir './checkpoints/OLMoE-perft-e' \
    --batch_size 16 \
    --micro_batch_size 16 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --embedded_routing_adapter True \
    --adapter_type 'LoRA' \
    --lora_r 16 \
    --lora_alpha 32
```

### Evaluation

```bash
# Commonsense evaluation
python commonsense_evaluate.py \
    --base_model 'allenai/OLMoE-1B-7B-0924' \
    --peft_model './checkpoints/OLMoE-lora'

# Math evaluation
cd ../math
python math_evaluate.py \
    --base_model 'mistralai/Mixtral-8x7B-v0.1' \
    --peft_model './checkpoints/Mixtral-qvlora'
```

## Supported Models

- **Mixtral-8x7B**: `mistralai/Mixtral-8x7B-v0.1` or `mistralai/Mixtral-8x7B-Instruct-v0.1`
- **OLMoE-1B-7B**: `allenai/OLMoE-1B-7B-0924`

## Configuration Options

### Adapter Types

- `LoRA`: Low-Rank Adaptation
- `Parallel_Adapter`: Parallel adapter layers

### PERFT Hyperparameters

PERFT supports three main strategies for routing adaptation modules:

- `--shared_adapter`: Enable shared adapters across layers (PERFT-S variant)
- `--shared_adapter_num`: Number of shared adapters
- `--shared_routing_adapter`: Enable shared/distributed routing adapters (PERFT-D variant)
- `--shared_routing_adapter_num_experts`: Number of experts in routing adapter
- `--shared_routing_adapter_num_experts_per_tok`: Number of experts per token
- `--embedded_routing_adapter`: Enable embedded routing adapters (PERFT-E variant)

### LoRA Hyperparameters

- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha scaling factor (default: 32)
- `--dropout`: Dropout rate (default: 0.05)

### Parallel Adapter Hyperparameters

- `--hidden_dim`: Hidden dimension of adapter (default: 16)
- `--dropout`: Dropout rate (default: 0.05)

### Training Hyperparameters

- `--batch_size`: Total batch size
- `--micro_batch_size`: Micro batch size for gradient accumulation
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate (default: 1e-5)
- `--cutoff_len`: Maximum sequence length (default: 256)
- `--val_set_size`: Validation set size
- `--eval_step`: Evaluation frequency in steps
- `--save_step`: Checkpoint saving frequency in steps

## Datasets

The repository is configured for evaluation on 14 commonsense and arithmetic reasoning tasks:

1. **Commonsense Reasoning**: Uses `commonsense_170k.json` format
2. **Math Problem Solving**: Uses `math_50k.json` format

Please refer to the respective task directories for data format requirements and licensing information (`DATA_LICENSE` files).

## Visualization

The repository includes a vector visualization tool for analyzing the dynamics between key memory vectors in experts and expert vectors in routers (as shown in Figure 1 of the paper):

```bash
python vector-visualize.py \
    --model_path './checkpoints/OLMoE-perft' \
    --data_type 'shared_routing_expert' \
    --output_dir './visualizations'
```

Supported data types: `'all'`, `'ffn'`, `'shared_expert'`, `'shared_routing_expert'`, `'embedded_expert'`

## Distributed Training

The repository includes SLURM job scripts for distributed training. Example:

```bash
# Submit job
sbatch OLMoE-1B-7B.parallel.sh
```

Job scripts are configured for multi-GPU training and can be modified based on your cluster setup.



## Contact

For questions or issues, please open an issue on the repository.
