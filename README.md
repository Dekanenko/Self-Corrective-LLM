# Self-Correcting LLM for Hallucination Mitigation

This repository contains the code and resources for the project "Self-Correcting LLM for Hallucination Mitigation". The project introduces a novel approach to enable Large Language Models (LLMs) to detect and correct their own hallucinations during the inference process. The core idea is to augment a base LLM with a specialized hallucination detection head and fine-tune it to recognize and rectify flawed generated text by inserting corrective instructions.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Resources](#resources)
- [Model Architecture](#model-architecture)
- [Two-Stage Training Process](#two-stage-training-process)
- [Dataset Preparation](#dataset-preparation)
- [Evaluation and Results](#evaluation-and-results)
- [Repository Structure](#repository-structure)
- [Scripts and Usage](#scripts-and-usage)
- [Dependencies](#dependencies)

## Project Overview

LLMs often "hallucinate" – generating text that is factually incorrect, nonsensical, or unfaithful to a provided source context. This project tackles this problem by building a self-correcting mechanism directly into the model's architecture.

The base model used is `meta-llama/Meta-Llama-3-8B-Instruct`. We introduce a custom **hallucination detection head** that works in tandem with the main language model head. This detector is trained to identify potential hallucinations at each generation step. When a likely hallucination is detected, the model is trained to automatically insert special instructions, such as `[remove sentence]` or `[remove response]`, into its own generated output. This allows for a post-processing step to easily clean the model's response, effectively enabling the model to correct itself in real-time.

The project is divided into three main parts:
1.  **Synthetic Dataset Preparation**: A custom pipeline for generating a large-scale synthetic dataset for training the self-correction capability.
2.  **Model Training**: A two-stage training process performed on AWS SageMaker to first train the hallucination detector and then jointly fine-tune the entire model.
3.  **Evaluation**: A comprehensive evaluation comparing the fine-tuned model against the base model and other state-of-the-art correction methods.

## Resources

-   **Fine-tuned Model**: The `SelfCorrectiveLlama` model, fine-tuned for self-correction, is available on the Hugging Face Hub. [Link to the model](<MathBite/self_corrective_llama_3.1_8B>)

## Key Features

-   **Self-Correction Mechanism**: The model can identify and flag its own hallucinations during inference.
-   **Hallucination Detection Head**: A dedicated neural network component trained specifically to predict hallucinatory content.
-   **Two-Stage Training**: An efficient training strategy that first bootstraps the detector and then fine-tunes the entire model using LoRA.
-   **Synthetic Data Pipeline**: A robust pipeline for generating high-quality training data for the self-correction task.
-   **Comprehensive Evaluation**: The model is evaluated against strong baselines, including RAPR and CoVE agents.

## Model Architecture

The model, named `SelfCorrectiveLlama`, inherits from the `transformers.LlamaForCausalLM` class. The key modification is the addition of a hallucination detection module.

-   **Base Model**: `meta-llama/Meta-Llama-3-8B-Instruct`.
-   **Hallucination Head**:
    -   It takes the final hidden states from the base LLaMA model as input.
    -   The hidden states are passed through a SwiGLU block for non-linear transformation.
    -   A final linear layer (`hallucination_detector`) projects the transformed hidden state into logits for three classes:
        1.  No hallucination.
        2.  Sentence-level hallucination (`[remove sentence]`).
        3.  Response-level hallucination (`[remove response]`).
-   **Gradient Isolation**: During the forward pass, the hidden states are detached before being fed to the hallucination head (`last_hidden.detach()`). This is a crucial design choice that prevents the gradients from the hallucination loss from flowing back into the base model's parameters during Stage 1 of training.

The inference logic is handled by a custom `generate` method, which:
1.  Calculates both token logits and hallucination logits at each step.
2.  Converts hallucination logits to probabilities using a softmax function.
3.  If the probability for a hallucination class exceeds a certain threshold (`deletion_threshold`) and a cooldown period (`correction_cooldown`) has passed, it inserts the corresponding correction instruction.
4.  Otherwise, it samples the next token from the standard token logits.

The full implementation can be found in `src/modeling.py`.

## Two-Stage Training Process

To effectively train the model, we employ a two-stage training strategy orchestrated by the `scripts/train.py` script, which is designed for AWS SageMaker.

-   **Stage 1: Detector Training**
    -   **Objective**: Train only the newly added hallucination detection head.
    -   **Method**: The base LLaMA model's weights are frozen (using 4-bit quantization with QLoRA). The trainer is configured with `alpha = 0.0`, meaning only the hallucination loss is used for backpropagation. This ensures that only the detector's weights are updated.
-   **Stage 2: Joint Fine-tuning**
    -   **Objective**: Reinforce the self-correction behavior by training the base model to accommodate the correction instructions.
    -   **Method**: We load the best checkpoint from Stage 1. The LoRA adapters for the base model are unfrozen. The trainer is configured with `alpha > 0` (e.g., 0.3), creating a combined loss from both the standard token prediction loss and the hallucination loss. This jointly fine-tunes the LoRA adapters and the hallucination head, teaching the model how to generate text that is amenable to the correction mechanism.

The custom `SelfCorrectionTrainer` in `src/trainer.py` manages the combined loss calculation and custom logging.

## Dataset Preparation

The training data is generated synthetically due to the lack of existing datasets for this specific self-correction task. The process is captured in the Jupyter notebooks under `notebooks/local_notebooks/`.

1.  **Initial Data Sources**:
    -   **Math QA**: [UMWP Dataset](<https://github.com/Yuki-Asuuna/UMWP/tree/main>)
    -   **Contextual QA**: [SQuAD Dataset (rajpurkar/squad-it)](<https://huggingface.co/datasets/rajpurkar/squad>)
2.  **Response Generation**: The base Llama 8B model is used to generate initial responses for the questions in the datasets.
3.  **Error Detection and Correction**: A pipeline (context_qa_correction_agent and math_qa_correction_agent) is used to analyze the generated responses. It identifies errors and creates "corrected" versions of the responses, embedding the `[remove sentence]` and `[remove response]` tags.
4.  **Data Formatting**: The notebooks `dataset_creation.ipynb` and `dataset_creation_stage_2.ipynb` process this raw output. They filter, sample, and format the data into the final structure required for the two training stages, creating balanced datasets of correct and incorrect examples.

## Evaluation and Results

The performance of the `SelfCorrectiveLlama` model was evaluated against the base Llama 3.1 8B Instruct model and three other correction/verification agents on a small, held-out test set.

-   **Comparison Agents**:
    -   **RAPR**: Researching and Revising What Language Models Say, Using Language Models. ([Paper Link](<https://arxiv.org/abs/2210.08726>))
    -   **CoVE**: Chain-of-Verification Reduces Hallucination in Large Language Models. ([Paper Link](<https://arxiv.org/abs/2309.11495>))
    -   **Custom Math Agent**: A custom-built agent for verifying and correcting mathematical reasoning.
    
    These agents are implemented in `src/agents/`.

### Results

The evaluation metrics were **Error Rate** (the proportion of responses with identified errors) and **Tokens Generated**. The results, as calculated in `notebooks/local_notebooks/evaluation.ipynb`, are as follows:

#### Math QA (UMWP Dataset)

| Model/Agent              | Error Rate |
| ------------------------ | ---------- |
| Base Llama 3.1 8B        | 47.6%      |
| **SelfCorrectiveLlama (Ours)** | **28.6%**      |
| Custom Math Agent        | 21.9%      |

*(Note: Add token comparison data here if available)*

#### Contextual QA (SQuAD Dataset)

| Model/Agent              | Error Rate |
| ------------------------ | ---------- |
| Base Llama 3.1 8B        | 30.5%      |
| **SelfCorrectiveLlama (Ours)** | **14.7%**      |
| RAPR Agent               | 15.8%      |
| CoVE Agent               | 10.5%      |

*(Note: Add token comparison data here if available)*

The results show that the `SelfCorrectiveLlama` significantly reduces the error rate compared to the base model on both tasks, demonstrating the effectiveness of the self-correction approach.

### Token Usage and Computational Cost

In addition to error rates, a critical aspect of this project is the computational cost, which can be approximated by token usage. The `SelfCorrectiveLlama` is designed to be a lightweight, single-pass solution, whereas agent-based pipelines like RAPR and CoVE require multiple expensive LLM calls.

#### Math QA Token Usage (Average per Query)

| Model/Agent | Input Tokens | Output Tokens | Total Tokens |
|---|---|---|---|
| Base Llama 3.1 8B | ~237 | ~196 | ~433 |
| **SelfCorrectiveLlama (Ours)** | **~330** | **~122** | **~452** |
| Custom Math Agent | ~1636 | ~521 | ~2157 |

#### Contextual QA Token Usage (Average per Query)

| Model/Agent | Input Tokens | Output Tokens | Total Tokens |
|---|---|---|---|
| Base Llama 3.1 8B | ~231 | ~10 | ~241 |
| **SelfCorrectiveLlama (Ours)** | **~324** | **~25** | **~349** |
| RAPR Agent | ~3814 | ~96 | ~3910 |
| CoVE Agent | ~2987 | ~98 | ~3085 |

### Conclusion

The `SelfCorrectiveLlama` demonstrates a significant reduction in hallucination-related errors compared to the base Llama 3.1 8B model across both mathematical and contextual reasoning tasks.

While multi-step agent pipelines like CoVE and the custom Math Agent can achieve slightly lower error rates in some cases, they do so at a substantially higher computational cost. The `SelfCorrectiveLlama` uses **5-8x fewer tokens** on average than these complex pipelines.

This makes the `SelfCorrectiveLlama` a highly efficient and practical solution. It offers a strong balance between performance and cost, providing a robust, single-pass method for mitigating hallucinations without the latency and expense of multi-agent systems.

## Repository Structure

```
├── configs/              # Configuration files for scripts
├── dataset/              # Raw, processed, and generated datasets
├── notebooks/            # Jupyter notebooks for analysis, dataset creation, and evaluation
│   ├── local_notebooks/
│   └── sagemaker_orchestration/
├── scripts/              # Python scripts for training, deployment, etc.
├── src/                  # Source code for the model, agents, and utilities
│   ├── agents/           # Implementations of CoVE, RAPR, and Math agents
│   ├── prompts/          # Prompts used by the agents
│   ├── utils/            # Utility functions for data processing
│   ├── modeling.py       # Definition of the SelfCorrectiveLlama model
│   ├── models.py         # Pydantic models for data structures
│   └── trainer.py        # Custom Hugging Face Trainer
└── requirements.txt      # Project dependencies
```

## Scripts and Usage

The `scripts/` directory contains key scripts for managing the project lifecycle.

-   `train.py`: The main script to run the two-stage training process on AWS SageMaker. It takes numerous arguments to control hyperparameters for both stages.
-   `build_and_push.py`: A utility script to convert the base Llama model to the `SelfCorrectiveLlama` architecture and upload it to the Hugging Face Hub. This script requires `trust_remote_code=True` to be used.
-   `merge_and_package.py`: Merges the trained LoRA adapters with the base model weights and packages it for deployment.
-   `push_training_data.py`: A script to upload the prepared training datasets to an S3 bucket for SageMaker to access.

## Dependencies

The main dependencies are listed in `requirements.txt`. Key libraries include:

-   `torch`
-   `transformers`
-   `peft` (for LoRA)
-   `bitsandbytes` (for quantization)
-   `datasets`
-   `langgraph` and `langchain` (for agents)
-   `sagemaker`

Install dependencies using:
```bash
pip install -r requirements.txt
```
