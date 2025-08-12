# scripts/train.py

import argparse
import os
import sys
import torch
from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType
import datasets

from src.trainer import SelfCorrectionTrainer, SelfCorrectionDataCollator

# --- Main Training Function ---
def main():
    # 1. Parse SageMaker-provided arguments
    parser = argparse.ArgumentParser()

    # --- SageMaker-specific arguments ---
     # The directory where the final model artifacts should be saved.
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    # The directory for other outputs like logs.
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    # Input channels for data.
    parser.add_argument("--dataset_path", type=str, default=os.environ.get("SM_CHANNEL_DATASET"))
    # A dedicated input channel for the base model.
    parser.add_argument("--base_model_path", type=str, default="/opt/ml/input/data/model")

    # Custom hyperparameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--alpha", type=float, default=0.7)

    args, _ = parser.parse_known_args()

    # 2. Load Tokenizer and Model
    print("--- Loading tokenizer and model ---")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
    )

    # 3. Configure PEFT/LoRA
    print("--- Configuring PEFT ---")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
        modules_to_save=["hallucination_detector"],
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    # 4. Load Datasets from SageMaker's input channels
    print("--- Loading dataset ---")
    dataset = datasets.load_from_disk(data_files=args.dataset_path)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    # 5. Set up Trainer
    print("--- Setting up Trainer ---")
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=8,
        optim="paged_adamw_8bit",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_dir=f"{args.output_data_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        report_to="tensorboard",
    )

    data_collator = SelfCorrectionDataCollator(tokenizer=tokenizer)

    trainer = SelfCorrectionTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        alpha=args.alpha,
    )

    # 6. Start Training
    print("--- Starting training ---")
    trainer.train()

    # 7. Save the final model
    print("--- Saving final model ---")
    # The Trainer automatically saves checkpoints. This is an explicit final save.
    trainer.save_model(args.model_dir)

if __name__ == "__main__":
    main()