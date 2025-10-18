# scripts/train.py

import argparse
import os
import json
import torch
import shutil
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel
import datasets
from transformers.trainer_utils import get_last_checkpoint
import wandb

from src.trainer import SelfCorrectionTrainer, SelfCorrectionDataCollator

# --- Main Training Function ---
def main():
    # --- Force Device Placement for PytorchDDP ---
    # Manually set the device for each process based on the 
    # LOCAL_RANK environment variable provided by torchrun.
    # This overrides the faulty default behavior where all processes
    # were piling onto GPU 0.
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        local_rank = 0 # Default to 0 if not in a distributed environment
            
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Two-stage training script for the Self-Corrective LLaMA model.")

    # --- SageMaker-specific arguments ---
     # The directory where the final model artifacts should be saved.
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    # The directory for other outputs like logs.
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    # Input channels for data.
    parser.add_argument("--dataset_path", type=str, default=os.environ.get("SM_CHANNEL_DATASET"))
    # A dedicated input channel for the base model.
    parser.add_argument("--base_model_path", type=str, default="/opt/ml/input/data/model")

    # --- Stage 1 Hyperparameters (Detector Training) ---
    parser.add_argument("--epochs_s1", type=float, default=1, help="Number of epochs for Stage 1.")
    parser.add_argument("--learning_rate_s1", type=float, default=2e-4, help="Learning rate for Stage 1.")
    
    # --- Stage 2 Hyperparameters (Joint Training) ---
    parser.add_argument("--epochs_s2", type=float, default=2, help="Number of epochs for Stage 2.")
    parser.add_argument("--learning_rate_s2", type=float, default=2e-5, help="Learning rate for Stage 2.")
    parser.add_argument("--alpha_s2", type=float, default=0.3, help="Alpha for Stage 2 (balances token and hallucination loss).")

    # --- Shared Hyperparameters ---
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--correction_weights", type=str, default='[1.0, 10.0, 6.0]', help='JSON string for a 3-element list for [no-op, del-s, del-a] weights.')
    parser.add_argument("--gradient_accumulation_steps_s1", type=int, default=8, help="Gradient accumulation steps for Stage 1.")
    parser.add_argument("--gradient_accumulation_steps_s2", type=int, default=4, help="Gradient accumulation steps for Stage 2.")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)

    # Exposing LoRA Config
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=40.0)

    args, _ = parser.parse_known_args()
    correction_weights = json.loads(args.correction_weights)
    
    # Define a dedicated directory for the Stage 1 checkpoints
    stage_1_checkpoints_path = os.path.join(args.output_data_dir, "s1_checkpoints")

    # 2. Initial Model & Tokenizer Setup (used for both stages)
    print("--- Loading tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # QLoRA configuration for 4-bit training
    print("--- Loading BNB Config ---")
    compute_dtype = getattr(torch, "bfloat16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        # Only the custom detector head and new embeddings are not quantized.
        llm_int8_skip_modules=[
            "hallucination_gate_proj",
            "hallucination_up_proj",
            "hallucination_down_proj",
            "hallucination_norm",
            "hallucination_detector",
        ],
    )

    model_config = AutoConfig.from_pretrained(args.base_model_path)

    print("--- Configuring PEFT ---")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=[
            "hallucination_gate_proj",
            "hallucination_up_proj",
            "hallucination_down_proj",
            "hallucination_detector",
            "hallucination_norm",
        ],
    )

    # 3. --- STAGE 1: DETECTOR TRAINING ---
    print("--- Loading Model for Stage 1 ---")
    model_s1 = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        config=model_config,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model_s1 = prepare_model_for_kbit_training(model_s1)
    peft_model_s1 = get_peft_model(model_s1, peft_config)
    peft_model_s1.print_trainable_parameters()

    print("--- Loading Stage 1 dataset ---")
    dataset_s1_path = os.path.join(args.dataset_path, "training_data_stage_1")
    print(f"Loading dataset from: {dataset_s1_path}")
    dataset_s1 = datasets.load_from_disk(dataset_s1_path)
    train_dataset_s1, eval_dataset_s1 = dataset_s1["train"], dataset_s1["test"]

    training_args_s1 = TrainingArguments(
        output_dir=stage_1_checkpoints_path,
        num_train_epochs=args.epochs_s1,
        learning_rate=args.learning_rate_s1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps_s1,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_dir=f"{args.output_data_dir}/logs/s1",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="wandb",
        run_name="self-correction-s1-detector",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        label_names=["labels", "hallucination_labels"],
        max_grad_norm=args.max_grad_norm,
    )

    trainer_s1 = SelfCorrectionTrainer(
        model=peft_model_s1,
        args=training_args_s1,
        train_dataset=train_dataset_s1,
        eval_dataset=eval_dataset_s1,
        tokenizer=tokenizer,
        data_collator=SelfCorrectionDataCollator(tokenizer=tokenizer),
        alpha=0.0, # Alpha=0 trains detector only
        correction_weights=correction_weights,
    )

    print("--- Starting Stage 1 Training ---")
    trainer_s1.train()
    print("--- Stage 1 Finished. ---")

    # --- End the W&B run for Stage 1 ---
    # This ensures that Stage 2 starts a new, separate run.
    if "wandb" in training_args_s1.report_to and local_rank == 0:
        print("--- Finishing Stage 1 W&B run ---")
        wandb.finish()

    # --- Post-Stage 1 Setup ---
    # Find the last checkpoint from the Stage 1 training run.
    last_checkpoint_s1 = get_last_checkpoint(stage_1_checkpoints_path)
    if last_checkpoint_s1 is None:
        raise ValueError("Could not find a valid checkpoint after Stage 1 training.")
    print(f"--- Found latest Stage 1 checkpoint at: {last_checkpoint_s1} ---")

    # 4. --- MEMORY CLEANUP ---
    print("--- Cleaning up memory between stages ---")
    del trainer_s1, model_s1, peft_model_s1
    gc.collect()
    torch.cuda.empty_cache()

    # 5. --- STAGE 2: JOINT TRAINING ---
    print("--- Loading base model for Stage 2 ---")
    # We must reload the base model from scratch to ensure a clean state.
    model_s2_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        config=model_config,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model_s2_base = prepare_model_for_kbit_training(model_s2_base)

    print(f"--- Loading and applying adapter from Stage 1 checkpoint: {last_checkpoint_s1} ---")
    # Load the PEFT model from the Stage 1 checkpoint and crucially, make it trainable.
    model_s2 = PeftModel.from_pretrained(model_s2_base, last_checkpoint_s1, is_trainable=True)
    model_s2.print_trainable_parameters()
    
    print("--- Loading Stage 2 dataset ---")
    dataset_s2_path = os.path.join(args.dataset_path, "training_data_stage_2")
    print(f"Loading dataset from: {dataset_s2_path}")
    dataset_s2 = datasets.load_from_disk(dataset_s2_path)
    train_dataset_s2, eval_dataset_s2 = dataset_s2["train"], dataset_s2["test"]

    training_args_s2 = TrainingArguments(
        output_dir=args.model_dir, # Stage 2 saves to the main model directory
        num_train_epochs=args.epochs_s2,
        learning_rate=args.learning_rate_s2,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps_s2,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_dir=f"{args.output_data_dir}/logs/s2",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        report_to="wandb",
        run_name="self-correction-s2-joint",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        label_names=["labels", "hallucination_labels"],
        max_grad_norm=args.max_grad_norm,
    )

    trainer_s2 = SelfCorrectionTrainer(
        model=model_s2,
        args=training_args_s2,
        train_dataset=train_dataset_s2,
        eval_dataset=eval_dataset_s2,
        tokenizer=tokenizer,
        data_collator=SelfCorrectionDataCollator(tokenizer=tokenizer),
        alpha=args.alpha_s2, # Alpha > 0 trains both LoRA and detector
        correction_weights=correction_weights,
    )

    print("--- Starting Stage 2 Training ---")
    trainer_s2.train()
    
    print("--- Two-stage training finished. Saving final model. ---")
    trainer_s2.save_model(args.model_dir)

if __name__ == "__main__":
    main()