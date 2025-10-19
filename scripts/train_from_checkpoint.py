# scripts/train_from_checkpoint.py

import argparse
import os
import json
import torch
import shutil
import gc
import tarfile
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel, prepare_model_for_kbit_training
import datasets
import wandb
import boto3
from urllib.parse import urlparse

from src.trainer import SelfCorrectionTrainer, SelfCorrectionDataCollator

def main():
    # --- Device Placement ---
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
            
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Resume training from a Stage 1 checkpoint.")

    # --- SageMaker-specific arguments ---
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--dataset_path", type=str, default=os.environ.get("SM_CHANNEL_DATASET"))
    parser.add_argument("--base_model_path", type=str, default="/opt/ml/input/data/model")
    
    # --- New Argument for Checkpoint ---
    parser.add_argument("--stage_1_checkpoint_path", type=str, required=True, help="Path to the model.tar.gz from Stage 1.")

    # --- Stage 1 Hyperparameters (Unused but kept for consistency) ---
    parser.add_argument("--epochs_s1", type=float, default=1, help="Number of epochs for Stage 1.")
    parser.add_argument("--learning_rate_s1", type=float, default=2e-4, help="Learning rate for Stage 1.")
    
    # --- Stage 2 Hyperparameters ---
    parser.add_argument("--epochs_s2", type=float, default=2, help="Number of epochs for Stage 2.")
    parser.add_argument("--learning_rate_s2", type=float, default=2e-5, help="Learning rate for Stage 2.")
    parser.add_argument("--alpha_s2", type=float, default=0.3, help="Alpha for Stage 2.")

    # --- Shared Hyperparameters ---
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--correction_weights", type=str, default='[1.0, 10.0, 6.0]', help='JSON string for correction weights.')
    parser.add_argument("--gradient_accumulation_steps_s1", type=int, default=8, help="Gradient accumulation for Stage 1 (Unused).")
    parser.add_argument("--gradient_accumulation_steps_s2", type=int, default=4, help="Gradient accumulation for Stage 2.")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=40.0)

    # --- LoRA Config ---
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    args, _ = parser.parse_known_args()
    correction_weights = json.loads(args.correction_weights)
    
    # --- Unpack the Stage 1 Checkpoint ---
    checkpoint_dir = os.path.join(args.output_data_dir, "unpacked_s1_checkpoint")
    last_checkpoint = None

    if local_rank == 0:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir)

        if args.stage_1_checkpoint_path.startswith("s3://"):
            print(f"--- Downloading and extracting S3 tarball from {args.stage_1_checkpoint_path} ---")
            parsed_url = urlparse(args.stage_1_checkpoint_path)
            bucket_name = parsed_url.netloc
            key = parsed_url.path.lstrip('/')
            
            s3_client = boto3.client('s3')
            tarball_path = os.path.join(checkpoint_dir, "model.tar.gz")
            s3_client.download_file(bucket_name, key, tarball_path)
            
            shutil.unpack_archive(tarball_path, checkpoint_dir)
            os.remove(tarball_path) # Clean up the downloaded tarball
        else:
            print(f"--- Unpacking local tarball from {args.stage_1_checkpoint_path} ---")
            shutil.unpack_archive(args.stage_1_checkpoint_path, checkpoint_dir)
        
        print(f"--- Checkpoint unpacked to {checkpoint_dir} ---")

    # Wait for the main process to finish unpacking. This is a blocking operation.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Add a delay to allow for network filesystem synchronization before proceeding.
    # This helps prevent other ranks from failing if they access the path before
    # the unpacked files are visible to them.
    print(f"--- Rank {local_rank} waiting 10 seconds for filesystem sync... ---")
    time.sleep(10)

    # Now, all processes will scan the directory for the latest checkpoint.
    last_checkpoint = get_last_checkpoint(checkpoint_dir)
    if last_checkpoint is None:
        print(f"--- Rank {local_rank}: No 'checkpoint-xxx' directory found. Assuming adapter is in {checkpoint_dir} ---")
        last_checkpoint = checkpoint_dir
    else:
        print(f"--- Rank {local_rank}: Found latest checkpoint in unpacked directory: {last_checkpoint} ---")
    
    # --- Initial Model & Tokenizer Setup ---
    print("--- Loading tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print("--- Loading BNB Config ---")
    compute_dtype = getattr(torch, "bfloat16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=[
            "hallucination_gate_proj", "hallucination_up_proj",
            "hallucination_down_proj", "hallucination_norm", "hallucination_detector",
        ],
    )
    model_config = AutoConfig.from_pretrained(args.base_model_path)

    # --- Load Model for Stage 2 ---
    print("--- Loading base model for Stage 2 ---")
    model_s2_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        config=model_config,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model_s2_base = prepare_model_for_kbit_training(model_s2_base)

    print(f"--- Loading and applying adapter from: {last_checkpoint} ---")
    model_s2 = PeftModel.from_pretrained(model_s2_base, last_checkpoint, is_trainable=True)
    model_s2.print_trainable_parameters()
    
    # --- Load Stage 2 Dataset ---
    print("--- Loading Stage 2 dataset ---")
    dataset_s2_path = os.path.join(args.dataset_path, "training_data_stage_2")
    print(f"Loading dataset from: {dataset_s2_path}")
    dataset_s2 = datasets.load_from_disk(dataset_s2_path)
    train_dataset_s2, eval_dataset_s2 = dataset_s2["train"], dataset_s2["test"]

    # --- Set up Trainer for Stage 2 ---
    training_args_s2 = TrainingArguments(
        output_dir=args.model_dir,
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
        run_name="self-correction-s2-resumed",
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
        alpha=args.alpha_s2,
        correction_weights=correction_weights,
    )

    print("--- Starting Resumed Stage 2 Training ---")
    trainer_s2.train()
    
    print("--- Training finished. Saving final model. ---")
    trainer_s2.save_model(args.model_dir)

if __name__ == "__main__":
    main()
