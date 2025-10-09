# scripts/fine_tune_pretrained.py

import argparse
import os
import json
import torch
import shutil
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
import gc

from src.trainer import SelfCorrectionTrainer, SelfCorrectionDataCollator

# --- Main Fine-tuning Function ---
def main():
    # --- Force Device Placement for PytorchDDP ---
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            
    # 1. Parse SageMaker-provided arguments
    parser = argparse.ArgumentParser()

    # --- SageMaker-specific arguments ---
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--dataset_path", type=str, default=os.environ.get("SM_CHANNEL_DATASET"))
    parser.add_argument("--base_model_path", type=str, default="/opt/ml/input/data/model")
    
    # --- Pre-trained model checkpoint path ---
    parser.add_argument("--pretrained_checkpoint_path", type=str, required=True, 
                       help="S3 path to the pre-trained model checkpoint (e.g., s3://bucket/path/model.tar.gz or s3://bucket/path/checkpoint/)")

    # Hyperparameters for fine-tuning
    parser.add_argument("--epochs", type=float, default=2, help="Number of epochs for fine-tuning.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Base learning rate.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Alpha.")

    # Shared Hyperparameters
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--correction_weights", type=str, default='[1.0, 10.0, 4.0, 1.0]')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
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
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Linear warmup over warmup_ratio fraction of total steps.")
    parser.add_argument("--max_grad_norm", type=float, default=40.0, help="The maximum gradient norm for clipping.")

    args, _ = parser.parse_known_args()

    # Parse the correction_weights from a JSON string
    correction_weights = json.loads(args.correction_weights)
    print(f"--- Correction weights: {correction_weights} ---")

    # 2. Load Tokenizer
    print("--- Loading tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Download and extract pre-trained checkpoint from S3
    print(f"--- Downloading pre-trained checkpoint from {args.pretrained_checkpoint_path} ---")
    import boto3
    import tempfile
    from urllib.parse import urlparse
    
    s3_client = boto3.client('s3')
    
    # Parse S3 path
    if args.pretrained_checkpoint_path.startswith('s3://'):
        parsed_url = urlparse(args.pretrained_checkpoint_path)
        bucket_name = parsed_url.netloc
        key = parsed_url.path.lstrip('/')
    else:
        raise ValueError("pretrained_checkpoint_path must be a valid S3 path starting with 's3://'")
    
    # Create temporary directory for checkpoint
    temp_checkpoint_dir = tempfile.mkdtemp()
    print(f"--- Temporary checkpoint directory: {temp_checkpoint_dir} ---")
    
    # Download and extract checkpoint
    try:
        if args.pretrained_checkpoint_path.endswith('.tar.gz'):
            # Handle model.tar.gz file
            print("--- Downloading and extracting model.tar.gz ---")
            tarball_path = os.path.join(temp_checkpoint_dir, "model.tar.gz")
            s3_client.download_file(bucket_name, key, tarball_path)
            
            # Extract the tarball
            extract_dir = os.path.join(temp_checkpoint_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            shutil.unpack_archive(tarball_path, extract_dir)
            
            # Find the latest checkpoint
            potential_checkpoint_paths = []
            for root, dirs, files in os.walk(extract_dir):
                if "adapter_config.json" in files:
                    potential_checkpoint_paths.append(root)
            
            if not potential_checkpoint_paths:
                raise ValueError("No checkpoints found with 'adapter_config.json' in the extracted archive")
            
            def get_step_from_path(path):
                """Extracts the step number from a path like '.../checkpoint-500'."""
                basename = os.path.basename(path)
                if basename.startswith("checkpoint-"):
                    try:
                        return int(basename.split('-')[-1])
                    except ValueError:
                        return -1
                return -1
            
            # Select the checkpoint with the highest step number
            latest_checkpoint_path = max(potential_checkpoint_paths, key=get_step_from_path)
            print(f"--- Found {len(potential_checkpoint_paths)} checkpoint(s). Using latest: {latest_checkpoint_path} ---")
            
            # Copy the latest checkpoint files to temp_checkpoint_dir
            for root, dirs, files in os.walk(latest_checkpoint_path):
                for file in files:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(temp_checkpoint_dir, file)
                    shutil.copy2(src_path, dst_path)
            
        else:
            # Handle regular checkpoint directory
            print("--- Downloading checkpoint directory ---")
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=key)
            
            if 'Contents' not in response:
                raise ValueError(f"No files found in S3 path: {args.pretrained_checkpoint_path}")
            
            # Download all files
            for obj in response['Contents']:
                file_key = obj['Key']
                local_path = os.path.join(temp_checkpoint_dir, os.path.basename(file_key))
                
                print(f"--- Downloading {file_key} to {local_path} ---")
                s3_client.download_file(bucket_name, file_key, local_path)
        
        print("--- Checkpoint download and extraction completed ---")
        
        # Debug: List all files in the checkpoint directory
        print("--- DEBUG: Files in checkpoint directory ---")
        for root, dirs, files in os.walk(temp_checkpoint_dir):
            level = root.replace(temp_checkpoint_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
    except Exception as e:
        print(f"--- Error downloading/extracting checkpoint: {e} ---")
        raise

    # 4. Load Base Model and Pre-trained Adapters
    print("--- Loading base model and pre-trained adapters ---")

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
            "hallucination_detector",
            "new_token_embeddings",
        ],
    )

    # Load the base model first, so it's always available
    print("--- Loading base model with BNB Config ---")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    print("--- Prepare model for kbit training ---")
    model = prepare_model_for_kbit_training(model)

    # 5. Load pre-trained adapters from checkpoint
    print("--- Loading pre-trained adapters from checkpoint ---")
    peft_model = None
    try:
        # Find the checkpoint directory
        checkpoint_dir = None
        for root, dirs, files in os.walk(temp_checkpoint_dir):
            if "adapter_config.json" in files:
                checkpoint_dir = root
                break
        
        if not checkpoint_dir:
            raise ValueError("No checkpoint directory found with adapter_config.json")
        
        print(f"--- Found checkpoint directory: {checkpoint_dir} ---")

        # Load the adapters onto the base model
        print(f"--- Loading adapters from: {checkpoint_dir} ---")
        peft_model = PeftModel.from_pretrained(model, checkpoint_dir)
        print("--- Adapters loaded successfully ---")

        peft_model.print_trainable_parameters()
        
        print("--- ✅ Pre-trained model and adapters loaded successfully ---")
        
    except Exception as e:
        print(f"--- Error loading pre-trained weights: {e} ---")
        raise
    finally:
        # Clean up checkpoint files immediately after loading to save disk space
        print("--- Cleaning up checkpoint files to save disk space ---")
        del model
        gc.collect()
        torch.cuda.empty_cache()

        if args.pretrained_checkpoint_path.endswith('.tar.gz'):
            # Remove the extracted directory and tarball
            extract_dir = os.path.join(temp_checkpoint_dir, "extracted")
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
                print("--- Removed extracted directory ---")
            
            tarball_path = os.path.join(temp_checkpoint_dir, "model.tar.gz")
            if os.path.exists(tarball_path):
                os.remove(tarball_path)
                print("--- Removed model.tar.gz ---")
        
        # Remove checkpoint files (keep only essential files if any)
        checkpoint_files = ["pytorch_model.bin", "adapter_model.bin", "adapter_config.json", "training_args.bin"]
        for file in os.listdir(temp_checkpoint_dir):
            file_path = os.path.join(temp_checkpoint_dir, file)
            if os.path.isfile(file_path) and file not in checkpoint_files:
                os.remove(file_path)
                print(f"--- Removed {file} ---")
        
        print("--- Checkpoint cleanup completed ---")

    if peft_model is None:
        raise RuntimeError("PEFT model could not be loaded.")

    peft_model.print_trainable_parameters()

    # 6. Load Dataset
    print("--- Loading fine-tuning dataset ---")
    dataset = datasets.load_from_disk(args.dataset_path)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    
    print(f"--- Train dataset size: {len(train_dataset)} ---")
    print(f"--- Eval dataset size: {len(eval_dataset)} ---")

    # 7. Setup Trainer (Fresh training with pre-loaded weights)
    print("--- Setting up Trainer ---")
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_dir=f"{args.output_data_dir}/logs",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="wandb",
        run_name="self-correction-fine-tuning",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        label_names=["labels", "hallucination_labels"],
        max_grad_norm=args.max_grad_norm,
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
        correction_weights=correction_weights,
    )

    # 8. Start Fine-tuning (Fresh training with pre-loaded weights)
    print("--- Starting fine-tuning ---")
    print("--- NOTE: This is fresh training starting from step 0, but with pre-loaded weights ---")
    trainer.train()
    
    # 9. Save the final model
    print("--- Fine-tuning finished. Saving final model. ---")
    trainer.save_model(args.model_dir)
    
    # 10. Cleanup temporary files
    print("--- Cleaning up temporary files ---")
    shutil.rmtree(temp_checkpoint_dir)
    print("--- Temporary files cleaned up ---")

if __name__ == "__main__":
    main()
