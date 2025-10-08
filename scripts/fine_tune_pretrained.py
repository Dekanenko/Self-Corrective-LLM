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
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for fine-tuning.")
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
        
    except Exception as e:
        print(f"--- Error downloading/extracting checkpoint: {e} ---")
        raise

    # 4. Load Base Model
    print("--- Loading base model ---")

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

    print("--- Loading base model with BNB Config ---")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    print("--- Prepare model for kbit training ---")
    model = prepare_model_for_kbit_training(model)

    # 5. Configure PEFT/LoRA
    print("--- Configuring PEFT ---")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj",
            "embed_tokens",
            "lm_head",
        ],
        modules_to_save=[
            "hallucination_gate_proj",
            "hallucination_up_proj",
            "hallucination_down_proj",
            "hallucination_detector",
            "new_token_embeddings"
        ],
    )
    
    print("--- Applying PEFT to base model ---")
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    # 6. Load pre-trained weights
    print("--- Loading pre-trained weights from checkpoint ---")
    try:
        # Load LoRA adapter weights first
        if "adapter_model.bin" in os.listdir(temp_checkpoint_dir):
            print("--- Loading LoRA adapter weights ---")
            
            # Load the adapter state dict directly
            adapter_state_dict = torch.load(os.path.join(temp_checkpoint_dir, "adapter_model.bin"), map_location="cpu")
            
            # Load adapter config to understand the structure
            adapter_config_path = os.path.join(temp_checkpoint_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                import json
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                print(f"--- Adapter config loaded: {adapter_config.get('peft_type', 'unknown')} ---")
            
            # Apply LoRA weights to the existing peft_model
            # The adapter_state_dict contains the LoRA weights with keys like "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
            missing_keys, unexpected_keys = peft_model.load_state_dict(adapter_state_dict, strict=False)
            
            if missing_keys:
                print(f"--- LoRA Missing keys: {len(missing_keys)} keys ---")
                if len(missing_keys) < 10:  # Only print if not too many
                    for key in missing_keys[:5]:
                        print(f"    - {key}")
            if unexpected_keys:
                print(f"--- LoRA Unexpected keys: {len(unexpected_keys)} keys ---")
                if len(unexpected_keys) < 10:  # Only print if not too many
                    for key in unexpected_keys[:5]:
                        print(f"    - {key}")
            
            print("--- LoRA adapter weights loaded successfully ---")
        
        # Load custom head weights from pytorch_model.bin
        print("--- Loading custom head weights ---")
        checkpoint = torch.load(os.path.join(temp_checkpoint_dir, "pytorch_model.bin"), map_location="cpu")
        model_state_dict = checkpoint.get("model", checkpoint)
        
        print(f"--- Checkpoint contains {len(model_state_dict)} parameters ---")
        
        # Load hallucination detector weights
        hallucination_weights = {}
        for key, value in model_state_dict.items():
            if any(module in key for module in ["hallucination_gate_proj", "hallucination_up_proj", 
                                              "hallucination_down_proj", "hallucination_detector", 
                                              "new_token_embeddings"]):
                hallucination_weights[key] = value
        
        print(f"--- Found {len(hallucination_weights)} hallucination detector weights in checkpoint ---")
        if hallucination_weights:
            print("--- Sample checkpoint hallucination weights:")
            for key in list(hallucination_weights.keys())[:3]:
                print(f"    - {key}")
        
        # Apply custom head weights
        missing_keys, unexpected_keys = peft_model.load_state_dict(hallucination_weights, strict=False)
        
        if missing_keys:
            print(f"--- Missing keys: {missing_keys} ---")
        if unexpected_keys:
            print(f"--- Unexpected keys: {unexpected_keys} ---")
        
        print("--- Pre-trained weights loaded successfully ---")
        
        # Validate that weights were loaded correctly
        print("--- Validating loaded weights ---")
        
        # Check hallucination detector weights
        hallucination_modules = ["hallucination_gate_proj", "hallucination_up_proj", 
                               "hallucination_down_proj", "hallucination_detector", 
                               "new_token_embeddings"]
        
        hallucination_params = []
        lora_params = []
        
        for name, param in peft_model.named_parameters():
            # Check for hallucination detector parameters
            for module in hallucination_modules:
                if module in name and param.requires_grad:
                    hallucination_params.append(name)
                    break
            
            # Check for LoRA parameters
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(name)
        
        print(f"--- Found {len(hallucination_params)} hallucination detector parameters ---")
        if hallucination_params:
            print("--- Sample hallucination parameters:")
            for param in hallucination_params[:3]:
                print(f"    - {param}")
        
        print(f"--- Found {len(lora_params)} LoRA parameters ---")
        if lora_params:
            print("--- Sample LoRA parameters:")
            for param in lora_params[:3]:
                print(f"    - {param}")
        
        # Check if LoRA adapter configuration exists
        if hasattr(peft_model, 'peft_config') and peft_model.peft_config:
            print("--- LoRA adapter configuration present ---")
        else:
            print("--- WARNING: No LoRA adapter configuration found ---")
        
        # Final validation summary
        if len(hallucination_params) > 0 and len(lora_params) > 0:
            print("--- ✅ SUCCESS: Both LoRA and hallucination detector weights loaded ---")
        elif len(hallucination_params) > 0:
            print("--- ⚠️  PARTIAL: Only hallucination detector weights loaded ---")
        elif len(lora_params) > 0:
            print("--- ⚠️  PARTIAL: Only LoRA weights loaded ---")
        else:
            print("--- ❌ ERROR: No pre-trained weights loaded ---")
        
        # Clean up checkpoint files immediately after loading to save disk space
        print("--- Cleaning up checkpoint files to save disk space ---")
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
        
    except Exception as e:
        print(f"--- Error loading pre-trained weights: {e} ---")
        raise

    # 7. Load Dataset
    print("--- Loading fine-tuning dataset ---")
    dataset = datasets.load_from_disk(args.dataset_path)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    
    print(f"--- Train dataset size: {len(train_dataset)} ---")
    print(f"--- Eval dataset size: {len(eval_dataset)} ---")

    # 8. Setup Trainer
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

    # 9. Start Fine-tuning
    print("--- Starting fine-tuning ---")
    trainer.train()
    
    # 10. Save the final model
    print("--- Fine-tuning finished. Saving final model. ---")
    trainer.save_model(args.model_dir)
    
    # 11. Cleanup temporary files
    import shutil
    shutil.rmtree(temp_checkpoint_dir)
    print("--- Temporary files cleaned up ---")

if __name__ == "__main__":
    main()
