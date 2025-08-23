# scripts/train.py

import argparse
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import datasets
import ast

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

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--correction_weights", type=str, default='[1.0, 10.0, 4.0, 1.0]')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--max_sequence_length", type=int, default=800, help="Maximum sequence length for padding and truncation.")

    # Exposing LoRA Config
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Linear warmup over warmup_ratio fraction of total steps.")

    args, _ = parser.parse_known_args()

    # Parse the correction_weights from a JSON string
    correction_weights = ast.literal_eval(args.correction_weights)

    # 2. Load Tokenizer and Model
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
        llm_int8_skip_modules=[
            "hallucination_gate_proj",
            "hallucination_up_proj",
            "hallucination_down_proj",
            "hallucination_detector",
            "new_token_embeddings",
        ],
    )

    print("--- Loading Model with BNB Config ---")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    print("--- Prepare model for kbit training ---")
    model = prepare_model_for_kbit_training(model)

    # 3. Configure PEFT/LoRA
    print("--- Configuring PEFT ---")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        # Apply LoRA to the standard transformer blocks for memory efficiency.
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
        # Fully fine-tune the custom detector.
        modules_to_save=[
            "hallucination_gate_proj",
            "hallucination_up_proj",
            "hallucination_down_proj",
            "hallucination_detector",
            "new_token_embeddings"
        ],
    )
    
    print("--- Applying PEFT ---")
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    # 4. Load Datasets from SageMaker's input channels
    print("--- Loading dataset ---")
    dataset = datasets.load_from_disk(args.dataset_path)
    print(dataset)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    # 5. Set up Trainer
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        label_names=["labels", "hallucination_labels"],
    )

    data_collator = SelfCorrectionDataCollator(tokenizer=tokenizer, max_sequence_length=args.max_sequence_length)

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

    # 6. Start Training
    print("--- Starting training ---")
    trainer.train()

    # 7. Save the final model
    print("--- Saving final model ---")
    # The Trainer automatically saves checkpoints. This is an explicit final save.
    trainer.save_model(args.model_dir)

if __name__ == "__main__":
    main()