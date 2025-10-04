# scripts/train.py

import argparse
import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import datasets
from transformers.trainer_utils import get_last_checkpoint

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

    # Hyperparameters - Stage 1: Skill Acquisition
    parser.add_argument("--epochs_s1", type=int, default=1, help="Number of epochs for Stage 1.")
    parser.add_argument("--learning_rate_s1", type=float, default=2e-5, help="Base learning rate for Stage 1.")
    parser.add_argument("--custom_head_learning_rate_s1", type=float, default=2e-4, help="Custom head learning rate for Stage 1.")

    # Hyperparameters - Stage 2: Stabilization
    parser.add_argument("--epochs_s2", type=int, default=2, help="Number of epochs for Stage 2.")
    parser.add_argument("--learning_rate_s2", type=float, default=2e-6, help="Base learning rate for Stage 2.")
    parser.add_argument("--custom_head_learning_rate_s2", type=float, default=2e-5, help="Custom head learning rate for Stage 2.")

    # Shared Hyperparameters
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--correction_weights", type=str, default='[1.0, 10.0]', help='JSON string for a 2-element list of weights for BCE loss [weight_for_0, weight_for_1]')
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
    # check if the correction weights are a list of floats
    print(f"--- Correction weights type: {correction_weights[0].__class__} ---")

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
        # Only the custom detector head and new embeddings are not quantized.
        llm_int8_skip_modules=[
            "hallucination_gate_proj",
            "hallucination_up_proj",
            "hallucination_down_proj",
            "hallucination_detector",
            "new_token_embeddings",
        ],
    )

    # Pass custom hyperparameters to the model config
    model_config = AutoConfig.from_pretrained(args.base_model_path)
    model_config.deletion_boost_multiplier = 1.0
    model_config.threshold = 0.6

    print("--- Loading Model with BNB Config ---")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        config=model_config,
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
            "embed_tokens",
            "lm_head",
        ],
        # The custom detector head and new embeddings are fully fine-tuned.
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

    # --- STAGE 1: SKILL ACQUISITION ---
    print("--- Loading Stage 1 dataset ---")
    dataset_s1_path = os.path.join(args.dataset_path, "_stage_1")
    dataset_s1 = datasets.load_from_disk(dataset_s1_path)
    train_dataset_s1, eval_dataset_s1 = dataset_s1["train"], dataset_s1["test"]
    
    print("--- Setting up Trainer for Stage 1 ---")
    training_args_s1 = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs_s1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate_s1,
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
        save_total_limit=3,
        report_to="wandb",
        run_name="self-correction-s1-skill-acquisition",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        label_names=["labels", "hallucination_labels"],
        max_grad_norm=args.max_grad_norm,
    )

    data_collator = SelfCorrectionDataCollator(tokenizer=tokenizer)

    trainer_s1 = SelfCorrectionTrainer(
        model=peft_model,
        args=training_args_s1,
        train_dataset=train_dataset_s1,
        eval_dataset=eval_dataset_s1,
        tokenizer=tokenizer,
        data_collator=data_collator,
        alpha=args.alpha,
        correction_weights=correction_weights,
        custom_head_learning_rate=args.custom_head_learning_rate_s1,
    )

    print("--- Starting training for Stage 1 ---")
    trainer_s1.train()
    print("--- Stage 1 finished ---")

    # --- MEMORY CLEANUP BETWEEN STAGES ---
    print("--- Cleaning up memory between stages ---")
    del trainer_s1
    del training_args_s1
    del train_dataset_s1
    del eval_dataset_s1
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # --- STAGE 2: STABILIZATION & INTEGRATION ---
    last_checkpoint = get_last_checkpoint(args.model_dir)
    print(f"--- Resuming from checkpoint for Stage 2: {last_checkpoint} ---")

    print("--- Loading Stage 2 dataset ---")
    dataset_s2_path = os.path.join(args.dataset_path, "_stage_2")
    dataset_s2 = datasets.load_from_disk(dataset_s2_path)
    train_dataset_s2, eval_dataset_s2 = dataset_s2["train"], dataset_s2["test"]

    print("--- Setting up Trainer for Stage 2 ---")
    training_args_s2 = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs_s2,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate_s2,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_dir=f"{args.output_data_dir}/logs/s2",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="wandb",
        run_name="self-correction-s2-stabilization",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        label_names=["labels", "hallucination_labels"],
        max_grad_norm=args.max_grad_norm,
    )

    trainer_s2 = SelfCorrectionTrainer(
        model=peft_model,
        args=training_args_s2,
        train_dataset=train_dataset_s2,
        eval_dataset=eval_dataset_s2,
        tokenizer=tokenizer,
        data_collator=data_collator,
        alpha=args.alpha,
        correction_weights=correction_weights,
        custom_head_learning_rate=args.custom_head_learning_rate_s2,
    )

    print("--- Starting training for Stage 2 ---")
    trainer_s2.train(resume_from_checkpoint=last_checkpoint)
    
    # 7. Save the final model
    print("--- Curriculum learning finished. Saving final model. ---")
    trainer_s2.save_model(args.model_dir)

if __name__ == "__main__":
    main()