import argparse
import json
import logging
import os
import shutil
import sys
import gc
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
from dotenv import load_dotenv
import boto3
from urllib.parse import urlparse
import tempfile

load_dotenv()

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Helper Functions ---
def hf_login():
    """Logs in to Hugging Face Hub using an environment variable token."""
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token is None:
        logger.error("HUGGINGFACE_TOKEN environment variable not set.")
        sys.exit("Please set the HUGGINGFACE_TOKEN environment variable.")
    
    logger.info("Logging in to Hugging Face Hub...")
    login(token=token)
    logger.info("Successfully logged in to Hugging Face Hub.")

def create_model_card(repo_id: str, base_model: str) -> str:
    """Generates a professional README.md model card."""
    return f"""---
license: llama3.1
language: en
base_model: {base_model}
---

# {repo_id}

This is a fine-tuned version of `{base_model}` that has been trained to detect and mitigate hallucinations in generated text.

## How to Use

Because this model uses a custom architecture, you **must** use `trust_remote_code=True` when loading it.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
)
```
"""

def download_s3_folder(s3_uri: str, local_path: str):
    """Downloads a folder from S3 to a local path."""
    logger.info(f"Downloading from S3 URI '{s3_uri}' to '{local_path}'...")
    parsed_url = urlparse(s3_uri)
    bucket_name = parsed_url.netloc
    prefix = parsed_url.path.lstrip('/')
    
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('/'): # Skip folders
                relative_path = os.path.relpath(key, prefix)
                local_file_path = os.path.join(local_path, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3_client.download_file(bucket_name, key, local_file_path)


def merge_and_package(config_path: str):
    """Main function to merge the LoRA adapter and package the final model."""
    
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    base_model_path = config["base_model_path"]
    adapter_model_path = config["adapter_model_path"]
    final_output_dir = config["final_output_dir"]
    hf_repo_id = config["hf_repo_id"]
    model_code_path = config["model_code_path"]
    base_model_name = config["base_model_name"]

    # Create a temporary directory within the project to ensure it's on the large volume
    project_tmp_dir = os.path.join(project_root, ".tmp_merge")
    if os.path.exists(project_tmp_dir):
        shutil.rmtree(project_tmp_dir) # Clean up from previous failed runs
    os.makedirs(project_tmp_dir)

    base_model_temp_dir = tempfile.mkdtemp(dir=project_tmp_dir)
    adapter_model_temp_dir = tempfile.mkdtemp(dir=project_tmp_dir)

    try:
        # --- Step 1: Download Models from S3 if necessary ---
        if base_model_path.startswith("s3://"):
            download_s3_folder(base_model_path, base_model_temp_dir)
            base_model_path = base_model_temp_dir  # Update path to local temp dir
        
        if adapter_model_path.startswith("s3://"):
            if adapter_model_path.endswith(".tar.gz"):
                logger.info(f"Downloading and extracting S3 tarball from {adapter_model_path}...")
                parsed_url = urlparse(adapter_model_path)
                bucket_name = parsed_url.netloc
                key = parsed_url.path.lstrip('/')
                
                s3_client = boto3.client('s3')
                tarball_path = os.path.join(adapter_model_temp_dir, "model.tar.gz")
                s3_client.download_file(bucket_name, key, tarball_path)
                
                shutil.unpack_archive(tarball_path, adapter_model_temp_dir)
                
                # SageMaker often nests the output, saving multiple checkpoints.
                # We must find the latest one to ensure we use the best model state.
                potential_adapter_paths = []
                for root, dirs, files in os.walk(adapter_model_temp_dir):
                    if "adapter_config.json" in files:
                        potential_adapter_paths.append(root)
                
                if not potential_adapter_paths:
                    logger.warning(f"Could not find 'adapter_config.json' in the extracted archive at {adapter_model_temp_dir}. Using the root directory and hoping for the best.")
                    adapter_model_path = adapter_model_temp_dir
                else:
                    def get_step_from_path(path):
                        """Extracts the step number from a path like '.../checkpoint-500'."""
                        basename = os.path.basename(path)
                        if basename.startswith("checkpoint-"):
                            try:
                                return int(basename.split('-')[-1])
                            except ValueError:
                                return -1 # Not a valid checkpoint folder
                        return -1 # Not a checkpoint folder

                    # Select the path with the highest step number.
                    # If no paths are 'checkpoint-XXX', they all get a score of -1,
                    # and max() will simply return one of them, which is a reasonable fallback.
                    best_adapter_path = max(potential_adapter_paths, key=get_step_from_path)
                    adapter_model_path = best_adapter_path
                    logger.info(f"Found {len(potential_adapter_paths)} potential adapter(s). Using latest checkpoint: {adapter_model_path}")

            else: # If it's a folder, not a tarball
                 download_s3_folder(adapter_model_path, adapter_model_temp_dir)
                 adapter_model_path = adapter_model_temp_dir # Update path to local temp dir
        
        hf_login()

        # --- Step 2: Load the Base Model and Tokenizer from local paths ---
        logger.info(f"Loading base model from local path: {base_model_path}...")

        # Define the quantization config, mirroring the training setup for compatibility
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            # This is crucial: we must not quantize the modules that were fully fine-tuned.
            llm_int8_skip_modules=[
                "hallucination_gate_proj",
                "hallucination_up_proj",
                "hallucination_down_proj",
                "hallucination_detector"
            ],
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            trust_remote_code=True,
            # Use "auto" to automatically place the model on the available GPU.
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # --- Step 3: Load the PEFT Model and Merge ---
        logger.info(f"Loading LoRA adapter from local path: {adapter_model_path}...")
        # Load the PEFT model, which will apply the adapter to the quantized base model
        peft_model = PeftModel.from_pretrained(base_model, adapter_model_path)

        logger.info("Merging the LoRA adapter into the base model...")
        # merge_and_unload will de-quantize the model and combine the weights
        merged_model = peft_model.merge_and_unload()
        logger.info("Merge complete.")

        # --- Step 4: Prepare Final Directory for Upload ---
        if os.path.exists(final_output_dir):
            logger.warning(f"Output directory {final_output_dir} will be overwritten.")
            shutil.rmtree(final_output_dir)
        os.makedirs(final_output_dir)

        logger.info(f"Saving merged model and tokenizer to {final_output_dir}...")
        merged_model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)

        # --- Step 5: Add Custom Code and Model Card ---
        logger.info(f"Copying custom model code to {final_output_dir}...")
        shutil.copy(model_code_path, os.path.join(final_output_dir, "modeling.py"))

        logger.info("Creating and writing model card...")
        readme_content = create_model_card(hf_repo_id, base_model_name)
        with open(os.path.join(final_output_dir, "README.md"), "w") as f:
            f.write(readme_content)

        # --- Step 6: Push to Hugging Face Hub ---
        logger.info(f"Uploading model to the Hub at {hf_repo_id}...")
        api = HfApi()
        api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=final_output_dir,
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message="Upload fine-tuned and merged model."
        )

        logger.info("Upload to Hugging Face Hub complete.")
        logger.info(f"Model is available at: https://huggingface.co/{hf_repo_id}")

        # --- Step 7: Local Staging Cleanup ---
        logger.info(f"Cleaning up local staging directory: {final_output_dir}")
        shutil.rmtree(final_output_dir)
        
        logger.info("Merge and package pipeline finished successfully!")
        
    finally:
        # --- Final S3 Download Cleanup ---
        logger.info("Cleaning up temporary download directories...")
        shutil.rmtree(project_tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model and upload to the Hugging Face Hub.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/merge_and_package_config.json",
        help="Path to the merge and package configuration JSON file."
    )
    args = parser.parse_args()
    merge_and_package(args.config)
