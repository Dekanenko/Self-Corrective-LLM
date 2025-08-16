import argparse
import json
import logging
import os
import shutil
import sys
import gc
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from dotenv import load_dotenv

# Set the MPS fallback for local execution on Apple Silicon.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
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

    hf_login()

    # --- Step 1: Load the Base Model and Tokenizer ---
    logger.info(f"Loading base model from {base_model_path}...")
    # It's crucial to load with trust_remote_code=True to get the custom architecture
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to('cpu') # Load to CPU to save GPU memory during merge
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # --- Step 2: Load the PEFT Model and Merge ---
    logger.info(f"Loading LoRA adapter from {adapter_model_path}...")
    # This loads the base model with the adapter layers attached
    peft_model = PeftModel.from_pretrained(base_model, adapter_model_path)

    logger.info("Merging the LoRA adapter into the base model...")
    # This combines the adapter weights with the base model weights
    merged_model = peft_model.merge_and_unload()
    logger.info("Merge complete.")

    # --- Step 3: Prepare Final Directory for Upload ---
    if os.path.exists(final_output_dir):
        logger.warning(f"Output directory {final_output_dir} will be overwritten.")
        shutil.rmtree(final_output_dir)
    os.makedirs(final_output_dir)

    logger.info(f"Saving merged model and tokenizer to {final_output_dir}...")
    merged_model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    # --- Step 4: Add Custom Code and Model Card ---
    logger.info(f"Copying custom model code to {final_output_dir}...")
    shutil.copy(model_code_path, os.path.join(final_output_dir, "modeling.py"))

    logger.info("Creating and writing model card...")
    readme_content = create_model_card(hf_repo_id, base_model.config._name_or_path)
    with open(os.path.join(final_output_dir, "README.md"), "w") as f:
        f.write(readme_content)

    # --- Step 5: Push to Hugging Face Hub ---
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

    # --- Step 6: Final Cleanup ---
    logger.info(f"Cleaning up local staging directory: {final_output_dir}")
    shutil.rmtree(final_output_dir)
    
    logger.info("Merge and package pipeline finished successfully!")


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
