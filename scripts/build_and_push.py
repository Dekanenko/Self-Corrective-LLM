import argparse
import json
import logging
import os
import shutil
import sys
import gc
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

# --- Setup ---

# Set up professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to Python path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modeling import SelfCorrectiveLlama

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

def create_model_card(repo_id: str, base_model: str, special_tokens: list) -> str:
    """Generates a professional README.md model card."""
    special_tokens_str = ", ".join(f"`{token}`" for token in special_tokens)
    # NOTE: It is your responsibility to use the correct license identifier for the base model.
    # For example, for Llama 3 8B Instruct, it is "meta-llama/llama-3-8b-instruct-license".
    # See the original model card on the Hub for the correct value.
    license_identifier = "llama3.1" # <-- ADJUST AS NEEDED

    return f"""---
license: {license_identifier}
language: en
base_model: {base_model}
---

# {repo_id}

This is a version of `{base_model}` modified with a custom architecture to support self-correction via hallucination detection.

This model, an instance of `SelfCorrectiveLlama`, includes a hallucination detection head that modifies the logits of special tokens to aid in content generation and revision.

## Special Tokens

The tokenizer has been expanded to include the following special tokens: {special_tokens_str}.

## How to Use

Because this model uses a custom architecture, you **must** use `trust_remote_code=True` when loading it. The required `modeling.py` file is included in this repository.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Important: You must trust the remote code
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
)

# You can now use the model for generation
# For example, to get hallucination probabilities:
# (sequences, p_halls) = model.generate(..., output_p_hall=True)
```

## Model Details

This model was programmatically converted and uploaded using a deployment script. The custom class `SelfCorrectiveLlama` can be found in the `modeling.py` file.

The code in `modeling.py` is licensed under the Apache 2.0 License. The model weights are subject to the original license of the base model.
"""

def build_and_push(config_path: str):
    """Main function to run the model conversion and deployment pipeline."""
    
    # 1. Load and Validate Configuration
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    base_model_name = config["base_model_name"]
    hf_repo_id = config["hf_repo_id"]
    local_output_dir = config["local_output_dir"]
    special_tokens = config["special_tokens"]
    model_code_path = config["model_code_path"]

    if not os.path.exists(model_code_path):
        logger.error(f"Model source code not found at {model_code_path}")
        sys.exit("Aborting: Check the `model_code_path` in your config.")

    # 2. Authenticate with Hugging Face
    hf_login()

    # 3. Perform Model Conversion
    logger.info(f"Starting model conversion for '{base_model_name}'")

    logger.info("Loading base tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    logger.info(f"Adding {len(special_tokens)} special tokens: {special_tokens}")
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    logger.info("Resizing token embeddings of the base model...")
    base_model.resize_token_embeddings(len(tokenizer))

    logger.info("Creating instance of the custom SelfCorrectiveLlama model...")
    # This is a critical step for Hub integration
    SelfCorrectiveLlama.register_for_auto_class("AutoModelForCausalLM")
    custom_model = SelfCorrectiveLlama(base_model.config)

    # Ensure the new model uses the same data type (e.g., bfloat16) as the base model
    logger.info(f"Casting custom model to the base model's dtype: {base_model.dtype}")
    custom_model = custom_model.to(base_model.dtype)

    logger.info("Loading state dict from base model into custom model...")
    incompatible_keys = custom_model.load_state_dict(base_model.state_dict(), strict=False)
    logger.info(f"State dict loaded. Mismatched keys (expected): {incompatible_keys}")
    
    # Explicitly free up memory by deleting the base model
    logger.info("Base model state copied. Deleting base model to free up memory...")
    del base_model
    gc.collect()
    
    # 4. Prepare Local Directory for Deployment
    if os.path.exists(local_output_dir):
        logger.warning(f"Output directory {local_output_dir} already exists. It will be overwritten.")
        shutil.rmtree(local_output_dir)
    os.makedirs(local_output_dir)

    logger.info(f"Saving custom model and tokenizer to {local_output_dir}...")
    custom_model.save_pretrained(local_output_dir)
    tokenizer.save_pretrained(local_output_dir)

    # 5. Add Custom Code and Model Card
    logger.info("Copying custom model code (`modeling.py`) to the output directory...")
    shutil.copy(model_code_path, os.path.join(local_output_dir, "modeling.py"))

    # Clean up the original source file if `save_pretrained` copied it automatically
    original_code_filename = os.path.basename(model_code_path)
    spurious_file_path = os.path.join(local_output_dir, original_code_filename)
    if os.path.exists(spurious_file_path):
        logger.info(f"Removing redundant source file '{original_code_filename}' from deployment folder...")
        os.remove(spurious_file_path)

    logger.info("Creating and writing model card (`README.md`)...")
    readme_content = create_model_card(hf_repo_id, base_model_name, special_tokens)
    with open(os.path.join(local_output_dir, "README.md"), "w") as f:
        f.write(readme_content)

    # 6. Deploy to Hugging Face Hub
    logger.info(f"Deploying model to Hugging Face Hub at repository: {hf_repo_id}")
    api = HfApi()
    
    logger.info("Creating repository on the Hub (if it doesn't exist)...")
    api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)

    logger.info(f"Uploading contents of {local_output_dir} to {hf_repo_id}...")
    api.upload_folder(
        folder_path=local_output_dir,
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Initial model conversion and upload."
    )

    logger.info("Upload to Hugging Face Hub complete.")
    logger.info(f"Model is available at: https://huggingface.co/{hf_repo_id}")

    # 7. Final Cleanup
    logger.info(f"Cleaning up local staging directory: {local_output_dir}")
    try:
        shutil.rmtree(local_output_dir)
        logger.info("Local directory successfully removed.")
    except OSError as e:
        logger.error(f"Error removing local directory {local_output_dir}: {e.strerror}")

    logger.info("Build and deploy pipeline finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and deploy a custom Llama model to the Hugging Face Hub.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/build_and_push_config.json",
        help="Path to the deployment configuration JSON file."
    )
    args = parser.parse_args()
    build_and_push(args.config)