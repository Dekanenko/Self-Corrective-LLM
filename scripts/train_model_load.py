import os
import subprocess
import logging
import json
import argparse
import shutil
from huggingface_hub import snapshot_download

# Set up logging to provide clear, timestamped output.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_command(command):
    """
    Helper function to run a shell command and log its output.
    Raises an exception if the command fails.
    """
    cmd_for_logging = command if isinstance(command, str) else ' '.join(command)
    logger.info(f"Running command: {cmd_for_logging}")
    try:
        process = subprocess.run(
            command, check=True, capture_output=True, text=True, shell=isinstance(command, str)
        )
        if process.stdout:
            logger.info(process.stdout)
        if process.stderr:
            logger.warning(process.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise

def main(args):
    """
    Downloads a model from Hugging Face and uploads its raw files to S3
    for use as a SageMaker training job input.
    """
    logger.info("Starting model preparation for SageMaker training...")
    
    # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from {args.config}")
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config}. Aborting.")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {args.config}. Please check its format. Aborting.")
        return

    model_id = config["model_id"]
    s3_bucket = config["s3_bucket"]
    s3_key_prefix = config["s3_key_prefix"]
    local_download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "model_files"))

    try:
        # --- 2. Download Model Files ---
        logger.info(f"Downloading model '{model_id}' from Hugging Face Hub...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_download_dir,
            local_dir_use_symlinks=False,
            # token="YOUR_HF_TOKEN_HERE" # Uncomment if the model is private
        )
        logger.info(f"Model successfully downloaded to: {local_download_dir}")

        # --- 3. Upload to S3 ---
        s3_uri = f"s3://{s3_bucket}/{s3_key_prefix}"
        logger.info(f"Uploading model files to '{s3_uri}'...")
        run_command(["aws", "s3", "cp", "--recursive", local_download_dir, s3_uri])

        logger.info("=" * 60)
        logger.info("✅ Process completed successfully!")
        logger.info(f"Model files are ready for training at: {s3_uri}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"An unexpected error occurred during the process: {e}", exc_info=True)
        logger.error("❌ Model preparation failed.")

    finally:
        # --- 4. Cleanup ---
        logger.info(f"Cleaning up local download directory: {local_download_dir}")
        if os.path.exists(local_download_dir):
            shutil.rmtree(local_download_dir)
            logger.info("Local directory successfully removed.")
        else:
            logger.info("Cleanup not needed, directory does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face model for SageMaker training.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../configs/train_model_load_config.json"),
        help="Path to the configuration JSON file."
    )
    args = parser.parse_args()
    main(args)