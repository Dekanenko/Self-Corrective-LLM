# scripts/extract_latest_checkpoint.py

import argparse
import json
import logging
import os
import shutil
import tempfile
import boto3
from urllib.parse import urlparse

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def extract_latest_checkpoint(input_s3_uri: str, output_s3_uri: str):
    """
    Extracts the latest checkpoint from a model.tar.gz file and uploads it to S3.
    
    Args:
        input_s3_uri: S3 URI to the model.tar.gz file (e.g., s3://bucket/path/model.tar.gz)
        output_s3_uri: S3 URI for the output directory (e.g., s3://bucket/path/checkpoint/)
    """
    
    # Parse S3 URIs
    input_parsed = urlparse(input_s3_uri)
    input_bucket = input_parsed.netloc
    input_key = input_parsed.path.lstrip('/')
    
    output_parsed = urlparse(output_s3_uri)
    output_bucket = output_parsed.netloc
    output_prefix = output_parsed.path.lstrip('/')
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")
    
    try:
        # --- Step 1: Download model.tar.gz from S3 ---
        logger.info(f"Downloading model.tar.gz from {input_s3_uri}...")
        s3_client = boto3.client('s3')
        
        tarball_path = os.path.join(temp_dir, "model.tar.gz")
        s3_client.download_file(input_bucket, input_key, tarball_path)
        logger.info("Download completed.")
        
        # --- Step 2: Extract the tarball ---
        logger.info("Extracting model.tar.gz...")
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        shutil.unpack_archive(tarball_path, extract_dir)
        logger.info("Extraction completed.")
        
        # --- Step 3: Find the latest checkpoint ---
        logger.info("Searching for checkpoints...")
        potential_checkpoint_paths = []
        
        for root, dirs, files in os.walk(extract_dir):
            if "adapter_config.json" in files:
                potential_checkpoint_paths.append(root)
        
        if not potential_checkpoint_paths:
            logger.error("No checkpoints found with 'adapter_config.json'!")
            raise ValueError("No valid checkpoints found in the extracted archive")
        
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
        logger.info(f"Found {len(potential_checkpoint_paths)} checkpoint(s). Using latest: {latest_checkpoint_path}")
        
        # --- Step 4: Upload the latest checkpoint to S3 ---
        logger.info(f"Uploading latest checkpoint to {output_s3_uri}...")
        
        # Upload all files from the checkpoint directory
        for root, dirs, files in os.walk(latest_checkpoint_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, latest_checkpoint_path)
                s3_key = f"{output_prefix.rstrip('/')}/{relative_path}"
                
                logger.info(f"Uploading {relative_path} to s3://{output_bucket}/{s3_key}")
                s3_client.upload_file(local_file_path, output_bucket, s3_key)
        
        logger.info("Upload completed successfully!")
        logger.info(f"Latest checkpoint is now available at: {output_s3_uri}")
        
        # --- Step 5: Print checkpoint info ---
        checkpoint_name = os.path.basename(latest_checkpoint_path)
        logger.info(f"Checkpoint name: {checkpoint_name}")
        
        # List files in the checkpoint
        checkpoint_files = []
        for root, dirs, files in os.walk(latest_checkpoint_path):
            for file in files:
                relative_path = os.path.relpath(os.path.join(root, file), latest_checkpoint_path)
                checkpoint_files.append(relative_path)
        
        logger.info(f"Checkpoint contains {len(checkpoint_files)} files:")
        for file in sorted(checkpoint_files):
            logger.info(f"  - {file}")
        
    except Exception as e:
        logger.error(f"Error during checkpoint extraction: {e}")
        raise
    
    finally:
        # Cleanup temporary directory
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        logger.info("Cleanup completed.")

def main():
    parser = argparse.ArgumentParser(
        description="Extract the latest checkpoint from a model.tar.gz file and upload to S3"
    )
    parser.add_argument(
        "--input_s3_uri",
        type=str,
        required=True,
        help="S3 URI to the model.tar.gz file (e.g., s3://bucket/path/model.tar.gz)"
    )
    parser.add_argument(
        "--output_s3_uri", 
        type=str,
        required=True,
        help="S3 URI for the output directory (e.g., s3://bucket/path/checkpoint/)"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting checkpoint extraction...")
    logger.info(f"Input: {args.input_s3_uri}")
    logger.info(f"Output: {args.output_s3_uri}")
    
    extract_latest_checkpoint(args.input_s3_uri, args.output_s3_uri)
    
    logger.info("Checkpoint extraction completed successfully!")

if __name__ == "__main__":
    main()
