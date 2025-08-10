import os
import subprocess
import shutil
import logging
import json

# --- Script Logic ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command, working_dir=".", shell=False):
    """Helper function to run a shell command and log its output."""
    cmd_for_logging = command if isinstance(command, str) else ' '.join(command)
    logging.info(f"Running command: {cmd_for_logging} in '{working_dir}'")

    process = subprocess.run(command, cwd=working_dir, check=True, capture_output=True, text=True, shell=shell)
    logging.info(process.stdout)
    if process.stderr:
        logging.warning(process.stderr)

def main():
    """
    Main function to execute the build and upload process.
    Reads configuration from `config.json` file in the same directory.
    
    Prerequisites for the EC2 instance:
    1. Git, Git LFS, and AWS CLI installed (`sudo yum install git git-lfs awscli -y`).
    2. Python and Pip installed (`sudo yum install python3-pip -y`).
    3. Hugging Face Hub library installed (`pip3 install huggingface-hub`).
    4. Logged in to Hugging Face (`huggingface-cli login`).
    5. Git credential helper configured (`git config --global credential.helper store`).
    """
    logging.info("Starting model build and upload process...")
    
    # --- Load Configuration from JSON file ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}. Please create it.")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {config_path}. Please check its format.")
        return

    # Unpack config variables
    model_id = config["model_id"]
    local_model_dir = config["local_model_dir"]
    s3_bucket = config["s3_bucket"]
    s3_key_prefix = config["s3_key_prefix"]
    inference_script_name = config["inference_script_name"]
    tarball_name = config["tarball_name"]

    # Define paths based on configuration
    model_repo_name = model_id.split('/')[-1]
    local_model_path = os.path.join(script_dir, local_model_dir)
    cloned_repo_path = os.path.join(local_model_path, model_repo_name)
    inference_script_source_path = os.path.join(script_dir, inference_script_name)

    # Use a try...finally block to ensure cleanup happens even if a step fails
    try:
        # --- 1. Preparation and Cleanup ---
        logging.info("Preparing local directories...")
        # Remove the cloned repo if it exists from a previous failed run
        if os.path.exists(cloned_repo_path):
            logging.warning(f"Found existing directory, removing: {cloned_repo_path}")
            shutil.rmtree(cloned_repo_path)
        # Ensure the parent model directory exists
        os.makedirs(local_model_path, exist_ok=True)

        # --- 2. Download the Model ---
        logging.info("Initializing Git LFS...")
        run_command(["git", "lfs", "install"])
        
        logging.info(f"Cloning model '{model_id}' from Hugging Face...")
        model_url = f"https://huggingface.co/{model_id}"
        run_command(["git", "clone", model_url], working_dir=local_model_path)

        # --- 3. Prepare the Package ---
        logging.info("Preparing model package with custom inference code...")
        code_dir_path = os.path.join(cloned_repo_path, "code")
        os.makedirs(code_dir_path, exist_ok=True)
        
        destination_script_path = os.path.join(code_dir_path, inference_script_name)
        logging.info(f"Copying '{inference_script_source_path}' to '{destination_script_path}'")
        shutil.copy(inference_script_source_path, destination_script_path)

        # --- 4. Create the Tarball ---
        logging.info(f"Creating tarball '{tarball_name}'...")
        # The command to create the tarball. We use `*` to package all contents.
        # This is run inside the cloned repository directory using shell expansion.
        run_command("tar czf " + tarball_name + " *", working_dir=cloned_repo_path, shell=True)
        
        # --- 5. Upload to S3 ---
        local_tarball_path = os.path.join(cloned_repo_path, tarball_name)
        s3_uri = f"s3://{s3_bucket}/{s3_key_prefix}/{tarball_name}"
        logging.info(f"Uploading '{local_tarball_path}' to '{s3_uri}'...")
        run_command(["aws", "s3", "cp", local_tarball_path, s3_uri])

        logging.info("==========================================================")
        logging.info("✅ Process completed successfully!")
        logging.info(f"Model package uploaded to: {s3_uri}")
        logging.info("==========================================================")

    except subprocess.CalledProcessError as e:
        logging.error("A command failed to execute.")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Return Code: {e.returncode}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        logging.error("❌ Build process failed.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        logging.error("❌ Build process failed.")

    finally:
        # --- 6. Cleanup ---
        logging.info("Cleaning up downloaded model directory...")
        if os.path.exists(cloned_repo_path):
            shutil.rmtree(cloned_repo_path)
            logging.info(f"Successfully removed: {cloned_repo_path}")
        else:
            logging.info("Cleanup not needed, directory does not exist.")


if __name__ == "__main__":
    main()
