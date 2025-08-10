import json
import random
import sys
import os
import argparse
import logging
from typing import List, Dict, Any

from datasets import Dataset, DatasetDict
from huggingface_hub import HfFolder
from dotenv import load_dotenv

# --- Constants ---
HUGGINGFACE_TOKEN_ENV_VAR = "HUGGINGFACE_TOKEN"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSON file and returns its content."""
    logging.info(f"Loading data from {file_path}...")
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}")
        sys.exit(1)

def format_single_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Formats a single raw sample into one or more training samples."""
    if sample.get("wrong_response_number", 0) == 0:
        return []

    sample_kwargs = {
        "question": sample["question"],
        "answer": str(sample.get("answer", [""])[0]),
    }
    if "is_answerable" in sample:
        sample_kwargs["is_answerable"] = sample["is_answerable"]
    else:
        sample_kwargs["context"] = sample.get("context", "")
    
    formatted_samples = []
    correct_response_index = 0
    for i, is_verified in enumerate(sample.get("verified_response_mask", [])):
        if is_verified:
            formatted_samples.append({
                "input": sample["input"],
                "incorrect_response": sample["responses_to_correct"][i],
                "errors": sample["errors_to_correct"][i],
                "correct_response": sample["corrected_responses"][correct_response_index],
                "additional_info": sample_kwargs,
            })
            correct_response_index += 1
    return formatted_samples

def process_dataset(raw_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Processes the entire raw dataset."""
    logging.info("Processing raw dataset...")
    final_dataset = []
    for sample in raw_dataset:
        final_dataset.extend(format_single_sample(sample))
    logging.info(f"Generated dataset with {len(final_dataset)} samples.")
    return final_dataset

def save_dataset_locally(dataset: List[Dict[str, Any]], output_path: str):
    """Saves the processed dataset to a local JSON file."""
    logging.info(f"Saving final dataset to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
    logging.info("Dataset saved successfully.")

def push_dataset_to_hub(dataset: List[Dict[str, Any]], repo_name: str):
    """Pushes the processed dataset to the Hugging Face Hub."""
    token = os.getenv(HUGGINGFACE_TOKEN_ENV_VAR)
    if not token:
        raise ValueError(f"Hugging Face token not found. Set the {HUGGINGFACE_TOKEN_ENV_VAR} environment variable.")
    
    logging.info("Connecting to Hugging Face Hub...")
    HfFolder.save_token(token)
    
    hf_dataset = Dataset.from_list(dataset)
    dataset_dict = DatasetDict({"train": hf_dataset})
    
    logging.info(f"Pushing dataset to Hugging Face Hub repo: {repo_name}")
    dataset_dict.push_to_hub(repo_name, private=True)
    logging.info("Dataset successfully pushed to the Hub.")

# --- Main Orchestration ---

def run_data_preparation(config: Dict[str, Any], project_root: str, push_to_hub: bool):
    """Orchestrates the data preparation pipeline."""
    random.seed(config['seed'])
    
    # 1. Load and merge datasets
    math_qa_path = os.path.join(project_root, config['math_qa_dataset_path'])
    context_qa_path = os.path.join(project_root, config['context_qa_dataset_path'])
    
    math_qa_dataset = load_json_file(math_qa_path)
    context_qa_dataset = load_json_file(context_qa_path)
    
    merged_dataset = math_qa_dataset + context_qa_dataset
    logging.info(f"Total samples after merging: {len(merged_dataset)}")
    
    # 2. Process dataset
    final_dataset = process_dataset(merged_dataset)
    
    # 3. Shuffle
    random.Random(config['seed']).shuffle(final_dataset)
    
    # 4. Save or push
    if push_to_hub:
        push_dataset_to_hub(final_dataset, config['huggingface_repo'])
    else:
        output_path = os.path.join(project_root, config['output_path'])
        save_dataset_locally(final_dataset, output_path)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    """Main entry point of the script."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Prepare training data and optionally push to Hugging Face Hub.")
    parser.add_argument("--config", type=str, default="configs/config.json", help="Path to the configuration file.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the dataset to the Hugging Face Hub.")
    
    args = parser.parse_args()

    # Set up project root path
    # This makes the script runnable from anywhere
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)
    
    run_data_preparation(config, project_root, args.push_to_hub)

if __name__ == "__main__":
    main()
