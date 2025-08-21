import json
import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

log_level = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

def model_fn(model_dir, context=None):
    """
    Load the 4-bit quantized model and tokenizer.
    """
    logger.info("--- Starting model_fn ---")

    # The model was quantized, so we need to load it with the correct config.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=[
            "hallucination_gate_proj",
            "hallucination_up_proj",
            "hallucination_down_proj",
            "hallucination_detector"
        ],
    )
    
    logger.info("Loading 4-bit quantized model from pretrained...")
    # Because this is a custom architecture, we must use trust_remote_code=True.
    # device_map="auto" will automatically place the model on the available GPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
    )
    logger.info("Model loaded.")
    model.eval()
    logger.info("Model set to eval mode.")
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("--- model_fn complete ---")
    
    return {"model": model, "tokenizer": tokenizer}

def input_fn(request_body, request_content_type):
    """
    Parse the request body.
    """
    logger.info("--- Starting input_fn ---")
    assert request_content_type == "application/json"
    
    logger.debug("Parsing request body...")
    data = json.loads(request_body)
    logger.debug(f"Parsed data: {data}")
    
    inputs = data.get("inputs")
    if inputs is None:
        raise ValueError("Request JSON must contain a 'inputs' field.")
    logger.debug(f"Extracted inputs: {inputs}")
        
    parameters = data.get("parameters", {})
    logger.debug(f"Extracted parameters: {parameters}")
    
    logger.info("--- input_fn complete ---")
    return {"inputs": inputs, "parameters": parameters}

def predict_fn(input_data, model_dict):
    """
    Run prediction on the processed input data.
    """
    logger.info("--- Starting predict_fn ---")
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model.device

    inputs = input_data["inputs"]
    parameters = input_data["parameters"]
    logger.info(f"Received {len(inputs)} inputs for prediction.")

    # If temperature is set for sampling, we must also set do_sample=True unless it's explicitly provided.
    if "temperature" in parameters and "do_sample" not in parameters:
        logger.info("'temperature' is set, so enabling 'do_sample=True' for generation.")
        parameters["do_sample"] = True

    logger.debug("Tokenizing inputs...")
    inputs = tokenizer(
        text=inputs, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=parameters.get("max_tokenization_length", 512),
    )
    parameters.pop("max_tokenization_length", None)
    logger.debug("Tokenization complete. Moving inputs to device...")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logger.debug(f"Inputs moved to device. Input shape: {inputs['input_ids'].shape}")
    
    logger.info("Starting model.generate()...")
    with torch.no_grad():
        outputs = model.generate(**inputs, **parameters, pad_token_id=tokenizer.eos_token_id)
    logger.info("model.generate() complete.")
        
    logger.debug("Decoding responses...")
    input_ids_len = inputs["input_ids"].shape[-1]
    responses = tokenizer.batch_decode(outputs[:, input_ids_len:], skip_special_tokens=True)
    logger.debug(f"Decoded {len(responses)} responses.")
    
    logger.info("--- predict_fn complete ---")
    return {"responses": responses}

def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result into the desired response format.
    """
    logger.info("--- Starting output_fn ---")
    assert response_content_type == "application/json"
    
    logger.debug("Serializing prediction...")
    output = json.dumps(prediction)
    logger.info("--- output_fn complete ---")
    return output 
