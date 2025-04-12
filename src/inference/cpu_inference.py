"""
CPU-optimized inference code for Legal Reasoning Model.
This module provides optimized code for deploying the model on CPU-based instances.
"""

import os
import json
import logging
import torch
from typing import Dict, List, Any, Tuple, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables to store model and tokenizer
model = None
tokenizer = None

def model_fn(model_dir: str) -> Tuple:
    """
    Load the model for inference.
    
    Args:
        model_dir: Directory where model artifacts are stored
        
    Returns:
        Tuple of model and tokenizer
    """
    global model, tokenizer
    
    logger.info(f"Loading model from {model_dir}")
    
    # Check if model is already loaded
    if model is not None and tokenizer is not None:
        return model, tokenizer
    
    # Load configuration
    config_path = os.path.join(model_dir, "legal_reasoning_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        language = config.get("language", "en")
    else:
        language = "en"
        logger.warning("No configuration file found, using default language: en")
    
    # Check if it's a PEFT model
    is_peft_model = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    
    # Determine if we should use quantization
    use_quantization = os.environ.get("USE_QUANTIZATION", "true").lower() == "true"
    quantization_bits = int(os.environ.get("QUANTIZATION_BITS", "8"))
    
    # Configure quantization
    quantization_config = None
    if use_quantization:
        logger.info(f"Using {quantization_bits}-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=quantization_bits == 8,
            load_in_4bit=quantization_bits == 4,
            bnb_4bit_compute_dtype=torch.float32,  # Use float32 for CPU
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    # Load model with CPU optimizations
    if is_peft_model:
        # For PEFT models, we need to load the base model first
        from peft import PeftModel, PeftConfig
        
        # Get base model path from config
        peft_config = PeftConfig.from_pretrained(model_dir)
        base_model_path = peft_config.base_model_name_or_path
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(
            base_model,
            model_dir,
            torch_dtype=torch.float32  # Use float32 for CPU
        )
    else:
        # Load regular model
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    
    # Move model to CPU explicitly
    model = model.to("cpu")
    
    # Set model to evaluation mode
    model.eval()
    
    # Apply additional CPU optimizations
    if hasattr(model, "config") and hasattr(model.config, "pretraining_tp"):
        # Enable tensor parallelism if the model supports it
        model.config.pretraining_tp = 1
    
    # Enable model optimizations for inference
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    
    # Log model size
    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model loaded with {model_size:.2f}M parameters")
    
    return model, tokenizer


def input_fn(request_body: str, request_content_type: str) -> Dict:
    """
    Deserialize and prepare the prediction input.
    
    Args:
        request_body: The request body
        request_content_type: The request content type
        
    Returns:
        Dictionary containing the request data
    """
    logger.info(f"Received request with content type: {request_content_type}")
    
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: Dict, model_and_tokenizer: Tuple) -> Dict:
    """
    Generate a prediction for the input data.
    
    Args:
        input_data: Input data in dictionary format
        model_and_tokenizer: Tuple of model and tokenizer
        
    Returns:
        Dictionary containing the prediction results
    """
    model, tokenizer = model_and_tokenizer
    
    # Extract parameters from input
    text = input_data.get("text", "")
    task = input_data.get("task", None)
    max_new_tokens = input_data.get("max_new_tokens", 512)
    temperature = input_data.get("temperature", 0.7)
    top_p = input_data.get("top_p", 0.9)
    language = input_data.get("language", "en")
    
    # Format prompt based on language and task
    if language == "en":
        system_message = "You are a legal reasoning assistant trained to analyze legal documents and provide insights."
    elif language == "de":
        system_message = "Du bist ein juristischer Assistent, der auf die Analyse deutscher Rechtsdokumente spezialisiert ist."
    else:
        system_message = "You are a legal reasoning assistant trained to analyze legal documents and provide insights."
    
    # Define task-specific instructions
    if task == "classification":
        if language == "en":
            instruction = "Classify the following legal document into one of these categories: Contract Law, Criminal Law, Constitutional Law, Administrative Law, or Tort Law."
        elif language == "de":
            instruction = "Klassifiziere das folgende Rechtsdokument in eine der folgenden Kategorien: Zivilrecht, Strafrecht, Verwaltungsrecht, Verfassungsrecht oder Arbeitsrecht."
    elif task == "summarization":
        if language == "en":
            instruction = "Provide a concise summary of the following legal document, highlighting key points and holdings."
        elif language == "de":
            instruction = "Fasse das folgende Rechtsdokument zusammen und hebe die wichtigsten Punkte und Entscheidungen hervor."
    elif task == "case_analysis":
        if language == "en":
            instruction = "Analyze the following legal case and explain the legal basis for the decision."
        elif language == "de":
            instruction = "Analysiere den folgenden Rechtsfall und erläutere die rechtlichen Grundlagen der Entscheidung."
    else:
        # Default instruction
        if language == "en":
            instruction = "Analyze the following legal text and provide insights."
        elif language == "de":
            instruction = "Analysiere den folgenden Rechtstext und gib Einblicke."
    
    # Format as a conversation for Qwen
    formatted_text = f"<|im_start|>system\n{system_message}\n<|im_end|>\n"
    formatted_text += f"<|im_start|>user\n{instruction}\n\n{text}\n<|im_end|>\n"
    formatted_text += "<|im_start|>assistant\n"
    
    # Tokenize input
    inputs = tokenizer(formatted_text, return_tensors="pt")
    
    # Generate with optimized settings for CPU
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,  # Enable KV caching for efficiency
                num_beams=1,  # Disable beam search for speed
                early_stopping=True
            )
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return {"error": str(e)}
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract assistant's response
    response = generated_text.split("<|im_start|>assistant\n")[1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    
    return {"response": response.strip()}


def output_fn(prediction: Dict, response_content_type: str) -> Tuple[str, str]:
    """
    Serialize the prediction result.
    
    Args:
        prediction: The prediction result
        response_content_type: The response content type
        
    Returns:
        Tuple of response body and content type
    """
    if response_content_type == "application/json":
        return json.dumps(prediction), response_content_type
    else:
        return json.dumps(prediction), "application/json"
