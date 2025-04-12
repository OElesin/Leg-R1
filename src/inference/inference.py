"""
Inference handler for SageMaker deployment.
This is the main entry point for SageMaker to run inference.
"""

import os
import json
import logging
from typing import Dict, Tuple

# Import CPU-optimized inference code
from cpu_inference import model_fn, input_fn, predict_fn, output_fn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# These functions will be used by SageMaker for inference
# They simply delegate to the CPU-optimized implementations

def model_fn(model_dir: str) -> Tuple:
    """Load the model for inference."""
    from cpu_inference import model_fn as cpu_model_fn
    return cpu_model_fn(model_dir)

def input_fn(request_body: str, request_content_type: str) -> Dict:
    """Deserialize and prepare the prediction input."""
    from cpu_inference import input_fn as cpu_input_fn
    return cpu_input_fn(request_body, request_content_type)

def predict_fn(input_data: Dict, model_and_tokenizer: Tuple) -> Dict:
    """Generate a prediction for the input data."""
    from cpu_inference import predict_fn as cpu_predict_fn
    return cpu_predict_fn(input_data, model_and_tokenizer)

def output_fn(prediction: Dict, response_content_type: str) -> Tuple[str, str]:
    """Serialize the prediction result."""
    from cpu_inference import output_fn as cpu_output_fn
    return cpu_output_fn(prediction, response_content_type)
