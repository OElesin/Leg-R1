#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to deploy the Legal Reasoning Model to a CPU-based SageMaker endpoint.
"""

import os
import argparse
import yaml
import logging
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cpu_deployment.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deploy Legal Reasoning Model to CPU-based SageMaker endpoint")
    
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model-data", type=str, required=True,
                        help="S3 URI of the model data")
    parser.add_argument("--endpoint-name", type=str, default="legal-reasoning-cpu-endpoint",
                        help="Name of the SageMaker endpoint")
    parser.add_argument("--instance-type", type=str, default="ml.c5.2xlarge",
                        help="Instance type for the endpoint")
    parser.add_argument("--instance-count", type=int, default=1,
                        help="Number of instances for the endpoint")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable 8-bit quantization")
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get AWS region
    region = config['aws']['region']
    
    # Get SageMaker execution role
    session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=session)
    role = sagemaker.get_execution_role()
    
    # Set environment variables for the container
    env = {
        'USE_QUANTIZATION': 'true' if args.quantize else 'false',
        'QUANTIZATION_BITS': '8' if args.quantize else '0',
        'MAX_LENGTH': str(config['model']['max_length']),
        'TEMPERATURE': str(config['model']['temperature']),
        'TOP_P': str(config['model']['top_p']),
        'LANGUAGE': config['model']['language']
    }
    
    # Create a SageMaker model
    logger.info(f"Creating SageMaker model with data: {args.model_data}")
    
    # Path to the inference code directory
    inference_code_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "src", "inference")
    
    # Create the model
    model = Model(
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
        model_data=args.model_data,
        role=role,
        env=env,
        source_dir=inference_code_path,
        entry_point="inference.py"
    )
    
    # Deploy to a SageMaker endpoint
    logger.info(f"Deploying model to endpoint: {args.endpoint_name}")
    predictor = model.deploy(
        initial_instance_count=args.instance_count,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        container_startup_health_check_timeout=600  # Extended timeout for model loading
    )
    
    logger.info(f"Model deployed to endpoint: {args.endpoint_name}")
    logger.info(f"Endpoint URL: https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{args.endpoint_name}/invocations")
    
    # Example invocation
    logger.info("Example invocation:")
    logger.info(f"aws sagemaker-runtime invoke-endpoint --endpoint-name {args.endpoint_name} --content-type 'application/json' --body '{{\"text\": \"Sample legal text\", \"task\": \"summarization\", \"max_new_tokens\": 512}}' output.json")


if __name__ == "__main__":
    main()
