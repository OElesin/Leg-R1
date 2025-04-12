#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to deploy the Legal Reasoning Model to SageMaker.
"""

import os
import argparse
import yaml
import logging
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deployment.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deploy Legal Reasoning Model to SageMaker")
    
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model-data", type=str, required=True,
                        help="S3 URI of the model data")
    parser.add_argument("--endpoint-name", type=str, default="legal-reasoning-endpoint",
                        help="Name of the SageMaker endpoint")
    parser.add_argument("--instance-type", type=str,
                        help="Instance type for the endpoint (overrides config)")
    parser.add_argument("--instance-count", type=int, default=1,
                        help="Number of instances for the endpoint")
    parser.add_argument("--language", type=str, choices=["en", "de"],
                        help="Language code (overrides config)")
    
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
    
    # Override configuration with command line arguments
    language = args.language or config['model']['language']
    instance_type = args.instance_type or config['aws']['instance_type']
    
    # Get AWS region
    region = config['aws']['region']
    
    # Get SageMaker execution role
    session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=session)
    role = sagemaker.get_execution_role()
    
    # Set environment variables based on language
    env = {
        'HF_MODEL_ID': 'Qwen/Qwen2.5-7B-Instruct',
        'HF_TASK': 'text-generation',
        'LANGUAGE': language,
        'MAX_LENGTH': str(config['model']['max_length']),
        'TEMPERATURE': str(config['model']['temperature']),
        'TOP_P': str(config['model']['top_p'])
    }
    
    # Create HuggingFace Model
    huggingface_model = HuggingFaceModel(
        model_data=args.model_data,
        role=role,
        transformers_version='4.28.1',
        pytorch_version='2.0.0',
        py_version='py310',
        env=env
    )
    
    # Deploy to a SageMaker endpoint
    logger.info(f"Deploying model to endpoint: {args.endpoint_name}")
    predictor = huggingface_model.deploy(
        initial_instance_count=args.instance_count,
        instance_type=instance_type,
        endpoint_name=args.endpoint_name,
        container_startup_health_check_timeout=600  # Extended timeout for large model loading
    )
    
    logger.info(f"Model deployed to endpoint: {args.endpoint_name}")
    logger.info(f"Endpoint URL: https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{args.endpoint_name}/invocations")


if __name__ == "__main__":
    main()
