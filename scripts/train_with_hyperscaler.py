#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train the Legal Reasoning Model using SageMaker Hyperscaler (model parallelism)
for optimal price-performance on ml.g5.8xlarge instances.
"""

import os
import argparse
import yaml
import logging
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperscaler_training.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Legal Reasoning Model with SageMaker Hyperscaler")
    
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--train-data", type=str, required=True,
                        help="S3 URI of the training data")
    parser.add_argument("--validation-data", type=str, required=True,
                        help="S3 URI of the validation data")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Hugging Face model ID or S3 URI of the base model")
    parser.add_argument("--output-path", type=str, required=True,
                        help="S3 URI for output artifacts")
    parser.add_argument("--job-name", type=str, default="legal-reasoning-training",
                        help="Name of the SageMaker training job")
    parser.add_argument("--instance-type", type=str, default="ml.g5.8xlarge",
                        help="Instance type for training")
    parser.add_argument("--instance-count", type=int, default=1,
                        help="Number of instances for training")
    parser.add_argument("--use-spot", action="store_true",
                        help="Use spot instances for training")
    parser.add_argument("--max-wait", type=int, default=36000,
                        help="Maximum wait time for spot instances")
    parser.add_argument("--max-run", type=int, default=36000,
                        help="Maximum run time for the training job")
    
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
    
    # Set up hyperparameters
    hyperparameters = {
        # Model configuration
        "model_id": args.model_id,
        "language": config['model']['language'],
        "max_seq_length": config['data']['max_seq_length'],
        
        # Training configuration
        "epochs": config['training']['num_train_epochs'],
        "per_device_train_batch_size": 1,  # Small per-device batch size for model parallelism
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,  # Increase for effective larger batch size
        "learning_rate": config['training']['learning_rate'],
        "weight_decay": config['training']['weight_decay'],
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "warmup_steps": config['training']['warmup_steps'],
        
        # LoRA configuration
        "use_lora": str(config['training']['use_lora']).lower(),
        "lora_r": config['training']['lora_rank'],
        "lora_alpha": config['training']['lora_alpha'],
        "lora_dropout": config['training']['lora_dropout'],
        "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        
        # Hyperscaler configuration
        "model_parallel_degree": 2,  # Split model across 2 GPUs on ml.g5.8xlarge
        "ddp_dist_backend": "nccl",
        "fp16": "true",  # Use mixed precision training
        "bf16": "false",  # Use BF16 if available (A10G supports it)
        
        # Optimization
        "deepspeed_config": "ds_z3_config.json",  # Will be created in the entry point
        "torch_distributed": "true",
        
        # Checkpointing
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 2,
        
        # Evaluation
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "logging_steps": 100,
        
        # Output
        "output_dir": "/opt/ml/model"
    }
    
    # Configure distribution for model parallelism
    distribution = {
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": {
                    "partitions": 2,                   # Split model across 2 GPUs
                    "microbatches": 4,                 # Process batch in smaller chunks
                    "optimize": "speed",               # Optimize for training speed
                    "pipeline_parallel_degree": 1,     # Pipeline parallelism degree
                    "tensor_parallel_degree": 2,       # Tensor parallelism degree
                    "ddp": True,                       # Enable DDP for data parallelism
                    "placement_strategy": "cluster",   # Optimize parameter placement
                    "activation_checkpointing": True,  # Enable activation checkpointing
                    "prescaled_batch": True,           # Pre-scale batch size
                    "checkpoint_attentions": True,     # Checkpoint attention layers
                    "fast_mode": True,                 # Enable fast mode
                    "shard_optimizer_state": True      # Shard optimizer state
                }
            }
        },
        "torch_distributed": {
            "enabled": True
        }
    }
    
    # Create HuggingFace estimator
    huggingface_estimator = HuggingFace(
        entry_point="train_hyperscaler.py",  # Custom training script with model parallelism
        source_dir="src/training",           # Directory containing the training code
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        role=role,
        transformers_version="4.28.1",
        pytorch_version="2.0.0",
        py_version="py310",
        hyperparameters=hyperparameters,
        distribution=distribution,
        use_spot_instances=args.use_spot,
        max_wait=args.max_wait if args.use_spot else None,
        max_run=args.max_run,
        checkpoint_s3_uri=f"{args.output_path}/checkpoints" if args.use_spot else None,
        output_path=args.output_path,
        base_job_name=args.job_name,
        sagemaker_session=sagemaker_session
    )
    
    # Define data channels
    data_channels = {
        "train": args.train_data,
        "validation": args.validation_data
    }
    
    # Start training job
    logger.info(f"Starting training job with name: {args.job_name}")
    logger.info(f"Using instance type: {args.instance_type}")
    logger.info(f"Using spot instances: {args.use_spot}")
    logger.info(f"Model will be saved to: {args.output_path}")
    
    huggingface_estimator.fit(
        inputs=data_channels,
        job_name=args.job_name
    )
    
    logger.info(f"Training job completed. Model artifacts saved to: {args.output_path}")


if __name__ == "__main__":
    main()
