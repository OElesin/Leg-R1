#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to prepare data for training with SageMaker Hyperscaler.
This script converts JSONL files to HuggingFace datasets and uploads them to S3.
"""

import os
import json
import argparse
import logging
import boto3
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preparation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare data for SageMaker Hyperscaler training")
    
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Directory to save processed datasets")
    parser.add_argument("--s3-bucket", type=str, required=True,
                        help="S3 bucket name")
    parser.add_argument("--s3-prefix", type=str, default="legal-reasoning-model/data",
                        help="S3 key prefix")
    parser.add_argument("--train-split", type=float, default=0.9,
                        help="Proportion of data to use for training")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--task", type=str, default=None,
                        help="Filter by specific task (e.g., 'case_analysis')")
    parser.add_argument("--language", type=str, default=None,
                        help="Filter by language (e.g., 'de')")
    
    return parser.parse_args()


def load_jsonl(file_path, max_samples=None, task=None, language=None):
    """Load data from JSONL file with optional filtering."""
    logger.info(f"Loading data from {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if max_samples is not None and i >= max_samples:
                break
                
            item = json.loads(line)
            
            # Apply filters if specified
            if task is not None and item.get('metadata', {}).get('task') != task:
                continue
                
            if language is not None and item.get('metadata', {}).get('language') != language:
                continue
                
            data.append(item)
    
    logger.info(f"Loaded {len(data)} samples")
    return data


def format_for_training(data):
    """Format data for training with SageMaker Hyperscaler."""
    logger.info("Formatting data for training")
    
    formatted_data = []
    for item in tqdm(data):
        conversations = item.get('conversations', [])
        
        # Extract system, user, and assistant messages
        system_msg = next((conv['content'] for conv in conversations if conv['role'] == 'system'), "")
        user_msg = next((conv['content'] for conv in conversations if conv['role'] == 'user'), "")
        assistant_msg = next((conv['content'] for conv in conversations if conv['role'] == 'assistant'), "")
        
        # Format as a single text for training
        formatted_text = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        formatted_text += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        formatted_text += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        
        # Create formatted item
        formatted_item = {
            'id': item.get('id', ''),
            'text': formatted_text,
            'metadata': item.get('metadata', {})
        }
        
        formatted_data.append(formatted_item)
    
    logger.info(f"Formatted {len(formatted_data)} samples")
    return formatted_data


def create_dataset(data, train_split=0.9):
    """Create HuggingFace dataset from formatted data."""
    logger.info("Creating HuggingFace dataset")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Split into train and validation
    train_size = int(len(df) * train_split)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the data
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Create dataset dictionary
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    logger.info(f"Created dataset with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    return dataset_dict


def save_and_upload(dataset_dict, output_dir, s3_bucket, s3_prefix):
    """Save dataset to disk and upload to S3."""
    logger.info(f"Saving dataset to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train and validation datasets
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    
    dataset_dict['train'].save_to_disk(train_dir)
    dataset_dict['validation'].save_to_disk(val_dir)
    
    logger.info("Dataset saved to disk")
    
    # Upload to S3
    logger.info(f"Uploading dataset to s3://{s3_bucket}/{s3_prefix}")
    
    s3 = boto3.client('s3')
    
    # Upload train dataset
    logger.info("Uploading train dataset")
    for root, _, files in os.walk(train_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, output_dir)
            s3_key = f"{s3_prefix}/{relative_path}"
            
            s3.upload_file(local_path, s3_bucket, s3_key)
    
    # Upload validation dataset
    logger.info("Uploading validation dataset")
    for root, _, files in os.walk(val_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, output_dir)
            s3_key = f"{s3_prefix}/{relative_path}"
            
            s3.upload_file(local_path, s3_bucket, s3_key)
    
    logger.info("Upload complete")
    
    # Return S3 URIs
    train_uri = f"s3://{s3_bucket}/{s3_prefix}/train"
    val_uri = f"s3://{s3_bucket}/{s3_prefix}/validation"
    
    return train_uri, val_uri


def main():
    """Main function."""
    args = parse_args()
    
    # Load data
    data = load_jsonl(args.input_file, args.max_samples, args.task, args.language)
    
    # Format data for training
    formatted_data = format_for_training(data)
    
    # Create dataset
    dataset_dict = create_dataset(formatted_data, args.train_split)
    
    # Save and upload dataset
    train_uri, val_uri = save_and_upload(dataset_dict, args.output_dir, args.s3_bucket, args.s3_prefix)
    
    logger.info(f"Training data URI: {train_uri}")
    logger.info(f"Validation data URI: {val_uri}")
    
    # Print command to start training
    logger.info("\nTo start training with SageMaker Hyperscaler, run:")
    logger.info(f"""
    python scripts/train_with_hyperscaler.py \\
        --train-data {train_uri} \\
        --validation-data {val_uri} \\
        --output-path s3://{args.s3_bucket}/{args.s3_prefix}/model \\
        --instance-type ml.g5.8xlarge \\
        --use-spot
    """)


if __name__ == "__main__":
    main()
