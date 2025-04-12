#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train the Legal Reasoning Model.
"""

import os
import argparse
import yaml
import logging

from src.training.trainer import LegalModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Legal Reasoning Model")
    
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model-name", type=str,
                        help="Name or path of the base model (overrides config)")
    parser.add_argument("--output-dir", type=str,
                        help="Directory to save the model (overrides config)")
    parser.add_argument("--train-data-path", type=str,
                        help="Path to training data (overrides config)")
    parser.add_argument("--eval-data-path", type=str,
                        help="Path to evaluation data (overrides config)")
    parser.add_argument("--language", type=str, choices=["en", "de"],
                        help="Language code (overrides config)")
    parser.add_argument("--no-lora", action="store_true",
                        help="Disable LoRA fine-tuning")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit precision")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit precision")
    
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
    model_name = args.model_name or config['model']['name']
    output_dir = args.output_dir or "models/legal_reasoning_model"
    train_data_path = args.train_data_path or config['data']['train_data_path']
    eval_data_path = args.eval_data_path or config['data']['eval_data_path']
    language = args.language or config['model']['language']
    use_lora = not args.no_lora if args.no_lora else config['training']['use_lora']
    load_in_8bit = args.load_in_8bit or config['training']['load_in_8bit']
    load_in_4bit = args.load_in_4bit or config['training']['load_in_4bit']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = LegalModelTrainer(
        model_name_or_path=model_name,
        output_dir=output_dir,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        use_lora=use_lora,
        lora_rank=config['training']['lora_rank'],
        lora_alpha=config['training']['lora_alpha'],
        lora_dropout=config['training']['lora_dropout'],
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        max_seq_length=config['data']['max_seq_length'],
        batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        num_train_epochs=config['training']['num_train_epochs'],
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        bf16=config['training']['bf16'],
        fp16=config['training']['fp16'],
        seed=config['training']['seed']
    )
    
    # Train model
    trainer.train()


if __name__ == "__main__":
    main()
