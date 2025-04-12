#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to evaluate the Legal Reasoning Model.
"""

import os
import argparse
import yaml
import logging
import json

from src.model.model import LegalReasoningModel
from src.evaluation.evaluator import LegalModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Legal Reasoning Model")
    
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--test-data-path", type=str,
                        help="Path to test data (overrides config)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--language", type=str, choices=["en", "de"],
                        help="Language code (overrides config)")
    parser.add_argument("--peft-model-path", type=str,
                        help="Path to PEFT adapter (optional)")
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
    test_data_path = args.test_data_path or config['data']['test_data_path']
    language = args.language or config['model']['language']
    load_in_8bit = args.load_in_8bit or config['training']['load_in_8bit']
    load_in_4bit = args.load_in_4bit or config['training']['load_in_4bit']
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = LegalReasoningModel.from_pretrained(
        model_name_or_path=args.model_path,
        peft_model_path=args.peft_model_path,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        language=language
    )
    
    # Initialize evaluator
    evaluator = LegalModelEvaluator(
        model=model,
        test_data_path=test_data_path,
        output_dir=args.output_dir,
        language=language
    )
    
    # Evaluate model
    metrics = evaluator.evaluate()
    
    # Print metrics
    logger.info("Evaluation metrics:")
    logger.info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
