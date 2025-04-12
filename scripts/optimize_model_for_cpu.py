#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to optimize the Legal Reasoning Model for CPU inference.
This script quantizes the model to INT8 and exports it to ONNX format.
"""

import os
import argparse
import yaml
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTQuantizer, ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_optimization.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimize Legal Reasoning Model for CPU inference")
    
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--output-path", type=str, default="models/optimized",
                        help="Path to save the optimized model")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize the model to INT8")
    parser.add_argument("--onnx", action="store_true",
                        help="Export the model to ONNX format")
    parser.add_argument("--peft-model-path", type=str,
                        help="Path to PEFT adapter (optional)")
    
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
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Step 1: Load the model
    logger.info(f"Loading model from {args.model_path}")
    
    # Check if it's a PEFT model
    if args.peft_model_path:
        from peft import PeftModel
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(
            base_model,
            args.peft_model_path,
            torch_dtype=torch.float32  # Use float32 for CPU
        )
        
        # Merge model and adapter for better performance
        logger.info("Merging PEFT adapter with base model")
        model = model.merge_and_unload()
    else:
        # Load regular model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Step 2: Apply CPU optimizations
    logger.info("Applying CPU optimizations")
    
    # Move model to CPU explicitly
    model = model.to("cpu")
    
    # Set model to evaluation mode
    model.eval()
    
    # Step 3: Quantize the model if requested
    if args.quantize:
        logger.info("Quantizing model to INT8")
        
        try:
            # Save model to temporary directory for quantization
            temp_dir = os.path.join(args.output_path, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            
            # Initialize quantizer
            quantizer = ORTQuantizer.from_pretrained(temp_dir)
            
            # Define quantization configuration
            qconfig = AutoQuantizationConfig.for_quantization(
                quantization_approach="dynamic",
                per_channel=False,
                operators_to_quantize=["MatMul", "Add", "Conv"]
            )
            
            # Quantize the model
            quantizer.quantize(
                save_dir=args.output_path,
                quantization_config=qconfig
            )
            
            logger.info(f"Quantized model saved to {args.output_path}")
        except Exception as e:
            logger.error(f"Error during quantization: {e}")
            logger.info("Falling back to PyTorch quantization")
            
            # Fallback to PyTorch quantization
            from torch.quantization import quantize_dynamic
            
            # Quantize the model
            quantized_model = quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            # Save the quantized model
            quantized_model.save_pretrained(args.output_path)
            tokenizer.save_pretrained(args.output_path)
            
            logger.info(f"PyTorch quantized model saved to {args.output_path}")
    
    # Step 4: Export to ONNX if requested
    if args.onnx:
        logger.info("Exporting model to ONNX format")
        
        try:
            # Export to ONNX
            from optimum.onnxruntime import ORTModelForCausalLM
            
            # Create ONNX directory
            onnx_dir = os.path.join(args.output_path, "onnx")
            os.makedirs(onnx_dir, exist_ok=True)
            
            # Export the model to ONNX
            ort_model = ORTModelForCausalLM.from_pretrained(
                args.model_path if not args.quantize else args.output_path,
                export=True,
                provider="CPUExecutionProvider"
            )
            
            # Save the ONNX model
            ort_model.save_pretrained(onnx_dir)
            tokenizer.save_pretrained(onnx_dir)
            
            logger.info(f"ONNX model saved to {onnx_dir}")
        except Exception as e:
            logger.error(f"Error during ONNX export: {e}")
    
    # Step 5: Save the model if not already saved
    if not args.quantize and not args.onnx:
        logger.info(f"Saving optimized model to {args.output_path}")
        model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
    
    # Copy configuration file
    if os.path.exists(os.path.join(args.model_path, "legal_reasoning_config.json")):
        import shutil
        shutil.copy(
            os.path.join(args.model_path, "legal_reasoning_config.json"),
            os.path.join(args.output_path, "legal_reasoning_config.json")
        )
    
    logger.info("Model optimization complete")


if __name__ == "__main__":
    main()
