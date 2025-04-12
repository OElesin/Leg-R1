#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for Legal Reasoning Model using SageMaker Hyperscaler (model parallelism).
This script is optimized for ml.g5.8xlarge instances to maximize cost efficiency.
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# Import SageMaker model parallelism library
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.nn import FusedLayerNorm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse arguments passed from the SageMaker estimator."""
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation-dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    
    # Model parameters
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    
    # LoRA parameters
    parser.add_argument("--use_lora", type=str, default="true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    
    # Hyperscaler parameters
    parser.add_argument("--model_parallel_degree", type=int, default=2)
    parser.add_argument("--fp16", type=str, default="true")
    parser.add_argument("--bf16", type=str, default="false")
    
    # Parse arguments
    args, _ = parser.parse_known_args()
    
    # Convert string to boolean
    args.use_lora = args.use_lora.lower() == "true"
    args.fp16 = args.fp16.lower() == "true"
    args.bf16 = args.bf16.lower() == "true"
    
    return args


def create_deepspeed_config():
    """Create DeepSpeed configuration for ZeRO-3 optimization."""
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "sub_group_size": 1e9,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True
        },
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": "auto"
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": "auto",
            "synchronize_checkpoint_boundary": True,
            "profile": False
        }
    }
    
    return config


def initialize_hyperscaler():
    """Initialize SageMaker model parallelism."""
    # Initialize SMP
    smp.init()
    
    # Set PyTorch CUDA device
    torch.cuda.set_device(smp.local_rank())
    
    # Set random seed for reproducibility
    set_seed(42 + smp.rank())
    
    # Log configuration
    logger.info(f"SMP configuration: {smp.get_config()}")
    logger.info(f"SMP rank: {smp.rank()}, local_rank: {smp.local_rank()}, size: {smp.size()}")
    logger.info(f"SMP model_parallel_rank: {smp.mp_rank()}, model_parallel_size: {smp.mp_size()}")
    logger.info(f"SMP dp_rank: {smp.dp_rank()}, dp_size: {smp.dp_size()}")
    
    # Create DeepSpeed config file
    if smp.rank() == 0:
        ds_config = create_deepspeed_config()
        with open("ds_z3_config.json", "w") as f:
            json.dump(ds_config, f, indent=4)


def load_datasets(train_dir, validation_dir):
    """Load datasets from disk."""
    logger.info(f"Loading training dataset from {train_dir}")
    train_dataset = load_from_disk(train_dir)
    
    logger.info(f"Loading validation dataset from {validation_dir}")
    validation_dataset = load_from_disk(validation_dir)
    
    return train_dataset, validation_dataset


def load_model_and_tokenizer(args):
    """Load model and tokenizer with SMP optimizations."""
    logger.info(f"Loading model: {args.model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure model loading with SMP
    with smp.tensor_parallelism():
        # Load model with appropriate precision
        torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        
        # Load model with quantization for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            use_cache=False  # Disable KV cache during training
        )
        
        # Replace layer norm with fused layer norm for better performance
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                setattr(model, name, FusedLayerNorm(module.normalized_shape, 
                                                   eps=module.eps, 
                                                   elementwise_affine=module.elementwise_affine))
        
        # Apply LoRA if enabled
        if args.use_lora:
            logger.info("Applying LoRA adapters")
            
            # Prepare model for k-bit training if using quantization
            model = prepare_model_for_kbit_training(model)
            
            # Configure LoRA
            target_modules = args.lora_target_modules.split(",")
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA adapters
            model = get_peft_model(model, lora_config)
            
            # Log trainable parameters
            model.print_trainable_parameters()
    
    # Wrap model with SMP DistributedModel
    model = smp.DistributedModel(model)
    
    return model, tokenizer


def train(args):
    """Train the model using SageMaker Hyperscaler."""
    # Initialize SageMaker model parallelism
    initialize_hyperscaler()
    
    # Load datasets
    train_dataset, validation_dataset = load_datasets(args.train_dir, args.validation_dir)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=f"{args.model_dir}/logs",
        logging_steps=100,
        report_to="tensorboard",
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed="ds_z3_config.json",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        group_by_length=True,  # Group similar length sequences for efficiency
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False  # Keep all columns for custom processing
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    if smp.rank() == 0:
        logger.info(f"Saving model to {args.model_dir}")
        
        # For LoRA models, save adapter config
        if args.use_lora:
            # Unwrap model from SMP
            unwrapped_model = model.module()
            
            # Save adapter
            unwrapped_model.save_pretrained(args.model_dir)
            
            # Save model config
            model_config = {
                "base_model_id": args.model_id,
                "language": args.language,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target_modules": args.lora_target_modules
            }
            
            with open(os.path.join(args.model_dir, "legal_reasoning_config.json"), "w") as f:
                json.dump(model_config, f, indent=4)
        else:
            # For full model, save the entire model
            unwrapped_model = model.module()
            unwrapped_model.save_pretrained(args.model_dir)
        
        # Save tokenizer
        tokenizer.save_pretrained(args.model_dir)
        
        logger.info("Model saved successfully")


def main():
    """Main function."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
