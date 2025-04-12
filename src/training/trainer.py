"""
Training module for Legal Reasoning Model.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
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
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LegalDataset(Dataset):
    """Dataset for legal data in conversation format."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file or directory
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        self.examples = []
        
        # Handle different data formats
        if os.path.isdir(data_path):
            # Directory with JSONL files
            for filename in os.listdir(data_path):
                if filename.endswith('.jsonl') or filename.endswith('.json'):
                    file_path = os.path.join(data_path, filename)
                    self._load_jsonl_file(file_path)
        elif data_path.endswith('.jsonl') or data_path.endswith('.json'):
            # Single JSONL file
            self._load_jsonl_file(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        logger.info(f"Loaded {len(self.examples)} examples")
    
    def _load_jsonl_file(self, file_path: str) -> None:
        """Load examples from a JSONL file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    self.examples.append(example)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {file_path}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        conversations = example['conversations']
        
        # Format as a conversation
        formatted_text = ""
        for message in conversations:
            role = message['role']
            content = message['content']
            
            if role == "system":
                formatted_text += f"<|im_start|>system\n{content}\n<|im_end|>\n"
            elif role == "user":
                formatted_text += f"<|im_start|>user\n{content}\n<|im_end|>\n"
            elif role == "assistant":
                formatted_text += f"<|im_start|>assistant\n{content}\n<|im_end|>\n"
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create input_ids and labels
        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]
        
        # For training, we want to predict the assistant's responses
        labels = input_ids.clone()
        
        # Set labels to -100 for non-assistant parts (we don't want to compute loss on these)
        assistant_positions = self._find_assistant_positions(formatted_text)
        for start, end in assistant_positions:
            # Convert text positions to token positions (approximate)
            token_start = min(len(self.tokenizer.encode(formatted_text[:start])), self.max_length - 1)
            token_end = min(len(self.tokenizer.encode(formatted_text[:end])), self.max_length)
            
            # Set everything outside assistant responses to -100
            labels[:token_start] = -100
            if token_end < self.max_length:
                labels[token_end:] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _find_assistant_positions(self, text: str) -> List[tuple]:
        """Find positions of assistant responses in the text."""
        positions = []
        start_marker = "<|im_start|>assistant\n"
        end_marker = "<|im_end|>"
        
        start = 0
        while True:
            start_pos = text.find(start_marker, start)
            if start_pos == -1:
                break
                
            start_pos += len(start_marker)
            end_pos = text.find(end_marker, start_pos)
            
            if end_pos == -1:
                break
                
            positions.append((start_pos, end_pos))
            start = end_pos + len(end_marker)
        
        return positions


class LegalModelTrainer:
    """Trainer for Legal Reasoning Model."""
    
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: str = "./legal_model",
        train_data_path: str = None,
        eval_data_path: str = None,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_seq_length: int = 2048,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 5e-6,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        num_train_epochs: int = 3,
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 500,
        save_total_limit: int = 3,
        bf16: bool = True,
        fp16: bool = False,
        seed: int = 42
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name_or_path: Name or path of the base model
            output_dir: Directory to save the model
            train_data_path: Path to training data
            eval_data_path: Path to evaluation data
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision
            max_seq_length: Maximum sequence length
            batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_ratio: Warmup ratio
            num_train_epochs: Number of training epochs
            logging_steps: Logging steps
            eval_steps: Evaluation steps
            save_steps: Save steps
            save_total_limit: Save total limit
            bf16: Whether to use bfloat16 precision
            fp16: Whether to use float16 precision
            seed: Random seed
        """
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.num_train_epochs = num_train_epochs
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.bf16 = bf16
        self.fp16 = fp16
        self.seed = seed
        
        # Set seed for reproducibility
        set_seed(self.seed)
        
        # Initialize model and tokenizer
        self._init_model_and_tokenizer()
    
    def _init_model_and_tokenizer(self) -> None:
        """Initialize model and tokenizer."""
        logger.info(f"Loading tokenizer from {self.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )
        
        # Add special tokens if not present
        special_tokens = ["<|im_start|>", "<|im_end|>"]
        special_tokens_dict = {}
        
        # Check if special tokens need to be added
        if not all(token in self.tokenizer.vocab for token in special_tokens):
            special_tokens_dict["additional_special_tokens"] = special_tokens
            self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # Determine quantization parameters
        quantization_config = None
        if self.load_in_8bit or self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.load_in_4bit else None
            )
        
        logger.info(f"Loading model from {self.model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Resize token embeddings if vocabulary was extended
        if len(self.tokenizer) > self.model.config.vocab_size:
            logger.info(f"Resizing token embeddings from {self.model.config.vocab_size} to {len(self.tokenizer)}")
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Prepare model for LoRA fine-tuning
        if self.use_lora:
            logger.info("Setting up LoRA fine-tuning")
            
            # Prepare model for k-bit training if using quantization
            if self.load_in_8bit or self.load_in_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # LoRA config
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            
            # Get PEFT model
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def train(self) -> None:
        """Train the model."""
        if not self.train_data_path:
            raise ValueError("Training data path not specified")
        
        # Load datasets
        logger.info("Loading datasets")
        train_dataset = LegalDataset(
            self.train_data_path,
            self.tokenizer,
            max_length=self.max_seq_length
        )
        
        eval_dataset = None
        if self.eval_data_path:
            eval_dataset = LegalDataset(
                self.eval_data_path,
                self.tokenizer,
                max_length=self.max_seq_length
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=self.logging_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.eval_steps if eval_dataset else None,
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            bf16=self.bf16,
            fp16=self.fp16,
            report_to="tensorboard",
            push_to_hub=False,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train model
        logger.info("Starting training")
        trainer.train()
        
        # Save model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save LoRA adapter separately if using LoRA
        if self.use_lora:
            logger.info(f"Saving LoRA adapter to {os.path.join(self.output_dir, 'lora_adapter')}")
            self.model.save_pretrained(os.path.join(self.output_dir, "lora_adapter"))
        
        logger.info("Training complete")
