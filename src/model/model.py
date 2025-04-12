"""
Legal Reasoning Model implementation.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LegalReasoningModel:
    """Legal Reasoning Model based on Qwen2.5-7B-Instruct."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "auto",
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        language: str = "en"
    ):
        """
        Initialize the Legal Reasoning Model.
        
        Args:
            model: Pre-trained model
            tokenizer: Tokenizer for the model
            device: Device to run the model on ('cpu', 'cuda', 'auto')
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            language: Language code (en, de)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.language = language
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initialized LegalReasoningModel on {self.device}")
        
        # Move model to device
        if self.device != "cpu":
            self.model = self.model.to(self.device)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        peft_model_path: Optional[str] = None,
        device: str = "auto",
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        language: str = "en",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ) -> "LegalReasoningModel":
        """
        Load a pre-trained Legal Reasoning Model.
        
        Args:
            model_name_or_path: Name or path of the base model
            peft_model_path: Path to PEFT adapter (optional)
            device: Device to run the model on ('cpu', 'cuda', 'auto')
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            language: Language code (en, de)
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision
            
        Returns:
            LegalReasoningModel instance
        """
        logger.info(f"Loading model from {model_name_or_path}")
        
        # Determine quantization parameters
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if load_in_4bit else None
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        
        # Load model with quantization if specified
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto" if device == "auto" else None,
            trust_remote_code=True
        )
        
        # Load PEFT adapter if specified
        if peft_model_path:
            logger.info(f"Loading PEFT adapter from {peft_model_path}")
            model = PeftModel.from_pretrained(model, peft_model_path)
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            language=language
        )
    
    @classmethod
    def load_from_path(cls, model_path: str, **kwargs) -> "LegalReasoningModel":
        """
        Load a saved Legal Reasoning Model.
        
        Args:
            model_path: Path to the saved model
            **kwargs: Additional arguments for model initialization
            
        Returns:
            LegalReasoningModel instance
        """
        # Check if it's a PEFT model
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            # Get base model path from config
            with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                config = json.load(f)
            
            base_model_path = config.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
            return cls.from_pretrained(base_model_path, peft_model_path=model_path, **kwargs)
        else:
            # Regular model
            return cls.from_pretrained(model_path, **kwargs)
    
    def save(self, output_dir: str, save_adapter_only: bool = True) -> None:
        """
        Save the model.
        
        Args:
            output_dir: Directory to save the model
            save_adapter_only: Whether to save only the adapter (for PEFT models)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if it's a PEFT model
        if hasattr(self.model, "save_pretrained") and hasattr(self.model, "is_peft_model") and self.model.is_peft_model:
            if save_adapter_only:
                logger.info(f"Saving PEFT adapter to {output_dir}")
                self.model.save_pretrained(output_dir)
            else:
                logger.info(f"Saving full PEFT model to {output_dir}")
                # Save the merged model
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(output_dir)
        else:
            logger.info(f"Saving model to {output_dir}")
            self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save model configuration
        config = {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "language": self.language
        }
        
        with open(os.path.join(output_dir, "legal_reasoning_config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    def _format_prompt(self, text: str, task: Optional[str] = None) -> str:
        """
        Format prompt for the model.
        
        Args:
            text: Input text
            task: Task type (classification, summarization, etc.)
            
        Returns:
            Formatted prompt
        """
        # Define system message based on language
        if self.language == "en":
            system_message = "You are a legal reasoning assistant trained to analyze legal documents and provide insights."
        elif self.language == "de":
            system_message = "Du bist ein juristischer Assistent, der auf die Analyse deutscher Rechtsdokumente spezialisiert ist."
        else:
            system_message = "You are a legal reasoning assistant trained to analyze legal documents and provide insights."
        
        # Define task-specific instructions
        if task == "classification":
            if self.language == "en":
                instruction = "Classify the following legal document into one of these categories: Contract Law, Criminal Law, Constitutional Law, Administrative Law, or Tort Law."
            elif self.language == "de":
                instruction = "Klassifiziere das folgende Rechtsdokument in eine der folgenden Kategorien: Zivilrecht, Strafrecht, Verwaltungsrecht, Verfassungsrecht oder Arbeitsrecht."
        elif task == "summarization":
            if self.language == "en":
                instruction = "Provide a concise summary of the following legal document, highlighting key points and holdings."
            elif self.language == "de":
                instruction = "Fasse das folgende Rechtsdokument zusammen und hebe die wichtigsten Punkte und Entscheidungen hervor."
        elif task == "case_analysis":
            if self.language == "en":
                instruction = "Analyze the following legal case and explain the legal basis for the decision."
            elif self.language == "de":
                instruction = "Analysiere den folgenden Rechtsfall und erläutere die rechtlichen Grundlagen der Entscheidung."
        else:
            # Default instruction
            if self.language == "en":
                instruction = "Analyze the following legal text and provide insights."
            elif self.language == "de":
                instruction = "Analysiere den folgenden Rechtstext und gib Einblicke."
        
        # Format as a conversation for Qwen
        formatted_text = f"<|im_start|>system\n{system_message}\n<|im_end|>\n"
        formatted_text += f"<|im_start|>user\n{instruction}\n\n{text}\n<|im_end|>\n"
        formatted_text += "<|im_start|>assistant\n"
        
        return formatted_text
    
    def predict(
        self,
        text: str,
        task: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Generate a prediction for the input text.
        
        Args:
            text: Input text
            task: Task type (classification, summarization, etc.)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (overrides instance value)
            top_p: Top-p sampling parameter (overrides instance value)
            
        Returns:
            Generated text
        """
        # Format prompt
        prompt = self._format_prompt(text, task)
        
        # Set generation parameters
        max_new_tokens = max_new_tokens or min(self.max_length, 1024)
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract assistant's response
        response = generated_text.split("<|im_start|>assistant\n")[1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        
        return response.strip()
    
    def batch_predict(
        self,
        texts: List[str],
        task: Optional[str] = None,
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Generate predictions for multiple input texts.
        
        Args:
            texts: List of input texts
            task: Task type (classification, summarization, etc.)
            batch_size: Batch size for generation
            **kwargs: Additional arguments for predict()
            
        Returns:
            List of generated texts
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = [self.predict(text, task, **kwargs) for text in batch_texts]
            results.extend(batch_results)
        
        return results
