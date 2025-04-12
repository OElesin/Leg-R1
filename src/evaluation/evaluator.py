"""
Evaluation module for Legal Reasoning Model.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error
)
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from ..model.model import LegalReasoningModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LegalModelEvaluator:
    """Evaluator for Legal Reasoning Model."""
    
    def __init__(
        self,
        model: LegalReasoningModel,
        test_data_path: str,
        output_dir: str = "./evaluation_results",
        language: str = "en"
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Legal Reasoning Model instance
            test_data_path: Path to test data
            output_dir: Directory to save evaluation results
            language: Language code (en, de)
        """
        self.model = model
        self.test_data_path = test_data_path
        self.output_dir = output_dir
        self.language = language
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        self.test_data = self._load_test_data()
        
        # Initialize metrics
        self.metrics = defaultdict(list)
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize BLEU smoothing function
        self.bleu_smoothing = SmoothingFunction().method1
    
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data."""
        logger.info(f"Loading test data from {self.test_data_path}")
        
        test_data = []
        
        # Handle different data formats
        if os.path.isdir(self.test_data_path):
            # Directory with JSONL files
            for filename in os.listdir(self.test_data_path):
                if filename.endswith('.jsonl') or filename.endswith('.json'):
                    file_path = os.path.join(self.test_data_path, filename)
                    self._load_jsonl_file(file_path, test_data)
        elif self.test_data_path.endswith('.jsonl') or self.test_data_path.endswith('.json'):
            # Single JSONL file
            self._load_jsonl_file(self.test_data_path, test_data)
        else:
            raise ValueError(f"Unsupported data format: {self.test_data_path}")
        
        logger.info(f"Loaded {len(test_data)} test examples")
        return test_data
    
    def _load_jsonl_file(self, file_path: str, data_list: List[Dict[str, Any]]) -> None:
        """Load examples from a JSONL file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    data_list.append(example)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {file_path}")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Returns:
            Dict containing evaluation metrics
        """
        logger.info("Starting evaluation")
        
        # Group examples by task
        task_examples = defaultdict(list)
        for example in self.test_data:
            task = example.get('metadata', {}).get('task', 'unknown')
            task_examples[task].append(example)
        
        # Evaluate each task separately
        for task, examples in task_examples.items():
            logger.info(f"Evaluating task: {task} with {len(examples)} examples")
            
            if task == "classification":
                self._evaluate_classification(examples)
            elif task == "summarization":
                self._evaluate_summarization(examples)
            elif task == "case_analysis":
                self._evaluate_case_analysis(examples)
            elif task == "statute_interpretation":
                self._evaluate_statute_interpretation(examples)
            else:
                logger.warning(f"Unknown task: {task}, using generic evaluation")
                self._evaluate_generic(examples, task)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics()
        
        # Save metrics
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(overall_metrics, f, indent=2)
        
        logger.info(f"Evaluation complete. Results saved to {self.output_dir}")
        return overall_metrics
    
    def _evaluate_classification(self, examples: List[Dict[str, Any]]) -> None:
        """Evaluate classification task."""
        inputs = []
        true_labels = []
        
        # Extract inputs and true labels
        for example in examples:
            # Get user input (the text to classify)
            user_content = next((msg['content'] for msg in example['conversations'] if msg['role'] == 'user'), "")
            inputs.append(user_content)
            
            # Get assistant response (the true label)
            assistant_content = next((msg['content'] for msg in example['conversations'] if msg['role'] == 'assistant'), "")
            
            # Extract category from response
            if self.language == "en":
                category_match = re.search(r'document falls under (\w+)', assistant_content)
            elif self.language == "de":
                category_match = re.search(r'Dokument fällt in die Kategorie (\w+)', assistant_content)
            else:
                category_match = re.search(r'document falls under (\w+)', assistant_content)
            
            true_label = category_match.group(1) if category_match else "unknown"
            true_labels.append(true_label)
        
        # Generate predictions
        predictions = self.model.batch_predict(inputs, task="classification")
        
        # Extract predicted labels
        predicted_labels = []
        for prediction in predictions:
            if self.language == "en":
                category_match = re.search(r'document falls under (\w+)', prediction)
            elif self.language == "de":
                category_match = re.search(r'Dokument fällt in die Kategorie (\w+)', prediction)
            else:
                category_match = re.search(r'document falls under (\w+)', prediction)
            
            predicted_label = category_match.group(1) if category_match else "unknown"
            predicted_labels.append(predicted_label)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
        
        # Store metrics
        self.metrics['classification_accuracy'].append(accuracy)
        self.metrics['classification_precision'].append(precision)
        self.metrics['classification_recall'].append(recall)
        self.metrics['classification_f1'].append(f1)
        
        # Generate confusion matrix
        self._generate_confusion_matrix(true_labels, predicted_labels, "classification")
    
    def _evaluate_summarization(self, examples: List[Dict[str, Any]]) -> None:
        """Evaluate summarization task."""
        inputs = []
        reference_summaries = []
        
        # Extract inputs and reference summaries
        for example in examples:
            # Get user input (the text to summarize)
            user_content = next((msg['content'] for msg in example['conversations'] if msg['role'] == 'user'), "")
            inputs.append(user_content)
            
            # Get assistant response (the reference summary)
            assistant_content = next((msg['content'] for msg in example['conversations'] if msg['role'] == 'assistant'), "")
            reference_summaries.append(assistant_content)
        
        # Generate predictions
        predicted_summaries = self.model.batch_predict(inputs, task="summarization")
        
        # Calculate ROUGE scores
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for reference, prediction in zip(reference_summaries, predicted_summaries):
            rouge_scores = self.rouge_scorer.score(reference, prediction)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
        
        # Calculate BLEU scores
        bleu_scores = []
        for reference, prediction in zip(reference_summaries, predicted_summaries):
            reference_tokens = reference.split()
            prediction_tokens = prediction.split()
            bleu_score = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=self.bleu_smoothing)
            bleu_scores.append(bleu_score)
        
        # Store metrics
        self.metrics['summarization_rouge1'].append(np.mean(rouge1_scores))
        self.metrics['summarization_rouge2'].append(np.mean(rouge2_scores))
        self.metrics['summarization_rougeL'].append(np.mean(rougeL_scores))
        self.metrics['summarization_bleu'].append(np.mean(bleu_scores))
    
    def _evaluate_case_analysis(self, examples: List[Dict[str, Any]]) -> None:
        """Evaluate case analysis task."""
        # Similar to summarization but with different metrics
        self._evaluate_generic(examples, "case_analysis")
    
    def _evaluate_statute_interpretation(self, examples: List[Dict[str, Any]]) -> None:
        """Evaluate statute interpretation task."""
        # Similar to summarization but with different metrics
        self._evaluate_generic(examples, "statute_interpretation")
    
    def _evaluate_generic(self, examples: List[Dict[str, Any]], task_name: str) -> None:
        """Generic evaluation for text generation tasks."""
        inputs = []
        reference_texts = []
        
        # Extract inputs and reference texts
        for example in examples:
            # Get user input
            user_content = next((msg['content'] for msg in example['conversations'] if msg['role'] == 'user'), "")
            inputs.append(user_content)
            
            # Get assistant response
            assistant_content = next((msg['content'] for msg in example['conversations'] if msg['role'] == 'assistant'), "")
            reference_texts.append(assistant_content)
        
        # Generate predictions
        predicted_texts = self.model.batch_predict(inputs, task=task_name)
        
        # Calculate ROUGE scores
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for reference, prediction in zip(reference_texts, predicted_texts):
            rouge_scores = self.rouge_scorer.score(reference, prediction)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
        
        # Store metrics
        self.metrics[f'{task_name}_rouge1'].append(np.mean(rouge1_scores))
        self.metrics[f'{task_name}_rouge2'].append(np.mean(rouge2_scores))
        self.metrics[f'{task_name}_rougeL'].append(np.mean(rougeL_scores))
    
    def _generate_confusion_matrix(self, true_labels: List[str], predicted_labels: List[str], task_name: str) -> None:
        """Generate confusion matrix."""
        # Get unique labels
        unique_labels = sorted(list(set(true_labels + predicted_labels)))
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {task_name}')
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f"{task_name}_confusion_matrix.png"))
        plt.close()
    
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall metrics."""
        overall_metrics = {}
        
        # Average metrics across all examples
        for metric_name, values in self.metrics.items():
            overall_metrics[metric_name] = np.mean(values)
        
        # Add task-specific metrics
        if 'classification_accuracy' in self.metrics:
            overall_metrics['classification_accuracy'] = np.mean(self.metrics['classification_accuracy'])
        
        if 'summarization_rouge1' in self.metrics:
            overall_metrics['summarization_rouge1'] = np.mean(self.metrics['summarization_rouge1'])
        
        return overall_metrics
