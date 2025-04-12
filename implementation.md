# Legal Reasoning Model Implementation Guide

This document provides detailed instructions for implementing the Legal Reasoning Model, including model architecture, training procedures, and deployment strategies.

## Table of Contents
1. [Model Architecture](#model-architecture)
2. [Training Pipeline](#training-pipeline)
3. [AWS SageMaker Setup](#aws-sagemaker-setup)
4. [Training Process](#training-process)
5. [Model Evaluation](#model-evaluation)
6. [Model Deployment](#model-deployment)
7. [Inference](#inference)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Model Architecture

### Base Model Selection
The Legal Reasoning Model is built on a transformer-based architecture, fine-tuned specifically for legal domain understanding. We use Qwen2.5-7B-Instruct as our foundation model and adapt it for legal reasoning tasks.

**Qwen2.5-7B-Instruct Advantages**:
- 7 billion parameters providing strong reasoning capabilities
- Instruction-tuned for better alignment with legal reasoning tasks
- Strong contextual understanding for complex legal documents
- Efficient token processing for long legal texts
- Multilingual capabilities for international legal applications

Our implementation leverages Qwen2.5-7B-Instruct for all legal tasks through specialized fine-tuning:
- Document classification: Qwen2.5-7B-Instruct with classification head
- Case outcome prediction: Qwen2.5-7B-Instruct with regression head
- Legal argument generation: Qwen2.5-7B-Instruct with specialized legal vocabulary

### Model Customizations
1. **Legal Vocabulary Enhancement**: We extend the Qwen2.5-7B-Instruct tokenizer vocabulary with 5,000+ legal-specific terms
2. **Parameter-Efficient Fine-Tuning (PEFT)**: Using LoRA (Low-Rank Adaptation) to efficiently fine-tune the large model
3. **Domain-Specific Embeddings**: Additional embedding layers for legal metadata (jurisdiction, court level, case age)
4. **Hierarchical Attention**: Special attention mechanisms for handling long legal documents
5. **Citation Graph Integration**: Graph neural network components to model legal citation networks

### Architecture Diagram
```
Input Text → Qwen2.5 Tokenizer → Embedding Layer → Qwen2.5-7B Transformer → Task-Specific Heads
                                    ↑                      ↑
                            Legal Metadata         Citation Graph
```

## Training Pipeline

### Data Flow
1. Raw legal documents from various sources
2. Preprocessing and cleaning (see [data-preparation.md](data-preparation.md))
3. Feature extraction and tokenization
4. Training/validation/test split
5. Model training with hyperparameter optimization
6. Evaluation on test set
7. Model export and versioning

### Training Objectives
The Qwen2.5-7B-Instruct model is trained with multiple objectives:
- **Classification Loss**: Cross-entropy for document classification tasks
- **Regression Loss**: Mean squared error for outcome prediction
- **Generation Loss**: Teacher forcing with label smoothing for text generation
- **Contrastive Loss**: For learning similarity between related legal concepts
- **Instruction Following Loss**: Special loss function to ensure the model follows legal-specific instructions

## AWS SageMaker Setup

### Prerequisites
1. **IAM Permissions**:
   - Create a role with the following policies:
     - AmazonSageMakerFullAccess
     - AmazonS3FullAccess
     - AmazonECR-FullAccess

2. **S3 Bucket Setup**:
   ```bash
   aws s3 mb s3://legal-reasoning-model-data
   ```

3. **Environment Configuration**:
   ```bash
   pip install sagemaker boto3 awscli
   aws configure
   ```

### SageMaker Project Structure
```
sagemaker/
├── code/
│   ├── train.py           # Training script
│   ├── model.py           # Model definition
│   ├── utils.py           # Utility functions
│   └── requirements.txt   # Dependencies
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
└── config/
    └── hyperparameters.json
```

## Training Process

### Preparing a Training Job

1. **Create a SageMaker Estimator for Qwen2.5-7B-Instruct**:

```python
import sagemaker
from sagemaker.huggingface import HuggingFace

role = sagemaker.get_execution_role()
bucket = 'legal-reasoning-model-data'
prefix = 'legal-model'

# Using Hugging Face estimator for Qwen2.5-7B-Instruct
estimator = HuggingFace(
    entry_point='train.py',
    source_dir='code',
    role=role,
    transformers_version='4.28.1',
    pytorch_version='2.0.0',
    py_version='py310',
    instance_count=2,
    instance_type='ml.g5.12xlarge',  # High-memory GPU instance for 7B model
    hyperparameters={
        'epochs': 3,
        'batch-size': 4,
        'learning-rate': 5e-6,
        'max-seq-length': 2048,
        'model-name': 'Qwen/Qwen2.5-7B-Instruct',
        'lora-rank': 16,
        'lora-alpha': 32,
        'lora-dropout': 0.1,
    },
    metric_definitions=[
        {'Name': 'train:loss', 'Regex': 'train_loss: ([0-9\\.]+)'},
        {'Name': 'validation:accuracy', 'Regex': 'validation_accuracy: ([0-9\\.]+)'}
    ]
)
```

2. **Launch Training Job**:

```python
data_channels = {
    'train': f's3://{bucket}/{prefix}/train',
    'validation': f's3://{bucket}/{prefix}/validation',
    'test': f's3://{bucket}/{prefix}/test'
}

estimator.fit(data_channels)
```

### Distributed Training
For the Qwen2.5-7B-Instruct model, we use SageMaker's distributed training capabilities with model parallelism:

```python
from sagemaker.huggingface import HuggingFace

distribution = {
    'torch_distributed': {
        'enabled': True
    },
    'smdistributed': {
        'modelparallel': {
            'enabled': True,
            'parameters': {
                'partitions': 2,
                'microbatches': 4,
                'placement_strategy': 'spread',
                'pipeline': 'interleaved',
                'optimize': 'speed',
                'ddp': True
            }
        }
    }
}

estimator = HuggingFace(
    # ... other parameters as above ...
    instance_count=4,  # Multiple instances for distributed training
    instance_type='ml.g5.24xlarge',  # Larger instance for model parallelism
    distribution=distribution
)
```

### Hyperparameter Tuning
We use SageMaker's hyperparameter tuning to optimize the Qwen2.5-7B-Instruct model performance:

```python
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, CategoricalParameter

hyperparameter_ranges = {
    'learning-rate': ContinuousParameter(1e-6, 1e-5, scaling_type='Logarithmic'),
    'lora-rank': CategoricalParameter([8, 16, 32]),
    'lora-alpha': CategoricalParameter([16, 32, 64]),
    'lora-dropout': ContinuousParameter(0.05, 0.2)
}

tuner = HyperparameterTuner(
    estimator,
    'validation:accuracy',
    hyperparameter_ranges,
    max_jobs=10,
    max_parallel_jobs=2,
    objective_type='Maximize'
)

tuner.fit(data_channels)
```

## Model Evaluation

### Evaluation Metrics
We evaluate the model on multiple dimensions:
- **Accuracy**: Overall correctness of predictions
- **F1 Score**: Balance between precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **BLEU/ROUGE**: For text generation quality
- **Domain-specific metrics**: Legal reasoning correctness evaluated by experts

### Evaluation Code
```python
def evaluate_model(model_path, test_data_path):
    # Load the Qwen2.5-7B-Instruct model with legal fine-tuning
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    test_data = load_dataset(test_data_path)
    
    results = {
        'accuracy': [],
        'f1': [],
        'auc': [],
        'legal_reasoning_score': []  # Domain-specific metric
    }
    
    for batch in test_data:
        # Format inputs as instructions for Qwen2.5-7B-Instruct
        inputs = [f"Analyze the following legal case and predict the outcome: {text}" 
                 for text in batch['input']]
        
        # Tokenize and generate predictions
        encoded_inputs = tokenizer(inputs, padding=True, truncation=True, 
                                  max_length=2048, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **encoded_inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        predictions = [tokenizer.decode(output, skip_special_tokens=True) 
                      for output in outputs]
        
        # Extract structured predictions from generated text
        structured_preds = extract_predictions_from_text(predictions)
        
        # Calculate metrics
        metrics = calculate_metrics(structured_preds, batch['labels'])
        for key in results:
            results[key].append(metrics[key])
    
    # Average results
    for key in results:
        results[key] = sum(results[key]) / len(results[key])
    
    return results
```

### Confusion Matrix Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
```

## Model Deployment

### SageMaker Endpoint Deployment
```python
from sagemaker.huggingface.model import HuggingFaceModel

# Create HuggingFace Model
huggingface_model = HuggingFaceModel(
    model_data=f's3://{bucket}/models/qwen-legal-model.tar.gz',
    role=role,
    transformers_version='4.28.1',
    pytorch_version='2.0.0',
    py_version='py310',
    env={
        'HF_MODEL_ID': 'Qwen/Qwen2.5-7B-Instruct',
        'HF_TASK': 'text-generation'
    }
)

# Deploy to a SageMaker endpoint with model optimizations
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g5.2xlarge',  # GPU instance for inference
    endpoint_name='legal-reasoning-endpoint',
    container_startup_health_check_timeout=600,  # Extended timeout for large model loading
)
```

### Batch Transform for Large-Scale Inference
```python
transformer = estimator.transformer(
    instance_count=1,
    instance_type='ml.c5.xlarge',
    output_path=f's3://{bucket}/batch-output/'
)

transformer.transform(
    data=f's3://{bucket}/batch-input/',
    content_type='application/json',
    split_type='Line'
)
```

### Multi-Model Endpoint for Cost Efficiency
```python
from sagemaker.multidatamodel import MultiDataModel

model_data_prefix = f's3://{bucket}/models/'
mme = MultiDataModel(
    model_data_prefix=model_data_prefix,
    image_uri=estimator.image_uri,
    role=role
)

predictor = mme.deploy(
    initial_instance_count=1,
    instance_type='ml.c5.xlarge',
    endpoint_name='legal-reasoning-mme'
)
```

## Inference

### Real-time Inference
```python
import json
import boto3

def predict(text, endpoint_name='legal-reasoning-endpoint'):
    runtime = boto3.client('runtime.sagemaker')
    
    # Format the prompt for Qwen2.5-7B-Instruct
    prompt = f"""<|im_start|>system
You are a legal reasoning assistant trained to analyze legal documents and provide insights.
<|im_end|>
<|im_start|>user
{text}
<|im_end|>
<|im_start|>assistant
"""
    
    payload = json.dumps({
        'inputs': prompt,
        'parameters': {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True
        }
    })
    
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload
    )
    
    result = json.loads(response['Body'].read().decode())
    
    # Extract the assistant's response
    generated_text = result[0]['generated_text']
    assistant_response = generated_text.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
    
    return assistant_response
```

### Batch Inference
```python
def batch_predict(input_file, output_file):
    # Prepare input data
    with open(input_file, 'r') as f:
        texts = [line.strip() for line in f]
    
    # Split into batches
    batch_size = 100
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    results = []
    for batch in batches:
        batch_results = predict_batch(batch)
        results.extend(batch_results)
    
    # Write results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
```

## Monitoring and Maintenance

### Model Monitoring
Set up SageMaker Model Monitor to track:
- Data quality
- Model quality
- Bias drift
- Feature attribution drift

```python
from sagemaker.model_monitor import DataCaptureConfig, DefaultModelMonitor

data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri=f's3://{bucket}/model-monitor'
)

model_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600
)

model_monitor.suggest_baseline(
    baseline_dataset=f's3://{bucket}/baseline/baseline.csv',
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=f's3://{bucket}/baseline/output',
    wait=True
)

monitoring_schedule_name = 'legal-model-monitoring-schedule'
model_monitor.create_monitoring_schedule(
    monitor_schedule_name=monitoring_schedule_name,
    endpoint_input=predictor.endpoint,
    statistics=model_monitor.baseline_statistics(),
    constraints=model_monitor.suggested_constraints(),
    schedule_cron_expression='cron(0 * ? * * *)'  # Hourly
)
```

### A/B Testing
```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor

# Create variant A (current model)
model_a = Model(
    model_data=f's3://{bucket}/models/model_v1.tar.gz',
    image_uri=estimator.image_uri,
    role=role
)

# Create variant B (new model)
model_b = Model(
    model_data=f's3://{bucket}/models/model_v2.tar.gz',
    image_uri=estimator.image_uri,
    role=role
)

# Create production variants
production_variants = [
    {
        'VariantName': 'ModelA',
        'ModelName': 'legal-model-v1',
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.c5.xlarge',
        'InitialVariantWeight': 0.7
    },
    {
        'VariantName': 'ModelB',
        'ModelName': 'legal-model-v2',
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.c5.xlarge',
        'InitialVariantWeight': 0.3
    }
]

# Deploy endpoint with both variants
sagemaker_client = boto3.client('sagemaker')
sagemaker_client.create_endpoint_config(
    EndpointConfigName='legal-ab-test-config',
    ProductionVariants=production_variants
)

sagemaker_client.create_endpoint(
    EndpointName='legal-ab-test-endpoint',
    EndpointConfigName='legal-ab-test-config'
)
```

### Model Updates and Versioning
```python
# Register model in SageMaker Model Registry
model_package_group_name = 'legal-reasoning-models'
model_package_description = 'Legal reasoning models for document analysis'

# Create model package group
sagemaker_client.create_model_package_group(
    ModelPackageGroupName=model_package_group_name,
    ModelPackageGroupDescription=model_package_description
)

# Register model version
model_package_input = {
    'ModelPackageGroupName': model_package_group_name,
    'ModelPackageDescription': 'Legal reasoning model v2.0',
    'InferenceSpecification': {
        'Containers': [
            {
                'Image': estimator.image_uri,
                'ModelDataUrl': f's3://{bucket}/models/model_v2.tar.gz'
            }
        ],
        'SupportedContentTypes': ['application/json'],
        'SupportedResponseMIMETypes': ['application/json']
    },
    'ModelApprovalStatus': 'PendingManualApproval'
}

create_model_package_response = sagemaker_client.create_model_package(**model_package_input)
model_package_arn = create_model_package_response['ModelPackageArn']
```

This implementation guide provides a comprehensive framework for building, training, evaluating, and deploying a legal reasoning model using AWS SageMaker. Follow these instructions to create a robust and scalable solution for legal AI applications.
