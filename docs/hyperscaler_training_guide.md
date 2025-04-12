# SageMaker Hyperscaler Training Guide

This guide explains how to train the Legal Reasoning Model using SageMaker Hyperscaler (model parallelism) on ml.g5.8xlarge instances for optimal price-performance.

## Overview

SageMaker Hyperscaler (also known as SageMaker model parallelism library) enables efficient training of large language models by distributing the model across multiple GPUs. This guide focuses on using ml.g5.8xlarge instances, which provide 2 NVIDIA A10G GPUs at a lower cost than larger instances.

## Prerequisites

- AWS account with SageMaker access
- S3 bucket for data and model artifacts
- IAM permissions for SageMaker and S3
- Processed training data in JSONL format

## Cost Comparison

| Instance Type | GPUs | vCPUs | Memory | Cost/Hour | Training Time | Total Cost |
|---------------|------|-------|--------|-----------|---------------|------------|
| ml.g5.12xlarge | 4 A10G | 48 | 192 GB | $8.64 | ~15-20 hours | ~$172.80 |
| ml.g5.8xlarge | 2 A10G | 32 | 128 GB | $5.76 | ~20-25 hours | ~$144.00 |
| ml.g5.8xlarge (spot) | 2 A10G | 32 | 128 GB | ~$1.73 | ~20-25 hours | ~$43.25 |

Using ml.g5.8xlarge with spot instances provides the best price-performance ratio, reducing costs by up to 75% compared to on-demand ml.g5.12xlarge instances.

## Step 1: Prepare Data for Hyperscaler

The first step is to prepare your data in a format optimized for SageMaker Hyperscaler.

```bash
python scripts/prepare_data_for_hyperscaler.py \
    --input-file data/german/processed/all_examples.jsonl \
    --output-dir data/hyperscaler \
    --s3-bucket your-bucket-name \
    --s3-prefix legal-reasoning-model/data \
    --language de
```

This script:
1. Loads data from the JSONL file
2. Formats it for training with Qwen2.5-7B
3. Creates train/validation splits
4. Saves the data in HuggingFace datasets format
5. Uploads the data to S3

## Step 2: Configure Hyperscaler Training

The training script (`train_with_hyperscaler.py`) is configured for optimal performance on ml.g5.8xlarge instances:

### Model Parallelism Configuration

```python
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
                "shard_optimizer_state": True      # Shard optimizer state
            }
        }
    }
}
```

### DeepSpeed ZeRO-3 Configuration

The training script also uses DeepSpeed ZeRO-3 for memory optimization:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

This configuration:
- Shards model parameters, gradients, and optimizer states across GPUs
- Offloads optimizer states and parameters to CPU when not needed
- Uses CPU pinned memory for efficient GPU-CPU transfers

## Step 3: Launch Training Job

Launch the training job with the following command:

```bash
python scripts/train_with_hyperscaler.py \
    --train-data s3://your-bucket/legal-reasoning-model/data/train \
    --validation-data s3://your-bucket/legal-reasoning-model/data/validation \
    --output-path s3://your-bucket/legal-reasoning-model/model \
    --instance-type ml.g5.8xlarge \
    --use-spot \
    --max-wait 36000
```

### Key Parameters

- `--instance-type ml.g5.8xlarge`: Uses 2 A10G GPUs
- `--use-spot`: Enables spot instances for cost savings
- `--max-wait 36000`: Maximum wait time for spot instances (10 hours)

## Step 4: Monitor Training

Monitor the training job in the SageMaker console or using the AWS CLI:

```bash
aws sagemaker describe-training-job --training-job-name legal-reasoning-training-YYYY-MM-DD-HH-MM-SS
```

You can also view logs in CloudWatch:

```bash
aws logs get-log-events --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name legal-reasoning-training-YYYY-MM-DD-HH-MM-SS/algo-1-XXXXXXXXXX
```

## Step 5: Evaluate and Deploy the Model

After training completes:

1. Download the model from S3:
   ```bash
   aws s3 cp --recursive s3://your-bucket/legal-reasoning-model/model ./models/trained
   ```

2. Evaluate the model:
   ```bash
   python scripts/evaluate_model.py --model-path models/trained
   ```

3. Deploy to a SageMaker endpoint:
   ```bash
   python scripts/deploy_model.py --model-path models/trained
   ```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size or gradient accumulation steps
   - Increase offloading to CPU
   - Enable activation checkpointing for more layers

2. **Slow Training**
   - Check GPU utilization with SageMaker Debugger
   - Optimize data loading with more workers
   - Adjust microbatch size

3. **Spot Instance Interruptions**
   - Ensure checkpoints are saved frequently
   - Set appropriate max wait time
   - Use checkpointing to resume training

## Conclusion

Using SageMaker Hyperscaler with ml.g5.8xlarge instances provides an optimal balance between cost and performance for training the Legal Reasoning Model. With spot instances, you can achieve significant cost savings while still getting good training performance.

For production deployments or faster training, you can scale up to ml.g5.12xlarge or ml.g5.16xlarge instances using the same configuration with minimal changes.
