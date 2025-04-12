# CPU Deployment Guide for Legal Reasoning Model

This guide explains how to deploy the Legal Reasoning Model on CPU-based SageMaker instances for cost-effective inference.

## Overview

While large language models like Qwen2.5-7B typically run best on GPU instances, there are scenarios where CPU deployment makes sense:

- Development and testing environments
- Low-traffic applications
- Cost-sensitive deployments
- Batch processing with relaxed latency requirements

This guide walks through the process of optimizing and deploying the Legal Reasoning Model on CPU instances.

## Prerequisites

- Trained Legal Reasoning Model (fine-tuned Qwen2.5-7B)
- AWS account with SageMaker access
- S3 bucket for model artifacts
- IAM permissions for SageMaker and ECR

## Step 1: Optimize the Model for CPU

The first step is to optimize the model for CPU inference by applying quantization and other optimizations.

```bash
python scripts/optimize_model_for_cpu.py \
    --model-path models/legal-reasoning-v1.0 \
    --output-path models/optimized \
    --quantize
```

This script:
- Loads the trained model
- Applies 8-bit quantization to reduce memory usage
- Optimizes the model for CPU inference
- Saves the optimized model to the specified output path

### Optimization Techniques

The script applies several optimization techniques:

1. **8-bit Quantization**: Reduces model precision from FP32 to INT8
2. **CPU-specific Optimizations**: Sets appropriate flags for CPU inference
3. **Low Memory Usage**: Configures the model for efficient memory usage
4. **PEFT Adapter Merging**: If using PEFT adapters, merges them with the base model

## Step 2: Test the Optimized Model Locally

Before deploying to SageMaker, it's a good idea to test the optimized model locally.

```bash
jupyter notebook notebooks/cpu_inference_test.ipynb
```

This notebook:
- Loads the optimized model
- Tests inference on sample legal documents
- Benchmarks performance across different tasks
- Analyzes memory usage

## Step 3: Deploy to SageMaker

Once you've verified the model works correctly, deploy it to a CPU-based SageMaker endpoint.

```bash
# Upload the optimized model to S3
aws s3 cp --recursive models/optimized s3://your-bucket/legal-reasoning-model/optimized/

# Deploy to SageMaker
python scripts/deploy_cpu_model.py \
    --model-data s3://your-bucket/legal-reasoning-model/optimized/ \
    --endpoint-name legal-reasoning-cpu \
    --instance-type ml.c5.2xlarge
```

### Choosing the Right Instance Type

For CPU-based deployment, consider these instance types:

- **ml.c5.xlarge**: 4 vCPU, 8 GiB memory (good for testing)
- **ml.c5.2xlarge**: 8 vCPU, 16 GiB memory (balanced)
- **ml.c5.4xlarge**: 16 vCPU, 32 GiB memory (better performance)
- **ml.m5.4xlarge**: 16 vCPU, 64 GiB memory (memory-optimized)

Choose based on your performance requirements and budget constraints.

## Step 4: Invoke the Endpoint

Once deployed, you can invoke the endpoint using the AWS SDK or CLI:

```python
import boto3
import json

# Create a SageMaker runtime client
runtime = boto3.client('sagemaker-runtime')

# Prepare the input
input_data = {
    "text": "Sample legal text...",
    "task": "summarization",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "language": "en"
}

# Invoke the endpoint
response = runtime.invoke_endpoint(
    EndpointName='legal-reasoning-cpu',
    ContentType='application/json',
    Body=json.dumps(input_data)
)

# Parse the response
result = json.loads(response['Body'].read().decode())
print(result['response'])
```

## Performance Considerations

When running on CPU, keep these considerations in mind:

1. **Latency**: Expect higher latency compared to GPU instances
2. **Throughput**: Lower tokens-per-second generation rate
3. **Batch Size**: Keep batch size small (1-2) for best performance
4. **Sequence Length**: Limit input sequence length to improve performance
5. **Concurrent Requests**: CPU instances can handle multiple concurrent requests

## Monitoring and Optimization

After deployment, monitor your endpoint's performance:

```bash
# View CloudWatch metrics for the endpoint
aws cloudwatch get-metric-statistics \
    --namespace AWS/SageMaker \
    --metric-name CPUUtilization \
    --dimensions Name=EndpointName,Value=legal-reasoning-cpu \
    --start-time 2025-04-12T00:00:00Z \
    --end-time 2025-04-12T23:59:59Z \
    --period 300 \
    --statistics Average
```

If performance is insufficient, consider:
1. Upgrading to a larger CPU instance
2. Switching to a GPU instance for production use
3. Further optimizing the model with distillation techniques
4. Implementing caching for common queries

## Conclusion

CPU-based deployment of the Legal Reasoning Model provides a cost-effective option for development, testing, and low-traffic applications. By applying quantization and optimization techniques, you can achieve reasonable performance while keeping costs down.
