# CPU Based Inference
## 1. CPU-Optimized Inference Code

Created src/inference/cpu_inference.py with:
• Optimized model loading for CPU
• Support for 8-bit quantization to reduce memory usage
• Efficient inference with proper caching
• Support for both regular and PEFT models

## 2. SageMaker Inference Handler

Created src/inference/inference.py that:
• Implements the SageMaker inference interface
• Delegates to the CPU-optimized implementation
• Handles serialization and deserialization

## 3. Docker Container Configuration

Created src/inference/Dockerfile and src/inference/requirements.txt for:
• Setting up the container environment
• Installing necessary dependencies
• Configuring the entry point

## 4. Model Optimization Script

Created scripts/optimize_model_for_cpu.py that:
• Quantizes the model to INT8 format
• Applies CPU-specific optimizations
• Optionally exports to ONNX format for further optimization
• Merges PEFT adapters with the base model if applicable

## 5. Deployment Script

Created scripts/deploy_cpu_model.py that:
• Deploys the optimized model to a CPU-based SageMaker endpoint
• Configures environment variables for the container
• Sets up appropriate instance types and resources

## 6. Testing Notebook

Created notebooks/cpu_inference_test.ipynb to:
• Test the optimized model locally before deployment
• Benchmark performance across different tasks
• Analyze memory usage and inference speed

## 7. Deployment Guide

Created docs/cpu_deployment_guide.md that:
• Explains the optimization and deployment process
• Provides guidance on instance selection
• Offers performance considerations and monitoring tips

## How to Use

1. Optimize the model:
```bash
python scripts/optimize_model_for_cpu.py --model-path models/legal-reasoning-v1.0 --output-path models/optimized --quantize
```
   

2. Test locally:
  Run the notebooks/cpu_inference_test.ipynb notebook to verify performance

3. Deploy to SageMaker:
  
```bash
   python scripts/deploy_cpu_model.py --model-data s3://your-bucket/legal-reasoning-model/optimized/ --endpoint-name legal-reasoning-cpu --instance-type ml.c5.2xlarge
```  


4. Invoke the endpoint:
  
```python
   response = runtime.invoke_endpoint(
       EndpointName='legal-reasoning-cpu',
       ContentType='application/json',
       Body=json.dumps({"text": "Sample legal text...", "task": "summarization"})
   )
```
   


These components provide a complete solution for deploying the Legal Reasoning Model on CPU-based SageMaker instances, with optimizations to 
ensure reasonable performance despite the CPU constraints.