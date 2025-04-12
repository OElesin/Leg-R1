#!/bin/bash

# Script to run SageMaker Hyperscaler training for Legal Reasoning Model
# This script automates the entire process from data preparation to training

# Exit on error
set -e

# Parse arguments
BUCKET_NAME=""
PREFIX="legal-reasoning-model"
INSTANCE_TYPE="ml.g5.8xlarge"
USE_SPOT="true"
LANGUAGE="de"
CONFIG_FILE="configs/hyperscaler_config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --bucket)
      BUCKET_NAME="$2"
      shift
      shift
      ;;
    --prefix)
      PREFIX="$2"
      shift
      shift
      ;;
    --instance-type)
      INSTANCE_TYPE="$2"
      shift
      shift
      ;;
    --no-spot)
      USE_SPOT="false"
      shift
      ;;
    --language)
      LANGUAGE="$2"
      shift
      shift
      ;;
    --config)
      CONFIG_FILE="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$BUCKET_NAME" ]; then
  echo "Error: --bucket argument is required"
  echo "Usage: $0 --bucket your-bucket-name [--prefix prefix] [--instance-type ml.g5.8xlarge] [--no-spot] [--language de] [--config configs/hyperscaler_config.yaml]"
  exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
fi

# Create timestamp for job name
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
JOB_NAME="legal-reasoning-${LANGUAGE}-${TIMESTAMP}"

echo "=== Legal Reasoning Model Training with SageMaker Hyperscaler ==="
echo "Bucket: $BUCKET_NAME"
echo "Prefix: $PREFIX"
echo "Instance Type: $INSTANCE_TYPE"
echo "Use Spot: $USE_SPOT"
echo "Language: $LANGUAGE"
echo "Job Name: $JOB_NAME"
echo "Config File: $CONFIG_FILE"
echo ""

# Step 1: Prepare data
echo "Step 1: Preparing data for Hyperscaler..."
python scripts/prepare_data_for_hyperscaler.py \
  --input-file data/german/processed/all_examples.jsonl \
  --output-dir data/hyperscaler \
  --s3-bucket $BUCKET_NAME \
  --s3-prefix $PREFIX/data \
  --language $LANGUAGE

# Step 2: Launch training job
echo "Step 2: Launching SageMaker training job..."

SPOT_ARGS=""
if [ "$USE_SPOT" = "true" ]; then
  SPOT_ARGS="--use-spot --max-wait 36000"
fi

python scripts/train_with_hyperscaler.py \
  --config $CONFIG_FILE \
  --train-data s3://$BUCKET_NAME/$PREFIX/data/train \
  --validation-data s3://$BUCKET_NAME/$PREFIX/data/validation \
  --output-path s3://$BUCKET_NAME/$PREFIX/model \
  --job-name $JOB_NAME \
  --instance-type $INSTANCE_TYPE \
  $SPOT_ARGS

echo "Training job launched: $JOB_NAME"
echo "Monitor progress in the SageMaker console or with:"
echo "aws sagemaker describe-training-job --training-job-name $JOB_NAME"
echo ""

# Step 3: Wait for training to complete (optional)
echo "Step 3: Waiting for training job to complete..."
echo "Press Ctrl+C to stop waiting (the job will continue running)"
echo ""

# Wait for training job to complete
aws sagemaker wait training-job-completed-or-stopped --training-job-name $JOB_NAME

# Check if job was successful
STATUS=$(aws sagemaker describe-training-job --training-job-name $JOB_NAME --query 'TrainingJobStatus' --output text)

if [ "$STATUS" = "Completed" ]; then
  echo "Training job completed successfully!"
  
  # Step 4: Download model
  echo "Step 4: Downloading model from S3..."
  mkdir -p models/trained-$TIMESTAMP
  aws s3 cp --recursive s3://$BUCKET_NAME/$PREFIX/model models/trained-$TIMESTAMP
  
  echo "Model downloaded to: models/trained-$TIMESTAMP"
  echo ""
  
  # Step 5: Suggest next steps
  echo "Step 5: Next steps"
  echo "To evaluate the model, run:"
  echo "python scripts/evaluate_model.py --model-path models/trained-$TIMESTAMP"
  echo ""
  echo "To deploy the model to SageMaker, run:"
  echo "python scripts/deploy_model.py --model-path models/trained-$TIMESTAMP --endpoint-name legal-reasoning-$LANGUAGE"
  echo ""
  echo "To optimize for CPU deployment, run:"
  echo "python scripts/optimize_model_for_cpu.py --model-path models/trained-$TIMESTAMP --output-path models/optimized-$TIMESTAMP --quantize"
else
  echo "Training job did not complete successfully. Status: $STATUS"
  echo "Check the SageMaker console for more details."
fi
