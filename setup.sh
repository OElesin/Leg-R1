#!/bin/bash

# Setup script for Legal Reasoning Model project

# Create directories if they don't exist
mkdir -p data/german/raw
mkdir -p data/german/processed
mkdir -p data/german/samples
mkdir -p models

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Virtual environment is now active."
echo "To deactivate the virtual environment, run 'deactivate'."
