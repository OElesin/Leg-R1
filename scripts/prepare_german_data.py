#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to prepare German legal data from www.rechtsprechung-im-internet.de.
"""

import os
import argparse
import logging
import requests
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
import boto3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("german_data_prep.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare German Legal Data")
    
    parser.add_argument("--download-dir", default="data/german/raw", 
                        help="Directory to save downloaded documents")
    parser.add_argument("--processed-dir", default="data/german/processed", 
                        help="Directory to save processed documents")
    parser.add_argument("--output-dir", default="data/german/training", 
                        help="Directory to save training data")
    parser.add_argument("--limit", type=int, default=1000, 
                        help="Maximum number of documents to download")
    parser.add_argument("--s3-bucket", default="legal-reasoning-model-data", 
                        help="S3 bucket name")
    parser.add_argument("--s3-prefix", default="german-legal-data", 
                        help="S3 key prefix")
    parser.add_argument("--upload-to-s3", action="store_true", 
                        help="Upload data to S3")
    parser.add_argument("--court-filter", nargs='+', 
                        help="Filter by courts")
    parser.add_argument("--year-filter", nargs='+', 
                        help="Filter by years")
    
    return parser.parse_args()


def download_toc():
    """Download the table of contents XML file."""
    logger.info("Downloading TOC from www.rechtsprechung-im-internet.de/rii-toc.xml")
    toc_url = "https://www.rechtsprechung-im-internet.de/rii-toc.xml"
    
    try:
        response = requests.get(toc_url)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        logger.info(f"Successfully downloaded TOC with {len(list(root))} entries")
        return root
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download TOC: {e}")
        raise


def main():
    """Main function."""
    args = parse_args()
    
    # Create directories
    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download TOC
    toc_root = download_toc()
    
    # Save TOC for reference
    with open(os.path.join(args.download_dir, "rii-toc.xml"), "wb") as f:
        f.write(ET.tostring(toc_root))
    
    logger.info("TOC downloaded and saved. Use the full data preparation pipeline from german-legal-data-preparation.md to process the data.")
    
    # Note: The full implementation would include:
    # 1. Downloading individual documents
    # 2. Preprocessing the documents
    # 3. Formatting for training
    # 4. Creating train/validation/test splits
    # 5. Uploading to S3
    # See german-legal-data-preparation.md for the complete implementation


if __name__ == "__main__":
    main()
