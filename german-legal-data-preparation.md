# German Legal Database Preparation Guide

This guide provides detailed instructions for preparing data from www.rechtsprechung-im-internet.de for training the Legal Reasoning Model based on Qwen2.5-7B-Instruct.

## Overview

The German legal database at www.rechtsprechung-im-internet.de contains a large collection of German court decisions in XML format. This guide outlines the process of:

1. Downloading the database's table of contents (TOC)
2. Analyzing the structure and content of the database
3. Downloading individual case documents
4. Preprocessing the German legal texts
5. Formatting the data for fine-tuning Qwen2.5-7B-Instruct
6. Creating train/validation/test splits
7. Uploading the prepared data to Amazon S3 for SageMaker training

## Prerequisites

- Python 3.8+
- AWS account with S3 access
- Sufficient storage space (at least 10GB recommended)
- Required Python packages:
  - requests
  - beautifulsoup4
  - tqdm
  - boto3
  - scikit-learn
  - transformers
  - peft (for training)

## Step 1: Understanding the Data Source

The www.rechtsprechung-im-internet.de database provides German court decisions through:

- A main XML table of contents file: `rii-toc.xml`
- Individual XML files for each court decision
- Structured metadata including court, date, case number, etc.
- Full text of decisions with typical German legal document structure

## Step 2: Data Collection

### Downloading the TOC

The first step is to download and parse the table of contents:

```bash
python german_legal_data_prep.py --download-dir german_legal_data/raw
```

This will:
- Download the TOC from www.rechtsprechung-im-internet.de/rii-toc.xml
- Parse the XML structure
- Analyze the content (document types, courts, years)
- Save statistics to `german_legal_data/processed/toc_stats.json`

### Filtering Options

You can filter the documents to download by:

- Court: `--court-filter BGH BVerfG`
- Year: `--year-filter 2020 2021 2022`
- Document type: `--doc-type-filter Urteil Beschluss`

Example:
```bash
python german_legal_data_prep.py --download-dir german_legal_data/raw --court-filter BGH --year-filter 2020 2021 --limit 1000
```

## Step 3: German-Specific Preprocessing

The preprocessing pipeline handles German legal texts with specialized functions:

1. **Document Structure Parsing**:
   - Extracts metadata (court, date, case number)
   - Identifies document sections (Tatbestand, Entscheidungsgründe, etc.)
   - Preserves legal citations and references

2. **Text Cleaning**:
   - Normalizes German characters (umlauts, ß)
   - Handles German quotation marks („")
   - Removes line numbers and formatting artifacts
   - Standardizes paragraph symbols (§)

3. **Section Extraction**:
   - Tatbestand (facts of the case)
   - Entscheidungsgründe (grounds for decision)
   - Tenor (ruling)
   - Leitsätze (headnotes/guiding principles)

## Step 4: Data Formatting for Qwen2.5-7B-Instruct

The data is formatted as instruction-response pairs in German:

```
<|im_start|>system
Du bist ein juristischer Assistent, der auf die Analyse deutscher Rechtsdokumente spezialisiert ist.
<|im_end|>
<|im_start|>user
[GERMAN_INSTRUCTION]
[GERMAN_LEGAL_TEXT]
<|im_end|>
<|im_start|>assistant
[GERMAN_RESPONSE]
<|im_end|>
```

### Task-Specific Templates

1. **Classification**:
   ```
   <|im_start|>user
   Klassifiziere das folgende Rechtsdokument in eine der folgenden Kategorien: Zivilrecht, Strafrecht, Verwaltungsrecht, Verfassungsrecht oder Arbeitsrecht.
   
   [LEGAL_TEXT]
   <|im_end|>
   ```

2. **Summarization**:
   ```
   <|im_start|>user
   Fasse das folgende Rechtsdokument zusammen und hebe die wichtigsten Punkte und Entscheidungen hervor.
   
   [LEGAL_TEXT]
   <|im_end|>
   ```

3. **Case Analysis**:
   ```
   <|im_start|>user
   Analysiere den folgenden Rechtsfall und erläutere die rechtlichen Grundlagen der Entscheidung.
   
   [LEGAL_TEXT]
   <|im_end|>
   ```

4. **Statute Interpretation**:
   ```
   <|im_start|>user
   Interpretiere die folgenden Gesetzesbestimmungen und erkläre ihre Anwendung.
   
   [LEGAL_TEXT]
   <|im_end|>
   ```

## Step 5: German Legal Vocabulary Enhancement

To improve the model's understanding of German legal terminology, we extend the tokenizer vocabulary:

1. **German Legal Terms**: Court names, document types, legal concepts
2. **Legal Paragraph References**: § symbols and article references
3. **Court Abbreviations**: BGH, BVerfG, OLG, etc.
4. **Compound Legal Terms**: Common in German legal language

## Step 6: Complete Pipeline Execution

Run the complete pipeline with:

```bash
python german_legal_data_prep.py \
  --download-dir german_legal_data/raw \
  --processed-dir german_legal_data/processed \
  --output-dir german_legal_data/training \
  --limit 5000 \
  --s3-bucket legal-reasoning-model-data \
  --s3-prefix german-legal-data \
  --upload-to-s3
```

This will:
1. Download the TOC and analyze it
2. Download up to 5000 documents
3. Process and format the documents
4. Create train/validation/test splits
5. Upload the data to S3

## Step 7: SageMaker Training Configuration

Configure SageMaker for training with German legal data:

```python
from sagemaker.huggingface import HuggingFace

# Get S3 paths from the pipeline output
with open('german_legal_data/processed/s3_paths.json', 'r') as f:
    s3_paths = json.load(f)

# Configure estimator for German legal model
estimator = HuggingFace(
    entry_point='train_german_legal.py',
    source_dir='code',
    role=role,
    transformers_version='4.28.1',
    pytorch_version='2.0.0',
    py_version='py310',
    instance_count=2,
    instance_type='ml.g5.12xlarge',
    hyperparameters={
        'epochs': 3,
        'batch-size': 4,
        'learning-rate': 5e-6,
        'max-seq-length': 2048,
        'model-name': 'Qwen/Qwen2.5-7B-Instruct',
        'use-lora': 'True',
        'lora-rank': 16,
        'lora-alpha': 32,
        'lora-dropout': 0.1,
        'bf16': 'True'
    }
)

# Start training
estimator.fit(s3_paths)
```

## Conclusion

This guide provides a comprehensive approach to preparing German legal data from www.rechtsprechung-im-internet.de for training a specialized legal reasoning model. By following these steps, you'll create a high-quality dataset that leverages the unique characteristics of German legal texts and enables effective fine-tuning of the Qwen2.5-7B-Instruct model for German legal reasoning tasks.
