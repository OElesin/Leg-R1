# Data Preparation Guide for Legal Reasoning Model

This document provides detailed instructions for preparing legal datasets to train the Legal Reasoning Model based on Qwen2.5-7B-Instruct.

## Table of Contents
1. [Data Requirements](#data-requirements)
2. [Data Collection](#data-collection)
3. [Data Preprocessing](#data-preprocessing)
4. [Data Formatting](#data-formatting)
5. [Data Splitting](#data-splitting)
6. [Data Augmentation](#data-augmentation)
7. [Uploading to Amazon S3](#uploading-to-amazon-s3)

## Data Requirements

### Dataset Size
- **Minimum recommended size**: 50,000 examples
- **Ideal size**: 200,000+ examples for robust performance
- **Distribution**: Balanced representation across legal domains and jurisdictions

### Data Types
The model requires diverse legal data sources:

1. **Case Law Documents**
   - Full text of judicial opinions
   - Case summaries
   - Headnotes and key points
   - Jurisdiction and court level metadata

2. **Statutes and Regulations**
   - Legislative text
   - Regulatory provisions
   - Historical versions for temporal analysis

3. **Legal Briefs and Memoranda**
   - Arguments presented
   - Citations to authority
   - Procedural context

4. **Legal Commentary**
   - Law review articles
   - Treatises
   - Expert analysis

5. **Legal Questions and Answers**
   - Bar exam questions
   - Legal advice scenarios
   - Hypothetical case analyses

### Data Quality Requirements
- **Accuracy**: Verified legal information from authoritative sources
- **Recency**: Include recent legal developments (ideally up to 2023)
- **Diversity**: Cover multiple jurisdictions and practice areas
- **Completeness**: Full text rather than excerpts when possible
- **Cleanliness**: Free from OCR errors and formatting issues

## Data Collection

### Authorized Sources
Only collect data from authorized sources with appropriate licensing:

1. **Public Domain Sources**
   - Government websites (courts, legislatures)
   - Public records
   - Historical legal documents

2. **Licensed Sources** (with proper permissions)
   - Legal databases (Westlaw, LexisNexis, etc.)
   - Academic repositories
   - Specialized legal collections

3. **Synthetic Data**
   - Generated examples for underrepresented scenarios
   - Hypothetical cases for ethical dilemmas

### Collection Methods
1. **API Integration**
   ```python
   import requests
   
   def fetch_case_law(api_key, jurisdiction, date_range):
       url = "https://legal-api.example.com/cases"
       headers = {"Authorization": f"Bearer {api_key}"}
       params = {
           "jurisdiction": jurisdiction,
           "start_date": date_range[0],
           "end_date": date_range[1],
           "format": "json"
       }
       
       response = requests.get(url, headers=headers, params=params)
       return response.json()
   ```

2. **Web Scraping** (where legally permitted)
   ```python
   import requests
   from bs4 import BeautifulSoup
   
   def scrape_public_court_opinions(court_url):
       response = requests.get(court_url)
       soup = BeautifulSoup(response.text, 'html.parser')
       
       opinions = []
       for opinion_div in soup.find_all('div', class_='opinion'):
           title = opinion_div.find('h3').text
           text = opinion_div.find('div', class_='opinion-text').text
           date = opinion_div.find('span', class_='date').text
           
           opinions.append({
               'title': title,
               'text': text,
               'date': date
           })
           
       return opinions
   ```

3. **Bulk Downloads**
   ```bash
   # Example: Downloading bulk data from a government repository
   wget -r -np -A .pdf https://www.courts.gov/opinions/2023/
   ```

### Metadata Collection
Ensure you collect essential metadata with each document:
- Document type (case, statute, brief, etc.)
- Jurisdiction
- Date
- Court/authority
- Citation information
- Legal domain/practice area
- Procedural posture
- Parties involved (anonymized if necessary)
## Data Preprocessing

### Cleaning Steps
1. **Text Normalization**
   - Convert to UTF-8 encoding
   - Standardize line endings
   - Normalize whitespace
   - Handle special characters

   ```python
   def normalize_text(text):
       # Convert to UTF-8
       text = text.encode('utf-8', errors='ignore').decode('utf-8')
       
       # Normalize whitespace
       text = re.sub(r'\s+', ' ', text)
       
       # Standardize line endings
       text = text.replace('\r\n', '\n').replace('\r', '\n')
       
       # Handle special characters
       text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
       
       return text.strip()
   ```

2. **Document Structure Parsing**
   - Identify document sections (headings, paragraphs, footnotes)
   - Extract citations
   - Recognize formatting elements

   ```python
   def parse_legal_document(text):
       sections = {}
       
       # Extract header information
       header_match = re.search(r'^(.*?)(?=OPINION|DECISION)', text, re.DOTALL | re.IGNORECASE)
       if header_match:
           sections['header'] = header_match.group(1).strip()
       
       # Extract opinion/body
       body_match = re.search(r'(?:OPINION|DECISION)(.*?)(?=CONCLUSION|$)', text, re.DOTALL | re.IGNORECASE)
       if body_match:
           sections['body'] = body_match.group(1).strip()
       
       # Extract conclusion
       conclusion_match = re.search(r'(?:CONCLUSION|ORDER)(.*?)$', text, re.DOTALL | re.IGNORECASE)
       if conclusion_match:
           sections['conclusion'] = conclusion_match.group(1).strip()
       
       return sections
   ```

3. **Citation Extraction**
   - Identify legal citations
   - Standardize citation format
   - Create citation graph

   ```python
   def extract_citations(text):
       # Common citation patterns
       patterns = [
           r'\d+ U\.S\. \d+',  # US Reports
           r'\d+ S\.Ct\. \d+',  # Supreme Court Reporter
           r'\d+ F\.\d[a-z]* \d+',  # Federal Reporter
           r'\d+ F\.Supp\.\d[a-z]* \d+'  # Federal Supplement
       ]
       
       citations = []
       for pattern in patterns:
           matches = re.findall(pattern, text)
           citations.extend(matches)
           
       return citations
   ```

4. **Anonymization**
   - Remove personally identifiable information (PII)
   - Replace names with placeholders
   - Mask sensitive information

   ```python
   def anonymize_text(text):
       # Replace personal names
       text = re.sub(r'Mr\.\s+[A-Z][a-z]+', 'Mr. [NAME]', text)
       text = re.sub(r'Ms\.\s+[A-Z][a-z]+', 'Ms. [NAME]', text)
       text = re.sub(r'Mrs\.\s+[A-Z][a-z]+', 'Mrs. [NAME]', text)
       
       # Replace addresses
       text = re.sub(r'\d+ [A-Za-z]+ (?:Street|Avenue|Road|Blvd)', '[ADDRESS]', text)
       
       # Replace phone numbers
       text = re.sub(r'\(\d{3}\) \d{3}-\d{4}', '[PHONE]', text)
       text = re.sub(r'\d{3}-\d{3}-\d{4}', '[PHONE]', text)
       
       # Replace emails
       text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
       
       return text
   ```

### Preprocessing Pipeline
Combine the above steps into a comprehensive preprocessing pipeline:

```python
def preprocess_legal_document(raw_text, anonymize=True):
    # Step 1: Normalize text
    text = normalize_text(raw_text)
    
    # Step 2: Parse document structure
    sections = parse_legal_document(text)
    
    # Step 3: Extract citations
    citations = extract_citations(text)
    
    # Step 4: Anonymize if required
    if anonymize:
        for section in sections:
            sections[section] = anonymize_text(sections[section])
    
    # Combine processed sections
    processed_text = '\n\n'.join(sections.values())
    
    return {
        'processed_text': processed_text,
        'sections': sections,
        'citations': citations
    }
```
## Data Formatting

### Input Format for Qwen2.5-7B-Instruct
The Qwen2.5-7B-Instruct model requires data to be formatted as instruction-response pairs. For legal reasoning tasks, we use the following format:

```
<|im_start|>system
You are a legal reasoning assistant trained to analyze legal documents and provide insights.
<|im_end|>
<|im_start|>user
[INSTRUCTION]
[LEGAL_TEXT]
<|im_end|>
<|im_start|>assistant
[RESPONSE]
<|im_end|>
```

Where:
- `[INSTRUCTION]` is the specific task (e.g., "Summarize this case", "Identify the legal issues", etc.)
- `[LEGAL_TEXT]` is the preprocessed legal document
- `[RESPONSE]` is the target output (e.g., summary, analysis, prediction)

### Task-Specific Formats

1. **Document Classification**
   ```
   <|im_start|>system
   You are a legal reasoning assistant trained to analyze legal documents and provide insights.
   <|im_end|>
   <|im_start|>user
   Classify the following legal document into one of these categories: Contract Law, Criminal Law, Constitutional Law, Administrative Law, or Tort Law.
   
   [LEGAL_TEXT]
   <|im_end|>
   <|im_start|>assistant
   Based on my analysis, this document falls under [CATEGORY] because [REASONING].
   <|im_end|>
   ```

2. **Case Outcome Prediction**
   ```
   <|im_start|>system
   You are a legal reasoning assistant trained to analyze legal documents and provide insights.
   <|im_end|>
   <|im_start|>user
   Predict the outcome of this case based on the facts and legal precedents.
   
   [CASE_FACTS]
   <|im_end|>
   <|im_start|>assistant
   Based on the facts presented and relevant precedents, I predict the outcome will be [OUTCOME]. This is because [REASONING].
   <|im_end|>
   ```

3. **Legal Summarization**
   ```
   <|im_start|>system
   You are a legal reasoning assistant trained to analyze legal documents and provide insights.
   <|im_end|>
   <|im_start|>user
   Provide a concise summary of the following legal document, highlighting key points and holdings.
   
   [LEGAL_TEXT]
   <|im_end|>
   <|im_start|>assistant
   Summary:
   [SUMMARY_TEXT]
   
   Key Holdings:
   1. [HOLDING_1]
   2. [HOLDING_2]
   3. [HOLDING_3]
   <|im_end|>
   ```

4. **Statute Interpretation**
   ```
   <|im_start|>system
   You are a legal reasoning assistant trained to analyze legal documents and provide insights.
   <|im_end|>
   <|im_start|>user
   Interpret the following statute and explain its application.
   
   [STATUTE_TEXT]
   <|im_end|>
   <|im_start|>assistant
   Interpretation:
   [INTERPRETATION]
   
   Application:
   [APPLICATION]
   
   Key Elements:
   1. [ELEMENT_1]
   2. [ELEMENT_2]
   3. [ELEMENT_3]
   <|im_end|>
   ```

### JSON Format for Training
For efficient training, convert the instruction-response pairs to JSON format:

```json
{
  "id": "example_001",
  "conversations": [
    {
      "role": "system",
      "content": "You are a legal reasoning assistant trained to analyze legal documents and provide insights."
    },
    {
      "role": "user",
      "content": "Classify the following legal document into one of these categories: Contract Law, Criminal Law, Constitutional Law, Administrative Law, or Tort Law.\n\n[LEGAL_TEXT]"
    },
    {
      "role": "assistant",
      "content": "Based on my analysis, this document falls under [CATEGORY] because [REASONING]."
    }
  ],
  "metadata": {
    "document_type": "case_law",
    "jurisdiction": "federal",
    "year": 2020,
    "source": "supreme_court",
    "task": "classification"
  }
}
```

### Conversion Script
Use the following script to convert preprocessed legal documents to the required format:

```python
import json
import uuid

def create_training_example(legal_text, task_type, metadata=None):
    """
    Create a training example in the required format for Qwen2.5-7B-Instruct.
    
    Args:
        legal_text (str): The preprocessed legal text
        task_type (str): One of 'classification', 'prediction', 'summarization', 'interpretation'
        metadata (dict): Additional metadata about the document
        
    Returns:
        dict: A formatted training example
    """
    system_message = "You are a legal reasoning assistant trained to analyze legal documents and provide insights."
    
    # Define task-specific templates
    templates = {
        'classification': {
            'instruction': "Classify the following legal document into one of these categories: Contract Law, Criminal Law, Constitutional Law, Administrative Law, or Tort Law.",
            'response': "Based on my analysis, this document falls under {category} because {reasoning}."
        },
        'prediction': {
            'instruction': "Predict the outcome of this case based on the facts and legal precedents.",
            'response': "Based on the facts presented and relevant precedents, I predict the outcome will be {outcome}. This is because {reasoning}."
        },
        'summarization': {
            'instruction': "Provide a concise summary of the following legal document, highlighting key points and holdings.",
            'response': "Summary:\n{summary}\n\nKey Holdings:\n1. {holding1}\n2. {holding2}\n3. {holding3}"
        },
        'interpretation': {
            'instruction': "Interpret the following statute and explain its application.",
            'response': "Interpretation:\n{interpretation}\n\nApplication:\n{application}\n\nKey Elements:\n1. {element1}\n2. {element2}\n3. {element3}"
        }
    }
    
    # Get the appropriate template
    template = templates.get(task_type, templates['classification'])
    
    # Create the example
    example = {
        "id": str(uuid.uuid4()),
        "conversations": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": f"{template['instruction']}\n\n{legal_text}"
            },
            {
                "role": "assistant",
                "content": template['response']  # This will be filled with actual data
            }
        ],
        "metadata": metadata or {}
    }
    
    # Add task type to metadata
    example["metadata"]["task"] = task_type
    
    return example
```
## Data Splitting

### Train/Validation/Test Split
For robust model evaluation, split your dataset into training, validation, and test sets:

```python
import random
from sklearn.model_selection import train_test_split

def split_dataset(examples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, stratify_by=None):
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        examples (list): List of formatted examples
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set
        test_ratio (float): Proportion for test set
        stratify_by (str): Metadata field to use for stratified splitting
        
    Returns:
        tuple: (train_examples, val_examples, test_examples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Extract stratification labels if specified
    if stratify_by and stratify_by in examples[0].get('metadata', {}):
        stratify = [ex['metadata'][stratify_by] for ex in examples]
    else:
        stratify = None
    
    # First split: separate training set
    train_examples, temp_examples = train_test_split(
        examples,
        train_size=train_ratio,
        random_state=42,
        stratify=stratify
    )
    
    # If we used stratification, update for the second split
    if stratify:
        stratify = [ex['metadata'][stratify_by] for ex in temp_examples]
    
    # Second split: separate validation and test sets
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_examples, test_examples = train_test_split(
        temp_examples,
        train_size=val_ratio_adjusted,
        random_state=42,
        stratify=stratify
    )
    
    print(f"Split dataset into {len(train_examples)} training examples, "
          f"{len(val_examples)} validation examples, and {len(test_examples)} test examples.")
    
    return train_examples, val_examples, test_examples
```

### Stratified Splitting
For legal datasets, it's important to maintain balanced representation across different legal domains, jurisdictions, or time periods. Use stratified splitting to ensure this balance:

```python
# Example: Split by legal domain
train_examples, val_examples, test_examples = split_dataset(
    all_examples, 
    stratify_by='document_type'
)

# Check distribution
def check_distribution(examples, field):
    distribution = {}
    for ex in examples:
        value = ex['metadata'].get(field, 'unknown')
        distribution[value] = distribution.get(value, 0) + 1
    
    for value, count in distribution.items():
        print(f"{value}: {count} ({count/len(examples)*100:.1f}%)")
    
    return distribution

print("Training set distribution by document type:")
check_distribution(train_examples, 'document_type')

print("\nValidation set distribution by document type:")
check_distribution(val_examples, 'document_type')

print("\nTest set distribution by document type:")
check_distribution(test_examples, 'document_type')
```

### Temporal Splitting
For legal applications, consider temporal splitting to evaluate how well the model generalizes to future cases:

```python
def temporal_split(examples, year_threshold=2020):
    """
    Split dataset based on the year of the document.
    Training: before year_threshold
    Validation: year_threshold
    Test: after year_threshold
    
    Args:
        examples (list): List of formatted examples
        year_threshold (int): Year to use as threshold
        
    Returns:
        tuple: (train_examples, val_examples, test_examples)
    """
    train_examples = []
    val_examples = []
    test_examples = []
    
    for ex in examples:
        year = ex['metadata'].get('year', 0)
        
        if year < year_threshold:
            train_examples.append(ex)
        elif year == year_threshold:
            val_examples.append(ex)
        else:
            test_examples.append(ex)
    
    print(f"Temporal split: {len(train_examples)} training examples (before {year_threshold}), "
          f"{len(val_examples)} validation examples ({year_threshold}), and "
          f"{len(test_examples)} test examples (after {year_threshold}).")
    
    return train_examples, val_examples, test_examples
```

## Data Augmentation

Legal datasets often have imbalanced representation across different legal domains or case types. Data augmentation can help address this issue.

### Synonym Replacement
Replace legal terms with their synonyms to create variations:

```python
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

def get_legal_synonyms(word):
    """Get synonyms for legal terms."""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

def augment_by_synonym_replacement(text, n_words=5, p_replace=0.3):
    """
    Replace random words with their synonyms.
    
    Args:
        text (str): Original text
        n_words (int): Number of words to replace
        p_replace (float): Probability of replacement
        
    Returns:
        str: Augmented text
    """
    words = text.split()
    new_words = words.copy()
    
    # Select random words to replace
    random_word_indices = random.sample(range(len(words)), min(n_words, len(words)))
    
    for idx in random_word_indices:
        word = words[idx]
        if len(word) < 4 or random.random() > p_replace:
            continue
            
        synonyms = get_legal_synonyms(word)
        if synonyms:
            new_words[idx] = random.choice(synonyms)
    
    return ' '.join(new_words)
```

### Back Translation
Use translation services to create paraphrased versions:

```python
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, source_lang="en", target_lang="fr"):
    """
    Augment text using back translation.
    
    Args:
        text (str): Original text
        source_lang (str): Source language code
        target_lang (str): Target language code
        
    Returns:
        str: Augmented text
    """
    # Load translation models
    model_name_to_target = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    model_name_from_target = f"Helsinki-NLP/opus-mt-{target_lang}-{source_lang}"
    
    tokenizer_to_target = MarianTokenizer.from_pretrained(model_name_to_target)
    model_to_target = MarianMTModel.from_pretrained(model_name_to_target)
    
    tokenizer_from_target = MarianTokenizer.from_pretrained(model_name_from_target)
    model_from_target = MarianMTModel.from_pretrained(model_name_from_target)
    
    # Translate to target language
    inputs = tokenizer_to_target(text, return_tensors="pt", padding=True)
    translated = model_to_target.generate(**inputs)
    target_text = tokenizer_to_target.decode(translated[0], skip_special_tokens=True)
    
    # Translate back to source language
    inputs = tokenizer_from_target(target_text, return_tensors="pt", padding=True)
    back_translated = model_from_target.generate(**inputs)
    back_translated_text = tokenizer_from_target.decode(back_translated[0], skip_special_tokens=True)
    
    return back_translated_text
```

### Legal-Specific Augmentation
Create variations by changing legal parameters:

```python
def augment_legal_case(case_text, case_metadata):
    """
    Create variations of legal cases by changing parameters.
    
    Args:
        case_text (str): Original case text
        case_metadata (dict): Case metadata
        
    Returns:
        list: List of (augmented_text, augmented_metadata) pairs
    """
    augmented_cases = []
    
    # Original case
    augmented_cases.append((case_text, case_metadata))
    
    # Change jurisdiction
    if 'jurisdiction' in case_metadata:
        jurisdictions = ['federal', 'state', 'international']
        for jurisdiction in jurisdictions:
            if jurisdiction != case_metadata['jurisdiction']:
                new_metadata = case_metadata.copy()
                new_metadata['jurisdiction'] = jurisdiction
                
                # Replace jurisdiction mentions in text
                new_text = case_text.replace(
                    f"jurisdiction of {case_metadata['jurisdiction']}", 
                    f"jurisdiction of {jurisdiction}"
                )
                
                augmented_cases.append((new_text, new_metadata))
    
    # Change time period
    if 'year' in case_metadata:
        years = [case_metadata['year'] - 10, case_metadata['year'] + 10]
        for year in years:
            if 1900 < year < 2023:  # Reasonable year range
                new_metadata = case_metadata.copy()
                new_metadata['year'] = year
                
                # Replace year mentions in text
                new_text = case_text.replace(
                    f"in {case_metadata['year']}", 
                    f"in {year}"
                )
                
                augmented_cases.append((new_text, new_metadata))
    
    return augmented_cases
```
## Uploading to Amazon S3

### Preparing Data for SageMaker
To use your prepared dataset with Amazon SageMaker, you need to upload it to Amazon S3 in an appropriate format.

### File Organization
Organize your files in the following structure:

```
s3://legal-reasoning-model-data/
├── raw/                      # Raw data files
├── processed/                # Processed data files
└── training-data/            # Training-ready data
    ├── train/                # Training set
    │   ├── data.json         # Training examples
    │   └── metadata.json     # Training metadata
    ├── validation/           # Validation set
    │   ├── data.json         # Validation examples
    │   └── metadata.json     # Validation metadata
    └── test/                 # Test set
        ├── data.json         # Test examples
        └── metadata.json     # Test metadata
```

### Converting to SageMaker Format
SageMaker expects data in specific formats. For text data, we'll use JSON Lines format:

```python
import json

def convert_to_jsonl(examples, output_file):
    """
    Convert examples to JSON Lines format.
    
    Args:
        examples (list): List of examples
        output_file (str): Path to output file
    """
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Converted {len(examples)} examples to {output_file}")
```

### Upload Script
Use the following script to upload your data to S3:

```python
import boto3
import os
import json

def upload_to_s3(local_directory, bucket_name, s3_prefix):
    """
    Upload data to S3 bucket.
    
    Args:
        local_directory (str): Local directory containing data files
        bucket_name (str): S3 bucket name
        s3_prefix (str): S3 key prefix
    """
    s3_client = boto3.client('s3')
    
    # Walk through local directory
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            # Construct the full local path
            local_path = os.path.join(root, filename)
            
            # Construct the S3 key
            relative_path = os.path.relpath(local_path, local_directory)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            
            # Upload file
            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
            s3_client.upload_file(local_path, bucket_name, s3_key)
    
    print(f"Upload complete to s3://{bucket_name}/{s3_prefix}")
```

### Complete Upload Process
Here's a complete script to prepare and upload your data:

```python
import os
import json
import boto3
from sklearn.model_selection import train_test_split

def prepare_and_upload_data(processed_examples, bucket_name, s3_prefix):
    """
    Prepare data splits and upload to S3.
    
    Args:
        processed_examples (list): List of processed examples
        bucket_name (str): S3 bucket name
        s3_prefix (str): S3 key prefix
    """
    # Create local directories
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/validation', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    # Split data
    train_examples, val_examples, test_examples = split_dataset(
        processed_examples, 
        train_ratio=0.8, 
        val_ratio=0.1, 
        test_ratio=0.1,
        stratify_by='document_type'
    )
    
    # Save to local files
    convert_to_jsonl(train_examples, 'data/train/data.jsonl')
    convert_to_jsonl(val_examples, 'data/validation/data.jsonl')
    convert_to_jsonl(test_examples, 'data/test/data.jsonl')
    
    # Create metadata files
    for split, examples in [('train', train_examples), ('validation', val_examples), ('test', test_examples)]:
        metadata = {
            'count': len(examples),
            'distribution': {},
            'fields': list(examples[0].keys()) if examples else []
        }
        
        # Calculate distribution of document types
        for ex in examples:
            doc_type = ex['metadata'].get('document_type', 'unknown')
            metadata['distribution'][doc_type] = metadata['distribution'].get(doc_type, 0) + 1
        
        # Save metadata
        with open(f'data/{split}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Upload to S3
    upload_to_s3('data', bucket_name, s3_prefix)
    
    # Return S3 paths
    return {
        'train': f's3://{bucket_name}/{s3_prefix}/train',
        'validation': f's3://{bucket_name}/{s3_prefix}/validation',
        'test': f's3://{bucket_name}/{s3_prefix}/test'
    }
```

### S3 Data Access for SageMaker
When setting up your SageMaker training job, use the S3 paths as data channels:

```python
data_channels = {
    'train': f's3://{bucket_name}/{s3_prefix}/train',
    'validation': f's3://{bucket_name}/{s3_prefix}/validation',
    'test': f's3://{bucket_name}/{s3_prefix}/test'
}

estimator.fit(data_channels)
```

### Data Versioning
Implement data versioning to track changes to your dataset:

```python
def version_dataset(bucket_name, s3_prefix, version_name):
    """
    Create a versioned copy of the dataset.
    
    Args:
        bucket_name (str): S3 bucket name
        s3_prefix (str): S3 key prefix of current data
        version_name (str): Version name or number
    
    Returns:
        str: S3 prefix of the versioned dataset
    """
    s3_client = boto3.client('s3')
    
    # New prefix for versioned data
    versioned_prefix = f"{s3_prefix}-{version_name}"
    
    # List all objects in the current prefix
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
    
    # Copy each object to the versioned prefix
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                source_key = obj['Key']
                
                # Create new key with versioned prefix
                relative_key = source_key[len(s3_prefix):]
                target_key = f"{versioned_prefix}{relative_key}"
                
                # Copy object
                print(f"Copying s3://{bucket_name}/{source_key} to s3://{bucket_name}/{target_key}")
                s3_client.copy_object(
                    CopySource={'Bucket': bucket_name, 'Key': source_key},
                    Bucket=bucket_name,
                    Key=target_key
                )
    
    print(f"Dataset versioned as s3://{bucket_name}/{versioned_prefix}")
    return versioned_prefix
```

## Conclusion

This data preparation guide provides a comprehensive framework for collecting, preprocessing, formatting, and uploading legal data for training the Legal Reasoning Model. Following these guidelines will ensure your data is properly structured for optimal training results with the Qwen2.5-7B-Instruct foundation model.

Remember to:
1. Collect diverse and high-quality legal data from authorized sources
2. Thoroughly preprocess and clean the data
3. Format the data as instruction-response pairs for Qwen2.5-7B-Instruct
4. Create balanced train/validation/test splits
5. Augment data to address imbalances
6. Upload the prepared data to Amazon S3 in the correct format

By following these steps, you'll create a robust dataset that enables effective fine-tuning of the Legal Reasoning Model for various legal tasks.
