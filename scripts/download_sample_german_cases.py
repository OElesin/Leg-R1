#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download and format sample German court decisions.
This script downloads the first 5 court decisions from the XML file,
extracts the content, and formats it for the Qwen2.5-7B-Instruct model.
"""

import os
import re
import xml.etree.ElementTree as ET
import requests
import zipfile
import io
import json
import logging
from bs4 import BeautifulSoup
from tqdm import tqdm
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sample_download.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Constants
XML_FILE_PATH = "downloaded_file.xml"
OUTPUT_DIR = "data/german/samples"
PROCESSED_DIR = "data/german/processed"
MAX_SAMPLES = 5


def parse_xml_file(xml_path):
    """Parse the XML file containing court decision links."""
    logger.info(f"Parsing XML file: {xml_path}")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract items (court decisions)
        items = []
        for i, item in enumerate(root.findall('item')):
            if i >= MAX_SAMPLES:
                break
                
            court = item.find('gericht').text if item.find('gericht') is not None else "Unknown"
            date = item.find('entsch-datum').text if item.find('entsch-datum') is not None else "Unknown"
            case_number = item.find('aktenzeichen').text if item.find('aktenzeichen') is not None else "Unknown"
            link = item.find('link').text if item.find('link') is not None else None
            
            if link:
                items.append({
                    'court': court,
                    'date': date,
                    'case_number': case_number,
                    'link': link
                })
        
        logger.info(f"Found {len(items)} items to download")
        return items
    
    except Exception as e:
        logger.error(f"Error parsing XML file: {e}")
        return []


def download_and_extract_zip(url, output_dir):
    """Download and extract a ZIP file from a URL."""
    logger.info(f"Downloading: {url}")
    
    try:
        # Download the ZIP file
        response = requests.get(url)
        response.raise_for_status()
        
        # Create a ZIP file in memory
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        
        # Extract all files to the output directory
        zip_file.extractall(output_dir)
        
        # Get the list of extracted files
        extracted_files = zip_file.namelist()
        
        # Find XML files
        xml_files = [f for f in extracted_files if f.endswith('.xml')]
        
        if xml_files:
            return os.path.join(output_dir, xml_files[0])
        else:
            logger.warning(f"No XML files found in ZIP from {url}")
            return None
    
    except Exception as e:
        logger.error(f"Error downloading or extracting ZIP file: {e}")
        return None


def parse_court_decision_xml(xml_path):
    """Parse the XML file containing a court decision."""
    logger.info(f"Parsing court decision XML: {xml_path}")
    
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse XML/HTML
        try:
            soup = BeautifulSoup(content, 'lxml-xml')  # Use lxml parser for XML
        except:
            try:
                soup = BeautifulSoup(content, 'xml')
            except:
                soup = BeautifulSoup(content, 'html.parser')
        
        # Extract metadata
        metadata = {}
        
        # Common metadata fields in German legal documents
        metadata_fields = {
            'Gericht': 'court',
            'Datum': 'date',
            'Aktenzeichen': 'case_number',
            'Dokumenttyp': 'document_type',
            'Verfahrensart': 'procedure_type',
            'ECLI': 'ecli'
        }
        
        # Extract metadata from XML
        for german_field, english_field in metadata_fields.items():
            field_elem = soup.find(german_field)
            if field_elem:
                metadata[english_field] = field_elem.text.strip()
        
        # Extract main text content
        text_content = ""
        
        # Look for common content containers
        content_tags = ['dokument', 'text', 'entscheidung', 'doktextdaten', 'body']
        for tag in content_tags:
            content_elem = soup.find(tag)
            if content_elem:
                text_content += content_elem.get_text(separator=' ', strip=True) + " "
        
        # If no specific content found, get all text
        if not text_content.strip():
            text_content = soup.get_text(separator=' ', strip=True)
        
        # Clean the text
        text_content = clean_text(text_content)
        
        # Extract sections
        sections = extract_sections(text_content)
        
        return {
            'metadata': metadata,
            'full_text': text_content,
            'sections': sections,
            'source_file': xml_path
        }
    
    except Exception as e:
        logger.error(f"Error parsing court decision XML: {e}")
        return None


def clean_text(text):
    """Clean German legal text."""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove line numbers often found in legal documents
    text = re.sub(r'^\s*\d+\s*', '', text, flags=re.MULTILINE)
    
    # Normalize German quotation marks
    text = text.replace('„', '"').replace('"', '"')
    
    # Fix common OCR errors in German legal texts
    text = text.replace('§', '§')  # Fix paragraph symbol
    
    return text.strip()


def extract_sections(text):
    """Extract common sections from German legal documents."""
    sections = {}
    
    # Common section patterns in German court decisions
    section_patterns = {
        'tatbestand': r'(?:Tatbestand|TATBESTAND).*?(?=(?:Entscheidungsgründe|ENTSCHEIDUNGSGRÜNDE|II\.|$))',
        'gruende': r'(?:Entscheidungsgründe|ENTSCHEIDUNGSGRÜNDE|Gründe|GRÜNDE).*?(?=(?:III\.|$))',
        'tenor': r'(?:Tenor|TENOR|Beschluss|BESCHLUSS).*?(?=(?:Tatbestand|TATBESTAND|I\.|$))',
        'leitsaetze': r'(?:Leitsätze|LEITSÄTZE|Leitsatz|LEITSATZ).*?(?=(?:Tenor|TENOR|$))'
    }
    
    # Extract each section using regex
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[section_name] = match.group(0).strip()
    
    return sections


def format_for_qwen(processed_doc, task_type="case_analysis"):
    """
    Format German legal document for Qwen2.5-7B-Instruct.
    
    Args:
        processed_doc: Processed document dictionary
        task_type: Type of task ('classification', 'summarization', 'case_analysis', etc.)
        
    Returns:
        dict: Formatted example for training
    """
    # System message in German
    system_message = "Du bist ein juristischer Assistent, der auf die Analyse deutscher Rechtsdokumente spezialisiert ist."
    
    # Define task-specific templates in German
    templates = {
        'classification': {
            'instruction': "Klassifiziere das folgende Rechtsdokument in eine der folgenden Kategorien: Zivilrecht, Strafrecht, Verwaltungsrecht, Verfassungsrecht oder Arbeitsrecht.",
            'response_template': "Basierend auf meiner Analyse fällt dieses Dokument in die Kategorie {category}, weil {reasoning}."
        },
        'summarization': {
            'instruction': "Fasse das folgende Rechtsdokument zusammen und hebe die wichtigsten Punkte und Entscheidungen hervor.",
            'response_template': "Zusammenfassung:\n{summary}\n\nWichtige Entscheidungen:\n1. {point1}\n2. {point2}\n3. {point3}"
        },
        'case_analysis': {
            'instruction': "Analysiere den folgenden Rechtsfall und erläutere die rechtlichen Grundlagen der Entscheidung.",
            'response_template': "Analyse:\n{analysis}\n\nRechtliche Grundlagen:\n{legal_basis}\n\nSchlussfolgerung:\n{conclusion}"
        },
        'statute_interpretation': {
            'instruction': "Interpretiere die folgenden Gesetzesbestimmungen und erkläre ihre Anwendung.",
            'response_template': "Interpretation:\n{interpretation}\n\nAnwendung:\n{application}\n\nWichtige Elemente:\n1. {element1}\n2. {element2}\n3. {element3}"
        }
    }
    
    # Get the appropriate template
    template = templates.get(task_type, templates['case_analysis'])
    
    # Prepare document text
    doc_text = processed_doc['full_text']
    
    # Limit text length to avoid token limits
    max_chars = 12000  # Approximate limit to stay within token limits
    if len(doc_text) > max_chars:
        doc_text = doc_text[:max_chars] + "..."
    
    # Create metadata
    metadata = {
        'document_type': processed_doc['metadata'].get('document_type', 'unknown'),
        'court': processed_doc['metadata'].get('court', 'unknown'),
        'date': processed_doc['metadata'].get('date', 'unknown'),
        'case_number': processed_doc['metadata'].get('case_number', 'unknown'),
        'language': 'de',
        'task': task_type,
        'source': 'rechtsprechung-im-internet'
    }
    
    # Create the example in Qwen2.5-7B-Instruct format
    example = {
        "id": str(uuid.uuid4()),
        "conversations": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": f"{template['instruction']}\n\n{doc_text}"
            },
            {
                "role": "assistant",
                "content": template['response_template']  # This will be filled with actual data or used as-is for training
            }
        ],
        "metadata": metadata
    }
    
    return example


def main():
    """Main function."""
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Parse XML file
    items = parse_xml_file(XML_FILE_PATH)
    
    # Download and process each item
    processed_docs = []
    formatted_examples = []
    
    for item in tqdm(items, desc="Processing court decisions"):
        # Create a subdirectory for each case
        case_dir = os.path.join(OUTPUT_DIR, item['case_number'].replace('/', '_'))
        os.makedirs(case_dir, exist_ok=True)
        
        # Download and extract the ZIP file
        xml_path = download_and_extract_zip(item['link'], case_dir)
        
        if xml_path:
            # Parse the court decision XML
            processed_doc = parse_court_decision_xml(xml_path)
            
            if processed_doc:
                # Add original metadata
                processed_doc['metadata'].update({
                    'court': item['court'],
                    'date': item['date'],
                    'case_number': item['case_number'],
                    'link': item['link']
                })
                
                processed_docs.append(processed_doc)
                
                # Save processed document
                processed_path = os.path.join(PROCESSED_DIR, f"{item['case_number'].replace('/', '_')}.json")
                with open(processed_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_doc, f, ensure_ascii=False, indent=2)
                
                # Format for different tasks
                tasks = ['classification', 'summarization', 'case_analysis', 'statute_interpretation']
                for task in tasks:
                    formatted_example = format_for_qwen(processed_doc, task)
                    formatted_examples.append(formatted_example)
                    
                    # Save formatted example
                    formatted_path = os.path.join(PROCESSED_DIR, f"{item['case_number'].replace('/', '_')}_{task}.jsonl")
                    with open(formatted_path, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(formatted_example, ensure_ascii=False))
    
    # Save all formatted examples to a single file
    all_examples_path = os.path.join(PROCESSED_DIR, 'all_examples.jsonl')
    with open(all_examples_path, 'w', encoding='utf-8') as f:
        for example in formatted_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"Processed {len(processed_docs)} court decisions")
    logger.info(f"Created {len(formatted_examples)} formatted examples")
    logger.info(f"All examples saved to {all_examples_path}")


if __name__ == "__main__":
    main()
