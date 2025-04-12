"""
Data preprocessing module for legal documents.
"""

import re
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class LegalDocumentPreprocessor:
    """Preprocessor for legal documents."""
    
    def __init__(self, language: str = "en"):
        """
        Initialize the preprocessor.
        
        Args:
            language: Language code (en, de)
        """
        self.language = language
        logger.info(f"Initialized LegalDocumentPreprocessor for {language} language")
    
    def preprocess(self, text: str, anonymize: bool = True) -> Dict[str, Any]:
        """
        Preprocess a legal document.
        
        Args:
            text: Raw text of the legal document
            anonymize: Whether to anonymize PII
            
        Returns:
            Dict containing processed text and metadata
        """
        # Step 1: Normalize text
        text = self._normalize_text(text)
        
        # Step 2: Parse document structure
        sections = self._parse_document_structure(text)
        
        # Step 3: Extract citations
        citations = self._extract_citations(text)
        
        # Step 4: Anonymize if required
        if anonymize:
            for section in sections:
                sections[section] = self._anonymize_text(sections[section])
        
        # Combine processed sections
        processed_text = '\n\n'.join(sections.values())
        
        return {
            'processed_text': processed_text,
            'sections': sections,
            'citations': citations
        }
    
    def preprocess_file(self, file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Preprocess a legal document file.
        
        Args:
            file_path: Path to the file
            output_path: Path to save the processed document (optional)
            
        Returns:
            Dict containing processed text and metadata
        """
        logger.info(f"Processing file: {file_path}")
        
        # Determine file type and read content
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_path.endswith('.html') or file_path.endswith('.htm'):
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
        elif file_path.endswith('.xml'):
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'xml')
                text = soup.get_text(separator=' ', strip=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Preprocess the text
        result = self.preprocess(text)
        
        # Add file metadata
        result['metadata'] = {
            'source_file': file_path,
            'file_type': os.path.splitext(file_path)[1][1:],
            'language': self.language
        }
        
        # Save processed document if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved processed document to {output_path}")
        
        return result
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text."""
        # Convert to UTF-8
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Standardize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Handle special characters based on language
        if self.language == "en":
            # Remove non-ASCII characters for English
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        elif self.language == "de":
            # Normalize German characters
            text = text.replace('„', '"').replace('"', '"')
            text = text.replace('§', '§')  # Fix paragraph symbol
        
        return text.strip()
    
    def _parse_document_structure(self, text: str) -> Dict[str, str]:
        """Parse document structure based on language."""
        sections = {}
        
        if self.language == "en":
            # Extract common sections in English legal documents
            section_patterns = {
                'header': r'^(.*?)(?=OPINION|DECISION)',
                'body': r'(?:OPINION|DECISION)(.*?)(?=CONCLUSION|$)',
                'conclusion': r'(?:CONCLUSION|ORDER)(.*?)$'
            }
            
            for section_name, pattern in section_patterns.items():
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    sections[section_name] = match.group(1).strip()
        
        elif self.language == "de":
            # Extract common sections in German legal documents
            section_patterns = {
                'tatbestand': r'(?:Tatbestand|TATBESTAND).*?(?=(?:Entscheidungsgründe|ENTSCHEIDUNGSGRÜNDE|II\.|$))',
                'gruende': r'(?:Entscheidungsgründe|ENTSCHEIDUNGSGRÜNDE|Gründe|GRÜNDE).*?(?=(?:III\.|$))',
                'tenor': r'(?:Tenor|TENOR|Beschluss|BESCHLUSS).*?(?=(?:Tatbestand|TATBESTAND|I\.|$))',
                'leitsaetze': r'(?:Leitsätze|LEITSÄTZE|Leitsatz|LEITSATZ).*?(?=(?:Tenor|TENOR|$))'
            }
            
            for section_name, pattern in section_patterns.items():
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    sections[section_name] = match.group(0).strip()
        
        # If no sections were found, use the whole text as body
        if not sections:
            sections['body'] = text
        
        return sections
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract legal citations based on language."""
        citations = []
        
        if self.language == "en":
            # Common US citation patterns
            patterns = [
                r'\d+ U\.S\. \d+',  # US Reports
                r'\d+ S\.Ct\. \d+',  # Supreme Court Reporter
                r'\d+ F\.\d[a-z]* \d+',  # Federal Reporter
                r'\d+ F\.Supp\.\d[a-z]* \d+'  # Federal Supplement
            ]
        elif self.language == "de":
            # Common German citation patterns
            patterns = [
                r'BGH.*?(?:Urteil|Beschluss).*?vom.*?\d{1,2}\.\d{1,2}\.\d{4}',  # BGH decisions
                r'BVerfG.*?(?:Urteil|Beschluss).*?vom.*?\d{1,2}\.\d{1,2}\.\d{4}',  # BVerfG decisions
                r'§\s*\d+[a-z]?(?:\s*(?:Abs\.|Absatz)\s*\d+)?(?:\s*(?:S\.|Satz)\s*\d+)?',  # Law references
                r'Art\.\s*\d+[a-z]?(?:\s*(?:Abs\.|Absatz)\s*\d+)?(?:\s*(?:S\.|Satz)\s*\d+)?'  # Article references
            ]
        
        # Extract citations using patterns
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return citations
    
    def _anonymize_text(self, text: str) -> str:
        """Anonymize personally identifiable information."""
        if self.language == "en":
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
        
        elif self.language == "de":
            # Replace German personal titles and names
            text = re.sub(r'Herr\s+[A-Z][a-zäöüß]+', 'Herr [NAME]', text)
            text = re.sub(r'Frau\s+[A-Z][a-zäöüß]+', 'Frau [NAME]', text)
            
            # Replace German addresses
            text = re.sub(r'\d+\s+[A-Za-zäöüß]+(?:straße|weg|allee|platz)', '[ADRESSE]', text)
            
            # Replace German phone numbers
            text = re.sub(r'0\d{2,4}[/-]?\d{6,8}', '[TELEFON]', text)
            
            # Replace emails
            text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
        
        return text


def batch_process_files(input_dir: str, output_dir: str, language: str = "en") -> List[Dict[str, Any]]:
    """
    Process all legal documents in a directory.
    
    Args:
        input_dir: Directory containing legal documents
        output_dir: Directory to save processed documents
        language: Language code (en, de)
        
    Returns:
        List of processed documents
    """
    logger.info(f"Batch processing files in {input_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = LegalDocumentPreprocessor(language=language)
    
    # Get all files
    files = []
    for ext in ['.txt', '.html', '.htm', '.xml']:
        files.extend([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(ext)])
    
    logger.info(f"Found {len(files)} files to process")
    
    # Process each file
    results = []
    for file_path in files:
        try:
            output_path = os.path.join(output_dir, os.path.basename(file_path) + '.json')
            result = preprocessor.preprocess_file(file_path, output_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Successfully processed {len(results)} files")
    return results
