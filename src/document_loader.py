"""
Document Loader Module
Handles loading and processing of various document formats
"""

import os
import csv
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  PDF support not available. Install 'pypdf' for PDF processing.")


class DocumentLoader:
    """Handles loading of TXT, PDF, and CSV documents"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.csv']
        if PDF_AVAILABLE:
            self.supported_formats.append('.pdf')
    
    def load_documents(self, data_folder: str = "data") -> List[Dict]:
        """Load all supported documents from the data folder"""
        data_path = Path(data_folder)
        if not data_path.exists():
            print(f"❌ Data folder '{data_folder}' not found!")
            return []
        
        documents = []
        file_count = 0
        
        # Load each supported file type
        for file_path in data_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    content = self._load_single_document(file_path)
                    if content:
                        documents.append({
                            'filename': file_path.name,
                            'content': content,
                            'file_type': file_path.suffix.lower(),
                            'size': len(content)
                        })
                        file_count += 1
                        print(f"✅ Loaded: {file_path.name}")
                    else:
                        print(f"⚠️  Skipped empty file: {file_path.name}")
                except Exception as e:
                    print(f"❌ Error loading {file_path.name}: {e}")
        
        print(f"📚 Successfully loaded {file_count} document(s)")
        self._show_disclaimer_if_needed(documents)
        
        return documents
    
    def _load_single_document(self, file_path: Path) -> Optional[str]:
        """Load a single document based on its file type"""
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.txt':
            return self._load_txt_file(file_path)
        elif file_extension == '.pdf':
            return self._load_pdf_file(file_path)
        elif file_extension == '.csv':
            return self._load_csv_file(file_path)
        else:
            print(f"⚠️  Unsupported file format: {file_extension}")
            return None
    
    def _load_txt_file(self, file_path: Path) -> str:
        """Load text file with encoding detection"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode {file_path.name} with any supported encoding")
    
    def _load_pdf_file(self, file_path: Path) -> str:
        """Load PDF file and extract text (ignoring images)"""
        if not PDF_AVAILABLE:
            raise ImportError("PDF support not available. Install 'pypdf' package.")
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        # Extract text only, ignore images and complex formatting
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(page_text.strip())
                    except Exception as e:
                        print(f"⚠️  Could not extract text from page {page_num + 1}: {e}")
                
                if not text_content:
                    print(f"⚠️  No text content found in {file_path.name}")
                    return ""
                
                return "\\n\\n".join(text_content)
        
        except Exception as e:
            raise ValueError(f"Error reading PDF {file_path.name}: {e}")
    
    def _load_csv_file(self, file_path: Path) -> str:
        """Load CSV file and convert to searchable text"""
        try:
            # Try to read with pandas first
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # Convert DataFrame to searchable text format
            text_parts = []
            
            # Add column headers information
            text_parts.append(f"CSV Data from {file_path.name}")
            text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
            text_parts.append("\\n" + "="*50 + "\\n")
            
            # Convert each row to text
            for index, row in df.iterrows():
                row_text = []
                for col, value in row.items():
                    if pd.notna(value):  # Skip NaN values
                        row_text.append(f"{col}: {value}")
                
                if row_text:  # Only add rows with data
                    text_parts.append(f"Row {index + 1}: " + " | ".join(row_text))
            
            return "\\n".join(text_parts)
        
        except Exception as e:
            # Fallback to basic CSV reading
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    rows = list(csv_reader)
                    
                    if not rows:
                        return ""
                    
                    text_parts = [f"CSV Data from {file_path.name}"]
                    
                    # Add header
                    if rows:
                        text_parts.append(f"Headers: {', '.join(rows[0])}")
                        text_parts.append("\\n" + "="*50 + "\\n")
                    
                    # Add data rows
                    for i, row in enumerate(rows[1:], 1):
                        if row:  # Skip empty rows
                            text_parts.append(f"Row {i}: {' | '.join(row)}")
                    
                    return "\\n".join(text_parts)
            
            except Exception as fallback_error:
                raise ValueError(f"Error reading CSV {file_path.name}: {fallback_error}")
    
    def _show_disclaimer_if_needed(self, documents: List[Dict]) -> None:
        """Show disclaimer for downloaded/external content"""
        external_indicators = [
            'garcia_pablo', 'download', 'example', 'sample', 'test', 
            'demo', 'internet', 'external'
        ]
        
        external_files = []
        for doc in documents:
            filename_lower = doc['filename'].lower()
            if any(indicator in filename_lower for indicator in external_indicators):
                external_files.append(doc['filename'])
        
        if external_files:
            print("\\n" + "="*60)
            print("⚠️  DISCLAIMER: Testing Content")
            print("="*60)
            print("The following files are used for testing/demonstration purposes only:")
            for filename in external_files:
                print(f"  • {filename}")
            print("\\nThese files:")
            print("  - May be downloaded from external sources")
            print("  - Are not owned by the project authors")
            print("  - Are used solely for testing functionality")
            print("  - Should be replaced with your own content for production use")
            print("="*60 + "\\n")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return self.supported_formats.copy()