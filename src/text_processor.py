"""
Text Processor Module
Handles text chunking and preprocessing for optimal embedding generation
"""

import re
from typing import List, Dict


class TextProcessor:
    """Handles text chunking with intelligent boundary detection"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+[\s\'"]*')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process all documents and create chunks with metadata"""
        all_chunks = []
        
        for doc in documents:
            # Clean the text first
            cleaned_text = self._clean_text(doc['content'])
            
            # Create chunks
            chunks = self._create_chunks(cleaned_text)
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'source': doc['filename'],
                    'file_type': doc['file_type'],
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                })
        
        print(f"📄 Created {len(all_chunks)} text chunks from {len(documents)} documents")
        return all_chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better processing"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with chunking
        text = re.sub(r'[^\w\s.,!?;:()\'"/-]', ' ', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create overlapping chunks with intelligent boundary detection"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If we're at the end of the text
            if end >= len(text):
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # Try to break at a good boundary
            optimal_end = self._find_optimal_break_point(text, start, end)
            
            chunk = text[start:optimal_end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = optimal_end - self.chunk_overlap
            
            # Ensure we don't go backwards
            if start < optimal_end - self.chunk_size:
                start = optimal_end
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 10]  # Filter very short chunks
    
    def _find_optimal_break_point(self, text: str, start: int, end: int) -> int:
        """Find the best place to break the text"""
        # Look for paragraph breaks first (best option)
        search_start = max(start + self.chunk_size // 2, end - 200)
        search_text = text[search_start:end + 100]  # Look a bit ahead
        
        # Look for paragraph breaks
        paragraph_matches = list(self.paragraph_breaks.finditer(search_text))
        if paragraph_matches:
            return search_start + paragraph_matches[0].end()
        
        # Look for sentence endings
        sentence_matches = list(self.sentence_endings.finditer(text[search_start:end + 50]))
        if sentence_matches:
            # Take the last sentence ending before our limit
            for match in reversed(sentence_matches):
                potential_end = search_start + match.end()
                if potential_end <= end + 50:
                    return potential_end
        
        # Look for word boundaries
        for i in range(end, max(start + self.chunk_size // 2, end - 100), -1):
            if text[i].isspace():
                return i
        
        # Last resort: break at character limit
        return end
    
    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk['text']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_text_length': sum(chunk_lengths),
            'sources': list(set(chunk['source'] for chunk in chunks))
        }