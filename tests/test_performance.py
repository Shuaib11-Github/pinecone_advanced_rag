import os
import re
from typing import List, Dict, Any
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import hashlib
import json

class AdvancedPDFProcessor:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunk_size = 512
        self.chunk_overlap = 50
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with proper error handling"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Advanced semantic-aware chunking"""
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_data = self._create_chunk_data(current_chunk.strip())
                chunks.append(chunk_data)
                # Keep overlap for context
                overlap_sentences = current_chunk.split('.')[-3:]
                current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
                current_length = len(current_chunk.split())
            else:
                current_chunk += sentence + ". "
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunk_data = self._create_chunk_data(current_chunk.strip())
            chunks.append(chunk_data)
        
        return chunks
    
    def _create_chunk_data(self, text: str) -> Dict[str, Any]:
        """Create chunk data with embeddings and metadata"""
        # Generate unique ID
        chunk_id = hashlib.md5(text.encode()).hexdigest()[:16]
        
        # Generate embedding
        embedding = self.embedding_model.encode([text])[0].tolist()
        
        return {
            "id": chunk_id,
            "text": text,
            "embedding": embedding,
            "length": len(text.split()),
            "metadata": {
                "chunk_type": "semantic",
                "word_count": len(text.split())
            }
        }
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Complete PDF processing pipeline"""
        print("📄 Extracting text from PDF...")
        text = self.extract_text_from_pdf(pdf_path)
        print(f"✅ Extracted {len(text)} characters")
        
        print("🔪 Performing semantic chunking...")
        chunks = self.semantic_chunking(text)
        print(f"✅ Created {len(chunks)} chunks")
        
        return chunks

# Test the processor
if __name__ == "__main__":
    processor = AdvancedPDFProcessor()
    chunks = processor.process_pdf("data/sample.pdf")
    print(f"First chunk: {chunks[0]['text'][:100]}...")