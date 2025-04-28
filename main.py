#!/usr/bin/env python3
"""
Simple Knowledge Base Bot - Initial Implementation
Supports TXT file processing with FAISS vector storage and OpenAI integration.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import openai
import faiss
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleKBBot:
    def __init__(self):
        """Initialize the Knowledge Base Bot"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            print("❌ Error: OPENAI_API_KEY not found or not configured!")
            print("💡 Please add your real OpenAI API key to the .env file")
            print("📖 For demo purposes, this will run in offline mode (embeddings disabled)")
            self.demo_mode = True
        else:
            # Initialize OpenAI client
            openai.api_key = self.api_key
            self.demo_mode = False
            print(f"✅ OpenAI API key loaded successfully")
            print(f"🤖 Using chat model: {self.model_name}")
            print(f"📊 Using embedding model: {self.embedding_model}")
        
        # Vector storage
        self.dimension = 1536  # Default OpenAI embedding dimension
        self.index = None
        self.documents = []
        self.chunks = []
        
        mode_text = "🔧 DEMO MODE" if self.demo_mode else "🤖 FULL MODE"
        print(f"{mode_text} Knowledge Base Bot initialized!")
    
    def load_txt_files(self, data_folder: str = "data") -> None:
        """Load all TXT files from the data folder"""
        data_path = Path(data_folder)
        if not data_path.exists():
            print(f"❌ Data folder '{data_folder}' not found!")
            return
        
        txt_files = list(data_path.glob("*.txt"))
        if not txt_files:
            print(f"❌ No TXT files found in '{data_folder}' folder!")
            return
        
        print(f"📁 Found {len(txt_files)} TXT file(s)")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        self.documents.append({
                            'filename': txt_file.name,
                            'content': content
                        })
                        print(f"✅ Loaded: {txt_file.name}")
                    else:
                        print(f"⚠️  Skipped empty file: {txt_file.name}")
            except Exception as e:
                print(f"❌ Error loading {txt_file.name}: {e}")
        
        print(f"📚 Successfully loaded {len(self.documents)} document(s)")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI"""
        if self.demo_mode:
            # Return dummy embedding for demo
            np.random.seed(hash(text) % 2**32)  # Consistent random based on text
            return np.random.random(self.dimension).tolist()
        
        try:
            response = openai.embeddings.create(
                model=self.embedding_model,  # Use configured embedding model
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return None
    
    def build_vector_index(self) -> None:
        """Build FAISS vector index from documents"""
        if not self.documents:
            print("❌ No documents loaded!")
            return
        
        print("🔄 Processing documents and generating embeddings...")
        
        # Process all documents into chunks
        all_chunks = []
        for doc in self.documents:
            doc_chunks = self.chunk_text(doc['content'])
            for chunk in doc_chunks:
                all_chunks.append({
                    'text': chunk,
                    'source': doc['filename']
                })
        
        self.chunks = all_chunks
        print(f"📄 Created {len(all_chunks)} text chunks")
        
        # Generate embeddings for all chunks
        embeddings = []
        for i, chunk in enumerate(all_chunks):
            print(f"🔄 Generating embedding {i+1}/{len(all_chunks)}", end='\r')
            embedding = self.generate_embedding(chunk['text'])
            if embedding:
                embeddings.append(embedding)
            else:
                print(f"\n❌ Failed to generate embedding for chunk {i+1}")
                return
        
        print(f"\n✅ Generated {len(embeddings)} embeddings")
        
        # Create FAISS index
        if embeddings:
            self.index = faiss.IndexFlatL2(self.dimension)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)
            print(f"🗃️  Vector index built with {self.index.ntotal} vectors")
        
        # Save index for future use
        try:
            faiss.write_index(self.index, "vector_store.faiss")
            print("💾 Vector index saved to vector_store.faiss")
        except Exception as e:
            print(f"⚠️  Could not save index: {e}")
    
    def search_similar(self, query: str, k: int = 3) -> List[dict]:
        """Search for similar chunks to the query"""
        if not self.index or not self.chunks:
            print("❌ Vector index not built! Please load documents first.")
            return []
        
        # Generate embedding for query
        query_embedding = self.generate_embedding(query)
        if not query_embedding:
            print("❌ Could not generate embedding for query")
            return []
        
        # Search in FAISS index
        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vector, k)
        
        # Return matching chunks
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                results.append({
                    'text': self.chunks[idx]['text'],
                    'source': self.chunks[idx]['source'],
                    'similarity': float(distance),
                    'rank': i + 1
                })
        
        return results
    
    def generate_answer(self, question: str, context_chunks: List[dict]) -> str:
        """Generate answer using OpenAI with retrieved context"""
        if not context_chunks:
            return "I couldn't find relevant information to answer your question."
        
        if self.demo_mode:
            # Simple demo response
            sources = [chunk['source'] for chunk in context_chunks]
            return f"Based on the available documents ({', '.join(set(sources))}), I found relevant information about your question. However, this is running in demo mode without OpenAI integration. Please add your OpenAI API key for full functionality."
        
        # Prepare context
        context = "\n\n".join([f"Source: {chunk['source']}\nContent: {chunk['text']}" 
                              for chunk in context_chunks])
        
        # Create prompt
        prompt = f"""Based on the following context from my knowledge base, please answer the question.

Context:
{context}

Question: {question}

Answer: Please provide a comprehensive answer based only on the information provided in the context. If the context doesn't contain enough information to answer the question, please say so."""

        try:
            response = openai.chat.completions.create(
                model=self.model_name,  # Use configured chat model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=self.temperature  # Use configured temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ Error generating response: {e}"
    
    def ask_question(self, question: str) -> None:
        """Process a question and provide an answer"""
        print(f"\n🔍 Searching for: {question}")
        
        # Search for relevant chunks
        similar_chunks = self.search_similar(question, k=3)
        
        if not similar_chunks:
            print("❌ No relevant information found.")
            return
        
        print(f"📄 Found {len(similar_chunks)} relevant chunk(s)")
        
        # Generate answer
        print("🤖 Generating answer...")
        answer = self.generate_answer(question, similar_chunks)
        
        print(f"\n💬 Answer:\n{answer}")
        
        # Show sources
        print(f"\n📚 Sources:")
        for chunk in similar_chunks:
            print(f"  • {chunk['source']} (rank {chunk['rank']})")
    
    def run_interactive(self) -> None:
        """Run interactive CLI"""
        print("\n" + "="*60)
        print("🤖 Knowledge Base Bot - Interactive Mode")
        print("💡 Ask me anything about your documents!")
        print("📝 Type 'exit', 'quit', or 'q' to stop")
        print("="*60)
        
        while True:
            try:
                question = input("\n❓ You: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Thank you for using Knowledge Base Bot!")
                    break
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                self.ask_question(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function"""
    print("🚀 Starting Knowledge Base Bot...")
    
    # Initialize bot
    bot = SimpleKBBot()
    
    # Load documents
    bot.load_txt_files()
    
    if not bot.documents:
        print("\n❌ No documents loaded. Please add TXT files to the 'data' folder.")
        return
    
    # Build vector index
    bot.build_vector_index()
    
    if not bot.index:
        print("\n❌ Failed to build vector index.")
        return
    
    # Run interactive mode
    bot.run_interactive()


if __name__ == "__main__":
    main()