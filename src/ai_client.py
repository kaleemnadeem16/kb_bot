"""
AI Client Module
Handles OpenAI API interactions for embeddings and completions
"""

import os
import time
from typing import List, Optional, Dict
import openai
import numpy as np


class AIClient:
    """Manages OpenAI API interactions with cost optimization and error handling"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.max_retries = 3
        self.retry_delay = 1
        
        # Initialize OpenAI client
        if not self.api_key or self.api_key.startswith("your_"):
            print("❌ OpenAI API key not found or invalid")
            print("💡 Add your API key to .env file for full functionality")
            self.demo_mode = True
        else:
            openai.api_key = self.api_key
            self.demo_mode = False
            print(f"✅ OpenAI client initialized")
            print(f"🤖 Chat model: {self.model_name}")
            print(f"📊 Embedding model: {self.embedding_model}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching and error handling"""
        if self.demo_mode:
            return self._generate_demo_embeddings(texts)
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"🔄 Generating embeddings in {total_batches} batch(es)...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"  📊 Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            
            batch_embeddings = self._generate_batch_embeddings(batch_texts)
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
            else:
                print(f"❌ Failed to generate embeddings for batch {batch_num}")
                return []
        
        print(f"✅ Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def _generate_batch_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for a batch of texts with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                
                embeddings = [data.embedding for data in response.data]
                return embeddings
                
            except openai.RateLimitError as e:
                wait_time = self.retry_delay * (2 ** attempt)
                print(f"⚠️  Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                time.sleep(wait_time)
                
            except openai.APIError as e:
                print(f"❌ OpenAI API error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                print(f"❌ Unexpected error generating embeddings: {e}")
                break
        
        return None
    
    def _generate_demo_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate consistent demo embeddings for testing"""
        embeddings = []
        for text in texts:
            # Use hash for consistency
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.random(1536).tolist()
            embeddings.append(embedding)
        
        print(f"🔧 Generated {len(embeddings)} demo embeddings")
        return embeddings
    
    def generate_response(self, question: str, context_chunks: List[Dict], max_tokens: int = 500) -> str:
        """Generate a response using OpenAI chat completion"""
        if self.demo_mode:
            return self._generate_demo_response(question, context_chunks)
        
        if not context_chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare context
        context_parts = []
        for chunk in context_chunks:
            source_info = f"Source: {chunk['source']}"
            if chunk.get('file_type'):
                source_info += f" ({chunk['file_type']})"
            
            context_parts.append(f"{source_info}\\nContent: {chunk['text']}")
        
        context = "\\n\\n".join(context_parts)
        
        # Create optimized prompt
        prompt = self._create_optimized_prompt(question, context)
        
        # Generate response with retry logic
        for attempt in range(self.max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a helpful assistant that answers questions based on provided context. Be accurate, concise, and cite sources when possible."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=self.temperature
                )
                
                return response.choices[0].message.content.strip()
                
            except openai.RateLimitError as e:
                wait_time = self.retry_delay * (2 ** attempt)
                print(f"⚠️  Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
                
            except openai.APIError as e:
                print(f"❌ OpenAI API error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                break
        
        return "❌ Sorry, I encountered an error while generating the response. Please try again."
    
    def _create_optimized_prompt(self, question: str, context: str) -> str:
        """Create an optimized prompt for better responses"""
        return f"""Based on the following context from my knowledge base, please answer the question.

Context:
{context}

Question: {question}

Instructions:
- Answer based only on the information provided in the context
- Be comprehensive but concise
- If the context doesn't contain enough information, say so clearly
- When possible, mention which source(s) your answer comes from
- Use clear, professional language

Answer:"""
    
    def _generate_demo_response(self, question: str, context_chunks: List[Dict]) -> str:
        """Generate a demo response when API is not available"""
        sources = list(set(chunk['source'] for chunk in context_chunks))
        source_list = ", ".join(sources)
        
        return f"""Based on the available documents ({source_list}), I found relevant information about your question: "{question}"

However, this is running in demo mode without OpenAI integration. The actual response would be generated using the GPT-{self.model_name} model with the retrieved context.

To get real AI-powered responses:
1. Add your OpenAI API key to the .env file
2. Ensure you have sufficient API credits
3. Restart the application

The system successfully retrieved {len(context_chunks)} relevant text chunks that would be used to generate a comprehensive answer."""
    
    def get_usage_info(self) -> Dict:
        """Get information about API usage and configuration"""
        return {
            'demo_mode': self.demo_mode,
            'chat_model': self.model_name,
            'embedding_model': self.embedding_model,
            'temperature': self.temperature,
            'max_retries': self.max_retries,
            'api_key_configured': bool(self.api_key and not self.api_key.startswith("your_"))
        }