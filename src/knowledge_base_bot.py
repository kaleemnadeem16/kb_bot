"""
Knowledge Base Bot - Main Application Class
Coordinates all components for intelligent document Q&A
"""

import os
from typing import List, Dict, Optional
from document_loader import DocumentLoader
from text_processor import TextProcessor
from vector_store import VectorStore
from ai_client import AIClient


class KnowledgeBaseBot:
    """Main application class that coordinates all components"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_processor = TextProcessor(
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
        )
        self.vector_store = VectorStore()
        self.ai_client = AIClient()
        
        # State
        self.documents = []
        self.chunks = []
        self.is_initialized = False
        
        print("🤖 Knowledge Base Bot initialized")
    
    def initialize(self, force_rebuild: bool = False) -> bool:
        """Initialize the bot by loading documents and building vector index"""
        print("🚀 Initializing Knowledge Base Bot...")
        
        # Load documents
        self.documents = self.document_loader.load_documents(self.data_folder)
        if not self.documents:
            print("❌ No documents found. Please add files to the data folder.")
            return False
        
        # Process documents into chunks
        self.chunks = self.text_processor.process_documents(self.documents)
        if not self.chunks:
            print("❌ No text chunks created from documents.")
            return False
        
        # Check if we need to rebuild the vector index
        if force_rebuild or self._should_rebuild_index():
            self._build_vector_index()
        else:
            print("📁 Using existing vector index")
        
        self.is_initialized = True
        self._show_initialization_summary()
        return True
    
    def _should_rebuild_index(self) -> bool:
        """Determine if vector index should be rebuilt"""
        stats = self.vector_store.get_stats()
        
        # Rebuild if no vectors or chunk count mismatch
        if stats.get('total_vectors', 0) == 0:
            return True
        
        if stats.get('total_chunks', 0) != len(self.chunks):
            print("🔄 Document changes detected, rebuilding index...")
            return True
        
        return False
    
    def _build_vector_index(self) -> None:
        """Build the vector index from processed chunks"""
        print("🔄 Building vector index...")
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Generate embeddings
        embeddings = self.ai_client.generate_embeddings(texts)
        if not embeddings:
            print("❌ Failed to generate embeddings")
            return
        
        # Build vector index
        self.vector_store.rebuild_index(embeddings, self.chunks)
        print("✅ Vector index built successfully")
    
    def ask_question(self, question: str, max_results: int = None) -> Dict:
        """Process a question and return detailed results"""
        if not self.is_initialized:
            return {
                'error': 'Bot not initialized. Call initialize() first.',
                'answer': None,
                'sources': []
            }
        
        if not question.strip():
            return {
                'error': 'Please provide a valid question.',
                'answer': None,
                'sources': []
            }
        
        max_results = max_results or int(os.getenv("MAX_RETRIEVAL_DOCS", "4"))
        
        print(f"🔍 Processing question: {question}")
        
        # Generate embedding for the question
        question_embeddings = self.ai_client.generate_embeddings([question])
        if not question_embeddings:
            return {
                'error': 'Failed to process question. Please try again.',
                'answer': None,
                'sources': []
            }
        
        # Search for similar chunks
        similar_chunks = self.vector_store.search_similar(
            question_embeddings[0], 
            k=max_results
        )
        
        if not similar_chunks:
            return {
                'error': None,
                'answer': 'I could not find relevant information to answer your question.',
                'sources': []
            }
        
        print(f"📄 Found {len(similar_chunks)} relevant chunks")
        
        # Generate response
        print("🤖 Generating answer...")
        answer = self.ai_client.generate_response(question, similar_chunks)
        
        # Prepare sources information
        sources = self._prepare_sources_info(similar_chunks)
        
        return {
            'error': None,
            'answer': answer,
            'sources': sources,
            'chunks_found': len(similar_chunks),
            'question': question
        }
    
    def _prepare_sources_info(self, chunks: List[Dict]) -> List[Dict]:
        """Prepare formatted source information"""
        sources = []
        for chunk in chunks:
            sources.append({
                'filename': chunk['source'],
                'file_type': chunk.get('file_type', 'unknown'),
                'chunk_id': chunk.get('chunk_id', 0),
                'similarity_score': chunk.get('similarity_score', 0),
                'rank': chunk.get('rank', 0)
            })
        return sources
    
    def _show_initialization_summary(self) -> None:
        """Show summary of initialization results"""
        doc_stats = {}
        for doc in self.documents:
            file_type = doc['file_type']
            doc_stats[file_type] = doc_stats.get(file_type, 0) + 1
        
        chunk_stats = self.text_processor.get_chunk_stats(self.chunks)
        vector_stats = self.vector_store.get_stats()
        
        print("\\n" + "="*50)
        print("📊 INITIALIZATION SUMMARY")
        print("="*50)
        print(f"📚 Documents loaded: {len(self.documents)}")
        for file_type, count in doc_stats.items():
            print(f"   {file_type}: {count} file(s)")
        
        print(f"📄 Text chunks created: {len(self.chunks)}")
        if chunk_stats:
            print(f"   Average chunk size: {chunk_stats.get('avg_chunk_length', 0):.0f} characters")
        
        print(f"🗃️  Vector index: {vector_stats.get('total_vectors', 0)} vectors")
        print(f"🤖 AI Client: {'Demo mode' if self.ai_client.demo_mode else 'Full mode'}")
        print("="*50 + "\\n")
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics about the bot"""
        return {
            'initialized': self.is_initialized,
            'documents': len(self.documents),
            'chunks': len(self.chunks),
            'vector_stats': self.vector_store.get_stats(),
            'ai_client': self.ai_client.get_usage_info(),
            'supported_formats': self.document_loader.get_supported_formats()
        }
    
    def run_interactive(self) -> None:
        """Run interactive CLI mode"""
        if not self.is_initialized:
            print("❌ Bot not initialized. Please run initialize() first.")
            return
        
        print("\\n" + "="*60)
        print("🤖 Knowledge Base Bot - Interactive Mode")
        print("💡 Ask me anything about your documents!")
        print("📝 Commands: 'exit', 'quit', 'q' to stop | 'stats' for information")
        print("="*60)
        
        while True:
            try:
                question = input("\\n❓ You: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\\n👋 Thank you for using Knowledge Base Bot!")
                    break
                
                if question.lower() == 'stats':
                    self._show_stats()
                    continue
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                # Process question
                result = self.ask_question(question)
                
                if result['error']:
                    print(f"❌ {result['error']}")
                else:
                    print(f"\\n💬 Answer:\\n{result['answer']}")
                    
                    if result['sources']:
                        print(f"\\n📚 Sources:")
                        for source in result['sources']:
                            print(f"  • {source['filename']} (rank {source['rank']})")
                
            except KeyboardInterrupt:
                print("\\n\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_stats(self) -> None:
        """Show detailed statistics"""
        stats = self.get_stats()
        
        print("\\n📊 BOT STATISTICS")
        print("-" * 30)
        print(f"Status: {'✅ Ready' if stats['initialized'] else '❌ Not initialized'}")
        print(f"Documents: {stats['documents']}")
        print(f"Text chunks: {stats['chunks']}")
        print(f"Vector index: {stats['vector_stats'].get('total_vectors', 0)} vectors")
        print(f"AI mode: {'🔧 Demo' if stats['ai_client']['demo_mode'] else '🤖 Full'}")
        print(f"Supported formats: {', '.join(stats['supported_formats'])}")
        
        if stats['vector_stats'].get('sources'):
            print(f"Sources: {', '.join(stats['vector_stats']['sources'])}")
        print("-" * 30)