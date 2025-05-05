"""
Vector Store Module
Handles FAISS operations for efficient similarity search
"""

import os
import pickle
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss


class VectorStore:
    """Manages FAISS vector storage and retrieval operations"""
    
    def __init__(self, dimension: int = 1536, index_path: str = "vector_store"):
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.chunks_metadata = []
        
        # Initialize FAISS index
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize or load existing FAISS index"""
        faiss_file = f"{self.index_path}.faiss"
        metadata_file = f"{self.index_path}.pkl"
        
        if os.path.exists(faiss_file) and os.path.exists(metadata_file):
            try:
                self.load_index()
                print(f"📁 Loaded existing index with {self.index.ntotal} vectors")
                return
            except Exception as e:
                print(f"⚠️  Could not load existing index: {e}")
        
        # Create new index
        self.index = faiss.IndexFlatL2(self.dimension)
        print(f"🆕 Created new FAISS index (dimension: {self.dimension})")
    
    def add_vectors(self, embeddings: List[List[float]], chunks_metadata: List[Dict]) -> None:
        """Add vectors to the FAISS index with metadata"""
        if not embeddings:
            print("⚠️  No embeddings provided")
            return
        
        if len(embeddings) != len(chunks_metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Validate dimensions
        if embeddings_array.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings_array.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Add to index
        self.index.add(embeddings_array)
        self.chunks_metadata.extend(chunks_metadata)
        
        print(f"✅ Added {len(embeddings)} vectors to index")
        print(f"🗃️  Total vectors in index: {self.index.ntotal}")
        
        # Auto-save after adding vectors
        self.save_index()
    
    def search_similar(self, query_embedding: List[float], k: int = 5, threshold: float = None) -> List[Dict]:
        """Search for similar vectors and return ranked results"""
        if not self.index or self.index.ntotal == 0:
            print("❌ No vectors in index")
            return []
        
        if len(query_embedding) != self.dimension:
            raise ValueError(f"Query embedding dimension {len(query_embedding)} doesn't match index dimension {self.dimension}")
        
        # Perform search
        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        # Process results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks_metadata):
                # Apply threshold if specified
                if threshold is not None and distance > threshold:
                    continue
                
                chunk_data = self.chunks_metadata[idx].copy()
                chunk_data.update({
                    'similarity_score': float(distance),
                    'rank': i + 1
                })
                results.append(chunk_data)
        
        print(f"🔍 Found {len(results)} similar chunks")
        return results
    
    def save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{self.index_path}.faiss")
            
            # Save metadata
            with open(f"{self.index_path}.pkl", 'wb') as f:
                pickle.dump(self.chunks_metadata, f)
            
            print(f"💾 Saved index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Error saving index: {e}")
    
    def load_index(self) -> None:
        """Load FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{self.index_path}.faiss")
            
            # Load metadata
            with open(f"{self.index_path}.pkl", 'rb') as f:
                self.chunks_metadata = pickle.load(f)
            
            print(f"📁 Loaded index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            self._initialize_index()
    
    def clear_index(self) -> None:
        """Clear the index and metadata"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks_metadata = []
        print("🗑️  Cleared vector index")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        if not self.index:
            return {'status': 'No index initialized'}
        
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'total_chunks': len(self.chunks_metadata),
            'index_type': type(self.index).__name__
        }
        
        if self.chunks_metadata:
            sources = [chunk['source'] for chunk in self.chunks_metadata]
            stats.update({
                'unique_sources': len(set(sources)),
                'sources': list(set(sources))
            })
        
        return stats
    
    def rebuild_index(self, embeddings: List[List[float]], chunks_metadata: List[Dict]) -> None:
        """Completely rebuild the index with new data"""
        print("🔄 Rebuilding vector index...")
        self.clear_index()
        self.add_vectors(embeddings, chunks_metadata)
        print("✅ Index rebuilt successfully")