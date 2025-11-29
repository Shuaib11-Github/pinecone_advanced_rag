import os
import time
from typing import List, Dict, Any
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer

class AdvancedRetriever:
    def __init__(self, pinecone_api_key: str, index_name: str, embedding_model='all-MiniLM-L6-v2'):
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.bm25_index = None
        self.chunk_texts = []
        
    def setup_hybrid_search(self, chunks: List[Dict[str, Any]]):
        """Setup BM25 for keyword search"""
        self.chunk_texts = [chunk['text'] for chunk in chunks]
        tokenized_corpus = [doc.split() for doc in self.chunk_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
    
    def vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Pure vector similarity search"""
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        
        latency = (time.time() - start_time) * 1000
        
        return [
            {
                'text': match.metadata.get('text', ''),
                'score': match.score,
                'type': 'vector',
                'latency': latency
            }
            for match in results.matches
        ]
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """BM25 keyword search"""
        start_time = time.time()
        
        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top_k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        latency = (time.time() - start_time) * 1000
        
        return [
            {
                'text': self.chunk_texts[idx],
                'score': scores[idx],
                'type': 'keyword',
                'latency': latency
            }
            for idx in top_indices if scores[idx] > 0
        ]
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Advanced hybrid search combining vector and keyword"""
        start_time = time.time()
        
        # Get results from both methods
        vector_results = self.vector_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # Normalize scores
        vector_scores = [r['score'] for r in vector_results]
        keyword_scores = [r['score'] for r in keyword_results]
        
        if vector_scores:
            max_vector = max(vector_scores)
            vector_results = [{'text': r['text'], 'score': r['score']/max_vector, 'type': 'vector'} 
                            for r in vector_results]
        
        if keyword_scores and max(keyword_scores) > 0:
            max_keyword = max(keyword_scores)
            keyword_results = [{'text': r['text'], 'score': r['score']/max_keyword, 'type': 'keyword'} 
                             for r in keyword_results]
        
        # Combine results
        combined = {}
        for result in vector_results + keyword_results:
            text = result['text']
            if text in combined:
                # Hybrid score: weighted combination
                if result['type'] == 'vector':
                    combined[text]['vector_score'] = result['score']
                else:
                    combined[text]['keyword_score'] = result['score']
            else:
                combined[text] = {
                    'text': text,
                    'vector_score': result['score'] if result['type'] == 'vector' else 0,
                    'keyword_score': result['score'] if result['type'] == 'keyword' else 0
                }
        
        # Calculate final hybrid scores
        final_results = []
        for text, scores in combined.items():
            hybrid_score = (alpha * scores['vector_score'] + 
                          (1 - alpha) * scores['keyword_score'])
            final_results.append({
                'text': text,
                'hybrid_score': hybrid_score,
                'vector_score': scores['vector_score'],
                'keyword_score': scores['keyword_score'],
                'type': 'hybrid'
            })
        
        # Sort by hybrid score and take top_k
        final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        final_results = final_results[:top_k]
        
        latency = (time.time() - start_time) * 1000
        for result in final_results:
            result['latency'] = latency
        
        return final_results
    
    def query_expansion(self, query: str) -> List[str]:
        """Simple query expansion for better retrieval"""
        # Basic expansion - in production, use LLM for this
        expansions = [query]
        
        # Add variations
        words = query.split()
        if len(words) > 1:
            # Reorder words
            expansions.append(' '.join(words[::-1]))
            
        # Add synonyms for common terms (simplified)
        synonym_map = {
            'machine learning': ['ML', 'artificial intelligence', 'AI'],
            'deep learning': ['neural networks', 'DL'],
            'how': ['what is', 'explain', 'describe']
        }
        
        for term, synonyms in synonym_map.items():
            if term in query.lower():
                for synonym in synonyms:
                    expanded = query.lower().replace(term, synonym)
                    expansions.append(expanded)
        
        return expansions[:3]  # Limit expansions
