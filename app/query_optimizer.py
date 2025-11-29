import re
import time
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class QueryOptimizer:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def analyze_query_type(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine optimal retrieval strategy"""
        query = query.lower().strip()
        
        analysis = {
            'type': 'factual',  # factual, conceptual, procedural, complex
            'complexity': 'simple',  # simple, medium, complex
            'expected_documents': 3,
            'search_strategy': 'hybrid',
            'alpha': 0.7  # hybrid search weight
        }
        
        # Query type detection
        if any(word in query for word in ['what is', 'define', 'meaning of']):
            analysis['type'] = 'conceptual'
            analysis['expected_documents'] = 2
        elif any(word in query for word in ['how to', 'steps to', 'process of']):
            analysis['type'] = 'procedural'
            analysis['expected_documents'] = 4
        elif any(word in query for word in ['compare', 'difference between', 'advantages']):
            analysis['type'] = 'complex'
            analysis['complexity'] = 'complex'
            analysis['expected_documents'] = 5
        
        # Complexity detection
        word_count = len(query.split())
        if word_count > 8:
            analysis['complexity'] = 'complex'
            analysis['expected_documents'] = 6
        elif word_count > 4:
            analysis['complexity'] = 'medium'
            analysis['expected_documents'] = 4
        
        # Strategy adjustment based on analysis
        if analysis['type'] == 'conceptual':
            analysis['alpha'] = 0.8  # More weight to vector search
        elif analysis['type'] == 'procedural':
            analysis['alpha'] = 0.5  # Balanced approach
        
        return analysis
    
    def optimize_query(self, query: str) -> str:
        """Optimize query for better retrieval"""
        # Remove unnecessary words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = query.split()
        optimized_words = [word for word in words if word.lower() not in stop_words]
        
        # Focus on key terms
        optimized_query = ' '.join(optimized_words)
        
        return optimized_query if len(optimized_words) > 1 else query
    
    def rerank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Re-rank results based on relevance to query"""
        if not results:
            return results
        
        # Simple re-ranking based on query term overlap
        query_terms = set(query.lower().split())
        
        for result in results:
            result_text = result['text'].lower()
            term_overlap = len(query_terms.intersection(set(result_text.split())))
            overlap_score = term_overlap / len(query_terms) if query_terms else 0
            
            # Combine with existing score
            existing_score = result.get('hybrid_score', result.get('score', 0))
            result['rerank_score'] = 0.7 * existing_score + 0.3 * overlap_score
        
        # Sort by rerank score
        results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        return results