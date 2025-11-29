import time
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class ResponseGenerator:
    def __init__(self):
        # Use cross-encoder for re-ranking if available
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except:
            self.reranker = None
    
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                         use_reranking: bool = True) -> Dict[str, Any]:
        """Generate final response with confidence scoring"""
        start_time = time.time()
        
        if not retrieved_docs:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'confidence': 0.0,
                'sources': [],
                'latency': (time.time() - start_time) * 1000
            }
        
        # Optional: Re-rank with cross-encoder for better quality
        if use_reranking and self.reranker and len(retrieved_docs) > 1:
            retrieved_docs = self.rerank_with_cross_encoder(query, retrieved_docs)
        
        # Generate answer (simplified - in production, use LLM like GPT)
        answer = self._synthesize_answer(query, retrieved_docs)
        
        # Calculate confidence
        confidence = self._calculate_confidence(retrieved_docs, query)
        
        # Prepare sources
        sources = [{'text': doc['text'][:200] + '...', 'score': doc.get('hybrid_score', doc.get('score', 0))} 
                  for doc in retrieved_docs[:3]]
        
        return {
            'answer': answer,
            'confidence': confidence,
            'sources': sources,
            'total_documents': len(retrieved_docs),
            'latency': (time.time() - start_time) * 1000
        }
    
    def rerank_with_cross_encoder(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank using cross-encoder for better relevance"""
        try:
            pairs = [(query, doc['text']) for doc in docs]
            scores = self.reranker.predict(pairs)
            
            for doc, score in zip(docs, scores):
                doc['cross_encoder_score'] = score
            
            docs.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            return docs
        except:
            return docs  # Fallback to original ranking
    
    def _synthesize_answer(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """Synthesize answer from retrieved documents (simplified version)"""
        # In production, replace this with proper LLM call
        top_doc = docs[0]['text']
        
        # Simple answer synthesis
        if 'what is' in query.lower():
            return f"Based on the document: {top_doc[:300]}..."
        elif 'how to' in query.lower():
            return f"The process involves: {top_doc[:400]}..."
        else:
            return f"Here's what I found: {top_doc[:350]}..."
    
    def _calculate_confidence(self, docs: List[Dict[str, Any]], query: str) -> float:
        """Calculate confidence score for the response"""
        if not docs:
            return 0.0
        
        # Base confidence on top document score
        top_score = docs[0].get('hybrid_score', docs[0].get('score', 0))
        
        # Adjust based on number of supporting documents
        supporting_docs = len([d for d in docs if d.get('hybrid_score', 0) > 0.5])
        support_factor = min(supporting_docs / 3, 1.0)
        
        # Query-term overlap
        query_terms = set(query.lower().split())
        top_doc_terms = set(docs[0]['text'].lower().split())
        overlap = len(query_terms.intersection(top_doc_terms)) / len(query_terms) if query_terms else 0
        
        confidence = (0.6 * top_score + 0.3 * support_factor + 0.1 * overlap)
        return min(confidence, 1.0)