import os
import time
import statistics
from dotenv import load_dotenv
from app.pdf_processor import AdvancedPDFProcessor
from app.advanced_retriever import AdvancedRetriever
from app.query_optimizer import QueryOptimizer
from app.response_generator import ResponseGenerator

class AdvancedRAGTester:
    def __init__(self):
        load_dotenv()
        
        # Initialize components
        self.processor = AdvancedPDFProcessor()
        self.retriever = AdvancedRetriever(
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            index_name=os.getenv('PINECONE_INDEX_NAME')
        )
        self.optimizer = QueryOptimizer()
        self.generator = ResponseGenerator()
        
        # Test queries
        self.test_queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain neural networks",
            "What are the applications of AI?",
            "Compare supervised and unsupervised learning"
        ]
    
    def test_complete_pipeline(self, pdf_path: str):
        """Test the complete Advanced RAG pipeline"""
        print("🚀 STARTING ADVANCED RAG PIPELINE TEST")
        print("=" * 60)
        
        # Step 1: Process PDF
        print("📄 Step 1: Processing PDF...")
        chunks = self.processor.process_pdf(pdf_path)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Step 2: Setup hybrid search
        print("🔍 Step 2: Setting up hybrid search...")
        self.retriever.setup_hybrid_search(chunks)
        print("✅ Hybrid search ready")
        
        # Step 3: Test queries
        print("🎯 Step 3: Testing queries...")
        print("=" * 60)
        
        results = {}
        
        for query in self.test_queries:
            print(f"\n🔍 Testing: '{query}'")
            
            # Time the complete pipeline
            start_time = time.time()
            
            # Query optimization
            query_analysis = self.optimizer.analyze_query_type(query)
            optimized_query = self.optimizer.optimize_query(query)
            
            # Retrieval
            retrieved_docs = self.retriever.hybrid_search(
                optimized_query, 
                top_k=query_analysis['expected_documents'],
                alpha=query_analysis['alpha']
            )
            
            # Re-ranking
            reranked_docs = self.optimizer.rerank_results(retrieved_docs, query)
            
            # Response generation
            response = self.generator.generate_response(query, reranked_docs)
            
            total_latency = (time.time() - start_time) * 1000
            
            # Store results
            results[query] = {
                'total_latency': total_latency,
                'retrieval_latency': retrieved_docs[0]['latency'] if retrieved_docs else 0,
                'response_latency': response['latency'],
                'confidence': response['confidence'],
                'documents_retrieved': len(retrieved_docs),
                'answer_preview': response['answer'][:100] + '...' if response['answer'] else 'No answer'
            }
            
            print(f"✅ Answer: {response['answer'][:100]}...")
            print(f"⏱️ Total Latency: {total_latency:.2f}ms")
            print(f"🎯 Confidence: {response['confidence']:.2f}")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 ADVANCED RAG PERFORMANCE SUMMARY")
        print("=" * 60)
        
        total_latencies = [r['total_latency'] for r in results.values()]
        retrieval_latencies = [r['retrieval_latency'] for r in results.values()]
        confidences = [r['confidence'] for r in results.values()]
        
        print(f"📈 Average Total Latency: {statistics.mean(total_latencies):.2f}ms")
        print(f"📈 Average Retrieval Latency: {statistics.mean(retrieval_latencies):.2f}ms")
        print(f"🎯 Average Confidence: {statistics.mean(confidences):.2f}")
        print(f"⚡ Best Latency: {min(total_latencies):.2f}ms")
        print(f"🐢 Worst Latency: {max(total_latencies):.2f}ms")
        print(f"📄 Avg Documents Retrieved: {statistics.mean([r['documents_retrieved'] for r in results.values()]):.1f}")
        
        print("\n" + "=" * 60)
        print("✅ ADVANCED RAG PIPELINE TEST COMPLETE")
        print("=" * 60)

# Run the test
if __name__ == "__main__":
    tester = AdvancedRAGTester()
    
    # Upload your PDF first
    from google.colab import files
    uploaded = files.upload()
    
    pdf_path = list(uploaded.keys())[0]
    results = tester.test_complete_pipeline(pdf_path)