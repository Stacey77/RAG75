"""
Production Client Example for 15-Step RAG Pipeline
Demonstrates how to use the production API
"""

import requests
import json
import time
from typing import Dict, Any

class RAGPipelineClient:
    """Client for interacting with the production RAG Pipeline API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed pipeline status"""
        response = self.session.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    def add_sources(self, sources: list, chunk_size: int = 600, overlap: int = 100) -> Dict[str, Any]:
        """Add sources to the pipeline"""
        payload = {
            "sources": sources,
            "chunk_size": chunk_size,
            "overlap": overlap
        }
        response = self.session.post(f"{self.base_url}/sources", json=payload)
        response.raise_for_status()
        return response.json()
    
    def query(self, query: str, top_k: int = 5, rerank_k: int = 3, use_mmr: bool = True) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        payload = {
            "query": query,
            "top_k": top_k,
            "rerank_k": rerank_k,
            "use_mmr": use_mmr
        }
        response = self.session.post(f"{self.base_url}/query", json=payload)
        response.raise_for_status()
        return response.json()
    
    def retrieve(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Retrieve raw content without LLM processing"""
        params = {"query": query, "k": k}
        response = self.session.get(f"{self.base_url}/retrieve", params=params)
        response.raise_for_status()
        return response.json()
    
    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update pipeline configuration"""
        response = self.session.post(f"{self.base_url}/config", json=config)
        response.raise_for_status()
        return response.json()
    
    def reset_pipeline(self) -> Dict[str, Any]:
        """Reset the pipeline"""
        response = self.session.delete(f"{self.base_url}/reset")
        response.raise_for_status()
        return response.json()

def demo_production_client():
    """Demonstrate production client usage"""
    print("🚀 Production RAG Pipeline Client Demo")
    print("=" * 50)
    
    # Initialize client
    client = RAGPipelineClient("http://localhost:8000")
    
    try:
        # Health check
        print("1. Health Check")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Pipeline Initialized: {health['pipeline_initialized']}")
        print(f"   Sources Loaded: {health['sources_loaded']}")
        print()
        
        # Get detailed status
        print("2. Pipeline Status")
        status = client.get_status()
        print(f"   Chunk Size: {status.get('chunk_size', 'N/A')}")
        print(f"   Top-K: {status.get('top_k', 'N/A')}")
        print(f"   Use MMR: {status.get('use_mmr', 'N/A')}")
        print()
        
        # Add sources (if needed)
        if status.get('sources_loaded', 0) == 0:
            print("3. Adding Sources")
            sources_result = client.add_sources(["README.md", "sample.txt"])
            print(f"   Sources Added: {sources_result['sources_added']}")
            print()
        
        # Test queries
        print("4. Test Queries")
        queries = [
            "What is this website about?",
            "What technologies are used?",
            "What features are included?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"   Query {i}: {query}")
            result = client.query(query, top_k=3, use_mmr=True)
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Sources Used: {result['sources_used']}")
            print(f"   Processing Time: {result['processing_time']:.2f}s")
            print()
        
        # Test content retrieval
        print("5. Content Retrieval")
        retrieval = client.retrieve("What technologies are used?", k=3)
        print(f"   Retrieved {retrieval['results_count']} chunks")
        for j, chunk in enumerate(retrieval['results'][:2], 1):
            print(f"   Chunk {j}: {chunk['source']} (Score: {chunk['score']:.3f})")
            print(f"   Content: {chunk['content'][:80]}...")
        print()
        
        # Test configuration update
        print("6. Configuration Update")
        new_config = {
            "chunk_size": 500,
            "top_k": 6,
            "rerank_k": 3,
            "use_mmr": True,
            "lambda_val": 0.7
        }
        config_result = client.update_config(new_config)
        print(f"   Config Updated: {config_result['message']}")
        print()
        
        print("✅ Production Client Demo Complete!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to RAG Pipeline API")
        print("   Make sure the production server is running:")
        print("   cd production && ./deploy.sh")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def benchmark_performance():
    """Benchmark production API performance"""
    print("🏃 Performance Benchmark")
    print("=" * 30)
    
    client = RAGPipelineClient("http://localhost:8000")
    
    try:
        # Test different configurations
        configs = [
            {"name": "Fast", "top_k": 3, "rerank_k": 2, "use_mmr": False},
            {"name": "Balanced", "top_k": 5, "rerank_k": 3, "use_mmr": True},
            {"name": "Thorough", "top_k": 8, "rerank_k": 4, "use_mmr": True}
        ]
        
        query = "What are the main features of this website?"
        
        for config in configs:
            print(f"Testing {config['name']} configuration...")
            
            times = []
            for i in range(5):
                start = time.time()
                result = client.query(
                    query, 
                    top_k=config['top_k'], 
                    rerank_k=config['rerank_k'], 
                    use_mmr=config['use_mmr']
                )
                times.append(result['processing_time'])
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"  Average: {avg_time:.2f}s")
            print(f"  Min: {min_time:.2f}s")
            print(f"  Max: {max_time:.2f}s")
            print()
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

if __name__ == "__main__":
    demo_production_client()
    print()
    benchmark_performance()
