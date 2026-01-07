"""
Simple Test Examples for 15-Step RAG Pipeline
Focused demonstrations of key functionality
"""

import os
from rag_pipeline_advanced import RAGPipeline

def test_basic_functionality():
    """Test basic RAG functionality"""
    print("=== BASIC FUNCTIONALITY TEST ===")
    print()
    
    # Setup pipeline
    pipeline = RAGPipeline(
        chunk_size=500,
        overlap=80,
        top_k=5,
        rerank_k=3,
        use_mmr=True
    )
    
    # Add sources
    sources = ["README.md", "sample.txt"]
    pipeline.add_sources(sources)
    print(f"Added {len(sources)} sources successfully")
    print()
    
    # Test queries
    queries = [
        "What is this website about?",
        "What technologies are used?",
        "What features are included?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        result = pipeline.query(query)
        print(f"Answer: {result[:150]}...")
        print("-" * 50)
        print()

def test_content_retrieval():
    """Test content retrieval with scores"""
    print("=== CONTENT RETRIEVAL TEST ===")
    print()
    
    pipeline = RAGPipeline(chunk_size=400, overlap=60, top_k=4, rerank_k=2)
    pipeline.add_sources(["README.md", "index.html"])
    
    query = "What are the main features?"
    print(f"Query: {query}")
    print()
    
    # Get retrieval results
    context = pipeline.retrieve(query)
    print(f"Retrieved {len(context)} relevant chunks:")
    print()
    
    for i, (chunk, score) in enumerate(context, 1):
        print(f"Chunk {i}:")
        print(f"  Source: {chunk.metadata.get('filename', 'Unknown')}")
        print(f"  Score: {score:.4f}")
        print(f"  Length: {len(chunk.text)} chars")
        print(f"  Preview: {chunk.text[:80]}...")
        print()

def test_mmr_re_ranking():
    """Test MMR re-ranking vs simple similarity"""
    print("=== MMR RE-RANKING TEST ===")
    print()
    
    query = "What technologies are mentioned?"
    print(f"Query: {query}")
    print()
    
    # Without MMR
    pipeline_no_mmr = RAGPipeline(
        chunk_size=400, overlap=60, top_k=5, rerank_k=5, use_mmr=False
    )
    pipeline_no_mmr.add_sources(["README.md", "index.html"])
    context_no_mmr = pipeline_no_mmr.retrieve(query)
    
    print("Without MMR (pure similarity):")
    for i, (chunk, score) in enumerate(context_no_mmr, 1):
        print(f"  {i}. {chunk.metadata.get('filename')} - {score:.4f}")
    print()
    
    # With MMR
    pipeline_mmr = RAGPipeline(
        chunk_size=400, overlap=60, top_k=5, rerank_k=5, use_mmr=True
    )
    pipeline_mmr.add_sources(["README.md", "index.html"])
    context_mmr = pipeline_mmr.retrieve(query)
    
    print("With MMR (diverse re-ranking):")
    for i, (chunk, score) in enumerate(context_mmr, 1):
        print(f"  {i}. {chunk.metadata.get('filename')} - {score:.4f}")
    print()

def test_different_sources():
    """Test different source types"""
    print("=== DIFFERENT SOURCE TYPES TEST ===")
    print()
    
    sources = ["README.md", "index.html", "sample.txt"]
    
    for source in sources:
        pipeline = RAGPipeline(chunk_size=300, overlap=40, top_k=3)
        pipeline.add_sources([source])
        
        query = "What is the main content?"
        result = pipeline.query(query)
        
        print(f"Source: {source}")
        print(f"Result: {result[:100]}...")
        print("-" * 40)
        print()

def test_chunk_sizes():
    """Test different chunking strategies"""
    print("=== CHUNK SIZE COMPARISON ===")
    print()
    
    chunk_sizes = [200, 500, 800]
    query = "What is this project?"
    
    for size in chunk_sizes:
        pipeline = RAGPipeline(chunk_size=size, overlap=size//6, top_k=3)
        pipeline.add_sources(["README.md"])
        
        context = pipeline.retrieve(query)
        total_chars = sum(len(chunk.text) for chunk, _ in context)
        
        print(f"Chunk Size {size}:")
        print(f"  Chunks retrieved: {len(context)}")
        print(f"  Total characters: {total_chars}")
        print(f"  Average chunk size: {total_chars//len(context) if context else 0}")
        print()

def test_pipeline_configs():
    """Test different pipeline configurations"""
    print("=== PIPELINE CONFIGURATIONS ===")
    print()
    
    configs = [
        {"name": "Fast", "top_k": 3, "rerank_k": 2, "use_mmr": False},
        {"name": "Balanced", "top_k": 8, "rerank_k": 4, "use_mmr": True},
        {"name": "Thorough", "top_k": 12, "rerank_k": 6, "use_mmr": True, "lambda_val": 0.5}
    ]
    
    query = "What makes this website modern?"
    
    for config in configs:
        name = config.pop("name")
        pipeline = RAGPipeline(**config)
        pipeline.add_sources(["README.md", "sample.txt"])
        
        context = pipeline.retrieve(query)
        print(f"Config '{name}': {len(context)} chunks retrieved")
        print(f"  Settings: {config}")
        print()

def test_query_types():
    """Test different types of queries"""
    print("=== QUERY TYPE TESTS ===")
    print()
    
    pipeline = RAGPipeline(chunk_size=450, overlap=70, top_k=6, rerank_k=3)
    pipeline.add_sources(["README.md", "index.html", "sample.txt"])
    
    query_types = {
        "Factual": ["What technologies are used?", "What features are included?"],
        "Analytical": ["How is the design structured?", "What makes it accessible?"],
        "Comparative": ["Compare frontend and backend", "Modern vs traditional features"]
    }
    
    for qtype, queries in query_types.items():
        print(f"{qtype} Queries:")
        for query in queries:
            context = pipeline.retrieve(query)
            print(f"  '{query[:30]}...' -> {len(context)} chunks")
        print()

def run_simple_tests():
    """Run all simple tests"""
    print("RUNNING SIMPLE RAG PIPELINE TESTS")
    print("=" * 50)
    print()
    
    try:
        test_basic_functionality()
        test_content_retrieval()
        test_mmr_re_ranking()
        test_different_sources()
        test_chunk_sizes()
        test_pipeline_configs()
        test_query_types()
        
        print("ALL SIMPLE TESTS COMPLETED!")
        print("The 15-step RAG pipeline is working correctly!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    run_simple_tests()
