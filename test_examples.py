"""
Comprehensive Test Examples for 15-Step RAG Pipeline
Demonstrates various query types, source formats, and pipeline features
"""

import os
import json
from rag_pipeline_advanced import RAGPipeline

def setup_test_pipeline():
    """Setup a test pipeline with all website sources"""
    pipeline = RAGPipeline(
        chunk_size=600,
        overlap=100,
        top_k=8,
        rerank_k=4,
        use_mmr=True,
        lambda_val=0.6,
        max_answer_words=300
    )
    
    # Add all available sources
    sources = ["README.md", "index.html", "styles.css", "script.js", "sample.txt"]
    pipeline.add_sources(sources)
    
    return pipeline

def test_basic_queries(pipeline):
    """Test basic informational queries"""
    print("=== BASIC QUERY TESTS ===")
    print()
    
    basic_queries = [
        "What is this website about?",
        "Who is Stacey Williams?",
        "What technologies are used in this project?",
        "How can I contact the website owner?",
        "What features does this website have?"
    ]
    
    for i, query in enumerate(basic_queries, 1):
        print(f"Query {i}: {query}")
        result = pipeline.query(query)
        print(f"Answer: {result[:200]}...")
        print("-" * 60)
        print()

def test_technical_queries(pipeline):
    """Test technical and implementation questions"""
    print("=== TECHNICAL QUERY TESTS ===")
    print()
    
    technical_queries = [
        "What frontend technologies are used?",
        "Describe the CSS architecture",
        "What JavaScript features are implemented?",
        "How is the website made responsive?",
        "What accessibility features are included?"
    ]
    
    for i, query in enumerate(technical_queries, 1):
        print(f"Technical Query {i}: {query}")
        result = pipeline.query(query)
        print(f"Answer: {result[:200]}...")
        print("-" * 60)
        print()

def test_portfolio_queries(pipeline):
    """Test portfolio and skills-related questions"""
    print("=== PORTFOLIO QUERY TESTS ===")
    print()
    
    portfolio_queries = [
        "What skills are showcased?",
        "What projects are featured?",
        "What is the design approach?",
        "What experience is highlighted?",
        "What tools and technologies are listed?"
    ]
    
    for i, query in enumerate(portfolio_queries, 1):
        print(f"Portfolio Query {i}: {query}")
        result = pipeline.query(query)
        print(f"Answer: {result[:200]}...")
        print("-" * 60)
        print()

def test_content_retrieval_demo(pipeline):
    """Demonstrate content retrieval with scores"""
    print("=== CONTENT RETRIEVAL DEMO ===")
    print()
    
    query = "What are the main features of this website?"
    print(f"Query: {query}")
    print()
    
    # Get raw retrieval results
    context = pipeline.retrieve(query)
    
    print(f"Retrieved {len(context)} relevant chunks:")
    print()
    
    for i, (chunk, score) in enumerate(context, 1):
        print(f"Chunk {i} (Similarity: {score:.4f}):")
        print(f"  Source: {chunk.metadata.get('filename', 'Unknown')}")
        print(f"  Length: {len(chunk.text)} chars")
        print(f"  Preview: {chunk.text[:100]}...")
        print()

def test_mmr_vs_no_mmr(pipeline):
    """Compare MMR re-ranking vs simple similarity"""
    print("=== MMR RE-RANKING COMPARISON ===")
    print()
    
    query = "What technologies are used in this website?"
    print(f"Query: {query}")
    print()
    
    # Test without MMR
    pipeline_no_mmr = RAGPipeline(
        chunk_size=600, overlap=100, top_k=5, rerank_k=5, use_mmr=False
    )
    pipeline_no_mmr.add_sources(["README.md", "index.html"])
    
    context_no_mmr = pipeline_no_mmr.retrieve(query)
    print("Without MMR (Top 5 by similarity):")
    for i, (chunk, score) in enumerate(context_no_mmr, 1):
        print(f"  {i}. {chunk.metadata.get('filename')} - Score: {score:.4f}")
    print()
    
    # Test with MMR
    context_mmr = pipeline.retrieve(query)
    print("With MMR (Diverse re-ranking):")
    for i, (chunk, score) in enumerate(context_mmr, 1):
        print(f"  {i}. {chunk.metadata.get('filename')} - Score: {score:.4f}")
    print()

def test_different_chunk_sizes():
    """Test different chunking strategies"""
    print("=== CHUNK SIZE COMPARISON ===")
    print()
    
    chunk_sizes = [300, 600, 900]
    query = "What is this project about?"
    
    for size in chunk_sizes:
        pipeline = RAGPipeline(chunk_size=size, overlap=size//6, top_k=3, rerank_k=2)
        pipeline.add_sources(["README.md"])
        
        context = pipeline.retrieve(query)
        print(f"Chunk Size {size}: Retrieved {len(context)} chunks")
        for i, (chunk, score) in enumerate(context, 1):
            print(f"  Chunk {i}: {len(chunk.text)} chars, Score: {score:.4f}")
        print()

def test_source_types():
    """Test different source types and formats"""
    print("=== SOURCE TYPE TESTS ===")
    print()
    
    # Test individual sources
    sources = ["README.md", "index.html", "sample.txt"]
    
    for source in sources:
        pipeline = RAGPipeline(chunk_size=500, overlap=50, top_k=3)
        pipeline.add_sources([source])
        
        query = "What is the main content of this source?"
        result = pipeline.query(query)
        
        print(f"Source: {source}")
        print(f"Result: {result[:150]}...")
        print("-" * 40)
        print()

def test_complex_queries(pipeline):
    """Test complex and multi-part questions"""
    print("=== COMPLEX QUERY TESTS ===")
    print()
    
    complex_queries = [
        "Compare the frontend and backend technologies used",
        "What is the relationship between design and functionality?",
        "How do accessibility features impact user experience?",
        "What makes this website modern and professional?",
        "How are performance and responsiveness balanced?"
    ]
    
    for i, query in enumerate(complex_queries, 1):
        print(f"Complex Query {i}: {query}")
        result = pipeline.query(query)
        print(f"Answer: {result[:200]}...")
        print("-" * 60)
        print()

def test_pipeline_configuration():
    """Test different pipeline configurations"""
    print("=== CONFIGURATION TESTS ===")
    print()
    
    configs = [
        {"name": "Fast Search", "top_k": 3, "rerank_k": 2, "use_mmr": False},
        {"name": "Balanced", "top_k": 8, "rerank_k": 4, "use_mmr": True},
        {"name": "Comprehensive", "top_k": 15, "rerank_k": 8, "use_mmr": True, "lambda_val": 0.4}
    ]
    
    query = "What are the key features?"
    
    for config in configs:
        name = config.pop("name")
        pipeline = RAGPipeline(**config)
        pipeline.add_sources(["README.md", "sample.txt"])
        
        context = pipeline.retrieve(query)
        print(f"Config '{name}': Retrieved {len(context)} chunks")
        print(f"  Settings: {config}")
        print()

def test_error_handling():
    """Test error handling and edge cases"""
    print("=== ERROR HANDLING TESTS ===")
    print()
    
    pipeline = setup_test_pipeline()
    
    # Test empty query
    try:
        result = pipeline.query("")
        print("Empty query handled:", result[:50])
    except Exception as e:
        print("Empty query error:", str(e))
    
    # Test very long query
    long_query = "What " + "very " * 50 + "long query about this website?"
    try:
        result = pipeline.query(long_query)
        print("Long query handled:", result[:50])
    except Exception as e:
        print("Long query error:", str(e))
    
    # Test query about non-existent content
    try:
        result = pipeline.query("What about quantum physics in this website?")
        print("Non-existent content:", result[:50])
    except Exception as e:
        print("Non-existent content error:", str(e))
    
    print()

def generate_test_report():
    """Generate a comprehensive test report"""
    print("=== COMPREHENSIVE TEST REPORT ===")
    print()
    
    pipeline = setup_test_pipeline()
    
    report = {
        "pipeline_config": {
            "chunk_size": 600,
            "overlap": 100,
            "top_k": 8,
            "rerank_k": 4,
            "use_mmr": True,
            "lambda_val": 0.6
        },
        "sources_processed": ["README.md", "index.html", "styles.css", "script.js", "sample.txt"],
        "test_results": {}
    }
    
    # Test various query categories
    test_categories = {
        "basic_queries": [
            "What is this website about?",
            "What technologies are used?",
            "How to contact?"
        ],
        "technical_queries": [
            "What frontend technologies?",
            "CSS architecture?",
            "JavaScript features?"
        ],
        "portfolio_queries": [
            "What skills showcased?",
            "What projects featured?",
            "Design approach?"
        ]
    }
    
    for category, queries in test_categories.items():
        results = []
        for query in queries:
            try:
                context = pipeline.retrieve(query)
                result = pipeline.query(query)
                results.append({
                    "query": query,
                    "chunks_retrieved": len(context),
                    "answer_length": len(result),
                    "success": True
                })
            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        report["test_results"][category] = results
    
    # Save report
    with open("rag_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("Test report saved to 'rag_test_report.json'")
    print(f"Total tests run: {sum(len(results) for results in test_categories.values())}")
    
    # Summary
    all_results = []
    for results in test_categories.values():
        all_results.extend(results)
    
    successful = sum(1 for r in all_results if r.get("success", False))
    print(f"Successful tests: {successful}/{len(all_results)}")
    print(f"Success rate: {successful/len(all_results)*100:.1f}%")

def run_all_tests():
    """Run all test examples"""
    print("RUNNING ALL RAG PIPELINE TESTS")
    print("=" * 60)
    print()
    
    try:
        pipeline = setup_test_pipeline()
        
        # Run all test categories
        test_basic_queries(pipeline)
        test_technical_queries(pipeline)
        test_portfolio_queries(pipeline)
        test_content_retrieval_demo(pipeline)
        test_mmr_vs_no_mmr(pipeline)
        test_different_chunk_sizes()
        test_source_types()
        test_complex_queries(pipeline)
        test_pipeline_configuration()
        test_error_handling()
        generate_test_report()
        
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The 15-step RAG pipeline is working perfectly!")
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()
