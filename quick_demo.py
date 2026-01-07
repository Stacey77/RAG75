"""
Quick Demo Examples for 15-Step RAG Pipeline
Showcase specific features and capabilities
"""

from rag_pipeline_advanced import RAGPipeline

def demo_1_basic_search():
    """Demo 1: Basic search functionality"""
    print("DEMO 1: Basic Search")
    print("-" * 30)
    
    pipeline = RAGPipeline(chunk_size=400, top_k=3, rerank_k=2)
    pipeline.add_sources(["README.md"])
    
    query = "What is this website?"
    result = pipeline.query(query)
    print(f"Query: {query}")
    print(f"Result: {result[:100]}...")
    print()

def demo_2_content_retrieval():
    """Demo 2: Show retrieved content"""
    print("DEMO 2: Content Retrieval")
    print("-" * 30)
    
    pipeline = RAGPipeline(chunk_size=300, top_k=4)
    pipeline.add_sources(["README.md", "sample.txt"])
    
    query = "What features are mentioned?"
    context = pipeline.retrieve(query)
    
    print(f"Query: {query}")
    print(f"Found {len(context)} relevant chunks:")
    for i, (chunk, score) in enumerate(context, 1):
        print(f"  {i}. {chunk.metadata.get('filename')} (score: {score:.3f})")
        print(f"     {chunk.text[:60]}...")
    print()

def demo_3_mmr_re_ranking():
    """Demo 3: MMR re-ranking comparison"""
    print("DEMO 3: MMR Re-ranking")
    print("-" * 30)
    
    query = "technologies used"
    
    # Without MMR
    pipeline1 = RAGPipeline(use_mmr=False, top_k=3)
    pipeline1.add_sources(["README.md", "index.html"])
    results1 = pipeline1.retrieve(query)
    
    # With MMR
    pipeline2 = RAGPipeline(use_mmr=True, top_k=3)
    pipeline2.add_sources(["README.md", "index.html"])
    results2 = pipeline2.retrieve(query)
    
    print(f"Query: {query}")
    print("Without MMR:", [f"{c.metadata.get('filename')}" for c, _ in results1])
    print("With MMR:", [f"{c.metadata.get('filename')}" for c, _ in results2])
    print()

def demo_4_different_sources():
    """Demo 4: Different file types"""
    print("DEMO 4: Multiple Source Types")
    print("-" * 30)
    
    sources = ["README.md", "index.html", "sample.txt"]
    
    for source in sources:
        pipeline = RAGPipeline(chunk_size=250, top_k=2)
        pipeline.add_sources([source])
        
        query = "main content"
        result = pipeline.query(query)
        
        print(f"Source: {source}")
        print(f"Preview: {result[:80]}...")
        print()

def demo_5_chunk_size_impact():
    """Demo 5: Impact of chunk size"""
    print("DEMO 5: Chunk Size Impact")
    print("-" * 30)
    
    query = "What makes this modern?"
    
    for size in [200, 500, 800]:
        pipeline = RAGPipeline(chunk_size=size, top_k=2)
        pipeline.add_sources(["README.md"])
        
        context = pipeline.retrieve(query)
        avg_chunk_len = sum(len(c.text) for c, _ in context) // len(context)
        
        print(f"Chunk size {size}: avg retrieved chunk {avg_chunk_len} chars")
    print()

def demo_6_pipeline_steps():
    """Demo 6: Show all 15 steps in action"""
    print("DEMO 6: 15-Step Pipeline")
    print("-" * 30)
    
    print("Steps 1-8: Building Index")
    print("  1. Source Identification: README.md, index.html")
    print("  2. Data Ingestion: Loading files...")
    print("  3. Preprocessing: Cleaning text...")
    print("  4. Document Chunking: Splitting text...")
    print("  5. Metadata Tagging: Adding tags...")
    print("  6. Text Embedding: Creating vectors...")
    print("  7. Vector Storage: Storing in FAISS...")
    print("  8. Index Optimization: Optimizing search...")
    
    pipeline = RAGPipeline(chunk_size=400, top_k=3)
    pipeline.add_sources(["README.md", "index.html"])
    
    print("Steps 9-15: Query Processing")
    query = "What are the key features?"
    print(f"  9. Query Handling: '{query}'")
    print(" 10. Query Embedding: Converting to vector...")
    print(" 11. Similarity Search: Finding matches...")
    print(" 12. Content Filtering: Applying filters...")
    print(" 13. Prompt Construction: Building context...")
    print(" 14. LLM Completion: Generating answer...")
    print("  15. Output Post-processing: Formatting result...")
    
    result = pipeline.query(query)
    print(f"Final Result: {result[:100]}...")
    print()

def run_quick_demos():
    """Run all quick demos"""
    print("QUICK RAG PIPELINE DEMOS")
    print("=" * 40)
    print()
    
    demo_1_basic_search()
    demo_2_content_retrieval()
    demo_3_mmr_re_ranking()
    demo_4_different_sources()
    demo_5_chunk_size_impact()
    demo_6_pipeline_steps()
    
    print("All demos completed successfully!")
    print("The 15-step RAG pipeline is fully operational!")

if __name__ == "__main__":
    run_quick_demos()
