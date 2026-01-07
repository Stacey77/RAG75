"""
RAG Pipeline Demo Script
Demonstrates the 15-step RAG pipeline with the personal website content
"""

import asyncio
import json
from rag_pipeline import RAGPipeline

async def demo_rag_pipeline():
    """Demo the complete RAG pipeline"""
    
    print("🚀 Starting RAG Pipeline Demo")
    print("=" * 50)
    
    # Load configuration
    with open('rag_config.json', 'r') as f:
        config = json.load(f)
    
    sources = config['sources']
    print(f"📁 Sources to process: {sources}")
    
    # Initialize pipeline
    pipeline = RAGPipeline(sources)
    
    # Build index
    print("\n🔧 Building RAG index...")
    try:
        await pipeline.build_index()
        print("✅ Index built successfully!")
    except Exception as e:
        print(f"❌ Error building index: {e}")
        return
    
    # Test queries
    test_queries = [
        "What is this project about?",
        "What technologies are used in this website?",
        "How can I contact the owner?",
        "What features does this website have?",
        "What skills are showcased?"
    ]
    
    print("\n🔍 Processing test queries...")
    print("-" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 30)
        
        try:
            result = await pipeline.query(query)
            print(f"Answer: {result['answer']}")
            print(f"Sources used: {result['sources_used']}")
            print(f"Confidence: {result['confidence_score']}")
            print(f"Processing time: {result['processing_time']}")
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    print("\n🎉 Demo completed!")

async def interactive_mode():
    """Interactive mode for custom queries"""
    
    print("\n🎯 Interactive RAG Mode")
    print("Type 'quit' to exit")
    print("-" * 30)
    
    # Load configuration and build pipeline
    with open('rag_config.json', 'r') as f:
        config = json.load(f)
    
    sources = config['sources']
    pipeline = RAGPipeline(sources)
    
    print("🔧 Building index...")
    await pipeline.build_index()
    print("✅ Ready for queries!")
    
    while True:
        try:
            query = input("\n🔍 Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print("🤔 Processing...")
            result = await pipeline.query(query)
            
            print(f"\n💬 Answer: {result['answer']}")
            print(f"📊 Confidence: {result['confidence_score']}")
            print(f"⏱️  Time: {result['processing_time']}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

async def main():
    """Main demo function"""
    print("🎭 RAG Pipeline Demo Options:")
    print("1. Run demo queries")
    print("2. Interactive mode")
    print("3. Both")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == '1':
        await demo_rag_pipeline()
    elif choice == '2':
        await interactive_mode()
    elif choice == '3':
        await demo_rag_pipeline()
        await interactive_mode()
    else:
        print("Running demo by default...")
        await demo_rag_pipeline()

if __name__ == "__main__":
    asyncio.run(main())
