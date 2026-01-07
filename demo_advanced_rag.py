"""
Advanced RAG Pipeline Demo Script
Demonstrates the production-ready 15-step RAG pipeline with advanced features
"""

import os
import logging
from rag_pipeline_advanced import RAGPipeline

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def demo_basic_usage():
    """Demonstrate basic RAG pipeline usage"""
    print("Advanced RAG Pipeline Demo")
    print("=" * 50)
    
    # Initialize pipeline with advanced settings
    pipeline = RAGPipeline(
        chunk_size=700,
        overlap=100,
        top_k=10,
        rerank_k=5,
        use_mmr=True,
        lambda_val=0.6,
        max_answer_words=250,
        model_name="gpt-4o-mini"
    )
    
    print("Adding sources...")
    # Add sources from the personal website
    sources = [
        "README.md",
        "index.html", 
        "styles.css",
        "script.js"
    ]
    
    try:
        pipeline.add_sources(sources)
        print(f"Successfully processed {len(sources)} sources")
    except Exception as e:
        print(f"Error processing sources: {e}")
        return
    
    # Test queries
    test_queries = [
        "What is this website about?",
        "What technologies are used in this project?",
        "How can I contact the website owner?",
        "What features does this website have?",
        "What skills are showcased in the portfolio?",
        "Describe the design approach used in this website"
    ]
    
    print("\nProcessing test queries...")
    print("-" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        try:
            result = pipeline.query(query)
            print(f"Answer: {result}")
        except Exception as e:
            print(f"Error processing query: {e}")

def demo_mixed_sources():
    """Demonstrate usage with mixed file and URL sources"""
    print("\nMixed Sources Demo")
    print("=" * 30)
    
    pipeline = RAGPipeline(
        chunk_size=600,
        overlap=80,
        top_k=8,
        rerank_k=4,
        use_mmr=True,
        lambda_val=0.7,
        max_answer_words=200
    )
    
    # Mix of local files and URLs (URLs will require internet connection)
    sources = [
        "README.md",
        {"uri": "https://httpbin.org/html", "label": "Sample HTML Page"},
        "index.html"
    ]
    
    print("Adding mixed sources...")
    try:
        pipeline.add_sources(sources)
        print("Mixed sources processed successfully")
        
        query = "What information is available from these sources?"
        result = pipeline.query(query)
        print(f"\nAnswer: {result}")
        
    except Exception as e:
        print(f"Error with mixed sources: {e}")

def demo_pdf_support():
    """Demonstrate PDF processing capabilities"""
    print("\nPDF Support Demo")
    print("=" * 25)
    
    # Check if any PDF files exist
    import glob
    pdf_files = glob.glob("*.pdf")
    
    if not pdf_files:
        print("No PDF files found. Skipping PDF demo.")
        return
    
    pipeline = RAGPipeline(
        chunk_size=800,
        overlap=150,
        top_k=6,
        rerank_k=3,
        use_mmr=False  # Simpler for PDF content
    )
    
    try:
        pipeline.add_sources(pdf_files)
        print(f"Processed {len(pdf_files)} PDF file(s)")
        
        query = "What are the main topics covered in these documents?"
        result = pipeline.query(query)
        print(f"\nAnswer: {result}")
        
    except Exception as e:
        print(f"Error processing PDFs: {e}")

def interactive_mode():
    """Interactive mode for custom queries"""
    print("\nInteractive RAG Mode")
    print("Type 'quit' to exit")
    print("-" * 30)
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        chunk_size=700,
        overlap=100,
        top_k=10,
        rerank_k=5,
        use_mmr=True,
        lambda_val=0.6,
        max_answer_words=300
    )
    
    # Add website sources
    sources = ["README.md", "index.html", "styles.css", "script.js"]
    
    print("Building index...")
    try:
        pipeline.add_sources(sources)
        print("Ready for queries!")
    except Exception as e:
        print(f"Error building index: {e}")
        return
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("Processing...")
            result = pipeline.query(query)
            print(f"\nAnswer: {result}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def check_environment():
    """Check if required dependencies are available"""
    print("Environment Check")
    print("-" * 25)
    
    deps = {
        'sentence-transformers': False,
        'faiss': False,
        'PyPDF2': False,
        'beautifulsoup4': False,
        'openai': False
    }
    
    try:
        import sentence_transformers
        deps['sentence-transformers'] = True
    except ImportError:
        pass
    
    try:
        import faiss
        deps['faiss'] = True
    except ImportError:
        pass
    
    try:
        import PyPDF2
        deps['PyPDF2'] = True
    except ImportError:
        pass
    
    try:
        import bs4
        deps['beautifulsoup4'] = True
    except ImportError:
        pass
    
    try:
        import openai
        deps['openai'] = True
    except ImportError:
        pass
    
    for dep, available in deps.items():
        status = "OK" if available else "MISSING"
        print(f"{status} {dep}")
    
    # Check OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print("OPENAI_API_KEY is set")
    else:
        print("OPENAI_API_KEY not set (will use fallback)")
    
    print("\nTo install missing dependencies:")
    print("pip install -r requirements.txt")

def main():
    """Main demo function"""
    setup_logging()
    
    print("\nAdvanced RAG Pipeline Demo")
    print("=" * 40)
    
    # Check environment first
    check_environment()
    
    print("\nDemo Options:")
    print("1. Basic usage with website files")
    print("2. Mixed sources (files + URLs)")
    print("3. PDF support demo")
    print("4. Interactive mode")
    print("5. Run all demos")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return
    
    if choice == '1':
        demo_basic_usage()
    elif choice == '2':
        demo_mixed_sources()
    elif choice == '3':
        demo_pdf_support()
    elif choice == '4':
        interactive_mode()
    elif choice == '5':
        demo_basic_usage()
        demo_mixed_sources()
        demo_pdf_support()
        interactive_mode()
    else:
        print("Running basic demo by default...")
        demo_basic_usage()

if __name__ == "__main__":
    main()
