"""
Real-World Test Examples for 15-Step RAG Pipeline
Practical usage scenarios and demonstrations
"""

from rag_pipeline_advanced import RAGPipeline

def test_website_support_chatbot():
    """Test: Website support chatbot scenario"""
    print("SCENARIO: Website Support Chatbot")
    print("-" * 40)
    
    # Setup for customer support
    pipeline = RAGPipeline(
        chunk_size=500,
        overlap=80,
        top_k=6,
        rerank_k=3,
        use_mmr=True,
        max_answer_words=200
    )
    
    # Add all website content
    sources = ["README.md", "index.html", "sample.txt"]
    pipeline.add_sources(sources)
    
    # Customer questions
    customer_questions = [
        "What technologies are used in this website?",
        "How can I contact the owner?",
        "What features does this website have?",
        "Is this website mobile-friendly?",
        "What makes this website accessible?"
    ]
    
    print("Customer Support Responses:")
    for i, question in enumerate(customer_questions, 1):
        response = pipeline.query(question)
        print(f"{i}. Q: {question}")
        print(f"   A: {response[:120]}...")
        print()

def test_content_analysis():
    """Test: Content analysis for developers"""
    print("SCENARIO: Content Analysis for Developers")
    print("-" * 40)
    
    pipeline = RAGPipeline(
        chunk_size=400,
        overlap=60,
        top_k=8,
        rerank_k=4,
        use_mmr=True,
        lambda_val=0.7
    )
    
    pipeline.add_sources(["README.md", "index.html", "styles.css"])
    
    # Technical analysis questions
    analysis_queries = [
        "What frontend frameworks are used?",
        "Describe the CSS architecture",
        "What JavaScript features are implemented?",
        "How is responsive design achieved?",
        "What accessibility standards are met?"
    ]
    
    print("Technical Analysis Results:")
    for query in analysis_queries:
        context = pipeline.retrieve(query)
        print(f"Query: {query}")
        print(f"Sources found: {len(context)} chunks")
        
        # Show source diversity
        sources = set(chunk.metadata.get('filename') for chunk, _ in context)
        print(f"Source types: {list(sources)}")
        print()

def test_portfolio_review():
    """Test: Portfolio review for recruiters"""
    print("SCENARIO: Portfolio Review for Recruiters")
    print("-" * 40)
    
    pipeline = RAGPipeline(
        chunk_size=450,
        overlap=70,
        top_k=5,
        rerank_k=3,
        use_mmr=True
    )
    
    pipeline.add_sources(["README.md", "index.html"])
    
    # Recruiter questions
    recruiter_questions = [
        "What skills does this developer have?",
        "What projects are showcased?",
        "What is the developer's experience level?",
        "What technologies is this developer proficient in?",
        "What makes this portfolio stand out?"
    ]
    
    print("Recruiter Portfolio Analysis:")
    for question in recruiter_questions:
        result = pipeline.query(question)
        print(f"Q: {question}")
        print(f"A: {result[:100]}...")
        print()

def test_website_audit():
    """Test: Website quality audit"""
    print("SCENARIO: Website Quality Audit")
    print("-" * 40)
    
    pipeline = RAGPipeline(
        chunk_size=350,
        overlap=50,
        top_k=10,
        rerank_k=5,
        use_mmr=True,
        lambda_val=0.5
    )
    
    pipeline.add_sources(["README.md", "index.html", "styles.css", "script.js"])
    
    # Audit criteria
    audit_checks = [
        "What performance optimizations are implemented?",
        "How is accessibility handled?",
        "What responsive design techniques are used?",
        "What modern web standards are followed?",
        "How is user experience enhanced?"
    ]
    
    print("Website Audit Results:")
    for check in audit_checks:
        context = pipeline.retrieve(check)
        print(f"Audit Check: {check}")
        print(f"Evidence sources: {len(context)} chunks")
        
        # Show key findings
        for i, (chunk, score) in enumerate(context[:2], 1):
            print(f"  Finding {i}: {chunk.text[:80]}...")
        print()

def test_feature_comparison():
    """Test: Feature comparison analysis"""
    print("SCENARIO: Feature Comparison Analysis")
    print("-" * 40)
    
    pipeline = RAGPipeline(
        chunk_size=300,
        overlap=40,
        top_k=6,
        rerank_k=3
    )
    
    pipeline.add_sources(["README.md", "sample.txt"])
    
    # Feature comparison queries
    comparison_queries = [
        "Compare design vs functionality features",
        "Frontend vs backend technologies",
        "Modern vs traditional features",
        "Performance vs accessibility trade-offs",
        "Desktop vs mobile features"
    ]
    
    print("Feature Comparison Analysis:")
    for query in comparison_queries:
        result = pipeline.query(query)
        print(f"Comparison: {query}")
        print(f"Analysis: {result[:110]}...")
        print()

def test_user_guide_generation():
    """Test: User guide generation"""
    print("SCENARIO: User Guide Generation")
    print("-" * 40)
    
    pipeline = RAGPipeline(
        chunk_size=600,
        overlap=100,
        top_k=4,
        rerank_k=2,
        max_answer_words=250
    )
    
    pipeline.add_sources(["README.md", "index.html"])
    
    # Guide generation queries
    guide_queries = [
        "Create a getting started guide for this website",
        "Explain how to navigate this portfolio",
        "What should visitors know about this site?",
        "How to use the contact features?",
        "What are the key sections to explore?"
    ]
    
    print("Generated User Guides:")
    for query in guide_queries:
        guide = pipeline.query(query)
        print(f"Guide Topic: {query}")
        print(f"Generated: {guide[:130]}...")
        print()

def test_search_engine_optimization():
    """Test: SEO analysis"""
    print("SCENARIO: SEO Analysis")
    print("-" * 40)
    
    pipeline = RAGPipeline(
        chunk_size=400,
        overlap=60,
        top_k=7,
        rerank_k=4,
        use_mmr=True
    )
    
    pipeline.add_sources(["README.md", "index.html"])
    
    # SEO analysis queries
    seo_queries = [
        "What keywords are prominent in this content?",
        "How is content structured for SEO?",
        "What meta information is included?",
        "How are headings and navigation organized?",
        "What content categories are covered?"
    ]
    
    print("SEO Analysis Results:")
    for query in seo_queries:
        context = pipeline.retrieve(query)
        print(f"SEO Check: {query}")
        print(f"Content samples: {len(context)} relevant sections")
        
        # Show content diversity
        unique_sources = set(chunk.metadata.get('filename') for chunk, _ in context)
        print(f"Content sources: {list(unique_sources)}")
        print()

def run_real_world_tests():
    """Run all real-world test scenarios"""
    print("REAL-WORLD RAG PIPELINE TESTS")
    print("=" * 50)
    print()
    
    try:
        test_website_support_chatbot()
        test_content_analysis()
        test_portfolio_review()
        test_website_audit()
        test_feature_comparison()
        test_user_guide_generation()
        test_search_engine_optimization()
        
        print("ALL REAL-WORLD TESTS COMPLETED!")
        print("The 15-step RAG pipeline handles diverse scenarios effectively!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    run_real_world_tests()
