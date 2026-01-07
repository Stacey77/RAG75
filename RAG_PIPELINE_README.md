# Advanced 15-Step RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline implementing all 15 steps from source identification to output post-processing.

## 🔄 Pipeline Steps

### Index Building (Steps 1-8)
1. **Source Identification** - Validates and classifies sources (files/URLs)
2. **Data Ingestion** - Loads content from various formats (PDF, HTML, MD, TXT)
3. **Preprocessing** - Text normalization and cleaning
4. **Document Chunking** - Intelligent text splitting with overlap
5. **Metadata Tagging** - Automatic metadata extraction and tagging
6. **Text Embedding** - Vector embeddings using sentence transformers
7. **Vector Storage** - FAISS-based vector indexing
8. **Index Optimization** - Performance optimization hooks

### Query Processing (Steps 9-15)
9. **Query Handling** - Query normalization and validation
10. **Query Embedding** - Convert queries to vector space
11. **Similarity Search** - Top-K vector similarity search
12. **Content Filtering & Re-ranking** - Content filtering with MMR re-ranking
13. **Prompt Construction** - Context-aware prompt building
14. **LLM Completion** - OpenAI API integration with fallback
15. **Output Post-processing** - Response formatting and citation

## 🚀 Features

- **Multi-format support**: PDF, HTML, Markdown, plain text
- **Web scraping**: URL-based content ingestion
- **Advanced chunking**: Intelligent text splitting with overlap
- **FAISS indexing**: High-performance vector search
- **MMR re-ranking**: Maximal Marginal Relevance for diverse results
- **Content filtering**: Built-in safety and quality filters
- **Citation system**: Automatic source attribution
- **OpenAI integration**: GPT-4 and other models support
- **Fallback mechanisms**: Graceful degradation without API keys

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (optional, for LLM completion)
export OPENAI_API_KEY="your-api-key-here"
```

## 🔧 Quick Start

### Basic Usage

```python
from rag_pipeline_advanced import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(
    chunk_size=700,
    overlap=100,
    top_k=10,
    rerank_k=5,
    use_mmr=True,
    lambda_val=0.6,
    max_answer_words=250
)

# Add sources
pipeline.add_sources([
    "README.md",
    "index.html",
    "styles.css",
    "script.js"
])

# Query the pipeline
result = pipeline.query("What is this website about?")
print(result)
```

### Mixed Sources

```python
# Mix files and URLs
sources = [
    "README.md",
    {"uri": "https://example.com", "label": "Example Site"},
    {"uri": "./docs/report.pdf", "label": "Annual Report"}
]

pipeline.add_sources(sources)
```

## 🎯 Advanced Configuration

### Chunking Strategy
```python
pipeline = RAGPipeline(
    chunk_size=800,      # Chunk size in characters
    overlap=150,         # Overlap between chunks
    top_k=15,           # Initial retrieval count
    rerank_k=8,         # Final results after re-ranking
    use_mmr=True,       # Enable MMR re-ranking
    lambda_val=0.7      # MMR diversity parameter (0-1)
)
```

### LLM Configuration
```python
pipeline = RAGPipeline(
    model_name="gpt-4o-mini",  # OpenAI model
    max_answer_words=300       # Response length limit
)
```

## 📊 File Structure

```
personal-website/
├── rag_pipeline_advanced.py    # Main pipeline implementation
├── demo_advanced_rag.py        # Demo and testing script
├── requirements.txt            # Python dependencies
├── RAG_PIPELINE_README.md      # This documentation
├── README.md                   # Website content (source)
├── index.html                  # Website content (source)
├── styles.css                  # Website content (source)
└── script.js                   # Website content (source)
```

## 🚀 Running Demos

```bash
# Run interactive demo
python demo_advanced_rag.py

# Options:
# 1. Basic usage with website files
# 2. Mixed sources (files + URLs)
# 3. PDF support demo
# 4. Interactive mode
# 5. Run all demos
```

## 🔍 Query Examples

The pipeline can answer questions like:

- "What is this website about?"
- "What technologies are used in this project?"
- "How can I contact the website owner?"
- "What features does this website have?"
- "What skills are showcased in the portfolio?"

## 🛠️ Dependencies

### Core Dependencies
- `numpy` - Numerical operations
- `sentence-transformers` - Text embeddings
- `faiss-cpu` - Vector indexing (or `faiss-gpu` for GPU)

### Content Processing
- `PyPDF2` - PDF text extraction
- `beautifulsoup4` - HTML parsing
- `lxml` - XML/HTML parser
- `requests` - HTTP client for URLs

### LLM Integration
- `openai` - OpenAI API client

## ⚙️ Configuration Options

### Vector Store
- **Metric**: cosine, inner product (ip), L2
- **Index type**: Flat (exact search), IVF, PQ (for large datasets)

### Embedding Models
- Default: `sentence-transformers/all-MiniLM-L6-v2`
- Alternatives: `all-mpnet-base-v2`, `multi-qa-mpnet-base-dot-v1`

### Re-ranking
- **MMR**: Maximal Marginal Re-ranking for diversity
- **Lambda**: Balance between relevance and diversity (0-1)

## 🔧 Customization

### Custom Embedding Model
```python
from rag_pipeline_advanced import Embedder

embedder = Embedder("sentence-transformers/all-mpnet-base-v2")
pipeline = RAGPipeline(embedder=embedder)
```

### Custom Content Filter
```python
def custom_filter(chunk):
    # Your filtering logic
    return len(chunk.text) > 50 and "important" in chunk.text.lower()

# Override in pipeline
pipeline.content_filter = custom_filter
```

## 📈 Performance Tips

1. **Chunk Size**: 600-800 characters works well for most content
2. **Overlap**: 10-20% of chunk size maintains context
3. **Top-K**: 10-15 initial candidates, 3-8 final results
4. **MMR**: Enable for diverse results, disable for pure relevance
5. **FAISS**: Use `faiss-gpu` for large datasets

## 🚨 Error Handling

The pipeline includes graceful fallbacks:
- Missing dependencies: Clear error messages with installation instructions
- Missing API keys: Fallback to local response generation
- Invalid sources: Skipped with logging
- Empty results: Informative "no context found" responses

## 📝 Logging

Configure logging for debugging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🤝 Contributing

The pipeline is modular and extensible:
- Add new source types in the ingestion step
- Implement custom chunking strategies
- Add new re-ranking algorithms
- Integrate different LLM providers

## 📄 License

This RAG pipeline implementation is provided as-is for educational and development purposes.
