import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
from rag_pipeline_advanced import RAGPipeline

print('=== RAG Pipeline Content Retrieval Demo ===')
print()

# Initialize pipeline
pipeline = RAGPipeline(chunk_size=400, overlap=50, top_k=3, rerank_k=2)

# Build index with website files
sources = ['README.md', 'sample.txt']
pipeline.add_sources(sources)

# Test retrieval with a simple query
query = 'What features does this website have?'
print(f'Query: {query}')

# Get the context chunks to show what was retrieved
context = pipeline.retrieve(query)
print(f'Retrieved {len(context)} relevant chunks:')
print()

for i, (chunk, score) in enumerate(context, 1):
    print(f'Chunk {i} (Score: {score:.3f}):')
    print(f'Source: {chunk.metadata.get("filename", "Unknown")}')
    print(f'Content: {chunk.text[:150]}...')
    print('-' * 40)

print('=== Pipeline Successfully Retrieved Relevant Content ===')
print('The system found and ranked the most relevant text chunks!')
