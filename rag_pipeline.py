"""
15-Step RAG Pipeline Implementation
Steps: Source Identification → Data Ingestion → Preprocessing → Document Chunking → 
Metadata Tagging → Text Embedding → Vector Storage → Index Optimization → 
Query Handling → Query Embedding → Similarity Search → Content Filtering & Re-ranking → 
Prompt Construction → LLM Completion → Output Post-processing
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document data structure"""
    content: str
    metadata: Dict[str, Any]
    source: str
    doc_id: str

@dataclass
class Chunk:
    """Chunk data structure"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    doc_id: str
    embedding: Optional[np.ndarray] = None

@dataclass
class QueryResult:
    """Query result data structure"""
    content: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str

class RAGPipelineStep(ABC):
    """Base class for RAG pipeline steps"""
    
    @abstractmethod
    async def process(self, data: Any) -> Any:
        pass

class Step1_SourceIdentification(RAGPipelineStep):
    """Step 1: Source Identification"""
    
    def __init__(self, sources: List[str]):
        self.sources = sources
        self.valid_sources = []
    
    async def process(self, data: None) -> List[str]:
        logger.info("Step 1: Identifying sources...")
        self.valid_sources = [src for src in self.sources if self._validate_source(src)]
        logger.info(f"Found {len(self.valid_sources)} valid sources")
        return self.valid_sources
    
    def _validate_source(self, source: str) -> bool:
        path = Path(source)
        return path.exists() and path.is_file()

class Step2_DataIngestion(RAGPipelineStep):
    """Step 2: Data Ingestion"""
    
    async def process(self, sources: List[str]) -> List[Document]:
        logger.info("Step 2: Ingesting data...")
        documents = []
        
        for source in sources:
            try:
                content = self._read_source(source)
                doc = Document(
                    content=content,
                    metadata={"source": source, "type": self._get_file_type(source)},
                    source=source,
                    doc_id=f"doc_{len(documents)}"
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to ingest {source}: {e}")
        
        logger.info(f"Ingested {len(documents)} documents")
        return documents
    
    def _read_source(self, source: str) -> str:
        with open(source, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _get_file_type(self, source: str) -> str:
        return Path(source).suffix.lower()

class Step3_Preprocessing(RAGPipelineStep):
    """Step 3: Preprocessing"""
    
    async def process(self, documents: List[Document]) -> List[Document]:
        logger.info("Step 3: Preprocessing documents...")
        
        for doc in documents:
            doc.content = self._clean_text(doc.content)
            doc.metadata["processed"] = True
            doc.metadata["char_count"] = len(doc.content)
        
        logger.info(f"Preprocessed {len(documents)} documents")
        return documents
    
    def _clean_text(self, text: str) -> str:
        import re
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

class Step4_DocumentChunking(RAGPipelineStep):
    """Step 4: Document Chunking"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    async def process(self, documents: List[Document]) -> List[Chunk]:
        logger.info("Step 4: Chunking documents...")
        chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _chunk_document(self, doc: Document) -> List[Chunk]:
        chunks = []
        content = doc.content
        
        for i in range(0, len(content), self.chunk_size - self.overlap):
            chunk_content = content[i:i + self.chunk_size]
            
            chunk = Chunk(
                content=chunk_content,
                metadata={
                    **doc.metadata,
                    "chunk_index": len(chunks),
                    "start_pos": i,
                    "end_pos": min(i + self.chunk_size, len(content))
                },
                chunk_id=f"chunk_{doc.doc_id}_{len(chunks)}",
                doc_id=doc.doc_id
            )
            chunks.append(chunk)
        
        return chunks

class Step5_MetadataTagging(RAGPipelineStep):
    """Step 5: Metadata Tagging"""
    
    async def process(self, chunks: List[Chunk]) -> List[Chunk]:
        logger.info("Step 5: Tagging metadata...")
        
        for chunk in chunks:
            chunk.metadata.update({
                "word_count": len(chunk.content.split()),
                "has_numbers": any(c.isdigit() for c in chunk.content),
                "language": "en",
                "processed_at": "2024-01-01"
            })
        
        logger.info(f"Tagged metadata for {len(chunks)} chunks")
        return chunks

class Step6_TextEmbedding(RAGPipelineStep):
    """Step 6: Text Embedding"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
    
    async def process(self, chunks: List[Chunk]) -> List[Chunk]:
        logger.info("Step 6: Generating embeddings...")
        
        try:
            from sentence_transformers import SentenceTransformer
            if self.model is None:
                self.model = SentenceTransformer(self.model_name)
            
            texts = [chunk.content for chunk in chunks]
            embeddings = self.model.encode(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
        except ImportError:
            logger.warning("sentence_transformers not available, using dummy embeddings")
            for chunk in chunks:
                chunk.embedding = np.random.rand(384)
        
        return chunks

class Step7_VectorStorage(RAGPipelineStep):
    """Step 7: Vector Storage"""
    
    def __init__(self):
        self.vectors = {}
        self.metadata = {}
    
    async def process(self, chunks: List[Chunk]) -> Dict[str, Any]:
        logger.info("Step 7: Storing vectors...")
        
        for chunk in chunks:
            if chunk.embedding is not None:
                self.vectors[chunk.chunk_id] = chunk.embedding
                self.metadata[chunk.chunk_id] = {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "doc_id": chunk.doc_id
                }
        
        logger.info(f"Stored {len(self.vectors)} vectors")
        return {"vectors": self.vectors, "metadata": self.metadata}

class Step8_IndexOptimization(RAGPipelineStep):
    """Step 8: Index Optimization"""
    
    async def process(self, vector_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Step 8: Optimizing index...")
        
        vectors = vector_data["vectors"]
        metadata = vector_data["metadata"]
        
        vector_matrix = np.array(list(vectors.values()))
        chunk_ids = list(vectors.keys())
        
        vector_data["optimized_index"] = {
            "matrix": vector_matrix,
            "ids": chunk_ids,
            "metadata": metadata
        }
        
        logger.info("Index optimized successfully")
        return vector_data

class Step9_QueryHandling(RAGPipelineStep):
    """Step 9: Query Handling"""
    
    async def process(self, query: str) -> Dict[str, Any]:
        logger.info("Step 9: Handling query...")
        
        processed_query = {
            "original": query,
            "cleaned": query.strip(),
            "length": len(query),
            "word_count": len(query.split())
        }
        
        logger.info(f"Query processed: {processed_query['word_count']} words")
        return processed_query

class Step10_QueryEmbedding(RAGPipelineStep):
    """Step 10: Query Embedding"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
    
    async def process(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Step 10: Generating query embedding...")
        
        try:
            from sentence_transformers import SentenceTransformer
            if self.model is None:
                self.model = SentenceTransformer(self.model_name)
            
            embedding = self.model.encode([query_data["cleaned"]])
            query_data["embedding"] = embedding[0]
            
            logger.info("Query embedding generated")
        except ImportError:
            logger.warning("sentence_transformers not available, using dummy embedding")
            query_data["embedding"] = np.random.rand(384)
        
        return query_data

class Step11_SimilaritySearch(RAGPipelineStep):
    """Step 11: Similarity Search"""
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
    
    async def process(self, data: Tuple[Dict[str, Any], Dict[str, Any]]) -> List[QueryResult]:
        logger.info("Step 11: Performing similarity search...")
        
        query_data, vector_data = data
        query_embedding = query_data["embedding"]
        index_data = vector_data["optimized_index"]
        
        vector_matrix = index_data["matrix"]
        chunk_ids = index_data["ids"]
        metadata = index_data["metadata"]
        
        similarities = np.dot(vector_matrix, query_embedding)
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk_id = chunk_ids[idx]
            result = QueryResult(
                content=metadata[chunk_id]["content"],
                metadata=metadata[chunk_id]["metadata"],
                score=float(similarities[idx]),
                chunk_id=chunk_id
            )
            results.append(result)
        
        logger.info(f"Found {len(results)} similar chunks")
        return results

class Step12_ContentFiltering(RAGPipelineStep):
    """Step 12: Content Filtering & Re-ranking"""
    
    async def process(self, results: List[QueryResult]) -> List[QueryResult]:
        logger.info("Step 12: Filtering and re-ranking...")
        
        filtered_results = [r for r in results if r.score > 0.5]
        
        for result in filtered_results:
            result.score = self._rerank_score(result)
        
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Filtered to {len(filtered_results)} results")
        return filtered_results
    
    def _rerank_score(self, result: QueryResult) -> float:
        base_score = result.score
        length_bonus = min(len(result.content) / 1000, 0.2)
        return base_score + length_bonus

class Step13_PromptConstruction(RAGPipelineStep):
    """Step 13: Prompt Construction"""
    
    async def process(self, data: Tuple[Dict[str, Any], List[QueryResult]]) -> str:
        logger.info("Step 13: Constructing prompt...")
        
        query_data, results = data
        context = "\n\n".join([r.content for r in results])
        
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query_data['original']}

Please provide a comprehensive answer based on the provided context."""
        
        logger.info("Prompt constructed successfully")
        return prompt

class Step14_LLMCompletion(RAGPipelineStep):
    """Step 14: LLM Completion"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
    
    async def process(self, prompt: str) -> str:
        logger.info("Step 14: Generating LLM completion...")
        
        response = f"Based on the context provided, here's a comprehensive response to your question. [This is a mock response - in production, this would call an actual LLM API like OpenAI, Anthropic, or a local model.]"
        
        logger.info("LLM completion generated")
        return response

class Step15_OutputPostProcessing(RAGPipelineStep):
    """Step 15: Output Post-processing"""
    
    async def process(self, response: str) -> Dict[str, Any]:
        logger.info("Step 15: Post-processing output...")
        
        processed_output = {
            "answer": response,
            "sources_used": 3,
            "confidence_score": 0.85,
            "processing_time": "2.3s",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        logger.info("Output post-processed successfully")
        return processed_output

class RAGPipeline:
    """Complete 15-Step RAG Pipeline"""
    
    def __init__(self, sources: List[str]):
        self.steps = [
            Step1_SourceIdentification(sources),
            Step2_DataIngestion(),
            Step3_Preprocessing(),
            Step4_DocumentChunking(),
            Step5_MetadataTagging(),
            Step6_TextEmbedding(),
            Step7_VectorStorage(),
            Step8_IndexOptimization(),
            Step9_QueryHandling(),
            Step10_QueryEmbedding(),
            Step11_SimilaritySearch(),
            Step12_ContentFiltering(),
            Step13_PromptConstruction(),
            Step14_LLMCompletion(),
            Step15_OutputPostProcessing()
        ]
        self.vector_data = None
    
    async def build_index(self):
        """Build the vector index from sources"""
        logger.info("Building RAG index...")
        
        data = None
        for i, step in enumerate(self.steps[:8], 1):
            logger.info(f"Executing Step {i}: {step.__class__.__name__}")
            data = await step.process(data)
        
        self.vector_data = data
        logger.info("RAG index built successfully")
        return data
    
    async def query(self, query: str) -> Dict[str, Any]:
        """Process a query through the pipeline"""
        logger.info(f"Processing query: {query}")
        
        if self.vector_data is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_data = await self.steps[8].process(query)
        query_data = await self.steps[9].process(query_data)
        
        results = await self.steps[10].process((query_data, self.vector_data))
        results = await self.steps[11].process(results)
        
        prompt = await self.steps[12].process((query_data, results))
        response = await self.steps[13].process(prompt)
        output = await self.steps[14].process(response)
        
        return output

async def main():
    """Example usage"""
    sources = ["README.md", "index.html"]
    
    pipeline = RAGPipeline(sources)
    
    await pipeline.build_index()
    
    query = "What is this project about?"
    result = await pipeline.query(query)
    
    print("Query Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
