"""
Production RAG Pipeline Application
15-Step RAG Pipeline ready for production deployment
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_pipeline_advanced import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="15-Step RAG Pipeline API",
    description="Production-ready RAG pipeline with 15 processing steps",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
rag_pipeline = None
pipeline_status = {"initialized": False, "sources_loaded": 0}

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    rerank_k: int = 3
    use_mmr: bool = True

class SourceRequest(BaseModel):
    sources: List[str]
    chunk_size: int = 600
    overlap: int = 100

class PipelineConfig(BaseModel):
    chunk_size: int = 600
    overlap: int = 100
    top_k: int = 8
    rerank_k: int = 4
    use_mmr: bool = True
    lambda_val: float = 0.6
    max_answer_words: int = 300

class QueryResponse(BaseModel):
    answer: str
    sources_used: int
    processing_time: float
    pipeline_config: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    pipeline_initialized: bool
    sources_loaded: int
    version: str

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    global rag_pipeline, pipeline_status
    
    try:
        logger.info("Initializing 15-Step RAG Pipeline...")
        
        # Default configuration
        config = PipelineConfig()
        
        # Initialize pipeline
        rag_pipeline = RAGPipeline(
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            top_k=config.top_k,
            rerank_k=config.rerank_k,
            use_mmr=config.use_mmr,
            lambda_val=config.lambda_val,
            max_answer_words=config.max_answer_words
        )
        
        # Load default sources if they exist
        default_sources = ["README.md", "index.html", "sample.txt"]
        available_sources = []
        
        for source in default_sources:
            if os.path.exists(source):
                available_sources.append(source)
        
        if available_sources:
            rag_pipeline.add_sources(available_sources)
            pipeline_status["sources_loaded"] = len(available_sources)
            logger.info(f"Loaded {len(available_sources)} default sources")
        
        pipeline_status["initialized"] = True
        logger.info("15-Step RAG Pipeline initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pipeline_status["initialized"] = False

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "15-Step RAG Pipeline API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if pipeline_status["initialized"] else "unhealthy",
        pipeline_initialized=pipeline_status["initialized"],
        sources_loaded=pipeline_status["sources_loaded"],
        version="1.0.0"
    )

@app.post("/config", response_model=Dict[str, str])
async def update_config(config: PipelineConfig):
    """Update pipeline configuration"""
    global rag_pipeline, pipeline_status
    
    try:
        logger.info(f"Updating pipeline configuration: {config}")
        
        # Reinitialize pipeline with new config
        rag_pipeline = RAGPipeline(
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            top_k=config.top_k,
            rerank_k=config.rerank_k,
            use_mmr=config.use_mmr,
            lambda_val=config.lambda_val,
            max_answer_words=config.max_answer_words
        )
        
        pipeline_status["initialized"] = True
        pipeline_status["sources_loaded"] = 0
        
        return {"message": "Configuration updated successfully"}
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sources", response_model=Dict[str, Any])
async def add_sources(request: SourceRequest, background_tasks: BackgroundTasks):
    """Add sources to the pipeline"""
    global rag_pipeline, pipeline_status
    
    if not pipeline_status["initialized"]:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        logger.info(f"Adding {len(request.sources)} sources")
        
        # Validate sources exist
        valid_sources = []
        for source in request.sources:
            if os.path.exists(source):
                valid_sources.append(source)
            else:
                logger.warning(f"Source not found: {source}")
        
        if not valid_sources:
            raise HTTPException(status_code=400, detail="No valid sources found")
        
        # Add sources to pipeline
        rag_pipeline.add_sources(valid_sources)
        pipeline_status["sources_loaded"] += len(valid_sources)
        
        return {
            "message": "Sources added successfully",
            "sources_added": len(valid_sources),
            "total_sources": pipeline_status["sources_loaded"],
            "valid_sources": valid_sources
        }
        
    except Exception as e:
        logger.error(f"Failed to add sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_pipeline(request: QueryRequest):
    """Query the RAG pipeline"""
    global rag_pipeline
    
    if not pipeline_status["initialized"]:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    if pipeline_status["sources_loaded"] == 0:
        raise HTTPException(status_code=400, detail="No sources loaded")
    
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Processing query: {request.query}")
        
        # Temporarily update pipeline settings for this query
        original_top_k = rag_pipeline.top_k
        original_rerank_k = rag_pipeline.rerank_k
        original_use_mmr = rag_pipeline.use_mmr
        
        rag_pipeline.top_k = request.top_k
        rag_pipeline.rerank_k = request.rerank_k
        rag_pipeline.use_mmr = request.use_mmr
        
        # Process query
        result = rag_pipeline.query(request.query)
        
        # Restore original settings
        rag_pipeline.top_k = original_top_k
        rag_pipeline.rerank_k = original_rerank_k
        rag_pipeline.use_mmr = original_use_mmr
        
        processing_time = time.time() - start_time
        
        logger.info(f"Query processed in {processing_time:.2f}s")
        
        return QueryResponse(
            answer=result,
            sources_used=len(rag_pipeline.retrieve(request.query)),
            processing_time=processing_time,
            pipeline_config={
                "chunk_size": rag_pipeline.chunk_size,
                "overlap": rag_pipeline.overlap,
                "top_k": request.top_k,
                "rerank_k": request.rerank_k,
                "use_mmr": request.use_mmr
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/retrieve")
async def retrieve_content(query: str, k: int = 5):
    """Retrieve raw content without LLM processing"""
    global rag_pipeline
    
    if not pipeline_status["initialized"]:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        logger.info(f"Retrieving content for: {query}")
        
        context = rag_pipeline.retrieve(query, k=k)
        
        results = []
        for i, (chunk, score) in enumerate(context, 1):
            results.append({
                "rank": i,
                "score": float(score),
                "source": chunk.metadata.get("filename", "Unknown"),
                "content": chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                "metadata": chunk.metadata
            })
        
        return {
            "query": query,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get detailed pipeline status"""
    global rag_pipeline, pipeline_status
    
    status = {
        "pipeline_initialized": pipeline_status["initialized"],
        "sources_loaded": pipeline_status["sources_loaded"],
        "version": "1.0.0"
    }
    
    if rag_pipeline:
        status.update({
            "chunk_size": rag_pipeline.chunk_size,
            "overlap": rag_pipeline.overlap,
            "top_k": rag_pipeline.top_k,
            "rerank_k": rag_pipeline.rerank_k,
            "use_mmr": rag_pipeline.use_mmr,
            "lambda_val": rag_pipeline.lambda_val,
            "max_answer_words": rag_pipeline.max_answer_words
        })
    
    return status

@app.delete("/reset")
async def reset_pipeline():
    """Reset the pipeline"""
    global rag_pipeline, pipeline_status
    
    try:
        logger.info("Resetting pipeline...")
        
        rag_pipeline = None
        pipeline_status = {"initialized": False, "sources_loaded": 0}
        
        # Reinitialize
        await startup_event()
        
        return {"message": "Pipeline reset successfully"}
        
    except Exception as e:
        logger.error(f"Failed to reset pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
