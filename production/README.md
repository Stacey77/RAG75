# 15-Step RAG Pipeline - Production Deployment

Production-ready deployment of the 15-step RAG pipeline with monitoring, scaling, and security features.

## 🚀 Quick Start

### 1. Prerequisites
- Docker & Docker Compose
- At least 4GB RAM
- OpenAI API Key (optional, for LLM completion)

### 2. Deployment
```bash
cd production
chmod +x deploy.sh
./deploy.sh
```

### 3. Verify Deployment
```bash
curl http://localhost:8000/health
```

## 📋 Services

### Core Services
- **RAG Pipeline API** (Port 8000) - Main 15-step pipeline
- **Nginx** (Port 80/443) - Load balancer and reverse proxy
- **Redis** (Port 6379) - Caching and session storage

### Monitoring
- **Prometheus** (Port 9090) - Metrics collection
- **Grafana** (Port 3000) - Visualization dashboard

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:

```bash
OPENAI_API_KEY=your_key_here
DEFAULT_CHUNK_SIZE=600
DEFAULT_TOP_K=8
DEFAULT_USE_MMR=true
```

### Pipeline Configuration
Update pipeline settings via API:

```bash
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_size": 500,
    "top_k": 6,
    "use_mmr": true,
    "lambda_val": 0.7
  }'
```

## 📡 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Add Sources
```bash
curl -X POST http://localhost:8000/sources \
  -H "Content-Type: application/json" \
  -d '{
    "sources": ["README.md", "docs/report.pdf"],
    "chunk_size": 600,
    "overlap": 100
  }'
```

### Query Pipeline
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What technologies are used?",
    "top_k": 5,
    "rerank_k": 3,
    "use_mmr": true
  }'
```

### Retrieve Content
```bash
curl "http://localhost:8000/retrieve?query=features&k=3"
```

## 🏗️ Architecture

### 15-Step Pipeline
1. **Source Identification** - Validate and classify sources
2. **Data Ingestion** - Load from files/URLs/PDFs
3. **Preprocessing** - Text normalization and cleaning
4. **Document Chunking** - Intelligent text splitting
5. **Metadata Tagging** - Automatic metadata extraction
6. **Text Embedding** - Vector generation
7. **Vector Storage** - FAISS indexing
8. **Index Optimization** - Performance tuning
9. **Query Handling** - Query normalization
10. **Query Embedding** - Query vectorization
11. **Similarity Search** - Vector similarity
12. **Content Filtering** - Quality filtering
13. **Re-ranking** - MMR diversity optimization
14. **Prompt Construction** - Context building
15. **LLM Completion** - Response generation

### Production Features
- **Load Balancing** - Nginx reverse proxy
- **Rate Limiting** - 10 requests/second per IP
- **Health Checks** - Automated monitoring
- **Logging** - Structured logging with levels
- **Metrics** - Prometheus integration
- **Security** - Headers and authentication ready
- **Scalability** - Horizontal scaling support

## 📊 Monitoring

### Grafana Dashboard
Access: http://localhost:3000
- Username: admin
- Password: admin123

### Key Metrics
- Request rate and latency
- Pipeline processing time
- Source ingestion performance
- Memory and CPU usage
- Error rates and types

### Prometheus Metrics
Access: http://localhost:9090

## 🔒 Security

### Implemented
- Rate limiting (10 req/s)
- Security headers
- Input validation
- Error handling
- CORS configuration

### Recommended
- SSL/TLS termination
- API authentication
- IP whitelisting
- Request signing

## 🚀 Scaling

### Horizontal Scaling
```bash
# Scale to 3 instances
docker-compose -f production/docker-compose.yml up -d --scale rag-pipeline=3
```

### Performance Tuning
- Adjust chunk sizes for content type
- Optimize top-k values for latency
- Tune MMR lambda for diversity
- Use Redis caching for repeated queries

## 🛠️ Management

### View Logs
```bash
docker-compose -f production/docker-compose.yml logs -f rag-pipeline
```

### Restart Services
```bash
docker-compose -f production/docker-compose.yml restart rag-pipeline
```

### Update Configuration
```bash
# Edit .env file
docker-compose -f production/docker-compose.yml up -d
```

### Backup Data
```bash
docker-compose -f production/docker-compose.yml exec rag-pipeline tar czf /backup/data.tar.gz /app/data
```

## 🔍 Troubleshooting

### Common Issues

#### API Not Responding
```bash
# Check service status
docker-compose -f production/docker-compose.yml ps

# View logs
docker-compose -f production/docker-compose.yml logs rag-pipeline

# Restart service
docker-compose -f production/docker-compose.yml restart rag-pipeline
```

#### Memory Issues
- Reduce chunk size
- Limit concurrent requests
- Increase available RAM

#### Slow Performance
- Optimize top-k values
- Enable Redis caching
- Use GPU for embeddings

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Service status
docker-compose -f production/docker-compose.yml exec rag-pipeline curl localhost:8000/health

# Resource usage
docker stats
```

## 📈 Performance

### Benchmarks
- **Query Latency**: 0.5-2 seconds (depends on content)
- **Throughput**: 10+ queries/second
- **Memory Usage**: 2-4GB (depends on content size)
- **Indexing Time**: 30-60 seconds per 1000 chunks

### Optimization Tips
1. **Chunk Size**: 400-800 chars optimal
2. **Top-K**: 5-10 balances quality/speed
3. **MMR**: Enable for diverse results
4. **Caching**: Enable Redis for repeated queries
5. **GPU**: Use for large-scale deployments

## 🔄 Updates

### Update Pipeline
```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose -f production/docker-compose.yml build
docker-compose -f production/docker-compose.yml up -d
```

### Add New Sources
```bash
curl -X POST http://localhost:8000/sources \
  -H "Content-Type: application/json" \
  -d '{"sources": ["new_document.pdf"]}'
```

## 📞 Support

### Client Examples
See `client_example.py` for production client usage.

### API Documentation
Interactive docs available at: http://localhost:8000/docs

### Monitoring
Check Grafana dashboard for system health and performance metrics.
