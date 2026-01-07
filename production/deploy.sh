#!/bin/bash

# Production Deployment Script for 15-Step RAG Pipeline
set -e

echo "🚀 Deploying 15-Step RAG Pipeline to Production..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs ssl

# Copy environment file if not exists
if [ ! -f .env ]; then
    echo "📝 Creating environment file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration before continuing."
    echo "Press Enter to continue or Ctrl+C to exit..."
    read
fi

# Build and start services
echo "🔨 Building Docker images..."
docker-compose -f production/docker-compose.yml build

echo "🚀 Starting services..."
docker-compose -f production/docker-compose.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🔍 Checking service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ RAG Pipeline API is healthy!"
else
    echo "❌ RAG Pipeline API is not responding"
    docker-compose -f production/docker-compose.yml logs rag-pipeline
    exit 1
fi

# Show status
echo "📊 Service Status:"
docker-compose -f production/docker-compose.yml ps

echo ""
echo "🎉 Deployment Complete!"
echo ""
echo "🌐 Available Services:"
echo "  - RAG Pipeline API: http://localhost:8000"
echo "  - API Documentation: http://localhost:8000/docs"
echo "  - Health Check: http://localhost:8000/health"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000 (admin/admin123)"
echo ""
echo "📝 Quick Test:"
echo "  curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"query\":\"What is this about?\"}'"
echo ""
echo "🔧 Management Commands:"
echo "  - View logs: docker-compose -f production/docker-compose.yml logs -f rag-pipeline"
echo "  - Stop services: docker-compose -f production/docker-compose.yml down"
echo "  - Restart: docker-compose -f production/docker-compose.yml restart"
