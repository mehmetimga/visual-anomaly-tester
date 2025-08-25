#!/bin/bash

# Visual Anomaly Testing Stack Setup Script
# This script sets up and runs the visual testing system locally

set -e

echo "🎯 Visual Anomaly Testing Stack Setup"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
echo
print_info "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker Desktop."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker Desktop."
    echo
    print_info "To start Docker:"
    echo "  - On macOS: Open Docker Desktop app"
    echo "  - On Linux: sudo systemctl start docker"
    echo "  - On Windows: Start Docker Desktop"
    exit 1
fi

print_status "Docker is installed and running"

# Check Docker Compose
if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not available"
    exit 1
fi

print_status "Docker Compose is available"

# Check Node.js (for UI tests)
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_status "Node.js is installed ($NODE_VERSION)"
else
    print_warning "Node.js not found. You'll need it to run UI tests."
    echo "  Install from: https://nodejs.org/"
fi

echo
print_info "Starting services..."

# Create necessary directories
mkdir -p data/{baselines,runs,approvals}
mkdir -p models
mkdir -p training_data
mkdir -p mlflow_data
mkdir -p qdrant_storage
mkdir -p logs

print_status "Created data directories"

# Start services
echo
print_info "Building and starting Docker services..."

if ! docker compose up --build -d; then
    print_error "Failed to start services"
    exit 1
fi

print_status "Services started"

# Wait for services to be ready
echo
print_info "Waiting for services to be ready..."

# Function to wait for service
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    echo -n "Waiting for $name"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo
            print_status "$name is ready"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo
    print_error "$name failed to start"
    return 1
}

# Wait for each service
wait_for_service "http://localhost:8080/health" "API Service"
wait_for_service "http://localhost:8000/health" "ML Service" 
wait_for_service "http://localhost:6333/health" "Qdrant Vector DB"
wait_for_service "http://localhost:5000" "MLflow Tracking"

echo
print_status "All services are running!"

echo
print_info "Service URLs:"
echo "  🌐 API Service:     http://localhost:8080"
echo "  🧠 ML Service:      http://localhost:8000"
echo "  🔍 Qdrant DB:       http://localhost:6333"
echo "  📊 MLflow:          http://localhost:5000"

echo
print_info "Health check results:"

# Check API health
if API_HEALTH=$(curl -s http://localhost:8080/health 2>/dev/null); then
    echo "  ✅ API Service: $(echo $API_HEALTH | jq -r '.status // "healthy"' 2>/dev/null || echo "healthy")"
else
    echo "  ❌ API Service: Not responding"
fi

# Check ML health
if ML_HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null); then
    echo "  ✅ ML Service: $(echo $ML_HEALTH | jq -r '.status // "healthy"' 2>/dev/null || echo "healthy")"
else
    echo "  ❌ ML Service: Not responding"
fi

# Check Qdrant health
if curl -s -f http://localhost:6333/health > /dev/null 2>&1; then
    echo "  ✅ Qdrant: healthy"
else
    echo "  ❌ Qdrant: Not responding"
fi

echo
print_info "Next steps:"
echo "  1. Install UI test dependencies:"
echo "     cd ui-tests && npm install && npx playwright install --with-deps"
echo
echo "  2. Set your target application URL:"
echo "     export APP_URL=http://localhost:3000  # Replace with your app URL"
echo
echo "  3. Run visual tests:"
echo "     cd ui-tests && npm test"
echo
echo "  4. View test artifacts:"
echo "     ls -la data/runs/"
echo
echo "  5. Access web interfaces:"
echo "     - MLflow UI: http://localhost:5000"
echo "     - Qdrant Dashboard: http://localhost:6333/dashboard"

print_status "Setup complete! 🎉"