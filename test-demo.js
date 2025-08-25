#!/usr/bin/env node

/**
 * Visual Anomaly Testing Demo Script
 * Demonstrates the system capabilities without requiring full Docker setup
 */

const http = require('http');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

// Colors for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

function colorLog(color, message) {
  console.log(colors[color] + message + colors.reset);
}

async function checkServiceHealth(url, name) {
  return new Promise((resolve) => {
    const request = http.get(url, (res) => {
      resolve({ name, status: res.statusCode === 200 ? 'healthy' : 'unhealthy', available: true });
    });
    
    request.on('error', () => {
      resolve({ name, status: 'unavailable', available: false });
    });
    
    request.setTimeout(5000, () => {
      request.destroy();
      resolve({ name, status: 'timeout', available: false });
    });
  });
}

async function startDemoServer() {
  return new Promise((resolve) => {
    const demoPath = path.join(__dirname, 'demo-app', 'index.html');
    
    if (!fs.existsSync(demoPath)) {
      colorLog('red', '❌ Demo app not found');
      resolve(null);
      return;
    }

    const server = http.createServer((req, res) => {
      if (req.url === '/' || req.url === '/index.html') {
        fs.readFile(demoPath, 'utf8', (err, content) => {
          if (err) {
            res.writeHead(500);
            res.end('Error loading demo app');
            return;
          }
          res.writeHead(200, { 'Content-Type': 'text/html' });
          res.end(content);
        });
      } else {
        res.writeHead(404);
        res.end('Not found');
      }
    });

    server.listen(3001, () => {
      colorLog('green', '✅ Demo app started at http://localhost:3001');
      resolve(server);
    });
  });
}

async function demonstrateSystem() {
  colorLog('cyan', '🎯 Visual Anomaly Testing System Demo');
  colorLog('cyan', '=====================================\n');

  // Check if Docker is available
  colorLog('blue', '🐳 Checking Docker status...');
  
  try {
    const dockerCheck = spawn('docker', ['--version'], { stdio: 'pipe' });
    dockerCheck.on('close', async (code) => {
      if (code === 0) {
        colorLog('green', '✅ Docker is available');
        
        // Check if services are running
        colorLog('blue', '\n🔍 Checking visual testing services...');
        
        const services = [
          { url: 'http://localhost:8080/health', name: 'API Service' },
          { url: 'http://localhost:8000/health', name: 'ML Service' },
          { url: 'http://localhost:6333/health', name: 'Qdrant Vector DB' },
          { url: 'http://localhost:5000', name: 'MLflow Tracking' }
        ];

        const healthChecks = await Promise.all(
          services.map(service => checkServiceHealth(service.url, service.name))
        );

        healthChecks.forEach(check => {
          if (check.available) {
            colorLog('green', `✅ ${check.name}: ${check.status}`);
          } else {
            colorLog('yellow', `⚠️  ${check.name}: ${check.status}`);
          }
        });

        const allHealthy = healthChecks.every(check => check.status === 'healthy');
        
        if (allHealthy) {
          colorLog('green', '\n🎉 All services are running! Ready for visual testing.');
          await runFullDemo();
        } else {
          colorLog('yellow', '\n⚠️  Some services are not running. Starting demo mode...');
          await runDemoMode();
        }
      } else {
        colorLog('yellow', '⚠️  Docker not available. Running demo mode...');
        await runDemoMode();
      }
    });

    dockerCheck.on('error', async () => {
      colorLog('yellow', '⚠️  Docker not available. Running demo mode...');
      await runDemoMode();
    });
    
  } catch (error) {
    colorLog('yellow', '⚠️  Docker not available. Running demo mode...');
    await runDemoMode();
  }
}

async function runFullDemo() {
  colorLog('blue', '\n🚀 Running full visual testing demo...');
  
  // Start demo app
  const server = await startDemoServer();
  
  colorLog('blue', '\n📋 System Overview:');
  console.log('   🏗️  Architecture: Microservices with Docker');
  console.log('   🧠 ML Pipeline: Computer Vision + Deep Learning');
  console.log('   🔍 Vector Search: Qdrant for similarity matching');
  console.log('   📊 Tracking: MLflow for experiment management');
  console.log('   🎭 Testing: Playwright with intelligent masking');
  
  colorLog('blue', '\n🔧 To run visual tests:');
  console.log('   1. cd ui-tests');
  console.log('   2. npm install');
  console.log('   3. npx playwright install --with-deps');
  console.log('   4. APP_URL=http://localhost:3001 npm test');
  
  colorLog('blue', '\n📈 Web Interfaces:');
  console.log('   • Demo App:      http://localhost:3001');
  console.log('   • API Health:    http://localhost:8080/health');
  console.log('   • ML Service:    http://localhost:8000/health');
  console.log('   • Qdrant UI:     http://localhost:6333/dashboard');
  console.log('   • MLflow UI:     http://localhost:5000');

  // Wait a bit then cleanup
  setTimeout(() => {
    if (server) {
      server.close();
      colorLog('blue', '\n👋 Demo complete. Demo server stopped.');
    }
  }, 5000);
}

async function runDemoMode() {
  colorLog('blue', '\n🎮 Demo Mode: System Overview');
  
  // Start demo app
  const server = await startDemoServer();
  
  colorLog('blue', '\n📋 Visual Anomaly Testing Stack Features:');
  console.log('');
  console.log('🧠 Advanced ML Pipeline:');
  console.log('   • Multi-metric analysis (SSIM, LPIPS, ViT distance)');
  console.log('   • Object detection with YOLO');
  console.log('   • Semantic similarity with CLIP embeddings');
  console.log('   • OCR text analysis');
  console.log('   • SAM segmentation for precise diff regions');
  console.log('');
  
  console.log('🎯 Intelligent Features:');
  console.log('   • Auto-detection of dynamic content');
  console.log('   • Smart masking (timestamps, user avatars, counters)');
  console.log('   • Rule-based triage with severity levels');
  console.log('   • Vector similarity search for historical context');
  console.log('   • Performance impact measurement');
  console.log('');
  
  console.log('🏗️  Production Architecture:');
  console.log('   • Docker containerized services');
  console.log('   • Qdrant vector database');
  console.log('   • MLflow experiment tracking');
  console.log('   • AutoML training pipeline');
  console.log('   • Enterprise security and monitoring');
  console.log('');
  
  colorLog('green', '🚀 To start the full system:');
  console.log('   1. Make sure Docker Desktop is running');
  console.log('   2. Run: ./setup.sh');
  console.log('   3. Or manually: docker compose up --build -d');
  console.log('');
  
  colorLog('blue', '📖 Key Files:');
  console.log('   • docker-compose.yml     - Service orchestration');
  console.log('   • services/api/index.js  - API with intelligent triage');
  console.log('   • services/ml/main.py    - ML service with 30+ features');
  console.log('   • ui-tests/tests/        - Playwright visual tests');
  console.log('   • setup.sh               - Automated setup script');

  if (server) {
    colorLog('cyan', '\n🌐 Demo app is running at: http://localhost:3001');
    colorLog('cyan', '   Open it in your browser to see the test target!');
    
    // Keep demo running
    colorLog('blue', '\nPress Ctrl+C to stop the demo server...');
    
    process.on('SIGINT', () => {
      server.close();
      colorLog('blue', '\n👋 Demo server stopped. Goodbye!');
      process.exit(0);
    });
  }
}

// Check if script should show help
if (process.argv.includes('--help') || process.argv.includes('-h')) {
  colorLog('cyan', '🎯 Visual Anomaly Testing Demo');
  colorLog('cyan', '==============================\n');
  console.log('Usage: node test-demo.js [options]\n');
  console.log('Options:');
  console.log('  --help, -h    Show this help message');
  console.log('');
  console.log('This script demonstrates the visual testing system capabilities.');
  console.log('It checks for running services and provides setup guidance.');
  process.exit(0);
}

// Run the demo
demonstrateSystem().catch(error => {
  colorLog('red', '❌ Demo failed: ' + error.message);
  process.exit(1);
});