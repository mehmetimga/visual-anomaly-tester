# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Advanced Visual Anomaly Testing Stack** - a self-hosted visual regression system that rivals Percy/Applitools using computer vision, deep learning, and intelligent triage. The system combines multiple ML models, vector similarity search, and advanced image analysis for enterprise-grade visual testing.

## Architecture Overview

### Core Services
- **`services/api/`** - Node.js/Express API (orchestration, storage, triage)  
- **`services/ml/`** - Python/FastAPI ML service (computer vision, deep learning)
- **`services/training/`** - ML training pipeline (AutoML, hyperparameter optimization)
- **`ui-tests/`** - Playwright visual testing suite with AI helpers
- **External**: Qdrant (vector DB), MLflow (experiment tracking)

### Technology Stack
- **Computer Vision**: OpenCV, SSIM, LPIPS, SAM segmentation
- **Deep Learning**: PyTorch, ViT, CLIP, YOLO object detection
- **ML Pipeline**: XGBoost, LightGBM, CatBoost ensemble training
- **Optimization**: Optuna hyperparameter tuning, active learning
- **Vector Search**: Qdrant for visual similarity matching
- **Testing**: Playwright with intelligent masking helpers

## Development Commands

### Service Management
```bash
# Start all services
docker compose up --build -d

# Individual services
docker compose up api ml qdrant mlflow -d
docker compose --profile training up training  # ML training

# Development mode
cd services/api && npm run dev
cd services/ml && uvicorn main:app --reload
```

### Testing Commands
```bash
cd ui-tests

# Install dependencies
npm install
npx playwright install --with-deps

# Run visual tests
APP_URL=http://localhost:3000 VISUAL_API=http://localhost:8080 npm test

# Test variations
npm run test:headed      # With browser UI
npm run test:debug       # Debug mode  
npm run test:ci          # CI reporter format
npm run show-report      # View HTML report
```

### ML Training
```bash
# Full training pipeline
cd services/training
python train_pipeline.py --experiment-name my-experiment --max-samples 10000

# Training options
python train_pipeline.py --help
```

### Utility Commands
```bash
# Health checks
curl http://localhost:8080/health  # API health
curl http://localhost:8000/health  # ML service health

# Baseline management
curl -X POST http://localhost:8080/approve \
  -H 'Content-Type: application/json' \
  -d '{"testId":"homepage","name":"test","runId":"abc123"}'

# Data cleanup
docker compose down -v  # Reset all data
```

## Key Implementation Patterns

### Visual Test Structure
```typescript
// Standard visual test pattern
await performVisualTest(page, 'test-name', {
  maskSelectors: ['[data-testid="dynamic"]', '.timestamp'],
  waitForSelectors: ['main', '.content-loaded'],
  customThresholds: { anomaly: 0.25, confidence: 0.8 }
});
```

### ML Service Integration
```javascript
// API calling ML service
const mlResult = await mlService.scoreImages(baselineB64, candidateB64, {
  masks: parsedMasks,
  use_sam_segmentation: true,
  anomaly_threshold: 0.3
});
```

### Intelligent Masking
```typescript
// Auto-mask dynamic content
const masks = await maskHelper.computeAdaptiveMasks(page, {
  detectText: true,        // Timestamps, counters
  detectImages: true,      // User avatars  
  detectAnimations: true   // CSS animations
});
```

## File Structure & Key Files

### Configuration Files
- `docker-compose.yml` - Service orchestration
- `ui-tests/playwright.config.ts` - Test configuration
- `services/*/requirements.txt` - Python dependencies
- `services/*/package.json` - Node.js dependencies

### Core Implementation
- `services/api/index.js` - Main API with advanced triage engine
- `services/ml/main.py` - ML service with 30+ CV features  
- `services/ml/download_models.py` - Pre-trained model management
- `services/training/train_pipeline.py` - AutoML training pipeline

### Testing Helpers
- `ui-tests/tests/helpers/visual-helper.ts` - Main visual testing logic
- `ui-tests/tests/helpers/mask-helper.ts` - Intelligent masking system
- `ui-tests/tests/helpers/performance-helper.ts` - Performance monitoring

### ML Models & Features
- **Object Detection**: YOLO for UI components
- **Embeddings**: CLIP for semantic similarity  
- **Perceptual**: LPIPS, ViT distance
- **Structural**: SSIM, pixel differences
- **Texture**: LBP, Gabor filters
- **Segmentation**: SAM for precise diff regions

## Advanced Features

### Multi-Model Ensemble
The system trains and compares multiple ML models:
- XGBoost with Optuna optimization (50+ trials)
- LightGBM with early stopping
- CatBoost with auto-tuning
- Ensemble averaging for final predictions

### Intelligent Triage System
```javascript
// Custom triage rules
const customRules = [
  {
    name: 'critical_layout_shift',
    severity: 'critical',
    condition: (metrics, analysis) => 
      metrics.anomaly_score > 0.5 && metrics.ssim < 0.8,
    message: 'Major layout disruption detected'
  }
];
```

### Vector Similarity Search
- Store CLIP embeddings in Qdrant
- Find similar historical failures
- Cluster visual anomaly patterns
- Guide active learning sample selection

## Environment Variables

### Required Configuration
```bash
# Service URLs
VISUAL_API=http://localhost:8080
ML_URL=http://ml:8000  
QDRANT_URL=http://qdrant:6333
MLFLOW_URL=http://mlflow:5000

# Test Configuration
APP_URL=http://localhost:3000
PROJECT_ID=visual-anomaly-tester
BRANCH=main

# ML Thresholds
ANOMALY_THRESHOLD=0.25
CONFIDENCE_THRESHOLD=0.8

# Advanced Options
ENABLE_SAM=true              # Segment Anything Model
ENABLE_TEXTURE=true          # Texture analysis
ENABLE_SIMILARITY=true       # Vector similarity search
MAX_SIMILAR_RESULTS=5
```

## Performance Considerations

### ML Service Optimization
- Model lazy loading to reduce memory
- GPU acceleration for deep learning models
- Feature caching for repeated comparisons
- Batch processing for multiple images

### API Performance
- Rate limiting and request validation
- Image optimization with Sharp
- Streaming for large artifacts
- Background processing for ML calls

### Test Performance
- Intelligent masking reduces processing
- Performance impact measurement built-in
- Parallel test execution supported
- Screenshot optimization

## Troubleshooting Common Issues

### ML Service Issues
- Check GPU availability: `nvidia-smi`
- Verify model downloads: `ls /app/models`
- Check memory usage for large images
- Validate input image formats (PNG recommended)

### API Issues
- Check Qdrant connection: `curl http://localhost:6333/health`
- Verify MLflow tracking: `curl http://localhost:5000`
- Check disk space for artifacts in `./data/`
- Monitor API logs: `docker compose logs -f api`

### Test Issues
- Verify app accessibility: `curl $APP_URL`
- Check Playwright browser installation
- Review test timeouts for slow ML processing
- Validate mask selectors exist on page

## Extension Points

### Adding Custom Features
1. Extend `FeatureEngineer` class in training pipeline
2. Add feature extraction in ML service
3. Update API to pass new options
4. Retrain models with new features

### Custom Triage Rules
1. Modify `TriageEngine` class in API
2. Add rule validation logic
3. Test with different anomaly patterns
4. Update documentation

### Model Integration
1. Add model download in `download_models.py`
2. Implement model wrapper in ML service
3. Add hyperparameter optimization
4. Update ensemble logic

This system represents a production-ready alternative to commercial visual testing services, with advanced ML capabilities and full customization control.