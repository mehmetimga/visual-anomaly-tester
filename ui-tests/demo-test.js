/**
 * Simple demo test to show visual testing capabilities
 * This runs without the full ML stack to demonstrate the concept
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

// Colors for output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

function colorLog(color, message) {
  console.log(colors[color] + message + colors.reset);
}

async function runSimpleVisualTest() {
  colorLog('cyan', '📸 Simple Visual Test Demo');
  colorLog('cyan', '==========================\n');

  let browser, page;
  
  try {
    // Launch browser
    colorLog('blue', '🚀 Launching browser...');
    browser = await chromium.launch({ 
      headless: true,
      args: ['--no-sandbox', '--disable-dev-shm-usage']
    });
    
    const context = await browser.newContext({
      viewport: { width: 1280, height: 720 }
    });
    
    page = await context.newPage();
    
    // Navigate to demo app
    colorLog('blue', '📍 Navigating to demo app...');
    await page.goto('http://localhost:3001', { 
      waitUntil: 'networkidle',
      timeout: 10000 
    });

    // Wait for page to stabilize
    colorLog('blue', '⏳ Stabilizing page...');
    
    // Disable animations for consistent testing
    await page.addStyleTag({
      content: `
        *, *::before, *::after {
          animation-duration: 0.01ms !important;
          animation-delay: -0.01ms !important;
          animation-iteration-count: 1 !important;
          background-attachment: initial !important;
          scroll-behavior: auto !important;
          transition-duration: 0s !important;
          transition-delay: 0s !important;
        }
      `
    });

    // Wait for content to load
    await page.waitForSelector('main', { timeout: 5000 });
    await page.waitForTimeout(1000); // Additional stability wait

    // Demonstrate masking capabilities
    colorLog('blue', '🎭 Demonstrating intelligent masking...');
    
    const dynamicElements = await page.evaluate(() => {
      const elements = [];
      
      // Find timestamp elements
      const timestamps = document.querySelectorAll('[data-testid="current-time"], .timestamp');
      timestamps.forEach(el => {
        const rect = el.getBoundingClientRect();
        elements.push({
          type: 'timestamp',
          selector: el.getAttribute('data-testid') || el.className,
          bounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height }
        });
      });
      
      // Find counters
      const counters = document.querySelectorAll('[data-testid*="count"], .metric');
      counters.forEach(el => {
        const rect = el.getBoundingClientRect();
        elements.push({
          type: 'counter',
          selector: el.getAttribute('data-testid') || el.className,
          bounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height }
        });
      });
      
      // Find user avatars
      const avatars = document.querySelectorAll('.user-avatar');
      avatars.forEach(el => {
        const rect = el.getBoundingClientRect();
        elements.push({
          type: 'avatar',
          selector: el.className,
          bounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height }
        });
      });
      
      return elements;
    });

    console.log(`   Found ${dynamicElements.length} dynamic elements to mask:`);
    dynamicElements.forEach(el => {
      console.log(`   • ${el.type}: ${el.selector} (${Math.round(el.bounds.width)}x${Math.round(el.bounds.height)})`);
    });

    // Take screenshot
    colorLog('blue', '📸 Capturing screenshot...');
    
    const screenshotDir = path.join(__dirname, 'demo-results');
    if (!fs.existsSync(screenshotDir)) {
      fs.mkdirSync(screenshotDir, { recursive: true });
    }

    const screenshot = await page.screenshot({
      path: path.join(screenshotDir, 'demo-screenshot.png'),
      fullPage: true
    });

    colorLog('green', '✅ Screenshot captured');

    // Simulate basic comparison (without ML service)
    colorLog('blue', '🔍 Simulating visual analysis...');
    
    // Mock analysis results
    const mockAnalysisResult = {
      metrics: {
        ssim: 0.95,
        pixel_diff_ratio: 0.02,
        anomaly_score: 0.15,
        is_anomaly: false,
        confidence: 0.87
      },
      analysis: {
        ui_components: [
          { class_name: 'button', confidence: 0.92, bbox: [100, 200, 150, 40] },
          { class_name: 'card', confidence: 0.89, bbox: [50, 100, 300, 200] }
        ],
        ocr_results: [
          { text: 'Visual Testing Demo', confidence: 0.95 },
          { text: 'Dashboard Metrics', confidence: 0.91 }
        ]
      },
      triage: {
        severity: 1,
        pass_ci: true,
        issues: [],
        recommendations: [
          'Consider masking timestamp elements for consistent testing',
          'Monitor dynamic counter values in metrics cards'
        ],
        summary: 'No critical visual anomalies detected'
      }
    };

    // Display results
    colorLog('green', '📊 Analysis Results:');
    console.log('   📈 Metrics:');
    console.log(`     • SSIM Score: ${mockAnalysisResult.metrics.ssim}`);
    console.log(`     • Pixel Diff: ${(mockAnalysisResult.metrics.pixel_diff_ratio * 100).toFixed(1)}%`);
    console.log(`     • Anomaly Score: ${mockAnalysisResult.metrics.anomaly_score}`);
    console.log(`     • Confidence: ${(mockAnalysisResult.metrics.confidence * 100).toFixed(0)}%`);
    
    console.log('   🔍 Detection:');
    console.log(`     • UI Components: ${mockAnalysisResult.analysis.ui_components.length} detected`);
    console.log(`     • Text Elements: ${mockAnalysisResult.analysis.ocr_results.length} detected`);
    
    console.log('   🎯 Triage:');
    console.log(`     • Severity: ${mockAnalysisResult.triage.severity}/3`);
    console.log(`     • CI Status: ${mockAnalysisResult.triage.pass_ci ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`     • Summary: ${mockAnalysisResult.triage.summary}`);

    if (mockAnalysisResult.triage.recommendations.length > 0) {
      console.log('   💡 Recommendations:');
      mockAnalysisResult.triage.recommendations.forEach(rec => {
        console.log(`     • ${rec}`);
      });
    }

    // Performance metrics
    const performanceMetrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0];
      return {
        domContentLoaded: Math.round(navigation.domContentLoadedEventEnd - navigation.navigationStart),
        loadComplete: Math.round(navigation.loadEventEnd - navigation.navigationStart),
        resources: performance.getEntriesByType('resource').length
      };
    });

    colorLog('blue', '⚡ Performance Impact:');
    console.log(`   • Page Load: ${performanceMetrics.loadComplete}ms`);
    console.log(`   • DOM Ready: ${performanceMetrics.domContentLoaded}ms`);
    console.log(`   • Resources: ${performanceMetrics.resources} loaded`);
    console.log(`   • Visual Test Overhead: ~200-500ms (estimated)`);

    colorLog('green', '\n🎉 Demo test completed successfully!');
    console.log(`   📁 Screenshot saved to: ${path.join(screenshotDir, 'demo-screenshot.png')}`);

  } catch (error) {
    colorLog('red', `❌ Test failed: ${error.message}`);
    throw error;
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

async function showFullSystemCapabilities() {
  colorLog('cyan', '\n🚀 Full System Capabilities');
  colorLog('cyan', '===========================\n');

  console.log('🧠 Advanced ML Analysis:');
  console.log('   • SSIM (Structural Similarity): Measures structural changes');
  console.log('   • LPIPS (Perceptual Distance): Deep learning perceptual similarity');
  console.log('   • ViT Distance: Vision Transformer semantic understanding');  
  console.log('   • CLIP Embeddings: 512-dimensional semantic vectors');
  console.log('   • YOLO Object Detection: UI component recognition');
  console.log('   • SAM Segmentation: Precise diff region identification');
  console.log('   • OCR Text Analysis: Content change detection');
  console.log('   • Texture Features: LBP, Gabor filters for micro-changes');
  console.log('');

  console.log('🎯 Intelligent Automation:');
  console.log('   • Auto-masking: 15+ patterns for dynamic content');
  console.log('   • Smart Triage: Rule-based severity assessment');
  console.log('   • Vector Search: Find similar historical failures');
  console.log('   • Active Learning: Query most informative samples');
  console.log('   • Performance Monitoring: Built-in overhead measurement');
  console.log('');

  console.log('🏗️  Production Features:');
  console.log('   • Ensemble Models: XGBoost, LightGBM, CatBoost');
  console.log('   • Hyperparameter Tuning: Optuna optimization');
  console.log('   • Experiment Tracking: MLflow integration');
  console.log('   • Data Validation: Quality checks and monitoring');
  console.log('   • Enterprise Security: Rate limiting, authentication');
  console.log('');

  colorLog('green', '🚀 To experience the full system:');
  console.log('   1. Start Docker Desktop');
  console.log('   2. Run: docker compose up --build -d');
  console.log('   3. Wait for all services to be healthy');
  console.log('   4. Run: cd ui-tests && npm test');
}

// Run the demo
if (require.main === module) {
  runSimpleVisualTest()
    .then(() => showFullSystemCapabilities())
    .catch(error => {
      console.error('Demo failed:', error);
      process.exit(1);
    });
}

module.exports = { runSimpleVisualTest };