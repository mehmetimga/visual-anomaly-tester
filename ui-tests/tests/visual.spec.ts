/**
 * Advanced Visual Anomaly Testing with ML-powered analysis
 * Supports multiple comparison strategies, adaptive thresholds, and intelligent triage
 */

import { test, expect, Page, BrowserContext } from '@playwright/test';
import { VisualTestHelper, TestConfig, SimilarityResult } from './helpers/visual-helper';
import { MaskHelper } from './helpers/mask-helper';
import { PerformanceHelper } from './helpers/performance-helper';

// Test configuration
const testConfig: TestConfig = {
  apiUrl: process.env.VISUAL_API || 'http://localhost:8080',
  appUrl: process.env.APP_URL || 'http://localhost:3000',
  projectId: process.env.PROJECT_ID || 'visual-anomaly-tester',
  branch: process.env.BRANCH || 'main',
  commitSha: process.env.COMMIT_SHA || 'unknown',
  thresholds: {
    anomaly: parseFloat(process.env.ANOMALY_THRESHOLD || '0.25'),
    confidence: parseFloat(process.env.CONFIDENCE_THRESHOLD || '0.8'),
  },
  options: {
    enableSamSegmentation: process.env.ENABLE_SAM === 'true',
    enableTextureAnalysis: process.env.ENABLE_TEXTURE === 'true',
    enableSimilaritySearch: process.env.ENABLE_SIMILARITY === 'true',
    maxSimilarResults: parseInt(process.env.MAX_SIMILAR_RESULTS || '5'),
  }
};

// Global visual test helper
let visualHelper: VisualTestHelper;
let maskHelper: MaskHelper;
let performanceHelper: PerformanceHelper;

test.beforeAll(async () => {
  visualHelper = new VisualTestHelper(testConfig);
  maskHelper = new MaskHelper();
  performanceHelper = new PerformanceHelper();
  
  // Initialize test run
  await visualHelper.initializeRun();
  console.log(`📸 Visual testing initialized for run: ${visualHelper.getCurrentRunId()}`);
});

test.afterAll(async () => {
  // Generate test summary
  await visualHelper.generateTestSummary();
});

// Test utilities
async function stabilizePage(page: Page): Promise<void> {
  // Wait for page to be fully loaded
  await page.waitForLoadState('networkidle');
  
  // Disable animations and transitions
  await page.addStyleTag({
    content: `
      *, *::before, *::after {
        animation-delay: -1ms !important;
        animation-duration: 1ms !important;
        animation-iteration-count: 1 !important;
        background-attachment: initial !important;
        scroll-behavior: auto !important;
        transition-duration: 0s !important;
        transition-delay: 0s !important;
      }
    `
  });
  
  // Stabilize dynamic content
  await page.evaluate(() => {
    // Mock Date.now for consistent timestamps
    const now = new Date('2024-01-01T12:00:00Z').getTime();
    Date.now = () => now;
    
    // Stop any intervals/timeouts
    const highestId = setTimeout(() => {}, 1);
    for (let i = 0; i < highestId; i++) {
      clearTimeout(i);
      clearInterval(i);
    }
  });
  
  // Wait for any lazy-loaded content
  await page.waitForTimeout(500);
}

async function performVisualTest(
  page: Page,
  testName: string,
  options: {
    maskSelectors?: string[];
    waitForSelectors?: string[];
    customThresholds?: { anomaly?: number; confidence?: number };
    skipSimilarity?: boolean;
  } = {}
): Promise<void> {
  const { maskSelectors = [], waitForSelectors = [], customThresholds, skipSimilarity } = options;
  
  // Wait for specific elements if provided
  for (const selector of waitForSelectors) {
    await page.waitForSelector(selector, { timeout: 30000 });
  }
  
  // Stabilize the page
  await stabilizePage(page);
  
  // Compute masks
  const masks = await maskHelper.computeMasks(page, maskSelectors);
  
  // Perform visual comparison with ML analysis
  const result = await visualHelper.captureAndAnalyze(
    page,
    testName,
    {
      masks,
      customThresholds,
      skipSimilarity
    }
  );
  
  // Enhanced assertions with detailed reporting
  if (result.metrics.is_anomaly) {
    const triageInfo = result.triage;
    const criticalIssues = triageInfo.issues.filter(issue => issue.severity === 'critical');
    
    if (criticalIssues.length > 0) {
      // Log detailed failure information
      console.error(`❌ Critical visual anomalies detected in ${testName}:`);
      console.error(`   Anomaly Score: ${result.metrics.anomaly_score?.toFixed(4)}`);
      console.error(`   Critical Issues: ${criticalIssues.map(i => i.message).join(', ')}`);
      console.error(`   Recommendations: ${triageInfo.recommendations.map(r => r.suggestion).join('; ')}`);
      
      // Attach artifacts for debugging
      if (result.files?.heatmap) {
        test.info().attach('visual-heatmap', {
          path: result.files.heatmap,
          contentType: 'image/png'
        });
      }
      
      // Find and display similar past failures
      if (!skipSimilarity && result.similarCases?.length > 0) {
        console.log(`🔍 Found ${result.similarCases.length} similar cases:`);
        result.similarCases.forEach((similar: SimilarityResult, index: number) => {
          console.log(`   ${index + 1}. ${similar.test_id}/${similar.name} (similarity: ${similar.score?.toFixed(3)})`);
        });
      }
      
      expect(result.triage.pass_ci, 
        `Visual anomaly detected: ${triageInfo.summary}`
      ).toBe(true);
    }
  }
  
  // Log success with performance metrics
  console.log(`✅ ${testName} passed - Anomaly: ${result.metrics.anomaly_score?.toFixed(4)}, ` +
             `Confidence: ${result.metrics.confidence?.toFixed(3)}`);
}

// Main visual test suites

test.describe('Homepage Visual Tests', () => {
  test('homepage renders correctly @visual @critical', async ({ page }) => {
    await page.goto('/');
    
    await performVisualTest(page, 'homepage-full', {
      maskSelectors: [
        '[data-testid="current-time"]',
        '.user-avatar',
        '.notification-badge'
      ],
      waitForSelectors: [
        'main',
        'header',
        'footer'
      ]
    });
  });
  
  test('homepage hero section @visual', async ({ page }) => {
    await page.goto('/');
    
    // Focus on hero section only
    const heroSection = page.locator('[data-testid="hero-section"]');
    await expect(heroSection).toBeVisible();
    
    await performVisualTest(page, 'homepage-hero', {
      maskSelectors: [
        '.dynamic-counter',
        '.typing-animation'
      ]
    });
  });
  
  test('homepage navigation @visual', async ({ page }) => {
    await page.goto('/');
    
    const navigation = page.locator('nav[role="navigation"]');
    await expect(navigation).toBeVisible();
    
    await performVisualTest(page, 'homepage-navigation', {
      waitForSelectors: ['nav a']
    });
  });
});

test.describe('Authentication Visual Tests', () => {
  test('login form visual state @visual', async ({ page }) => {
    await page.goto('/login');
    
    await performVisualTest(page, 'login-form-default', {
      waitForSelectors: [
        'form[data-testid="login-form"]',
        'input[type="email"]',
        'input[type="password"]'
      ]
    });
  });
  
  test('login form validation states @visual', async ({ page }) => {
    await page.goto('/login');
    
    // Trigger validation by submitting empty form
    await page.click('button[type="submit"]');
    await page.waitForSelector('.error-message');
    
    await performVisualTest(page, 'login-form-validation-errors', {
      customThresholds: { anomaly: 0.15 } // More lenient for expected changes
    });
  });
});

test.describe('Dashboard Visual Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Mock login or setup authenticated state
    await page.goto('/dashboard');
    // Add authentication setup here
  });
  
  test('dashboard overview @visual @dashboard', async ({ page }) => {
    await performVisualTest(page, 'dashboard-overview', {
      maskSelectors: [
        '[data-testid="last-updated"]',
        '.real-time-chart',
        '.user-activity-feed'
      ],
      waitForSelectors: [
        '.dashboard-grid',
        '.stat-card',
        '.chart-container'
      ]
    });
  });
  
  test('dashboard responsive layout @visual @responsive', async ({ page }) => {
    // Test different viewport sizes
    const viewports = [
      { width: 1920, height: 1080, name: 'desktop-large' },
      { width: 1366, height: 768, name: 'laptop' },
      { width: 768, height: 1024, name: 'tablet' },
      { width: 375, height: 667, name: 'mobile' }
    ];
    
    for (const viewport of viewports) {
      await page.setViewportSize(viewport);
      await page.waitForTimeout(500); // Allow layout to settle
      
      await performVisualTest(page, `dashboard-${viewport.name}`, {
        maskSelectors: [
          '[data-testid="last-updated"]',
          '.responsive-chart'
        ]
      });
    }
  });
});

test.describe('Component Visual Tests', () => {
  test('button variations @visual @components', async ({ page }) => {
    await page.goto('/components/buttons');
    
    await performVisualTest(page, 'button-components', {
      waitForSelectors: [
        '.btn-primary',
        '.btn-secondary',
        '.btn-disabled'
      ]
    });
  });
  
  test('modal dialogs @visual @components', async ({ page }) => {
    await page.goto('/components/modals');
    
    // Open modal
    await page.click('[data-testid="open-modal"]');
    await page.waitForSelector('.modal[aria-hidden="false"]');
    
    await performVisualTest(page, 'modal-open-state', {
      customThresholds: { anomaly: 0.2 } // Account for overlay effects
    });
  });
  
  test('form components @visual @components', async ({ page }) => {
    await page.goto('/components/forms');
    
    await performVisualTest(page, 'form-components', {
      maskSelectors: [
        'input[type="datetime-local"]', // Dynamic default values
        '.generated-id'
      ],
      waitForSelectors: [
        'input',
        'select',
        'textarea',
        '.form-group'
      ]
    });
  });
});

test.describe('Data Visualization Tests', () => {
  test('charts and graphs @visual @charts', async ({ page }) => {
    await page.goto('/analytics');
    
    // Wait for charts to render
    await page.waitForFunction(() => {
      const charts = document.querySelectorAll('.chart-container canvas, .chart-container svg');
      return charts.length > 0 && Array.from(charts).every(chart => 
        chart.getBoundingClientRect().width > 0
      );
    });
    
    await performVisualTest(page, 'analytics-charts', {
      maskSelectors: [
        '.chart-tooltip', // Dynamic tooltips
        '.data-timestamp',
        '.loading-skeleton'
      ],
      customThresholds: { anomaly: 0.3 } // Charts can have minor pixel differences
    });
  });
});

test.describe('Error States Visual Tests', () => {
  test('404 error page @visual @error', async ({ page }) => {
    await page.goto('/non-existent-page');
    await expect(page.locator('h1')).toContainText('404');
    
    await performVisualTest(page, '404-error-page');
  });
  
  test('network error state @visual @error', async ({ page }) => {
    // Simulate network failure
    await page.route('**/api/**', route => route.abort('failed'));
    await page.goto('/dashboard');
    
    // Wait for error state to appear
    await page.waitForSelector('.error-state, .network-error');
    
    await performVisualTest(page, 'network-error-state', {
      skipSimilarity: true // Error states might be unique
    });
  });
});

test.describe('Performance Impact Tests', () => {
  test('visual testing performance impact @performance', async ({ page, context }) => {
    // Measure performance impact of visual testing
    const metrics = await performanceHelper.measurePageLoad(page, '/');
    
    await performVisualTest(page, 'homepage-performance-test');
    
    // Log performance metrics
    console.log(`⚡ Page load metrics: FCP: ${metrics.fcp}ms, LCP: ${metrics.lcp}ms`);
    
    // Ensure visual testing doesn't significantly impact performance
    expect(metrics.fcp).toBeLessThan(3000);
    expect(metrics.lcp).toBeLessThan(4000);
  });
});

test.describe('Cross-Browser Visual Consistency', () => {
  ['chromium', 'firefox', 'webkit'].forEach(browserName => {
    test(`cross-browser consistency ${browserName} @visual @cross-browser`, async ({ page }) => {
      await page.goto('/');
      
      await performVisualTest(page, `homepage-${browserName}`, {
        maskSelectors: [
          '.browser-specific-feature',
          '[data-testid="user-agent"]'
        ],
        customThresholds: { anomaly: 0.4 } // More lenient for cross-browser differences
      });
    });
  });
});

// Utilities for batch operations
test.describe('Batch Visual Updates', () => {
  test('batch approve all intentional changes @utility', async () => {
    // This test helps approve multiple visual changes at once
    // Run this manually when you've made intentional design changes
    
    const changedTests = [
      { testId: 'homepage', name: 'homepage-full' },
      { testId: 'dashboard', name: 'dashboard-overview' },
      // Add more as needed
    ];
    
    for (const testCase of changedTests) {
      const result = await visualHelper.approveBaseline(
        testCase.testId,
        testCase.name,
        {
          approved_by: process.env.USER || 'automated',
          notes: 'Batch approval of intentional design changes'
        }
      );
      
      console.log(`✅ Approved baseline for ${testCase.testId}/${testCase.name}`);
    }
  });
});