/**
 * Performance Helper for Visual Testing
 * Measures the performance impact of visual testing operations
 */

import { Page, BrowserContext } from '@playwright/test';

export interface PerformanceMetrics {
  fcp: number; // First Contentful Paint
  lcp: number; // Largest Contentful Paint
  fid: number; // First Input Delay
  cls: number; // Cumulative Layout Shift
  ttfb: number; // Time to First Byte
  domContentLoaded: number;
  load: number;
  networkRequests: number;
  totalTransferSize: number;
  jsHeapUsed: number;
}

export class PerformanceHelper {
  async measurePageLoad(page: Page, url: string): Promise<PerformanceMetrics> {
    // Start performance monitoring
    await page.goto(url, { waitUntil: 'networkidle' });
    
    // Collect Web Vitals and other metrics
    const metrics = await page.evaluate(() => {
      return new Promise<PerformanceMetrics>((resolve) => {
        // Wait for all metrics to be available
        setTimeout(() => {
          const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
          
          const result: Partial<PerformanceMetrics> = {
            ttfb: navigation.responseStart - navigation.requestStart,
            domContentLoaded: navigation.domContentLoadedEventEnd - navigation.navigationStart,
            load: navigation.loadEventEnd - navigation.navigationStart,
            fcp: 0,
            lcp: 0,
            fid: 0,
            cls: 0,
            networkRequests: performance.getEntriesByType('resource').length,
            totalTransferSize: performance.getEntriesByType('resource')
              .reduce((total, resource) => total + (resource as any).transferSize || 0, 0),
            jsHeapUsed: (performance as any).memory?.usedJSHeapSize || 0
          };

          // Try to get Web Vitals if available
          if ('PerformanceObserver' in window) {
            let fcpCollected = false;
            let lcpCollected = false;
            let clsCollected = false;

            const observer = new PerformanceObserver((list) => {
              for (const entry of list.getEntries()) {
                if (entry.entryType === 'paint' && entry.name === 'first-contentful-paint') {
                  result.fcp = entry.startTime;
                  fcpCollected = true;
                } else if (entry.entryType === 'largest-contentful-paint') {
                  result.lcp = entry.startTime;
                  lcpCollected = true;
                } else if (entry.entryType === 'layout-shift' && !(entry as any).hadRecentInput) {
                  result.cls = (result.cls || 0) + (entry as any).value;
                  clsCollected = true;
                }
              }

              if (fcpCollected && lcpCollected && clsCollected) {
                observer.disconnect();
                resolve(result as PerformanceMetrics);
              }
            });

            observer.observe({ entryTypes: ['paint', 'largest-contentful-paint', 'layout-shift'] });

            // Fallback timeout
            setTimeout(() => {
              observer.disconnect();
              resolve(result as PerformanceMetrics);
            }, 5000);
          } else {
            resolve(result as PerformanceMetrics);
          }
        }, 1000);
      });
    });

    return metrics;
  }

  async measureScreenshotPerformance(page: Page, iterations: number = 3): Promise<{
    averageTime: number;
    minTime: number;
    maxTime: number;
    totalTime: number;
  }> {
    const times: number[] = [];

    for (let i = 0; i < iterations; i++) {
      const startTime = performance.now();
      
      await page.screenshot({
        fullPage: true,
        type: 'png'
      });
      
      const endTime = performance.now();
      times.push(endTime - startTime);
    }

    return {
      averageTime: times.reduce((a, b) => a + b, 0) / times.length,
      minTime: Math.min(...times),
      maxTime: Math.max(...times),
      totalTime: times.reduce((a, b) => a + b, 0)
    };
  }

  async measureMemoryUsage(context: BrowserContext): Promise<{
    usedJSHeapSize: number;
    totalJSHeapSize: number;
    jsHeapSizeLimit: number;
  }> {
    const page = await context.newPage();
    
    const memoryInfo = await page.evaluate(() => {
      if ('memory' in performance) {
        return {
          usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
          totalJSHeapSize: (performance as any).memory.totalJSHeapSize,
          jsHeapSizeLimit: (performance as any).memory.jsHeapSizeLimit
        };
      }
      return {
        usedJSHeapSize: 0,
        totalJSHeapSize: 0,
        jsHeapSizeLimit: 0
      };
    });

    await page.close();
    return memoryInfo;
  }

  async measureNetworkImpact(page: Page, operation: () => Promise<void>): Promise<{
    requestCount: number;
    totalBytes: number;
    duration: number;
  }> {
    const networkRequests: Array<{ url: string; size: number; duration: number }> = [];
    
    // Monitor network requests
    page.on('response', async (response) => {
      try {
        const request = response.request();
        const timing = response.timing();
        
        networkRequests.push({
          url: request.url(),
          size: parseInt(response.headers()['content-length'] || '0'),
          duration: timing.responseEnd
        });
      } catch (error) {
        // Ignore errors for measuring
      }
    });

    const startTime = performance.now();
    await operation();
    const endTime = performance.now();

    const totalBytes = networkRequests.reduce((sum, req) => sum + req.size, 0);

    return {
      requestCount: networkRequests.length,
      totalBytes,
      duration: endTime - startTime
    };
  }

  async monitorCPUUsage(page: Page, duration: number = 5000): Promise<number[]> {
    const cpuSamples: number[] = [];
    
    const startTime = Date.now();
    
    while (Date.now() - startTime < duration) {
      const cpuUsage = await page.evaluate(() => {
        // Approximate CPU usage by measuring script execution time
        const start = performance.now();
        let iterations = 0;
        const testEnd = start + 10; // 10ms test
        
        while (performance.now() < testEnd) {
          iterations++;
        }
        
        return iterations;
      });
      
      cpuSamples.push(cpuUsage);
      await page.waitForTimeout(100);
    }
    
    return cpuSamples;
  }

  async generatePerformanceReport(metrics: PerformanceMetrics): Promise<string> {
    const report = `
Performance Report
==================

Core Web Vitals:
- First Contentful Paint (FCP): ${metrics.fcp.toFixed(0)}ms
- Largest Contentful Paint (LCP): ${metrics.lcp.toFixed(0)}ms
- First Input Delay (FID): ${metrics.fid.toFixed(0)}ms
- Cumulative Layout Shift (CLS): ${metrics.cls.toFixed(3)}

Loading Performance:
- Time to First Byte (TTFB): ${metrics.ttfb.toFixed(0)}ms
- DOM Content Loaded: ${metrics.domContentLoaded.toFixed(0)}ms
- Page Load Complete: ${metrics.load.toFixed(0)}ms

Resource Usage:
- Network Requests: ${metrics.networkRequests}
- Total Transfer Size: ${(metrics.totalTransferSize / 1024).toFixed(2)} KB
- JS Heap Used: ${(metrics.jsHeapUsed / 1024 / 1024).toFixed(2)} MB

Performance Score:
${this.calculatePerformanceScore(metrics)}
    `.trim();

    return report;
  }

  private calculatePerformanceScore(metrics: PerformanceMetrics): string {
    let score = 100;
    
    // Deduct points based on Core Web Vitals thresholds
    if (metrics.fcp > 1800) score -= 20;
    else if (metrics.fcp > 1000) score -= 10;
    
    if (metrics.lcp > 2500) score -= 25;
    else if (metrics.lcp > 1500) score -= 15;
    
    if (metrics.cls > 0.25) score -= 25;
    else if (metrics.cls > 0.1) score -= 15;
    
    if (metrics.ttfb > 800) score -= 15;
    else if (metrics.ttfb > 400) score -= 10;

    const grade = score >= 90 ? 'A' : score >= 80 ? 'B' : score >= 70 ? 'C' : score >= 60 ? 'D' : 'F';
    
    return `Score: ${Math.max(0, score)}/100 (Grade: ${grade})`;
  }

  async benchmarkVisualTestingOverhead(
    page: Page,
    baselineOperation: () => Promise<void>,
    visualTestOperation: () => Promise<void>,
    iterations: number = 5
  ): Promise<{
    baselineAverage: number;
    visualTestAverage: number;
    overhead: number;
    overheadPercentage: number;
  }> {
    // Benchmark baseline operation
    const baselineTimes: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await baselineOperation();
      baselineTimes.push(performance.now() - start);
      await page.waitForTimeout(100); // Cool down
    }

    // Benchmark visual test operation
    const visualTestTimes: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await visualTestOperation();
      visualTestTimes.push(performance.now() - start);
      await page.waitForTimeout(100); // Cool down
    }

    const baselineAverage = baselineTimes.reduce((a, b) => a + b, 0) / baselineTimes.length;
    const visualTestAverage = visualTestTimes.reduce((a, b) => a + b, 0) / visualTestTimes.length;
    const overhead = visualTestAverage - baselineAverage;
    const overheadPercentage = (overhead / baselineAverage) * 100;

    return {
      baselineAverage,
      visualTestAverage,
      overhead,
      overheadPercentage
    };
  }

  async waitForPageStability(page: Page, options: {
    networkIdle?: number;
    maxWait?: number;
    stabilityThreshold?: number;
  } = {}): Promise<void> {
    const {
      networkIdle = 500,
      maxWait = 10000,
      stabilityThreshold = 100
    } = options;

    const startTime = Date.now();
    let lastActivityTime = Date.now();
    let isStable = false;

    // Monitor network activity
    const networkPromise = page.waitForLoadState('networkidle', { timeout: networkIdle });

    // Monitor layout stability
    const layoutStabilityPromise = page.evaluate((threshold) => {
      return new Promise<void>((resolve) => {
        let cumulativeShift = 0;
        let stableTime = Date.now();
        
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.entryType === 'layout-shift' && !(entry as any).hadRecentInput) {
              cumulativeShift += (entry as any).value;
              stableTime = Date.now();
            }
          }
        });

        observer.observe({ entryTypes: ['layout-shift'] });

        const checkStability = () => {
          if (Date.now() - stableTime > threshold) {
            observer.disconnect();
            resolve();
          } else {
            setTimeout(checkStability, 50);
          }
        };

        checkStability();
      });
    }, stabilityThreshold);

    try {
      await Promise.race([
        Promise.all([networkPromise, layoutStabilityPromise]),
        page.waitForTimeout(maxWait)
      ]);
    } catch (error) {
      // Timeout is acceptable, continue
    }
  }
}