/**
 * Advanced Visual Testing Helper
 * Provides intelligent visual comparison with ML-powered analysis
 */

import { Page } from '@playwright/test';
import axios, { AxiosResponse } from 'axios';
import sharp from 'sharp';
import { createLogger, format, transports } from 'winston';
import FormData from 'form-data';

export interface TestConfig {
  apiUrl: string;
  appUrl: string;
  projectId: string;
  branch: string;
  commitSha: string;
  thresholds: {
    anomaly: number;
    confidence: number;
  };
  options: {
    enableSamSegmentation: boolean;
    enableTextureAnalysis: boolean;
    enableSimilaritySearch: boolean;
    maxSimilarResults: number;
  };
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface VisualComparisonResult {
  status: string;
  metrics: {
    anomaly_score: number;
    is_anomaly: boolean;
    confidence: number;
    ssim: number;
    lpips: number;
    vit_cosine_distance: number;
    pixel_diff_ratio: number;
    [key: string]: any;
  };
  analysis: {
    ui_components: Array<{
      bbox: number[];
      confidence: number;
      class_name: string;
      type: string;
    }>;
    ocr_results: Array<{
      bbox: number[][];
      text: string;
      confidence: number;
    }>;
    feature_count: number;
  };
  triage: {
    severity: number;
    pass_ci: boolean;
    issues: Array<{
      severity: string;
      rule: string;
      message: string;
      level: number;
    }>;
    recommendations: Array<{
      type: string;
      priority: string;
      suggestion: string;
      [key: string]: any;
    }>;
    summary: string;
  };
  embeddings: {
    clip_512: number[];
  };
  files: {
    baseline: string;
    candidate: string;
    heatmap: string;
  };
  similarCases?: SimilarityResult[];
  metadata: {
    run_id: string;
    test_id: string;
    name: string;
    timestamp: string;
  };
}

export interface SimilarityResult {
  id: string;
  score: number;
  payload: {
    test_id: string;
    name: string;
    timestamp: string;
    metrics: any;
    run_id: string;
  };
}

export interface TestSummary {
  runId: string;
  projectId: string;
  branch: string;
  timestamp: string;
  totalTests: number;
  passedTests: number;
  failedTests: number;
  anomaliesDetected: number;
  averageAnomalyScore: number;
  criticalIssues: number;
  recommendations: string[];
}

export class VisualTestHelper {
  private logger = createLogger({
    level: 'info',
    format: format.combine(
      format.timestamp(),
      format.errors({ stack: true }),
      format.json()
    ),
    transports: [
      new transports.Console({
        format: format.simple()
      }),
      new transports.File({ filename: 'visual-tests.log' })
    ]
  });

  private currentRunId: string | null = null;
  private testResults: VisualComparisonResult[] = [];
  private httpClient = axios.create({
    timeout: 300000, // 5 minutes
    headers: {
      'User-Agent': 'Visual-Anomaly-Tester/2.0'
    }
  });

  constructor(private config: TestConfig) {
    this.logger.info('Visual Test Helper initialized', {
      apiUrl: config.apiUrl,
      projectId: config.projectId,
      branch: config.branch
    });
  }

  async initializeRun(): Promise<string> {
    try {
      const response = await this.httpClient.post(`${this.config.apiUrl}/runs`, {
        project_id: this.config.projectId,
        branch: this.config.branch,
        commit_sha: this.config.commitSha,
        metadata: {
          user_agent: 'Playwright Visual Tests',
          test_framework: 'playwright',
          environment: process.env.NODE_ENV || 'test',
          ci: !!process.env.CI
        }
      });

      this.currentRunId = response.data.runId;
      this.logger.info('Test run initialized', { runId: this.currentRunId });
      
      return this.currentRunId;
    } catch (error) {
      this.logger.error('Failed to initialize test run', { error: error.message });
      throw new Error(`Failed to initialize test run: ${error.message}`);
    }
  }

  getCurrentRunId(): string | null {
    return this.currentRunId;
  }

  async captureAndAnalyze(
    page: Page,
    testName: string,
    options: {
      masks?: BoundingBox[];
      customThresholds?: { anomaly?: number; confidence?: number };
      skipSimilarity?: boolean;
      fullPage?: boolean;
      clip?: { x: number; y: number; width: number; height: number };
    } = {}
  ): Promise<VisualComparisonResult> {
    if (!this.currentRunId) {
      throw new Error('Test run not initialized. Call initializeRun() first.');
    }

    const {
      masks = [],
      customThresholds = {},
      skipSimilarity = false,
      fullPage = true,
      clip
    } = options;

    try {
      // Capture screenshot with optimization
      const screenshotOptions: any = {
        fullPage,
        type: 'png',
        ...(clip && { clip })
      };

      let screenshot = await page.screenshot(screenshotOptions);
      
      // Optimize screenshot for faster processing
      screenshot = await sharp(screenshot)
        .png({ quality: 90, progressive: true })
        .toBuffer();

      // Prepare form data
      const formData = new FormData();
      formData.append('runId', this.currentRunId);
      formData.append('testId', `${this.config.projectId}@${testName}`);
      formData.append('name', testName);
      formData.append('masks', JSON.stringify(masks));
      formData.append('image', screenshot, {
        filename: `${testName}.png`,
        contentType: 'image/png'
      });

      // Add viewport information
      const viewport = page.viewportSize();
      if (viewport) {
        formData.append('viewport', JSON.stringify(viewport));
      }

      // Add analysis options
      const analysisOptions = {
        use_sam_segmentation: this.config.options.enableSamSegmentation,
        enable_texture_analysis: this.config.options.enableTextureAnalysis,
        anomaly_threshold: customThresholds.anomaly || this.config.thresholds.anomaly,
        confidence_threshold: customThresholds.confidence || this.config.thresholds.confidence
      };
      formData.append('options', JSON.stringify(analysisOptions));

      this.logger.info('Capturing visual snapshot', {
        testName,
        runId: this.currentRunId,
        masksCount: masks.length,
        screenshotSize: screenshot.length
      });

      // Submit to visual analysis API
      const response = await this.httpClient.post(
        `${this.config.apiUrl}/snapshots`,
        formData,
        {
          headers: {
            ...formData.getHeaders(),
            'Content-Length': formData.getLengthSync()
          }
        }
      );

      const result: VisualComparisonResult = response.data;

      // Fetch similar cases if enabled and anomaly detected
      if (!skipSimilarity && 
          this.config.options.enableSimilaritySearch && 
          result.embeddings?.clip_512 && 
          result.metrics.is_anomaly) {
        
        try {
          const similarityResponse = await this.httpClient.post(
            `${this.config.apiUrl}/similar`,
            {
              vector: result.embeddings.clip_512,
              limit: this.config.options.maxSimilarResults,
              filter: {
                must_not: [
                  {
                    match: {
                      run_id: this.currentRunId
                    }
                  }
                ]
              }
            }
          );

          result.similarCases = similarityResponse.data.results;
          
          this.logger.info('Similar cases found', {
            testName,
            similarCount: result.similarCases?.length || 0
          });
        } catch (similarityError) {
          this.logger.warn('Failed to fetch similar cases', {
            testName,
            error: similarityError.message
          });
        }
      }

      // Store result for summary
      this.testResults.push(result);

      // Enhanced logging
      this.logger.info('Visual analysis completed', {
        testName,
        status: result.status,
        isAnomaly: result.metrics.is_anomaly,
        anomalyScore: result.metrics.anomaly_score?.toFixed(4),
        confidence: result.metrics.confidence?.toFixed(3),
        passCi: result.triage?.pass_ci,
        criticalIssues: result.triage?.issues?.filter(i => i.severity === 'critical').length || 0
      });

      return result;
    } catch (error) {
      this.logger.error('Visual analysis failed', {
        testName,
        runId: this.currentRunId,
        error: error.message,
        stack: error.stack
      });
      throw new Error(`Visual analysis failed for ${testName}: ${error.message}`);
    }
  }

  async approveBaseline(
    testId: string,
    name: string,
    approvalInfo: {
      approved_by?: string;
      notes?: string;
    } = {}
  ): Promise<any> {
    if (!this.currentRunId) {
      throw new Error('Test run not initialized');
    }

    try {
      const response = await this.httpClient.post(`${this.config.apiUrl}/approve`, {
        testId: `${this.config.projectId}@${testId}`,
        name,
        runId: this.currentRunId,
        ...approvalInfo
      });

      this.logger.info('Baseline approved', {
        testId,
        name,
        runId: this.currentRunId,
        approvedBy: approvalInfo.approved_by
      });

      return response.data;
    } catch (error) {
      this.logger.error('Failed to approve baseline', {
        testId,
        name,
        error: error.message
      });
      throw new Error(`Failed to approve baseline: ${error.message}`);
    }
  }

  async generateTestSummary(): Promise<TestSummary> {
    if (!this.currentRunId || this.testResults.length === 0) {
      throw new Error('No test results available for summary');
    }

    const totalTests = this.testResults.length;
    const anomaliesDetected = this.testResults.filter(r => r.metrics.is_anomaly).length;
    const passedTests = this.testResults.filter(r => r.triage?.pass_ci !== false).length;
    const failedTests = totalTests - passedTests;
    
    const anomalyScores = this.testResults
      .map(r => r.metrics.anomaly_score || 0)
      .filter(score => score > 0);
    const averageAnomalyScore = anomalyScores.length > 0 
      ? anomalyScores.reduce((a, b) => a + b, 0) / anomalyScores.length 
      : 0;

    const criticalIssues = this.testResults
      .flatMap(r => r.triage?.issues || [])
      .filter(issue => issue.severity === 'critical').length;

    // Collect unique recommendations
    const allRecommendations = this.testResults
      .flatMap(r => r.triage?.recommendations || [])
      .map(rec => rec.suggestion);
    const recommendations = [...new Set(allRecommendations)];

    const summary: TestSummary = {
      runId: this.currentRunId,
      projectId: this.config.projectId,
      branch: this.config.branch,
      timestamp: new Date().toISOString(),
      totalTests,
      passedTests,
      failedTests,
      anomaliesDetected,
      averageAnomalyScore,
      criticalIssues,
      recommendations
    };

    // Log summary
    this.logger.info('Test run summary generated', summary);

    // Print console summary
    console.log('\n📊 Visual Testing Summary');
    console.log('========================');
    console.log(`Run ID: ${summary.runId}`);
    console.log(`Project: ${summary.projectId} (${summary.branch})`);
    console.log(`Tests: ${summary.totalTests} total, ${summary.passedTests} passed, ${summary.failedTests} failed`);
    console.log(`Anomalies: ${summary.anomaliesDetected} detected (avg score: ${summary.averageAnomalyScore.toFixed(4)})`);
    console.log(`Critical Issues: ${summary.criticalIssues}`);
    
    if (summary.recommendations.length > 0) {
      console.log('\n💡 Recommendations:');
      summary.recommendations.slice(0, 5).forEach((rec, i) => {
        console.log(`   ${i + 1}. ${rec}`);
      });
    }
    
    console.log(`\n🔗 View detailed results: ${this.config.apiUrl}/artifacts/runs/${this.currentRunId}`);
    console.log('========================\n');

    return summary;
  }

  async getArtifactUrl(filename: string): string {
    if (!this.currentRunId) {
      throw new Error('Test run not initialized');
    }
    return `${this.config.apiUrl}/artifacts/runs/${this.currentRunId}/${filename}`;
  }

  async waitForMLProcessing(maxWaitTime: number = 60000): Promise<void> {
    // Wait for any background ML processing to complete
    await new Promise(resolve => setTimeout(resolve, Math.min(maxWaitTime, 2000)));
  }

  // Utility methods for advanced testing scenarios

  async compareWithBaseline(
    currentImage: Buffer,
    baselineImage: Buffer,
    options: {
      masks?: BoundingBox[];
      threshold?: number;
    } = {}
  ): Promise<{
    isDifferent: boolean;
    similarity: number;
    diffImage: Buffer | null;
  }> {
    try {
      // Use Sharp for quick pixel-level comparison
      const currentSharp = sharp(currentImage);
      const baselineSharp = sharp(baselineImage);
      
      const currentMeta = await currentSharp.metadata();
      const baselineMeta = await baselineSharp.metadata();
      
      // Ensure same dimensions
      if (currentMeta.width !== baselineMeta.width || 
          currentMeta.height !== baselineMeta.height) {
        throw new Error('Images have different dimensions');
      }
      
      // Simple pixel difference calculation
      const currentBuffer = await currentSharp.raw().toBuffer();
      const baselineBuffer = await baselineSharp.raw().toBuffer();
      
      let differentPixels = 0;
      const totalPixels = currentBuffer.length / 3; // RGB
      
      for (let i = 0; i < currentBuffer.length; i += 3) {
        const rDiff = Math.abs(currentBuffer[i] - baselineBuffer[i]);
        const gDiff = Math.abs(currentBuffer[i + 1] - baselineBuffer[i + 1]);
        const bDiff = Math.abs(currentBuffer[i + 2] - baselineBuffer[i + 2]);
        
        if (rDiff > 10 || gDiff > 10 || bDiff > 10) {
          differentPixels++;
        }
      }
      
      const similarity = 1 - (differentPixels / totalPixels);
      const isDifferent = similarity < (options.threshold || 0.99);
      
      return {
        isDifferent,
        similarity,
        diffImage: null // Could generate diff image here
      };
    } catch (error) {
      this.logger.error('Baseline comparison failed', { error: error.message });
      throw error;
    }
  }

  async batchApproveChanges(changes: Array<{
    testId: string;
    name: string;
    notes?: string;
  }>): Promise<void> {
    this.logger.info('Starting batch approval', { count: changes.length });
    
    for (const change of changes) {
      try {
        await this.approveBaseline(change.testId, change.name, {
          approved_by: process.env.USER || 'batch-approval',
          notes: change.notes || 'Batch approval of design changes'
        });
      } catch (error) {
        this.logger.error('Failed to approve in batch', {
          testId: change.testId,
          name: change.name,
          error: error.message
        });
      }
    }
    
    this.logger.info('Batch approval completed', { count: changes.length });
  }
}