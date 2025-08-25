/**
 * Enterprise Visual Anomaly Detection API
 * Advanced image comparison with ML integration, model versioning, and comprehensive analytics
 */

import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import multer from 'multer';
import sharp from 'sharp';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import axios from 'axios';
import mkdirp from 'mkdirp';
import { v4 as uuidv4 } from 'uuid';
import winston from 'winston';
import expressWinston from 'express-winston';
import cron from 'node-cron';
import Joi from 'joi';
import 'express-async-errors';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================================
// Configuration
// ============================================================================

const config = {
  port: process.env.PORT || 8080,
  dataDir: process.env.DATA_DIR || path.join(__dirname, 'data'),
  modelRegistry: process.env.MODEL_REGISTRY || path.join(__dirname, 'models'),
  mlUrl: process.env.ML_URL || 'http://localhost:8000',
  qdrantUrl: process.env.QDRANT_URL || 'http://localhost:6333',
  mlflowUrl: process.env.MLFLOW_URL || 'http://localhost:5000',
  maxImageSize: 10 * 1024 * 1024, // 10MB
  maxImagesPerRun: 100,
  retentionDays: 30,
  collections: {
    uiDiffs: 'ui_diffs_v2',
    models: 'model_embeddings'
  }
};

// ============================================================================
// Logging Setup
// ============================================================================

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

// ============================================================================
// Express App Setup
// ============================================================================

const app = express();

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "blob:"],
    },
  },
}));

app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || '*',
  credentials: true,
}));

app.use(compression());

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // Limit each IP to 1000 requests per windowMs
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
});
app.use(limiter);

// Request logging
app.use(expressWinston.logger({
  winstonInstance: logger,
  meta: true,
  msg: 'HTTP {{req.method}} {{req.url}}',
  expressFormat: true,
  colorize: false,
}));

// Body parsing
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// File upload configuration
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: { fileSize: config.maxImageSize },
  fileFilter: (req, file, cb) => {
    const allowedMimes = ['image/png', 'image/jpeg', 'image/webp'];
    if (allowedMimes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only PNG, JPEG, and WebP are allowed.'), false);
    }
  }
});

// ============================================================================
// Validation Schemas
// ============================================================================

const schemas = {
  runCreate: Joi.object({
    project_id: Joi.string().required(),
    branch: Joi.string().default('main'),
    commit_sha: Joi.string().optional(),
    metadata: Joi.object().optional()
  }),

  snapshot: Joi.object({
    runId: Joi.string().uuid().required(),
    testId: Joi.string().required(),
    name: Joi.string().required(),
    masks: Joi.string().default('[]'),
    viewport: Joi.object({
      width: Joi.number().integer().min(1),
      height: Joi.number().integer().min(1)
    }).optional(),
    options: Joi.object({
      use_sam_segmentation: Joi.boolean().default(false),
      enable_texture_analysis: Joi.boolean().default(true),
      anomaly_threshold: Joi.number().min(0).max(1).default(0.3),
      confidence_threshold: Joi.number().min(0).max(1).default(0.8)
    }).optional()
  }),

  approve: Joi.object({
    testId: Joi.string().required(),
    name: Joi.string().required(),
    runId: Joi.string().uuid().required(),
    approved_by: Joi.string().optional(),
    notes: Joi.string().max(500).optional()
  }),

  triage: Joi.object({
    metrics: Joi.object().required(),
    ui_components: Joi.array().optional(),
    ocr_results: Joi.array().optional(),
    custom_rules: Joi.array().optional()
  })
};

// ============================================================================
// Utility Functions
// ============================================================================

const utils = {
  async ensureDir(dirPath) {
    await mkdirp(dirPath);
  },

  getBaselinePath(testId, name) {
    return path.join(config.dataDir, 'baselines', testId, `${name}.png`);
  },

  getCandidatePath(runId, name) {
    return path.join(config.dataDir, 'runs', runId, `${name}.png`);
  },

  getHeatmapPath(runId, name) {
    return path.join(config.dataDir, 'runs', runId, `${name}.heatmap.png`);
  },

  getMetadataPath(runId, name) {
    return path.join(config.dataDir, 'runs', runId, `${name}.metadata.json`);
  },

  async optimizeImage(buffer, maxWidth = 1920, maxHeight = 1080, quality = 90) {
    return sharp(buffer)
      .resize(maxWidth, maxHeight, { 
        fit: 'inside',
        withoutEnlargement: true
      })
      .png({ quality, progressive: true })
      .toBuffer();
  },

  async fileExists(filePath) {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  },

  sanitizeFilename(filename) {
    return filename.replace(/[^a-z0-9.-]/gi, '_').toLowerCase();
  }
};

// ============================================================================
// Qdrant Integration
// ============================================================================

class QdrantManager {
  constructor() {
    this.baseUrl = config.qdrantUrl;
  }

  async ensureCollection(collectionName, vectorSize = 512) {
    try {
      await axios.put(`${this.baseUrl}/collections/${collectionName}`, {
        vectors: { 
          size: vectorSize, 
          distance: 'Cosine',
          on_disk: true
        },
        optimizers_config: {
          deleted_threshold: 0.2,
          vacuum_min_vector_number: 1000,
          default_segment_number: 2
        },
        replication_factor: 1
      });
      logger.info(`Collection ${collectionName} ensured`);
    } catch (error) {
      if (error.response?.status !== 409) { // 409 = already exists
        logger.error(`Failed to create collection ${collectionName}:`, error.message);
        throw error;
      }
    }
  }

  async upsertPoint(collectionName, pointId, vector, payload) {
    try {
      const response = await axios.put(
        `${this.baseUrl}/collections/${collectionName}/points`,
        {
          points: [{
            id: pointId,
            vector,
            payload
          }]
        }
      );
      return response.data;
    } catch (error) {
      logger.error(`Failed to upsert point to ${collectionName}:`, error.message);
      throw error;
    }
  }

  async searchSimilar(collectionName, vector, limit = 10, filter = null) {
    try {
      const payload = {
        vector,
        limit,
        with_payload: true,
        with_vector: false
      };
      
      if (filter) {
        payload.filter = filter;
      }

      const response = await axios.post(
        `${this.baseUrl}/collections/${collectionName}/points/search`,
        payload
      );
      return response.data.result;
    } catch (error) {
      logger.error(`Failed to search ${collectionName}:`, error.message);
      throw error;
    }
  }

  async deleteOldPoints(collectionName, cutoffDate) {
    try {
      await axios.post(`${this.baseUrl}/collections/${collectionName}/points/delete`, {
        filter: {
          range: {
            timestamp: {
              lt: cutoffDate.toISOString()
            }
          }
        }
      });
    } catch (error) {
      logger.error(`Failed to delete old points from ${collectionName}:`, error.message);
    }
  }
}

const qdrant = new QdrantManager();

// ============================================================================
// ML Service Integration
// ============================================================================

class MLService {
  constructor() {
    this.baseUrl = config.mlUrl;
    this.timeout = 240000; // 4 minutes
  }

  async scoreImages(baselineB64, candidateB64, options = {}) {
    try {
      const payload = {
        baseline_png_b64: baselineB64,
        candidate_png_b64: candidateB64,
        masks: options.masks || [],
        use_sam_segmentation: options.use_sam_segmentation || false,
        enable_texture_analysis: options.enable_texture_analysis || true,
        anomaly_threshold: options.anomaly_threshold || 0.3,
        confidence_threshold: options.confidence_threshold || 0.8
      };

      const response = await axios.post(`${this.baseUrl}/score`, payload, {
        timeout: this.timeout,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      });

      return response.data;
    } catch (error) {
      logger.error('ML scoring failed:', error.message);
      if (error.response) {
        logger.error('ML service response:', error.response.data);
      }
      throw new Error(`ML scoring failed: ${error.message}`);
    }
  }

  async trainModel(imagesB64, labels, modelName) {
    try {
      const response = await axios.post(`${this.baseUrl}/train_anomaly_detector`, {
        images_b64: imagesB64,
        labels,
        model_name: modelName
      }, {
        timeout: 600000, // 10 minutes for training
        headers: { 'Content-Type': 'application/json' }
      });

      return response.data;
    } catch (error) {
      logger.error('ML training failed:', error.message);
      throw new Error(`ML training failed: ${error.message}`);
    }
  }

  async healthCheck() {
    try {
      const response = await axios.get(`${this.baseUrl}/health`, { timeout: 5000 });
      return response.data;
    } catch (error) {
      return { status: 'unhealthy', error: error.message };
    }
  }
}

const mlService = new MLService();

// ============================================================================
// Advanced Triage Engine
// ============================================================================

class TriageEngine {
  constructor() {
    this.rules = this.loadTriageRules();
  }

  loadTriageRules() {
    return {
      // Critical issues - block deployment
      critical: [
        {
          name: 'layout_shift',
          condition: (metrics) => metrics.anomaly_score > 0.5 && metrics.ssim < 0.8,
          message: 'Significant layout shift detected'
        },
        {
          name: 'missing_content',
          condition: (metrics, analysis) => {
            const hasText = analysis.ocr_results?.length > 0;
            const hasUI = analysis.ui_components?.length > 0;
            return metrics.anomaly_score > 0.4 && !hasText && !hasUI;
          },
          message: 'Potential missing content or rendering failure'
        }
      ],

      // Warning issues - require review
      warning: [
        {
          name: 'text_changes',
          condition: (metrics, analysis) => {
            const ocrResults = analysis.ocr_results || [];
            const textChangeIndicators = ocrResults.length > 0 && metrics.pixel_diff_ratio > 0.02;
            return textChangeIndicators && metrics.anomaly_score > 0.2;
          },
          message: 'Text content changes detected'
        },
        {
          name: 'ui_component_shift',
          condition: (metrics, analysis) => {
            const hasUIComponents = analysis.ui_components?.length > 0;
            return hasUIComponents && metrics.vit_cosine_distance > 0.3;
          },
          message: 'UI component positioning changed'
        }
      ],

      // Info issues - track but don't block
      info: [
        {
          name: 'minor_pixel_diff',
          condition: (metrics) => metrics.pixel_diff_ratio < 0.01 && metrics.anomaly_score < 0.15,
          message: 'Minor visual differences detected'
        },
        {
          name: 'dynamic_content',
          condition: (metrics, analysis) => {
            const dynamicIndicators = analysis.ocr_results?.some(ocr => 
              /\d{1,2}:\d{2}|\$\d+|\d+%|loading|pending/i.test(ocr.text)
            );
            return dynamicIndicators;
          },
          message: 'Dynamic content detected (timestamps, prices, loading states)'
        }
      ]
    };
  }

  analyze(metrics, analysis, customRules = []) {
    const issues = [];
    let maxSeverity = 0;

    // Check predefined rules
    for (const [severity, rules] of Object.entries(this.rules)) {
      const severityLevel = severity === 'critical' ? 3 : severity === 'warning' ? 2 : 1;
      
      for (const rule of rules) {
        if (rule.condition(metrics, analysis)) {
          issues.push({
            severity,
            rule: rule.name,
            message: rule.message,
            level: severityLevel
          });
          maxSeverity = Math.max(maxSeverity, severityLevel);
        }
      }
    }

    // Apply custom rules
    for (const customRule of customRules) {
      if (typeof customRule.condition === 'function' && customRule.condition(metrics, analysis)) {
        issues.push({
          severity: customRule.severity || 'info',
          rule: customRule.name,
          message: customRule.message,
          level: customRule.level || 1,
          custom: true
        });
      }
    }

    // Generate recommendations
    const recommendations = this.generateRecommendations(metrics, analysis, issues);

    // Determine if CI should pass
    const passCI = maxSeverity < 3; // Only critical issues block CI

    return {
      severity: maxSeverity,
      pass_ci: passCI,
      issues,
      recommendations,
      summary: this.generateSummary(metrics, analysis, issues)
    };
  }

  generateRecommendations(metrics, analysis, issues) {
    const recommendations = [];

    // Dynamic content recommendations
    if (issues.some(i => i.rule === 'dynamic_content')) {
      recommendations.push({
        type: 'masking',
        priority: 'medium',
        suggestion: 'Consider masking dynamic content areas (timestamps, prices, counters)',
        selectors: ['.timestamp', '.price', '.loading', '[data-dynamic]']
      });
    }

    // Layout shift recommendations
    if (issues.some(i => i.rule === 'layout_shift')) {
      recommendations.push({
        type: 'investigation',
        priority: 'high',
        suggestion: 'Investigate CSS changes, viewport size, or async content loading',
        actions: ['Check CSS commits', 'Verify viewport consistency', 'Review async loading']
      });
    }

    // Performance recommendations
    if (metrics.anomaly_score > 0.3) {
      recommendations.push({
        type: 'threshold',
        priority: 'low',
        suggestion: 'Consider adjusting anomaly threshold for this test case',
        current_threshold: 0.3,
        suggested_threshold: Math.min(0.5, metrics.anomaly_score + 0.1)
      });
    }

    return recommendations;
  }

  generateSummary(metrics, analysis, issues) {
    const score = metrics.anomaly_score?.toFixed(3) || '0.000';
    const pixelRatio = (metrics.pixel_diff_ratio || 0).toFixed(3);
    const confidence = (metrics.confidence || 0).toFixed(3);
    
    const criticalCount = issues.filter(i => i.severity === 'critical').length;
    const warningCount = issues.filter(i => i.severity === 'warning').length;
    
    return `Anomaly: ${score}, Pixels: ${pixelRatio}, Confidence: ${confidence}. ` +
           `Issues: ${criticalCount} critical, ${warningCount} warnings`;
  }
}

const triageEngine = new TriageEngine();

// ============================================================================
// API Routes
// ============================================================================

// Health check
app.get('/health', async (req, res) => {
  try {
    const mlHealth = await mlService.healthCheck();
    
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      version: '2.0.0',
      services: {
        ml: mlHealth,
        qdrant: { status: 'connected' }, // Add actual check if needed
        storage: { 
          data_dir: config.dataDir,
          model_registry: config.modelRegistry
        }
      },
      stats: {
        uptime: process.uptime(),
        memory: process.memoryUsage()
      }
    });
  } catch (error) {
    logger.error('Health check failed:', error);
    res.status(503).json({
      status: 'unhealthy',
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Create new test run
app.post('/runs', async (req, res) => {
  try {
    const { error, value } = schemas.runCreate.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const runId = uuidv4();
    const runData = {
      id: runId,
      ...value,
      status: 'created',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };

    // Create run directory
    const runDir = path.join(config.dataDir, 'runs', runId);
    await utils.ensureDir(runDir);

    // Save run metadata
    await fs.writeFile(
      path.join(runDir, 'run.json'),
      JSON.stringify(runData, null, 2)
    );

    logger.info(`Created run ${runId} for project ${value.project_id}`);
    res.json({ runId, ...runData });
  } catch (error) {
    logger.error('Failed to create run:', error);
    res.status(500).json({ error: 'Failed to create run' });
  }
});

// Upload and process snapshot
app.post('/snapshots', upload.single('image'), async (req, res) => {
  try {
    const { error, value } = schemas.snapshot.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { runId, testId, name, masks, viewport, options } = value;
    
    if (!req.file) {
      return res.status(400).json({ error: 'Image file is required' });
    }

    // Optimize and save candidate image
    const optimizedImage = await utils.optimizeImage(req.file.buffer);
    const candidatePath = utils.getCandidatePath(runId, utils.sanitizeFilename(name));
    
    await utils.ensureDir(path.dirname(candidatePath));
    await fs.writeFile(candidatePath, optimizedImage);

    // Check for existing baseline
    const baselinePath = utils.getBaselinePath(testId, utils.sanitizeFilename(name));
    const baselineExists = await utils.fileExists(baselinePath);

    if (!baselineExists) {
      // Create baseline from first image
      await utils.ensureDir(path.dirname(baselinePath));
      await fs.writeFile(baselinePath, optimizedImage);
      
      logger.info(`Created baseline for ${testId}/${name}`);
      return res.json({
        status: 'baseline_created',
        is_anomaly: false,
        runId,
        testId,
        name,
        files: {
          baseline: baselinePath,
          candidate: candidatePath
        }
      });
    }

    // Prepare for ML scoring
    const baselineB64 = (await fs.readFile(baselinePath)).toString('base64');
    const candidateB64 = optimizedImage.toString('base64');
    const parsedMasks = JSON.parse(masks);

    // Call ML service for scoring
    logger.info(`Scoring ${testId}/${name} with ML service`);
    const mlResult = await mlService.scoreImages(baselineB64, candidateB64, {
      masks: parsedMasks,
      ...options
    });

    // Save artifacts
    const heatmapPath = utils.getHeatmapPath(runId, utils.sanitizeFilename(name));
    if (mlResult.artifacts?.heatmap_png_b64) {
      await fs.writeFile(heatmapPath, Buffer.from(mlResult.artifacts.heatmap_png_b64, 'base64'));
    }

    // Prepare metadata
    const metadata = {
      run_id: runId,
      test_id: testId,
      name,
      timestamp: new Date().toISOString(),
      viewport,
      masks: parsedMasks,
      ml_result: mlResult,
      files: {
        baseline: baselinePath,
        candidate: candidatePath,
        heatmap: heatmapPath
      }
    };

    // Save metadata
    const metadataPath = utils.getMetadataPath(runId, utils.sanitizeFilename(name));
    await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2));

    // Initialize Qdrant collections
    await qdrant.ensureCollection(config.collections.uiDiffs, 512);

    // Store in Qdrant for similarity search
    const pointId = `${runId}::${testId}::${name}`;
    const embeddings = mlResult.embeddings?.clip_512;
    
    if (embeddings) {
      await qdrant.upsertPoint(
        config.collections.uiDiffs,
        pointId,
        embeddings,
        {
          ...metadata,
          // Remove binary data for storage efficiency
          ml_result: {
            ...mlResult,
            artifacts: undefined
          }
        }
      );
    }

    // Run triage analysis
    const triageResult = triageEngine.analyze(
      mlResult.metrics,
      mlResult.analysis,
      options?.custom_rules
    );

    // Build response
    const response = {
      status: 'scored',
      ...mlResult,
      triage: triageResult,
      files: metadata.files,
      metadata: {
        run_id: runId,
        test_id: testId,
        name,
        timestamp: metadata.timestamp
      }
    };

    logger.info(`Completed scoring for ${testId}/${name}: anomaly=${mlResult.metrics.anomaly_score?.toFixed(3)}, pass_ci=${triageResult.pass_ci}`);
    res.json(response);

  } catch (error) {
    logger.error('Snapshot processing failed:', error);
    res.status(500).json({
      error: 'Snapshot processing failed',
      detail: error.message
    });
  }
});

// Approve changes and update baseline
app.post('/approve', async (req, res) => {
  try {
    const { error, value } = schemas.approve.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { testId, name, runId, approved_by, notes } = value;
    const sanitizedName = utils.sanitizeFilename(name);
    
    const candidatePath = utils.getCandidatePath(runId, sanitizedName);
    const baselinePath = utils.getBaselinePath(testId, sanitizedName);

    // Check if candidate exists
    if (!(await utils.fileExists(candidatePath))) {
      return res.status(404).json({ error: 'Candidate image not found' });
    }

    // Copy candidate to baseline
    await utils.ensureDir(path.dirname(baselinePath));
    const candidateBuffer = await fs.readFile(candidatePath);
    await fs.writeFile(baselinePath, candidateBuffer);

    // Log approval
    const approvalRecord = {
      test_id: testId,
      name,
      run_id: runId,
      approved_by: approved_by || 'system',
      notes,
      approved_at: new Date().toISOString(),
      baseline_updated: baselinePath
    };

    // Save approval record
    const approvalsDir = path.join(config.dataDir, 'approvals');
    await utils.ensureDir(approvalsDir);
    const approvalFile = path.join(approvalsDir, `${testId}_${sanitizedName}_${runId}.json`);
    await fs.writeFile(approvalFile, JSON.stringify(approvalRecord, null, 2));

    logger.info(`Approved baseline update for ${testId}/${name} by ${approved_by || 'system'}`);
    res.json({
      status: 'baseline_updated',
      ...approvalRecord
    });

  } catch (error) {
    logger.error('Approval failed:', error);
    res.status(500).json({
      error: 'Approval failed',
      detail: error.message
    });
  }
});

// Find similar visual cases
app.post('/similar', async (req, res) => {
  try {
    const { vector, limit = 10, filter } = req.body;
    
    if (!vector || !Array.isArray(vector)) {
      return res.status(400).json({ error: 'Valid vector array is required' });
    }

    await qdrant.ensureCollection(config.collections.uiDiffs, vector.length);
    
    const results = await qdrant.searchSimilar(
      config.collections.uiDiffs,
      vector,
      Math.min(limit, 50), // Cap at 50 results
      filter
    );

    res.json({
      results,
      count: results.length,
      query_vector_size: vector.length
    });

  } catch (error) {
    logger.error('Similarity search failed:', error);
    res.status(500).json({
      error: 'Similarity search failed',
      detail: error.message
    });
  }
});

// Advanced triage analysis
app.post('/triage', async (req, res) => {
  try {
    const { error, value } = schemas.triage.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { metrics, ui_components, ocr_results, custom_rules } = value;
    
    const analysis = {
      ui_components: ui_components || [],
      ocr_results: ocr_results || []
    };

    const triageResult = triageEngine.analyze(metrics, analysis, custom_rules || []);
    
    res.json(triageResult);

  } catch (error) {
    logger.error('Triage analysis failed:', error);
    res.status(500).json({
      error: 'Triage analysis failed',
      detail: error.message
    });
  }
});

// Train ML model
app.post('/train', async (req, res) => {
  try {
    const { images_b64, labels, model_name = `model_${Date.now()}` } = req.body;
    
    if (!images_b64 || !labels || images_b64.length !== labels.length) {
      return res.status(400).json({ 
        error: 'images_b64 and labels arrays must be provided and have equal length' 
      });
    }

    logger.info(`Starting training for model ${model_name} with ${images_b64.length} samples`);
    
    const trainingResult = await mlService.trainModel(images_b64, labels, model_name);
    
    res.json({
      status: 'training_started',
      model_name,
      ...trainingResult
    });

  } catch (error) {
    logger.error('Model training failed:', error);
    res.status(500).json({
      error: 'Model training failed',
      detail: error.message
    });
  }
});

// Serve artifacts
app.get('/artifacts/*', async (req, res) => {
  try {
    const artifactPath = path.join(config.dataDir, req.params[0] || '');
    
    // Security check: ensure path is within data directory
    const resolvedPath = path.resolve(artifactPath);
    const resolvedDataDir = path.resolve(config.dataDir);
    
    if (!resolvedPath.startsWith(resolvedDataDir)) {
      return res.status(403).json({ error: 'Access denied' });
    }

    if (!(await utils.fileExists(artifactPath))) {
      return res.status(404).json({ error: 'Artifact not found' });
    }

    // Set appropriate content type based on file extension
    const ext = path.extname(artifactPath).toLowerCase();
    const contentTypes = {
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.webp': 'image/webp',
      '.json': 'application/json'
    };

    const contentType = contentTypes[ext] || 'application/octet-stream';
    res.setHeader('Content-Type', contentType);
    res.setHeader('Cache-Control', 'public, max-age=86400'); // 24 hours
    
    const fileStream = (await import('fs')).createReadStream(artifactPath);
    fileStream.pipe(res);

  } catch (error) {
    logger.error('Artifact serving failed:', error);
    res.status(500).json({ error: 'Failed to serve artifact' });
  }
});

// Get run statistics
app.get('/stats', async (req, res) => {
  try {
    const runsDir = path.join(config.dataDir, 'runs');
    const baselinesDir = path.join(config.dataDir, 'baselines');
    
    // This is a simple implementation - for production, consider using a database
    let runCount = 0;
    let baselineCount = 0;
    
    try {
      const runs = await fs.readdir(runsDir);
      runCount = runs.length;
    } catch (e) {
      // Directory doesn't exist yet
    }
    
    try {
      const baselines = await fs.readdir(baselinesDir);
      baselineCount = baselines.length;
    } catch (e) {
      // Directory doesn't exist yet
    }

    res.json({
      statistics: {
        total_runs: runCount,
        total_baselines: baselineCount,
        data_directory: config.dataDir,
        uptime_seconds: Math.floor(process.uptime())
      },
      configuration: {
        max_image_size: config.maxImageSize,
        max_images_per_run: config.maxImagesPerRun,
        retention_days: config.retentionDays
      }
    });

  } catch (error) {
    logger.error('Stats retrieval failed:', error);
    res.status(500).json({ error: 'Failed to retrieve statistics' });
  }
});

// ============================================================================
// Error Handling
// ============================================================================

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    path: req.path,
    method: req.method
  });
});

// Global error handler
app.use((error, req, res, next) => {
  logger.error('Unhandled error:', error);
  
  if (error instanceof multer.MulterError) {
    return res.status(400).json({
      error: 'File upload error',
      detail: error.message
    });
  }
  
  res.status(500).json({
    error: 'Internal server error',
    detail: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
  });
});

// ============================================================================
// Cleanup Tasks
// ============================================================================

// Clean up old data periodically
cron.schedule('0 2 * * *', async () => { // Daily at 2 AM
  try {
    logger.info('Starting cleanup of old data...');
    
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - config.retentionDays);
    
    // Clean up old Qdrant points
    await qdrant.deleteOldPoints(config.collections.uiDiffs, cutoffDate);
    
    // TODO: Clean up old files from filesystem
    
    logger.info('Cleanup completed');
  } catch (error) {
    logger.error('Cleanup failed:', error);
  }
});

// ============================================================================
// Server Startup
// ============================================================================

const server = app.listen(config.port, async () => {
  logger.info(`Visual Anomaly Detection API v2.0.0 listening on port ${config.port}`);
  logger.info(`Data directory: ${config.dataDir}`);
  logger.info(`ML service: ${config.mlUrl}`);
  logger.info(`Qdrant service: ${config.qdrantUrl}`);
  
  // Ensure required directories exist
  await utils.ensureDir(config.dataDir);
  await utils.ensureDir(path.join(config.dataDir, 'baselines'));
  await utils.ensureDir(path.join(config.dataDir, 'runs'));
  await utils.ensureDir(path.join(config.dataDir, 'approvals'));
  await utils.ensureDir(path.dirname(path.join(__dirname, 'logs/app.log')));
  
  // Initialize Qdrant collections
  try {
    await qdrant.ensureCollection(config.collections.uiDiffs, 512);
    logger.info('Qdrant collections initialized');
  } catch (error) {
    logger.warn('Failed to initialize Qdrant collections:', error.message);
  }
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    logger.info('Process terminated');
    process.exit(0);
  });
});

export default app;