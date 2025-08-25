/**
 * Intelligent Masking Helper
 * Automatically compute masks for dynamic content areas
 */

import { Page, Locator } from '@playwright/test';
import { BoundingBox } from './visual-helper';

export interface MaskRule {
  selector: string;
  reason: string;
  priority: number;
  dynamic?: boolean;
}

export class MaskHelper {
  // Common dynamic content patterns
  private readonly DEFAULT_MASK_RULES: MaskRule[] = [
    // Timestamps and dates
    { selector: '[data-testid*="time"], [data-testid*="date"]', reason: 'timestamp', priority: 1 },
    { selector: '.timestamp, .datetime, .time-ago', reason: 'timestamp', priority: 1 },
    { selector: 'time[datetime]', reason: 'timestamp', priority: 1 },
    
    // User-specific content
    { selector: '.user-avatar, .profile-picture, [data-testid="avatar"]', reason: 'user_content', priority: 2 },
    { selector: '.username, .user-name, [data-testid*="user"]', reason: 'user_content', priority: 2 },
    
    // Dynamic counters and metrics
    { selector: '.counter, .metric, .stat-value', reason: 'dynamic_numbers', priority: 2 },
    { selector: '[data-testid*="count"], [data-testid*="total"]', reason: 'dynamic_numbers', priority: 2 },
    
    // Loading states
    { selector: '.loading, .spinner, .skeleton', reason: 'loading_state', priority: 1, dynamic: true },
    { selector: '[data-loading="true"], [data-testid*="loading"]', reason: 'loading_state', priority: 1, dynamic: true },
    
    // Notifications and alerts
    { selector: '.notification, .alert, .toast', reason: 'notification', priority: 1, dynamic: true },
    { selector: '[role="alert"], [data-testid*="notification"]', reason: 'notification', priority: 1, dynamic: true },
    
    // Charts and visualizations (often have micro-animations)
    { selector: 'canvas, .chart, .graph, [data-testid*="chart"]', reason: 'visualization', priority: 3 },
    
    // Maps and interactive elements
    { selector: '.map, .interactive-map, [data-testid*="map"]', reason: 'interactive_content', priority: 3 },
    
    // Real-time data
    { selector: '.real-time, .live-data, [data-testid*="live"]', reason: 'real_time_data', priority: 1 },
    
    // Generated IDs and random content
    { selector: '[id*="random"], [class*="generated"]', reason: 'generated_content', priority: 2 },
    
    // Advertisements
    { selector: '.ad, .advertisement, .banner-ad', reason: 'advertisement', priority: 2 },
    
    // Video and media elements
    { selector: 'video, .video-player, .media-player', reason: 'media_content', priority: 3 },
    
    // Progress indicators
    { selector: '.progress, .progress-bar, [role="progressbar"]', reason: 'progress_indicator', priority: 2 }
  ];

  async computeMasks(page: Page, customSelectors: string[] = []): Promise<BoundingBox[]> {
    const masks: BoundingBox[] = [];
    
    // Combine custom selectors with default rules
    const allSelectors = [
      ...customSelectors.map(selector => ({ selector, reason: 'custom', priority: 1 })),
      ...this.DEFAULT_MASK_RULES
    ];
    
    // Sort by priority (lower number = higher priority)
    allSelectors.sort((a, b) => a.priority - b.priority);
    
    for (const rule of allSelectors) {
      try {
        const elements = await page.locator(rule.selector).all();
        
        for (const element of elements) {
          const box = await element.boundingBox();
          if (box && this.isValidBoundingBox(box)) {
            // Expand mask slightly to ensure complete coverage
            const expandedBox = this.expandBoundingBox(box, 2);
            
            // Check if this mask overlaps significantly with existing masks
            if (!this.hasSignificantOverlap(expandedBox, masks)) {
              masks.push(expandedBox);
              
              console.log(`🎭 Masking element: ${rule.selector} (${rule.reason}) - ` +
                         `${expandedBox.width}x${expandedBox.height} at (${expandedBox.x}, ${expandedBox.y})`);
            }
          }
        }
      } catch (error) {
        // Selector might not exist on this page, continue
        continue;
      }
    }
    
    // Apply intelligent mask merging
    const mergedMasks = this.mergeMasks(masks);
    
    console.log(`🎭 Applied ${mergedMasks.length} masks (merged from ${masks.length} candidates)`);
    return mergedMasks;
  }

  async computeAdaptiveMasks(page: Page, options: {
    detectText?: boolean;
    detectImages?: boolean;
    detectAnimations?: boolean;
    minSize?: number;
  } = {}): Promise<BoundingBox[]> {
    const {
      detectText = true,
      detectImages = true,
      detectAnimations = true,
      minSize = 10
    } = options;
    
    const masks: BoundingBox[] = [];
    
    // Detect potentially dynamic text content
    if (detectText) {
      const textMasks = await this.detectDynamicText(page);
      masks.push(...textMasks);
    }
    
    // Detect images that might change
    if (detectImages) {
      const imageMasks = await this.detectDynamicImages(page);
      masks.push(...imageMasks);
    }
    
    // Detect elements with animations
    if (detectAnimations) {
      const animationMasks = await this.detectAnimatedElements(page);
      masks.push(...animationMasks);
    }
    
    // Filter by minimum size
    const filteredMasks = masks.filter(mask => 
      mask.width >= minSize && mask.height >= minSize
    );
    
    return this.mergeMasks(filteredMasks);
  }

  private async detectDynamicText(page: Page): Promise<BoundingBox[]> {
    const masks: BoundingBox[] = [];
    
    // Patterns that indicate dynamic text
    const dynamicTextPatterns = [
      /\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?/i, // Time
      /\d{1,2}\/\d{1,2}\/\d{2,4}/i, // Date
      /\$\d+(\.\d{2})?/, // Money
      /\d+%/, // Percentage
      /\d+\s+(seconds?|minutes?|hours?|days?)\s+ago/i, // Relative time
      /(loading|pending|processing)/i, // Status text
      /^\d+$/ // Pure numbers (counters)
    ];
    
    try {
      // Get all text nodes
      const textElements = await page.locator('*:has-text("")').all();
      
      for (const element of textElements) {
        const text = await element.textContent();
        if (!text) continue;
        
        // Check if text matches dynamic patterns
        const isDynamic = dynamicTextPatterns.some(pattern => pattern.test(text.trim()));
        
        if (isDynamic) {
          const box = await element.boundingBox();
          if (box && this.isValidBoundingBox(box)) {
            masks.push(box);
          }
        }
      }
    } catch (error) {
      console.warn('Failed to detect dynamic text:', error.message);
    }
    
    return masks;
  }

  private async detectDynamicImages(page: Page): Promise<BoundingBox[]> {
    const masks: BoundingBox[] = [];
    
    try {
      // Look for images with dynamic sources
      const images = await page.locator('img').all();
      
      for (const img of images) {
        const src = await img.getAttribute('src');
        const alt = await img.getAttribute('alt');
        
        // Check for dynamic image indicators
        const isDynamic = src && (
          src.includes('timestamp=') ||
          src.includes('cache-bust=') ||
          src.includes('random=') ||
          src.includes('avatar') ||
          src.includes('profile') ||
          (alt && (alt.includes('avatar') || alt.includes('profile')))
        );
        
        if (isDynamic) {
          const box = await img.boundingBox();
          if (box && this.isValidBoundingBox(box)) {
            masks.push(box);
          }
        }
      }
    } catch (error) {
      console.warn('Failed to detect dynamic images:', error.message);
    }
    
    return masks;
  }

  private async detectAnimatedElements(page: Page): Promise<BoundingBox[]> {
    const masks: BoundingBox[] = [];
    
    try {
      // Detect elements with CSS animations or transitions
      const animatedElements = await page.evaluate(() => {
        const elements: Element[] = [];
        
        document.querySelectorAll('*').forEach(el => {
          const computedStyle = window.getComputedStyle(el);
          const hasAnimation = computedStyle.animationName !== 'none' && 
                              computedStyle.animationName !== '';
          const hasTransition = computedStyle.transitionProperty !== 'none' && 
                               computedStyle.transitionProperty !== '';
          
          if (hasAnimation || hasTransition) {
            elements.push(el);
          }
        });
        
        return elements.map(el => {
          const rect = el.getBoundingClientRect();
          return {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height
          };
        });
      });
      
      masks.push(...animatedElements.filter(box => this.isValidBoundingBox(box)));
    } catch (error) {
      console.warn('Failed to detect animated elements:', error.message);
    }
    
    return masks;
  }

  private isValidBoundingBox(box: BoundingBox): boolean {
    return box.width > 0 && 
           box.height > 0 && 
           box.x >= 0 && 
           box.y >= 0 &&
           box.width < 10000 && 
           box.height < 10000; // Sanity check
  }

  private expandBoundingBox(box: BoundingBox, pixels: number): BoundingBox {
    return {
      x: Math.max(0, box.x - pixels),
      y: Math.max(0, box.y - pixels),
      width: box.width + (pixels * 2),
      height: box.height + (pixels * 2)
    };
  }

  private hasSignificantOverlap(newBox: BoundingBox, existingMasks: BoundingBox[], threshold: number = 0.8): boolean {
    for (const mask of existingMasks) {
      const overlapArea = this.getOverlapArea(newBox, mask);
      const newBoxArea = newBox.width * newBox.height;
      const overlapRatio = overlapArea / newBoxArea;
      
      if (overlapRatio > threshold) {
        return true;
      }
    }
    return false;
  }

  private getOverlapArea(box1: BoundingBox, box2: BoundingBox): number {
    const xOverlap = Math.max(0, Math.min(box1.x + box1.width, box2.x + box2.width) - Math.max(box1.x, box2.x));
    const yOverlap = Math.max(0, Math.min(box1.y + box1.height, box2.y + box2.height) - Math.max(box1.y, box2.y));
    return xOverlap * yOverlap;
  }

  private mergeMasks(masks: BoundingBox[]): BoundingBox[] {
    if (masks.length <= 1) return masks;
    
    const mergedMasks: BoundingBox[] = [];
    const processed = new Set<number>();
    
    for (let i = 0; i < masks.length; i++) {
      if (processed.has(i)) continue;
      
      let currentMask = { ...masks[i] };
      processed.add(i);
      
      // Find overlapping masks to merge
      for (let j = i + 1; j < masks.length; j++) {
        if (processed.has(j)) continue;
        
        if (this.shouldMergeMasks(currentMask, masks[j])) {
          currentMask = this.combineBoundingBoxes(currentMask, masks[j]);
          processed.add(j);
        }
      }
      
      mergedMasks.push(currentMask);
    }
    
    return mergedMasks;
  }

  private shouldMergeMasks(box1: BoundingBox, box2: BoundingBox, threshold: number = 0.3): boolean {
    const overlapArea = this.getOverlapArea(box1, box2);
    const box1Area = box1.width * box1.height;
    const box2Area = box2.width * box2.height;
    const smallerArea = Math.min(box1Area, box2Area);
    
    return overlapArea / smallerArea > threshold;
  }

  private combineBoundingBoxes(box1: BoundingBox, box2: BoundingBox): BoundingBox {
    const x = Math.min(box1.x, box2.x);
    const y = Math.min(box1.y, box2.y);
    const right = Math.max(box1.x + box1.width, box2.x + box2.width);
    const bottom = Math.max(box1.y + box1.height, box2.y + box2.height);
    
    return {
      x,
      y,
      width: right - x,
      height: bottom - y
    };
  }

  // Utility methods for specific masking scenarios

  async maskDynamicContent(page: Page): Promise<BoundingBox[]> {
    return this.computeMasks(page);
  }

  async maskUserSpecificContent(page: Page): Promise<BoundingBox[]> {
    const userContentSelectors = [
      '.user-avatar',
      '.profile-picture',
      '.username',
      '.user-email',
      '[data-testid*="user"]',
      '[data-user-id]'
    ];
    
    return this.computeMasks(page, userContentSelectors);
  }

  async maskTemporalContent(page: Page): Promise<BoundingBox[]> {
    const temporalSelectors = [
      '.timestamp',
      '.datetime',
      '.time-ago',
      '.last-updated',
      '[data-testid*="time"]',
      '[data-testid*="date"]',
      'time[datetime]'
    ];
    
    return this.computeMasks(page, temporalSelectors);
  }

  async maskLoadingStates(page: Page): Promise<BoundingBox[]> {
    const loadingSelectors = [
      '.loading',
      '.spinner',
      '.skeleton',
      '.loading-placeholder',
      '[data-loading="true"]',
      '[aria-busy="true"]'
    ];
    
    return this.computeMasks(page, loadingSelectors);
  }

  // Advanced masking for specific UI patterns

  async maskDataTables(page: Page, options: {
    maskRowNumbers?: boolean;
    maskTimestamps?: boolean;
    maskUserColumns?: boolean;
  } = {}): Promise<BoundingBox[]> {
    const masks: BoundingBox[] = [];
    
    try {
      const tables = await page.locator('table, [role="table"]').all();
      
      for (const table of tables) {
        if (options.maskRowNumbers) {
          const rowNumbers = await table.locator('td:first-child:has-text(/^\\d+$/)').all();
          for (const cell of rowNumbers) {
            const box = await cell.boundingBox();
            if (box) masks.push(box);
          }
        }
        
        if (options.maskTimestamps) {
          const timestampCells = await table.locator('td:has-text(/(ago|AM|PM|:\\d{2})/)').all();
          for (const cell of timestampCells) {
            const box = await cell.boundingBox();
            if (box) masks.push(box);
          }
        }
      }
    } catch (error) {
      console.warn('Failed to mask data tables:', error.message);
    }
    
    return masks;
  }

  async maskChartElements(page: Page): Promise<BoundingBox[]> {
    const chartSelectors = [
      'canvas',
      '.chart',
      '.graph',
      '.plot',
      'svg[class*="chart"]',
      '[data-testid*="chart"]',
      '[data-testid*="graph"]'
    ];
    
    return this.computeMasks(page, chartSelectors);
  }
}