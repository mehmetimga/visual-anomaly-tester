/**
 * Global teardown for Playwright visual tests
 * Cleanup and summary reporting
 */

import { FullConfig } from '@playwright/test';

async function globalTeardown(config: FullConfig) {
  console.log('🧹 Cleaning up visual testing environment...');

  // Log final summary
  console.log('📊 Visual testing session complete');
  
  // Could add cleanup tasks here like:
  // - Compress old test artifacts
  // - Send summary notifications
  // - Update dashboards
}

export default globalTeardown;