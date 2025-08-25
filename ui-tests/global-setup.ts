/**
 * Global setup for Playwright visual tests
 * Prepares test environment and ensures services are ready
 */

import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  console.log('🔧 Setting up visual testing environment...');

  // Check if services are running
  const services = [
    { name: 'API Service', url: 'http://localhost:8080/health' },
    { name: 'ML Service', url: 'http://localhost:8000/health' },
    { name: 'Qdrant DB', url: 'http://localhost:6333/health' }
  ];

  console.log('⏳ Checking service availability...');
  
  for (const service of services) {
    try {
      const response = await fetch(service.url);
      if (response.ok) {
        console.log(`✅ ${service.name}: Ready`);
      } else {
        console.log(`⚠️  ${service.name}: Available but not healthy`);
      }
    } catch (error) {
      console.log(`❌ ${service.name}: Not available`);
      console.log(`   Make sure to start services with: docker compose up -d`);
    }
  }

  // Warm up browser for consistent performance
  console.log('🔥 Warming up browser...');
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  
  // Pre-load any common resources
  try {
    await page.goto('about:blank');
    await page.evaluate(() => {
      // Pre-warm JavaScript engine
      const start = performance.now();
      while (performance.now() - start < 100) {
        Math.random();
      }
    });
  } catch (error) {
    console.log('⚠️  Browser warm-up failed:', error.message);
  }

  await browser.close();
  console.log('✅ Global setup complete');
}

export default globalSetup;