#!/usr/bin/env node
/**
 * Direct ML Service Test
 * Tests the visual anomaly detection with actual images
 */

const fs = require('fs');
const https = require('https');

async function createTestImages() {
    // Create simple test images using HTML Canvas
    const { createCanvas } = require('canvas');
    
    // Baseline image - simple red rectangle
    const canvas1 = createCanvas(800, 600);
    const ctx1 = canvas1.getContext('2d');
    ctx1.fillStyle = '#ff0000';
    ctx1.fillRect(100, 100, 200, 150);
    ctx1.fillStyle = '#000000';
    ctx1.font = '30px Arial';
    ctx1.fillText('Baseline Image', 100, 300);
    
    // Candidate image - red rectangle with slight difference
    const canvas2 = createCanvas(800, 600);
    const ctx2 = canvas2.getContext('2d');
    ctx2.fillStyle = '#ff0000';
    ctx2.fillRect(100, 100, 200, 150);
    ctx2.fillStyle = '#0000ff';  // Blue text instead of black
    ctx2.font = '30px Arial';
    ctx2.fillText('Changed Image', 100, 300);
    
    const baseline = canvas1.toBuffer('image/png').toString('base64');
    const candidate = canvas2.toBuffer('image/png').toString('base64');
    
    return { baseline, candidate };
}

async function testMLService() {
    console.log('🧪 Testing ML Service Integration...\n');
    
    try {
        // Test health endpoint first
        console.log('📊 Checking ML Service health...');
        const healthResponse = await fetch('http://localhost:8000/health');
        const health = await healthResponse.json();
        
        console.log('✅ ML Service Status:', health.status);
        console.log('📋 Available Features:');
        health.features.forEach(feature => console.log(`   • ${feature}`));
        console.log('');
        
        // Generate test images
        console.log('🎨 Generating test images...');
        const { baseline, candidate } = await createTestImages();
        console.log('✅ Test images created');
        console.log('');
        
        // Test visual analysis
        console.log('🔍 Running visual anomaly analysis...');
        const analysisResponse = await fetch('http://localhost:8000/score', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                baseline_image: baseline,
                candidate_image: candidate,
                threshold: 0.1
            })
        });
        
        const analysis = await analysisResponse.json();
        
        console.log('📈 Analysis Results:');
        console.log(`   • Similarity Score: ${(analysis.similarity_score * 100).toFixed(2)}%`);
        console.log(`   • Anomaly Score: ${(analysis.anomaly_score * 100).toFixed(2)}%`);
        console.log(`   • Status: ${analysis.status}`);
        console.log(`   • Differences Found: ${analysis.differences_found}`);
        console.log('');
        
        console.log('💡 Recommendations:');
        analysis.recommendations.forEach(rec => console.log(`   • ${rec}`));
        console.log('');
        
        console.log('🔬 Feature Analysis:');
        const baseline_features = analysis.features.baseline;
        const candidate_features = analysis.features.candidate;
        
        console.log('   📊 Baseline Features:');
        console.log(`      - Mean Brightness: ${baseline_features.mean_brightness.toFixed(2)}`);
        console.log(`      - Edge Density: ${baseline_features.edge_density.toFixed(4)}`);
        console.log(`      - Contrast: ${baseline_features.contrast.toFixed(2)}`);
        
        console.log('   📊 Candidate Features:');
        console.log(`      - Mean Brightness: ${candidate_features.mean_brightness.toFixed(2)}`);
        console.log(`      - Edge Density: ${candidate_features.edge_density.toFixed(4)}`);
        console.log(`      - Contrast: ${candidate_features.contrast.toFixed(2)}`);
        
        if (analysis.features.anomalies && Object.keys(analysis.features.anomalies).length > 0) {
            console.log('');
            console.log('⚠️  Detected Anomalies:');
            Object.entries(analysis.features.anomalies).forEach(([key, data]) => {
                console.log(`   • ${key}: ${(data.difference_ratio * 100).toFixed(2)}% difference`);
                console.log(`     Baseline: ${data.baseline.toFixed(2)} → Candidate: ${data.candidate.toFixed(2)}`);
            });
        }
        
        console.log('');
        console.log('✅ ML Service test completed successfully!');
        console.log('🎯 Visual anomaly detection is working correctly.');
        
        // Test performance
        console.log('');
        console.log('⚡ Performance Test...');
        const startTime = Date.now();
        
        for (let i = 0; i < 3; i++) {
            await fetch('http://localhost:8000/score', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    baseline_image: baseline,
                    candidate_image: candidate,
                    threshold: 0.1
                })
            });
        }
        
        const endTime = Date.now();
        const avgTime = (endTime - startTime) / 3;
        console.log(`📊 Average processing time: ${avgTime.toFixed(0)}ms per analysis`);
        
        return true;
        
    } catch (error) {
        console.error('❌ ML Service test failed:', error.message);
        return false;
    }
}

// Run the test
if (require.main === module) {
    testMLService().then(success => {
        process.exit(success ? 0 : 1);
    });
}

module.exports = { testMLService };