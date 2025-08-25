#!/usr/bin/env node
/**
 * Training Data Generator for Visual Anomaly Detection
 * Creates diverse test images and runs ML analysis to build training dataset
 */

const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');
const path = require('path');

class TrainingDataGenerator {
    constructor() {
        this.outputDir = path.join(__dirname, '..', 'training_data');
        this.resultsDir = path.join(this.outputDir, 'results');
        this.imagesDir = path.join(this.outputDir, 'images');
        
        // Ensure directories exist
        [this.outputDir, this.resultsDir, this.imagesDir].forEach(dir => {
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
        });

        this.testCases = [];
        this.results = [];
    }

    // Generate various UI component types
    generateButton(canvas, ctx, variant = 'primary') {
        const colors = {
            primary: { bg: '#007bff', text: '#ffffff', border: '#0056b3' },
            secondary: { bg: '#6c757d', text: '#ffffff', border: '#545b62' },
            success: { bg: '#28a745', text: '#ffffff', border: '#1e7e34' },
            danger: { bg: '#dc3545', text: '#ffffff', border: '#bd2130' },
            warning: { bg: '#ffc107', text: '#212529', border: '#d39e00' }
        };

        const color = colors[variant] || colors.primary;
        const x = 50, y = 50, width = 120, height = 40, radius = 5;

        // Button background
        ctx.fillStyle = color.bg;
        ctx.strokeStyle = color.border;
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        ctx.roundRect(x, y, width, height, radius);
        ctx.fill();
        ctx.stroke();

        // Button text
        ctx.fillStyle = color.text;
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Click Me', x + width/2, y + height/2);

        return { type: 'button', variant, bounds: { x, y, width, height } };
    }

    generateCard(canvas, ctx, variant = 'normal') {
        const x = 30, y = 30, width = 280, height = 180, radius = 8;

        // Card background
        ctx.fillStyle = variant === 'error' ? '#f8d7da' : '#ffffff';
        ctx.strokeStyle = variant === 'error' ? '#dc3545' : '#dee2e6';
        ctx.lineWidth = 1;
        ctx.shadowColor = 'rgba(0,0,0,0.1)';
        ctx.shadowBlur = 10;
        ctx.shadowOffsetY = 2;
        
        ctx.beginPath();
        ctx.roundRect(x, y, width, height, radius);
        ctx.fill();
        ctx.stroke();
        
        ctx.shadowColor = 'transparent'; // Reset shadow

        // Card header
        ctx.fillStyle = variant === 'error' ? '#dc3545' : '#495057';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Card Title', x + 20, y + 35);

        // Card content
        ctx.fillStyle = '#6c757d';
        ctx.font = '14px Arial';
        ctx.fillText('This is card content with some', x + 20, y + 65);
        ctx.fillText('descriptive text about the item.', x + 20, y + 85);

        // Status indicator
        const statusColor = variant === 'error' ? '#dc3545' : '#28a745';
        ctx.fillStyle = statusColor;
        ctx.beginPath();
        ctx.arc(x + width - 30, y + 25, 6, 0, Math.PI * 2);
        ctx.fill();

        return { type: 'card', variant, bounds: { x, y, width, height } };
    }

    generateForm(canvas, ctx, variant = 'normal') {
        const x = 40, y = 40, width = 240, height = 160;

        // Form background
        ctx.fillStyle = '#f8f9fa';
        ctx.strokeStyle = '#dee2e6';
        ctx.lineWidth = 1;
        ctx.fillRect(x, y, width, height);
        ctx.strokeRect(x, y, width, height);

        // Form title
        ctx.fillStyle = '#212529';
        ctx.font = 'bold 16px Arial';
        ctx.fillText('Login Form', x + 15, y + 25);

        // Input fields
        const inputHeight = 30, inputWidth = width - 30;
        
        // Username field
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(x + 15, y + 40, inputWidth, inputHeight);
        ctx.strokeStyle = variant === 'error' ? '#dc3545' : '#ced4da';
        ctx.strokeRect(x + 15, y + 40, inputWidth, inputHeight);
        
        ctx.fillStyle = '#6c757d';
        ctx.font = '12px Arial';
        ctx.fillText('Username', x + 20, y + 57);

        // Password field
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(x + 15, y + 80, inputWidth, inputHeight);
        ctx.strokeStyle = variant === 'error' ? '#dc3545' : '#ced4da';
        ctx.strokeRect(x + 15, y + 80, inputWidth, inputHeight);
        
        ctx.fillStyle = '#6c757d';
        ctx.fillText('Password', x + 20, y + 97);

        // Submit button
        const btnColor = variant === 'error' ? '#dc3545' : '#007bff';
        ctx.fillStyle = btnColor;
        ctx.fillRect(x + 15, y + 120, 80, 25);
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        ctx.fillText('Login', x + 45, y + 135);

        return { type: 'form', variant, bounds: { x, y, width, height } };
    }

    generateNavigation(canvas, ctx, variant = 'normal') {
        const x = 0, y = 0, width = canvas.width, height = 60;

        // Navigation background
        ctx.fillStyle = variant === 'dark' ? '#212529' : '#f8f9fa';
        ctx.fillRect(x, y, width, height);
        
        // Navigation items
        const textColor = variant === 'dark' ? '#ffffff' : '#212529';
        ctx.fillStyle = textColor;
        ctx.font = 'bold 16px Arial';
        ctx.fillText('Brand', x + 20, y + 35);

        ctx.font = '14px Arial';
        ctx.fillText('Home', x + 120, y + 35);
        ctx.fillText('About', x + 170, y + 35);
        ctx.fillText('Contact', x + 220, y + 35);

        // Active indicator
        ctx.fillStyle = '#007bff';
        ctx.fillRect(x + 115, y + 45, 35, 3);

        return { type: 'navigation', variant, bounds: { x, y, width, height } };
    }

    generateChart(canvas, ctx, variant = 'bar') {
        const x = 50, y = 50, width = 220, height = 140;

        // Chart background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(x, y, width, height);
        ctx.strokeStyle = '#dee2e6';
        ctx.strokeRect(x, y, width, height);

        if (variant === 'bar') {
            // Bar chart
            const bars = [40, 80, 60, 100, 30];
            const barWidth = 25;
            const spacing = 10;
            
            bars.forEach((barHeight, i) => {
                const barX = x + 30 + i * (barWidth + spacing);
                const barY = y + height - 20 - barHeight;
                
                ctx.fillStyle = `hsl(${210 + i * 30}, 70%, 50%)`;
                ctx.fillRect(barX, barY, barWidth, barHeight);
            });
        } else if (variant === 'line') {
            // Line chart
            ctx.strokeStyle = '#007bff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            const points = [30, 60, 40, 90, 70, 50];
            points.forEach((point, i) => {
                const pointX = x + 20 + i * 35;
                const pointY = y + height - 20 - point;
                if (i === 0) ctx.moveTo(pointX, pointY);
                else ctx.lineTo(pointX, pointY);
            });
            ctx.stroke();

            // Data points
            ctx.fillStyle = '#007bff';
            points.forEach((point, i) => {
                const pointX = x + 20 + i * 35;
                const pointY = y + height - 20 - point;
                ctx.beginPath();
                ctx.arc(pointX, pointY, 3, 0, Math.PI * 2);
                ctx.fill();
            });
        }

        // Chart title
        ctx.fillStyle = '#212529';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Performance Metrics', x + width/2, y + 15);

        return { type: 'chart', variant, bounds: { x, y, width, height } };
    }

    // Generate different types of anomalies
    generateAnomalyVariant(baseImage, anomalyType) {
        const canvas = createCanvas(baseImage.width, baseImage.height);
        const ctx = canvas.getContext('2d');
        
        // Copy base image
        ctx.drawImage(baseImage, 0, 0);

        switch (anomalyType) {
            case 'color_shift':
                // Apply color filter
                ctx.globalCompositeOperation = 'multiply';
                ctx.fillStyle = 'rgba(255, 200, 200, 0.3)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.globalCompositeOperation = 'source-over';
                break;

            case 'missing_element':
                // Cover part of the image
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(50, 50, 100, 60);
                break;

            case 'extra_element':
                // Add unexpected element
                ctx.fillStyle = '#ff0000';
                ctx.font = 'bold 16px Arial';
                ctx.fillText('ERROR', 200, 100);
                break;

            case 'size_change':
                // Slightly stretch content
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.scale(1.1, 0.9);
                ctx.putImageData(imageData, 0, 0);
                break;

            case 'position_shift':
                // Move content slightly
                const originalData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.putImageData(originalData, 5, -3);
                break;

            case 'brightness_change':
                // Adjust brightness
                const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                for (let i = 0; i < imgData.data.length; i += 4) {
                    imgData.data[i] = Math.min(255, imgData.data[i] * 1.3);     // R
                    imgData.data[i + 1] = Math.min(255, imgData.data[i + 1] * 1.3); // G
                    imgData.data[i + 2] = Math.min(255, imgData.data[i + 2] * 1.3); // B
                }
                ctx.putImageData(imgData, 0, 0);
                break;
        }

        return canvas;
    }

    // Generate comprehensive test dataset
    async generateTestCases() {
        console.log('🎨 Generating diverse test image dataset...\n');

        const components = [
            { name: 'button', variants: ['primary', 'secondary', 'danger', 'warning'] },
            { name: 'card', variants: ['normal', 'error'] },
            { name: 'form', variants: ['normal', 'error'] },
            { name: 'navigation', variants: ['normal', 'dark'] },
            { name: 'chart', variants: ['bar', 'line'] }
        ];

        const anomalies = [
            'color_shift', 'missing_element', 'extra_element', 
            'size_change', 'position_shift', 'brightness_change'
        ];

        let caseId = 1;

        for (const component of components) {
            for (const variant of component.variants) {
                // Generate baseline image
                const canvas = createCanvas(320, 240);
                const ctx = canvas.getContext('2d');
                
                // White background
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Generate component
                let componentInfo;
                switch (component.name) {
                    case 'button':
                        componentInfo = this.generateButton(canvas, ctx, variant);
                        break;
                    case 'card':
                        componentInfo = this.generateCard(canvas, ctx, variant);
                        break;
                    case 'form':
                        componentInfo = this.generateForm(canvas, ctx, variant);
                        break;
                    case 'navigation':
                        componentInfo = this.generateNavigation(canvas, ctx, variant);
                        break;
                    case 'chart':
                        componentInfo = this.generateChart(canvas, ctx, variant);
                        break;
                }

                // Save baseline image
                const baselineFileName = `baseline_${component.name}_${variant}_${caseId}.png`;
                const baselinePath = path.join(this.imagesDir, baselineFileName);
                const baselineBuffer = canvas.toBuffer('image/png');
                fs.writeFileSync(baselinePath, baselineBuffer);

                // Generate anomaly variants
                for (const anomalyType of anomalies) {
                    const anomalyCanvas = this.generateAnomalyVariant(canvas, anomalyType);
                    const anomalyFileName = `anomaly_${component.name}_${variant}_${anomalyType}_${caseId}.png`;
                    const anomalyPath = path.join(this.imagesDir, anomalyFileName);
                    const anomalyBuffer = anomalyCanvas.toBuffer('image/png');
                    fs.writeFileSync(anomalyPath, anomalyBuffer);

                    // Create test case
                    const testCase = {
                        id: caseId,
                        component: component.name,
                        variant: variant,
                        anomaly_type: anomalyType,
                        baseline_image: baselineFileName,
                        candidate_image: anomalyFileName,
                        expected_anomaly: true,
                        component_info: componentInfo,
                        created_at: new Date().toISOString()
                    };

                    this.testCases.push(testCase);
                    caseId++;
                }

                // Also create a "no anomaly" test case (baseline vs baseline with slight noise)
                const noiseCanvas = createCanvas(canvas.width, canvas.height);
                const noiseCtx = noiseCanvas.getContext('2d');
                noiseCtx.drawImage(canvas, 0, 0);
                
                // Add very slight noise that shouldn't be detected as anomaly
                const imageData = noiseCtx.getImageData(0, 0, canvas.width, canvas.height);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    if (Math.random() < 0.001) { // Very sparse noise
                        const noise = (Math.random() - 0.5) * 10;
                        imageData.data[i] = Math.max(0, Math.min(255, imageData.data[i] + noise));
                        imageData.data[i + 1] = Math.max(0, Math.min(255, imageData.data[i + 1] + noise));
                        imageData.data[i + 2] = Math.max(0, Math.min(255, imageData.data[i + 2] + noise));
                    }
                }
                noiseCtx.putImageData(imageData, 0, 0);

                const noiseFileName = `noise_${component.name}_${variant}_${caseId}.png`;
                const noisePath = path.join(this.imagesDir, noiseFileName);
                const noiseBuffer = noiseCanvas.toBuffer('image/png');
                fs.writeFileSync(noisePath, noiseBuffer);

                const noAnomalyCase = {
                    id: caseId,
                    component: component.name,
                    variant: variant,
                    anomaly_type: 'none',
                    baseline_image: baselineFileName,
                    candidate_image: noiseFileName,
                    expected_anomaly: false,
                    component_info: componentInfo,
                    created_at: new Date().toISOString()
                };

                this.testCases.push(noAnomalyCase);
                caseId++;
            }
        }

        console.log(`✅ Generated ${this.testCases.length} test cases`);
        console.log(`📁 Images saved to: ${this.imagesDir}`);
        return this.testCases;
    }

    // Run ML analysis on all test cases
    async runMLAnalysis() {
        console.log('\n🤖 Running ML analysis on test dataset...\n');

        let processed = 0;
        const total = this.testCases.length;

        for (const testCase of this.testCases) {
            try {
                // Read images
                const baselinePath = path.join(this.imagesDir, testCase.baseline_image);
                const candidatePath = path.join(this.imagesDir, testCase.candidate_image);
                
                const baselineBuffer = fs.readFileSync(baselinePath);
                const candidateBuffer = fs.readFileSync(candidatePath);
                
                const baselineB64 = baselineBuffer.toString('base64');
                const candidateB64 = candidateBuffer.toString('base64');

                // Analyze with ML service
                const response = await fetch('http://localhost:8000/score', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        baseline_image: baselineB64,
                        candidate_image: candidateB64,
                        threshold: 0.1
                    })
                });

                if (!response.ok) {
                    throw new Error(`ML service error: ${response.statusText}`);
                }

                const analysis = await response.json();

                // Store result
                const result = {
                    ...testCase,
                    ml_analysis: analysis,
                    prediction: analysis.anomaly_score > 0.2 ? 'anomaly' : 'normal',
                    correct_prediction: (analysis.anomaly_score > 0.2) === testCase.expected_anomaly,
                    processing_time: Date.now()
                };

                this.results.push(result);
                processed++;

                if (processed % 10 === 0 || processed === total) {
                    console.log(`📊 Processed ${processed}/${total} test cases (${((processed/total)*100).toFixed(1)}%)`);
                }

            } catch (error) {
                console.error(`❌ Error processing test case ${testCase.id}:`, error.message);
            }
        }

        console.log(`✅ Analysis completed: ${this.results.length} results generated`);
    }

    // Generate training dataset and performance report
    generateReport() {
        console.log('\n📈 Generating performance report...\n');

        const stats = {
            total_cases: this.results.length,
            anomaly_cases: this.results.filter(r => r.expected_anomaly).length,
            normal_cases: this.results.filter(r => r.expected_anomaly === false).length,
            correct_predictions: this.results.filter(r => r.correct_prediction).length,
            false_positives: this.results.filter(r => !r.expected_anomaly && r.prediction === 'anomaly').length,
            false_negatives: this.results.filter(r => r.expected_anomaly && r.prediction === 'normal').length,
            accuracy: 0,
            precision: 0,
            recall: 0,
            f1_score: 0
        };

        if (stats.total_cases > 0) {
            stats.accuracy = stats.correct_predictions / stats.total_cases;
            
            const truePositives = this.results.filter(r => r.expected_anomaly && r.prediction === 'anomaly').length;
            stats.precision = stats.false_positives > 0 ? truePositives / (truePositives + stats.false_positives) : 1.0;
            stats.recall = stats.false_negatives > 0 ? truePositives / (truePositives + stats.false_negatives) : 1.0;
            stats.f1_score = (stats.precision + stats.recall) > 0 ? 2 * (stats.precision * stats.recall) / (stats.precision + stats.recall) : 0;
        }

        // Performance by component type
        const componentStats = {};
        for (const result of this.results) {
            if (!componentStats[result.component]) {
                componentStats[result.component] = {
                    total: 0,
                    correct: 0,
                    avg_similarity: 0,
                    avg_anomaly_score: 0
                };
            }
            
            componentStats[result.component].total++;
            if (result.correct_prediction) componentStats[result.component].correct++;
            componentStats[result.component].avg_similarity += result.ml_analysis.similarity_score;
            componentStats[result.component].avg_anomaly_score += result.ml_analysis.anomaly_score;
        }

        // Calculate averages
        for (const component in componentStats) {
            const cs = componentStats[component];
            cs.accuracy = cs.correct / cs.total;
            cs.avg_similarity /= cs.total;
            cs.avg_anomaly_score /= cs.total;
        }

        // Performance by anomaly type
        const anomalyStats = {};
        for (const result of this.results.filter(r => r.expected_anomaly)) {
            if (!anomalyStats[result.anomaly_type]) {
                anomalyStats[result.anomaly_type] = {
                    total: 0,
                    detected: 0,
                    avg_score: 0
                };
            }
            
            anomalyStats[result.anomaly_type].total++;
            if (result.prediction === 'anomaly') anomalyStats[result.anomaly_type].detected++;
            anomalyStats[result.anomaly_type].avg_score += result.ml_analysis.anomaly_score;
        }

        for (const anomaly in anomalyStats) {
            const as = anomalyStats[anomaly];
            as.detection_rate = as.detected / as.total;
            as.avg_score /= as.total;
        }

        // Save training dataset
        const trainingDataset = {
            metadata: {
                generated_at: new Date().toISOString(),
                total_samples: this.results.length,
                image_resolution: '320x240',
                components: Object.keys(componentStats),
                anomaly_types: Object.keys(anomalyStats)
            },
            statistics: stats,
            component_performance: componentStats,
            anomaly_performance: anomalyStats,
            samples: this.results
        };

        const datasetPath = path.join(this.resultsDir, 'training_dataset.json');
        fs.writeFileSync(datasetPath, JSON.stringify(trainingDataset, null, 2));

        // Generate report
        const report = `# Visual Anomaly Detection - Training Data Report

## 📊 Dataset Overview
- **Total Samples**: ${stats.total_cases}
- **Anomaly Cases**: ${stats.anomaly_cases}
- **Normal Cases**: ${stats.normal_cases}
- **Generated**: ${new Date().toLocaleString()}

## 🎯 Model Performance
- **Accuracy**: ${(stats.accuracy * 100).toFixed(2)}%
- **Precision**: ${(stats.precision * 100).toFixed(2)}%
- **Recall**: ${(stats.recall * 100).toFixed(2)}%
- **F1 Score**: ${(stats.f1_score * 100).toFixed(2)}%

### Confusion Matrix
- True Positives: ${this.results.filter(r => r.expected_anomaly && r.prediction === 'anomaly').length}
- False Positives: ${stats.false_positives}
- False Negatives: ${stats.false_negatives}
- True Negatives: ${this.results.filter(r => !r.expected_anomaly && r.prediction === 'normal').length}

## 🧩 Performance by Component Type

| Component | Samples | Accuracy | Avg Similarity | Avg Anomaly Score |
|-----------|---------|----------|----------------|-------------------|
${Object.entries(componentStats).map(([comp, stats]) => 
    `| ${comp} | ${stats.total} | ${(stats.accuracy * 100).toFixed(1)}% | ${stats.avg_similarity.toFixed(3)} | ${stats.avg_anomaly_score.toFixed(3)} |`
).join('\n')}

## 🔍 Anomaly Detection Performance

| Anomaly Type | Cases | Detection Rate | Avg Score |
|--------------|-------|----------------|-----------|
${Object.entries(anomalyStats).map(([anomaly, stats]) => 
    `| ${anomaly} | ${stats.total} | ${(stats.detection_rate * 100).toFixed(1)}% | ${stats.avg_score.toFixed(3)} |`
).join('\n')}

## 📁 Files Generated
- **Training Dataset**: \`${datasetPath}\`
- **Images Directory**: \`${this.imagesDir}\`
- **Total Images**: ${fs.readdirSync(this.imagesDir).length}

## 🚀 Next Steps
1. Use training dataset for model fine-tuning
2. Analyze misclassified cases for improvement
3. Expand dataset with real-world UI screenshots
4. Implement active learning for difficult cases
`;

        const reportPath = path.join(this.resultsDir, 'performance_report.md');
        fs.writeFileSync(reportPath, report);

        console.log('📋 PERFORMANCE SUMMARY');
        console.log('═══════════════════════');
        console.log(`📊 Total Samples: ${stats.total_cases}`);
        console.log(`🎯 Accuracy: ${(stats.accuracy * 100).toFixed(2)}%`);
        console.log(`🎪 Precision: ${(stats.precision * 100).toFixed(2)}%`);
        console.log(`📈 Recall: ${(stats.recall * 100).toFixed(2)}%`);
        console.log(`🏆 F1 Score: ${(stats.f1_score * 100).toFixed(2)}%`);
        console.log('');
        console.log('📁 FILES GENERATED:');
        console.log(`   • Training Dataset: ${datasetPath}`);
        console.log(`   • Performance Report: ${reportPath}`);
        console.log(`   • Images (${fs.readdirSync(this.imagesDir).length}): ${this.imagesDir}`);

        return { stats, componentStats, anomalyStats };
    }

    async run() {
        console.log('🎯 Visual Anomaly Detection - Training Data Generator');
        console.log('=====================================================\n');

        try {
            // Generate test cases and images
            await this.generateTestCases();
            
            // Run ML analysis
            await this.runMLAnalysis();
            
            // Generate report and dataset
            const results = this.generateReport();
            
            console.log('\n🎉 Training data generation completed successfully!');
            console.log('💡 Use the generated dataset for model training and evaluation.');
            
            return results;

        } catch (error) {
            console.error('❌ Training data generation failed:', error);
            throw error;
        }
    }
}

// Run the generator
if (require.main === module) {
    const generator = new TrainingDataGenerator();
    generator.run().then(() => {
        process.exit(0);
    }).catch(error => {
        console.error('Generation failed:', error);
        process.exit(1);
    });
}

module.exports = { TrainingDataGenerator };