#!/usr/bin/env python3
"""
Simple ML Service for Visual Anomaly Testing
Headless version using PIL and scikit-image only (no OpenCV)
"""

import io
import base64
from pathlib import Path
from typing import List, Dict, Optional
import asyncio

import numpy as np
from PIL import Image, ImageEnhance
from skimage import filters, measure, feature
from scipy import ndimage
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Visual Anomaly ML Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (loaded on startup)
models = {}

class AnalysisRequest(BaseModel):
    baseline_image: str  # base64 encoded
    candidate_image: str  # base64 encoded
    threshold: float = 0.1
    mask_regions: Optional[List[Dict]] = None

class AnalysisResponse(BaseModel):
    similarity_score: float
    anomaly_score: float
    features: Dict
    status: str
    differences_found: int
    recommendations: List[str]

def decode_image(image_b64: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    try:
        img_data = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

def calculate_image_features(img: np.ndarray) -> Dict:
    """Calculate basic image features using PIL and scikit-image"""
    # Convert to grayscale for feature extraction
    if len(img.shape) == 3:
        gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img
        
    features = {}
    
    # Basic statistics
    features['mean_brightness'] = float(np.mean(gray))
    features['std_brightness'] = float(np.std(gray))
    features['min_brightness'] = float(np.min(gray))
    features['max_brightness'] = float(np.max(gray))
    
    # Histogram features
    hist, _ = np.histogram(gray, bins=32, range=(0, 255))
    features['histogram_entropy'] = float(-np.sum(hist * np.log(hist + 1e-10)))
    
    # Edge detection using Sobel filter
    edges = filters.sobel(gray)
    features['edge_density'] = float(np.mean(edges))
    
    # Texture features
    features['contrast'] = float(np.std(gray))
    
    return features

def calculate_pixel_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate pixel-level similarity"""
    # Ensure images are same size
    if img1.shape != img2.shape:
        # Resize to match smaller image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_h, target_w = min(h1, h2), min(w1, w2)
        
        img1_pil = Image.fromarray(img1)
        img2_pil = Image.fromarray(img2)
        
        img1 = np.array(img1_pil.resize((target_w, target_h)))
        img2 = np.array(img2_pil.resize((target_w, target_h)))
    
    # Convert to grayscale
    if len(img1.shape) == 3:
        gray1 = np.dot(img1[...,:3], [0.2989, 0.5870, 0.1140])
        gray2 = np.dot(img2[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        gray1, gray2 = img1, img2
    
    # Calculate MSE and SSIM approximation
    mse = np.mean((gray1 - gray2) ** 2)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse + 1e-10))
    
    # Normalize to 0-1 similarity score
    similarity = min(1.0, psnr / 40.0)  # PSNR of 40+ is very good
    
    return float(similarity)

def detect_anomalies(features1: Dict, features2: Dict) -> Dict:
    """Simple anomaly detection based on feature differences"""
    anomalies = {}
    threshold = 0.2  # 20% difference threshold
    
    for key in features1:
        if key in features2:
            val1, val2 = features1[key], features2[key]
            if val1 != 0:  # Avoid division by zero
                diff = abs(val1 - val2) / abs(val1)
                if diff > threshold:
                    anomalies[key] = {
                        'baseline': val1,
                        'candidate': val2,
                        'difference_ratio': diff
                    }
    
    return anomalies

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    try:
        print("🤖 Starting Simple ML Service (Headless Mode)")
        print("📊 Basic image processing with PIL and scikit-image")
        models['initialized'] = True
        print("✅ Service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        models['error'] = str(e)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Visual Anomaly ML Service (Simple)",
        "models_loaded": models.get('initialized', False),
        "features": [
            "Basic image similarity",
            "Feature-based anomaly detection", 
            "Pixel-level comparison",
            "Statistical analysis"
        ]
    }

@app.post("/score", response_model=AnalysisResponse)
async def analyze_images(request: AnalysisRequest):
    """Analyze two images for visual anomalies"""
    try:
        if not models.get('initialized'):
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # Decode images
        baseline_img = decode_image(request.baseline_image)
        candidate_img = decode_image(request.candidate_image)
        
        # Calculate features
        baseline_features = calculate_image_features(baseline_img)
        candidate_features = calculate_image_features(candidate_img)
        
        # Calculate similarity
        pixel_similarity = calculate_pixel_similarity(baseline_img, candidate_img)
        
        # Detect anomalies
        anomalies = detect_anomalies(baseline_features, candidate_features)
        anomaly_score = len(anomalies) / max(len(baseline_features), 1)
        
        # Generate recommendations
        recommendations = []
        if pixel_similarity < 0.8:
            recommendations.append("Significant pixel-level differences detected")
        if anomaly_score > 0.3:
            recommendations.append("Multiple feature anomalies found")
        if not recommendations:
            recommendations.append("Images appear visually similar")
        
        return AnalysisResponse(
            similarity_score=pixel_similarity,
            anomaly_score=anomaly_score,
            features={
                "baseline": baseline_features,
                "candidate": candidate_features,
                "anomalies": anomalies
            },
            status="completed",
            differences_found=len(anomalies),
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/train_anomaly_detector")
async def train_detector(background_tasks: BackgroundTasks):
    """Placeholder for training endpoint"""
    return {
        "status": "success",
        "message": "Training not implemented in simple mode",
        "model_version": "simple_v1.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)