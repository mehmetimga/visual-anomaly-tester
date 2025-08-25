"""
Advanced Visual Anomaly Detection ML Service
Incorporates computer vision, deep learning, and AI best practices.
"""

import base64
import io
import math
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import asyncio

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from skimage import filters, measure, feature
from scipy import ndimage
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Deep Learning Models
import timm
import lpips as lp
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
import easyocr

# Segmentation
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

# ML Pipeline
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from scipy.spatial.distance import cosine

# API Framework
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Visual Anomaly Detection API",
    description="Enterprise-grade visual regression testing with AI/ML",
    version="2.0.0"
)

# ============================================================================
# Configuration & Models
# ============================================================================

MODEL_REGISTRY = Path("/app/models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

class ModelManager:
    """Centralized model management with lazy loading."""
    
    def __init__(self):
        self._lpips = None
        self._vit_models = {}
        self._clip_model = None
        self._clip_processor = None
        self._yolo = None
        self._ui_detector = None
        self._sam = None
        self._ocr = None
        self._anomaly_detector = None
        
    def get_lpips(self):
        if self._lpips is None:
            self._lpips = lp.LPIPS(net='alex').to(DEVICE)
            self._lpips.eval()
        return self._lpips
    
    def get_vit(self, model_name="vit_base_patch16_224"):
        if model_name not in self._vit_models:
            self._vit_models[model_name] = timm.create_model(
                model_name, pretrained=True
            ).to(DEVICE).eval()
        return self._vit_models[model_name]
    
    def get_clip(self):
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model.eval()
        return self._clip_model, self._clip_processor
    
    def get_yolo(self):
        if self._yolo is None:
            model_path = MODEL_REGISTRY / "yolov8s.pt"
            if model_path.exists():
                self._yolo = YOLO(str(model_path))
            else:
                self._yolo = YOLO("yolov8s.pt")
        return self._yolo
    
    def get_ui_detector(self):
        if self._ui_detector is None:
            ui_model_path = MODEL_REGISTRY / "ui_components_yolo.pt"
            if ui_model_path.exists():
                self._ui_detector = YOLO(str(ui_model_path))
            else:
                self._ui_detector = self.get_yolo()  # Fallback
        return self._ui_detector
    
    def get_sam(self):
        if self._sam is None and SAM_AVAILABLE:
            sam_checkpoint = MODEL_REGISTRY / "sam_vit_h_4b8939.pth"
            if sam_checkpoint.exists():
                sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
                sam.to(device=DEVICE)
                self._sam = SamPredictor(sam)
        return self._sam
    
    def get_ocr(self):
        if self._ocr is None:
            self._ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        return self._ocr

models = ModelManager()

# ============================================================================
# Data Models
# ============================================================================

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class ScoreRequest(BaseModel):
    baseline_png_b64: str
    candidate_png_b64: str
    masks: List[BoundingBox] = Field(default_factory=list)
    
    # Metric weights (normalized)
    ssim_weight: float = 0.25
    lpips_weight: float = 0.25
    vit_weight: float = 0.20
    pixel_weight: float = 0.10
    texture_weight: float = 0.10
    semantic_weight: float = 0.10
    
    # Advanced options
    use_sam_segmentation: bool = False
    enable_texture_analysis: bool = True
    anomaly_threshold: float = 0.3
    confidence_threshold: float = 0.8

class TrainingRequest(BaseModel):
    images_b64: List[str]
    labels: List[int]  # 0: normal, 1: anomaly
    model_name: str = "anomaly_detector_v1"

# ============================================================================
# Image Processing Utilities
# ============================================================================

def b64_to_array(b64_str: str) -> np.ndarray:
    """Convert base64 string to OpenCV array."""
    try:
        img_data = base64.b64decode(b64_str)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def apply_masks(img: np.ndarray, masks: List[BoundingBox]) -> np.ndarray:
    """Apply masks by filling regions with median color."""
    if not masks:
        return img
    
    masked_img = img.copy()
    for mask in masks:
        x, y, w, h = mask.x, mask.y, mask.width, mask.height
        # Ensure bounds are valid
        x = max(0, min(x, img.shape[1] - 1))
        y = max(0, min(y, img.shape[0] - 1))
        w = max(1, min(w, img.shape[1] - x))
        h = max(1, min(h, img.shape[0] - y))
        
        # Fill with median color from surrounding area
        patch = img[max(0, y-10):y+h+10, max(0, x-10):x+w+10]
        if patch.size > 0:
            median_color = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
            masked_img[y:y+h, x:x+w] = median_color
    
    return masked_img

def enhance_image_quality(img: np.ndarray) -> np.ndarray:
    """Apply image enhancement for better analysis."""
    # Convert to PIL for enhancement
    pil_img = Image.fromarray(img)
    
    # Apply enhancement chain
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.1)
    
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.05)
    
    return np.array(pil_img)

# ============================================================================
# Advanced Feature Extraction
# ============================================================================

def extract_texture_features(img: np.ndarray) -> Dict[str, float]:
    """Extract texture features using LBP and Gabor filters."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Local Binary Pattern
    lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
    
    # Gabor filters
    gabor_responses = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        real, _ = gabor(gray, frequency=0.1, theta=theta)
        gabor_responses.append(np.mean(np.abs(real)))
    
    return {
        "lbp_uniformity": float(np.sum(lbp_hist[:10])),  # First 10 bins are uniform
        "lbp_entropy": float(-np.sum(lbp_hist * np.log(lbp_hist + 1e-10))),
        "gabor_energy": float(np.mean(gabor_responses)),
        "gabor_variance": float(np.var(gabor_responses))
    }

def compute_perceptual_hash(img: np.ndarray, hash_size: int = 16) -> str:
    """Compute perceptual hash for duplicate detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    
    # Compute differences
    diff = resized[:, 1:] > resized[:, :-1]
    return ''.join(str(int(x)) for x in diff.flatten())

# ============================================================================
# ML-Based Scoring
# ============================================================================

def compute_ssim_enhanced(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Enhanced SSIM computation with multi-scale analysis."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Standard SSIM
    ssim_val, ssim_map = ssim(gray1, gray2, data_range=255, full=True)
    
    # Multi-scale SSIM
    scales = [1.0, 0.5, 0.25]
    ms_ssim_values = []
    
    for scale in scales:
        if scale != 1.0:
            h, w = int(gray1.shape[0] * scale), int(gray1.shape[1] * scale)
            s1 = cv2.resize(gray1, (w, h))
            s2 = cv2.resize(gray2, (w, h))
        else:
            s1, s2 = gray1, gray2
        
        if min(s1.shape) >= 7:  # Minimum size for SSIM
            ms_val = ssim(s1, s2, data_range=255)
            ms_ssim_values.append(ms_val)
    
    return {
        "ssim": float(ssim_val),
        "ssim_variance": float(np.var(ssim_map)),
        "ms_ssim": float(np.mean(ms_ssim_values)) if ms_ssim_values else float(ssim_val)
    }

def compute_lpips_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute LPIPS perceptual distance."""
    lpips_model = models.get_lpips()
    
    def preprocess(img):
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        )
        return img_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
    
    with torch.no_grad():
        tensor1 = preprocess(img1).to(DEVICE)
        tensor2 = preprocess(img2).to(DEVICE)
        distance = lpips_model(tensor1, tensor2)
        return float(distance.cpu().item())

def compute_vit_features(img: np.ndarray, model_name: str = "vit_base_patch16_224") -> np.ndarray:
    """Extract ViT features for semantic comparison."""
    vit_model = models.get_vit(model_name)
    
    # Preprocessing
    transform = timm.data.resolve_data_config({}, model=vit_model)
    transform = timm.data.create_transform(**transform)
    
    pil_img = Image.fromarray(img)
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        features = vit_model.forward_features(img_tensor)
        # Global average pooling
        features = features.mean(dim=1)  # [1, hidden_dim]
        features = F.normalize(features, p=2, dim=1)
    
    return features.cpu().numpy().flatten()

def compute_semantic_similarity(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Compute semantic similarity using CLIP."""
    clip_model, clip_processor = models.get_clip()
    
    pil_img1 = Image.fromarray(img1)
    pil_img2 = Image.fromarray(img2)
    
    inputs = clip_processor(images=[pil_img1, pil_img2], return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        image_features = F.normalize(image_features, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            image_features[0:1], image_features[1:2], dim=1
        )
    
    return {
        "clip_similarity": float(similarity.cpu().item()),
        "clip_distance": float(1.0 - similarity.cpu().item())
    }

# ============================================================================
# Object Detection & Segmentation
# ============================================================================

def detect_ui_components(img: np.ndarray) -> List[Dict]:
    """Detect UI components using fine-tuned YOLO."""
    ui_detector = models.get_ui_detector()
    
    results = ui_detector.predict(source=img, verbose=False)
    detections = []
    
    for result in results:
        if result.boxes is not None:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if conf > 0.5:  # Confidence threshold
                    detections.append({
                        "bbox": [float(x) for x in box],
                        "confidence": float(conf),
                        "class_id": int(cls),
                        "class_name": result.names[int(cls)],
                        "type": "ui_component"
                    })
    
    return detections

def segment_differences(img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
    """Use SAM to segment difference regions."""
    if not SAM_AVAILABLE:
        return None
        
    sam_predictor = models.get_sam()
    if sam_predictor is None:
        return None
    
    # Compute difference map
    diff = cv2.absdiff(
        cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    )
    
    # Find significant difference regions
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Use largest contour as prompt for SAM
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 100:  # Skip small differences
        return None
    
    # Get centroid as point prompt
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
        
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Run SAM segmentation
    sam_predictor.set_image(img2)
    masks, scores, _ = sam_predictor.predict(
        point_coords=np.array([[cx, cy]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    
    # Return best mask
    if len(masks) > 0:
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx].astype(np.uint8) * 255
    
    return None

# ============================================================================
# Anomaly Detection Pipeline
# ============================================================================

def extract_comprehensive_features(img1: np.ndarray, img2: np.ndarray, 
                                   masks: List[BoundingBox]) -> Dict[str, float]:
    """Extract comprehensive feature vector for ML-based anomaly detection."""
    # Apply masks
    img1_masked = apply_masks(img1, masks)
    img2_masked = apply_masks(img2, masks)
    
    features = {}
    
    # 1. SSIM metrics
    ssim_metrics = compute_ssim_enhanced(img1_masked, img2_masked)
    features.update(ssim_metrics)
    
    # 2. Pixel-level metrics
    gray1 = cv2.cvtColor(img1_masked, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_masked, cv2.COLOR_RGB2GRAY)
    
    pixel_diff = cv2.absdiff(gray1, gray2)
    features.update({
        "pixel_diff_mean": float(np.mean(pixel_diff)),
        "pixel_diff_std": float(np.std(pixel_diff)),
        "pixel_diff_max": float(np.max(pixel_diff)),
        "changed_pixels_ratio": float(np.sum(pixel_diff > 10) / pixel_diff.size)
    })
    
    # 3. Perceptual metrics
    features["lpips"] = compute_lpips_score(img1_masked, img2_masked)
    
    # 4. Semantic similarity
    semantic_metrics = compute_semantic_similarity(img1_masked, img2_masked)
    features.update(semantic_metrics)
    
    # 5. Texture analysis
    texture1 = extract_texture_features(img1_masked)
    texture2 = extract_texture_features(img2_masked)
    
    for key in texture1:
        features[f"texture1_{key}"] = texture1[key]
        features[f"texture2_{key}"] = texture2[key]
        features[f"texture_diff_{key}"] = abs(texture1[key] - texture2[key])
    
    # 6. ViT features distance
    vit_feat1 = compute_vit_features(img1_masked)
    vit_feat2 = compute_vit_features(img2_masked)
    features["vit_cosine_distance"] = float(cosine(vit_feat1, vit_feat2))
    
    # 7. Perceptual hashes
    hash1 = compute_perceptual_hash(img1_masked)
    hash2 = compute_perceptual_hash(img2_masked)
    hamming_dist = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    features["perceptual_hash_distance"] = float(hamming_dist / len(hash1))
    
    return features

# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/score")
async def score_images(request: ScoreRequest):
    """Advanced visual anomaly scoring with ML pipeline."""
    try:
        # Decode images
        baseline_img = b64_to_array(request.baseline_png_b64)
        candidate_img = b64_to_array(request.candidate_png_b64)
        
        # Ensure same dimensions
        if candidate_img.shape[:2] != baseline_img.shape[:2]:
            candidate_img = cv2.resize(
                candidate_img, 
                (baseline_img.shape[1], baseline_img.shape[0]), 
                interpolation=cv2.INTER_AREA
            )
        
        # Enhance image quality
        baseline_img = enhance_image_quality(baseline_img)
        candidate_img = enhance_image_quality(candidate_img)
        
        # Extract comprehensive features
        features = extract_comprehensive_features(baseline_img, candidate_img, request.masks)
        
        # Compute weighted anomaly score
        anomaly_score = (
            request.ssim_weight * (1.0 - features.get("ssim", 1.0)) +
            request.lpips_weight * features.get("lpips", 0.0) +
            request.vit_weight * features.get("vit_cosine_distance", 0.0) +
            request.pixel_weight * features.get("changed_pixels_ratio", 0.0) +
            request.texture_weight * features.get("texture_diff_lbp_entropy", 0.0) +
            request.semantic_weight * features.get("clip_distance", 0.0)
        )
        
        # Generate diff heatmap
        gray1 = cv2.cvtColor(baseline_img, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(candidate_img, cv2.COLOR_RGB2GRAY)
        diff_map = cv2.absdiff(gray1, gray2)
        diff_map = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
        
        # Overlay on original
        overlay = cv2.addWeighted(baseline_img, 0.6, heatmap, 0.4, 0)
        _, heatmap_buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        heatmap_b64 = base64.b64encode(heatmap_buffer.tobytes()).decode('ascii')
        
        # Object detection
        ui_detections = detect_ui_components(candidate_img)
        
        # OCR analysis  
        ocr_reader = models.get_ocr()
        ocr_results = []
        try:
            ocr_data = ocr_reader.readtext(candidate_img)
            for (bbox, text, conf) in ocr_data:
                if conf > 0.5:
                    ocr_results.append({
                        "bbox": [[float(x), float(y)] for x, y in bbox],
                        "text": text,
                        "confidence": float(conf)
                    })
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
        
        # SAM segmentation (optional)
        sam_mask = None
        if request.use_sam_segmentation:
            sam_mask = segment_differences(baseline_img, candidate_img)
            if sam_mask is not None:
                _, mask_buffer = cv2.imencode('.png', sam_mask)
                sam_mask = base64.b64encode(mask_buffer.tobytes()).decode('ascii')
        
        # CLIP embeddings for similarity search
        clip_model, clip_processor = models.get_clip()
        pil_candidate = Image.fromarray(candidate_img)
        inputs = clip_processor(images=pil_candidate, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            clip_features = clip_model.get_image_features(**inputs)
            clip_features = F.normalize(clip_features, p=2, dim=1)
            clip_embedding = clip_features.cpu().numpy().flatten().tolist()
        
        # Build response
        response = {
            "metrics": {
                "anomaly_score": float(anomaly_score),
                "is_anomaly": anomaly_score >= request.anomaly_threshold,
                "confidence": float(1.0 - min(anomaly_score / request.anomaly_threshold, 1.0)),
                **{k: float(v) for k, v in features.items()}
            },
            "analysis": {
                "ui_components": ui_detections,
                "ocr_results": ocr_results,
                "feature_count": len(features)
            },
            "embeddings": {
                "clip_512": clip_embedding
            },
            "artifacts": {
                "heatmap_png_b64": heatmap_b64,
                "sam_mask_b64": sam_mask
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Scoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@app.post("/train_anomaly_detector")
async def train_anomaly_detector(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train ML model for anomaly detection using labeled data."""
    try:
        # Start MLflow run
        with mlflow.start_run(run_name=f"training_{request.model_name}"):
            # Process training images
            features_list = []
            labels = np.array(request.labels)
            
            logger.info(f"Processing {len(request.images_b64)} training images...")
            
            for i, img_b64 in enumerate(request.images_b64):
                img = b64_to_array(img_b64)
                # For training, we compare against a reference (first image)
                if i == 0:
                    reference_img = img
                    continue
                
                features = extract_comprehensive_features(reference_img, img, [])
                features_list.append(list(features.values()))
            
            if len(features_list) < 2:
                raise HTTPException(status_code=400, detail="Need at least 2 images for training")
            
            # Prepare training data
            X = np.array(features_list)
            y = labels[1:]  # Skip first image (reference)
            
            # Feature scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Isolation Forest for anomaly detection
            clf = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            clf.fit(X_scaled)
            
            # Evaluate
            predictions = clf.predict(X_scaled)
            predictions_binary = (predictions == -1).astype(int)  # -1 is anomaly
            
            # Log metrics
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("contamination", 0.1)
            
            if len(np.unique(y)) > 1:  # If we have both classes
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                accuracy = accuracy_score(y, predictions_binary)
                precision = precision_score(y, predictions_binary, zero_division=0)
                recall = recall_score(y, predictions_binary, zero_division=0)
                
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
            
            # Save model
            model_path = MODEL_REGISTRY / f"{request.model_name}.pkl"
            import joblib
            joblib.dump({
                'classifier': clf,
                'scaler': scaler,
                'feature_names': list(features.keys())
            }, model_path)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(clf, "anomaly_detector")
            
            return {
                "status": "training_completed",
                "model_path": str(model_path),
                "n_samples": len(X),
                "n_features": X.shape[1],
                "mlflow_run_id": mlflow.active_run().info.run_id
            }
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": {
            "lpips": models._lpips is not None,
            "clip": models._clip_model is not None,
            "yolo": models._yolo is not None,
            "sam": models._sam is not None,
            "ocr": models._ocr is not None
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)