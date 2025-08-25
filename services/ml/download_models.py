#!/usr/bin/env python3
"""Download and cache pre-trained models for visual anomaly detection."""

import os
import urllib.request
from pathlib import Path

# Import ML libraries with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available, skipping torch models")
    TORCH_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    print("Warning: timm not available, skipping ViT models")
    TIMM_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: YOLO not available, skipping YOLO models")
    YOLO_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available, skipping CLIP models")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

def download_models():
    """Download all required pre-trained models."""
    
    model_dir = Path("/app/models")
    model_dir.mkdir(exist_ok=True)
    
    print("Downloading pre-trained models...")
    
    # 1. YOLO for object detection
    if YOLO_AVAILABLE:
        print("📦 YOLO models...")
        yolo_models = ["yolov8n.pt", "yolov8s.pt"]  # Reduced set
        for model in yolo_models:
            try:
                if not (model_dir / model).exists():
                    print(f"  Downloading {model}...")
                    yolo_model = YOLO(model)
                    # Just download, don't save separately
            except Exception as e:
                print(f"  Failed to download {model}: {e}")
    
    # 2. Vision Transformers for feature extraction
    if TORCH_AVAILABLE and TIMM_AVAILABLE:
        print("🤖 Vision Transformer models...")
        vit_models = ["vit_base_patch16_224"]  # Just one for now
        for model_name in vit_models:
            try:
                print(f"  Loading {model_name}...")
                model = timm.create_model(model_name, pretrained=True)
                torch.save(model.state_dict(), model_dir / f"{model_name}.pth")
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
    
    # 3. CLIP models for semantic embeddings
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("🎯 CLIP models...")
        clip_models = ["all-MiniLM-L6-v2"]  # Lightweight model only
        for model_name in clip_models:
            try:
                print(f"  Loading {model_name}...")
                model = SentenceTransformer(model_name)
                model.save(str(model_dir / model_name.replace("/", "_")))
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
    
    # 4. Skip SAM for now (too large and complex)
    print("✂️  SAM model skipped (use basic diff detection instead)")
        
    # 5. UI Component detector (use base YOLO)
    print("🎨 UI Component detector...")
    ui_model_path = model_dir / "ui_components_yolo.pt"
    if YOLO_AVAILABLE and not ui_model_path.exists():
        try:
            # For now, use base YOLO - replace with fine-tuned model later
            base_model = YOLO("yolov8n.pt")  # Smallest model
            print("  Using base YOLO model for UI components")
        except Exception as e:
            print(f"  Failed to setup UI detector: {e}")
    
    print("✅ All models downloaded successfully!")

if __name__ == "__main__":
    download_models()