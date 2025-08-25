#!/usr/bin/env python3
"""Download and cache pre-trained models for visual anomaly detection."""

import os
import torch
import timm
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from segment_anything import sam_model_registry, SamPredictor
import urllib.request
from pathlib import Path

def download_models():
    """Download all required pre-trained models."""
    
    model_dir = Path("/app/models")
    model_dir.mkdir(exist_ok=True)
    
    print("Downloading pre-trained models...")
    
    # 1. YOLO for object detection
    print("📦 YOLO models...")
    yolo_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    for model in yolo_models:
        if not (model_dir / model).exists():
            YOLO(model).save(str(model_dir / model))
    
    # 2. Vision Transformers for feature extraction
    print("🤖 Vision Transformer models...")
    vit_models = [
        "vit_base_patch16_224",
        "vit_large_patch16_224", 
        "deit3_base_patch16_224",
        "swin_base_patch4_window7_224"
    ]
    for model_name in vit_models:
        print(f"  Loading {model_name}...")
        model = timm.create_model(model_name, pretrained=True)
        torch.save(model.state_dict(), model_dir / f"{model_name}.pth")
    
    # 3. CLIP models for semantic embeddings
    print("🎯 CLIP models...")
    clip_models = [
        "clip-ViT-B-32",
        "clip-ViT-L-14",
        "all-MiniLM-L6-v2"  # Lightweight alternative
    ]
    for model_name in clip_models:
        print(f"  Loading {model_name}...")
        model = SentenceTransformer(model_name)
        model.save(str(model_dir / model_name.replace("/", "_")))
    
    # 4. Segment Anything Model (SAM)
    print("✂️  SAM model...")
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam_url = f"https://dl.fbaipublicfiles.com/segment_anything/{sam_checkpoint}"
    sam_path = model_dir / sam_checkpoint
    
    if not sam_path.exists():
        print(f"  Downloading {sam_checkpoint}...")
        urllib.request.urlretrieve(sam_url, sam_path)
        
    # 5. Custom fine-tuned UI component detector (placeholder)
    print("🎨 UI Component detector...")
    ui_model_path = model_dir / "ui_components_yolo.pt"
    if not ui_model_path.exists():
        # For now, use base YOLO - replace with fine-tuned model later
        YOLO("yolov8s.pt").save(str(ui_model_path))
    
    print("✅ All models downloaded successfully!")

if __name__ == "__main__":
    download_models()