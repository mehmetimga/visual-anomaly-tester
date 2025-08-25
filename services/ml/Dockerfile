FROM --platform=linux/arm64 python:3.10-slim

# System dependencies for OpenCV, ML, and basic functionality on ARM64
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Essential OpenCV dependencies for ARM64/headless operation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for headless operation
ENV OPENCV_IO_ENABLE_JASPER=1
ENV OPENCV_IO_ENABLE_OPENEXR=1

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download pre-trained models (simple mode)
RUN python download_models_simple.py

EXPOSE 8000
CMD ["uvicorn", "main_simple:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]