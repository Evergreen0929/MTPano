#!/bin/bash

set -e
echo "==========================================="
echo "       Environment Setup for MTPano"
echo "==========================================="

echo "[1/3] Installing dependencies..."
pip install tqdm Pillow easydict pyyaml imageio scikit-image tensorboard wandb open3d
pip install opencv-python==4.10.0.84
pip install transformers==4.57.6 timm==1.0.24 huggingface_hub==0.34.0 matplotlib==3.9.2 numpy==1.26.4 einops==0.8.0

echo "✅ Dependencies installed successfully."

# 3. Hugging Face 登录
echo "==========================================="
echo "[2/3] Hugging Face Login"
echo "Prepare Access Token (Write Access Recommended)"
echo "-------------------------------------------"
huggingface-cli login

# 4. WandB 登录
echo "==========================================="
echo "[3/3] Weights & Biases (WandB) Login, if you don't need it, Ctrl+C to skip."
echo "Prepare API Key"
echo "-------------------------------------------"
wandb login

echo "==========================================="
echo "🎉 Environment Setup Completed!"
echo "==========================================="