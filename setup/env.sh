#!/bin/bash

# PiAnnotate - Environment Setup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_DIR="$PROJECT_ROOT/web"
cd "$PROJECT_ROOT"
ENV_NAME="piannotate"

echo "🎹 PiAnnotate Setup"
echo "   Project path: $PROJECT_ROOT"
echo "   Web path: $WEB_DIR"
echo ""

# ============================================
# 1. Python Environment Setup
# ============================================
echo "📦 Setting up Python environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "   Environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "   Creating conda environment '$ENV_NAME'..."
conda create -n $ENV_NAME python=3.10 -y
fi

# Activate environment and install packages
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install Python packages
echo "   Installing Python packages..."
pip install orjson tqdm flask numpy msgpack gdown mido

# Install PyTorch for model training (CUDA 11.8)
echo "   Installing PyTorch for model training..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install optional packages for training
echo "   Installing optional training packages..."
pip install tensorboard

echo "✅ Python environment ready!"
echo ""

# ============================================
# 2. Node.js Dependencies Setup
# ============================================
echo "📦 Setting up Node.js dependencies..."

# Check if node is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js first."
    exit 1
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Please install npm first."
    exit 1
fi

echo "   Node.js version: $(node --version)"
echo "   npm version: $(npm --version)"

# Install npm packages
cd "$WEB_DIR"
if [ -f "package.json" ]; then
    echo "   Installing npm packages..."
    npm install --silent
    echo "✅ Node.js dependencies ready!"
else
    echo "❌ package.json not found in $WEB_DIR"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 Setup complete!"
echo ""
echo "   To run the visualizer:"
echo "   bash start.sh"
echo ""
echo "=========================================="
