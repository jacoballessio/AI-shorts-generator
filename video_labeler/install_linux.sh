#!/bin/bash
# Installation script for Video Labeler with SmolVLM (Linux/Mac)

echo "========================================"
echo "Video Labeler SmolVLM Installation"
echo "========================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found!"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

echo "Detected Python version:"
python3 --version
echo ""

# Ask about GPU
read -p "Do you have an NVIDIA GPU with CUDA? (y/n): " has_gpu

if [[ $has_gpu == "y" || $has_gpu == "Y" ]]; then
    echo ""
    echo "Installing PyTorch with CUDA support..."
    echo "This will download ~2GB"
    echo ""
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo ""
    echo "Installing PyTorch (CPU only)..."
    echo "Note: This will be slower than GPU!"
    echo ""
    pip3 install torch torchvision
fi

echo ""
echo "Installing Transformers and dependencies..."
pip3 install transformers accelerate pillow

echo ""
echo "Installing MoviePy for video processing..."
pip3 install moviepy

echo ""
echo "Installing Whisper for transcription..."
pip3 install openai-whisper

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Testing installation..."
echo ""

python3 test_smolvlm.py

echo ""
echo "If all tests passed, you're ready to go!"
echo "See examples.py for usage examples."
echo "See SMOLVLM_SETUP.md for optimization tips."
echo ""
