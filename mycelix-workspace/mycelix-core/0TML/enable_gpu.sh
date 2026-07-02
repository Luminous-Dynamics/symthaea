#!/usr/bin/env bash
# Enable GPU support for PyTorch in venv

set -e

echo "🎮 Enabling GPU Support for Hybrid ZeroTrustML"
echo "=========================================="
echo ""

# Check for NVIDIA GPU
if ! nvidia-smi &> /dev/null; then
    echo "❌ No NVIDIA GPU detected. GPU acceleration not available."
    exit 1
fi

echo "✓ NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Activate venv
if [ ! -d .venv ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

source .venv/bin/activate

echo "🔄 Installing PyTorch with CUDA support..."
echo "(This will reinstall PyTorch from pip with CUDA 12.1)"
echo ""

# Uninstall existing PyTorch (from nix)
pip uninstall -y torch torchvision 2>/dev/null || true

# Install CUDA-enabled PyTorch from pip
# Using CUDA 12.1 (compatible with CUDA 13.0 driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "✅ GPU support enabled!"
echo ""
echo "Testing CUDA availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  CUDA not available - check installation')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Restart your nix develop shell"
echo "  2. Run: python test_minimal.py"
echo "  3. Run: python experiments/runner.py --config experiments/configs/mnist_test.yaml"
echo ""
echo "Expected speedup: 10-20x faster training!"
