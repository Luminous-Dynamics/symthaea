#!/usr/bin/env bash
# Launch full MNIST IID experiment with GPU acceleration
# 100 rounds, 7 baselines, ~30-60 minutes expected on RTX 2070

set -e

echo "🚀 Launching Full MNIST IID Experiment on GPU"
echo "=============================================="
echo ""

# Check GPU is available
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "⚠️  WARNING: GPU not detected!"
    echo "This will run on CPU and take 8-16 hours."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    GPU_NAME=$(python -c 'import torch; print(torch.cuda.get_device_name(0))')
    echo "✓ GPU detected: $GPU_NAME"
    echo "✓ Expected runtime: 30-60 minutes"
fi

echo ""
echo "Experiment configuration:"
echo "  - Dataset: MNIST (60,000 training samples)"
echo "  - Clients: 10 (IID split)"
echo "  - Baselines: 7 (FedAvg, FedProx, SCAFFOLD, Krum, Multi-Krum, Bulyan, Median)"
echo "  - Rounds: 100"
echo "  - Evaluation: Every 10 rounds"
echo ""

read -p "Start experiment? (Y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    exit 0
fi

echo ""
echo "🏃 Starting experiment..."
echo ""

# Run with unbuffered output so we can see progress
python -u experiments/runner.py --config experiments/configs/mnist_iid.yaml

echo ""
echo "✅ Experiment complete!"
echo ""
echo "Results saved to: results/iid/"
echo ""
echo "To analyze results:"
echo "  python experiments/utils/analyze_results.py \\"
echo "    --results results/iid/mnist_iid_*.json \\"
echo "    --output results/analysis/mnist_iid/"
