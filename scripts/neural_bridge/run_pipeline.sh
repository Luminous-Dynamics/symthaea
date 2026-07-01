#!/usr/bin/env bash
set -euo pipefail

# Neural Bridge: Complete Training Pipeline
# ==========================================
#
# This script runs the complete neural bridge training pipeline:
# 1. Extract activations from LLM
# 2. Train hyperdimensional probe
# 3. Verify Rust integration
#
# Usage:
#   ./run_pipeline.sh                     # Use defaults
#   ./run_pipeline.sh --model gemma-2b    # Specify model
#   ./run_pipeline.sh --simulate          # Use simulated activations (no GPU)

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL="google/gemma-2b"
LAYER=12
EPOCHS=100
LR=0.01
SIMULATE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --simulate)
            SIMULATE=true
            shift
            ;;
        -h|--help)
            echo "Neural Bridge Training Pipeline"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL     HuggingFace model name (default: google/gemma-2b)"
            echo "  --layer N         Layer to extract from (default: 12)"
            echo "  --epochs N        Training epochs (default: 100)"
            echo "  --lr RATE         Learning rate (default: 0.01)"
            echo "  --simulate        Use simulated activations (no GPU needed)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}"
echo "═══════════════════════════════════════════════════════════"
echo "  Neural Bridge: LLM -> HDC Direct Projection Training"
echo "═══════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Layer: $LAYER"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Simulate: $SIMULATE"
echo ""

# Check Python environment
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Check for numpy
if ! python -c "import numpy" 2>/dev/null; then
    echo -e "${RED}Error: numpy not installed${NC}"
    echo "Install with: pip install numpy"
    exit 1
fi

# Step 1: Extract activations
echo -e "${YELLOW}Step 1: Extracting LLM activations...${NC}"
echo ""

if [ "$SIMULATE" = true ]; then
    echo "  (Using simulated activations - no actual LLM inference)"
fi

python scripts/neural_bridge/activation_extractor.py \
    --model "$MODEL" \
    --layer "$LAYER" \
    --output "data/neural_bridge/concept_activations.npz"

echo ""

# Step 2: Train probe
echo -e "${YELLOW}Step 2: Training hyperdimensional probe...${NC}"
echo ""

DEVICE="cpu"
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="cuda"
    echo "  (Using CUDA)"
else
    echo "  (Using CPU)"
fi

python scripts/neural_bridge/train_probe.py \
    --activations "data/neural_bridge/concept_activations.npz" \
    --output "models/neural_bridge/probe_weights.npy" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --device "$DEVICE"

echo ""

# Step 3: Verify Rust integration
echo -e "${YELLOW}Step 3: Verifying Rust integration...${NC}"
echo ""

if cargo test -p symthaea --lib perception::neural_bridge 2>/dev/null; then
    echo -e "${GREEN}  Rust tests passed!${NC}"
else
    echo -e "${YELLOW}  Rust tests not run (compilation may be needed)${NC}"
    echo "  Run: cargo test -p symthaea --lib perception::neural_bridge"
fi

echo ""

# Summary
echo -e "${GREEN}"
echo "═══════════════════════════════════════════════════════════"
echo "  Neural Bridge Training Complete!"
echo "═══════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""
echo "Files created:"
echo "  - data/neural_bridge/concept_activations.npz"
echo "  - models/neural_bridge/probe_weights.npy"
echo "  - models/neural_bridge/probe_weights.json"
echo ""
echo "Usage in Rust:"
echo '  let bridge = NeuralBridge::load("models/neural_bridge/probe_weights.npy")?;'
echo '  let hdc = bridge.project_to_packed(&activation)?;'
echo ""
echo "Next steps:"
echo "  1. Add more concepts to concept_dataset.py"
echo "  2. Use real LLM activations (remove --simulate)"
echo "  3. Integrate with Symthaea's consciousness processing"
echo ""
