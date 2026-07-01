#!/usr/bin/env bash
# download-models.sh - Download ONNX models for Symthaea perception
#
# Downloads:
# - SigLIP (image embeddings)
# - Moondream (image captioning / VQA)
# - Qwen3-Embedding (text embeddings)

set -euo pipefail

MODEL_DIR="${SYMTHAEA_MODEL_DIR:-$HOME/.cache/symthaea/models}"
mkdir -p "$MODEL_DIR"

echo "==> Symthaea Model Downloader"
echo "    Target: $MODEL_DIR"
echo ""

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "ERROR: huggingface-cli not found. Install with: pip install huggingface-hub"
    exit 1
fi

echo "[1/3] Downloading SigLIP (image embeddings)..."
if [ -d "$MODEL_DIR/siglip" ]; then
    echo "      Already exists, skipping."
else
    huggingface-cli download google/siglip-so400m-patch14-384 \
        --local-dir "$MODEL_DIR/siglip" \
        --include "*.onnx" "*.json" "*.txt" "preprocessor_config.json"
    echo "      Done."
fi

echo ""
echo "[2/3] Downloading Moondream (image captioning)..."
if [ -d "$MODEL_DIR/moondream" ]; then
    echo "      Already exists, skipping."
else
    huggingface-cli download vikhyatk/moondream2 \
        --local-dir "$MODEL_DIR/moondream" \
        --include "*.onnx" "*.json" "*.txt"
    echo "      Done."
fi

echo ""
echo "[3/3] Downloading Qwen3-Embedding (text embeddings)..."
if [ -d "$MODEL_DIR/qwen3-embedding" ]; then
    echo "      Already exists, skipping."
else
    # Use the smaller 0.6B model for efficiency
    huggingface-cli download Alibaba-NLP/gte-Qwen2-1.5B-instruct \
        --local-dir "$MODEL_DIR/qwen3-embedding" \
        --include "*.onnx" "*.json" "*.txt" "tokenizer*"
    echo "      Done."
fi

echo ""
echo "==> All models downloaded to $MODEL_DIR"
echo ""
echo "Set environment variable to use custom path:"
echo "  export SYMTHAEA_MODEL_DIR=$MODEL_DIR"
