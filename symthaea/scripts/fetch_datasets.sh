#!/usr/bin/env bash
# Fetch public benchmark datasets for Symthaea external validation.
# Usage: ./scripts/fetch_datasets.sh [dataset_name]
# Without arguments, fetches all missing datasets.
#
# Datasets are downloaded to data/benchmarks/<name>/
# Large datasets (>1GB) are skipped unless explicitly requested.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data/benchmarks"

mkdir -p "$DATA_DIR"

# ── Color helpers ──────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; }

# ── Dataset fetchers ───────────────────────────────────────────────────

fetch_arc() {
    local dir="$DATA_DIR/arc/repo"
    if [ -d "$dir/data/training" ]; then
        ok "ARC-AGI already present ($(ls "$dir/data/training" | wc -l) tasks)"
        return
    fi
    echo "Fetching ARC-AGI (Chollet 2019)..."
    mkdir -p "$DATA_DIR/arc"
    git clone --depth 1 https://github.com/fchollet/ARC-AGI "$dir"
    ok "ARC-AGI: $(ls "$dir/data/training" | wc -l) training tasks"
}

fetch_gsm8k() {
    local file="$DATA_DIR/gsm8k/test.jsonl"
    if [ -f "$file" ]; then
        ok "GSM8K already present ($(wc -l < "$file") problems)"
        return
    fi
    echo "Fetching GSM8K (Cobbe et al. 2021)..."
    mkdir -p "$DATA_DIR/gsm8k"
    curl -sL "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl" -o "$file"
    ok "GSM8K: $(wc -l < "$file") test problems"
}

fetch_mnist() {
    local dir="$DATA_DIR/mnist"
    if [ -f "$dir/train-images-idx3-ubyte" ]; then
        ok "MNIST already present"
        return
    fi
    echo "Fetching MNIST..."
    mkdir -p "$dir"
    for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
        curl -sL "https://storage.googleapis.com/cvdf-datasets/mnist/${f}.gz" | gunzip > "$dir/$f"
    done
    ok "MNIST: 60K train + 10K test images"
}

fetch_ethics() {
    local dir="$DATA_DIR/ethics"
    if [ -d "$dir" ] && [ "$(ls "$dir" | wc -l)" -gt 3 ]; then
        ok "Hendrycks ETHICS already present"
        return
    fi
    echo "Fetching Hendrycks ETHICS..."
    mkdir -p "$dir"
    # ETHICS dataset from Hendrycks et al. 2021
    local tmp=$(mktemp -d)
    curl -sL "https://people.eecs.berkeley.edu/~hendrycks/ethics.tar" -o "$tmp/ethics.tar"
    tar xf "$tmp/ethics.tar" -C "$dir" --strip-components=1
    rm -rf "$tmp"
    ok "Hendrycks ETHICS: $(ls "$dir" | wc -l) categories"
}

fetch_truthfulqa() {
    local dir="$DATA_DIR/truthfulqa"
    if [ -d "$dir" ] && [ "$(ls "$dir"/*.csv 2>/dev/null | wc -l)" -gt 0 ]; then
        ok "TruthfulQA already present"
        return
    fi
    echo "Fetching TruthfulQA (Lin et al. 2022)..."
    mkdir -p "$dir"
    curl -sL "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv" -o "$dir/TruthfulQA.csv"
    ok "TruthfulQA: $(wc -l < "$dir/TruthfulQA.csv") questions"
}

fetch_isolet() {
    local dir="$DATA_DIR/isolet"
    if [ -d "$dir" ] && [ "$(ls "$dir"/*.data 2>/dev/null | wc -l)" -gt 0 ]; then
        ok "ISOLET already present"
        return
    fi
    echo "Fetching ISOLET (UCI ML Repository)..."
    mkdir -p "$dir"
    curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet1+2+3+4.data.Z" | uncompress > "$dir/isolet_train.data" 2>/dev/null || true
    curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet5.data.Z" | uncompress > "$dir/isolet_test.data" 2>/dev/null || true
    ok "ISOLET: spoken letter recognition dataset"
}

# ── Main ───────────────────────────────────────────────────────────────

DATASETS=(arc gsm8k mnist ethics truthfulqa isolet)

if [ $# -gt 0 ]; then
    # Fetch specific dataset
    for name in "$@"; do
        func="fetch_${name}"
        if declare -f "$func" > /dev/null 2>&1; then
            $func
        else
            echo "Unknown dataset: $name"
            echo "Available: ${DATASETS[*]}"
            exit 1
        fi
    done
else
    echo "Symthaea Dataset Fetcher"
    echo "========================"
    echo "Checking ${#DATASETS[@]} datasets..."
    echo ""
    for name in "${DATASETS[@]}"; do
        "fetch_${name}"
    done
    echo ""
    echo "Done. See data/DATASET_REGISTRY.md for full inventory."
fi
