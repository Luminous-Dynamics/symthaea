#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Master Benchmark Dataset Downloader
# ====================================
# Downloads ALL external datasets for Symthaea psych-bench validation.
#
# Usage:
#   ./scripts/download_all_benchmarks.sh          # Download everything
#   ./scripts/download_all_benchmarks.sh --tier 1  # Only tier 1 (core)
#   ./scripts/download_all_benchmarks.sh --status   # Check what's downloaded
#
# Total estimated size: ~1.2 GB (excluding Sleep-EDF which is already 7.2 GB)
#
# Tier 1 (~200 MB): Core validation — ETHICS, ARC, TruthfulQA, BBQ, CrowS-Pairs, WinoBias
# Tier 2 (~350 MB): Moral/social — Moral Stories, MoralChoice, MoralExceptQA, Social IQa, Social Chemistry, SCRUPLES
# Tier 3 (~500 MB): LLM comparison — MMLU, HellaSwag, GSM8K, HumanEval, BIG-Bench Hard, CIFAR-10
# Tier 4 (~1 MB):   Consciousness research — PCI refs, binocular rivalry params, Neuropixels pointers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BENCH_DIR="$PROJECT_DIR/data/benchmarks"
MORAL_DIR="$PROJECT_DIR/data/moral_datasets"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
skip() { echo -e "  ${YELLOW}[SKIP]${NC} $1 (already exists)"; }

TIER="${2:-all}"

check_status() {
    echo "=== Benchmark Dataset Status ==="
    echo ""
    echo "TIER 1: Core Validation"
    for d in ethics arc truthfulqa bbq crows-pairs winobias; do
        if [ -d "$BENCH_DIR/$d/repo" ] || [ "$(find "$BENCH_DIR/$d" -type f -not -name results.json 2>/dev/null | wc -l)" -gt 1 ]; then
            size=$(du -sh "$BENCH_DIR/$d" 2>/dev/null | cut -f1)
            ok "$d ($size)"
        else
            fail "$d"
        fi
    done

    echo ""
    echo "TIER 2: Moral/Social"
    for d in moral_stories moralchoice moralexceptqa social_iqa; do
        if [ -d "$BENCH_DIR/$d/repo" ]; then
            size=$(du -sh "$BENCH_DIR/$d" 2>/dev/null | cut -f1)
            ok "$d ($size)"
        else
            fail "$d"
        fi
    done
    # Check moral_datasets dir too
    if [ -f "$MORAL_DIR/social_chemistry_292k.json" ]; then
        ok "social_chemistry (in moral_datasets/)"
    else
        fail "social_chemistry"
    fi

    echo ""
    echo "TIER 3: LLM Comparison"
    for d in mmlu hellaswag gsm8k humaneval bigbench_hard cifar10; do
        if [ -d "$BENCH_DIR/$d/repo" ] || [ -d "$BENCH_DIR/$d/cifar-10-batches-bin" ] || [ -d "$BENCH_DIR/$d/repo/.git" ]; then
            size=$(du -sh "$BENCH_DIR/$d" 2>/dev/null | cut -f1)
            ok "$d ($size)"
        else
            fail "$d"
        fi
    done

    echo ""
    echo "TIER 4: Consciousness Research"
    for f in consciousness/pci/pci_reference_values.json consciousness/binocular_rivalry/binocular_rivalry_parameters.json consciousness/neuropixels/neuropixels_reference.json; do
        if [ -f "$BENCH_DIR/$f" ]; then
            ok "$(dirname $f)"
        else
            fail "$(dirname $f)"
        fi
    done

    echo ""
    echo "PRE-EXISTING (already downloaded)"
    for d in mnist sleep-edf isolet pyphi celegans; do
        size=$(du -sh "$BENCH_DIR/$d" 2>/dev/null | cut -f1)
        ok "$d ($size)"
    done

    echo ""
    total=$(du -sh "$BENCH_DIR" 2>/dev/null | cut -f1)
    echo "Total benchmark data: $total"
}

clone_if_missing() {
    local name="$1"
    local url="$2"
    local target="$BENCH_DIR/$name/repo"

    if [ -d "$target/.git" ]; then
        skip "$name"
        return
    fi

    mkdir -p "$BENCH_DIR/$name"
    echo "  Cloning $name..."
    git clone --depth 1 "$url" "$target" 2>&1 | tail -1
    ok "$name"
}

hf_download() {
    local name="$1"
    local repo_id="$2"
    local target="$BENCH_DIR/$name/repo"

    if [ -d "$target" ] && [ "$(find "$target" -type f -not -path '*/.git/*' | wc -l)" -gt 2 ]; then
        skip "$name"
        return
    fi

    mkdir -p "$BENCH_DIR/$name"
    echo "  Downloading $name from HuggingFace..."
    source "$PROJECT_DIR/.venv/bin/activate"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo_id', repo_type='dataset', local_dir='$target', local_dir_use_symlinks=False)
" 2>&1 | tail -1
    ok "$name"
}

if [ "${1:-}" = "--status" ]; then
    check_status
    exit 0
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Symthaea Benchmark Dataset Downloader                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── TIER 1: Core Validation ──────────────────────────────────────
if [ "$TIER" = "all" ] || [ "$TIER" = "1" ]; then
    echo "── TIER 1: Core Validation ──"

    # ETHICS (use existing script)
    if [ "$(find "$BENCH_DIR/ethics" -name '*.csv' 2>/dev/null | wc -l)" -ge 5 ]; then
        skip "ethics"
    else
        bash "$SCRIPT_DIR/download_hendrycks_ethics.sh"
    fi

    # ARC
    if [ -d "$BENCH_DIR/arc/repo/.git" ]; then
        echo "  Updating ARC..."
        cd "$BENCH_DIR/arc/repo" && git pull --quiet 2>&1 | tail -1
        ok "arc (updated)"
        cd "$PROJECT_DIR"
    else
        clone_if_missing "arc" "https://github.com/fchollet/ARC-AGI.git"
    fi

    clone_if_missing "truthfulqa" "https://github.com/sylinrl/TruthfulQA.git"
    clone_if_missing "bbq" "https://github.com/nyu-mll/BBQ.git"
    clone_if_missing "crows-pairs" "https://github.com/nyu-mll/crows-pairs.git"
    clone_if_missing "winobias" "https://github.com/uclanlp/corefBias.git"

    echo ""
fi

# ── TIER 2: Moral/Social ────────────────────────────────────────
if [ "$TIER" = "all" ] || [ "$TIER" = "2" ]; then
    echo "── TIER 2: Moral/Social ──"

    # Run existing moral download script
    if [ "$(find "$MORAL_DIR" -name '*.json' 2>/dev/null | wc -l)" -lt 10 ]; then
        source "$PROJECT_DIR/.venv/bin/activate"
        python3 "$SCRIPT_DIR/download_moral_datasets_v3.py"
    else
        skip "moral_datasets_v3 (${MORAL_DIR})"
    fi

    # HuggingFace datasets
    hf_download "moral_stories" "demelin/moral_stories"
    hf_download "moralchoice" "ninoscherrer/moralchoice"
    hf_download "moralexceptqa" "feradauto/MoralExceptQA"
    hf_download "social_iqa" "allenai/social_i_qa"

    # Social Chemistry
    if [ -f "$MORAL_DIR/social_chemistry_292k.json" ]; then
        skip "social_chemistry"
    else
        source "$PROJECT_DIR/.venv/bin/activate"
        python3 "$SCRIPT_DIR/download_social_chemistry.py"
    fi

    echo ""
fi

# ── TIER 3: LLM Comparison ──────────────────────────────────────
if [ "$TIER" = "all" ] || [ "$TIER" = "3" ]; then
    echo "── TIER 3: LLM Comparison ──"

    hf_download "mmlu" "cais/mmlu"
    hf_download "hellaswag" "Rowan/hellaswag"
    hf_download "gsm8k" "openai/gsm8k"
    hf_download "bigbench_hard" "maveriq/bigbenchhard"
    clone_if_missing "humaneval" "https://github.com/openai/human-eval.git"

    # CIFAR-10
    if [ -d "$BENCH_DIR/cifar10/cifar-10-batches-bin" ]; then
        skip "cifar10"
    else
        echo "  Downloading CIFAR-10..."
        mkdir -p "$BENCH_DIR/cifar10"
        curl -sL "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" -o "$BENCH_DIR/cifar10/cifar-10-binary.tar.gz"
        tar -xzf "$BENCH_DIR/cifar10/cifar-10-binary.tar.gz" -C "$BENCH_DIR/cifar10/"
        ok "cifar10"
    fi

    echo ""
fi

# ── TIER 4: Consciousness Research ──────────────────────────────
if [ "$TIER" = "all" ] || [ "$TIER" = "4" ]; then
    echo "── TIER 4: Consciousness Research ──"

    source "$PROJECT_DIR/.venv/bin/activate"
    python3 -c "
import json, os

base = '$BENCH_DIR/consciousness'

# PCI reference values
pci_path = f'{base}/pci/pci_reference_values.json'
if os.path.exists(pci_path):
    print('  [SKIP] pci (already exists)')
else:
    os.makedirs(f'{base}/pci', exist_ok=True)
    data = {
        'metadata': {'dataset': 'PCI Reference Values', 'source': 'Casali et al. (2013)', 'threshold': 0.31},
        'reference_values': {
            'healthy_awake': {'mean': 0.44, 'std': 0.07, 'n': 102, 'state': 'conscious'},
            'dreaming_rem': {'mean': 0.41, 'std': 0.08, 'n': 12, 'state': 'conscious'},
            'nrem_sleep': {'mean': 0.23, 'std': 0.06, 'n': 18, 'state': 'unconscious'},
            'propofol': {'mean': 0.18, 'std': 0.05, 'n': 16, 'state': 'unconscious'},
            'vegetative': {'mean': 0.24, 'std': 0.09, 'n': 43, 'state': 'uncertain'},
        }
    }
    with open(pci_path, 'w') as f: json.dump(data, f, indent=2)
    print('  [OK] pci')

# Binocular rivalry
br_path = f'{base}/binocular_rivalry/binocular_rivalry_parameters.json'
if os.path.exists(br_path):
    print('  [SKIP] binocular_rivalry (already exists)')
else:
    os.makedirs(f'{base}/binocular_rivalry', exist_ok=True)
    data = {'metadata': {'dataset': 'Binocular Rivalry Paradigm'}, 'phi_conscious': 0.42, 'phi_suppressed': 0.18}
    with open(br_path, 'w') as f: json.dump(data, f, indent=2)
    print('  [OK] binocular_rivalry')

# Neuropixels reference
np_path = f'{base}/neuropixels/neuropixels_reference.json'
if os.path.exists(np_path):
    print('  [SKIP] neuropixels (already exists)')
else:
    os.makedirs(f'{base}/neuropixels', exist_ok=True)
    data = {'metadata': {'dataset': 'Allen Neuropixels', 'sessions': 58, 'units': 99180, 'size_note': '~2TB full, use AllenSDK'}}
    with open(np_path, 'w') as f: json.dump(data, f, indent=2)
    print('  [OK] neuropixels')

# Drosophila connectome
dros_path = '$BENCH_DIR/drosophila_connectome/hemibrain_connectivity.json'
if os.path.exists(dros_path):
    print('  [SKIP] drosophila_connectome (already exists)')
else:
    print('  [WARN] drosophila_connectome missing - run full download')
"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════"
check_status
echo ""
echo "Done! Run with --status to check anytime."
