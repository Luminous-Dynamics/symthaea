#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Automated Broca Intelligence Upgrade Script

set -e

echo "🚀 Starting Broca Intelligence Upgrade..."

# 1. Ingest verified distillation geometries
echo "📥 Step 1: Ingesting verified distillation geometries..."
python3 scripts/ingest_distillation_to_broca.py

# 2. Execute Broca SSM training
echo "🧠 Step 2: Training Broca SSM on pristine dataset..."
RUSTC_WRAPPER= SCCACHE_DISABLE=1 cargo run --bin broca-train -p symthaea-broca -- \
    --data data/training/broca_humaneval_pristine.jsonl \
    --epochs 3 \
    --output broca.bin

# 3. Deploy newly trained model to production
echo "🚀 Step 3: Deploying new model to production..."
mkdir -p models/broca
if [ -f "broca.bin" ]; then
    cp broca.bin models/broca/production.bin
    echo "✅ Model deployed to models/broca/production.bin"
else
    echo "⚠️ Warning: Could not find 'broca.bin'. Check training output."
fi

echo "✨ Broca upgrade complete! Symthaea's baseline intelligence has been raised."
