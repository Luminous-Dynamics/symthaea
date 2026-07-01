#!/usr/bin/env bash
# Download DEAM (Database for Emotional Analysis of Music) dataset
# Source: University of Geneva, MediaEval Emotion in Music task
# Reference: Aljanaki, Yang & Soleymani (2017)
#
# Contents:
#   - 1,802 audio excerpts (45s MP3, ~1.3GB)
#   - Continuous valence-arousal annotations at 2Hz (~5MB)
#   - Metadata (genre, tags) (~1MB)
#   - Pre-extracted features (~600MB) [optional]
#
# Usage: ./scripts/download_deam.sh [--with-features]

set -euo pipefail

BASE_URL="https://cvml.unige.ch/databases/DEAM"
DATA_DIR="data/deam"
WITH_FEATURES=false

if [[ "${1:-}" == "--with-features" ]]; then
    WITH_FEATURES=true
fi

mkdir -p "$DATA_DIR"

echo "=== DEAM Dataset Download ==="
echo "Target: $DATA_DIR"
echo ""

# 1. Audio (1.3 GB)
if [ -d "$DATA_DIR/MEMD_audio" ] || [ -d "$DATA_DIR/audio" ]; then
    echo "[skip] Audio already downloaded"
else
    echo "[1/4] Downloading audio (1.3 GB)..."
    curl -L --progress-bar -o "$DATA_DIR/DEAM_audio.zip" "$BASE_URL/DEAM_audio.zip"
    echo "       Extracting..."
    unzip -q -o "$DATA_DIR/DEAM_audio.zip" -d "$DATA_DIR/"
    rm "$DATA_DIR/DEAM_audio.zip"
    echo "       Done."
fi

# 2. Annotations (5 MB)
if [ -d "$DATA_DIR/DEAM_Annotations" ] || [ -d "$DATA_DIR/annotations" ]; then
    echo "[skip] Annotations already downloaded"
else
    echo "[2/4] Downloading annotations (5 MB)..."
    curl -L --progress-bar -o "$DATA_DIR/DEAM_Annotations.zip" "$BASE_URL/DEAM_Annotations.zip"
    echo "       Extracting..."
    unzip -q -o "$DATA_DIR/DEAM_Annotations.zip" -d "$DATA_DIR/"
    rm "$DATA_DIR/DEAM_Annotations.zip"
    echo "       Done."
fi

# 3. Metadata (1 MB)
if [ -f "$DATA_DIR/metadata" ] || [ -d "$DATA_DIR/metadata" ]; then
    echo "[skip] Metadata already downloaded"
else
    echo "[3/4] Downloading metadata..."
    curl -L --progress-bar -o "$DATA_DIR/metadata.zip" "$BASE_URL/metadata.zip"
    echo "       Extracting..."
    unzip -q -o "$DATA_DIR/metadata.zip" -d "$DATA_DIR/"
    rm "$DATA_DIR/metadata.zip"
    echo "       Done."
fi

# 4. Features (601 MB, optional)
if $WITH_FEATURES; then
    if [ -d "$DATA_DIR/features" ]; then
        echo "[skip] Features already downloaded"
    else
        echo "[4/4] Downloading pre-extracted features (601 MB)..."
        curl -L --progress-bar -o "$DATA_DIR/features.zip" "$BASE_URL/features.zip"
        echo "       Extracting..."
        unzip -q -o "$DATA_DIR/features.zip" -d "$DATA_DIR/"
        rm "$DATA_DIR/features.zip"
        echo "       Done."
    fi
else
    echo "[4/4] Skipping features (use --with-features to include)"
fi

# 5. Manual
if [ ! -f "$DATA_DIR/manual.pdf" ]; then
    curl -sL -o "$DATA_DIR/manual.pdf" "$BASE_URL/manual.pdf" 2>/dev/null || true
fi

echo ""
echo "=== Download complete ==="
echo "Audio:       $(find "$DATA_DIR" -name '*.mp3' 2>/dev/null | wc -l) files"
echo "Annotations: $(find "$DATA_DIR" -name '*.csv' 2>/dev/null | wc -l) CSV files"
echo ""
echo "Dataset structure:"
ls -la "$DATA_DIR/" 2>/dev/null || true
