#!/usr/bin/env bash
# Download SPARC (Spitzer Photometry and Accurate Rotation Curves) dataset
# Source: Lelli, McGaugh & Schombert (2016), AJ 152, 157
# http://astroweb.cwru.edu/SPARC/
#
# Contents:
#   - SPARC_Lelli2016c.mrt  — galaxy table (distance, inclination, L[3.6], quality) (~30 KB)
#   - Rotmod_LTG.zip        — per-galaxy rotation-curve mass models (~0.3 MB)
#
# Usage: ./scripts/download_sparc.sh

set -euo pipefail

BASE_URL="https://astroweb.cwru.edu/SPARC"
FALLBACK_URL="https://astroweb.case.edu/SPARC"
DATA_DIR="data/benchmarks/sparc"

mkdir -p "$DATA_DIR"

echo "=== SPARC Dataset Download ==="
echo "Target: $DATA_DIR"
echo ""

fetch() {
    local file="$1" out="$2"
    curl -fL --progress-bar -o "$out" "$BASE_URL/$file" \
        || curl -fL --progress-bar -o "$out" "$FALLBACK_URL/$file"
}

# 1. Galaxy metadata table
if [ -f "$DATA_DIR/SPARC_Lelli2016c.mrt" ]; then
    echo "[skip] SPARC_Lelli2016c.mrt already present"
else
    echo "[1/2] Downloading galaxy table (~30 KB)..."
    fetch "SPARC_Lelli2016c.mrt" "$DATA_DIR/SPARC_Lelli2016c.mrt"
    echo "       Done."
fi

# 2. Rotation-curve mass models
if find "$DATA_DIR" -name '*_rotmod.dat' 2>/dev/null | grep -q .; then
    echo "[skip] rotmod files already present"
else
    echo "[2/2] Downloading Rotmod_LTG.zip (~0.3 MB)..."
    fetch "Rotmod_LTG.zip" "$DATA_DIR/Rotmod_LTG.zip"
    echo "       Extracting..."
    unzip -q -o "$DATA_DIR/Rotmod_LTG.zip" -d "$DATA_DIR/"
    rm "$DATA_DIR/Rotmod_LTG.zip"
    echo "       Done."
fi

echo ""
echo "=== Download complete ==="
echo "Galaxy table: $(ls -la "$DATA_DIR/SPARC_Lelli2016c.mrt" 2>/dev/null | awk '{print $5}' || echo missing) bytes"
echo "Rotmod files: $(find "$DATA_DIR" -name '*_rotmod.dat' | wc -l)"
