#!/usr/bin/env bash
# Generate all figures for the ALIFE 2026 paper
# Usage: ./make_figures.sh
# Requires: Python 3 with matplotlib + numpy (use nix-shell if needed)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ALIFE 2026 Figure Generation ==="

# Step 1: Generate figures from data
echo "[1/2] Generating figures..."
if command -v python3 &>/dev/null && python3 -c "import matplotlib" 2>/dev/null; then
    python3 generate_figures.py
else
    echo "  matplotlib not found, trying nix-shell..."
    nix-shell -p python3Packages.matplotlib python3Packages.numpy --run "python3 generate_figures.py"
fi

# Step 2: Verify all expected figures exist
echo "[2/2] Verifying figures..."
EXPECTED="fig1_ablation.pdf fig2_curvature.pdf fig3_dynamics.pdf fig4_cooperation.pdf fig5_phase_transition.pdf"
MISSING=0
for fig in $EXPECTED; do
    if [ -f "$fig" ]; then
        echo "  OK: $fig"
    else
        echo "  MISSING: $fig"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo "WARNING: $MISSING figures missing!"
    exit 1
else
    echo "All figures generated successfully."
fi
