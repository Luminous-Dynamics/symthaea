#!/usr/bin/env bash
# Generate all figures for the PGx Health Equity paper
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== PGx Health Equity Figure Generation ==="
if command -v python3 &>/dev/null && python3 -c "import matplotlib" 2>/dev/null; then
    python3 generate_figures.py
else
    nix-shell -p python3Packages.matplotlib python3Packages.numpy --run "python3 generate_figures.py"
fi

echo "Verifying figures..."
for fig in figures/figure{1,2,3,4}_*.pdf; do
    if [ -f "$fig" ]; then echo "  OK: $fig"; else echo "  MISSING: $fig"; fi
done
echo "Done."
