#!/usr/bin/env bash
# Regenerate docs/TECHNICAL_REPORT.pdf from TECHNICAL_REPORT.md.
#
# Requires pandoc (system-wide) + typst (fetched ephemerally via `nix shell`,
# not installed globally — matches the project's NixOS immutable-root rule).
set -euo pipefail
cd "$(dirname "$0")"

nix shell nixpkgs#typst --command pandoc TECHNICAL_REPORT.md \
    --pdf-engine=typst \
    -o TECHNICAL_REPORT.pdf \
    --metadata title="Spark Engine Technical Report: LCF Anomaly Investigation" \
    --metadata author="Luminous Dynamics" \
    --metadata date="2026-07-07" \
    -V papersize=us-letter \
    -V margin-x=1in -V margin-y=1in \
    -V mainfont="Noto Sans"

echo "Wrote $(dirname "$0")/TECHNICAL_REPORT.pdf"
