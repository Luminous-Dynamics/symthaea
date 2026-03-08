#!/usr/bin/env bash
# Convert all 5 Tier-1 IDDs to filing-ready provisional patent PDFs
# Usage: nix-shell -p texliveSmall pandoc --run "bash convert-to-pdf.sh"

set -euo pipefail

PATENT_DIR="/srv/luminous-dynamics/patents"
OUTPUT_DIR="$PATENT_DIR/filing-ready"
TIER1_DIR="$PATENT_DIR/tier-1"

IDDS=(
  "P-001_hdc-ltc-neuron"
  "P-002_moral-algebra"
  "P-003_ltc-vocal-tract"
  "P-004_consciousness-equation"
  "P-005_consciousness-fl"
)

for idd in "${IDDS[@]}"; do
  echo "Converting $idd..."

  INPUT="$TIER1_DIR/$idd/IDD.md"
  CLEANED="$OUTPUT_DIR/${idd}_cleaned.md"
  PDF="$OUTPUT_DIR/${idd}_provisional.pdf"

  if [ ! -f "$INPUT" ]; then
    echo "  SKIP: $INPUT not found"
    continue
  fi

  # Clean up the IDD for filing:
  # 1. Remove "Invention Disclosure Document" subtitle (it's now a provisional application)
  # 2. Remove confidentiality notice at the bottom
  # 3. Remove "Key Source Files" section (internal paths)
  # 4. Add provisional application header
  sed \
    -e 's/## Invention Disclosure Document/## Provisional Patent Application/' \
    -e '/This document is confidential and privileged/d' \
    -e '/Document prepared for patent counsel/d' \
    -e '/^All files located under/d' \
    "$INPUT" > "$CLEANED"

  # Convert to PDF via pandoc + pdflatex
  pandoc "$CLEANED" \
    -o "$PDF" \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V header-includes='\usepackage{fancyhdr}\pagestyle{fancy}\fancyhead[R]{PROVISIONAL APPLICATION}\fancyfoot[C]{\thepage}' \
    -V title="" \
    --highlight-style=tango \
    2>&1

  if [ -f "$PDF" ]; then
    PAGES=$(pdfinfo "$PDF" 2>/dev/null | grep Pages | awk '{print $2}' || echo "?")
    SIZE=$(du -h "$PDF" | cut -f1)
    echo "  OK: $PDF ($PAGES pages, $SIZE)"
  else
    echo "  FAIL: PDF not generated"
  fi
done

echo ""
echo "All done. PDFs in: $OUTPUT_DIR"
