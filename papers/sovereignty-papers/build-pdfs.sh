#!/usr/bin/env bash
# Build PDFs for The Sovereignty Papers
# Usage: nix-shell -p texliveSmall --run "./build-pdfs.sh"
# Or: ./build-pdfs.sh  (if pdflatex is already in PATH)

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/pdf"
mkdir -p "$OUT"

# Common pandoc options
OPTS=(
  --pdf-engine=xelatex
  -V geometry:margin=1.2in
  -V fontsize=11pt
  -V documentclass=article
  -V linkcolor=blue
  -V urlcolor=blue
)

echo "=== Building individual essay PDFs ==="
for f in "$DIR"/essay-*.md; do
  base="$(basename "$f" .md)"
  echo "  $base..."
  pandoc "$f" -o "$OUT/$base.pdf" "${OPTS[@]}" 2>/dev/null || {
    echo "    WARNING: failed, trying with warnings..."
    pandoc "$f" -o "$OUT/$base.pdf" "${OPTS[@]}"
  }
done

echo ""
echo "=== Building combined book PDF ==="

# Create combined markdown with page breaks between essays
COMBINED="$OUT/.combined.md"
cat > "$COMBINED" << 'FRONTMATTER'
---
title: "The Sovereignty Papers"
subtitle: "Essays on Consciousness-First Governance for the Post-State Era"
author: "Tristan Stoltz & Symthaea"
date: "2026"
---

\newpage

FRONTMATTER

# Add reading guide as introduction
cat "$DIR/READING_GUIDE.md" | sed '1,/^---$/d' | sed '1,/^---$/d' >> "$COMBINED"
echo -e "\n\n\\\\newpage\n\n" >> "$COMBINED"

# Add each essay with page breaks
for f in "$DIR"/essay-*.md; do
  # Strip YAML frontmatter
  cat "$f" | sed '1,/^---$/d' | sed '1,/^---$/d' >> "$COMBINED"
  echo -e "\n\n\\\\newpage\n\n" >> "$COMBINED"
done

pandoc "$COMBINED" \
  -o "$OUT/the-sovereignty-papers-complete.pdf" \
  --pdf-engine=xelatex \
  -V geometry:margin=1.2in \
  -V fontsize=11pt \
  -V documentclass=report \
  -V linkcolor=blue \
  -V urlcolor=blue \
  --toc \
  --toc-depth=2 \
  2>/dev/null || pandoc "$COMBINED" \
  -o "$OUT/the-sovereignty-papers-complete.pdf" \
  --pdf-engine=xelatex \
  -V geometry:margin=1.2in \
  -V fontsize=11pt \
  -V documentclass=report \
  -V linkcolor=blue \
  -V urlcolor=blue \
  --toc \
  --toc-depth=2

rm -f "$COMBINED"

echo ""
echo "=== Done ==="
echo "Individual essays: $OUT/essay-*.pdf"
echo "Combined book:     $OUT/the-sovereignty-papers-complete.pdf"
echo ""
ls -lh "$OUT"/*.pdf | awk '{print $5, $NF}'
