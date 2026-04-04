#!/usr/bin/env bash
# Quick screenshot of Terra Atlas globe via headless Brave
# Usage: ./scripts/screenshot.sh [output.png]
OUT="${1:-/tmp/terra-screenshot.png}"
timeout 60 brave --headless=new \
  --enable-unsafe-swiftshader \
  --screenshot="$OUT" \
  --window-size=1280,800 \
  --virtual-time-budget=25000 \
  "http://localhost:8091" 2>/dev/null
if [ -f "$OUT" ]; then
  echo "Screenshot saved: $OUT ($(du -h "$OUT" | cut -f1))"
else
  echo "FAILED: No screenshot produced"
  exit 1
fi
