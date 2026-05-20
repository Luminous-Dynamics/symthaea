#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# A/B lane for repair memory:
#   1. run the forced repair lane without memory and export lessons
#   2. rerun the same lane with those lessons loaded as repair memory
#   3. assert memory is used and quality does not regress

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${CODING_BACKEND_AB_OUT_DIR:-/tmp/symthaea-coding-backend-repair-memory-ab}"
WITHOUT_DIR="$OUT_DIR/without-memory"
WITH_DIR="$OUT_DIR/with-memory"
WITHOUT_REPORT="$WITHOUT_DIR/benchmark-coding-backends-repair.json"
WITH_REPORT="$WITH_DIR/benchmark-coding-backends-repair.json"
MEMORY_JSONL="$WITHOUT_DIR/repair-lessons-repair.jsonl"

mkdir -p "$WITHOUT_DIR" "$WITH_DIR"

echo "[repair-memory-ab] baseline run without repair memory"
unset SYMTHAEA_REPAIR_MEMORY_JSONL || true
CODING_BACKEND_LANE=repair \
CODING_BACKEND_OUT_DIR="$WITHOUT_DIR" \
CODING_BACKEND_REPORT="$WITHOUT_REPORT" \
CODING_BACKEND_REPAIR_LESSONS="$MEMORY_JSONL" \
scripts/coding_backend_regression.sh

echo "[repair-memory-ab] comparison run with repair memory"
SYMTHAEA_REPAIR_MEMORY_JSONL="$MEMORY_JSONL" \
CODING_BACKEND_LANE=repair \
CODING_BACKEND_OUT_DIR="$WITH_DIR" \
CODING_BACKEND_REPORT="$WITH_REPORT" \
CODING_BACKEND_SKIP_BASELINE=1 \
scripts/coding_backend_regression.sh

echo "[repair-memory-ab] checking A/B result"
python3 scripts/check_repair_memory_ab.py \
    --without-memory "$WITHOUT_REPORT" \
    --with-memory "$WITH_REPORT" \
    --min-memory-hits "${CODING_BACKEND_AB_MIN_MEMORY_HITS:-8}" \
    --min-memory-helped-tasks "${CODING_BACKEND_AB_MIN_MEMORY_HELPED_TASKS:-1}" \
    --summary-out "$OUT_DIR/repair-memory-ab-summary.json" \
    --require-no-pass-regression \
    --require-no-attempt-regression

echo "[repair-memory-ab] PASS"
echo "  without-memory report: $WITHOUT_REPORT"
echo "  with-memory report:    $WITH_REPORT"
echo "  A/B summary:           $OUT_DIR/repair-memory-ab-summary.json"
echo "  memory jsonl:          $MEMORY_JSONL"
