#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Report-first Broca checkpoint promotion workflow:
#   repair benchmark -> repair-memory A/B -> repair training smoke -> selector.
#
# By default this writes reports only. Set BROCA_PROMOTE=1 to copy the selected
# checkpoint into BROCA_PROMOTION_DIR.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${BROCA_PROMOTION_OUT_DIR:-/tmp/symthaea-broca-promotion}"
LESSON_STORE="${BROCA_REPAIR_LESSON_STORE:-$OUT_DIR/repair-lessons-store.jsonl}"
PROMOTION_DIR="${BROCA_PROMOTION_DIR:-$OUT_DIR/promoted}"
PROMOTE="${BROCA_PROMOTE:-0}"
MODE="${BROCA_PROMOTION_MODE:-fast}"

case "$MODE" in
    fast)
        DEFAULT_EPOCHS=1
        DEFAULT_NEGATIVE_SAMPLES=1
        DEFAULT_EVAL_LIMIT=1
        DEFAULT_MAX_GEN_TOKENS=4
        DEFAULT_BACKEND=cpu
        DEFAULT_USE_NIX=0
        ;;
    manual)
        DEFAULT_EPOCHS=3
        DEFAULT_NEGATIVE_SAMPLES=16
        DEFAULT_EVAL_LIMIT=8
        DEFAULT_MAX_GEN_TOKENS=16
        DEFAULT_BACKEND=auto
        DEFAULT_USE_NIX=1
        ;;
    *)
        echo "[broca-promotion] BROCA_PROMOTION_MODE must be fast or manual, got: $MODE" >&2
        exit 2
        ;;
esac

REQUIRE_TRAINED_IMPROVEMENT="${BROCA_PROMOTION_REQUIRE_TRAINED_IMPROVEMENT:-$PROMOTE}"

mkdir -p "$OUT_DIR"

AB_DIR="$OUT_DIR/repair-memory-ab"
SMOKE_DIR="$OUT_DIR/repair-train"
AB_SUMMARY="$AB_DIR/repair-memory-ab-summary.json"
AB_WITHOUT_LESSONS="$AB_DIR/without-memory/repair-lessons-repair.jsonl"
AB_WITH_LESSONS="$AB_DIR/with-memory/repair-lessons-repair.jsonl"
CODING_REPORT="$SMOKE_DIR/coding/benchmark-coding-backends-repair.json"
SELECTION_REPORT="$SMOKE_DIR/broca/selected-checkpoint.json"

echo "[broca-promotion] running repair-memory A/B"
echo "[broca-promotion] mode: $MODE"
CODING_BACKEND_AB_OUT_DIR="$AB_DIR" \
scripts/coding_backend_repair_memory_ab.sh

echo "[broca-promotion] merging repair lessons into store"
python3 scripts/merge_repair_lessons.py \
    --store "$LESSON_STORE" \
    --input "$AB_WITHOUT_LESSONS" \
    --input "$AB_WITH_LESSONS"

echo "[broca-promotion] training/evaluating repair checkpoint"
SYMTHAEA_REPAIR_MEMORY_JSONL="$LESSON_STORE" \
BROCA_REPAIR_SMOKE_OUT_DIR="$SMOKE_DIR" \
BROCA_REPAIR_SMOKE_SELECT=1 \
BROCA_REPAIR_SMOKE_BASELINE="${BROCA_PROMOTION_BASELINE:-1}" \
BROCA_REPAIR_SMOKE_CANONICAL="${BROCA_PROMOTION_CANONICAL:-1}" \
BROCA_REPAIR_SMOKE_CODING_REPORT="$CODING_REPORT" \
BROCA_REPAIR_SMOKE_REPAIR_MEMORY_AB_SUMMARY="$AB_SUMMARY" \
BROCA_REPAIR_SMOKE_REQUIRE_CODING_GATE=1 \
BROCA_REPAIR_SMOKE_REQUIRE_CODE_SIGNAL=1 \
BROCA_REPAIR_SMOKE_REQUIRE_REPAIR_MEMORY_GATE=1 \
BROCA_REPAIR_SMOKE_REQUIRE_TRAINED_IMPROVEMENT="$REQUIRE_TRAINED_IMPROVEMENT" \
BROCA_REPAIR_SMOKE_MIN_CODING_SCORE="${BROCA_PROMOTION_MIN_CODING_SCORE:-0.9}" \
BROCA_REPAIR_SMOKE_MIN_CODING_QUALITY_PASS_RATE="${BROCA_PROMOTION_MIN_QUALITY_PASS_RATE:-1.0}" \
BROCA_REPAIR_SMOKE_MIN_REPAIR_MEMORY_HITS="${BROCA_PROMOTION_MIN_REPAIR_MEMORY_HITS:-8}" \
BROCA_REPAIR_SMOKE_MIN_REPAIR_MEMORY_SUCCESS_RATE="${BROCA_PROMOTION_MIN_REPAIR_MEMORY_SUCCESS_RATE:-1.0}" \
BROCA_REPAIR_SMOKE_MAX_REPAIR_MEMORY_HURT_TASKS="${BROCA_PROMOTION_MAX_REPAIR_MEMORY_HURT_TASKS:-0}" \
BROCA_REPAIR_SMOKE_EPOCHS="${BROCA_PROMOTION_EPOCHS:-$DEFAULT_EPOCHS}" \
BROCA_REPAIR_SMOKE_NEGATIVE_SAMPLES="${BROCA_PROMOTION_NEGATIVE_SAMPLES:-$DEFAULT_NEGATIVE_SAMPLES}" \
BROCA_REPAIR_SMOKE_EVAL_LIMIT="${BROCA_PROMOTION_EVAL_LIMIT:-$DEFAULT_EVAL_LIMIT}" \
BROCA_REPAIR_SMOKE_MAX_GEN_TOKENS="${BROCA_PROMOTION_MAX_GEN_TOKENS:-$DEFAULT_MAX_GEN_TOKENS}" \
BROCA_REPAIR_SMOKE_BACKEND="${BROCA_PROMOTION_BACKEND:-$DEFAULT_BACKEND}" \
BROCA_REPAIR_SMOKE_USE_NIX="${BROCA_PROMOTION_USE_NIX:-$DEFAULT_USE_NIX}" \
scripts/broca_repair_train_smoke.sh

SELECTED_CHECKPOINT="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "$SELECTION_REPORT"
)"

echo "[broca-promotion] selected checkpoint: $SELECTED_CHECKPOINT"
echo "[broca-promotion] selection report:    $SELECTION_REPORT"
echo "[broca-promotion] repair lesson store: $LESSON_STORE"

if [[ "$PROMOTE" == "1" ]]; then
    mkdir -p "$PROMOTION_DIR"
    cp "$SELECTED_CHECKPOINT" "$PROMOTION_DIR/broca-selected.bin"
    cp "$SELECTION_REPORT" "$PROMOTION_DIR/selected-checkpoint.json"
    cp "$CODING_REPORT" "$PROMOTION_DIR/coding-report.json"
    cp "$AB_SUMMARY" "$PROMOTION_DIR/repair-memory-ab-summary.json"
    echo "[broca-promotion] promoted to: $PROMOTION_DIR"
else
    echo "[broca-promotion] report-only mode; set BROCA_PROMOTE=1 to copy the selected checkpoint"
fi
