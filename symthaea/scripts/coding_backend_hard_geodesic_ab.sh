#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# A/B lane for AST-HDC geodesic fast-fail:
#   1. run shadow mode to estimate false positives without changing behavior
#   2. run hard rejection mode to short-circuit low-prior candidates
#   3. assert quality does not regress and emit a compact summary

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${CODING_BACKEND_GEODESIC_AB_OUT_DIR:-/tmp/symthaea-coding-backend-geodesic-ab}"
LANE="${CODING_BACKEND_GEODESIC_AB_LANE:-hard}"
THRESHOLD="${CODING_BACKEND_GEODESIC_AB_THRESHOLD:-0.25}"
SHADOW_DIR="$OUT_DIR/shadow"
HARD_DIR="$OUT_DIR/hard"
SHADOW_REPORT="$SHADOW_DIR/benchmark-coding-backends-$LANE.json"
HARD_REPORT="$HARD_DIR/benchmark-coding-backends-$LANE.json"
SUMMARY_JSON="$OUT_DIR/hard-geodesic-ab-summary.json"

mkdir -p "$SHADOW_DIR" "$HARD_DIR"

echo "[hard-geodesic-ab] shadow run"
CODING_BACKEND_LANE="$LANE" \
CODING_BACKEND_OUT_DIR="$SHADOW_DIR" \
CODING_BACKEND_REPORT="$SHADOW_REPORT" \
CODING_BACKEND_GEODESIC_REJECTION_SHADOW=1 \
CODING_BACKEND_STRUCTURAL_PRIOR_THRESHOLD="$THRESHOLD" \
CODING_BACKEND_SKIP_BASELINE="${CODING_BACKEND_GEODESIC_AB_SKIP_BASELINE:-1}" \
scripts/coding_backend_regression.sh

echo "[hard-geodesic-ab] hard rejection run"
CODING_BACKEND_LANE="$LANE" \
CODING_BACKEND_OUT_DIR="$HARD_DIR" \
CODING_BACKEND_REPORT="$HARD_REPORT" \
CODING_BACKEND_HARD_GEODESIC_REJECTION=1 \
CODING_BACKEND_STRUCTURAL_PRIOR_THRESHOLD="$THRESHOLD" \
CODING_BACKEND_SKIP_BASELINE=1 \
scripts/coding_backend_regression.sh

echo "[hard-geodesic-ab] writing comparison summary"
python3 - "$SHADOW_REPORT" "$HARD_REPORT" "$SUMMARY_JSON" "$THRESHOLD" <<'PY'
import json
import sys
from pathlib import Path

shadow_path, hard_path, summary_path = map(Path, sys.argv[1:4])
threshold = float(sys.argv[4])
shadow = json.loads(shadow_path.read_text())
hard = json.loads(hard_path.read_text())

def number(doc, key, default=0):
    value = doc.get(key)
    return default if value is None else value

summary = {
    "lane": shadow.get("benchmark"),
    "threshold": threshold,
    "shadow_report": str(shadow_path),
    "hard_report": str(hard_path),
    "shadow_hits": number(shadow, "geodesic_rejection_shadow_hits"),
    "shadow_true_positives": number(shadow, "geodesic_rejection_shadow_true_positives"),
    "shadow_false_positives": number(shadow, "geodesic_rejection_shadow_false_positives"),
    "hard_rejections": number(hard, "hard_geodesic_rejections"),
    "quality_pass_rate_shadow": number(shadow, "quality_pass_rate", 0.0),
    "quality_pass_rate_hard": number(hard, "quality_pass_rate", 0.0),
    "pass_rate_shadow": number(shadow, "pass_rate", 0.0),
    "pass_rate_hard": number(hard, "pass_rate", 0.0),
    "mean_attempts_shadow": number(shadow, "mean_attempts_per_task", 0.0),
    "mean_attempts_hard": number(hard, "mean_attempts_per_task", 0.0),
}
summary["quality_pass_rate_delta"] = (
    summary["quality_pass_rate_hard"] - summary["quality_pass_rate_shadow"]
)
summary["pass_rate_delta"] = summary["pass_rate_hard"] - summary["pass_rate_shadow"]
summary["mean_attempts_delta"] = (
    summary["mean_attempts_hard"] - summary["mean_attempts_shadow"]
)

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

if summary["quality_pass_rate_delta"] < -0.001:
    raise SystemExit("hard geodesic rejection regressed quality pass rate")
if summary["pass_rate_delta"] < -0.001:
    raise SystemExit("hard geodesic rejection regressed pass rate")
if summary["shadow_false_positives"] > 0 and summary["hard_rejections"] > 0:
    raise SystemExit(
        "hard geodesic rejection active while shadow mode detected false positives; raise threshold conservatism or keep shadow-only"
    )
PY

echo "[hard-geodesic-ab] PASS"
echo "  shadow report: $SHADOW_REPORT"
echo "  hard report:   $HARD_REPORT"
echo "  summary:       $SUMMARY_JSON"
