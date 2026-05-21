#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# A/B lane for coding structural memory:
#   1. run a deterministic benchmark and export distillation + compact prototypes
#   2. rerun the same lane with both memories loaded into a fresh orchestrator
#   3. emit a compact comparison JSON for trend tracking

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${CODING_BACKEND_STRUCTURAL_AB_OUT_DIR:-/tmp/symthaea-coding-backend-structural-memory-ab}"
LANE="${CODING_BACKEND_STRUCTURAL_AB_LANE:-hard}"
WITHOUT_DIR="$OUT_DIR/without-memory"
WITH_DIR="$OUT_DIR/with-memory"
WITHOUT_REPORT="$WITHOUT_DIR/benchmark-coding-backends-$LANE.json"
WITH_REPORT="$WITH_DIR/benchmark-coding-backends-$LANE.json"
DISTILLATION_JSONL="$WITHOUT_DIR/distillation-$LANE.jsonl"
PROTOTYPES_JSON="$WITHOUT_DIR/structural-prototypes-$LANE.json"
SUMMARY_JSON="$OUT_DIR/structural-memory-ab-summary.json"

mkdir -p "$WITHOUT_DIR" "$WITH_DIR"

echo "[structural-memory-ab] baseline run without structural memory"
CODING_BACKEND_LANE="$LANE" \
CODING_BACKEND_OUT_DIR="$WITHOUT_DIR" \
CODING_BACKEND_REPORT="$WITHOUT_REPORT" \
CODING_BACKEND_DISTILLATION_OUT="$DISTILLATION_JSONL" \
CODING_BACKEND_STRUCTURAL_PROTOTYPES_OUT="$PROTOTYPES_JSON" \
CODING_BACKEND_SKIP_BASELINE="${CODING_BACKEND_STRUCTURAL_AB_SKIP_BASELINE:-1}" \
scripts/coding_backend_regression.sh

echo "[structural-memory-ab] comparison run with imported structural memory"
CODING_BACKEND_LANE="$LANE" \
CODING_BACKEND_OUT_DIR="$WITH_DIR" \
CODING_BACKEND_REPORT="$WITH_REPORT" \
CODING_BACKEND_DISTILLATION_IN="$DISTILLATION_JSONL" \
CODING_BACKEND_STRUCTURAL_PROTOTYPES_IN="$PROTOTYPES_JSON" \
CODING_BACKEND_SKIP_BASELINE=1 \
scripts/coding_backend_regression.sh

echo "[structural-memory-ab] writing comparison summary"
python3 - "$WITHOUT_REPORT" "$WITH_REPORT" "$SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

without_path, with_path, summary_path = map(Path, sys.argv[1:])
without = json.loads(without_path.read_text())
with_memory = json.loads(with_path.read_text())

def number(report, key, default=0.0):
    value = report.get(key)
    return default if value is None else value

summary = {
    "lane": with_memory.get("benchmark"),
    "without_memory_report": str(without_path),
    "with_memory_report": str(with_path),
    "distillation_imported": with_memory.get("distillation_imported", 0),
    "structural_prototype_imported": with_memory.get("structural_prototype_imported", False),
    "pass_rate_without": number(without, "pass_rate"),
    "pass_rate_with": number(with_memory, "pass_rate"),
    "quality_pass_rate_without": number(without, "quality_pass_rate"),
    "quality_pass_rate_with": number(with_memory, "quality_pass_rate"),
    "mean_attempts_without": number(without, "mean_attempts_per_task"),
    "mean_attempts_with": number(with_memory, "mean_attempts_per_task"),
    "structural_prior_score_without": number(without, "mean_structural_prior_score", None),
    "structural_prior_score_with": number(with_memory, "mean_structural_prior_score", None),
    "structural_prior_observations_without": without.get("structural_prior_observations", 0),
    "structural_prior_observations_with": with_memory.get("structural_prior_observations", 0),
}
summary["pass_rate_delta"] = summary["pass_rate_with"] - summary["pass_rate_without"]
summary["quality_pass_rate_delta"] = (
    summary["quality_pass_rate_with"] - summary["quality_pass_rate_without"]
)
summary["mean_attempts_delta"] = (
    summary["mean_attempts_with"] - summary["mean_attempts_without"]
)
if summary["structural_prior_score_without"] is not None and summary["structural_prior_score_with"] is not None:
    summary["structural_prior_score_delta"] = (
        summary["structural_prior_score_with"] - summary["structural_prior_score_without"]
    )
else:
    summary["structural_prior_score_delta"] = None

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

if summary["distillation_imported"] <= 0:
    raise SystemExit("structural memory A/B failed: no distillation records imported")
if not summary["structural_prototype_imported"]:
    raise SystemExit("structural memory A/B failed: compact prototypes were not imported")
if summary["pass_rate_delta"] < -0.001:
    raise SystemExit("structural memory A/B failed: pass rate regressed")
if summary["quality_pass_rate_delta"] < -0.001:
    raise SystemExit("structural memory A/B failed: quality pass rate regressed")
PY

echo "[structural-memory-ab] PASS"
echo "  without-memory report: $WITHOUT_REPORT"
echo "  with-memory report:    $WITH_REPORT"
echo "  summary:               $SUMMARY_JSON"
echo "  distillation jsonl:    $DISTILLATION_JSONL"
echo "  prototypes json:       $PROTOTYPES_JSON"
