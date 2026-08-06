#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/broca_runtime_crust.sh
broca_resolve_runtime

BASELINE_ROOT="${BROCA_BASELINE_ROOT:-crates/domains/symthaea-broca/data/models/baselines}"
BASELINE_NAME="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="$BASELINE_ROOT/$BASELINE_NAME"

if [[ -e "$OUT_DIR" ]]; then
  echo "[broca] baseline already exists: $OUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[broca] capturing baseline artifacts in $OUT_DIR"
scripts/broca_measurement_artifacts.sh "$OUT_DIR"

{
  echo "schema_version=1"
  echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "baseline_name=$BASELINE_NAME"
  echo "git_rev=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  echo "broca_eval_limit=${BROCA_EVAL_LIMIT:-8}"
  echo "broca_max_direct_drift=${BROCA_MAX_DIRECT_DRIFT:-1.10}"
  echo "broca_max_mamba_drift=${BROCA_MAX_MAMBA_DRIFT:-1.10}"
  echo "broca_max_hallucination_rate=${BROCA_MAX_HALLUCINATION_RATE:-1.0}"
  echo "broca_min_structured_validity=${BROCA_MIN_STRUCTURED_VALIDITY:-0.5}"
  echo "broca_min_structured_required_role_rate=${BROCA_MIN_STRUCTURED_REQUIRED_ROLE_RATE:-1.0}"
  echo "broca_min_structured_translation_validity=${BROCA_MIN_STRUCTURED_TRANSLATION_VALIDITY:-0.75}"
  echo "broca_min_structured_translation_grounding_rate=${BROCA_MIN_STRUCTURED_TRANSLATION_GROUNDING_RATE:-1.0}"
  echo "broca_max_structured_translation_drift=${BROCA_MAX_STRUCTURED_TRANSLATION_DRIFT:-0.25}"
  echo "broca_include_structured_molecule=${BROCA_INCLUDE_STRUCTURED_MOLECULE:-0}"
  echo "broca_cargo_unlocked=${BROCA_CARGO_UNLOCKED:-0}"
  broca_write_runtime_manifest
  echo "broca_skip_exercism=${BROCA_SKIP_EXERCISM:-0}"
  echo "broca_exercism_attempts=${BROCA_EXERCISM_ATTEMPTS:-1}"
  echo "broca_exercism_max_exercises=${BROCA_EXERCISM_MAX_EXERCISES:-0}"
  echo "broca_exercism_timeout=${BROCA_EXERCISM_TIMEOUT:-}"
  echo "broca_external_baseline=${BROCA_EXTERNAL_BASELINE:-0}"
} > "$OUT_DIR/baseline-manifest.env"

echo "[broca] baseline ready: $OUT_DIR"
