#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/broca_runtime_crust.sh

OUT_DIR="${1:-target/broca-measurements}"
mkdir -p "$OUT_DIR"

export RUSTC_WRAPPER="${RUSTC_WRAPPER:-}"
export SCCACHE_DISABLE="${SCCACHE_DISABLE:-1}"

echo "[broca] writing measurement artifacts to $OUT_DIR"

cargo_locked_args=()
if [[ "${BROCA_CARGO_UNLOCKED:-0}" != "1" ]]; then
  cargo_locked_args+=(--locked)
fi

broca_resolve_runtime
broca_print_runtime

write_failed_exercism_artifact() {
  local reason="$1"
  cat > "$OUT_DIR/exercism-bench.json" <<JSON
{
  "schema_version": 1,
  "evidence_level": "failed-measurement",
  "measured": false,
  "benchmark": "broca-exercism-rust-fixtures",
  "total_exercises": 0,
  "compile_successes": 0,
  "test_successes": 0,
  "cases": [],
  "error": "$reason"
}
JSON
}

{
  echo "schema_version=1"
  echo "created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
  echo "broca_baseline_artifact_dir=${BROCA_BASELINE_ARTIFACT_DIR:-}"
  echo "broca_require_all_comparison_metrics=${BROCA_REQUIRE_ALL_COMPARISON_METRICS:-}"
  echo "broca_fail_on_regression=${BROCA_FAIL_ON_REGRESSION:-}"
} > "$OUT_DIR/measurement-manifest.env"

decoder_cmd=(
  cargo run "${cargo_locked_args[@]}" -p symthaea-broca --features "$BROCA_DECODER_FEATURES" --bin broca-decoder-ab --
  --decoder structured
  --eval-limit "${BROCA_EVAL_LIMIT:-8}"
  --max-direct-drift "${BROCA_MAX_DIRECT_DRIFT:-1.10}"
  --max-mamba-drift "${BROCA_MAX_MAMBA_DRIFT:-1.10}"
  --max-hallucination-rate "${BROCA_MAX_HALLUCINATION_RATE:-1.0}"
  --min-structured-validity "${BROCA_MIN_STRUCTURED_VALIDITY:-0.5}"
  --min-structured-required-role-rate "${BROCA_MIN_STRUCTURED_REQUIRED_ROLE_RATE:-1.0}"
  --min-structured-translation-validity "${BROCA_MIN_STRUCTURED_TRANSLATION_VALIDITY:-0.75}"
  --min-structured-translation-grounding-rate "${BROCA_MIN_STRUCTURED_TRANSLATION_GROUNDING_RATE:-1.0}"
  --max-structured-translation-drift "${BROCA_MAX_STRUCTURED_TRANSLATION_DRIFT:-0.25}"
  --json-out "$OUT_DIR/decoder-ab.json"
)
if [[ "${BROCA_INCLUDE_STRUCTURED_MOLECULE:-0}" == "1" ]]; then
  decoder_cmd+=(--include-structured-molecule)
fi
"${decoder_cmd[@]}"

if [[ "${BROCA_SKIP_EXERCISM:-0}" == "1" ]]; then
  cat > "$OUT_DIR/exercism-bench.json" <<'JSON'
{
  "schema_version": 1,
  "evidence_level": "skipped",
  "measured": false,
  "benchmark": "broca-exercism-rust-fixtures",
  "total_exercises": 0,
  "compile_successes": 0,
  "test_successes": 0,
  "cases": []
}
JSON
else
  exercism_cmd=(
    cargo run "${cargo_locked_args[@]}" -p symthaea-broca --features "$BROCA_EXERCISM_FEATURES" --bin broca-exercism-bench --
    --max-attempts "${BROCA_EXERCISM_ATTEMPTS:-1}"
    --max-exercises "${BROCA_EXERCISM_MAX_EXERCISES:-0}"
    --json-out "$OUT_DIR/exercism-bench.json"
  )
  if [[ -n "${BROCA_EXERCISM_TIMEOUT:-}" ]]; then
    set +e
    timeout "$BROCA_EXERCISM_TIMEOUT" "${exercism_cmd[@]}"
    status=$?
    set -e
    if [[ "$status" -ne 0 ]]; then
      write_failed_exercism_artifact "exercism measurement failed or timed out with status $status"
      if [[ "${BROCA_EXERCISM_REQUIRED:-0}" == "1" ]]; then
        exit "$status"
      fi
    fi
  else
    set +e
    "${exercism_cmd[@]}"
    status=$?
    set -e
    if [[ "$status" -ne 0 ]]; then
      write_failed_exercism_artifact "exercism measurement failed with status $status"
      if [[ "${BROCA_EXERCISM_REQUIRED:-0}" == "1" ]]; then
        exit "$status"
      fi
    fi
  fi
fi

if [[ "${BROCA_EXTERNAL_BASELINE:-0}" == "1" ]]; then
  cargo run "${cargo_locked_args[@]}" -p symthaea-broca --bin broca-external-baseline -- \
    --url "${BROCA_EXTERNAL_BASELINE_URL:-http://127.0.0.1:11434/api/generate}" \
    --model "${BROCA_EXTERNAL_BASELINE_MODEL:-gemma3:latest}" \
    --prompt "${BROCA_EXTERNAL_BASELINE_PROMPT:-Translate this semantic state into a concise response.}" \
    --json-out "$OUT_DIR/external-baseline.json"
fi

if [[ -n "${BROCA_BASELINE_ARTIFACT_DIR:-}" ]]; then
  cargo run "${cargo_locked_args[@]}" -p symthaea-broca --bin broca-checkpoint-compare -- \
    --baseline-dir "$BROCA_BASELINE_ARTIFACT_DIR" \
    --candidate-dir "$OUT_DIR" \
    --json-out "$OUT_DIR/checkpoint-compare.json" \
    --max-drift-regression "${BROCA_MAX_DRIFT_REGRESSION:-0.05}" \
    --max-hallucination-regression "${BROCA_MAX_HALLUCINATION_REGRESSION:-0.05}" \
    --max-compile-rate-regression "${BROCA_MAX_COMPILE_RATE_REGRESSION:-0.0}" \
    --max-test-rate-regression "${BROCA_MAX_TEST_RATE_REGRESSION:-0.0}" \
    --max-structured-confidence-regression "${BROCA_MAX_STRUCTURED_CONFIDENCE_REGRESSION:-0.05}" \
    --max-structured-validity-regression "${BROCA_MAX_STRUCTURED_VALIDITY_REGRESSION:-0.05}" \
    --max-structured-required-role-rate-regression "${BROCA_MAX_STRUCTURED_REQUIRED_ROLE_RATE_REGRESSION:-0.0}" \
    --max-structured-translation-validity-regression "${BROCA_MAX_STRUCTURED_TRANSLATION_VALIDITY_REGRESSION:-0.02}" \
    --max-structured-translation-grounding-rate-regression "${BROCA_MAX_STRUCTURED_TRANSLATION_GROUNDING_RATE_REGRESSION:-0.0}" \
    --max-structured-translation-drift-regression "${BROCA_MAX_STRUCTURED_TRANSLATION_DRIFT_REGRESSION:-0.02}" \
    ${BROCA_REQUIRE_ALL_COMPARISON_METRICS:+--require-all-metrics} \
    ${BROCA_FAIL_ON_REGRESSION:+--fail-on-regression}
fi

scripts/broca_measurement_summary.py "$OUT_DIR"

echo "[broca] measurement artifacts complete"
