#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "\${BASH_SOURCE[0]}")/../.." && pwd)"
GATE="$ROOT/scripts/research/photonic-preregistration-gate.sh"
TEMPLATE="$ROOT/docs/research/quantum/benchmarks/PHOT-CTRL-003A-PREREGISTRATION-TEMPLATE.json"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

command -v jq >/dev/null
bash -n "$GATE"
jq -e . "$TEMPLATE" >/dev/null

bash "$GATE" --assert-template-blocked "$TEMPLATE"

if bash "$GATE" --require-ready "$TEMPLATE" >"$TMP/template.out" 2>&1; then
  echo "FAIL: unfinished template unexpectedly passed the readiness gate." >&2
  exit 1
fi
grep -q '^BLOCKED:' "$TMP/template.out"
echo "PASS: unfinished template is rejected."

HASH="$(printf 'a%.0s' {1..64})"
COMMIT="$(printf 'b%.0s' {1..40})"

# Synthetic values are test-only and must never be used to authorize real benchmark runs.
jq \
  --arg h "$HASH" \
  --arg commit "$COMMIT" \
  '
  .status = "preregistered"
  | .benchmark_sources |= map(.instance_manifest_sha256 = $h | .catalog_snapshot_sha256 = $h)
  | .canonicalization.graph_encoding = "SYMTHEA-MIS-GRAPH-V1"
  | .canonicalization.canonicalization_code_sha256 = $h
  | .canonicalization.per_instance_graph_sha256 = ([.benchmark_sources[].instance_ids[]] | map({key: ., value: $h}) | from_entries)
  | .canonicalization.target_manifest_sha256 = $h
  | .target_policy.per_instance_targets = ([.benchmark_sources[].instance_ids[]] | map({key: ., value: 1}) | from_entries)
  | .target_policy.independent_verifier_source_sha256 = $h
  | .target_policy.independent_verifier_build_sha256 = $h
  | .splits.development_families = ["family-dev"]
  | .splits.validation_families = ["family-validation"]
  | .splits.confirmation_families = ["family-confirmation"]
  | .splits.split_manifest_sha256 = $h
  | .arms |= map(if .operational == true then .source_revision = $commit | .configuration_sha256 = $h else . end)
  | .budgets.solver_budget_per_trial = 1000
  | .budgets.tuning_budget_per_operational_arm = 5000
  | .budgets.timeout_seconds_per_trial = 60
  | .budgets.maximum_retries = 2
  | .budgets.queue_policy = "record-and-charge-queue"
  | .budgets.compile_initialization_amortization_policy = "include-deployment; standardized-excludes"
  | .budgets.matched_hardware_resource_policy = "same-host-same-thread-count"
  | .repetitions.seed_list = [range(1; 11)]
  | .repetitions.arm_order_randomization_policy = "seeded-block-randomization"
  | .primary_endpoint.horizon_seconds = 3600
  | .primary_endpoint.number_of_bootstrap_replicates = 1000
  | .cost_boundary.energy_scope = "not-measured-explicitly"
  | .environment.operating_system = "synthetic-linux"
  | .environment.cpu_model = "synthetic-ci-cpu"
  | .environment.ram_bytes = 1000000000
  | .environment.thread_count = 2
  | .environment.container_or_nix_derivation_sha256 = $h
  | .environment.dependency_lock_sha256 = $h
  | .environment.compiler_versions = {"rustc": "1.96.0"}
  | .result_artifacts.per_attempt_record_schema_version = "test-v1"
  | .result_artifacts.raw_event_log_location = "artifacts/raw-events/"
  | .result_artifacts.verifier_receipt_location = "artifacts/receipts/"
  | .result_artifacts.result_manifest_sha256 = $h
  ' "$TEMPLATE" > "$TMP/complete.json"

bash "$GATE" --require-ready "$TMP/complete.json" "$TMP/ready-receipt.json"
jq -e '
  .outcome == "preregistration-complete"
  and .gate_id == "PHOT-CTRL-004"
  and (.manifest_sha256 | test("^[0-9a-f]{64}$"))
  and (.checker_sha256 | test("^[0-9a-f]{64}$"))
  and (.scope | contains("completeness-only"))
' "$TMP/ready-receipt.json" >/dev/null
echo "PASS: synthetic complete fixture produces a content-bound readiness receipt."

jq '.status = "preregistered"' "$TEMPLATE" > "$TMP/falsely-registered.json"
if bash "$GATE" --require-ready "$TMP/falsely-registered.json" >"$TMP/falsely-registered.out" 2>&1; then
  echo "FAIL: status-only preregistration unexpectedly passed." >&2
  exit 1
fi
grep -q 'canonical graph digests for every pinned instance' "$TMP/falsely-registered.out"
echo "PASS: changing status alone does not bypass completeness checks."

jq '.splits.confirmation_families = ["family-dev"]' "$TMP/complete.json" > "$TMP/overlap.json"
if bash "$GATE" --require-ready "$TMP/overlap.json" >"$TMP/overlap.out" 2>&1; then
  echo "FAIL: overlapping graph-family split unexpectedly passed." >&2
  exit 1
fi
grep -q 'family splits are nonempty, unique, disjoint and hashed' "$TMP/overlap.out"
echo "PASS: overlapping family splits are rejected."

FIRST_ID="$(jq -r '.benchmark_sources[0].instance_ids[0]' "$TMP/complete.json")"
jq --arg id "$FIRST_ID" 'del(.canonicalization.per_instance_graph_sha256[$id])' "$TMP/complete.json" > "$TMP/missing-digest.json"
if bash "$GATE" --require-ready "$TMP/missing-digest.json" >"$TMP/missing-digest.out" 2>&1; then
  echo "FAIL: missing per-instance graph digest unexpectedly passed." >&2
  exit 1
fi
grep -q 'canonical graph digests for every pinned instance' "$TMP/missing-digest.out"
echo "PASS: missing graph digests are rejected."

echo "All PHOT-CTRL-004 readiness-gate tests passed."
