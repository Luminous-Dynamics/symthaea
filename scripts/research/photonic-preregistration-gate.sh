#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  photonic-preregistration-gate.sh --assert-template-blocked MANIFEST.json
  photonic-preregistration-gate.sh --require-ready MANIFEST.json [RECEIPT.json]

The checker only evaluates frozen preregistration completeness. It never launches a solver
or reads confirmation outcomes. A generated receipt is a content-bound preflight record,
not a signature, approval, or solver-run authorization by itself.
USAGE
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

MODE="$1"
MANIFEST="$2"
RECEIPT="\${3:-}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest file does not exist: $MANIFEST" >&2
  exit 2
fi
if ! jq -e 'type == "object"' "$MANIFEST" >/dev/null 2>&1; then
  echo "ERROR: manifest is not valid JSON object: $MANIFEST" >&2
  exit 2
fi

if [[ "$MODE" == "--assert-template-blocked" ]]; then
  status="$(jq -r '.status // ""' "$MANIFEST")"
  if [[ "$status" != "template-not-preregistered" ]]; then
    echo "ERROR: expected the source template to remain template-not-preregistered; found '$status'." >&2
    exit 1
  fi
  echo "PASS: source template is explicitly blocked from confirmation use."
  exit 0
fi

if [[ "$MODE" != "--require-ready" ]]; then
  usage
  exit 2
fi

missing=()
require() {
  local field="$1"
  local expression="$2"
  if ! jq -e "$expression" "$MANIFEST" >/dev/null 2>&1; then
    missing+=("$field")
  fi
}

require "status" '.status == "preregistered"'
require "protocol_id/schema_version/freeze_required_before_confirmation" '.protocol_id == "PHOT-CTRL-003A" and (.schema_version | type == "string" and length > 0) and .freeze_required_before_confirmation == true'
require "benchmark_sources" '(.benchmark_sources | type == "array" and length > 0)'
require "benchmark source immutable revisions" '(.benchmark_sources | all(.[]; (.revision_or_release | type == "string" and test("^[0-9a-f]{40}([0-9a-f]{24})?$"))))'
require "benchmark instance manifests and catalog hashes" '(.benchmark_sources | all(.[]; (.instance_ids | type == "array" and length > 0) and (.instance_manifest_sha256 | type == "string" and test("^[0-9a-f]{64}$")) and (.catalog_snapshot_sha256 | type == "string" and test("^[0-9a-f]{64}$"))))'
require "unique benchmark instance IDs" '. as $root | ([$root.benchmark_sources[].instance_ids[]]) as $ids | (($ids | unique | length) == ($ids | length))'
require "canonical graph encoding and code digest" '(.canonicalization.graph_encoding | type == "string" and length > 0 and . != "TBD") and (.canonicalization.canonicalization_code_sha256 | type == "string" and test("^[0-9a-f]{64}$"))'
require "canonical graph digests for every pinned instance" '. as $root | ([$root.benchmark_sources[].instance_ids[]]) as $ids | all($ids[]; . as $id | (($root.canonicalization.per_instance_graph_sha256[$id] // "") | test("^[0-9a-f]{64}$")))'
require "target manifest digest and target value for every instance" '. as $root | (($root.canonicalization.target_manifest_sha256 | type == "string" and test("^[0-9a-f]{64}$")) and ($root.target_policy.per_instance_targets | type == "object" and length > 0) and ([$root.benchmark_sources[].instance_ids[]] as $ids | all($ids[]; . as $id | (($root.target_policy.per_instance_targets[$id] // 0) | (type == "number" and floor == . and . >= 1)))))'
require "independent verifier source/build digests" '(.target_policy.independent_verifier_source_sha256 | type == "string" and test("^[0-9a-f]{64}$")) and (.target_policy.independent_verifier_build_sha256 | type == "string" and test("^[0-9a-f]{64}$"))'
require "family split unit" '.splits.unit == "graph-family"'
require "family splits are nonempty, unique, disjoint and hashed" '. as $root | (["development_families","validation_families","confirmation_families"] | all(.[]; . as $k | ($root.splits[$k] | type == "array" and length > 0 and (unique | length) == length))) and ([$root.splits.development_families[]?, $root.splits.validation_families[]?, $root.splits.confirmation_families[]?] as $families | ($families | unique | length) == ($families | length)) and ($root.splits.split_manifest_sha256 | type == "string" and test("^[0-9a-f]{64}$"))'
require "all operational solver arms pinned" '(.arms | type == "array" and length > 0) and (.arms | any(.[]; .id == "B" and .operational == true)) and (.arms | all(.[]; (.operational != true) or ((.source_revision | type == "string" and test("^[0-9a-f]{40}([0-9a-f]{24})?$")) and (.configuration_sha256 | type == "string" and test("^[0-9a-f]{64}$")))))'
require "positive matched solver/tuning/time budgets and declared policies" '(.budgets.solver_budget_per_trial | type == "number" and . > 0) and (.budgets.tuning_budget_per_operational_arm | type == "number" and . > 0) and (.budgets.timeout_seconds_per_trial | type == "number" and . > 0) and (.budgets.maximum_retries | type == "number" and floor == . and . >= 0) and (.budgets.queue_policy | type == "string" and length > 0 and (ascii_downcase | contains("tbd") | not)) and (.budgets.compile_initialization_amortization_policy | type == "string" and length > 0 and (ascii_downcase | contains("tbd") | not)) and (.budgets.matched_hardware_resource_policy | type == "string" and length > 0 and (ascii_downcase | contains("tbd") | not))'
require "repetition count, distinct seeds and arm-order randomization" '(.repetitions.minimum_independent_stochastic_runs_per_arm_instance | type == "number" and floor == . and . >= 10) and (.repetitions.seed_list | type == "array" and length >= 10 and (unique | length) == length) and (.repetitions.arm_order_randomization_policy | type == "string" and length > 0 and (ascii_downcase | contains("tbd") | not))'
require "primary metric horizon/comparator/uncertainty plan" '. as $root | $root.primary_endpoint.name == "restricted-mean-all-in-time-to-verified-target" and ($root.primary_endpoint.horizon_seconds | type == "number" and . > 0) and ($root.primary_endpoint.comparison_arm | type == "string" and length > 0) and ($root.arms | any(.[]; .id == $root.primary_endpoint.comparison_arm and .operational == true)) and ($root.primary_endpoint.uncertainty_method | type == "string" and length > 0) and ($root.primary_endpoint.confidence_level | type == "number" and . > 0 and . < 1) and ($root.primary_endpoint.number_of_bootstrap_replicates | type == "number" and floor == . and . >= 1000)'
require "decision gates frozen" '.decision_gates.zero_false_target_certificates == true and (.decision_gates.target_hit_probability_difference_lower_bound_minimum | type == "number" and . >= -1 and . <= 0) and (.decision_gates.champion_over_candidate_restricted_mean_time_ratio_goal | type == "number" and . > 1) and .decision_gates["95_percent_interval_must_exclude_no_improvement"] == true and .decision_gates.trajectory_features_must_beat_final_spin_and_graph_only_controls == true and .decision_gates.adaptive_acquisition_must_beat_fixed_K_at_matched_budget == true and .decision_gates.shuffled_feature_control_must_not_reproduce_treatment_effect == true and .decision_gates.publish_negative_results == true'
require "cost boundary and energy scope explicit" '(.cost_boundary.standardized_algorithm_runtime_excludes | type == "array" and length > 0) and (.cost_boundary.deployment_all_in_latency_includes | type == "array" and length > 0) and .cost_boundary.unknown_cost_is_zero == false and (.cost_boundary.energy_scope | type == "string" and length > 0 and (ascii_downcase | contains("tbd") | not))'
require "execution environment/compiler/lockfile frozen" '(.environment.operating_system | type == "string" and length > 0) and (.environment.cpu_model | type == "string" and length > 0) and (.environment.ram_bytes | type == "number" and floor == . and . > 0) and (.environment.thread_count | type == "number" and floor == . and . > 0) and (.environment.container_or_nix_derivation_sha256 | type == "string" and test("^[0-9a-f]{64}$")) and (.environment.dependency_lock_sha256 | type == "string" and test("^[0-9a-f]{64}$")) and (.environment.compiler_versions | type == "object" and length > 0 and all(.[]; type == "string" and length > 0))'
require "result record/event log/receipt schema and output locations" '(.result_artifacts.per_attempt_record_schema_version | type == "string" and length > 0) and (.result_artifacts.raw_event_log_location | type == "string" and length > 0) and (.result_artifacts.verifier_receipt_location | type == "string" and length > 0) and (.result_artifacts.result_manifest_sha256 | type == "string" and test("^[0-9a-f]{64}$"))'

if (( \${#missing[@]} > 0 )); then
  printf 'BLOCKED: preregistration is incomplete (%d requirement groups failed):\n' "\${#missing[@]}" >&2
  for field in "\${missing[@]}"; do
    printf '  - %s\n' "$field" >&2
  done
  exit 1
fi

manifest_sha256="$(sha256sum "$MANIFEST" | awk '{print $1}')"
checker_path="$(cd -- "$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)/$(basename -- "\${BASH_SOURCE[0]}")"
checker_sha256="$(sha256sum "$checker_path" | awk '{print $1}')"
echo "PASS: preregistration manifest is complete under PHOT-CTRL-004."
echo "Manifest SHA-256: $manifest_sha256"
if [[ -n "$RECEIPT" ]]; then
  jq -n \
    --arg protocol_id "PHOT-CTRL-003A" \
    --arg gate_id "PHOT-CTRL-004" \
    --arg manifest_sha256 "$manifest_sha256" \
    --arg checker_sha256 "$checker_sha256" \
    '{
      schema_version: "1.0",
      protocol_id: $protocol_id,
      gate_id: $gate_id,
      outcome: "preregistration-complete",
      manifest_sha256: $manifest_sha256,
      checker_sha256: $checker_sha256,
      scope: "completeness-only; not independent review, statistical power, execution authorization, or a performance result"
    }' > "$RECEIPT"
  echo "Readiness receipt written: $RECEIPT"
fi
