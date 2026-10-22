#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

manifest="tools/allostatic-control-bench/Cargo.toml"
source_files=(
  ".github/workflows/allostatic-control-bench.yml"
  "tools/allostatic-control-bench/Cargo.toml"
  "tools/allostatic-control-bench/Cargo.lock"
  "tools/allostatic-control-bench/src/lib.rs"
  "tools/allostatic-control-bench/src/calibration.rs"
  "tools/allostatic-control-bench/src/certificate_lease.rs"
  "tools/allostatic-control-bench/src/model_mismatch.rs"
  "tools/allostatic-control-bench/src/multistep_viability.rs"
  "tools/allostatic-control-bench/src/robust_viability.rs"
  "tools/allostatic-control-bench/src/main.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-control-report.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-model-mismatch-bench.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-multistep-viability-bench.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-certificate-lease-bench.rs"
  "tools/allostatic-control-bench/qualify.sh"
  "docs/research/ALLOSTATIC_CONTROL_BENCHMARK_V1.md"
  "docs/research/ROBUST_VIABILITY_ENVELOPE_V1.md"
  "docs/research/MULTISTEP_VIABILITY_TUBES_V1.md"
  "docs/research/ASSUMPTION_BOUND_CERTIFICATE_LEASES_V1.md"
)
expected_head="${1:-$(git rev-parse HEAD)}"
actual_head="$(git rev-parse HEAD)"
if [[ "$actual_head" != "$expected_head" ]]; then
  printf 'FAIL: expected checkout %s, found %s\n' "$expected_head" "$actual_head" >&2
  exit 2
fi

for file in "${source_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    printf 'FAIL: missing qualification input: %s\n' "$file" >&2
    exit 2
  fi
done

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
out_csv="${2:-allostatic-control-v1.csv}"
out_base="${out_csv%.csv}"
summary_out="${out_base}-summary.csv"
paired_out="${out_base}-paired-deltas.csv"
mismatch_out="${out_base}-mismatch-frontier.csv"
mismatch_summary_out="${out_base}-mismatch-frontier-summary.csv"
horizon_out="${out_base}-multistep-horizon.csv"
horizon_summary_out="${out_base}-multistep-horizon-summary.csv"
certificate_lease_out="${out_base}-certificate-leases.csv"
certificate_lease_summary_out="${out_base}-certificate-leases-summary.csv"
receipt="${out_csv}.receipt.txt"

cargo fmt --manifest-path "$manifest" -- --check
cargo test --locked --manifest-path "$manifest"
cargo clippy --locked --manifest-path "$manifest" --all-targets -- -D warnings

cargo run --locked --release --manifest-path "$manifest" --bin allostatic-control-bench > "$tmpdir/run-a.csv"
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-control-bench > "$tmpdir/run-b.csv"
if ! cmp -s "$tmpdir/run-a.csv" "$tmpdir/run-b.csv"; then
  printf 'FAIL: repeated benchmark output is not byte-identical\n' >&2
  exit 1
fi

# Require the full scenario × seed × policy Cartesian product and a stable schema.
if ! awk -F, '
  NR == 1 { if (NF != 29) failed = 1; next }
  {
    if (NF != 29) failed = 1
    if ($1 != "calm" && $1 != "acute" && $1 != "chronic" &&
        $1 != "repeated" && $1 != "regime_shift") failed = 1
    if ($2 !~ /^[0-9]+$/ || $2 < 0 || $2 >= 32) failed = 1
    if ($3 != "task_only" && $3 != "stress_reactive" &&
        $3 != "allostatic_reactive" && $3 != "predictive_viability" &&
        $3 != "shuffled_forecast_control" && $3 != "oracle_predictive") failed = 1
    if ($4 != "true" && $4 != "false") failed = 1
    key = $1 SUBSEP $2 SUBSEP $3 SUBSEP $4
    if (seen[key]++) failed = 1
    rows++
  }
  END { if (NR != 1921 || rows != 1920 || failed) exit 1 }
' "$tmpdir/run-a.csv"; then
  printf 'FAIL: CSV schema or scenario/seed/policy/shield coverage is invalid\n' >&2
  exit 1
fi

cargo run --locked --release --manifest-path "$manifest" --bin allostatic-control-report -- \
  "$tmpdir/run-a.csv" "$tmpdir/summary-a.csv" "$tmpdir/paired-a.csv" > "$tmpdir/report-a.log"
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-control-report -- \
  "$tmpdir/run-b.csv" "$tmpdir/summary-b.csv" "$tmpdir/paired-b.csv" > "$tmpdir/report-b.log"
if ! cmp -s "$tmpdir/summary-a.csv" "$tmpdir/summary-b.csv" ||
   ! cmp -s "$tmpdir/paired-a.csv" "$tmpdir/paired-b.csv"; then
  printf 'FAIL: repeated distributional report is not byte-identical\n' >&2
  exit 1
fi

# Compare global and action-specific envelopes at matched 90% and 95% joint targets,
# across five operating points. Out-of-envelope profiles are negative controls.
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-model-mismatch-bench -- \
  "$tmpdir/mismatch-a.csv" "$tmpdir/mismatch-summary-a.csv"
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-model-mismatch-bench -- \
  "$tmpdir/mismatch-b.csv" "$tmpdir/mismatch-summary-b.csv"
if ! cmp -s "$tmpdir/mismatch-a.csv" "$tmpdir/mismatch-b.csv" ||
   ! cmp -s "$tmpdir/mismatch-summary-a.csv" "$tmpdir/mismatch-summary-b.csv"; then
  printf 'FAIL: repeated matched-coverage frontier is not byte-identical\n' >&2
  exit 1
fi

# Five operating points × five stress levels × three actions; profiles are paired
# across all four envelope methods and all operating points.
if ! awk -F, '
  NR == 1 { if (NF != 22) failed = 1; next }
  {
    if (NF != 22) failed = 1
    if ($1 != "within_envelope" && $1 != "out_of_envelope") failed = 1
    if ($2 != "critical_low_reserve_high_load" && $2 != "reserve_limited" &&
        $2 != "load_limited" && $2 != "balanced" && $2 != "high_margin") failed = 1
    if ($5 !~ /^[0-9]+$/) failed = 1
    if ($7 != "push" && $7 != "maintain" && $7 != "emergency_recover") failed = 1
    for (i = 8; i <= 22; i++) if ($i != "true" && $i != "false") failed = 1
    if ($11 == "true" && $9 != "true") failed = 1
    if ($12 == "true" && $10 != "true") failed = 1
    if ($1 == "within_envelope") {
      within_rows++
      if (($9 == "true" || $10 == "true" || $11 == "true" || $12 == "true") &&
          $13 == "false") failed = 1
    }
    if ($1 == "out_of_envelope") out_rows++
    key = $1 SUBSEP $2 SUBSEP $5 SUBSEP $6 SUBSEP $7
    if (seen[key]++) failed = 1
  }
  END { if (NR != 24901 || within_rows != 22500 || out_rows != 2400 || failed) exit 1 }
' "$tmpdir/mismatch-a.csv"; then
  printf 'FAIL: matched-coverage frontier schema, coverage, monotonicity, or safety contract is invalid\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 35) failed = 1; next }
  {
    if (NF != 35) failed = 1
    if ($1 != "within_envelope" && $1 != "out_of_envelope") failed = 1
    if ($8 > $7 || $9 > $7 || $10 > $7 || $11 > $7) failed = 1
    if ($10 > $8 || $11 > $9) failed = 1
    if ($2 == "all_states") {
      aggregate_rows++
      if ($1 == "within_envelope") {
        if ($5 != 1500 || $6 != 22500 ||
            $14 != 0 || $15 != 0 || $16 != 0 || $17 != 0) failed = 1
      } else {
        if ($5 != 160 || $6 != 2400 ||
            ($14 + $15 + $16 + $17) < 1) failed = 1
      }
    } else {
      state_rows++
      if ($5 != ($1 == "within_envelope" ? 300 : 32) ||
          $6 != ($1 == "within_envelope" ? 4500 : 480)) failed = 1
      if ($10 > $8 || $11 > $9) failed = 1
      if ($1 == "within_envelope" &&
          ($14 != 0 || $15 != 0 || $16 != 0 || $17 != 0)) failed = 1
    }
  }
  END { if (NR != 13 || aggregate_rows != 2 || state_rows != 10 || failed) exit 1 }
' "$tmpdir/mismatch-summary-a.csv"; then
  printf 'FAIL: matched-coverage per-state safety/availability summary is invalid\n' >&2
  exit 1
fi

# Measure viable-action shrinkage and bounded search cost across five operating
# states, 24 deterministic dynamics panels and horizons 1, 2, 3, 5, and 8.
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-multistep-viability-bench -- \
  "$tmpdir/horizon-a.csv" "$tmpdir/horizon-summary-a.csv"
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-multistep-viability-bench -- \
  "$tmpdir/horizon-b.csv" "$tmpdir/horizon-summary-b.csv"
if ! cmp -s "$tmpdir/horizon-a.csv" "$tmpdir/horizon-b.csv" ||
   ! cmp -s "$tmpdir/horizon-summary-a.csv" "$tmpdir/horizon-summary-b.csv"; then
  printf 'FAIL: repeated multi-step raw and summary frontiers are not byte-identical\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 11) failed = 1; next }
  {
    if (NF != 11) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin") failed = 1
    if ($2 !~ /^[0-9]+$/ || $2 >= 24) failed = 1
    if ($3 != 1 && $3 != 2 && $3 != 3 && $3 != 5 && $3 != 8) failed = 1
    for (i = 4; i <= 6; i++) if ($i != "true" && $i != "false") failed = 1
    if ($7 != (($4 == "true") + ($5 == "true") + ($6 == "true"))) failed = 1
    if (($8 == "true") != ($7 == 0)) failed = 1
    if ($9 !~ /^[0-9]+$/ || $9 > 10000) failed = 1
    if ($10 !~ /^[0-9]+$/ || $10 > 10000) failed = 1
    if ($11 !~ /^[0-9]+$/ || $11 > $3) failed = 1
    key = $1 SUBSEP $2
    if (!(key in previous_horizon)) {
      profile_keys++
      if ($3 != 1) failed = 1
    } else {
      if (($3 == 2 && previous_horizon[key] != 1) ||
          ($3 == 3 && previous_horizon[key] != 2) ||
          ($3 == 5 && previous_horizon[key] != 3) ||
          ($3 == 8 && previous_horizon[key] != 5)) failed = 1
      if ($4 == "true" && previous_push[key] != "true") failed = 1
      if ($5 == "true" && previous_maintain[key] != "true") failed = 1
      if ($6 == "true" && previous_recovery[key] != "true") failed = 1
    }
    previous_horizon[key] = $3
    previous_push[key] = $4
    previous_maintain[key] = $5
    previous_recovery[key] = $6
    unique_key = $1 SUBSEP $2 SUBSEP $3
    if (seen[unique_key]++) failed = 1
    rows++
  }
  END { if (NR != 601 || rows != 600 || profile_keys != 120 || failed) exit 1 }
' "$tmpdir/horizon-a.csv"; then
  printf 'FAIL: multi-step horizon schema, monotonicity, or bounded-search contract is invalid\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 15) failed = 1; next }
  {
    if (NF != 15) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin" &&
        $1 != "all_states") failed = 1
    if ($2 != 1 && $2 != 2 && $2 != 3 && $2 != 5 && $2 != 8) failed = 1
    expected_profiles = ($1 == "all_states" ? 120 : 24)
    if ($3 != expected_profiles || $4 != expected_profiles * 3) failed = 1
    if ($5 < 0 || $5 > $4) failed = 1
    if ($6 < 0 || $6 > 1) failed = 1
    if ($7 < 0 || $7 > $3) failed = 1
    if ($8 < 0 || $8 > 1) failed = 1
    if (($5 / $4 - $6) > 0.00001 || ($6 - $5 / $4) > 0.00001) failed = 1
    if (($7 / $3 - $8) > 0.00001 || ($8 - $7 / $3) > 0.00001) failed = 1
    if ($9 < 0 || $9 > $11 || $10 > $11 || $11 > 10000) failed = 1
    if ($12 < 0 || $12 > $14 || $13 > $14 || $14 > 10000) failed = 1
    if ($15 < 0 || $15 > $2) failed = 1
    key = $1
    if (!(key in previous_horizon)) {
      groups++
      if ($2 != 1) failed = 1
    } else {
      if (($2 == 2 && previous_horizon[key] != 1) ||
          ($2 == 3 && previous_horizon[key] != 2) ||
          ($2 == 5 && previous_horizon[key] != 3) ||
          ($2 == 8 && previous_horizon[key] != 5)) failed = 1
      if ($5 > previous_certified[key]) failed = 1
    }
    previous_horizon[key] = $2
    previous_certified[key] = $5
    rowkey = $1 SUBSEP $2
    if (seen[rowkey]++) failed = 1
    rows++
  }
  END { if (NR != 31 || rows != 30 || groups != 6 || failed) exit 1 }
' "$tmpdir/horizon-summary-a.csv"; then
  printf 'FAIL: multi-step aggregate availability/runtime summary is invalid\n' >&2
  exit 1
fi

# Certificate leases must become unusable when any supporting assumption changes.
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-certificate-lease-bench -- \
  "$tmpdir/certificate-a.csv" "$tmpdir/certificate-summary-a.csv"
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-certificate-lease-bench -- \
  "$tmpdir/certificate-b.csv" "$tmpdir/certificate-summary-b.csv"
if ! cmp -s "$tmpdir/certificate-a.csv" "$tmpdir/certificate-b.csv" ||
   ! cmp -s "$tmpdir/certificate-summary-a.csv" "$tmpdir/certificate-summary-b.csv"; then
  printf 'FAIL: repeated certificate lease outputs are not byte-identical\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 11) failed = 1; next }
  {
    if (NF != 11) failed = 1
    if ($1 != "valid_within_age" && $1 != "state_revision_changed" &&
        $1 != "dynamics_revision_changed" && $1 != "limits_revision_changed" &&
        $1 != "calibration_revision_changed" && $1 != "authority_revision_changed" &&
        $1 != "monitor_epoch_changed" && $1 != "supporting_evidence_changed" &&
        $1 != "lease_expired" && $1 != "clock_regressed" &&
        $1 != "longer_horizon_requested" && $1 != "uncertified_action_not_issued" &&
        $1 != "zero_digest_context_rejected") failed = 1
    if ($2 != "true" && $2 != "false") failed = 1
    if ($3 != "true" && $3 != "false") failed = 1
    if ($4 != "push" && $4 != "maintain" && $4 != "emergency_recover") failed = 1
    if ($5 != $6) failed = 1
    if ($7 != "true" && $7 != "false") failed = 1
    if ($8 !~ /^[0-9]+$/ || $9 !~ /^[0-9]+$/ ||
        $10 !~ /^[0-9]+$/ || $11 !~ /^[0-9]+$/) failed = 1
    if (($6 == "accepted") != ($7 == "true")) failed = 1
    if (($2 == "true") != ($3 == "true")) failed = 1
    if ($1 == "valid_within_age" &&
        ($2 != "true" || $3 != "true" || $6 != "accepted" || $8 != 101 || $9 != 101)) failed = 1
    if ($1 == "state_revision_changed" && $6 != "state_revision_changed") failed = 1
    if ($1 == "dynamics_revision_changed" && $6 != "dynamics_revision_changed") failed = 1
    if ($1 == "limits_revision_changed" && $6 != "limits_revision_changed") failed = 1
    if ($1 == "calibration_revision_changed" && $6 != "calibration_revision_changed") failed = 1
    if ($1 == "authority_revision_changed" && $6 != "authority_revision_changed") failed = 1
    if ($1 == "monitor_epoch_changed" && $6 != "monitor_epoch_changed") failed = 1
    if ($1 == "supporting_evidence_changed" && $6 != "supporting_evidence_changed") failed = 1
    if ($1 == "lease_expired" && ($6 != "expired" || $9 <= $8)) failed = 1
    if ($1 == "clock_regressed" && ($6 != "clock_regressed" || $9 >= 100)) failed = 1
    if ($1 == "longer_horizon_requested" && ($6 != "horizon_insufficient" || $11 <= $10)) failed = 1
    if ($1 == "uncertified_action_not_issued" && ($2 != "false" || $6 != "issue_rejected" || $4 != "push")) failed = 1
    if ($1 == "zero_digest_context_rejected" && ($2 != "false" || $3 != "false" || $6 != "invalid_context")) failed = 1
    seen[$1]++
    rows++
  }
  END { if (NR != 14 || rows != 13 || failed) exit 1; for (key in seen) if (seen[key] != 1) exit 1 }
' "$tmpdir/certificate-a.csv"; then
  printf 'FAIL: certificate lease invalidation, horizon or fail-closed issuance contract is invalid\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 6) failed = 1; next }
  {
    if (NF != 6 || $3 != $4 || $6 != "true") failed = 1
    rows++
  }
  END { if (NR != 14 || rows != 13 || failed) exit 1 }
' "$tmpdir/certificate-summary-a.csv"; then
  printf 'FAIL: certificate lease summary does not match expected rejection reasons\n' >&2
  exit 1
fi

# Summary must contain 5 scenarios × 6 policies × 2 shield settings.
if ! awk -F, 'NR == 1 { if (NF != 28) failed = 1; next }
  { if (NF != 28) failed = 1; rows++ }
  END { if (NR != 61 || rows != 60 || failed) exit 1 }
' "$tmpdir/summary-a.csv"; then
  printf 'FAIL: summary schema or condition coverage is invalid\n' >&2
  exit 1
fi
# Paired report must contain one shield delta row per scenario × policy.
if ! awk -F, 'NR == 1 { if (NF != 18) failed = 1; next }
  { if (NF != 18) failed = 1; rows++ }
  END { if (NR != 31 || rows != 30 || failed) exit 1 }
' "$tmpdir/paired-a.csv"; then
  printf 'FAIL: paired-delta schema or coverage is invalid\n' >&2
  exit 1
fi

cp "$tmpdir/run-a.csv" "$out_csv"
cp "$tmpdir/summary-a.csv" "$summary_out"
cp "$tmpdir/paired-a.csv" "$paired_out"
cp "$tmpdir/mismatch-a.csv" "$mismatch_out"
cp "$tmpdir/mismatch-summary-a.csv" "$mismatch_summary_out"
cp "$tmpdir/horizon-a.csv" "$horizon_out"
cp "$tmpdir/horizon-summary-a.csv" "$horizon_summary_out"
cp "$tmpdir/certificate-a.csv" "$certificate_lease_out"
cp "$tmpdir/certificate-summary-a.csv" "$certificate_lease_summary_out"
{
  printf 'status=pass\n'
  printf 'git_commit=%s\n' "$actual_head"
  printf 'git_branch=%s\n' "$(git branch --show-current)"
  printf 'rustc=%s\n' "$(rustc -V)"
  printf 'cargo=%s\n' "$(cargo -V)"
  printf 'target=%s\n' "$(rustc -vV | sed -n 's/^host: //p')"
  printf 'command=cargo run --locked --release --manifest-path %s\n' "$manifest"
  printf 'deterministic_repeat=byte-identical\n'
  printf 'csv_sha256=%s\n' "$(sha256sum "$out_csv" | awk '{print $1}')"
  printf 'csv_rows_excluding_header=%s\n' "$(( $(wc -l < "$out_csv") - 1 ))"
  printf 'summary_sha256=%s\n' "$(sha256sum "$summary_out" | awk '{print $1}')"
  printf 'summary_rows_excluding_header=%s\n' "$(( $(wc -l < "$summary_out") - 1 ))"
  printf 'paired_deltas_sha256=%s\n' "$(sha256sum "$paired_out" | awk '{print $1}')"
  printf 'paired_delta_rows_excluding_header=%s\n' "$(( $(wc -l < "$paired_out") - 1 ))"
  printf 'mismatch_frontier_sha256=%s\n' "$(sha256sum "$mismatch_out" | awk '{print $1}')"
  printf 'mismatch_frontier_rows_excluding_header=%s\n' "$(( $(wc -l < "$mismatch_out") - 1 ))"
  printf 'mismatch_summary_sha256=%s\n' "$(sha256sum "$mismatch_summary_out" | awk '{print $1}')"
  printf 'mismatch_summary_rows_excluding_header=%s\n' "$(( $(wc -l < "$mismatch_summary_out") - 1 ))"
  printf 'multistep_frontier_sha256=%s\n' "$(sha256sum "$horizon_out" | awk '{print $1}')"
  printf 'multistep_frontier_rows_excluding_header=%s\n' "$(( $(wc -l < "$horizon_out") - 1 ))"
  printf 'multistep_summary_sha256=%s\n' "$(sha256sum "$horizon_summary_out" | awk '{print $1}')"
  printf 'multistep_summary_rows_excluding_header=%s\n' "$(( $(wc -l < "$horizon_summary_out") - 1 ))"
  printf 'certificate_lease_sha256=%s\n' "$(sha256sum "$certificate_lease_out" | awk '{print $1}')"
  printf 'certificate_lease_rows_excluding_header=%s\n' "$(( $(wc -l < "$certificate_lease_out") - 1 ))"
  printf 'certificate_lease_summary_sha256=%s\n' "$(sha256sum "$certificate_lease_summary_out" | awk '{print $1}')"
  printf 'certificate_lease_summary_rows_excluding_header=%s\n' "$(( $(wc -l < "$certificate_lease_summary_out") - 1 ))"
  printf '\nsource_sha256:\n'
  sha256sum "${source_files[@]}"
} > "$receipt"

printf 'Qualified episodes: %s\n' "$out_csv"
printf 'Distributional summary: %s\n' "$summary_out"
printf 'Paired shield deltas: %s\n' "$paired_out"
printf 'Model-mismatch safety frontier: %s\n' "$mismatch_out"
printf 'Model-mismatch frontier summary: %s\n' "$mismatch_summary_out"
printf 'Multi-step viability horizon frontier: %s\n' "$horizon_out"
printf 'Multi-step viability horizon summary: %s\n' "$horizon_summary_out"
printf 'Certificate lease frontier: %s\n' "$certificate_lease_out"
printf 'Certificate lease summary: %s\n' "$certificate_lease_summary_out"
printf 'Provenance receipt: %s\n' "$receipt"
