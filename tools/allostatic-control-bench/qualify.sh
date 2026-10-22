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
  "tools/allostatic-control-bench/src/drift_monitor.rs"
  "tools/allostatic-control-bench/src/bounded_risk_eprocess.rs"
  "tools/allostatic-control-bench/src/scope_monitor.rs"
  "tools/allostatic-control-bench/src/model_mismatch.rs"
  "tools/allostatic-control-bench/src/multistep_viability.rs"
  "tools/allostatic-control-bench/src/robust_viability.rs"
  "tools/allostatic-control-bench/src/main.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-control-report.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-model-mismatch-bench.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-multistep-viability-bench.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-ood-fallback-bench.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-task-scope-bench.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-sequential-scope-bench.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-risk-eprocess-bench.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-alpha-spending-bench.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-persistent-eprocess-bench.rs"
  "tools/allostatic-control-bench/qualify.sh"
  "docs/research/ALLOSTATIC_CONTROL_BENCHMARK_V1.md"
  "docs/research/ROBUST_VIABILITY_ENVELOPE_V1.md"
  "docs/research/MULTISTEP_VIABILITY_TUBES_V1.md"
  "docs/research/OOD_REACHABILITY_FALLBACK_V1.md"
  "docs/research/TASK_RELEVANT_SCOPE_MONITOR_V1.md"
  "docs/research/SEQUENTIAL_SCOPE_CUSUM_V1.md"
  "docs/research/BOUNDED_RISK_EPROCESS_V1.md"
  "docs/research/EPROCESS_LIFETIME_ALPHA_SPENDING_V1.md"
  "docs/research/EPROCESS_PERSISTENT_VS_RESET_V1.md"
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
ood_out="${out_base}-ood-fallback.csv"
ood_summary_out="${out_base}-ood-fallback-summary.csv"
scope_out="${out_base}-task-scope.csv"
scope_summary_out="${out_base}-task-scope-summary.csv"
sequential_out="${out_base}-sequential-scope.csv"
sequential_summary_out="${out_base}-sequential-scope-summary.csv"
eprocess_out="${out_base}-risk-eprocess.csv"
eprocess_summary_out="${out_base}-risk-eprocess-summary.csv"
alpha_spending_out="${out_base}-alpha-spending.csv"
alpha_spending_summary_out="${out_base}-alpha-spending-summary.csv"
persistent_eprocess_out="${out_base}-persistent-eprocess.csv"
persistent_eprocess_summary_out="${out_base}-persistent-eprocess-summary.csv"
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

# Calibrated OOD monitor and a separately widened reachability fallback.
# Actual synthetic transition parameters change under moderate/strong drift.
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-ood-fallback-bench -- \
  "$tmpdir/ood-a.csv" "$tmpdir/ood-summary-a.csv"
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-ood-fallback-bench -- \
  "$tmpdir/ood-b.csv" "$tmpdir/ood-summary-b.csv"
if ! cmp -s "$tmpdir/ood-a.csv" "$tmpdir/ood-b.csv" ||
   ! cmp -s "$tmpdir/ood-summary-a.csv" "$tmpdir/ood-summary-b.csv"; then
  printf 'FAIL: repeated OOD fallback raw and summary outputs are not byte-identical\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 28) failed = 1; next }
  {
    if (NF != 28) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin") failed = 1
    if ($2 != "nominal" && $2 != "moderate_shift" && $2 != "strong_shift") failed = 1
    if ($3 !~ /^[0-9]+$/ || $3 >= 300) failed = 1
    if ($4 !~ /^[0-9]*[.]?[0-9]+$/ || $5 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($6 != "true" && $6 != "false") failed = 1
    if (($6 == "true" && $7 != "reachability_fallback") ||
        ($6 == "false" && $7 != "predictive")) failed = 1
    if ($8 != "push") failed = 1
    for (i = 9; i <= 10; i++) if ($i !~ /^[0-3]$/) failed = 1
    for (i = 11; i <= 13; i++)
      if ($i != "push" && $i != "maintain" && $i != "emergency_recover" && $i != "none") failed = 1
    for (i = 14; i <= 17; i++) if ($i != "true" && $i != "false") failed = 1
    if ($18 !~ /^[0-3]$/ || $19 !~ /^[0-3]$/) failed = 1
    for (i = 20; i <= 28; i++) if ($i != "true" && $i != "false") failed = 1
    if (($4 > $5) != ($6 == "true")) failed = 1
    if (($6 == "false" && $13 != $11) || ($6 == "true" && $13 != $12)) failed = 1
    if (($13 == "none") != ($15 == "true")) failed = 1
    if ($15 == "true" && $14 == "true") failed = 1
    if (($13 != "none" && $20 == "false") != ($22 == "true")) failed = 1
    if ($21 == "true" && $20 == "false") failed = 1
    if ($6 == "true" && $13 != "none" && $21 != "true") failed = 1
    if ($11 == "none" && ($25 != "false" || $26 != "false")) failed = 1
    if ($2 == "nominal" && $11 != "none" && $25 != "true") failed = 1
    if ($12 != "none" && ($27 != "true" || $28 != "true")) failed = 1
    if ($2 == "nominal" &&
        ($22 == "true" || $23 != "true" || $24 != "true" ||
         ($11 != "none" && $25 != "true"))) failed = 1
    if ($2 != "nominal" && ($23 != "false" || $24 != "true")) failed = 1
    if ($2 == "strong_shift" && ($6 != "true" || $22 == "true")) failed = 1
    key = $1 SUBSEP $2 SUBSEP $3
    if (seen[key]++) failed = 1
    rows++
  }
  END { if (NR != 4501 || rows != 4500 || failed) exit 1 }
' "$tmpdir/ood-a.csv"; then
  printf 'FAIL: OOD switch, active-filter certification, actual safety or raw coverage is invalid\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 31) failed = 1; next }
  {
    if (NF != 31) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin") failed = 1
    if ($2 != "nominal" && $2 != "moderate_shift" && $2 != "strong_shift") failed = 1
    if ($3 != 300 || $4 < 0 || $4 > 300 || $5 < 0 || $5 > 300 || $6 < 0 || $6 > 300) failed = 1
    if ($5 > $4 || $6 > $3) failed = 1
    if ($9 != $4 || $10 > $9) failed = 1
    if ($11 < 0 || $11 > 300 || $12 < 0 || $12 > 300) failed = 1
    if ($13 < 0 || $13 > 3 || $14 < 0 || $14 > 3 || $15 < 0 || $15 > 3 || $16 < 0 || $16 > 3) failed = 1
    if ($17 < 0 || $17 > 300 || $18 < 0 || $18 > $17 ||
        $19 < 0 || $19 > $18 || $20 < 0 || $20 > $17) failed = 1
    if ($22 < 0 || $22 > 300 || $23 != 300) failed = 1
    if ($24 < 0 || $24 > 300 || $25 < 0 || $25 > $24 ||
        $27 < 0 || $27 > $24 || $28 < 0 || $28 > 300 ||
        $29 < 0 || $29 > $28 || $31 < 0 || $31 > $28) failed = 1
    if ($29 != 0 || $31 != $28) failed = 1
    if ($24 == 0) {
      if ($26 != "NA") failed = 1
    } else if (($25 / $24 - $26 > 0.00001) || ($26 - $25 / $24 > 0.00001)) failed = 1
    if ($28 == 0) {
      if ($30 != "NA") failed = 1
    } else if (($29 / $28 - $30 > 0.00001) || ($30 - $29 / $28 > 0.00001)) failed = 1
    if (($18 + $20) != $17) failed = 1
    if ($17 == 0) {
      if ($21 != "NA" || $20 != 0) failed = 1
    } else if (($20 / $17 - $21 > 0.00001) || ($21 - $20 / $17 > 0.00001)) failed = 1
    if ($2 == "nominal" &&
        ($5 != $4 || $6 != 0 || $7 > 0.10 || $20 != 0 || $22 != 300)) failed = 1
    if ($2 == "moderate_shift" &&
        ($5 != 0 || $6 > 150 || $22 != 0)) failed = 1
    if ($2 == "strong_shift" &&
        ($5 != 0 || $6 != 0 || $20 != 0 || $22 != 0)) failed = 1
    if ($2 == "nominal" && $25 != 0) failed = 1
    if ($2 == "strong_shift") strong_predictive_false_certificates += $25
    key = $1 SUBSEP $2
    if (seen[key]++) failed = 1
    rows++
  }
  END {
    if (NR != 16 || rows != 15 || strong_predictive_false_certificates < 1 || failed) exit 1
  }
' "$tmpdir/ood-summary-a.csv"; then
  printf 'FAIL: OOD fallback per-state alarm, miss, true-safety or availability summary is invalid\n' >&2
  exit 1
fi

# Compare generic feature-space OOD scores with task-relevant one-step
# model-scope residuals. Calibration is episode-level and test cases include a
# stealth shift with nominal-distribution feature scores but shifted dynamics.
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-task-scope-bench -- \
  "$tmpdir/scope-a.csv" "$tmpdir/scope-summary-a.csv"
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-task-scope-bench -- \
  "$tmpdir/scope-b.csv" "$tmpdir/scope-summary-b.csv"
if ! cmp -s "$tmpdir/scope-a.csv" "$tmpdir/scope-b.csv" ||
   ! cmp -s "$tmpdir/scope-summary-a.csv" "$tmpdir/scope-summary-b.csv"; then
  printf 'FAIL: repeated task-scope raw and summary outputs are not byte-identical\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 19) failed = 1; next }
  {
    if (NF != 19) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin") failed = 1
    if ($2 != "nominal" && $2 != "moderate_shift" &&
        $2 != "strong_shift" && $2 != "stealth_shift") failed = 1
    if ($3 !~ /^[0-9]+$/ || $3 >= 300) failed = 1
    for (i = 4; i <= 5; i++) if ($i !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    for (i = 6; i <= 11; i++) if ($i != "true" && $i != "false") failed = 1
    for (i = 12; i <= 15; i++) if ($i !~ /^[0-8]$/) failed = 1
    for (i = 16; i <= 19; i++) if ($i != "true" && $i != "false") failed = 1
    if (($4 > $5) != ($6 == "true")) failed = 1
    if ((($7 > $8) || ($19 == "true")) != ($9 == "true")) failed = 1
    if (($6 == "true" || $9 == "true") != ($10 == "true")) failed = 1
    if ($12 != (($6 == "true") ? 0 : 8)) failed = 1
    if (($13 < 8) != ($9 == "true")) failed = 1
    expected_combined = ($12 < $13 ? $12 : $13)
    if ($14 != expected_combined) failed = 1
    if (($15 < 8) != ($19 == "true")) failed = 1
    if ($11 != ($2 == "nominal" ? "false" : "true")) failed = 1
    if (($15 < 8 && $16 == "true") != ($15 < 8 && $12 <= $15)) failed = 1
    if (($15 < 8 && $17 == "true") != ($15 < 8 && $13 <= $15)) failed = 1
    if (($15 < 8 && $18 == "true") != ($15 < 8 && $14 <= $15)) failed = 1
    pair_key = $1 SUBSEP $3
    if ($2 == "nominal") nominal_feature_alarm[pair_key] = $6
    if ($2 == "stealth_shift" && nominal_feature_alarm[pair_key] != $6) failed = 1
    key = $1 SUBSEP $2 SUBSEP $3
    if (seen[key]++) failed = 1
    rows++
  }
  END { if (NR != 6001 || rows != 6000 || failed) exit 1 }
' "$tmpdir/scope-a.csv"; then
  printf 'FAIL: task-scope raw schema, paired stealth score, envelope or alarm-step contract is invalid\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 23) failed = 1; next }
  {
    if (NF != 23) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin") failed = 1
    if ($2 != "nominal" && $2 != "moderate_shift" &&
        $2 != "strong_shift" && $2 != "stealth_shift") failed = 1
    if ($3 != 300) failed = 1
    for (i = 4; i <= 19; i++) if ($i !~ /^[0-9]+$/) failed = 1
    if ($4 > 300 || $5 > 300 || $6 > 300 ||
        $7 > $4 || $8 > $5 || $9 > $6 ||
        $10 > 300 || $11 > 300 || $12 > 300 ||
        $13 > 300 || $14 > $13 || $15 > $13 || $16 > $13 ||
        $17 > $13 || $18 > $13 || $19 > $13) failed = 1
    if (($14 + $17) != $13 || ($15 + $18) != $13 ||
        ($16 + $19) != $13) failed = 1
    if ($18 != 0 || $19 != 0) failed = 1
    if ($20 !~ /^[0-9]*[.]?[0-9]+$/ || $21 !~ /^[0-9]*[.]?[0-9]+$/ ||
        $22 !~ /^[0-9]*[.]?[0-9]+$/ || $23 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($2 == "nominal" && ($7 > 30 || $8 > 30 || $9 > 30)) failed = 1
    if (($2 == "moderate_shift" || $2 == "strong_shift" || $2 == "stealth_shift") &&
        ($11 > 30 || $12 > 30)) failed = 1
    if ($2 == "moderate_shift" && $10 > 150) failed = 1
    if ($2 == "strong_shift" && $10 != 0) failed = 1
    if ($2 == "stealth_shift" && ($10 < 240 || $11 > 30)) failed = 1
    if ($2 == "nominal" && $11 > 30) failed = 1
    key = $1 SUBSEP $2
    if (seen[key]++) failed = 1
    rows++
  }
  END { if (NR != 21 || rows != 20 || failed) exit 1 }
' "$tmpdir/scope-summary-a.csv"; then
  printf 'FAIL: task-scope per-state false-alarm, missed-shift or safety-event summary is invalid\n' >&2
  exit 1
fi

# Calibrated sequential CUSUM over residuals. Pointwise and cumulative
# thresholds are independently calibrated from nominal episode maxima.
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-sequential-scope-bench -- \
  "$tmpdir/sequential-a.csv" "$tmpdir/sequential-summary-a.csv"
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-sequential-scope-bench -- \
  "$tmpdir/sequential-b.csv" "$tmpdir/sequential-summary-b.csv"
if ! cmp -s "$tmpdir/sequential-a.csv" "$tmpdir/sequential-b.csv" ||
   ! cmp -s "$tmpdir/sequential-summary-a.csv" "$tmpdir/sequential-summary-b.csv"; then
  printf 'FAIL: repeated sequential-scope raw and summary outputs are not byte-identical\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 14) failed = 1; next }
  NR == 2 {
    pointwise_threshold = $6
    cusum_threshold = $10
  }
  {
    if (NF != 14) failed = 1
    if (($6 - pointwise_threshold > 0.000001) ||
        (pointwise_threshold - $6 > 0.000001) ||
        ($10 - cusum_threshold > 0.000001) ||
        (cusum_threshold - $10 > 0.000001)) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin") failed = 1
    if ($2 != "nominal" && $2 != "gradual_shift" &&
        $2 != "abrupt_shift" && $2 != "stealth_shift") failed = 1
    if ($3 !~ /^[0-9]+$/ || $3 < 50000 || $3 >= 50300) failed = 1
    if ($4 !~ /^[0-7]$/) failed = 1
    if ($5 !~ /^[0-9]*[.]?[0-9]+$/ || $6 !~ /^[0-9]*[.]?[0-9]+$/ ||
        $9 !~ /^[0-9]*[.]?[0-9]+$/ || $10 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($7 != "true" && $7 != "false") failed = 1
    if ($11 != "true" && $11 != "false") failed = 1
    if ($8 !~ /^[0-8]$/ || $12 !~ /^[0-8]$/) failed = 1
    if ($13 != "true" && $13 != "false") failed = 1
    if ($14 != "true" && $14 != "false") failed = 1
    if (($5 > $6) != ($7 == "true")) failed = 1
    if (($2 == "nominal") != ($13 == "false")) failed = 1
    if ($9 < 0 || ($4 == 0 && $9 != 0)) failed = 1
    if ($10 < 0) failed = 1

    key = $1 SUBSEP $2 SUBSEP $3
    if ($4 == 0) {
      next_step[key] = 0
      prev_cusum[key] = 0
      first_point[key] = 8
      first_cusum[key] = 8
      episode_max[key] = 0
      episode_below[key] = "true"
      recorded_point_first[key] = $8
      recorded_cusum_first[key] = $12
    } else if ($4 != next_step[key]) failed = 1

    if ($5 > episode_max[key]) episode_max[key] = $5
    if ($5 > $6) episode_below[key] = "false"
    expected_cusum = prev_cusum[key] + $5 - 0.010
    if (expected_cusum < 0) expected_cusum = 0
    if (($9 - expected_cusum > 0.000001) || (expected_cusum - $9 > 0.000001)) failed = 1
    prev_cusum[key] = $9
    if ($7 == "true" && first_point[key] == 8) first_point[key] = $4
    if ($11 == "true" && first_cusum[key] == 8) first_cusum[key] = $4
    if ($8 != recorded_point_first[key] || $12 != recorded_cusum_first[key]) failed = 1
    if ($14 != episode_below[key]) {
      # The full-episode flag is only final on step 7; compare there.
      if ($4 == 7) failed = 1
    }

    rowkey = key SUBSEP $4
    if (seen[rowkey]++) failed = 1
    pairkey = $1 SUBSEP $3 SUBSEP $4
    if ($2 == "nominal") {
      nominal_score[pairkey] = $5
      nominal_point_alarm[pairkey] = $7
      nominal_cusum_stat[pairkey] = $9
      nominal_cusum_alarm[pairkey] = $11
      nominal_point_first[pairkey] = $8
      nominal_cusum_first[pairkey] = $12
      nominal_below[pairkey] = $14
    }
    if ($2 == "stealth_shift") {
      if (nominal_score[pairkey] != $5 ||
          nominal_point_alarm[pairkey] != $7 ||
          nominal_cusum_stat[pairkey] != $9 ||
          nominal_cusum_alarm[pairkey] != $11 ||
          nominal_point_first[pairkey] != $8 ||
          nominal_cusum_first[pairkey] != $12 ||
          nominal_below[pairkey] != $14) failed = 1
    }
    next_step[key] = $4 + 1
    rows++
  }
  END {
    if (NR != 48001 || rows != 48000 || failed) exit 1
    for (key in next_step) {
      if (next_step[key] != 8 || first_point[key] != recorded_point_first[key] ||
          first_cusum[key] != recorded_cusum_first[key]) exit 1
      if ((episode_max[key] <= pointwise_threshold) != (episode_below[key] == "true")) exit 1
    }
  }
' "$tmpdir/sequential-a.csv"; then
  printf 'FAIL: sequential-scope raw schema, CUSUM recurrence, step accounting or stealth pairing is invalid\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 24) failed = 1; next }
  {
    if (NF != 24) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin") failed = 1
    if ($2 != "nominal" && $2 != "gradual_shift" &&
        $2 != "abrupt_shift" && $2 != "stealth_shift") failed = 1
    if ($3 != 300) failed = 1
    for (i = 4; i <= 9; i++) if ($i !~ /^[0-9]+$/ || $i > 300) failed = 1
    if ($6 > $4 || $7 > $5) failed = 1
    if ($2 == "nominal" &&
        ($6 != $4 || $7 != $5 || $8 != 0 || $9 != 0)) failed = 1
    if ($2 != "nominal" &&
        ($6 != 0 || $7 != 0 || $4 + $8 != 300 || $5 + $9 != 300)) failed = 1
    if ($10 != "NA" && $10 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($11 != "NA" && $11 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($12 != "NA" && $12 !~ /^[0-9]+$/) failed = 1
    if ($13 != "NA" && $13 !~ /^[0-9]+$/) failed = 1
    if ($14 != "NA" && $14 !~ /^[0-9]+$/) failed = 1
    if ($15 != "NA" && $15 !~ /^[0-9]+$/) failed = 1
    for (i = 16; i <= 18; i++) if ($i !~ /^[0-9]+$/ || $i > 300) failed = 1
    if ($19 !~ /^[0-9]*[.]?[0-9]+$/ || $20 !~ /^[0-9]*[.]?[0-9]+$/ ||
        $21 !~ /^[0-9]*[.]?[0-9]+$/ || $22 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($23 < 0 || $23 > 300 || $24 < 0 || $24 > $23) failed = 1
    if ($2 == "nominal" && ($4 > 30 || $5 > 30 || $8 != 0 || $9 != 0)) failed = 1
    if ($2 == "gradual_shift" &&
        ($4 != 0 || $8 != 300 || $5 < 240 || $9 > 60 ||
         $23 != 300 || $24 < 240)) failed = 1
    if ($2 == "abrupt_shift" && ($4 < 270 || $5 < 270 || $8 > 30 || $9 > 30)) failed = 1
    if ($2 == "stealth_shift" && ($4 > 30 || $5 > 30 || $8 < 270 || $9 < 270)) failed = 1
    key = $1 SUBSEP $2
    if (seen[key]++) failed = 1
    if ($2 == "nominal") {
      nominal_point[key] = $4
      nominal_cusum[key] = $5
    }
    if ($2 == "stealth_shift" &&
        (nominal_point[$1 SUBSEP "nominal"] != $4 ||
         nominal_cusum[$1 SUBSEP "nominal"] != $5)) failed = 1
    rows++
  }
  END { if (NR != 21 || rows != 20 || failed) exit 1 }
' "$tmpdir/sequential-summary-a.csv"; then
  printf 'FAIL: sequential-scope summary schema, detection targets or matched-control gates are invalid\n' >&2
  exit 1
fi

# Apply a predeclared mixture of bounded-risk e-processes to the exact same
# sequential residual rows. The three bet fractions are fixed before evaluation.
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-risk-eprocess-bench -- \
  "$tmpdir/sequential-a.csv" "$tmpdir/eprocess-a.csv" "$tmpdir/eprocess-summary-a.csv"
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-risk-eprocess-bench -- \
  "$tmpdir/sequential-a.csv" "$tmpdir/eprocess-b.csv" "$tmpdir/eprocess-summary-b.csv"
if ! cmp -s "$tmpdir/eprocess-a.csv" "$tmpdir/eprocess-b.csv" ||
   ! cmp -s "$tmpdir/eprocess-summary-a.csv" "$tmpdir/eprocess-summary-b.csv"; then
  printf 'FAIL: repeated bounded-risk mixture e-process outputs are not byte-identical\n' >&2
  exit 1
fi
if ! cmp -s "$tmpdir/sequential-a.csv" <(cut -d, -f1-14 "$tmpdir/eprocess-a.csv"); then
  printf 'FAIL: e-process output is not paired to the exact CUSUM raw trace\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 30) failed = 1; next }
  NR == 2 {
    log_threshold = $19
    risk_bound = $20
    fraction_1 = $21
    fraction_2 = $22
    fraction_3 = $23
  }
  {
    if (NF != 30) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin") failed = 1
    if ($2 != "nominal" && $2 != "gradual_shift" &&
        $2 != "abrupt_shift" && $2 != "stealth_shift") failed = 1
    if ($3 !~ /^[0-9]+$/ || $3 < 50000 || $3 >= 50300) failed = 1
    if ($4 !~ /^[0-7]$/) failed = 1
    if ($5 !~ /^[0-9]*[.]?[0-9]+$/ || $5 < 0 || $5 > 1) failed = 1
    if ($6 !~ /^[0-9]*[.]?[0-9]+$/ || $6 < 0) failed = 1
    if ($7 != "true" && $7 != "false") failed = 1
    if ($8 !~ /^[0-8]$/) failed = 1
    if ($9 !~ /^[0-9]*[.]?[0-9]+$/ || $10 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($11 != "true" && $11 != "false") failed = 1
    if ($12 !~ /^[0-8]$/) failed = 1
    if ($13 != "true" && $13 != "false") failed = 1
    if ($14 != "true" && $14 != "false") failed = 1
    expected_loss = $5 / 0.05
    if (expected_loss > 1) expected_loss = 1
    if (($15 - expected_loss > 0.000001) || (expected_loss - $15 > 0.000001)) failed = 1
    if ($16 !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    if ($17 != "true" && $17 != "false") failed = 1
    if ($18 !~ /^[0-8]$/) failed = 1
    if (($19 - log(100) > 0.000001) || (log(100) - $19 > 0.000001)) failed = 1
    if (($19 - log_threshold > 0.000001) || (log_threshold - $19 > 0.000001)) failed = 1
    if (($20 - 0.15 > 0.000001) || (0.15 - $20 > 0.000001)) failed = 1
    if (($21 - 0.25 > 0.000001) || (0.25 - $21 > 0.000001)) failed = 1
    if (($22 - 0.50 > 0.000001) || (0.50 - $22 > 0.000001)) failed = 1
    if (($23 - 0.75 > 0.000001) || (0.75 - $23 > 0.000001)) failed = 1
    if (($20 - risk_bound > 0.000001) || (risk_bound - $20 > 0.000001)) failed = 1
    if (($21 - fraction_1 > 0.000001) || (fraction_1 - $21 > 0.000001) ||
        ($22 - fraction_2 > 0.000001) || (fraction_2 - $22 > 0.000001) ||
        ($23 - fraction_3 > 0.000001) || (fraction_3 - $23 > 0.000001)) failed = 1
    for (i = 24; i <= 27; i++) if ($i !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    if ($28 != "true" && $28 != "false") failed = 1
    if ($29 !~ /^[0-8]$/) failed = 1
    if (($30 - 0.5 > 0.000001) || (0.5 - $30 > 0.000001)) failed = 1

    key = $1 SUBSEP $2 SUBSEP $3
    if ($4 == 0) {
      next_step[key] = 0
      previous_1[key] = 0
      previous_2[key] = 0
      previous_3[key] = 0
      previous_fixed[key] = 0
      first_alarm[key] = 8
      first_fixed_alarm[key] = 8
      recorded_first_alarm[key] = $18
      recorded_fixed_first_alarm[key] = $29
      fixed_latched[key] = "false"
      recorded_below[key] = $14
      latched[key] = "false"
      episode_max_score[key] = 0
      episode_all_below[key] = "true"
      threshold_by_episode[key] = $6
    } else if ($4 != next_step[key]) failed = 1
    if (($6 - threshold_by_episode[key] > 0.000001) ||
        (threshold_by_episode[key] - $6 > 0.000001)) failed = 1

    factor_1 = 1 + (fraction_1 / risk_bound) * ($15 - risk_bound)
    factor_2 = 1 + (fraction_2 / risk_bound) * ($15 - risk_bound)
    factor_3 = 1 + (fraction_3 / risk_bound) * ($15 - risk_bound)
    if (factor_1 <= 0 || factor_2 <= 0 || factor_3 <= 0) failed = 1
    expected_1 = previous_1[key] + log(factor_1)
    expected_2 = previous_2[key] + log(factor_2)
    expected_3 = previous_3[key] + log(factor_3)
    if (($24 - expected_1 > 0.000001) || (expected_1 - $24 > 0.000001)) failed = 1
    if (($25 - expected_2 > 0.000001) || (expected_2 - $25 > 0.000001)) failed = 1
    if (($26 - expected_3 > 0.000001) || (expected_3 - $26 > 0.000001)) failed = 1
    previous_1[key] = $24
    previous_2[key] = $25
    previous_3[key] = $26

    max_component = $24
    if ($25 > max_component) max_component = $25
    if ($26 > max_component) max_component = $26
    expected_mix = max_component + log((
      exp($24 - max_component) +
      exp($25 - max_component) +
      exp($26 - max_component)) / 3)
    if (($16 - expected_mix > 0.000002) || (expected_mix - $16 > 0.000002)) failed = 1

    expected_alarm = (latched[key] == "true" || $16 >= log_threshold) ? "true" : "false"
    if ($17 != expected_alarm) failed = 1
    if ($17 == "true" && first_alarm[key] == 8) first_alarm[key] = $4
    latched[key] = expected_alarm
    if ($18 != recorded_first_alarm[key]) failed = 1

    factor_fixed = 1 + (0.5 / risk_bound) * ($15 - risk_bound)
    if (factor_fixed <= 0) failed = 1
    expected_fixed = previous_fixed[key] + log(factor_fixed)
    if (($27 - expected_fixed > 0.000001) || (expected_fixed - $27 > 0.000001)) failed = 1
    previous_fixed[key] = $27
    expected_fixed_alarm = (fixed_latched[key] == "true" || $27 >= log_threshold) ? "true" : "false"
    if ($28 != expected_fixed_alarm) failed = 1
    if ($28 == "true" && first_fixed_alarm[key] == 8) first_fixed_alarm[key] = $4
    fixed_latched[key] = expected_fixed_alarm
    if ($29 != recorded_fixed_first_alarm[key]) failed = 1
    if ($14 != recorded_below[key]) failed = 1
    if ($5 > episode_max_score[key]) episode_max_score[key] = $5
    if ($5 > $6) episode_all_below[key] = "false"

    pair_key = $1 SUBSEP $3 SUBSEP $4
    if ($2 == "nominal") {
      nominal_score[pair_key] = $5
      nominal_loss[pair_key] = $15
      nominal_mix[pair_key] = $16
      nominal_alarm[pair_key] = $17
      nominal_first_alarm[pair_key] = $18
      nominal_1[pair_key] = $24
      nominal_2[pair_key] = $25
      nominal_3[pair_key] = $26
      nominal_fixed_log[pair_key] = $27
      nominal_fixed_alarm[pair_key] = $28
      nominal_fixed_first[pair_key] = $29
    }
    if ($2 == "stealth_shift" &&
        (nominal_score[pair_key] != $5 ||
         nominal_loss[pair_key] != $15 ||
         nominal_mix[pair_key] != $16 ||
         nominal_alarm[pair_key] != $17 ||
         nominal_first_alarm[pair_key] != $18 ||
         nominal_1[pair_key] != $24 ||
         nominal_2[pair_key] != $25 ||
         nominal_3[pair_key] != $26 ||
         nominal_fixed_log[pair_key] != $27 ||
         nominal_fixed_alarm[pair_key] != $28 ||
         nominal_fixed_first[pair_key] != $29)) failed = 1

    if ($4 == 7 && ($14 == "true") != (episode_max_score[key] <= threshold_by_episode[key])) failed = 1
    if ($4 == 7 && (episode_all_below[key] == "true") != ($14 == "true")) failed = 1
    row_key = key SUBSEP $4
    if (seen[row_key]++) failed = 1
    next_step[key] = $4 + 1
    rows++
  }
  END {
    if (NR != 48001 || rows != 48000 || failed) exit 1
    for (key in next_step) {
      if (next_step[key] != 8 ||
          first_alarm[key] != recorded_first_alarm[key] ||
          first_fixed_alarm[key] != recorded_fixed_first_alarm[key]) exit 1
    }
  }
' "$tmpdir/eprocess-a.csv"; then
  printf 'FAIL: bounded-risk mixture e-process schema, component recurrence, log-mixture, threshold or stealth pairing is invalid\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 29) failed = 1; next }
  {
    if (NF != 29) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin") failed = 1
    if ($2 != "nominal" && $2 != "gradual_shift" &&
        $2 != "abrupt_shift" && $2 != "stealth_shift") failed = 1
    if ($3 != 300) failed = 1
    for (i = 4; i <= 6; i++) if ($i !~ /^[0-9]+$/ || $i > 300) failed = 1
    if ($2 == "nominal") {
      if ($5 != $4 || $6 != 0 || $4 > 30) failed = 1
      if ($21 != $20 || $22 != 0 || $20 > 30) failed = 1
    } else {
      if ($4 + $6 != 300 || $5 != 0) failed = 1
      if ($20 + $22 != 300 || $21 != 0) failed = 1
    }
    if ($7 != "NA" && $7 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($8 != "NA" && $8 !~ /^[0-9]+$/) failed = 1
    if ($9 != "NA" && $9 !~ /^[0-9]+$/) failed = 1
    if ($4 == 0 && ($7 != "NA" || $8 != "NA" || $9 != "NA")) failed = 1
    if ($4 > 0 && ($7 == "NA" || $8 == "NA" || $9 == "NA")) failed = 1
    if ($10 < 0 || $10 > 300 || $11 < 0 || $11 > $10) failed = 1
    if ($12 !~ /^-?[0-9]*[.]?[0-9]+$/ || $13 !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    if (($14 - log(100) > 0.000001) || (log(100) - $14 > 0.000001)) failed = 1
    if (($15 - 0.15 > 0.000001) || (0.15 - $15 > 0.000001)) failed = 1
    if (($16 - 0.25 > 0.000001) || (0.25 - $16 > 0.000001) ||
        ($17 - 0.50 > 0.000001) || (0.50 - $17 > 0.000001) ||
        ($18 - 0.75 > 0.000001) || (0.75 - $18 > 0.000001)) failed = 1
    if (($19 - 0.05 > 0.000001) || (0.05 - $19 > 0.000001)) failed = 1
    if ($23 != "NA" && $23 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($24 != "NA" && $24 !~ /^[0-9]+$/) failed = 1
    if ($25 != "NA" && $25 !~ /^[0-9]+$/) failed = 1
    if ($20 == 0 && ($23 != "NA" || $24 != "NA" || $25 != "NA")) failed = 1
    if ($20 > 0 && ($23 == "NA" || $24 == "NA" || $25 == "NA")) failed = 1
    if ($26 < 0 || $26 > 300 || $26 > $10) failed = 1
    if ($27 !~ /^-?[0-9]*[.]?[0-9]+$/ || $28 !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    if (($29 - 0.5 > 0.000001) || (0.5 - $29 > 0.000001)) failed = 1

    if ($2 == "nominal" && ($4 > 30 || $5 != $4 || $6 != 0)) failed = 1
    if ($2 == "gradual_shift" &&
        ($4 < 240 || $6 > 60 || $10 != 300 || $11 < 240 ||
         $20 < 120 || $20 > 240 || $22 > 180 || $26 != $20 ||
         $4 < $20 + 60)) failed = 1
    if ($2 == "abrupt_shift" &&
        ($4 < 270 || $6 > 30 || $20 < 270 || $22 > 30)) failed = 1
    if ($2 == "stealth_shift" &&
        ($4 > 30 || $6 < 270 || $20 > 30 || $22 < 270)) failed = 1
    key = $1 SUBSEP $2
    if (seen[key]++) failed = 1
    if ($2 == "nominal") {
      nominal_alarms[$1] = $4
      nominal_misses[$1] = $6
      nominal_fixed_alarms[$1] = $20
      nominal_fixed_misses[$1] = $22
    }
    if ($2 == "stealth_shift" &&
        ($4 != nominal_alarms[$1] || $6 != nominal_misses[$1] ||
         $20 != nominal_fixed_alarms[$1] ||
         $22 != nominal_fixed_misses[$1])) failed = 1
    rows++
  }
  END { if (NR != 21 || rows != 20 || failed) exit 1 }
' "$tmpdir/eprocess-summary-a.csv"; then
  printf 'FAIL: mixture e-process summary schema, detection behavior or matched stealth-control gates are invalid\n' >&2
  exit 1
fi

# Compare a countably summable alpha-spending schedule with the same e-process
# reset per episode at a fixed threshold. Inputs are the exact same CUSUM traces.
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-alpha-spending-bench -- \
  "$tmpdir/sequential-a.csv" "$tmpdir/alpha-spending-a.csv" "$tmpdir/alpha-spending-summary-a.csv"
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-alpha-spending-bench -- \
  "$tmpdir/sequential-a.csv" "$tmpdir/alpha-spending-b.csv" "$tmpdir/alpha-spending-summary-b.csv"
if ! cmp -s "$tmpdir/alpha-spending-a.csv" "$tmpdir/alpha-spending-b.csv" ||
   ! cmp -s "$tmpdir/alpha-spending-summary-a.csv" "$tmpdir/alpha-spending-summary-b.csv"; then
  printf 'FAIL: repeated alpha-spending outputs are not byte-identical\n' >&2
  exit 1
fi
if ! cmp -s "$tmpdir/sequential-a.csv" <(cut -d, -f1-14 "$tmpdir/alpha-spending-a.csv"); then
  printf 'FAIL: alpha-spending output is not paired to the exact CUSUM source trace\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 31) failed = 1; next }
  {
    if (NF != 31) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin") failed = 1
    if ($2 != "nominal" && $2 != "gradual_shift" &&
        $2 != "abrupt_shift" && $2 != "stealth_shift") failed = 1
    if ($3 !~ /^[0-9]+$/ || $3 < 50000 || $3 >= 50300) failed = 1
    if ($4 !~ /^[0-7]$/) failed = 1
    if ($5 !~ /^[0-9]*[.]?[0-9]+$/ || $5 < 0 || $5 > 1) failed = 1
    if ($6 !~ /^[0-9]*[.]?[0-9]+$/ || $6 < 0) failed = 1
    for (i = 7; i <= 7; i++) if ($i != "true" && $i != "false") failed = 1
    if ($8 !~ /^[0-8]$/ || $9 !~ /^[0-9]*[.]?[0-9]+$/ ||
        $10 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($11 != "true" && $11 != "false") failed = 1
    if ($12 !~ /^[0-8]$/) failed = 1
    if ($13 != "true" && $13 != "false") failed = 1
    if ($14 != "true" && $14 != "false") failed = 1
    expected_loss = $5 / 0.05
    if (expected_loss > 1) expected_loss = 1
    if (($15 - expected_loss > 0.000001) || (expected_loss - $15 > 0.000001)) failed = 1
    expected_budget_index = $3 - 49999
    if ($16 != expected_budget_index || $16 < 1 || $16 > 300) failed = 1
    expected_spend = 0.05 / (5 * $16 * ($16 + 1))
    if (($17 - expected_spend > 0.0000000001) ||
        (expected_spend - $17 > 0.0000000001)) failed = 1
    expected_alpha_threshold = log(5 * $16 * ($16 + 1) / 0.05)
    if (($19 - expected_alpha_threshold > 0.000001) ||
        (expected_alpha_threshold - $19 > 0.000001)) failed = 1
    if (($23 - log(100) > 0.000001) || (log(100) - $23 > 0.000001)) failed = 1
    if ($18 !~ /^-?[0-9]*[.]?[0-9]+$/ ||
        $22 !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    if ($20 != "true" && $20 != "false") failed = 1
    if ($21 !~ /^[0-8]$/) failed = 1
    if ($24 != "true" && $24 != "false") failed = 1
    if ($25 !~ /^[0-8]$/) failed = 1
    for (i = 26; i <= 31; i++) if ($i !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    if ($19 < log(5 / 0.05)) failed = 1

    key = $1 SUBSEP $2 SUBSEP $3
    if ($4 == 0) {
      next_step[key] = 0
      previous_spend_1[key] = 0
      previous_spend_2[key] = 0
      previous_spend_3[key] = 0
      previous_reset_1[key] = 0
      previous_reset_2[key] = 0
      previous_reset_3[key] = 0
      first_spend[key] = 8
      first_reset[key] = 8
      recorded_first_spend[key] = $21
      recorded_first_reset[key] = $25
      latched_spend[key] = "false"
      latched_reset[key] = "false"
      recorded_below[key] = $14
      episode_all_below[key] = "true"
      previous_budget[key] = $16
      episode_max_score[key] = 0
    } else if ($4 != next_step[key]) failed = 1
    if ($16 != previous_budget[key]) failed = 1
    if ($21 != recorded_first_spend[key] || $25 != recorded_first_reset[key]) failed = 1

    factor_1 = 1 + (0.25 / 0.15) * ($15 - 0.15)
    factor_2 = 1 + (0.50 / 0.15) * ($15 - 0.15)
    factor_3 = 1 + (0.75 / 0.15) * ($15 - 0.15)
    spend_1 = previous_spend_1[key] + log(factor_1)
    spend_2 = previous_spend_2[key] + log(factor_2)
    spend_3 = previous_spend_3[key] + log(factor_3)
    reset_1 = previous_reset_1[key] + log(factor_1)
    reset_2 = previous_reset_2[key] + log(factor_2)
    reset_3 = previous_reset_3[key] + log(factor_3)
    if (($26 - spend_1 > 0.000001) || (spend_1 - $26 > 0.000001) ||
        ($27 - spend_2 > 0.000001) || (spend_2 - $27 > 0.000001) ||
        ($28 - spend_3 > 0.000001) || (spend_3 - $28 > 0.000001)) failed = 1
    if (($29 - reset_1 > 0.000001) || (reset_1 - $29 > 0.000001) ||
        ($30 - reset_2 > 0.000001) || (reset_2 - $30 > 0.000001) ||
        ($31 - reset_3 > 0.000001) || (reset_3 - $31 > 0.000001)) failed = 1
    previous_spend_1[key] = $26
    previous_spend_2[key] = $27
    previous_spend_3[key] = $28
    previous_reset_1[key] = $29
    previous_reset_2[key] = $30
    previous_reset_3[key] = $31

    max_spend_component = $26
    if ($27 > max_spend_component) max_spend_component = $27
    if ($28 > max_spend_component) max_spend_component = $28
    expected_spend_mix = max_spend_component + log((
      exp($26 - max_spend_component) +
      exp($27 - max_spend_component) +
      exp($28 - max_spend_component)) / 3)
    if (($18 - expected_spend_mix > 0.000002) ||
        (expected_spend_mix - $18 > 0.000002)) failed = 1

    max_reset_component = $29
    if ($30 > max_reset_component) max_reset_component = $30
    if ($31 > max_reset_component) max_reset_component = $31
    expected_reset_mix = max_reset_component + log((
      exp($29 - max_reset_component) +
      exp($30 - max_reset_component) +
      exp($31 - max_reset_component)) / 3)
    if (($22 - expected_reset_mix > 0.000002) ||
        (expected_reset_mix - $22 > 0.000002)) failed = 1

    expected_spend_alarm = (latched_spend[key] == "true" || $18 >= $19) ? "true" : "false"
    expected_reset_alarm = (latched_reset[key] == "true" || $22 >= $23) ? "true" : "false"
    if ($20 != expected_spend_alarm || $24 != expected_reset_alarm) failed = 1
    if ($20 == "true" && first_spend[key] == 8) first_spend[key] = $4
    if ($24 == "true" && first_reset[key] == 8) first_reset[key] = $4
    latched_spend[key] = expected_spend_alarm
    latched_reset[key] = expected_reset_alarm
    if ($5 > episode_max_score[key]) episode_max_score[key] = $5
    if ($5 > $6) episode_all_below[key] = "false"
    if ($14 != recorded_below[key]) failed = 1
    if ($4 == 7 && (($14 == "true") != (episode_all_below[key] == "true"))) failed = 1

    pair_key = $1 SUBSEP $3 SUBSEP $4
    if ($2 == "nominal") {
      nominal_score[pair_key] = $5
      nominal_loss[pair_key] = $15
      nominal_budget[pair_key] = $16
      nominal_spend_log[pair_key] = $18
      nominal_spend_alarm[pair_key] = $20
      nominal_spend_first[pair_key] = $21
      nominal_reset_log[pair_key] = $22
      nominal_reset_alarm[pair_key] = $24
      nominal_reset_first[pair_key] = $25
    }
    if ($2 == "stealth_shift" &&
        (nominal_score[pair_key] != $5 ||
         nominal_loss[pair_key] != $15 ||
         nominal_budget[pair_key] != $16 ||
         nominal_spend_log[pair_key] != $18 ||
         nominal_spend_alarm[pair_key] != $20 ||
         nominal_spend_first[pair_key] != $21 ||
         nominal_reset_log[pair_key] != $22 ||
         nominal_reset_alarm[pair_key] != $24 ||
         nominal_reset_first[pair_key] != $25)) failed = 1

    row_key = key SUBSEP $4
    if (seen[row_key]++) failed = 1
    next_step[key] = $4 + 1
    rows++
  }
  END {
    if (NR != 48001 || rows != 48000 || failed) exit 1
    for (key in next_step) {
      if (next_step[key] != 8 ||
          first_spend[key] != recorded_first_spend[key] ||
          first_reset[key] != recorded_first_reset[key]) exit 1
    }
  }
' "$tmpdir/alpha-spending-a.csv"; then
  printf 'FAIL: alpha-spending raw schema, component recurrence, threshold or paired-controls contract is invalid\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 19) failed = 1; next }
  {
    if (NF != 19) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin") failed = 1
    if ($2 != "nominal" && $2 != "gradual_shift" &&
        $2 != "abrupt_shift" && $2 != "stealth_shift") failed = 1
    if ($3 != 300) failed = 1
    for (i = 4; i <= 9; i++) if ($i !~ /^[0-9]+$/ || $i > 300) failed = 1
    if ($2 == "nominal") {
      if ($5 != $4 || $6 != 0 || $8 != $7 || $9 != 0 || $4 > 30 || $7 > 30) failed = 1
    } else {
      if ($5 != 0 || $4 + $6 != 300 || $8 != 0 || $7 + $9 != 300) failed = 1
    }
    if ($4 > $7 || $6 < $9) failed = 1
    if ($10 != "NA" && $10 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($11 != "NA" && $11 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($12 < 0 || $12 > 300 || $13 < 0 || $13 > $12 ||
        $14 < 0 || $14 > $12) failed = 1
    if ($15 !~ /^[0-9]*[.]?[0-9]+$/ ||
        $16 != 300 ||
        $17 !~ /^-?[0-9]*[.]?[0-9]+$/ ||
        $18 !~ /^-?[0-9]*[.]?[0-9]+$/ ||
        $19 !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    expected_spend_sum = 0.05 / 5 * (1 - 1 / 301)
    if (($15 - expected_spend_sum > 0.0000001) ||
        (expected_spend_sum - $15 > 0.0000001)) failed = 1
    key = $1 SUBSEP $2
    if (seen[key]++) failed = 1
    if ($2 == "nominal") {
      nominal_spending[$1] = $4
      nominal_reset[$1] = $7
    }
    if ($2 == "stealth_shift" &&
        ($4 != nominal_spending[$1] || $7 != nominal_reset[$1])) failed = 1
    rows++
  }
  END { if (NR != 21 || rows != 20 || failed) exit 1 }
' "$tmpdir/alpha-spending-summary-a.csv"; then
  printf 'FAIL: alpha-spending summary schema, budget arithmetic or paired shift controls are invalid\n' >&2
  exit 1
fi

# Compare a continuous per-stream e-process with summably alpha-spent episode resets.
# Each of the 20 state/condition streams is a separate test stream (K=20); the
# persistent monitor does not reset at episode boundaries.
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-persistent-eprocess-bench -- \
  "$tmpdir/sequential-a.csv" "$tmpdir/persistent-eprocess-a.csv" "$tmpdir/persistent-eprocess-summary-a.csv"
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-persistent-eprocess-bench -- \
  "$tmpdir/sequential-a.csv" "$tmpdir/persistent-eprocess-b.csv" "$tmpdir/persistent-eprocess-summary-b.csv"
if ! cmp -s "$tmpdir/persistent-eprocess-a.csv" "$tmpdir/persistent-eprocess-b.csv" ||
   ! cmp -s "$tmpdir/persistent-eprocess-summary-a.csv" "$tmpdir/persistent-eprocess-summary-b.csv"; then
  printf 'FAIL: repeated persistent-e-process outputs are not byte-identical\n' >&2
  exit 1
fi
if ! cmp -s "$tmpdir/sequential-a.csv" <(cut -d, -f1-14 "$tmpdir/persistent-eprocess-a.csv"); then
  printf 'FAIL: persistent-e-process output is not paired to the exact CUSUM trace\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 31) failed = 1; next }
  {
    if (NF != 31) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin") failed = 1
    if ($2 != "nominal" && $2 != "gradual_shift" &&
        $2 != "abrupt_shift" && $2 != "stealth_shift") failed = 1
    if ($3 < 50000 || $3 >= 50300 || $3 !~ /^[0-9]+$/ ||
        $4 !~ /^[0-7]$/) failed = 1
    if ($5 !~ /^[0-9]*[.]?[0-9]+$/ || $5 < 0 || $5 > 1 ||
        $6 !~ /^[0-9]*[.]?[0-9]+$/ || $6 < 0) failed = 1
    if ($7 != "true" && $7 != "false") failed = 1
    if ($8 !~ /^[0-8]$/ || $9 !~ /^[0-9]*[.]?[0-9]+$/ ||
        $10 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($11 != "true" && $11 != "false") failed = 1
    if ($12 !~ /^[0-8]$/ || ($13 != "true" && $13 != "false") ||
        ($14 != "true" && $14 != "false")) failed = 1
    expected_loss = $5 / 0.05
    if (expected_loss > 1) expected_loss = 1
    if (($15 - expected_loss > 0.000001) || (expected_loss - $15 > 0.000001)) failed = 1
    if ($16 != ($3 - 49999) || $16 < 1 || $16 > 300) failed = 1
    if (($18 - log(20 / 0.05) > 0.000001) ||
        (log(20 / 0.05) - $18 > 0.000001)) failed = 1
    if ($19 != "true" && $19 != "false") failed = 1
    if ($20 != "NA" && $20 !~ /^[0-9]+$/) failed = 1
    if ($24 !~ /^-?[0-9]*[.]?[0-9]+$/ ||
        $25 !~ /^[0-9]*[.]?[0-9]+$/ ||
        ($26 != "true" && $26 != "false") ||
        $27 !~ /^[0-8]$/ ||
        $28 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    expected_reset_budget = 0.05 / (20 * $16 * ($16 + 1))
    expected_reset_threshold = log(20 * $16 * ($16 + 1) / 0.05)
    if (($25 - expected_reset_threshold > 0.000001) ||
        (expected_reset_threshold - $25 > 0.000001)) failed = 1
    if (($28 - expected_reset_budget > 0.0000000001) ||
        (expected_reset_budget - $28 > 0.0000000001)) failed = 1
    for (i = 21; i <= 23; i++) if ($i !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    for (i = 29; i <= 31; i++) if ($i !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1

    cell = $1 SUBSEP $2
    episode = $3 SUBSEP cell
    stream_step = ($16 - 1) * 8 + $4
    if ($4 == 0 && $16 == 1) {
      next_stream[cell] = 0
      pprev_1[cell] = 0
      pprev_2[cell] = 0
      pprev_3[cell] = 0
      first_persistent[cell] = 2400
      recorded_persistent_first[cell] = $20
      persistent_latched[cell] = "false"
    }
    if (stream_step != next_stream[cell]) failed = 1
    next_stream[cell] = stream_step + 1

    pf1 = 1 + (0.25 / 0.15) * ($15 - 0.15)
    pf2 = 1 + (0.50 / 0.15) * ($15 - 0.15)
    pf3 = 1 + (0.75 / 0.15) * ($15 - 0.15)
    if (pf1 <= 0 || pf2 <= 0 || pf3 <= 0) failed = 1
    pexp1 = pprev_1[cell] + log(pf1)
    pexp2 = pprev_2[cell] + log(pf2)
    pexp3 = pprev_3[cell] + log(pf3)
    if (($21 - pexp1 > 0.000001) || (pexp1 - $21 > 0.000001) ||
        ($22 - pexp2 > 0.000001) || (pexp2 - $22 > 0.000001) ||
        ($23 - pexp3 > 0.000001) || (pexp3 - $23 > 0.000001)) failed = 1
    pprev_1[cell] = $21
    pprev_2[cell] = $22
    pprev_3[cell] = $23
    pmax = $21
    if ($22 > pmax) pmax = $22
    if ($23 > pmax) pmax = $23
    pmix = pmax + log((exp($21 - pmax) + exp($22 - pmax) + exp($23 - pmax)) / 3)
    if (($17 - pmix > 0.000002) || (pmix - $17 > 0.000002)) failed = 1
    palarm = (persistent_latched[cell] == "true" || $17 >= $18) ? "true" : "false"
    if ($19 != palarm) failed = 1
    if ($19 == "true" && first_persistent[cell] == 2400) first_persistent[cell] = stream_step
    persistent_latched[cell] = palarm
    if ($20 != recorded_persistent_first[cell]) failed = 1

    if ($4 == 0) {
      rprev_1[episode] = 0
      rprev_2[episode] = 0
      rprev_3[episode] = 0
      first_reset[episode] = 8
      recorded_reset_first[episode] = $27
      reset_latched[episode] = "false"
      below_episode[episode] = "true"
      reset_budget[episode] = $28
    }
    if ($28 != reset_budget[episode]) failed = 1
    rfx1 = 1 + (0.25 / 0.15) * ($15 - 0.15)
    rfx2 = 1 + (0.50 / 0.15) * ($15 - 0.15)
    rfx3 = 1 + (0.75 / 0.15) * ($15 - 0.15)
    rexp1 = rprev_1[episode] + log(rfx1)
    rexp2 = rprev_2[episode] + log(rfx2)
    rexp3 = rprev_3[episode] + log(rfx3)
    if (($29 - rexp1 > 0.000001) || (rexp1 - $29 > 0.000001) ||
        ($30 - rexp2 > 0.000001) || (rexp2 - $30 > 0.000001) ||
        ($31 - rexp3 > 0.000001) || (rexp3 - $31 > 0.000001)) failed = 1
    rprev_1[episode] = $29
    rprev_2[episode] = $30
    rprev_3[episode] = $31
    rmax = $29
    if ($30 > rmax) rmax = $30
    if ($31 > rmax) rmax = $31
    rmix = rmax + log((exp($29 - rmax) + exp($30 - rmax) + exp($31 - rmax)) / 3)
    if (($24 - rmix > 0.000002) || (rmix - $24 > 0.000002)) failed = 1
    rexp_alarm = (reset_latched[episode] == "true" || $24 >= $25) ? "true" : "false"
    if ($26 != rexp_alarm || $27 != recorded_reset_first[episode]) failed = 1
    if ($26 == "true" && first_reset[episode] == 8) first_reset[episode] = $4
    reset_latched[episode] = rexp_alarm
    if ($5 > $6) below_episode[episode] = "false"
    pair_key = $1 SUBSEP $3 SUBSEP $4
    if ($2 == "nominal") {
      nominal_score[pair_key] = $5
      nominal_persistent[pair_key] = $17
      nominal_persistent_alarm[pair_key] = $19
      nominal_persistent_first[pair_key] = $20
      nominal_reset[pair_key] = $24
      nominal_reset_alarm[pair_key] = $26
      nominal_reset_first[pair_key] = $27
    }
    if ($2 == "stealth_shift" &&
        (nominal_score[pair_key] != $5 ||
         nominal_persistent[pair_key] != $17 ||
         nominal_persistent_alarm[pair_key] != $19 ||
         nominal_persistent_first[pair_key] != $20 ||
         nominal_reset[pair_key] != $24 ||
         nominal_reset_alarm[pair_key] != $26 ||
         nominal_reset_first[pair_key] != $27)) failed = 1

    row_key = episode SUBSEP $4
    if (seen[row_key]++) failed = 1
    rows++
  }
  END {
    if (NR != 48001 || rows != 48000 || failed) exit 1
    for (cell in next_stream) {
      if (next_stream[cell] != 2400) exit 1
      if (first_persistent[cell] == 2400) {
        if (recorded_persistent_first[cell] != "NA") exit 1
      } else if (first_persistent[cell] != recorded_persistent_first[cell]) exit 1
    }
    for (episode in first_reset) {
      if (first_reset[episode] != recorded_reset_first[episode]) exit 1
    }
  }
' "$tmpdir/persistent-eprocess-a.csv"; then
  printf 'FAIL: persistent/reset raw recurrence, threshold, latch or paired-control gates are invalid\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 20) failed = 1; next }
  {
    if (NF != 20 || $3 != 300) failed = 1
    if ($1 != "critical_low_reserve_high_load" && $1 != "reserve_limited" &&
        $1 != "load_limited" && $1 != "balanced" && $1 != "high_margin") failed = 1
    if ($2 != "nominal" && $2 != "gradual_shift" &&
        $2 != "abrupt_shift" && $2 != "stealth_shift") failed = 1
    if ($4 != "true" && $4 != "false") failed = 1
    if ($5 != "true" && $5 != "false") failed = 1
    if ($6 != "true" && $6 != "false") failed = 1
    if (($2 == "nominal" && ($5 != $4 || $6 != "false")) ||
        ($2 != "nominal" && ($5 != "false" || ($6 == "true") == ($4 == "true")))) failed = 1
    if ($7 != "NA" && ($7 !~ /^[0-9]+$/ || $7 >= 300)) failed = 1
    if ($8 != "NA" && ($8 !~ /^[0-7]$/)) failed = 1
    if (($4 == "true") != ($7 != "NA") || ($4 == "true") != ($8 != "NA")) failed = 1
    for (i = 9; i <= 11; i++) if ($i !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    if ($12 < 0 || $12 > 300 || $13 < 0 || $13 > 300 || $14 < 0 || $14 > 300) failed = 1
    if ($13 > $12 || $14 > $12 || $13 + $14 > 300) failed = 1
    if ($15 != "NA" && $15 !~ /^[0-9]*[.]?[0-9]+$/) failed = 1
    if ($12 == 0 && $15 != "NA") failed = 1
    if ($12 > 0 && $15 == "NA") failed = 1
    if ($16 < 0 || $16 > 300 || $17 < 0 || $17 > $16) failed = 1
    if (($18 - (0.05 / 20 * (1 - 1 / 301)) > 0.0000001) ||
        ((0.05 / 20 * (1 - 1 / 301)) - $18 > 0.0000001)) failed = 1
    if ($19 != 300 || $20 !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    if ($2 == "nominal" && ($12 != $13 || $14 != 0)) failed = 1
    if ($2 != "nominal" && ($13 != 0 || $12 + $14 != 300)) failed = 1
    key = $1 SUBSEP $2
    if (seen[key]++) failed = 1
    if ($2 == "nominal") {
      nominal_persistent[$1] = $4
      nominal_reset[$1] = $12
    }
    if ($2 == "stealth_shift" &&
        ($4 != nominal_persistent[$1] || $12 != nominal_reset[$1])) failed = 1
    rows++
  }
  END { if (NR != 21 || rows != 20 || failed) exit 1 }
' "$tmpdir/persistent-eprocess-summary-a.csv"; then
  printf 'FAIL: persistent/reset summary familywise-budget or paired-control contract is invalid\n' >&2
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
cp "$tmpdir/ood-a.csv" "$ood_out"
cp "$tmpdir/ood-summary-a.csv" "$ood_summary_out"
cp "$tmpdir/scope-a.csv" "$scope_out"
cp "$tmpdir/scope-summary-a.csv" "$scope_summary_out"
cp "$tmpdir/sequential-a.csv" "$sequential_out"
cp "$tmpdir/sequential-summary-a.csv" "$sequential_summary_out"
cp "$tmpdir/eprocess-a.csv" "$eprocess_out"
cp "$tmpdir/eprocess-summary-a.csv" "$eprocess_summary_out"
cp "$tmpdir/alpha-spending-a.csv" "$alpha_spending_out"
cp "$tmpdir/alpha-spending-summary-a.csv" "$alpha_spending_summary_out"
cp "$tmpdir/persistent-eprocess-a.csv" "$persistent_eprocess_out"
cp "$tmpdir/persistent-eprocess-summary-a.csv" "$persistent_eprocess_summary_out"
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
  printf 'ood_frontier_sha256=%s\n' "$(sha256sum "$ood_out" | awk '{print $1}')"
  printf 'ood_frontier_rows_excluding_header=%s\n' "$(( $(wc -l < "$ood_out") - 1 ))"
  printf 'ood_summary_sha256=%s\n' "$(sha256sum "$ood_summary_out" | awk '{print $1}')"
  printf 'ood_summary_rows_excluding_header=%s\n' "$(( $(wc -l < "$ood_summary_out") - 1 ))"
  printf 'task_scope_sha256=%s\n' "$(sha256sum "$scope_out" | awk '{print $1}')"
  printf 'task_scope_rows_excluding_header=%s\n' "$(( $(wc -l < "$scope_out") - 1 ))"
  printf 'task_scope_summary_sha256=%s\n' "$(sha256sum "$scope_summary_out" | awk '{print $1}')"
  printf 'task_scope_summary_rows_excluding_header=%s\n' "$(( $(wc -l < "$scope_summary_out") - 1 ))"
  printf 'sequential_scope_sha256=%s\n' "$(sha256sum "$sequential_out" | awk '{print $1}')"
  printf 'sequential_scope_rows_excluding_header=%s\n' "$(( $(wc -l < "$sequential_out") - 1 ))"
  printf 'sequential_scope_summary_sha256=%s\n' "$(sha256sum "$sequential_summary_out" | awk '{print $1}')"
  printf 'sequential_scope_summary_rows_excluding_header=%s\n' "$(( $(wc -l < "$sequential_summary_out") - 1 ))"
  printf 'risk_eprocess_sha256=%s\n' "$(sha256sum "$eprocess_out" | awk '{print $1}')"
  printf 'risk_eprocess_rows_excluding_header=%s\n' "$(( $(wc -l < "$eprocess_out") - 1 ))"
  printf 'risk_eprocess_summary_sha256=%s\n' "$(sha256sum "$eprocess_summary_out" | awk '{print $1}')"
  printf 'risk_eprocess_summary_rows_excluding_header=%s\n' "$(( $(wc -l < "$eprocess_summary_out") - 1 ))"
  printf 'alpha_spending_sha256=%s\n' "$(sha256sum "$alpha_spending_out" | awk '{print $1}')"
  printf 'alpha_spending_rows_excluding_header=%s\n' "$(( $(wc -l < "$alpha_spending_out") - 1 ))"
  printf 'alpha_spending_summary_sha256=%s\n' "$(sha256sum "$alpha_spending_summary_out" | awk '{print $1}')"
  printf 'alpha_spending_summary_rows_excluding_header=%s\n' "$(( $(wc -l < "$alpha_spending_summary_out") - 1 ))"
  printf 'persistent_eprocess_sha256=%s\n' "$(sha256sum "$persistent_eprocess_out" | awk '{print $1}')"
  printf 'persistent_eprocess_rows_excluding_header=%s\n' "$(( $(wc -l < "$persistent_eprocess_out") - 1 ))"
  printf 'persistent_eprocess_summary_sha256=%s\n' "$(sha256sum "$persistent_eprocess_summary_out" | awk '{print $1}')"
  printf 'persistent_eprocess_summary_rows_excluding_header=%s\n' "$(( $(wc -l < "$persistent_eprocess_summary_out") - 1 ))"
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
printf 'OOD fallback frontier: %s\n' "$ood_out"
printf 'OOD fallback summary: %s\n' "$ood_summary_out"
printf 'Task-relevant scope frontier: %s\n' "$scope_out"
printf 'Task-relevant scope summary: %s\n' "$scope_summary_out"
printf 'Sequential CUSUM scope frontier: %s\n' "$sequential_out"
printf 'Sequential CUSUM scope summary: %s\n' "$sequential_summary_out"
printf 'Bounded-risk e-process frontier: %s\n' "$eprocess_out"
printf 'Bounded-risk e-process summary: %s\n' "$eprocess_summary_out"
printf 'Lifetime alpha-spending frontier: %s\n' "$alpha_spending_out"
printf 'Lifetime alpha-spending summary: %s\n' "$alpha_spending_summary_out"
printf 'Persistent-vs-reset e-process frontier: %s\n' "$persistent_eprocess_out"
printf 'Persistent-vs-reset e-process summary: %s\n' "$persistent_eprocess_summary_out"
printf 'Provenance receipt: %s\n' "$receipt"
