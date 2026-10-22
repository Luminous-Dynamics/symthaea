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
  "tools/allostatic-control-bench/src/bin/allostatic-eprocess-shift-power-surface.rs"
  "tools/allostatic-control-bench/qualify.sh"
  "docs/research/ALLOSTATIC_CONTROL_BENCHMARK_V1.md"
  "docs/research/ROBUST_VIABILITY_ENVELOPE_V1.md"
  "docs/research/MULTISTEP_VIABILITY_TUBES_V1.md"
  "docs/research/OOD_REACHABILITY_FALLBACK_V1.md"
  "docs/research/TASK_RELEVANT_SCOPE_MONITOR_V1.md"
  "docs/research/SEQUENTIAL_SCOPE_CUSUM_V1.md"
  "docs/research/BOUNDED_RISK_EPROCESS_V1.md"
  "docs/research/EPROCESS_SHIFT_POWER_SURFACE_V1.md"
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
power_surface_out="${out_base}-eprocess-shift-power.csv"
power_surface_summary_out="${out_base}-eprocess-shift-power-summary.csv"
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

# Sweep predeclared e-process betting rates over shift magnitudes and temporal
# profiles. This contrasts per-episode risk testing with change detection; it
# does not re-use the held-out episodes to select a betting rate.
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-eprocess-shift-power-surface -- \
  "$tmpdir/power-a.csv" "$tmpdir/power-summary-a.csv"
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-eprocess-shift-power-surface -- \
  "$tmpdir/power-b.csv" "$tmpdir/power-summary-b.csv"
if ! cmp -s "$tmpdir/power-a.csv" "$tmpdir/power-b.csv" ||
   ! cmp -s "$tmpdir/power-summary-a.csv" "$tmpdir/power-summary-b.csv"; then
  printf 'FAIL: repeated e-process shift-surface outputs are not byte-identical\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 44) failed = 1; next }
  {
    if (NF != 44) failed = 1
    if ($1 != "null_low" && $1 != "null_boundary" &&
        $1 != "persistent_0175" && $1 != "persistent_020" &&
        $1 != "persistent_025" && $1 != "persistent_035" &&
        $1 != "ramp_to_020" && $1 != "ramp_to_030" &&
        $1 != "ramp_to_045" && $1 != "burst_008" &&
        $1 != "burst_024" && $1 != "recurrent_bursts" &&
        $1 != "stealth_paired") failed = 1
    if ($2 != "true" && $2 != "false") failed = 1
    if ($3 != "true" && $3 != "false") failed = 1
    if ($4 != "true" && $4 != "false") failed = 1
    if ($5 !~ /^[0-9]+$/ || $5 < 0 || $5 >= 5000) failed = 1
    if ($6 !~ /^[0-9]+$/ || $7 != 256) failed = 1
    for (i = 8; i <= 10; i++) {
      if ($i !~ /^[0-9]*[.]?[0-9]+$/ || $i < 0 || $i > 1) failed = 1
    }
    if ($9 < $8 || $11 !~ /^[0-9]+$/ || $11 > 256) failed = 1
    for (i = 12; i <= 12; i++) if ($i !~ /^[0-9]+$/ || $i < 0 || $i > 256) failed = 1
    if ($13 != "true" && $13 != "false") failed = 1
    for (i = 14; i <= 18; i++) if ($i !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    if ($19 !~ /^[0-9]+$/ || $19 < 0 || $19 > 256) failed = 1
    if ($20 != "true" && $20 != "false") failed = 1
    for (i = 21; i <= 25; i++) if ($i !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    for (i = 26; i <= 40; i++) {
      if ((i - 26) % 3 == 0) {
        if ($i !~ /^[0-9]+$/ || $i < 0 || $i > 256) failed = 1
      } else if ((i - 26) % 3 == 1) {
        if ($i != "true" && $i != "false") failed = 1
      } else if ($i !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    }
    if (($12 < 256) != ($13 == "true")) failed = 1
    if (($19 < 256) != ($20 == "true")) failed = 1
    if (($26 < 256) != ($27 == "true")) failed = 1
    if (($29 < 256) != ($30 == "true")) failed = 1
    if (($32 < 256) != ($33 == "true")) failed = 1
    if (($35 < 256) != ($36 == "true")) failed = 1
    if (($38 < 256) != ($39 == "true")) failed = 1
    if (($41 - log(100) > 0.000001) || (log(100) - $41 > 0.000001)) failed = 1
    if (($42 - 0.15 > 0.000001) || (0.15 - $42 > 0.000001)) failed = 1
    if (($43 - 0.05 > 0.000001) || (0.05 - $43 > 0.000001)) failed = 1
    if ($44 != 5) failed = 1

    max_low = $16
    if ($17 > max_low) max_low = $17
    if ($18 > max_low) max_low = $18
    low_mix = max_low + log((exp($16 - max_low) + exp($17 - max_low) +
      exp($18 - max_low)) / 3)
    if (($14 - low_mix > 0.000001) || (low_mix - $14 > 0.000001)) failed = 1
    max_high = $23
    if ($24 > max_high) max_high = $24
    if ($25 > max_high) max_high = $25
    high_mix = max_high + log((exp($23 - max_high) + exp($24 - max_high) +
      exp($25 - max_high)) / 3)
    if (($21 - high_mix > 0.000001) || (high_mix - $21 > 0.000001)) failed = 1

    if ($1 == "null_low" &&
        ($2 != "false" || $3 != "false" || $4 != "true" ||
         $8 != 0.1 || $9 != 0.1)) failed = 1
    if ($1 == "null_boundary" &&
        ($2 != "false" || $3 != "false" || $4 != "true" ||
         $8 != 0.15 || $9 != 0.15)) failed = 1
    if ($1 == "stealth_paired" &&
        ($2 != "true" || $3 != "false" || $4 != "false" ||
         $8 != 0.1 || $9 != 0.1)) failed = 1
    if ($1 != "null_low" && $1 != "null_boundary" && $1 != "stealth_paired" &&
        ($2 != "true" || $3 != "true" || $4 != "false")) failed = 1

    key = $1 SUBSEP $5
    if (seen[key]++) failed = 1
    if ($1 == "null_low") {
      for (i = 5; i <= 44; i++) paired_null[$5 SUBSEP i] = $i
    }
    if ($1 == "stealth_paired") {
      for (i = 5; i <= 44; i++) {
        if ($i != paired_null[$5 SUBSEP i]) failed = 1
      }
    }
    scenario_rows[$1]++
    rows++
  }
  END {
    if (NR != 65001 || rows != 65000 || failed) exit 1
    for (name in scenario_rows) if (scenario_rows[name] != 5000) exit 1
    if (scenario_rows["null_low"] != 5000 ||
        scenario_rows["null_boundary"] != 5000 ||
        scenario_rows["persistent_0175"] != 5000 ||
        scenario_rows["persistent_020"] != 5000 ||
        scenario_rows["persistent_025"] != 5000 ||
        scenario_rows["persistent_035"] != 5000 ||
        scenario_rows["ramp_to_020"] != 5000 ||
        scenario_rows["ramp_to_030"] != 5000 ||
        scenario_rows["ramp_to_045"] != 5000 ||
        scenario_rows["burst_008"] != 5000 ||
        scenario_rows["burst_024"] != 5000 ||
        scenario_rows["recurrent_bursts"] != 5000 ||
        scenario_rows["stealth_paired"] != 5000) exit 1
  }
' "$tmpdir/power-a.csv"; then
  printf 'FAIL: e-process power-surface raw schema, threshold, mixture arithmetic or stealth pairing is invalid\n' >&2
  exit 1
fi

if ! awk -F, '
  NR == 1 { if (NF != 87) failed = 1; next }
  {
    if (NF != 87) failed = 1
    if ($1 != "null_low" && $1 != "null_boundary" &&
        $1 != "persistent_0175" && $1 != "persistent_020" &&
        $1 != "persistent_025" && $1 != "persistent_035" &&
        $1 != "ramp_to_020" && $1 != "ramp_to_030" &&
        $1 != "ramp_to_045" && $1 != "burst_008" &&
        $1 != "burst_024" && $1 != "recurrent_bursts" &&
        $1 != "stealth_paired") failed = 1
    if ($2 != "true" && $2 != "false") failed = 1
    if ($3 != "true" && $3 != "false") failed = 1
    if ($4 != "true" && $4 != "false") failed = 1
    if ($5 != 5000) failed = 1
    if ($6 !~ /^[0-9]*[.]?[0-9]+$/ || $7 !~ /^[0-9]*[.]?[0-9]+$/ ||
        $8 !~ /^[0-9]*[.]?[0-9]+$/ || $8 < $6) failed = 1
    if (($1 == "null_low" || $1 == "null_boundary") &&
        ($2 != "false" || $3 != "false" || $4 != "true")) failed = 1
    if ($1 == "stealth_paired" &&
        ($2 != "true" || $3 != "false" || $4 != "false")) failed = 1
    if ($1 != "null_low" && $1 != "null_boundary" && $1 != "stealth_paired" &&
        ($2 != "true" || $3 != "true" || $4 != "false")) failed = 1

    for (start = 11; start <= 77; start += 11) {
      alarms = $(start)
      rate = $(start + 1)
      lower = $(start + 2)
      upper = $(start + 3)
      false_alarms = $(start + 4)
      risk_misses = $(start + 5)
      latent_misses = $(start + 6)
      delay = $(start + 7)
      p95 = $(start + 8)
      median = $(start + 9)
      mean_log = $(start + 10)
      if (alarms !~ /^[0-9]+$/ || alarms > 5000 ||
          false_alarms !~ /^[0-9]+$/ || false_alarms > 5000 ||
          risk_misses !~ /^[0-9]+$/ || risk_misses > 5000 ||
          latent_misses !~ /^[0-9]+$/ || latent_misses > 5000) failed = 1
      if (rate !~ /^[0-9]*[.]?[0-9]+$/ || lower !~ /^[0-9]*[.]?[0-9]+$/ ||
          upper !~ /^[0-9]*[.]?[0-9]+$/ || lower > rate || rate > upper ||
          (rate - alarms/5000 > 0.000001) || (alarms/5000 - rate > 0.000001)) failed = 1
      if ($3 == "true") {
        if (false_alarms != 0 || risk_misses != 5000 - alarms ||
            latent_misses != 5000 - alarms) failed = 1
      } else {
        if (false_alarms != alarms || risk_misses != 0) failed = 1
        if ($2 == "true" && latent_misses != 5000 - alarms) failed = 1
        if ($2 == "false" && latent_misses != 0) failed = 1
      }
      if ($4 == "true" && false_alarms > 75) failed = 1
      if (alarms == 0 && (delay != "NA" || p95 != "NA" || median != "NA")) failed = 1
      if (alarms > 0 &&
          (delay !~ /^[0-9]*[.]?[0-9]+$/ || p95 !~ /^[0-9]+$/ ||
           median !~ /^[0-9]+$/ || delay < 0 || delay > 255 ||
           p95 < 0 || p95 > 255 || median < 0 || median > 255)) failed = 1
      if (mean_log !~ /^-?[0-9]*[.]?[0-9]+$/) failed = 1
    }

    # Power gates are deliberately limited to predeclared strong cells.
    # Moderate alternatives and bursts are measured rather than forced to pass.
    if ($1 == "persistent_025" && ($11 < 3750 || $44 < 4000)) failed = 1
    if ($1 == "persistent_035" &&
        ($11 < 4500 || $22 < 4500 || $44 < 4500 || $55 < 4500)) failed = 1
    key = $1
    if (seen[key]++) failed = 1
    if ($1 == "null_low") {
      for (i = 11; i <= 87; i++) null_low_summary[i] = $i
    }
    if ($1 == "stealth_paired") {
      for (i = 11; i <= 87; i++) {
        # Latent-shift misses should differ: the paired null has no latent shift.
        if ((i - 11) % 11 != 6 && $i != null_low_summary[i]) failed = 1
      }
    }
    rows++
  }
  END { if (NR != 14 || rows != 13 || failed) exit 1 }
' "$tmpdir/power-summary-a.csv"; then
  printf 'FAIL: e-process power-surface summary schema, confidence intervals, null rate or power gates are invalid\n' >&2
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
cp "$tmpdir/power-a.csv" "$power_surface_out"
cp "$tmpdir/power-summary-a.csv" "$power_surface_summary_out"
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
  printf 'eprocess_power_surface_sha256=%s\n' "$(sha256sum "$power_surface_out" | awk '{print $1}')"
  printf 'eprocess_power_surface_rows_excluding_header=%s\n' "$(( $(wc -l < "$power_surface_out") - 1 ))"
  printf 'eprocess_power_surface_summary_sha256=%s\n' "$(sha256sum "$power_surface_summary_out" | awk '{print $1}')"
  printf 'eprocess_power_surface_summary_rows_excluding_header=%s\n' "$(( $(wc -l < "$power_surface_summary_out") - 1 ))"
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
printf 'E-process shift power surface: %s\n' "$power_surface_out"
printf 'E-process shift power summary: %s\n' "$power_surface_summary_out"
printf 'Provenance receipt: %s\n' "$receipt"
