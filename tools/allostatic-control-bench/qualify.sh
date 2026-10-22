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
  "tools/allostatic-control-bench/qualify.sh"
  "docs/research/ALLOSTATIC_CONTROL_BENCHMARK_V1.md"
  "docs/research/ROBUST_VIABILITY_ENVELOPE_V1.md"
  "docs/research/MULTISTEP_VIABILITY_TUBES_V1.md"
  "docs/research/OOD_REACHABILITY_FALLBACK_V1.md"
  "docs/research/TASK_RELEVANT_SCOPE_MONITOR_V1.md"
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
    if (($7 > $8) != ($9 == "true")) failed = 1
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
printf 'Provenance receipt: %s\n' "$receipt"
