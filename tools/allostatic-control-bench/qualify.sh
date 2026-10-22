#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

manifest="tools/allostatic-control-bench/Cargo.toml"
source_files=(
  ".github/workflows/allostatic-control-bench.yml"
  "tools/allostatic-control-bench/Cargo.toml"
  "tools/allostatic-control-bench/Cargo.lock"
  "tools/allostatic-control-bench/external-benchmark-protocol-v1.json"
  "tools/allostatic-control-bench/src/lib.rs"
  "tools/allostatic-control-bench/src/calibration.rs"
  "tools/allostatic-control-bench/src/model_mismatch.rs"
  "tools/allostatic-control-bench/src/robust_viability.rs"
  "tools/allostatic-control-bench/src/paired_statistics.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-paired-eval-bench.rs"
  "docs/research/PAIRED_RUNTIME_ASSURANCE_INFERENCE_V1.md"
  "tools/allostatic-control-bench/src/main.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-control-report.rs"
  "tools/allostatic-control-bench/src/bin/allostatic-model-mismatch-bench.rs"
  "tools/allostatic-control-bench/qualify.sh"
  "docs/research/ALLOSTATIC_CONTROL_BENCHMARK_V1.md"
  "docs/research/ROBUST_VIABILITY_ENVELOPE_V1.md"
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
paired_eval_summary_out="${out_base}-paired-eval-selftest-summary.csv"
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

# Exercise paired inference on a deterministic synthetic fixture. This tests the analysis tool only.
fixture="$tmpdir/paired-eval-fixture.csv"
printf 'scenario,seed,policy,shielded,task_reward,unsafe_episode\n' > "$fixture"
for scenario in nominal shifted; do
  for shielded in false true; do
    for seed in $(seq 0 29); do
      baseline_reward="$(awk -v s="$seed" 'BEGIN { printf "%.6f", 1 + s / 100 }')"
      candidate_reward="$(awk -v s="$seed" -v c="$scenario" 'BEGIN { d = (c == "nominal" ? 0.10 : 0.05); printf "%.6f", 1 + s / 100 + d }')"
      baseline_violation=false
      candidate_violation=false
      if [[ "$scenario" == "shifted" ]] && (( seed % 5 == 0 )); then baseline_violation=true; fi
      if [[ "$scenario" == "shifted" ]] && (( seed % 10 == 0 )); then candidate_violation=true; fi
      printf '%s,%s,global_margin,%s,%s,%s\n' "$scenario" "$seed" "$shielded" "$baseline_reward" "$baseline_violation" >> "$fixture"
      printf '%s,%s,conditioned_policy,%s,%s,%s\n' "$scenario" "$seed" "$shielded" "$candidate_reward" "$candidate_violation" >> "$fixture"
    done
  done
done
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-paired-eval-bench -- "$fixture" "$tmpdir/paired-eval-a.csv" task_reward unsafe_episode global_margin conditioned_policy 10000
cargo run --locked --release --manifest-path "$manifest" --bin allostatic-paired-eval-bench -- "$fixture" "$tmpdir/paired-eval-b.csv" task_reward unsafe_episode global_margin conditioned_policy 10000
if ! cmp -s "$tmpdir/paired-eval-a.csv" "$tmpdir/paired-eval-b.csv"; then
  printf 'FAIL: repeated paired-inference report is not byte-identical\n' >&2
  exit 1
fi
if ! awk -F, 'NR == 1 { if (NF != 22) failed = 1; next }
  { if (NF != 22 || $6 != 30) failed = 1; rows++ }
  END { if (NR != 5 || rows != 4 || failed) exit 1 }
' "$tmpdir/paired-eval-a.csv"; then
  printf 'FAIL: paired-inference schema or condition coverage is invalid\n' >&2
  exit 1
fi

cp "$tmpdir/paired-eval-a.csv" "$paired_eval_summary_out"
cp "$tmpdir/run-a.csv" "$out_csv"
cp "$tmpdir/summary-a.csv" "$summary_out"
cp "$tmpdir/paired-a.csv" "$paired_out"
cp "$tmpdir/mismatch-a.csv" "$mismatch_out"
cp "$tmpdir/mismatch-summary-a.csv" "$mismatch_summary_out"
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
  printf 'paired_eval_selftest_sha256=%s\n' "$(sha256sum "$paired_eval_summary_out" | awk '{print $1}')"
  printf 'paired_eval_selftest_rows_excluding_header=%s\n' "$(( $(wc -l < "$paired_eval_summary_out") - 1 ))"
  printf '\nsource_sha256:\n'
  sha256sum "${source_files[@]}"
} > "$receipt"

printf 'Qualified episodes: %s\n' "$out_csv"
printf 'Distributional summary: %s\n' "$summary_out"
printf 'Paired shield deltas: %s\n' "$paired_out"
printf 'Model-mismatch safety frontier: %s\n' "$mismatch_out"
printf 'Model-mismatch frontier summary: %s\n' "$mismatch_summary_out"
printf 'Provenance receipt: %s\n' "$receipt"
printf 'Paired-inference self-test summary (synthetic only): %s\n' "$paired_eval_summary_out"
