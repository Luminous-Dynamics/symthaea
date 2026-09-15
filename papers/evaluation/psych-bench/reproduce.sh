#!/usr/bin/env bash
# Transactional reproduction for the Psych-Bench paper workflow.
#
# Default mode executes the four qualified command surfaces in a detached worktree
# at the exact current Git subject. Generated paper CSVs and convenience outputs are
# staged outside the canonical paper-data directory. A complete run emits a
# BLAKE3-bound reproduction receipt and a separate promotion manifest.
#
# This script NEVER promotes staged CSVs into papers/data/psych_bench automatically.
# Promotion remains a separate, deliberate action after receipt verification.
#
# Fast contract test:
#   ./reproduce.sh --self-test --output-dir /tmp/psych-bench-repro-selftest
#
# Real staged execution:
#   ./reproduce.sh
#   ./reproduce.sh --output-dir target/my-reproduction-run
#
# A successful receipt proves consistency/provenance for the bound transaction.
# It does not itself establish scientific validity, human comparability, causal
# interpretation, independent replication, or benchmark-suite compatibility.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../" && pwd)"
MODE="execute"
OUTPUT_DIR=""

usage() {
  cat <<'EOF'
usage: reproduce.sh [--self-test] [--output-dir PATH]

  --self-test        Exercise detached-worktree staging and receipt verification
                     with synthetic artifacts; does not run the expensive battery.
  --output-dir PATH  Final completed run directory. Must not already exist and
                     must resolve outside canonical paper-data/manuscript trees.
EOF
}

while (($#)); do
  case "$1" in
    --self-test)
      MODE="self_test"
      shift
      ;;
    --output-dir)
      if (($# < 2)); then
        printf '%s\n' '--output-dir requires a path' >&2
        exit 2
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$REPO"
SOURCE_SHA="$(git rev-parse HEAD)"
SOURCE_TREE="$(git rev-parse 'HEAD^{tree}')"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  printf '%s\n' 'tracked worktree is dirty; refusing reproduction' >&2
  exit 2
fi

RUSTC_SHORT="$(rustc --version)"
CARGO_SHORT="$(cargo --version)"
if [[ "$RUSTC_SHORT" != rustc\ 1.96.0* ]]; then
  printf 'expected Rust 1.96.0, found: %s\n' "$RUSTC_SHORT" >&2
  exit 2
fi
if [[ "$CARGO_SHORT" != cargo\ 1.96.0* ]]; then
  printf 'expected Cargo 1.96.0, found: %s\n' "$CARGO_SHORT" >&2
  exit 2
fi

SHORT_SHA="${SOURCE_SHA:0:12}"
RUN_PARENT="$REPO/target/psych-bench-reproduction"
if [[ -n "$OUTPUT_DIR" ]]; then
  if [[ "$OUTPUT_DIR" = /* ]]; then
    FINAL_DIR="$OUTPUT_DIR"
  else
    FINAL_DIR="$REPO/$OUTPUT_DIR"
  fi
else
  RUN_ID="${SHORT_SHA}-$(date -u +%Y%m%dT%H%M%SZ)-${MODE}"
  FINAL_DIR="$RUN_PARENT/$RUN_ID"
fi

# Canonicalize before any staging directory is created. This closes `..` and
# symlink-prefix escapes that could otherwise place an untracked reproduction
# bundle inside the canonical evidence/manuscript trees.
FINAL_DIR="$(realpath -m -- "$FINAL_DIR")"
CANONICAL_DATA="$(realpath -m -- "$REPO/papers/data/psych_bench")"
MANUSCRIPT_DIR="$(realpath -m -- "$REPO/papers/evaluation/psych-bench")"
case "$FINAL_DIR/" in
  "$CANONICAL_DATA/"*|"$MANUSCRIPT_DIR/"*)
    printf 'output directory must resolve outside canonical paper trees: %s\n' "$FINAL_DIR" >&2
    exit 2
    ;;
esac

if [[ -e "$FINAL_DIR" ]]; then
  printf 'output directory already exists: %s\n' "$FINAL_DIR" >&2
  exit 2
fi
FINAL_PARENT="$(dirname "$FINAL_DIR")"
mkdir -p "$FINAL_PARENT"
STAGING="$(mktemp -d "$FINAL_PARENT/.psych-bench-repro-staging.XXXXXX")"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/symthaea-psych-repro.XXXXXX")"
WORKTREE="$TMP_ROOT/repo"
FINALIZED=false

cleanup() {
  if [[ -n "${WORKTREE:-}" && -e "$WORKTREE/.git" ]]; then
    git -C "$REPO" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
  fi
  if [[ "${FINALIZED:-false}" != true && -n "${STAGING:-}" && -d "$STAGING" ]]; then
    rm -rf "$STAGING"
  fi
  if [[ -n "${TMP_ROOT:-}" && -d "$TMP_ROOT" ]]; then
    rm -rf "$TMP_ROOT"
  fi
}
trap cleanup EXIT INT TERM

git worktree add --detach "$WORKTREE" "$SOURCE_SHA" >/dev/null
if [[ "$(git -C "$WORKTREE" rev-parse HEAD)" != "$SOURCE_SHA" ]]; then
  printf '%s\n' 'detached staging worktree resolved the wrong subject' >&2
  exit 2
fi
if [[ -n "$(git -C "$WORKTREE" status --porcelain)" ]]; then
  printf '%s\n' 'detached staging worktree is not clean' >&2
  exit 2
fi

ARTIFACTS="$STAGING/artifacts"
PAPER_ARTIFACTS="$ARTIFACTS/paper_csv"
JOURNAL="$STAGING/step-journal.tsv"
RUSTC_FILE="$STAGING/rustc-version.txt"
CARGO_FILE="$STAGING/cargo-version.txt"
RECEIPT="$STAGING/reproduction_receipt.json"
PROMOTION="$STAGING/promotion_manifest.json"
STAGED_PAPER_DATA="$WORKTREE/papers/data/psych_bench"
mkdir -p "$ARTIFACTS" "$PAPER_ARTIFACTS" "$STAGED_PAPER_DATA"
: > "$JOURNAL"
rustc --version --verbose > "$RUSTC_FILE"
cargo --version > "$CARGO_FILE"

# Remove inherited historical CSVs from the detached worktree. Any CSV present
# after the generator runs is therefore a product of this reproduction attempt.
find "$STAGED_PAPER_DATA" -maxdepth 1 \( -type f -o -type l \) -name '*.csv' -delete

export CARGO_TARGET_DIR="$REPO/target/psych-bench-reproduction-build/$SOURCE_SHA"

run_real_reproduction() {
  cd "$WORKTREE"

  echo "[1/4] Full core benchmark runner (default configuration)"
  cargo run --locked --release --example run_psych_benchmarks \
    --package symthaea-psych-bench -- --json-output "$ARTIFACTS/out_default.json"
  printf 'core_runner\t0\n' >> "$JOURNAL"

  echo "[2/4] Full paper CSV generator"
  cargo run --locked --release --example psych_bench_paper_data \
    --package symthaea-psych-bench
  printf 'paper_csv\t0\n' >> "$JOURNAL"

  echo "[3/4] Multi-seed robustness (42, 123, 456, 789, 1024)"
  cargo run --locked --release --example multi_seed_robustness \
    --package symthaea-psych-bench > "$ARTIFACTS/out_stability.md"
  printf 'multi_seed_robustness\t0\n' >> "$JOURNAL"

  echo "[4/4] Seven-benchmark Qualia Confidence Matrix (seed 42)"
  cargo run --locked --release --example qualia_confidence_report \
    --package symthaea-psych-bench -- --seed 42 --json \
    > "$ARTIFACTS/out_qualia_confidence.json"
  printf 'qualia_confidence\t0\n' >> "$JOURNAL"
}

run_synthetic_self_test() {
  printf '{"self_test":true,"step":"core_runner"}\n' > "$ARTIFACTS/out_default.json"
  printf 'core_runner\t0\n' >> "$JOURNAL"

  local csv_names=(
    ablation_domains.csv
    cognitive_profile.csv
    correlations.csv
    neuromod_curves.csv
    neuromod_profiles.csv
    normative_zscores.csv
    reliability.csv
    sat_arcfluid.csv
    sat_curves.csv
    sat_flanker.csv
    sat_nback.csv
    sat_stroop.csv
    sat_visualsearch.csv
    sat_wcst.csv
  )
  local name
  for name in "${csv_names[@]}"; do
    printf 'self_test,artifact\n1,%s\n' "$name" > "$STAGED_PAPER_DATA/$name"
  done
  printf 'paper_csv\t0\n' >> "$JOURNAL"

  printf '# synthetic multi-seed self-test\n' > "$ARTIFACTS/out_stability.md"
  printf 'multi_seed_robustness\t0\n' >> "$JOURNAL"

  printf '{"self_test":true,"step":"qualia_confidence"}\n' \
    > "$ARTIFACTS/out_qualia_confidence.json"
  printf 'qualia_confidence\t0\n' >> "$JOURNAL"
}

if [[ "$MODE" == "self_test" ]]; then
  run_synthetic_self_test
else
  run_real_reproduction
fi

# Copy only generated CSVs after all four producer steps have succeeded. The Rust
# receipt helper owns the exact expected artifact census and fails on extras/missing.
cp -- "$STAGED_PAPER_DATA"/*.csv "$PAPER_ARTIFACTS/"

cd "$WORKTREE"
cargo run --locked --quiet --package symthaea-psych-bench \
  --example reproduction_receipt -- \
  build "$MODE" "$ARTIFACTS" "$JOURNAL" "$SOURCE_SHA" "$SOURCE_TREE" \
  "$WORKTREE/papers/evaluation/psych-bench/reproduce.sh" \
  "$WORKTREE/Cargo.lock" "$RUSTC_FILE" "$CARGO_FILE" "$RECEIPT" "$PROMOTION"

cargo run --locked --quiet --package symthaea-psych-bench \
  --example reproduction_receipt -- \
  verify "$MODE" "$ARTIFACTS" "$JOURNAL" "$SOURCE_SHA" "$SOURCE_TREE" \
  "$WORKTREE/papers/evaluation/psych-bench/reproduce.sh" \
  "$WORKTREE/Cargo.lock" "$RUSTC_FILE" "$CARGO_FILE" "$RECEIPT" "$PROMOTION"

# Recheck the invoking worktree before making the completed staged bundle visible.
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$SOURCE_SHA" ]]; then
  printf '%s\n' 'source subject changed during reproduction' >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]; then
  printf '%s\n' 'tracked source worktree changed during reproduction' >&2
  exit 2
fi

mv "$STAGING" "$FINAL_DIR"
FINALIZED=true

echo
echo "=== Transaction complete ==="
echo "Mode: $MODE"
echo "Source subject: $SOURCE_SHA"
echo "Staged bundle: $FINAL_DIR"
echo "Receipt: $FINAL_DIR/reproduction_receipt.json"
echo "Promotion manifest: $FINAL_DIR/promotion_manifest.json"
echo "Canonical papers/data/psych_bench was NOT modified by this script."
