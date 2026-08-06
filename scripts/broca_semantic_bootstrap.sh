#!/usr/bin/env bash
# broca_semantic_bootstrap.sh: one-time semantic_hv bootstrap pass.
#
# Part 4 of the curriculum-conditioned Broca fine-tuning bridge (see
# /home/tstoltz/.claude/plans/fuzzy-beaming-brook.md). Curriculum-derived
# training pairs now carry a real `semantic_hv` content-conditioning signal
# (Part 3), and the wiring is proven correct in isolation (Phase D) — but
# the regular curriculum cycle's deliberately ultra-conservative single
# epoch / lr=0.00003 (tuned specifically to avoid re-collapsing an
# already-converged model) isn't enough signal to teach that model to use
# a genuinely new region of its own input space (Phase E: a real cycle
# trained fine by that metric but showed zero real improvement on the
# actual gate). A toy sweep (Phase F.1) found several moderate (lr,
# epochs) combinations that DO teach a converged toy model to differentiate
# on semantic_hv without collapsing what it already knew.
#
# This script is that moderate bootstrap, applied ONCE to the real
# production checkpoint — NOT a replacement for `broca_curriculum_cycle.sh`,
# which keeps its conservative defaults for ordinary incremental updates on
# top of whatever this bootstrap (if promoted) leaves as the new baseline.
# Run manually and rarely — this is not a recurring cadence, and there is
# no systemd timer for it.
#
# Reuses the exact same candidate-directory layout, gate
# (broca_measurement_artifacts.sh / broca_checkpoint_compare.rs), and
# human-promotion step (broca_promote_candidate.sh) as the regular cycle —
# this script NEVER touches the production checkpoint directly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BROCA_DATA_DIR="${BROCA_DATA_DIR:-crates/domains/symthaea-broca/data}"
PRODUCTION_CHECKPOINT="$BROCA_DATA_DIR/models/broca-checkpoint-latest.bin"
BASE_CORPUS="$BROCA_DATA_DIR/train-combined-v8.jsonl"
EVAL_FILE="$BROCA_DATA_DIR/eval-creativity-v1.jsonl"
CURRICULUM_DERIVED_DIR="$BROCA_DATA_DIR/curriculum-derived"
CANDIDATES_ROOT="$BROCA_DATA_DIR/checkpoints/candidates"
BASELINE_ROOT="${BROCA_BASELINE_ROOT:-crates/domains/symthaea-broca/data/models/baselines}"
BASELINE_POINTER="$BROCA_DATA_DIR/models/measurements/CURRENT_BASELINE"

# The one real holdout batch with genuine semantic_hv, produced by the
# original (interrupted, then retried) Phase E cycle. Override if a newer
# semantic_hv-bearing holdout exists by the time this runs.
BOOTSTRAP_HOLDOUT="${BROCA_BOOTSTRAP_HOLDOUT:-$CURRICULUM_DERIVED_DIR/holdout/20260707T134451Z.holdout.jsonl}"

# Starting point per Phase F.1's toy sweep (all three candidates tried —
# lr=0.001/epochs=5, lr=0.003/epochs=8, lr=0.005/epochs=5 — succeeded at
# toy scale). These are NOT proven to transfer to the real, much larger,
# more thoroughly converged production checkpoint; the gate below decides
# that empirically, not this default. Expect to need a few real attempts
# at different values before something actually passes the improvement
# gate, same as any other genuinely new training recipe.
#
# 2026-07-08: two full-network real attempts (lr=0.0005 and lr=0.0001,
# both with freeze_embeddings=false, network_lr_scale=0.3 default) both
# destabilized -- fast at 0.0005, slower but still worsening across 2
# epochs at 0.0001. Mechanistic read (Phase F.3 toy test,
# test_bootstrap_recipe_sweep_frozen_embeddings): in both failed runs
# embeddings moved at the FULL base lr while the network moved at only
# 0.3x that -- so embeddings were actually the faster-moving component,
# despite the network being what needs to learn to use semantic_hv.
# Embeddings feed both the token-output layer and semantic_hv's own
# encoding, so a shifting embedding table is a plausible destabilizer.
# BROCA_BOOTSTRAP_FREEZE_EMBEDDINGS / BROCA_BOOTSTRAP_NETWORK_LR_SCALE
# let a real attempt isolate that: freeze embeddings entirely and drive
# the network's OWN effective lr (= BOOTSTRAP_LR * network_lr_scale) up
# directly, without ever perturbing embeddings.
BOOTSTRAP_EPOCHS="${BROCA_BOOTSTRAP_EPOCHS:-8}"
BOOTSTRAP_LR="${BROCA_BOOTSTRAP_LR:-0.0005}"
BOOTSTRAP_FREEZE_EMBEDDINGS="${BROCA_BOOTSTRAP_FREEZE_EMBEDDINGS:-0}"
BOOTSTRAP_NETWORK_LR_SCALE="${BROCA_BOOTSTRAP_NETWORK_LR_SCALE:-0.3}"
BOOTSTRAP_NETWORK_WARMUP="${BROCA_BOOTSTRAP_NETWORK_WARMUP:-0}"

cargo_locked_args=()
if [[ "${BROCA_CARGO_UNLOCKED:-0}" != "1" ]]; then
  cargo_locked_args+=(--locked)
fi

# ── Lock: shares the regular cycle's lock file deliberately — a bootstrap
# run and a regular curriculum cycle must never execute concurrently
# against the same production checkpoint / curriculum-derived directory. ──
mkdir -p target
LOCK_FILE="target/broca-curriculum-cycle.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "[broca-bootstrap] another cycle or bootstrap run is already in progress (lock held at $LOCK_FILE); exiting" >&2
  exit 1
fi
echo "[broca-bootstrap] lock acquired: $LOCK_FILE"

# ── Invariant: this script must never modify the production checkpoint. ──
production_hash_before=""
if [[ -f "$PRODUCTION_CHECKPOINT" ]]; then
  production_hash_before="$(sha256sum "$PRODUCTION_CHECKPOINT" | cut -d' ' -f1)"
fi
check_production_untouched() {
  if [[ -n "$production_hash_before" ]]; then
    local after
    after="$(sha256sum "$PRODUCTION_CHECKPOINT" | cut -d' ' -f1)"
    if [[ "$after" != "$production_hash_before" ]]; then
      echo "[broca-bootstrap] INVARIANT VIOLATION: production checkpoint changed during this script ($production_hash_before -> $after)" >&2
      exit 99
    fi
  fi
}
trap check_production_untouched EXIT

if [[ ! -s "$BOOTSTRAP_HOLDOUT" ]]; then
  echo "[broca-bootstrap] FAIL: semantic_hv holdout file not found or empty: $BOOTSTRAP_HOLDOUT" >&2
  echo "[broca-bootstrap] set BROCA_BOOTSTRAP_HOLDOUT to a real holdout file containing semantic_hv-bearing entries" >&2
  exit 1
fi

# ── Merge: base corpus + ALL accumulated curriculum-derived data (both
# channels-only historical batches and the semantic_hv-bearing batch) —
# identical merge logic to the regular cycle script, nothing new. No
# "check for new objectives" step: this is a deliberate one-time
# bootstrap over what already exists, not tied to curriculum growth. ──
shopt -s nullglob
derived_jsonl_files=("$CURRICULUM_DERIVED_DIR"/*.jsonl)
derived_manifest_files=("$CURRICULUM_DERIVED_DIR"/*.manifest.json)
shopt -u nullglob

if [[ ${#derived_jsonl_files[@]} -eq 0 ]]; then
  echo "[broca-bootstrap] no curriculum-derived pairs on disk; nothing to bootstrap with" >&2
  exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
candidate_dir="$CANDIDATES_ROOT/bootstrap-$timestamp"
mkdir -p "$candidate_dir/measurement"
candidate_checkpoint="$candidate_dir/candidate.bin"

merged_data="$candidate_dir/merged-training-data.jsonl"
cat "$BASE_CORPUS" "${derived_jsonl_files[@]}" > "$merged_data"

base_corpus_lines="$(wc -l < "$BASE_CORPUS")"
merged_lines="$(wc -l < "$merged_data")"

python3 - "$candidate_dir" "$BASE_CORPUS" "$base_corpus_lines" "$merged_lines" "${derived_jsonl_files[@]}" <<'PY'
import json, hashlib, sys

candidate_dir, base_corpus, base_lines, merged_lines, *derived_files = sys.argv[1:]

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "kind": "semantic_bootstrap",
    "base_corpus_path": base_corpus,
    "base_corpus_lines": int(base_lines),
    "base_corpus_sha256": sha256_of(base_corpus),
    "merged_lines": int(merged_lines),
    "curriculum_derived_files": [
        {"path": p, "sha256": sha256_of(p)} for p in derived_files
    ],
}
with open(f"{candidate_dir}/merged-training-manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
PY

if [[ ${#derived_manifest_files[@]} -gt 0 ]]; then
  python3 - "$candidate_dir" "${derived_manifest_files[@]}" <<'PY'
import json, sys

candidate_dir, *manifest_files = sys.argv[1:]
entries = []
for path in manifest_files:
    with open(path) as f:
        entries.extend(json.load(f))
with open(f"{candidate_dir}/curriculum-derived-manifest.json", "w") as f:
    json.dump(entries, f, indent=2)
PY
else
  echo "[]" > "$candidate_dir/curriculum-derived-manifest.json"
fi

echo "[broca-bootstrap] candidate directory: $candidate_dir ($merged_lines merged pairs)"

# ── Train: same backend-selection logic as the regular cycle script. ──
TRAIN_BACKEND="${BROCA_TRAIN_BACKEND:-cpu}"
train_feature_args=(--features simd)
train_nix_attr=".#broca-gpu"
case "$TRAIN_BACKEND" in
  cpu) ;;
  gpu)
    train_feature_args=(--features simd,gpu)
    export CUDA_COMPUTE_CAP="${BROCA_TRAIN_CUDA_COMPUTE_CAP:-75}"
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"
    ;;
  auto)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      TRAIN_BACKEND="gpu"
      train_feature_args=(--features simd,gpu)
      export CUDA_COMPUTE_CAP="${BROCA_TRAIN_CUDA_COMPUTE_CAP:-75}"
      export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"
    else
      TRAIN_BACKEND="cpu"
    fi
    ;;
  *)
    echo "[broca-bootstrap] BROCA_TRAIN_BACKEND must be cpu, gpu, or auto" >&2
    exit 2
    ;;
esac

run_train() {
  if [[ "$TRAIN_BACKEND" == "gpu" && -z "${IN_NIX_SHELL:-}" ]]; then
    nix develop "$train_nix_attr" --command "$@"
  else
    "$@"
  fi
}

train_extra_args=(--network-lr-scale "$BOOTSTRAP_NETWORK_LR_SCALE")
if [[ "$BOOTSTRAP_NETWORK_WARMUP" != "0" ]]; then
  train_extra_args+=(--network-warmup "$BOOTSTRAP_NETWORK_WARMUP")
fi
if [[ "$BOOTSTRAP_FREEZE_EMBEDDINGS" == "1" ]]; then
  train_extra_args+=(--freeze-embeddings)
fi

echo "[broca-bootstrap] training candidate checkpoint (epochs=$BOOTSTRAP_EPOCHS lr=$BOOTSTRAP_LR network_lr_scale=$BOOTSTRAP_NETWORK_LR_SCALE freeze_embeddings=$BOOTSTRAP_FREEZE_EMBEDDINGS network_warmup=$BOOTSTRAP_NETWORK_WARMUP, backend=$TRAIN_BACKEND)..."
run_train cargo run "${cargo_locked_args[@]}" -p symthaea-broca "${train_feature_args[@]}" --bin broca-train -- \
  --data "$merged_data" \
  --eval "$EVAL_FILE" \
  --epochs "$BOOTSTRAP_EPOCHS" \
  --lr "$BOOTSTRAP_LR" \
  --resume "$PRODUCTION_CHECKPOINT" \
  --output "$candidate_checkpoint" \
  "${train_extra_args[@]}"

# ── Cheap invariant checks before the expensive measurement run. ──
echo "[broca-bootstrap] running cheap invariant checks..."

if [[ ! -s "$candidate_checkpoint" ]]; then
  echo "[broca-bootstrap] FAIL: candidate checkpoint missing or empty at $candidate_checkpoint" >&2
  exit 1
fi

candidate_hash="$(sha256sum "$candidate_checkpoint" | cut -d' ' -f1)"
if [[ "$candidate_hash" == "$production_hash_before" ]]; then
  echo "[broca-bootstrap] FAIL: candidate checkpoint is byte-identical to production; training had no effect" >&2
  exit 1
fi

if [[ "$merged_lines" -lt "$base_corpus_lines" ]]; then
  echo "[broca-bootstrap] FAIL: merged dataset ($merged_lines lines) is smaller than base corpus ($base_corpus_lines lines)" >&2
  exit 1
fi

if [[ ! -s "$EVAL_FILE" ]]; then
  echo "[broca-bootstrap] FAIL: eval file $EVAL_FILE missing or empty" >&2
  exit 1
fi

echo "[broca-bootstrap] cheap invariants passed"

# ── Ensure a production baseline exists to compare against. ──
mkdir -p "$(dirname "$BASELINE_POINTER")"
baseline_dir=""
if [[ -f "$BASELINE_POINTER" ]]; then
  baseline_dir="$(cat "$BASELINE_POINTER")"
fi
export BROCA_DECODER_FEATURES="${BROCA_DECODER_FEATURES:-simd}"

if [[ -z "$baseline_dir" || ! -d "$baseline_dir" ]]; then
  baseline_name="production-$(date -u +%Y%m%dT%H%M%SZ)"
  echo "[broca-bootstrap] no valid production baseline recorded; capturing one now ($baseline_name)..."
  BROCA_CHECKPOINT_PATH="$PRODUCTION_CHECKPOINT" BROCA_SKIP_EXERCISM="${BROCA_SKIP_EXERCISM:-1}" scripts/broca_capture_baseline.sh "$baseline_name"
  baseline_dir="$BASELINE_ROOT/$baseline_name"
  echo "$baseline_dir" > "$BASELINE_POINTER"
fi
echo "[broca-bootstrap] using production baseline: $baseline_dir"

# ── Topic-coverage evidence on the semantic_hv holdout: production vs.
# candidate, exactly as Phase E validated actually catches a
# non-improving candidate (do not skip this the way an interrupted-then-
# retried regular cycle silently did — that gap is what let a broken
# candidate report PASSED once already). ──
echo "[broca-bootstrap] checking topic coverage on the semantic_hv holdout (production baseline)..."
baseline_topic_coverage_report="$candidate_dir/topic-coverage-baseline-report.json"
cargo run "${cargo_locked_args[@]}" -p symthaea-broca --features simd --bin broca-topic-coverage -- \
  --checkpoint "$PRODUCTION_CHECKPOINT" \
  --holdout "$BOOTSTRAP_HOLDOUT" \
  --json-out "$baseline_topic_coverage_report"
echo "[broca-bootstrap] production baseline topic coverage report: $baseline_topic_coverage_report"

echo "[broca-bootstrap] checking topic coverage on the semantic_hv holdout (candidate)..."
candidate_topic_coverage_report="$candidate_dir/topic-coverage-report.json"
cargo run "${cargo_locked_args[@]}" -p symthaea-broca --features simd --bin broca-topic-coverage -- \
  --checkpoint "$candidate_checkpoint" \
  --holdout "$BOOTSTRAP_HOLDOUT" \
  --json-out "$candidate_topic_coverage_report"
echo "[broca-bootstrap] candidate topic coverage report: $candidate_topic_coverage_report"

# ── Measure: same gate as the regular cycle, fail-closed. ──
echo "[broca-bootstrap] measuring candidate checkpoint (regression + improvement gates)..."
set +e
BROCA_CHECKPOINT_PATH="$candidate_checkpoint" \
BROCA_BASELINE_ARTIFACT_DIR="$baseline_dir" \
BROCA_FAIL_ON_REGRESSION=1 \
BROCA_SKIP_EXERCISM="${BROCA_SKIP_EXERCISM:-1}" \
BROCA_BASELINE_TOPIC_COVERAGE="$baseline_topic_coverage_report" \
BROCA_CANDIDATE_TOPIC_COVERAGE="$candidate_topic_coverage_report" \
BROCA_REQUIRE_IMPROVEMENT_EVALUATED=1 \
BROCA_MIN_KEYWORD_OVERLAP_IMPROVEMENT="${BROCA_MIN_KEYWORD_OVERLAP_IMPROVEMENT:-0.05}" \
BROCA_MIN_COHERENCE_IMPROVEMENT="${BROCA_MIN_COHERENCE_IMPROVEMENT:-0.02}" \
  scripts/broca_measurement_artifacts.sh "$candidate_dir/measurement"
measure_status=$?
set -e

if [[ "${BROCA_SKIP_EXERCISM:-1}" == "1" ]]; then
  echo "[broca-bootstrap] NOTE: exercism-bench skipped by default (measures the separate Liquid-Mamba pathway, not this checkpoint)."
fi

recorded_checkpoint="$(python3 -c "import json; print(json.load(open('$candidate_dir/measurement/decoder-ab.json')).get('checkpoint_path') or '')" 2>/dev/null || echo "")"
if [[ "$recorded_checkpoint" != "$candidate_checkpoint" ]]; then
  echo "[broca-bootstrap] FAIL: decoder-ab.json checkpoint_path ('$recorded_checkpoint') does not match candidate ('$candidate_checkpoint') — measurement gate is not checkpoint-aware" >&2
  exit 1
fi

if [[ "$measure_status" -ne 0 ]]; then
  echo "[broca-bootstrap] FAIL: candidate did not pass the regression and/or improvement gate (see $candidate_dir/measurement/checkpoint-compare.json)" >&2
  exit 1
fi

promotion_passed="$(python3 -c "import json; print(json.load(open('$candidate_dir/measurement/checkpoint-compare.json'))['promotion']['passed'])" 2>/dev/null || echo "False")"
if [[ "$promotion_passed" != "True" ]]; then
  regression_passed="$(python3 -c "import json; print(json.load(open('$candidate_dir/measurement/checkpoint-compare.json'))['promotion'].get('regression_passed'))" 2>/dev/null || echo "unknown")"
  improvement_passed="$(python3 -c "import json; print(json.load(open('$candidate_dir/measurement/checkpoint-compare.json'))['promotion'].get('improvement_passed'))" 2>/dev/null || echo "unknown")"
  echo "[broca-bootstrap] FAIL: promotion.passed is not true in checkpoint-compare.json (regression_passed=$regression_passed, improvement_passed=$improvement_passed)" >&2
  echo "[broca-bootstrap] this means the bootstrap recipe (epochs=$BOOTSTRAP_EPOCHS lr=$BOOTSTRAP_LR) didn't work well enough this attempt — try different BROCA_BOOTSTRAP_EPOCHS/BROCA_BOOTSTRAP_LR values, informed by Phase F.1's toy sweep, not by re-running the same values again" >&2
  exit 1
fi

# ── Collapse/repetition check on the semantic_hv holdout — same check the
# regular cycle applies, same reason (regression metrics alone don't
# detect degenerate/repetitive output). ──
collapse_check_status=0
python3 - "$candidate_topic_coverage_report" <<'PY' || collapse_check_status=$?
import json, sys
from collections import Counter

path = sys.argv[1]
with open(path) as f:
    report = json.load(f)

cases = report.get("cases", [])
if not cases:
    print("[broca-bootstrap] collapse check: no cases in topic-coverage report, skipping")
    sys.exit(0)

texts = [c.get("generated_text", "") for c in cases]
counts = Counter(texts)
most_common_text, most_common_count = counts.most_common(1)[0]
duplicate_ratio = most_common_count / len(texts)

print(
    f"[broca-bootstrap] collapse check: {len(texts)} cases, "
    f"most-repeated generated_text appears {most_common_count} times "
    f"(ratio {duplicate_ratio:.2f})"
)

MAX_DUPLICATE_RATIO = 0.25
if duplicate_ratio > MAX_DUPLICATE_RATIO and most_common_text.strip():
    print(
        f"[broca-bootstrap] FAIL: {most_common_count}/{len(texts)} objectives "
        f"produced byte-identical generated text (ratio {duplicate_ratio:.2f} "
        f"> {MAX_DUPLICATE_RATIO}) -- candidate has likely collapsed. Example: "
        f"{most_common_text[:200]!r}",
        file=sys.stderr,
    )
    sys.exit(1)
PY
if [[ "$collapse_check_status" -ne 0 ]]; then
  echo "[broca-bootstrap] FAIL: collapse/repetition check failed (see above) — refusing to write PROMOTION_READY.json" >&2
  exit 1
fi

# ── Write the promotion-ready marker. Still touches nothing under
# production — a human runs broca_promote_candidate.sh to act on this. ──
python3 - "$candidate_dir" "$candidate_checkpoint" <<'PY'
import json, sys
from datetime import datetime, timezone

candidate_dir, candidate_checkpoint = sys.argv[1:]

with open(f"{candidate_dir}/curriculum-derived-manifest.json") as f:
    contributing = [e["objective_id"] for e in json.load(f)]

with open(f"{candidate_dir}/measurement/checkpoint-compare.json") as f:
    compare = json.load(f)

def load_topic_coverage_summary(path):
    try:
        with open(path) as f:
            report = json.load(f)
    except FileNotFoundError:
        return None
    return {
        "report_path": path,
        "num_objectives": report.get("num_objectives"),
        "avg_keyword_overlap": report.get("avg_keyword_overlap"),
        "avg_generated_coherence": report.get("avg_generated_coherence"),
        "veto_count": report.get("veto_count"),
    }

topic_coverage = load_topic_coverage_summary(f"{candidate_dir}/topic-coverage-report.json")
topic_coverage_baseline = load_topic_coverage_summary(
    f"{candidate_dir}/topic-coverage-baseline-report.json"
)

promotion = compare.get("promotion", {})

promotion_ready = {
    "kind": "semantic_bootstrap",
    "candidate_checkpoint": candidate_checkpoint,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "contributing_objective_ids": contributing,
    "checkpoint_compare": compare,
    "regression_passed": promotion.get("regression_passed"),
    "improvement_passed": promotion.get("improvement_passed"),
    "improvement_evaluated": promotion.get("improvement_evaluated"),
    "topic_coverage": topic_coverage,
    "topic_coverage_baseline": topic_coverage_baseline,
}
with open(f"{candidate_dir}/PROMOTION_READY.json", "w") as f:
    json.dump(promotion_ready, f, indent=2)
PY

echo "[broca-bootstrap] PASSED — candidate ready for human review: $candidate_dir"
echo "[broca-bootstrap] read $candidate_dir/topic-coverage-report.json before promoting — it's evidence of what the model actually generates now, not just gate-passing numbers"
echo "[broca-bootstrap] to promote: scripts/broca_promote_candidate.sh $candidate_dir --i-understand-this-requires-restart"
