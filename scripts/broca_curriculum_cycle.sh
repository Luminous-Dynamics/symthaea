#!/usr/bin/env bash
# broca_curriculum_cycle.sh: curriculum-conditioned Broca fine-tuning, one cycle.
#
# Pipeline: check for new curriculum objectives -> synthesize training pairs
# (broca-curriculum-sync) -> merge with the base corpus -> fine-tune a
# candidate checkpoint (broca-train --resume) -> gate it against the
# production checkpoint (broca_measurement_artifacts.sh, checkpoint-aware
# since Phase 0) -> if it passes, write PROMOTION_READY.json.
#
# This script NEVER touches the production checkpoint. Promotion is always a
# separate, explicit human action via scripts/broca_promote_candidate.sh.
# See /home/tstoltz/.claude/plans/fuzzy-beaming-brook.md for the full design.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BROCA_DATA_DIR="${BROCA_DATA_DIR:-crates/domains/symthaea-broca/data}"
PRODUCTION_CHECKPOINT="$BROCA_DATA_DIR/models/broca-checkpoint-latest.bin"
BASE_CORPUS="$BROCA_DATA_DIR/train-combined-v8.jsonl"
EVAL_FILE="$BROCA_DATA_DIR/eval-creativity-v1.jsonl"
CURRICULUM_DERIVED_DIR="$BROCA_DATA_DIR/curriculum-derived"
CANDIDATES_ROOT="$BROCA_DATA_DIR/checkpoints/candidates"
BASELINE_ROOT="${BROCA_BASELINE_ROOT:-target/broca-baselines}"
BASELINE_POINTER="$BROCA_DATA_DIR/models/measurements/CURRENT_BASELINE"

MIN_NEW_OBJECTIVES="${BROCA_CURRICULUM_MIN_NEW_OBJECTIVES:-20}"
CYCLE_EPOCHS="${BROCA_CYCLE_EPOCHS:-3}"
CYCLE_LR="${BROCA_CYCLE_LR:-0.0002}"

cargo_locked_args=()
if [[ "${BROCA_CARGO_UNLOCKED:-0}" != "1" ]]; then
  cargo_locked_args+=(--locked)
fi

# ── Lock: never allow two cycles (scheduled + manual, or two schedules) to
# race on shared state (curriculum-derived files, bridge state, production
# checkpoint reads). ──────────────────────────────────────────────────────
mkdir -p target
LOCK_FILE="target/broca-curriculum-cycle.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "[broca-cycle] another cycle is already running (lock held at $LOCK_FILE); exiting" >&2
  exit 1
fi
echo "[broca-cycle] lock acquired: $LOCK_FILE"

# ── Invariant: this script must never modify the production checkpoint.
# Checked unconditionally on every exit path, not just the happy path. ────
production_hash_before=""
if [[ -f "$PRODUCTION_CHECKPOINT" ]]; then
  production_hash_before="$(sha256sum "$PRODUCTION_CHECKPOINT" | cut -d' ' -f1)"
fi
check_production_untouched() {
  if [[ -n "$production_hash_before" ]]; then
    local after
    after="$(sha256sum "$PRODUCTION_CHECKPOINT" | cut -d' ' -f1)"
    if [[ "$after" != "$production_hash_before" ]]; then
      echo "[broca-cycle] INVARIANT VIOLATION: production checkpoint changed during this script ($production_hash_before -> $after)" >&2
      exit 99
    fi
  fi
}
trap check_production_untouched EXIT

# ── Step 1: cheap check for new objectives before doing anything expensive ─
sync_cmd=(cargo run "${cargo_locked_args[@]}" --no-default-features --features ssm_language --bin broca-curriculum-sync --)

export BROCA_BRIDGE_STATE_PATH="${BROCA_BRIDGE_STATE_PATH:-$BROCA_DATA_DIR/broca_curriculum_bridge_state.json}"
export BROCA_CURRICULUM_DERIVED_DIR="$CURRICULUM_DERIVED_DIR"
export BROCA_BASE_CORPUS_PATH="$BASE_CORPUS"

echo "[broca-cycle] checking for new curriculum objectives..."
new_count="$("${sync_cmd[@]}" --count-only 2>/dev/null | tail -1)"
if ! [[ "$new_count" =~ ^[0-9]+$ ]]; then
  echo "[broca-cycle] could not determine new-objective count (got '$new_count'); treating as 0" >&2
  new_count=0
fi
echo "[broca-cycle] new objectives pending: $new_count (threshold: $MIN_NEW_OBJECTIVES)"

if [[ "$new_count" -lt "$MIN_NEW_OBJECTIVES" ]]; then
  echo "[broca-cycle] below threshold; nothing to do this cycle"
  exit 0
fi

# ── Step 2: synthesize training pairs for real ─────────────────────────────
echo "[broca-cycle] synthesizing training pairs for new objectives..."
"${sync_cmd[@]}"

shopt -s nullglob
derived_jsonl_files=("$CURRICULUM_DERIVED_DIR"/*.jsonl)
derived_manifest_files=("$CURRICULUM_DERIVED_DIR"/*.manifest.json)
shopt -u nullglob

if [[ ${#derived_jsonl_files[@]} -eq 0 ]]; then
  echo "[broca-cycle] no curriculum-derived pairs on disk; nothing to train on"
  exit 0
fi

# ── Step 3: self-contained candidate directory ─────────────────────────────
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
candidate_dir="$CANDIDATES_ROOT/$timestamp"
mkdir -p "$candidate_dir/measurement"
candidate_checkpoint="$candidate_dir/candidate.bin"

# Full-history replay: base corpus + every curriculum-derived batch so far,
# not just this cycle's new batch (see plan rationale — --resume only
# reinforces each cycle's data once, so prior batches need replay too).
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

echo "[broca-cycle] candidate directory: $candidate_dir ($merged_lines merged pairs)"

# ── Step 4: incremental fine-tune, resumed from production ─────────────────
# BROCA_TRAIN_FEATURES defaults to CPU-only "simd". Note "gpu" is a separate
# Cargo feature from "mamba"/"mamba-cpu" (gpu = candle-core/cuda + candle-nn,
# nothing more) — it does NOT pull in the liquid_mamba module, so it doesn't
# hit the broken symthaea-broca-tools migration (dangling sovereignty_bridge/
# wasm_architect/etc. references) that "simd" alone already sidesteps. Once
# the GPU driver/library mismatch on this machine is cleared by a reboot,
# set BROCA_TRAIN_FEATURES=simd,gpu to use it — full-corpus CPU training is
# impractically slow (a 1-epoch pass over the ~5.8k-pair base corpus alone
# was still running after 48 CPU-minutes in testing).
TRAIN_FEATURES="${BROCA_TRAIN_FEATURES:-simd}"
echo "[broca-cycle] training candidate checkpoint (epochs=$CYCLE_EPOCHS lr=$CYCLE_LR, features=$TRAIN_FEATURES)..."
cargo run "${cargo_locked_args[@]}" -p symthaea-broca --features "$TRAIN_FEATURES" --bin broca-train -- \
  --data "$merged_data" \
  --eval "$EVAL_FILE" \
  --epochs "$CYCLE_EPOCHS" \
  --lr "$CYCLE_LR" \
  --resume "$PRODUCTION_CHECKPOINT" \
  --output "$candidate_checkpoint"

# ── Step 5: cheap invariant checks before the expensive measurement run ────
echo "[broca-cycle] running cheap invariant checks..."

if [[ ! -s "$candidate_checkpoint" ]]; then
  echo "[broca-cycle] FAIL: candidate checkpoint missing or empty at $candidate_checkpoint" >&2
  exit 1
fi

candidate_hash="$(sha256sum "$candidate_checkpoint" | cut -d' ' -f1)"
if [[ "$candidate_hash" == "$production_hash_before" ]]; then
  echo "[broca-cycle] FAIL: candidate checkpoint is byte-identical to production; training had no effect" >&2
  exit 1
fi

if [[ "$merged_lines" -lt "$base_corpus_lines" ]]; then
  echo "[broca-cycle] FAIL: merged dataset ($merged_lines lines) is smaller than base corpus ($base_corpus_lines lines)" >&2
  exit 1
fi

if [[ ! -s "$EVAL_FILE" ]]; then
  echo "[broca-cycle] FAIL: eval file $EVAL_FILE missing or empty" >&2
  exit 1
fi

echo "[broca-cycle] cheap invariants passed"

# ── Step 6: ensure a production baseline exists to compare against ────────
mkdir -p "$(dirname "$BASELINE_POINTER")"
baseline_dir=""
if [[ -f "$BASELINE_POINTER" ]]; then
  baseline_dir="$(cat "$BASELINE_POINTER")"
fi
if [[ -z "$baseline_dir" || ! -d "$baseline_dir" ]]; then
  baseline_name="production-$(date -u +%Y%m%dT%H%M%SZ)"
  echo "[broca-cycle] no valid production baseline recorded; capturing one now ($baseline_name)..."
  BROCA_CHECKPOINT_PATH="$PRODUCTION_CHECKPOINT" scripts/broca_capture_baseline.sh "$baseline_name"
  baseline_dir="$BASELINE_ROOT/$baseline_name"
  echo "$baseline_dir" > "$BASELINE_POINTER"
fi
echo "[broca-cycle] using production baseline: $baseline_dir"

# ── Step 7: measure the candidate against that baseline, fail-closed ───────
echo "[broca-cycle] measuring candidate checkpoint..."
set +e
BROCA_CHECKPOINT_PATH="$candidate_checkpoint" \
BROCA_BASELINE_ARTIFACT_DIR="$baseline_dir" \
BROCA_FAIL_ON_REGRESSION=1 \
BROCA_SKIP_EXERCISM="${BROCA_SKIP_EXERCISM:-1}" \
  scripts/broca_measurement_artifacts.sh "$candidate_dir/measurement"
measure_status=$?
set -e

if [[ "${BROCA_SKIP_EXERCISM:-1}" == "1" ]]; then
  echo "[broca-cycle] NOTE: exercism-bench skipped by default for this bridge (BROCA_SKIP_EXERCISM=1) — it measures the separate Liquid-Mamba pathway, not this checkpoint, so it isn't part of the gate. Set BROCA_SKIP_EXERCISM=0 to still capture it for visibility."
fi

# Cross-check the gate actually looked at THIS checkpoint, not just accepted
# the flag — belt-and-suspenders against Phase 0's wiring silently regressing.
recorded_checkpoint="$(python3 -c "import json; print(json.load(open('$candidate_dir/measurement/decoder-ab.json')).get('checkpoint_path') or '')" 2>/dev/null || echo "")"
if [[ "$recorded_checkpoint" != "$candidate_checkpoint" ]]; then
  echo "[broca-cycle] FAIL: decoder-ab.json checkpoint_path ('$recorded_checkpoint') does not match candidate ('$candidate_checkpoint') — measurement gate is not checkpoint-aware" >&2
  exit 1
fi

if [[ "$measure_status" -ne 0 ]]; then
  echo "[broca-cycle] FAIL: candidate did not pass the regression gate (see $candidate_dir/measurement/checkpoint-compare.json)" >&2
  exit 1
fi

promotion_passed="$(python3 -c "import json; print(json.load(open('$candidate_dir/measurement/checkpoint-compare.json'))['promotion']['passed'])" 2>/dev/null || echo "False")"
if [[ "$promotion_passed" != "True" ]]; then
  echo "[broca-cycle] FAIL: promotion.passed is not true in checkpoint-compare.json" >&2
  exit 1
fi

# ── Step 8: write the promotion-ready marker (still touches nothing under
# production — a human runs broca_promote_candidate.sh to act on this) ─────
python3 - "$candidate_dir" "$candidate_checkpoint" <<'PY'
import json, sys
from datetime import datetime, timezone

candidate_dir, candidate_checkpoint = sys.argv[1:]

with open(f"{candidate_dir}/curriculum-derived-manifest.json") as f:
    contributing = [e["objective_id"] for e in json.load(f)]

with open(f"{candidate_dir}/measurement/checkpoint-compare.json") as f:
    compare = json.load(f)

promotion_ready = {
    "candidate_checkpoint": candidate_checkpoint,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "contributing_objective_ids": contributing,
    "checkpoint_compare": compare,
}
with open(f"{candidate_dir}/PROMOTION_READY.json", "w") as f:
    json.dump(promotion_ready, f, indent=2)
PY

echo "[broca-cycle] PASSED — candidate ready for human review: $candidate_dir"
echo "[broca-cycle] to promote: scripts/broca_promote_candidate.sh $candidate_dir --i-understand-this-requires-restart"
