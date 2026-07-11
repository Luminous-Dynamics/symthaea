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
# Defaults lowered 2026-07-05 after the first real cycle (177 real objectives,
# candidate 20260705T094221Z): 3 epochs @ lr=0.0002 drove val_loss from a
# healthy start to 17.87 (train loss looked fine at 2.54) -- the gate's
# automated metrics didn't catch it (known blind spot, see
# fuzzy-beaming-brook.md), but manual inspection of topic-coverage-report.json
# showed the candidate had collapsed to repeating the same degenerate string
# regardless of input topic. Training data itself was verified coherent, so
# this is a hyperparameter problem, not a data-quality one: --resume starts
# from an already-converged model, and the original epochs/lr (tuned for
# training from scratch) are too aggressive for continued fine-tuning on a
# comparatively small new slice. NOT re-validated with a fresh run this
# session (that's itself a multi-hour job) -- treat as a first evidence-based
# adjustment, not a proven-correct value.
CYCLE_EPOCHS="${BROCA_CYCLE_EPOCHS:-1}"
CYCLE_LR="${BROCA_CYCLE_LR:-0.00003}"

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
mkdir -p "$CURRICULUM_DERIVED_DIR/holdout"
shopt -s nullglob
holdout_files_before=("$CURRICULUM_DERIVED_DIR"/holdout/*.holdout.jsonl)
shopt -u nullglob

"${sync_cmd[@]}"

# holdout/ is a separate subdirectory from the training-pair jsonl files
# specifically so this glob (used for the training merge below) can never
# accidentally include held-out text — see broca_curriculum_sync.rs's note.
shopt -s nullglob
derived_jsonl_files=("$CURRICULUM_DERIVED_DIR"/*.jsonl)
derived_manifest_files=("$CURRICULUM_DERIVED_DIR"/*.manifest.json)
holdout_files_after=("$CURRICULUM_DERIVED_DIR"/holdout/*.holdout.jsonl)
shopt -u nullglob

# The one holdout file broca-curriculum-sync just wrote this cycle (if any
# new objectives produced enough accepted text to spare a holdout) — used
# later to check whether THIS cycle's new material actually landed, not
# full history.
new_holdout_file=""
for f in "${holdout_files_after[@]}"; do
  found=0
  for old in "${holdout_files_before[@]}"; do
    [[ "$f" == "$old" ]] && found=1 && break
  done
  if [[ "$found" -eq 0 ]]; then
    new_holdout_file="$f"
    break
  fi
done

# Fallback for a retried/resumed cycle: if this invocation's sync step was a
# no-op (e.g. re-running after training was killed mid-epoch, with all
# objectives already marked consumed from the interrupted attempt), the
# before/after diff above finds nothing even though a real, never-measured
# holdout batch from the ORIGINAL synthesis sits on disk and is already part
# of this candidate's merged training data. Silently skipping the
# improvement/collapse gates in that case is exactly what let candidate
# 20260707T155814Z report "PASSED" while its real generated text had
# near-zero coherence (avg_generated_coherence=0.0076) — the gates never ran
# at all, not that they ran and missed it. Fall back to the most recently
# written holdout file on disk rather than skip gating entirely.
if [[ -z "$new_holdout_file" && ${#holdout_files_after[@]} -gt 0 ]]; then
  new_holdout_file="$(ls -t "${holdout_files_after[@]}" | head -1)"
  echo "[broca-cycle] NOTE: no holdout file was newly written this invocation (sync was a no-op) — falling back to the most recent one on disk for gating: $new_holdout_file"
fi

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
# BROCA_TRAIN_BACKEND: cpu (default), gpu, or auto — mirrors the backend
# selection in the existing scripts/broca_train_and_gate.sh. "gpu" is a
# separate Cargo feature from "mamba"/"mamba-cpu" (gpu = candle-core/cuda +
# candle-nn, nothing more) — it does NOT pull in the liquid_mamba module, so
# it doesn't hit the broken symthaea-broca-tools migration (dangling
# sovereignty_bridge/wasm_architect/etc. references) that "simd" alone
# already sidesteps.
#
# GPU requires the nix develop .#broca-gpu shell specifically, not the
# default shell: cudaforge (candle-kernels' build-dependency) resolves nvcc
# via `which nvcc` before checking CUDA_HOME, and on NixOS `nvcc` is on PATH
# via the merged-profile symlink /run/current-system/sw/bin/nvcc — deriving
# cuda_root as nvcc_path.parent().parent() from THAT path lands on
# /run/current-system/sw, which has no CUDA headers, so plain `nix develop`
# fails compiling candle-kernels' .cu files with "cuda_runtime.h: No such
# file or directory". devShells."broca-gpu" works around this by prepending
# cudaPackages.cuda_nvcc's real store path onto PATH ahead of the system
# profile. This is a pre-existing bug in the vendored cudaforge crate, not
# something to patch here — verified 2026-07-03 against the current CUDA
# 12.9 / driver 595.84 setup.
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
    echo "[broca-cycle] BROCA_TRAIN_BACKEND must be cpu, gpu, or auto" >&2
    exit 2
    ;;
esac

# Only wrap in nix develop for the gpu path (per CLAUDE.md: direct cargo for
# everything else — no reason to pay nix-shell entry cost on the CPU path
# that already works directly). Skips the wrap if already inside a nix shell.
run_train() {
  if [[ "$TRAIN_BACKEND" == "gpu" && -z "${IN_NIX_SHELL:-}" ]]; then
    nix develop "$train_nix_attr" --command "$@"
  else
    "$@"
  fi
}

echo "[broca-cycle] training candidate checkpoint (epochs=$CYCLE_EPOCHS lr=$CYCLE_LR, backend=$TRAIN_BACKEND)..."
run_train cargo run "${cargo_locked_args[@]}" -p symthaea-broca "${train_feature_args[@]}" --bin broca-train -- \
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
# BROCA_DECODER_FEATURES=simd (not the auto-detected mamba-cpu default):
# this bridge's gate only ever requests --decoder direct,structured, never
# mamba, so it doesn't need the Liquid-Mamba pathway at all — and mamba-cpu
# currently fails to compile regardless (dangling symthaea-broca-tools
# references from an incomplete crate-split migration in liquid_mamba.rs,
# unrelated to this bridge). This is the deliberately-correct feature set
# for this gate, not just a workaround, so it stays "simd" even after GPU
# comes back — see broca_curriculum_sync's equivalent note.
export BROCA_DECODER_FEATURES="${BROCA_DECODER_FEATURES:-simd}"

if [[ -z "$baseline_dir" || ! -d "$baseline_dir" ]]; then
  baseline_name="production-$(date -u +%Y%m%dT%H%M%SZ)"
  echo "[broca-cycle] no valid production baseline recorded; capturing one now ($baseline_name)..."
  BROCA_CHECKPOINT_PATH="$PRODUCTION_CHECKPOINT" BROCA_SKIP_EXERCISM="${BROCA_SKIP_EXERCISM:-1}" scripts/broca_capture_baseline.sh "$baseline_name"
  baseline_dir="$BASELINE_ROOT/$baseline_name"
  echo "$baseline_dir" > "$BASELINE_POINTER"
fi
echo "[broca-cycle] using production baseline: $baseline_dir"

# ── Step 6.5: topic-coverage evidence, generated BEFORE the gate so the
# presence-of-improvement check (Step 7) has both sides to compare. Uses
# only this cycle's new holdout batch (not full history) since the question
# is "did THIS cycle's learning work," not the whole corpus's. Two runs
# against the SAME holdout file: production checkpoint (the "did production
# already know this" baseline) and the candidate (the "did fine-tuning
# actually land it" side) — this is the evidence the regression gate
# structurally cannot produce, since regression bounds only look at drift
# from the checkpoint's OWN prior behavior, not at performance on material
# neither checkpoint was ever trained on identically. See
# broca_checkpoint_compare.rs's module docs and
# DISCOVERY_AND_SELF_IMPROVEMENT_PLAN_2026-07-06.md Tier 1.1.
baseline_topic_coverage_report=""
candidate_topic_coverage_report=""
if [[ -n "$new_holdout_file" ]]; then
  echo "[broca-cycle] checking topic coverage on this cycle's new material (production baseline)..."
  baseline_topic_coverage_report="$candidate_dir/topic-coverage-baseline-report.json"
  cargo run "${cargo_locked_args[@]}" -p symthaea-broca --features simd --bin broca-topic-coverage -- \
    --checkpoint "$PRODUCTION_CHECKPOINT" \
    --holdout "$new_holdout_file" \
    --json-out "$baseline_topic_coverage_report"
  echo "[broca-cycle] production baseline topic coverage report: $baseline_topic_coverage_report"

  echo "[broca-cycle] checking topic coverage on this cycle's new material (candidate)..."
  candidate_topic_coverage_report="$candidate_dir/topic-coverage-report.json"
  cargo run "${cargo_locked_args[@]}" -p symthaea-broca --features simd --bin broca-topic-coverage -- \
    --checkpoint "$candidate_checkpoint" \
    --holdout "$new_holdout_file" \
    --json-out "$candidate_topic_coverage_report"
  echo "[broca-cycle] candidate topic coverage report: $candidate_topic_coverage_report"
else
  echo "[broca-cycle] NOTE: no holdout batch from this cycle (every objective produced fewer than 2 accepted texts) — skipping topic-coverage reports and the improvement gate for this cycle"
fi

# ── Step 7: measure the candidate against that baseline, fail-closed. This
# is BOTH gates at once: checkpoint-compare computes regression_passed
# (absence-of-regression, the historical behavior) AND, when the topic
# coverage reports above exist, improvement_passed (presence-of-improvement
# on held-out material). promotion.passed = regression_passed &&
# improvement_passed — see broca_checkpoint_compare.rs. ─────────────────────
echo "[broca-cycle] measuring candidate checkpoint (regression + improvement gates)..."
set +e
BROCA_CHECKPOINT_PATH="$candidate_checkpoint" \
BROCA_BASELINE_ARTIFACT_DIR="$baseline_dir" \
BROCA_FAIL_ON_REGRESSION=1 \
BROCA_SKIP_EXERCISM="${BROCA_SKIP_EXERCISM:-1}" \
BROCA_BASELINE_TOPIC_COVERAGE="$baseline_topic_coverage_report" \
BROCA_CANDIDATE_TOPIC_COVERAGE="$candidate_topic_coverage_report" \
BROCA_REQUIRE_IMPROVEMENT_EVALUATED="${new_holdout_file:+1}" \
BROCA_MIN_KEYWORD_OVERLAP_IMPROVEMENT="${BROCA_MIN_KEYWORD_OVERLAP_IMPROVEMENT:-0.05}" \
BROCA_MIN_COHERENCE_IMPROVEMENT="${BROCA_MIN_COHERENCE_IMPROVEMENT:-0.02}" \
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
  echo "[broca-cycle] FAIL: candidate did not pass the regression and/or improvement gate (see $candidate_dir/measurement/checkpoint-compare.json)" >&2
  exit 1
fi

promotion_passed="$(python3 -c "import json; print(json.load(open('$candidate_dir/measurement/checkpoint-compare.json'))['promotion']['passed'])" 2>/dev/null || echo "False")"
if [[ "$promotion_passed" != "True" ]]; then
  regression_passed="$(python3 -c "import json; print(json.load(open('$candidate_dir/measurement/checkpoint-compare.json'))['promotion'].get('regression_passed'))" 2>/dev/null || echo "unknown")"
  improvement_passed="$(python3 -c "import json; print(json.load(open('$candidate_dir/measurement/checkpoint-compare.json'))['promotion'].get('improvement_passed'))" 2>/dev/null || echo "unknown")"
  echo "[broca-cycle] FAIL: promotion.passed is not true in checkpoint-compare.json (regression_passed=$regression_passed, improvement_passed=$improvement_passed)" >&2
  exit 1
fi

# ── Step 7.5: collapse/repetition check on this cycle's new material. Not
# diagnostic-only, this one actually blocks promotion. Added 2026-07-05
# after candidate 20260705T094221Z passed checkpoint-compare's regression
# gate (decoder-ab metrics: the structured path is checkpoint-independent
# by design, and the direct path's coherence/hallucination metrics don't
# detect degenerate output) while its topic-coverage-report.json showed it
# had actually collapsed — dozens of unrelated objectives all generating
# the exact same garbage string. checkpoint-compare.json's regression
# metrics have no signal for this failure mode at all; this is the cheapest
# real check that would have caught it: if a large fraction of distinct
# objectives produce byte-identical generated text, the model isn't
# distinguishing inputs anymore. Reads the candidate topic-coverage report
# generated in Step 6.5 above — no need to regenerate it here.
if [[ -n "$new_holdout_file" ]]; then
  collapse_check_status=0
  python3 - "$candidate_topic_coverage_report" <<'PY' || collapse_check_status=$?
import json, sys
from collections import Counter

path = sys.argv[1]
with open(path) as f:
    report = json.load(f)

cases = report.get("cases", [])
if not cases:
    print("[broca-cycle] collapse check: no cases in topic-coverage report, skipping")
    sys.exit(0)

texts = [c.get("generated_text", "") for c in cases]
counts = Counter(texts)
most_common_text, most_common_count = counts.most_common(1)[0]
duplicate_ratio = most_common_count / len(texts)

print(
    f"[broca-cycle] collapse check: {len(texts)} cases, "
    f"most-repeated generated_text appears {most_common_count} times "
    f"(ratio {duplicate_ratio:.2f})"
)

MAX_DUPLICATE_RATIO = 0.25
if duplicate_ratio > MAX_DUPLICATE_RATIO and most_common_text.strip():
    print(
        f"[broca-cycle] FAIL: {most_common_count}/{len(texts)} objectives "
        f"produced byte-identical generated text (ratio {duplicate_ratio:.2f} "
        f"> {MAX_DUPLICATE_RATIO}) -- candidate has likely collapsed to "
        "repeating one degenerate output regardless of input. Example: "
        f"{most_common_text[:200]!r}",
        file=sys.stderr,
    )
    sys.exit(1)
PY
  if [[ "$collapse_check_status" -ne 0 ]]; then
    echo "[broca-cycle] FAIL: collapse/repetition check failed (see above) — refusing to write PROMOTION_READY.json" >&2
    exit 1
  fi
else
  echo "[broca-cycle] NOTE: no holdout batch from this cycle (every objective produced fewer than 2 accepted texts) — skipping topic-coverage report"
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
    "candidate_checkpoint": candidate_checkpoint,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "contributing_objective_ids": contributing,
    "checkpoint_compare": compare,
    # Convenience top-level verdicts (also present nested inside
    # checkpoint_compare.promotion) so a human/script scanning
    # PROMOTION_READY.json doesn't have to know the nested schema to see
    # whether this candidate cleared BOTH the absence-of-regression gate
    # and the presence-of-improvement gate (Tier 1.1) distinctly.
    "regression_passed": promotion.get("regression_passed"),
    "improvement_passed": promotion.get("improvement_passed"),
    "improvement_evaluated": promotion.get("improvement_evaluated"),
    "topic_coverage": topic_coverage,
    "topic_coverage_baseline": topic_coverage_baseline,
}
with open(f"{candidate_dir}/PROMOTION_READY.json", "w") as f:
    json.dump(promotion_ready, f, indent=2)
PY

echo "[broca-cycle] PASSED — candidate ready for human review: $candidate_dir"
if [[ -n "$new_holdout_file" ]]; then
  echo "[broca-cycle] read $candidate_dir/topic-coverage-report.json before promoting — it's evidence of what the new material actually looks like generated, not just regression-absence numbers"
fi
echo "[broca-cycle] to promote: scripts/broca_promote_candidate.sh $candidate_dir --i-understand-this-requires-restart"
