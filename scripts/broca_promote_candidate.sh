#!/usr/bin/env bash
# broca_promote_candidate.sh: promote a gated Broca candidate checkpoint to production.
#
# This is the ONLY step in the curriculum-fine-tuning bridge that touches the
# production checkpoint, and it is always a deliberate, human-invoked action —
# never run automatically by broca_curriculum_cycle.sh or any scheduler.
# See /home/tstoltz/.claude/plans/fuzzy-beaming-brook.md for the full design.
#
# Usage:
#   scripts/broca_promote_candidate.sh <candidate-dir> --i-understand-this-requires-restart
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BROCA_DATA_DIR="${BROCA_DATA_DIR:-crates/domains/symthaea-broca/data}"
PRODUCTION_CHECKPOINT="$BROCA_DATA_DIR/models/broca-checkpoint-latest.bin"
BASELINE_ROOT="${BROCA_BASELINE_ROOT:-target/broca-baselines}"
BASELINE_POINTER="$BROCA_DATA_DIR/models/measurements/CURRENT_BASELINE"
PROMOTION_LOG="$BROCA_DATA_DIR/models/promotion-log.jsonl"

usage() {
  echo "Usage: $0 <candidate-dir> --i-understand-this-requires-restart" >&2
  echo "" >&2
  echo "The confirmation flag is mandatory. This script refuses to act without it —" >&2
  echo "a copy-pasted command must not be able to promote a live cognitive subsystem" >&2
  echo "silently. No live Broca consumer (REPL, fused_cognitive_node, etc.) hot-reloads" >&2
  echo "its checkpoint; promoting means whatever process is using Broca must restart" >&2
  echo "to pick up the new weights." >&2
  exit 2
}

candidate_dir="${1:-}"
confirm_flag="${2:-}"

if [[ -z "$candidate_dir" || "$confirm_flag" != "--i-understand-this-requires-restart" ]]; then
  usage
fi

promotion_ready="$candidate_dir/PROMOTION_READY.json"
if [[ ! -f "$promotion_ready" ]]; then
  echo "[broca-promote] REFUSED: no PROMOTION_READY.json in $candidate_dir" >&2
  exit 1
fi

passed="$(python3 -c "import json; print(json.load(open('$promotion_ready'))['checkpoint_compare']['promotion']['passed'])" 2>/dev/null || echo "False")"
if [[ "$passed" != "True" ]]; then
  echo "[broca-promote] REFUSED: $promotion_ready does not record promotion.passed == true" >&2
  exit 1
fi

candidate_checkpoint="$(python3 -c "import json; print(json.load(open('$promotion_ready'))['candidate_checkpoint'])")"
if [[ ! -s "$candidate_checkpoint" ]]; then
  echo "[broca-promote] REFUSED: candidate checkpoint missing or empty at $candidate_checkpoint" >&2
  exit 1
fi

if [[ ! -s "$PRODUCTION_CHECKPOINT" ]]; then
  echo "[broca-promote] REFUSED: no production checkpoint found at $PRODUCTION_CHECKPOINT to back up" >&2
  exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_path="$BROCA_DATA_DIR/models/broca-checkpoint-latest.bin.backup-$timestamp"

echo "[broca-promote] backing up production checkpoint -> $backup_path"
cp "$PRODUCTION_CHECKPOINT" "$backup_path"

echo "[broca-promote] promoting candidate -> $PRODUCTION_CHECKPOINT"
cp "$candidate_checkpoint" "$PRODUCTION_CHECKPOINT"

new_hash="$(sha256sum "$PRODUCTION_CHECKPOINT" | cut -d' ' -f1)"
candidate_hash="$(sha256sum "$candidate_checkpoint" | cut -d' ' -f1)"
if [[ "$new_hash" != "$candidate_hash" ]]; then
  echo "[broca-promote] FATAL: production checkpoint hash after copy does not match candidate; restoring backup" >&2
  cp "$backup_path" "$PRODUCTION_CHECKPOINT"
  exit 1
fi

# Write the audit-trail entry now, before attempting baseline recapture below.
# The checkpoint swap above is the part of "promotion" that actually matters
# and has already succeeded (and was gated) — the log must record it even if
# the best-effort baseline recapture that follows fails for an unrelated
# reason (e.g. a transient build issue). Do not let bookkeeping for the next
# cycle block or hide a promotion that already happened.
mkdir -p "$(dirname "$PROMOTION_LOG")"
python3 - "$PROMOTION_LOG" "$promotion_ready" "$candidate_checkpoint" "$candidate_hash" "$backup_path" "$timestamp" <<'PY'
import json, sys

log_path, promotion_ready_path, candidate_checkpoint, candidate_hash, backup_path, timestamp = sys.argv[1:]

with open(promotion_ready_path) as f:
    promotion_ready = json.load(f)

entry = {
    "promoted_at_utc": timestamp,
    "candidate_checkpoint": candidate_checkpoint,
    "candidate_sha256": candidate_hash,
    "previous_checkpoint_backup": backup_path,
    "contributing_objective_ids": promotion_ready.get("contributing_objective_ids", []),
    "checkpoint_compare": promotion_ready.get("checkpoint_compare"),
}
with open(log_path, "a") as f:
    f.write(json.dumps(entry) + "\n")
PY

echo "[broca-promote] DONE. Production checkpoint updated: $PRODUCTION_CHECKPOINT ($new_hash)"
echo "[broca-promote] Previous checkpoint backed up at: $backup_path"
echo "[broca-promote] Provenance recorded in: $PROMOTION_LOG"
echo "[broca-promote] REMINDER: no live Broca consumer hot-reloads its checkpoint."
echo "[broca-promote]          Restart any running process using --llm-backend broca"
echo "[broca-promote]          (or similar) to pick up the new weights."

# Recapture the production baseline for the *next* cycle's comparison. This
# is best-effort: it must not roll back an already-succeeded, already-logged
# promotion. If it fails, invalidate the stale pointer (rather than leaving
# it pointing at a baseline that no longer matches production) so the next
# broca_curriculum_cycle.sh run self-heals by recapturing a fresh one, and
# exit non-zero so a human notices and can investigate/retry manually.
#
# BROCA_DECODER_FEATURES=simd (not the auto-detected mamba-cpu default): this
# bridge's gate only ever needs --decoder direct,structured, never mamba, and
# mamba-cpu currently fails to compile regardless (dangling
# symthaea-broca-tools references from an incomplete crate-split migration in
# liquid_mamba.rs, unrelated to this bridge) — see broca_curriculum_sync's
# equivalent note.
export BROCA_DECODER_FEATURES="${BROCA_DECODER_FEATURES:-simd}"

baseline_name="production-$timestamp"
echo "[broca-promote] recapturing production baseline ($baseline_name) for the next cycle's comparison..."
mkdir -p "$(dirname "$BASELINE_POINTER")"
if BROCA_CHECKPOINT_PATH="$PRODUCTION_CHECKPOINT" BROCA_SKIP_EXERCISM="${BROCA_SKIP_EXERCISM:-1}" scripts/broca_capture_baseline.sh "$baseline_name"; then
  echo "$BASELINE_ROOT/$baseline_name" > "$BASELINE_POINTER"
  echo "[broca-promote] baseline recaptured: $BASELINE_ROOT/$baseline_name"
else
  rm -f "$BASELINE_POINTER"
  echo "[broca-promote] WARNING: baseline recapture failed. The promotion above already succeeded and is recorded in $PROMOTION_LOG, but no fresh baseline is recorded — CURRENT_BASELINE has been invalidated so the next cycle run will recapture one automatically. You may also run scripts/broca_capture_baseline.sh manually." >&2
  exit 3
fi
