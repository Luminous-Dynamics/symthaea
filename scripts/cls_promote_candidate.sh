#!/usr/bin/env bash
# cls_promote_candidate.sh: promote a gated CLS threshold-phenotype candidate
# to the path the live cognitive loop reads.
#
# Tier 1.2 (DISCOVERY_AND_SELF_IMPROVEMENT_PLAN_2026-07-06.md), mirroring the
# Broca curriculum bridge's promotion shape (see broca_promote_candidate.sh):
# this is the ONLY step in the CLS threshold-evolution pipeline that touches
# the file `ThresholdOverrides::from_env()` reads, and it is always a
# deliberate, human-invoked action — never run automatically by evolve_cls or
# cls_promotion_gate.
#
# Usage:
#   scripts/cls_promote_candidate.sh <candidate-dir> --i-understand-this-is-live
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLS_DATA_DIR="${CLS_DATA_DIR:-crates/domains/symthaea-neuroevolution/data/cls-thresholds}"
ACTIVE_DIR="$CLS_DATA_DIR/active"
ACTIVE_PATH="$ACTIVE_DIR/threshold-overrides-active.json"
PROMOTION_LOG="$CLS_DATA_DIR/promotion-log.jsonl"

usage() {
  echo "Usage: $0 <candidate-dir> --i-understand-this-is-live" >&2
  echo "" >&2
  echo "The confirmation flag is mandatory. This script refuses to act without it —" >&2
  echo "a copy-pasted command must not be able to promote live cognitive-loop" >&2
  echo "thresholds silently. No running CognitiveLoopService hot-reloads this file" >&2
  echo "(ThresholdOverrides::from_env() is read once, at construction time) —" >&2
  echo "promoting means whatever process constructs a CognitiveLoopService must" >&2
  echo "restart (with SYMTHAEA_THRESHOLD_OVERRIDES_PATH pointed at $ACTIVE_PATH)" >&2
  echo "to pick up the promoted thresholds." >&2
  exit 2
}

candidate_dir="${1:-}"
confirm_flag="${2:-}"

if [[ -z "$candidate_dir" || "$confirm_flag" != "--i-understand-this-is-live" ]]; then
  usage
fi

promotion_ready="$candidate_dir/PROMOTION_READY.json"
if [[ ! -f "$promotion_ready" ]]; then
  echo "[cls-promote] REFUSED: no PROMOTION_READY.json in $candidate_dir (run cls_promotion_gate first)" >&2
  exit 1
fi

passed="$(python3 -c "import json; print(json.load(open('$promotion_ready'))['passed'])" 2>/dev/null || echo "False")"
if [[ "$passed" != "True" ]]; then
  echo "[cls-promote] REFUSED: $promotion_ready does not record passed == true" >&2
  exit 1
fi

candidate_phenotype="$(python3 -c "import json; print(json.load(open('$promotion_ready'))['candidate_phenotype_path'])")"
if [[ ! -s "$candidate_phenotype" ]]; then
  echo "[cls-promote] REFUSED: candidate phenotype missing or empty at $candidate_phenotype" >&2
  exit 1
fi

mkdir -p "$ACTIVE_DIR"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_path=""
if [[ -s "$ACTIVE_PATH" ]]; then
  backup_path="$ACTIVE_DIR/threshold-overrides-active.json.backup-$timestamp"
  echo "[cls-promote] backing up previously active thresholds -> $backup_path"
  cp "$ACTIVE_PATH" "$backup_path"
else
  echo "[cls-promote] no previously active thresholds found at $ACTIVE_PATH — nothing to back up (first promotion)"
fi

# The candidate phenotype JSON loads directly as ThresholdOverrides (same 18
# fields, same names/types modulo Option<T> — see
# cls_evolution_harness.rs module docs and
# threshold_overrides.rs::test_threshold_phenotype_json_loads_as_overrides).
# No conversion needed: copy verbatim.
echo "[cls-promote] promoting candidate -> $ACTIVE_PATH"
cp "$candidate_phenotype" "$ACTIVE_PATH"

new_hash="$(sha256sum "$ACTIVE_PATH" | cut -d' ' -f1)"
candidate_hash="$(sha256sum "$candidate_phenotype" | cut -d' ' -f1)"
if [[ "$new_hash" != "$candidate_hash" ]]; then
  echo "[cls-promote] FATAL: active file hash after copy does not match candidate" >&2
  if [[ -n "$backup_path" ]]; then
    echo "[cls-promote] restoring backup" >&2
    cp "$backup_path" "$ACTIVE_PATH"
  else
    rm -f "$ACTIVE_PATH"
  fi
  exit 1
fi

mkdir -p "$(dirname "$PROMOTION_LOG")"
python3 - "$PROMOTION_LOG" "$promotion_ready" "$candidate_phenotype" "$candidate_hash" "$backup_path" "$timestamp" "$ACTIVE_PATH" <<'PY'
import json, sys

log_path, promotion_ready_path, candidate_phenotype, candidate_hash, backup_path, timestamp, active_path = sys.argv[1:]

with open(promotion_ready_path) as f:
    promotion_ready = json.load(f)

entry = {
    "promoted_at_utc": timestamp,
    "candidate_phenotype": candidate_phenotype,
    "candidate_sha256": candidate_hash,
    "active_path": active_path,
    "previous_active_backup": backup_path or None,
    "recorded_fitness": promotion_ready.get("recorded_fitness"),
    "fresh_fitness": promotion_ready.get("fresh_fitness"),
    "tolerance": promotion_ready.get("tolerance"),
}
with open(log_path, "a") as f:
    f.write(json.dumps(entry) + "\n")
PY

echo "[cls-promote] DONE. Active thresholds updated: $ACTIVE_PATH ($new_hash)"
if [[ -n "$backup_path" ]]; then
  echo "[cls-promote] Previous active thresholds backed up at: $backup_path"
fi
echo "[cls-promote] Provenance recorded in: $PROMOTION_LOG"
case "$ACTIVE_PATH" in
  /*) active_path_abs="$ACTIVE_PATH" ;;
  *) active_path_abs="$ROOT/$ACTIVE_PATH" ;;
esac
echo "[cls-promote] REMINDER: no running CognitiveLoopService hot-reloads this file."
echo "[cls-promote]          Restart with SYMTHAEA_THRESHOLD_OVERRIDES_PATH=$active_path_abs"
echo "[cls-promote]          to pick up the promoted thresholds."
