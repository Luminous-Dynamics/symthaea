#!/usr/bin/env bash
# broca_distill_cycle.sh — close the distillation loop (Improvement Plan Tier 2.2).
#
# The flywheel writes and forgets: DistillationCollector and the code
# orchestrator append TrainingPair JSONL that no training job ever reads.
# This script is the missing closure: merge accumulated distillation pairs
# (+ optionally a fresh teacher batch from broca-collect) with the base
# corpus, dedup, resume-train, and eval before/after.
#
# Promotion stays MANUAL by design (self-mod safety): output goes to a new
# timestamped checkpoint + manifest; this script never touches
# broca-checkpoint-latest.bin. Inspect the eval delta, then promote by hand.
#
# Usage:
#   ./scripts/broca_distill_cycle.sh                       # flywheel + base corpus
#   EXTRA=/path/pairs.jsonl ./scripts/broca_distill_cycle.sh   # + teacher batch
#   EPOCHS=3 NO_BASE=1 ./scripts/broca_distill_cycle.sh    # quick: new data only
#
# Env knobs:
#   FLYWHEEL  distillation file   (default ~/.local/share/symthaea/distillation.jsonl)
#   EXTRA     extra pair file(s), colon-separated (e.g. a broca-collect batch)
#   NO_BASE   set to 1 to skip the base corpus (fast incremental runs)
#   RESUME    checkpoint to resume (default data/models/broca-checkpoint-latest.bin)
#   EPOCHS    training epochs      (default 5)
#   LR        learning rate        (default 0.0005 — resume-friendly, half of cold-start)
#   EVAL      eval set             (default crates/domains/symthaea-broca/data/eval-nsm-v1.jsonl)

set -euo pipefail
cd "$(dirname "$0")/.."   # symthaea/

FLYWHEEL="${FLYWHEEL:-$HOME/.local/share/symthaea/distillation.jsonl}"
EXTRA="${EXTRA:-}"
NO_BASE="${NO_BASE:-0}"
BASE="crates/domains/symthaea-broca/data/train-combined-v8.jsonl"
RESUME="${RESUME:-data/models/broca-checkpoint-latest.bin}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-0.0005}"
EVAL="${EVAL:-crates/domains/symthaea-broca/data/eval-nsm-v1.jsonl}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="data/models"
OUT="${OUT_DIR}/broca-distilled-${STAMP}.bin"
MANIFEST="${OUT_DIR}/broca-distilled-${STAMP}.manifest.json"
MERGED="$(mktemp /tmp/broca-distill-merged-XXXX.jsonl)"
trap 'rm -f "$MERGED"' EXIT

# NixOS CUDA runtime linkage (same fix as train_broca_gpu.sh).
export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"

# Locate binaries: honor CARGO_TARGET_DIR (session isolation, monorepo Rule 5) —
# train_broca_gpu.sh's hardcoded ./target/release breaks under it.
TARGET="${CARGO_TARGET_DIR:-target}"
for bin in broca-train broca-eval; do
    if ! test -x "$TARGET/release/$bin"; then
        echo "Building $bin (release, gpu features)..."
        cargo build --release -p symthaea-broca --bin "$bin" --features "simd,parallel,gpu"
    fi
done
TRAIN="$TARGET/release/broca-train"
EVAL_BIN="$TARGET/release/broca-eval"

# ---- Merge + dedup by target_text ----
sources=()
[[ -f "$FLYWHEEL" ]] && sources+=("$FLYWHEEL") || echo "note: flywheel file absent ($FLYWHEEL) — continuing without it"
if [[ -n "$EXTRA" ]]; then
    IFS=':' read -ra extra_files <<< "$EXTRA"
    for f in "${extra_files[@]}"; do
        [[ -f "$f" ]] || { echo "EXTRA file not found: $f" >&2; exit 1; }
        sources+=("$f")
    done
fi
[[ "$NO_BASE" != "1" ]] && sources+=("$BASE")
if [[ ${#sources[@]} -eq 0 ]]; then
    echo "No input sources at all (flywheel absent, no EXTRA, NO_BASE=1) — nothing to train on." >&2
    exit 1
fi

echo "Merging ${#sources[@]} source(s): ${sources[*]}"
python3 - "$MERGED" "${sources[@]}" <<'PY'
import hashlib, json, sys
out_path, *paths = sys.argv[1:]
seen, kept, dropped, bad = set(), 0, 0, 0
with open(out_path, "w") as out:
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = hashlib.sha256(obj.get("target_text", line).encode()).hexdigest()
            except json.JSONDecodeError:
                bad += 1
                continue
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            out.write(line + "\n")
            kept += 1
print(f"merged: kept={kept} deduped={dropped} malformed={bad}")
PY
PAIRS=$(wc -l < "$MERGED")

# ---- Eval BEFORE (baseline checkpoint) ----
echo; echo "=== Eval BEFORE (baseline: $RESUME) ==="
BEFORE_JSON="${OUT_DIR}/broca-distilled-${STAMP}.eval-before.json"
"$EVAL_BIN" --checkpoint "$RESUME" --eval "$EVAL" --json-out "$BEFORE_JSON" || echo "baseline eval failed (continuing)"

# ---- Train ----
echo; echo "=== Training: $PAIRS pairs, $EPOCHS epochs, resume=$RESUME ==="
RUST_LOG="${RUST_LOG:-info}" "$TRAIN" \
    --data "$MERGED" \
    --resume "$RESUME" \
    --eval "$EVAL" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --output "$OUT"

# ---- Eval AFTER ----
echo; echo "=== Eval AFTER (new: $OUT) ==="
AFTER_JSON="${OUT_DIR}/broca-distilled-${STAMP}.eval-after.json"
"$EVAL_BIN" --checkpoint "$OUT" --eval "$EVAL" --json-out "$AFTER_JSON" || echo "post eval failed"

# ---- Provenance manifest ----
python3 - "$MANIFEST" <<PY
import json, hashlib, os
data = {
  "created": "$STAMP",
  "sources": "${sources[*]}".split(),
  "merged_pairs": $PAIRS,
  "merged_sha256": hashlib.sha256(open("$MERGED","rb").read()).hexdigest(),
  "resume_from": "$RESUME",
  "epochs": $EPOCHS,
  "lr": $LR,
  "eval_set": "$EVAL",
  "output": "$OUT",
  "eval_before": "$BEFORE_JSON" if os.path.exists("$BEFORE_JSON") else None,
  "eval_after": "$AFTER_JSON" if os.path.exists("$AFTER_JSON") else None,
  "promoted": False,
}
json.dump(data, open("$MANIFEST","w"), indent=2)
print("manifest:", "$MANIFEST")
PY

echo
echo "=== Before/after comparison ==="
python3 - <<PY
import json, os
def load(p):
    try: return json.load(open(p))
    except Exception: return None
def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix[:-1]] = d
    return out
b, a = load("$BEFORE_JSON"), load("$AFTER_JSON")
if not (b and a):
    print("comparison unavailable (an eval failed)"); raise SystemExit
fb, fa = flatten(b), flatten(a)
for k in sorted(set(fb) & set(fa)):
    print(f"  {k:<40} before={fb[k]:<12.6g} after={fa[k]:<12.6g} delta={fa[k]-fb[k]:+.6g}")
PY

echo
echo "New checkpoint: $OUT (NOT promoted — inspect deltas, then promote manually)"
