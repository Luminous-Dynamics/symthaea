#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Fast end-to-end Broca training smoke:
#   curriculum data -> short train -> checkpoint reload.
#
# Optional slower quality mode:
#   BROCA_SMOKE_CANONICAL=1 scripts/broca_train_smoke.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RUSTC_WRAPPER="${BROCA_SMOKE_RUSTC_WRAPPER:-}"
export SCCACHE_DISABLE="${SCCACHE_DISABLE:-1}"

OUT_DIR="${BROCA_SMOKE_OUT_DIR:-/tmp/symthaea-broca-smoke}"
TARGET_DIR="${BROCA_SMOKE_TARGET_DIR:-/tmp/symthaea-broca-gpu-target}"
PAIR_COUNT="${BROCA_SMOKE_PAIRS:-8}"
EVAL_LIMIT="${BROCA_SMOKE_EVAL_LIMIT:-8}"
MAX_GEN_TOKENS="${BROCA_SMOKE_MAX_GEN_TOKENS:-8}"
EPOCHS="${BROCA_SMOKE_EPOCHS:-1}"
NEGATIVE_SAMPLES="${BROCA_SMOKE_NEGATIVE_SAMPLES:-64}"
USE_NIX="${BROCA_SMOKE_USE_NIX:-1}"
RUN_CANONICAL="${BROCA_SMOKE_CANONICAL:-0}"
RUN_BASELINE="${BROCA_SMOKE_BASELINE:-0}"
RUN_SELECTION="${BROCA_SMOKE_SELECT:-0}"
BACKEND="${BROCA_SMOKE_BACKEND:-auto}"
EVAL_LANE="${BROCA_SMOKE_EVAL_LANE:-fast}"
GPU_COMPUTE_CAP="${BROCA_SMOKE_CUDA_COMPUTE_CAP:-75}"

FULL_DATA="$OUT_DIR/curriculum-full.jsonl"
TRAIN_DATA="$OUT_DIR/train-smoke.jsonl"
EXTERNAL_TRAIN_DATA="${BROCA_SMOKE_TRAIN_DATA:-}"
BASELINE_CKPT="$OUT_DIR/broca-baseline.bin"
TRAINED_CKPT="$OUT_DIR/broca-trained.bin"
BASELINE_JSON="$OUT_DIR/baseline-quality.json"
TRAINED_JSON="$OUT_DIR/trained-quality.json"
SELECTION_JSON="$OUT_DIR/selected-checkpoint.json"
CANONICAL="crates/symthaea-broca/tests/fixtures/eval-canonical-v1.jsonl"

mkdir -p "$OUT_DIR"

select_backend() {
    case "$BACKEND" in
        cpu)
            CARGO_FEATURE_ARGS=()
            NIX_SHELL_ATTR="."
            ;;
        gpu)
            CARGO_FEATURE_ARGS=(--features gpu)
            NIX_SHELL_ATTR=".#broca-gpu"
            export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$TARGET_DIR}"
            export CUDA_COMPUTE_CAP="$GPU_COMPUTE_CAP"
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"
            ;;
        auto)
            if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
                CARGO_FEATURE_ARGS=(--features gpu)
                NIX_SHELL_ATTR=".#broca-gpu"
                export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$TARGET_DIR}"
                export CUDA_COMPUTE_CAP="$GPU_COMPUTE_CAP"
                export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"
                BACKEND="gpu"
            else
                CARGO_FEATURE_ARGS=()
                NIX_SHELL_ATTR="."
                BACKEND="cpu"
            fi
            ;;
        *)
            echo "BROCA_SMOKE_BACKEND must be auto, gpu, or cpu" >&2
            exit 2
            ;;
    esac
}

select_backend

if [[ "${BROCA_SMOKE_CODE_SHEAF_EVAL:-0}" == "1" ]]; then
    CARGO_FEATURE_ARGS+=(--features code-sheaf-eval)
fi

case "$EVAL_LANE" in
    fast | full) ;;
    *)
        echo "BROCA_SMOKE_EVAL_LANE must be fast or full" >&2
        exit 2
        ;;
esac

run() {
    if [[ "$USE_NIX" == "1" && -z "${IN_NIX_SHELL:-}" ]]; then
        nix develop "$NIX_SHELL_ATTR" -c "$@"
    else
        "$@"
    fi
}

echo "[broca-smoke] output: $OUT_DIR"
echo "[broca-smoke] backend: $BACKEND"
echo "[broca-smoke] eval lane: $EVAL_LANE"
if [[ -n "$EXTERNAL_TRAIN_DATA" ]]; then
    TRAIN_DATA="$EXTERNAL_TRAIN_DATA"
    PAIR_COUNT="$(wc -l < "$TRAIN_DATA" | tr -d ' ')"
    echo "[broca-smoke] using external training data: $TRAIN_DATA"
else
    echo "[broca-smoke] generating curriculum data"
    run cargo run -p symthaea-broca "${CARGO_FEATURE_ARGS[@]}" --bin broca-train -- --curriculum "$FULL_DATA"

    head -n "$PAIR_COUNT" "$FULL_DATA" > "$TRAIN_DATA"
fi
echo "[broca-smoke] using $PAIR_COUNT training pairs"

if [[ "$RUN_BASELINE" == "1" ]]; then
    echo "[broca-smoke] saving baseline checkpoint"
    run cargo run -p symthaea-broca "${CARGO_FEATURE_ARGS[@]}" --bin broca-train -- \
        --data "$TRAIN_DATA" \
        --epochs 0 \
        --output "$BASELINE_CKPT" \
        --no-save-adam \
        --samples 0

    echo "[broca-smoke] checking baseline checkpoint reload"
    run cargo run -p symthaea-broca "${CARGO_FEATURE_ARGS[@]}" --bin broca-eval -- \
        --checkpoint "$BASELINE_CKPT" \
        --samples 0
fi

echo "[broca-smoke] training checkpoint"
run cargo run -p symthaea-broca "${CARGO_FEATURE_ARGS[@]}" --bin broca-train -- \
    --data "$TRAIN_DATA" \
    --epochs "$EPOCHS" \
    --bptt-window 8 \
    --negative-samples "$NEGATIVE_SAMPLES" \
    --lr 0.001 \
    --network-lr-scale 0.2 \
    --diagnostics \
    --no-save-adam \
    --output "$TRAINED_CKPT" \
    --samples 0

echo "[broca-smoke] checking trained checkpoint reload"
run cargo run -p symthaea-broca "${CARGO_FEATURE_ARGS[@]}" --bin broca-eval -- \
    --checkpoint "$TRAINED_CKPT" \
    --samples 0

if [[ "$RUN_CANONICAL" == "1" ]]; then
    canonical_args=()
    if [[ "$EVAL_LANE" == "fast" ]]; then
        canonical_args+=(--teacher-forced-only)
    fi
    eval_features=()
    if [[ "$BACKEND" == "gpu" ]]; then
        eval_features+=(gpu)
    fi
    if [[ "${BROCA_SMOKE_CODE_SHEAF_EVAL:-0}" == "1" ]]; then
        eval_features+=(code-sheaf-eval)
    fi
    eval_feature_set="$(IFS=,; echo "${eval_features[*]}")"
    if [[ -n "${BROCA_SMOKE_MIN_CODE_SHEAF_FUNCTION_COHERENCE_RATE:-}" ]]; then
        canonical_args+=(
            --min-code-sheaf-function-coherence-rate
            "$BROCA_SMOKE_MIN_CODE_SHEAF_FUNCTION_COHERENCE_RATE"
        )
    fi

    if [[ "$RUN_BASELINE" == "1" ]]; then
        echo "[broca-smoke] running canonical quality suite for baseline checkpoint"
        run env \
            BROCA_EVAL_BACKEND="$BACKEND" \
            BROCA_EVAL_LANE="$EVAL_LANE" \
            BROCA_EVAL_FEATURES="$eval_feature_set" \
            BROCA_TRAIN_PAIR_COUNT="$PAIR_COUNT" \
            BROCA_TRAIN_EPOCHS=0 \
            cargo run -p symthaea-broca "${CARGO_FEATURE_ARGS[@]}" --bin broca-eval -- \
            --checkpoint "$BASELINE_CKPT" \
            --canonical-eval "$CANONICAL" \
            --eval-limit "$EVAL_LIMIT" \
            --max-gen-tokens "$MAX_GEN_TOKENS" \
            --json-out "$BASELINE_JSON" \
            "${canonical_args[@]}"
    fi

    echo "[broca-smoke] running canonical quality suite for trained checkpoint"
    run env \
        BROCA_EVAL_BACKEND="$BACKEND" \
        BROCA_EVAL_LANE="$EVAL_LANE" \
        BROCA_EVAL_FEATURES="$eval_feature_set" \
        BROCA_TRAIN_PAIR_COUNT="$PAIR_COUNT" \
        BROCA_TRAIN_EPOCHS="$EPOCHS" \
        cargo run -p symthaea-broca "${CARGO_FEATURE_ARGS[@]}" --bin broca-eval -- \
        --checkpoint "$TRAINED_CKPT" \
        --canonical-eval "$CANONICAL" \
        --eval-limit "$EVAL_LIMIT" \
        --max-gen-tokens "$MAX_GEN_TOKENS" \
        --json-out "$TRAINED_JSON" \
        "${canonical_args[@]}"

    if [[ "$RUN_SELECTION" == "1" ]]; then
        selection_reports=("$TRAINED_JSON")
        if [[ "$RUN_BASELINE" == "1" ]]; then
            selection_reports=("$BASELINE_JSON" "$TRAINED_JSON")
        fi
        selection_args=()
        if [[ -n "${BROCA_SMOKE_CODING_REPORT:-}" ]]; then
            selection_args+=(--coding-report "$BROCA_SMOKE_CODING_REPORT")
        fi
        if [[ -n "${BROCA_SMOKE_REPAIR_MEMORY_AB_SUMMARY:-}" ]]; then
            selection_args+=(--repair-memory-ab-summary "$BROCA_SMOKE_REPAIR_MEMORY_AB_SUMMARY")
        fi
        if [[ "${BROCA_SMOKE_REQUIRE_CODING_GATE:-0}" == "1" ]]; then
            selection_args+=(--require-coding-gate)
        fi
        if [[ "${BROCA_SMOKE_REQUIRE_CODE_SIGNAL:-0}" == "1" ]]; then
            selection_args+=(--require-code-signal)
        fi
        if [[ "${BROCA_SMOKE_REQUIRE_CODING_EVAL_GATE:-0}" == "1" ]]; then
            selection_args+=(--require-coding-eval-gate)
        fi
        if [[ "${BROCA_SMOKE_REQUIRE_REPAIR_MEMORY_GATE:-0}" == "1" ]]; then
            selection_args+=(--require-repair-memory-gate)
        fi
        if [[ "${BROCA_SMOKE_REQUIRE_TRAINED_IMPROVEMENT:-0}" == "1" ]]; then
            selection_args+=(
                --require-trained-improvement
                --baseline-report "$BASELINE_JSON"
                --trained-report "$TRAINED_JSON"
            )
        fi
        selection_args+=(
            --min-coding-score "${BROCA_SMOKE_MIN_CODING_SCORE:-0}"
            --min-quality-pass-rate "${BROCA_SMOKE_MIN_CODING_QUALITY_PASS_RATE:-0}"
            --min-memory-hits "${BROCA_SMOKE_MIN_REPAIR_MEMORY_HITS:-0}"
            --min-memory-success-rate "${BROCA_SMOKE_MIN_REPAIR_MEMORY_SUCCESS_RATE:-0}"
            --max-memory-hurt-tasks "${BROCA_SMOKE_MAX_REPAIR_MEMORY_HURT_TASKS:-0}"
        )
        echo "[broca-smoke] selecting checkpoint with coding quality signals"
        python3 scripts/select_broca_checkpoint_by_quality.py \
            --json \
            "${selection_args[@]}" \
            "${selection_reports[@]}" > "$SELECTION_JSON"
        python3 -m json.tool "$SELECTION_JSON"
    fi
fi

echo "[broca-smoke] complete"
echo "  train data:       $TRAIN_DATA"
echo "  trained ckpt:     $TRAINED_CKPT"
if [[ "$RUN_BASELINE" == "1" ]]; then
    echo "  baseline ckpt:    $BASELINE_CKPT"
fi
if [[ "$RUN_CANONICAL" == "1" ]]; then
    if [[ "$RUN_BASELINE" == "1" ]]; then
        echo "  baseline report:  $BASELINE_JSON"
    fi
    echo "  trained report:   $TRAINED_JSON"
    if [[ "$RUN_SELECTION" == "1" ]]; then
        echo "  selection report: $SELECTION_JSON"
    fi
fi
