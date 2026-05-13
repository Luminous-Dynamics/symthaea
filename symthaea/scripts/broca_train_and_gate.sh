#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Tier 0 Broca automation: train a checkpoint, run canonical eval, and fail
# the process if configured quality gates do not pass.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RUSTC_WRAPPER="${BROCA_GATE_RUSTC_WRAPPER:-}"
export SCCACHE_DISABLE="${SCCACHE_DISABLE:-1}"

OUT_DIR="${BROCA_GATE_OUT_DIR:-/tmp/symthaea-broca-gate}"
TARGET_DIR="${BROCA_GATE_TARGET_DIR:-/tmp/symthaea-broca-gpu-target}"
PAIR_COUNT="${BROCA_GATE_PAIRS:-16}"
EPOCHS="${BROCA_GATE_EPOCHS:-1}"
EVAL_LIMIT="${BROCA_GATE_EVAL_LIMIT:-4}"
MAX_GEN_TOKENS="${BROCA_GATE_MAX_GEN_TOKENS:-4}"
USE_NIX="${BROCA_GATE_USE_NIX:-1}"
BACKEND="${BROCA_GATE_BACKEND:-auto}"
EVAL_LANE="${BROCA_GATE_EVAL_LANE:-fast}"
GPU_COMPUTE_CAP="${BROCA_GATE_CUDA_COMPUTE_CAP:-75}"

FULL_DATA="$OUT_DIR/curriculum-full.jsonl"
TRAIN_DATA="$OUT_DIR/train-gate.jsonl"
CHECKPOINT="$OUT_DIR/broca-gated.bin"
REPORT="${BROCA_GATE_REPORT:-$OUT_DIR/quality-$EVAL_LANE.json}"
CANONICAL="${BROCA_GATE_CANONICAL:-crates/symthaea-broca/tests/fixtures/eval-canonical-v1.jsonl}"

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
            echo "BROCA_GATE_BACKEND must be auto, gpu, or cpu" >&2
            exit 2
            ;;
    esac
}

select_backend

case "$EVAL_LANE" in
    fast | full) ;;
    *)
        echo "BROCA_GATE_EVAL_LANE must be fast or full" >&2
        exit 2
        ;;
esac

if [[ "${BROCA_GATE_CODE_SHEAF_EVAL:-0}" == "1" ]]; then
    CARGO_FEATURE_ARGS+=(--features code-sheaf-eval)
fi

run() {
    if [[ "$USE_NIX" == "1" && -z "${IN_NIX_SHELL:-}" ]]; then
        nix develop "$NIX_SHELL_ATTR" -c "$@"
    else
        "$@"
    fi
}

threshold_args=()
if [[ "${BROCA_GATE_REPORT_ONLY:-0}" == "1" ]]; then
    threshold_args+=(--report-only)
fi
if [[ "$EVAL_LANE" == "full" && "${BROCA_GATE_TEACHER_FORCED_ONLY:-0}" == "1" ]]; then
    echo "BROCA_GATE_TEACHER_FORCED_ONLY=1 conflicts with BROCA_GATE_EVAL_LANE=full" >&2
    exit 2
fi
if [[ "$EVAL_LANE" == "fast" ]]; then
    threshold_args+=(--teacher-forced-only)
fi

add_threshold() {
    local env_name="$1"
    local flag="$2"
    local value="${!env_name:-}"
    if [[ -n "$value" ]]; then
        threshold_args+=("$flag" "$value")
    fi
}

add_threshold BROCA_GATE_MAX_GATED_PERPLEXITY --max-gated-perplexity
add_threshold BROCA_GATE_MIN_GATED_COHERENCE --min-gated-coherence
add_threshold BROCA_GATE_MIN_GATED_ENGLISH_RATIO --min-gated-english-ratio
add_threshold BROCA_GATE_MAX_GATED_HALLUCINATION_RATE --max-gated-hallucination-rate
add_threshold BROCA_GATE_MAX_COHERENCE_REGRESSION --max-coherence-regression
add_threshold BROCA_GATE_MAX_TARGET_OVERLAP_REGRESSION --max-target-overlap-regression
add_threshold BROCA_GATE_MIN_MORAL_REFUSAL_RATE --min-moral-refusal-rate
add_threshold BROCA_GATE_MAX_CODE_SHEAF_INCOHERENCE_RATE --max-code-sheaf-incoherence-rate
add_threshold BROCA_GATE_MIN_STRUCTURED_OUTPUT_VALIDITY_RATE --min-structured-output-validity-rate
add_threshold BROCA_GATE_MIN_CODE_SHEAF_FUNCTION_COHERENCE_RATE --min-code-sheaf-function-coherence-rate

echo "[broca-gate] output: $OUT_DIR"
echo "[broca-gate] backend: $BACKEND"
echo "[broca-gate] eval lane: $EVAL_LANE"
echo "[broca-gate] generating curriculum data"
run cargo run -p symthaea-broca "${CARGO_FEATURE_ARGS[@]}" --bin broca-train -- --curriculum "$FULL_DATA"

head -n "$PAIR_COUNT" "$FULL_DATA" > "$TRAIN_DATA"
echo "[broca-gate] training $PAIR_COUNT pairs for $EPOCHS epoch(s)"
run cargo run -p symthaea-broca "${CARGO_FEATURE_ARGS[@]}" --bin broca-train -- \
    --data "$TRAIN_DATA" \
    --epochs "$EPOCHS" \
    --bptt-window 8 \
    --negative-samples 64 \
    --lr 0.001 \
    --network-lr-scale 0.2 \
    --diagnostics \
    --no-save-adam \
    --output "$CHECKPOINT" \
    --samples 0

echo "[broca-gate] running canonical quality gate"
eval_features=()
if [[ "$BACKEND" == "gpu" ]]; then
    eval_features+=(gpu)
fi
if [[ "${BROCA_GATE_CODE_SHEAF_EVAL:-0}" == "1" ]]; then
    eval_features+=(code-sheaf-eval)
fi
eval_feature_set="$(IFS=,; echo "${eval_features[*]}")"

if run env \
    BROCA_EVAL_BACKEND="$BACKEND" \
    BROCA_EVAL_LANE="$EVAL_LANE" \
    BROCA_EVAL_FEATURES="$eval_feature_set" \
    BROCA_TRAIN_PAIR_COUNT="$PAIR_COUNT" \
    BROCA_TRAIN_EPOCHS="$EPOCHS" \
    cargo run -p symthaea-broca "${CARGO_FEATURE_ARGS[@]}" --bin broca-eval -- \
    --checkpoint "$CHECKPOINT" \
    --canonical-eval "$CANONICAL" \
    --eval-limit "$EVAL_LIMIT" \
    --max-gen-tokens "$MAX_GEN_TOKENS" \
    --json-out "$REPORT" \
    "${threshold_args[@]}"; then
    echo "[broca-gate] PASS"
else
    status=$?
    echo "[broca-gate] FAIL (exit $status)"
    echo "  report: $REPORT"
    exit "$status"
fi

echo "  checkpoint: $CHECKPOINT"
echo "  report:     $REPORT"
