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
USE_NIX="${BROCA_GATE_USE_NIX:-1}"
BACKEND="${BROCA_GATE_BACKEND:-auto}"
EVAL_LANE="${BROCA_GATE_EVAL_LANE:-fast}"
GPU_COMPUTE_CAP="${BROCA_GATE_CUDA_COMPUTE_CAP:-75}"
RECIPE="${BROCA_GATE_RECIPE:-smoke}"

case "$RECIPE" in
    smoke)
        DEFAULT_PAIRS=16
        DEFAULT_EPOCHS=1
        DEFAULT_EVAL_LIMIT=4
        DEFAULT_MAX_GEN_TOKENS=4
        DEFAULT_BPTT_WINDOW=8
        DEFAULT_NEGATIVE_SAMPLES=64
        DEFAULT_LR=0.001
        DEFAULT_NETWORK_LR_SCALE=0.2
        DEFAULT_NETWORK_LAYERS=3
        DEFAULT_NEURONS_PER_LAYER=8
        DEFAULT_COHERENCE_ALIGNMENT=0.0
        DEFAULT_ALIGNMENT_START=0.0
        DEFAULT_CONTRASTIVE=0.0
        DEFAULT_CONTRASTIVE_MARGIN=0.0
        DEFAULT_SCHEDULED_SAMPLING=0.0
        DEFAULT_LABEL_SMOOTHING=0.0
        DEFAULT_THOUGHT_LOGIT_AUX=0.0
        DEFAULT_THOUGHT_LOGIT_RESIDUAL=0.0
        DEFAULT_MERGE_BIAS=1.5
        DEFAULT_PAIR_SELECTION=head
        ;;
    baseline-v1-small)
        DEFAULT_PAIRS=128
        DEFAULT_EPOCHS=2
        DEFAULT_EVAL_LIMIT=16
        DEFAULT_MAX_GEN_TOKENS=8
        DEFAULT_BPTT_WINDOW=12
        DEFAULT_NEGATIVE_SAMPLES=96
        DEFAULT_LR=0.001
        DEFAULT_NETWORK_LR_SCALE=0.25
        DEFAULT_NETWORK_LAYERS=3
        DEFAULT_NEURONS_PER_LAYER=12
        DEFAULT_COHERENCE_ALIGNMENT=0.0
        DEFAULT_ALIGNMENT_START=0.0
        DEFAULT_CONTRASTIVE=0.0
        DEFAULT_CONTRASTIVE_MARGIN=0.0
        DEFAULT_SCHEDULED_SAMPLING=0.0
        DEFAULT_LABEL_SMOOTHING=0.0
        DEFAULT_THOUGHT_LOGIT_AUX=0.0
        DEFAULT_THOUGHT_LOGIT_RESIDUAL=0.0
        DEFAULT_MERGE_BIAS=1.5
        DEFAULT_PAIR_SELECTION=head
        ;;
    baseline-v1-binding)
        DEFAULT_PAIRS=128
        DEFAULT_EPOCHS=2
        DEFAULT_EVAL_LIMIT=16
        DEFAULT_MAX_GEN_TOKENS=8
        DEFAULT_BPTT_WINDOW=16
        DEFAULT_NEGATIVE_SAMPLES=256
        DEFAULT_LR=0.0008
        DEFAULT_NETWORK_LR_SCALE=0.35
        DEFAULT_NETWORK_LAYERS=3
        DEFAULT_NEURONS_PER_LAYER=12
        DEFAULT_COHERENCE_ALIGNMENT=0.05
        DEFAULT_ALIGNMENT_START=0.15
        DEFAULT_CONTRASTIVE=0.03
        DEFAULT_CONTRASTIVE_MARGIN=0.05
        DEFAULT_SCHEDULED_SAMPLING=0.0
        DEFAULT_LABEL_SMOOTHING=0.02
        DEFAULT_THOUGHT_LOGIT_AUX=0.25
        DEFAULT_THOUGHT_LOGIT_RESIDUAL=0.35
        DEFAULT_MERGE_BIAS=1.5
        DEFAULT_PAIR_SELECTION=interleaved
        ;;
    baseline-v1-medium)
        DEFAULT_PAIRS=512
        DEFAULT_EPOCHS=3
        DEFAULT_EVAL_LIMIT=60
        DEFAULT_MAX_GEN_TOKENS=16
        DEFAULT_BPTT_WINDOW=16
        DEFAULT_NEGATIVE_SAMPLES=128
        DEFAULT_LR=0.0008
        DEFAULT_NETWORK_LR_SCALE=0.3
        DEFAULT_NETWORK_LAYERS=4
        DEFAULT_NEURONS_PER_LAYER=16
        DEFAULT_COHERENCE_ALIGNMENT=0.0
        DEFAULT_ALIGNMENT_START=0.0
        DEFAULT_CONTRASTIVE=0.0
        DEFAULT_CONTRASTIVE_MARGIN=0.0
        DEFAULT_SCHEDULED_SAMPLING=0.0
        DEFAULT_LABEL_SMOOTHING=0.0
        DEFAULT_THOUGHT_LOGIT_AUX=0.0
        DEFAULT_THOUGHT_LOGIT_RESIDUAL=0.0
        DEFAULT_MERGE_BIAS=1.5
        DEFAULT_PAIR_SELECTION=interleaved
        ;;
    custom)
        DEFAULT_PAIRS=16
        DEFAULT_EPOCHS=1
        DEFAULT_EVAL_LIMIT=4
        DEFAULT_MAX_GEN_TOKENS=4
        DEFAULT_BPTT_WINDOW=8
        DEFAULT_NEGATIVE_SAMPLES=64
        DEFAULT_LR=0.001
        DEFAULT_NETWORK_LR_SCALE=0.2
        DEFAULT_NETWORK_LAYERS=3
        DEFAULT_NEURONS_PER_LAYER=8
        DEFAULT_COHERENCE_ALIGNMENT=0.0
        DEFAULT_ALIGNMENT_START=0.0
        DEFAULT_CONTRASTIVE=0.0
        DEFAULT_CONTRASTIVE_MARGIN=0.0
        DEFAULT_SCHEDULED_SAMPLING=0.0
        DEFAULT_LABEL_SMOOTHING=0.0
        DEFAULT_THOUGHT_LOGIT_AUX=0.0
        DEFAULT_THOUGHT_LOGIT_RESIDUAL=0.0
        DEFAULT_MERGE_BIAS=1.5
        DEFAULT_PAIR_SELECTION=head
        ;;
    *)
        echo "BROCA_GATE_RECIPE must be smoke, baseline-v1-small, baseline-v1-binding, baseline-v1-medium, or custom" >&2
        exit 2
        ;;
esac

PAIR_COUNT="${BROCA_GATE_PAIRS:-$DEFAULT_PAIRS}"
EPOCHS="${BROCA_GATE_EPOCHS:-$DEFAULT_EPOCHS}"
EVAL_LIMIT="${BROCA_GATE_EVAL_LIMIT:-$DEFAULT_EVAL_LIMIT}"
MAX_GEN_TOKENS="${BROCA_GATE_MAX_GEN_TOKENS:-$DEFAULT_MAX_GEN_TOKENS}"
BPTT_WINDOW="${BROCA_GATE_BPTT_WINDOW:-$DEFAULT_BPTT_WINDOW}"
NEGATIVE_SAMPLES="${BROCA_GATE_NEGATIVE_SAMPLES:-$DEFAULT_NEGATIVE_SAMPLES}"
LR="${BROCA_GATE_LR:-$DEFAULT_LR}"
NETWORK_LR_SCALE="${BROCA_GATE_NETWORK_LR_SCALE:-$DEFAULT_NETWORK_LR_SCALE}"
NETWORK_LAYERS="${BROCA_GATE_NETWORK_LAYERS:-$DEFAULT_NETWORK_LAYERS}"
NEURONS_PER_LAYER="${BROCA_GATE_NEURONS_PER_LAYER:-$DEFAULT_NEURONS_PER_LAYER}"
COHERENCE_ALIGNMENT="${BROCA_GATE_COHERENCE_ALIGNMENT:-$DEFAULT_COHERENCE_ALIGNMENT}"
ALIGNMENT_START="${BROCA_GATE_ALIGNMENT_START:-$DEFAULT_ALIGNMENT_START}"
CONTRASTIVE="${BROCA_GATE_CONTRASTIVE:-$DEFAULT_CONTRASTIVE}"
CONTRASTIVE_MARGIN="${BROCA_GATE_CONTRASTIVE_MARGIN:-$DEFAULT_CONTRASTIVE_MARGIN}"
SCHEDULED_SAMPLING="${BROCA_GATE_SCHEDULED_SAMPLING:-$DEFAULT_SCHEDULED_SAMPLING}"
LABEL_SMOOTHING="${BROCA_GATE_LABEL_SMOOTHING:-$DEFAULT_LABEL_SMOOTHING}"
THOUGHT_LOGIT_AUX="${BROCA_GATE_THOUGHT_LOGIT_AUX:-$DEFAULT_THOUGHT_LOGIT_AUX}"
THOUGHT_LOGIT_RESIDUAL="${BROCA_GATE_THOUGHT_LOGIT_RESIDUAL:-$DEFAULT_THOUGHT_LOGIT_RESIDUAL}"
MERGE_BIAS="${BROCA_GATE_MERGE_BIAS:-$DEFAULT_MERGE_BIAS}"
PAIR_SELECTION="${BROCA_GATE_PAIR_SELECTION:-$DEFAULT_PAIR_SELECTION}"

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
echo "[broca-gate] recipe: $RECIPE"
echo "[broca-gate] pair selection: $PAIR_SELECTION"
echo "[broca-gate] thought-logit aux/residual: $THOUGHT_LOGIT_AUX / $THOUGHT_LOGIT_RESIDUAL"
echo "[broca-gate] generating curriculum data"
run cargo run -p symthaea-broca "${CARGO_FEATURE_ARGS[@]}" --bin broca-train -- --curriculum "$FULL_DATA"

case "$PAIR_SELECTION" in
    head)
        head -n "$PAIR_COUNT" "$FULL_DATA" > "$TRAIN_DATA"
        ;;
    strided | interleaved)
        awk -v n="$PAIR_COUNT" '
            { lines[NR] = $0 }
            END {
                if (n <= 0 || NR == 0) {
                    exit
                }
                if (n >= NR) {
                    for (i = 1; i <= NR; i++) {
                        print lines[i]
                    }
                    exit
                }
                if (n == 1) {
                    print lines[1]
                    exit
                }
                step = 997
                while (gcd(step, NR) != 1) {
                    step += 2
                }
                for (i = 0; i < n; i++) {
                    idx = ((i * step) % NR) + 1
                    print lines[idx]
                }
            }
            function gcd(a, b, t) {
                while (b != 0) {
                    t = a % b
                    a = b
                    b = t
                }
                return a
            }
        ' "$FULL_DATA" > "$TRAIN_DATA"
        ;;
    *)
        echo "BROCA_GATE_PAIR_SELECTION must be head, strided, or interleaved" >&2
        exit 2
        ;;
esac
echo "[broca-gate] training $PAIR_COUNT pairs for $EPOCHS epoch(s)"
run cargo run -p symthaea-broca "${CARGO_FEATURE_ARGS[@]}" --bin broca-train -- \
    --data "$TRAIN_DATA" \
    --epochs "$EPOCHS" \
    --bptt-window "$BPTT_WINDOW" \
    --negative-samples "$NEGATIVE_SAMPLES" \
    --lr "$LR" \
    --network-lr-scale "$NETWORK_LR_SCALE" \
    --network-layers "$NETWORK_LAYERS" \
    --neurons-per-layer "$NEURONS_PER_LAYER" \
    --coherence-alignment "$COHERENCE_ALIGNMENT" \
    --alignment-start "$ALIGNMENT_START" \
    --contrastive "$CONTRASTIVE" \
    --contrastive-margin "$CONTRASTIVE_MARGIN" \
    --scheduled-sampling "$SCHEDULED_SAMPLING" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --thought-logit-aux "$THOUGHT_LOGIT_AUX" \
    --thought-logit-residual "$THOUGHT_LOGIT_RESIDUAL" \
    --merge-bias "$MERGE_BIAS" \
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
    BROCA_TRAIN_RECIPE="$RECIPE" \
    BROCA_TRAIN_PAIR_COUNT="$PAIR_COUNT" \
    BROCA_TRAIN_EPOCHS="$EPOCHS" \
    BROCA_TRAIN_BPTT_WINDOW="$BPTT_WINDOW" \
    BROCA_TRAIN_NEGATIVE_SAMPLES="$NEGATIVE_SAMPLES" \
    BROCA_TRAIN_LR="$LR" \
    BROCA_TRAIN_NETWORK_LR_SCALE="$NETWORK_LR_SCALE" \
    BROCA_TRAIN_NETWORK_LAYERS="$NETWORK_LAYERS" \
    BROCA_TRAIN_NEURONS_PER_LAYER="$NEURONS_PER_LAYER" \
    BROCA_TRAIN_COHERENCE_ALIGNMENT="$COHERENCE_ALIGNMENT" \
    BROCA_TRAIN_ALIGNMENT_START="$ALIGNMENT_START" \
    BROCA_TRAIN_CONTRASTIVE="$CONTRASTIVE" \
    BROCA_TRAIN_CONTRASTIVE_MARGIN="$CONTRASTIVE_MARGIN" \
    BROCA_TRAIN_SCHEDULED_SAMPLING="$SCHEDULED_SAMPLING" \
    BROCA_TRAIN_LABEL_SMOOTHING="$LABEL_SMOOTHING" \
    BROCA_TRAIN_THOUGHT_LOGIT_AUX="$THOUGHT_LOGIT_AUX" \
    BROCA_TRAIN_THOUGHT_LOGIT_RESIDUAL="$THOUGHT_LOGIT_RESIDUAL" \
    BROCA_TRAIN_PAIR_SELECTION="$PAIR_SELECTION" \
    BROCA_TRAIN_MERGE_BIAS="$MERGE_BIAS" \
    cargo run -p symthaea-broca "${CARGO_FEATURE_ARGS[@]}" --bin broca-eval -- \
    --checkpoint "$CHECKPOINT" \
    --canonical-eval "$CANONICAL" \
    --eval-limit "$EVAL_LIMIT" \
    --max-gen-tokens "$MAX_GEN_TOKENS" \
    --json-out "$REPORT" \
    --thought-logit-residual "$THOUGHT_LOGIT_RESIDUAL" \
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
