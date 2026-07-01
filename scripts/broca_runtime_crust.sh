#!/usr/bin/env bash
# Shared Broca runtime selection helpers.
#
# Source this file from Broca shell entrypoints. It centralizes backend
# detection so measurement, baselines, and training do not drift.

broca_detect_backend() {
  local requested="${BROCA_MAMBA_BACKEND:-auto}"

  case "$requested" in
    gpu|cuda|mamba)
      if broca_gpu_available; then
        printf 'gpu\t%s\n' "requested gpu and CUDA runtime is visible"
      else
        echo "[broca] BROCA_MAMBA_BACKEND=$requested requested GPU, but CUDA is not usable" >&2
        echo "[broca] require nvidia-smi, nvcc, and at least one visible NVIDIA GPU" >&2
        exit 2
      fi
      ;;
    cpu|mamba-cpu)
      printf 'cpu\t%s\n' "requested cpu"
      ;;
    auto|"")
      if broca_gpu_available; then
        printf 'gpu\t%s\n' "auto detected nvidia-smi, nvcc, and a visible NVIDIA GPU"
      else
        printf 'cpu\t%s\n' "auto fallback: CUDA runtime or GPU not visible"
      fi
      ;;
    *)
      echo "[broca] invalid BROCA_MAMBA_BACKEND=$requested" >&2
      echo "[broca] expected auto, gpu, or cpu" >&2
      exit 2
      ;;
  esac
}

broca_gpu_available() {
  command -v nvidia-smi >/dev/null 2>&1 \
    && command -v nvcc >/dev/null 2>&1 \
    && nvidia-smi -L >/dev/null 2>&1
}

broca_resolve_runtime() {
  local detected
  detected="$(broca_detect_backend)"
  BROCA_SELECTED_BACKEND="${detected%%$'\t'*}"
  BROCA_BACKEND_REASON="${detected#*$'\t'}"
  export BROCA_SELECTED_BACKEND
  export BROCA_BACKEND_REASON

  if [[ "$BROCA_SELECTED_BACKEND" == "gpu" ]]; then
    BROCA_MAMBA_FEATURE="mamba"
  else
    BROCA_MAMBA_FEATURE="mamba-cpu"
  fi
  export BROCA_MAMBA_FEATURE

  BROCA_DECODER_FEATURES="${BROCA_DECODER_FEATURES:-$BROCA_MAMBA_FEATURE}"
  BROCA_EXERCISM_FEATURES="${BROCA_EXERCISM_FEATURES:-$BROCA_MAMBA_FEATURE,code-sheaf-eval}"
  export BROCA_DECODER_FEATURES
  export BROCA_EXERCISM_FEATURES
}

broca_print_runtime() {
  echo "[broca] selected Mamba backend: $BROCA_SELECTED_BACKEND ($BROCA_MAMBA_FEATURE)"
  echo "[broca] backend reason: $BROCA_BACKEND_REASON"
}

broca_write_runtime_manifest() {
  echo "broca_mamba_backend=${BROCA_MAMBA_BACKEND:-auto}"
  echo "broca_selected_backend=$BROCA_SELECTED_BACKEND"
  echo "broca_backend_reason=$BROCA_BACKEND_REASON"
  echo "broca_mamba_feature=$BROCA_MAMBA_FEATURE"
  echo "broca_decoder_features=$BROCA_DECODER_FEATURES"
  echo "broca_exercism_features=$BROCA_EXERCISM_FEATURES"
}
