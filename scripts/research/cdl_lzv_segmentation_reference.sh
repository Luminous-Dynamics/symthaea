#!/usr/bin/env bash
set -euo pipefail

# WZS-inspired leading-zero-variance segmentation control.
# This is not the WZS implementation and does not encode/compress the data.

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${ROOT}/research/cdl-bench/reference/lzv_segmentation_reference.cpp"
CXX="${CXX:-g++}"
OUT_DIR="${CDL_LZV_OUT:-${ROOT}/target/cdl-lzv-reference}"

command -v "${CXX}" >/dev/null 2>&1 || { echo "missing compiler: ${CXX}" >&2; exit 2; }
[[ -f "${SOURCE}" ]] || { echo "missing source: ${SOURCE}" >&2; exit 2; }
mkdir -p "${OUT_DIR}"

"${CXX}" -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  "${SOURCE}" -o "${OUT_DIR}/lzv_segmentation_reference"
"${OUT_DIR}/lzv_segmentation_reference"
