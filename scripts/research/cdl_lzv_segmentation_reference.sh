#!/usr/bin/env bash
set -euo pipefail

# Independent WZS-inspired prefix-only LZV segmentation control.
# This does not reproduce the WZS paper implementation and does not measure
# compression ratios; it checks deterministic segmentation boundaries only.

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${ROOT}/research/cdl-bench/reference/lzv_segmentation_reference.cpp"
CXX="${CXX:-g++}"

command -v "${CXX}" >/dev/null 2>&1 || { echo "missing C++ compiler: ${CXX}" >&2; exit 2; }
[[ -f "${SOURCE}" ]] || { echo "missing source: ${SOURCE}" >&2; exit 2; }

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
"${CXX}" -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  "${SOURCE}" -o "${TMP}/lzv_segmentation_reference"
"${TMP}/lzv_segmentation_reference"
