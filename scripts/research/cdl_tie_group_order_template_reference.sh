#!/usr/bin/env bash
set -euo pipefail

# Standalone test of repeated timestamp tie-group order templates.
# It validates the schedule primitive, not the complete PRN1 frame.
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${ROOT}/research/cdl-bench/reference/tie_group_order_template_reference.cpp"
CXX="${CXX:-g++}"
OUT_DIR="${CDL_TIE_TEMPLATE_OUT:-${ROOT}/target/cdl-tie-template-reference}"

command -v "${CXX}" >/dev/null 2>&1 || { echo "missing compiler: ${CXX}" >&2; exit 2; }
[[ -f "${SOURCE}" ]] || { echo "missing source: ${SOURCE}" >&2; exit 2; }
mkdir -p "${OUT_DIR}"
"${CXX}" -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror "${SOURCE}" \
  -o "${OUT_DIR}/tie_group_order_template_reference"
"${OUT_DIR}/tie_group_order_template_reference"
