#!/usr/bin/env bash
set -euo pipefail

# Paper-derived predictive flag mapper diagnostic only.
# No ACTF/XOR payload branches or Rabbit TOE are implemented here.

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${ROOT}/research/cdl-bench/reference/rabbit_branch_mapper_reference.cpp"
CXX="${CXX:-g++}"
OUT_DIR="${CDL_RABBIT_OUT:-${ROOT}/target/cdl-rabbit-branch-reference}"

command -v "${CXX}" >/dev/null 2>&1 || { echo "missing compiler: ${CXX}" >&2; exit 2; }
[[ -f "${SOURCE}" ]] || { echo "missing source: ${SOURCE}" >&2; exit 2; }
mkdir -p "${OUT_DIR}"

"${CXX}" -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  "${SOURCE}" -o "${OUT_DIR}/rabbit_branch_mapper_reference"
"${OUT_DIR}/rabbit_branch_mapper_reference"
