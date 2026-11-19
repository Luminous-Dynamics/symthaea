#!/usr/bin/env bash
set -euo pipefail

# Independent reference for Rabbit's four-branch prefix remapping only.
# This does not implement the floating-point payload branches or TOE.

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${ROOT}/research/cdl-bench/reference/rabbit_flag_mapping_reference.cpp"
CXX="${CXX:-g++}"

command -v "${CXX}" >/dev/null 2>&1 || { echo "missing C++ compiler: ${CXX}" >&2; exit 2; }
[[ -f "${SOURCE}" ]] || { echo "missing source: ${SOURCE}" >&2; exit 2; }

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
"${CXX}" -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  "${SOURCE}" -o "${TMP}/rabbit_flag_mapping_reference"
"${TMP}/rabbit_flag_mapping_reference"
