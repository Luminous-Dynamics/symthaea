#!/usr/bin/env bash
set -euo pipefail

# Reproduce the synthetic typed-record state-delta reference and compare
# Zstandard at one fixed level on raw record bytes vs delta-transformed bytes.
# This is a transform+Zstd diagnostic, not a replacement for the Rust CDL codec.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${ROOT}/research/cdl-bench/reference/typed_record_delta_reference.cpp"
CXX="${CXX:-g++}"
ZSTD="${ZSTD:-zstd}"
LEVEL="${ZSTD_LEVEL:-19}"

command -v "${CXX}" >/dev/null 2>&1 || { echo "missing compiler: ${CXX}" >&2; exit 2; }
command -v "${ZSTD}" >/dev/null 2>&1 || { echo "missing zstd executable: ${ZSTD}" >&2; exit 2; }
[[ -f "${SOURCE}" ]] || { echo "missing reference source: ${SOURCE}" >&2; exit 2; }

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
mkdir -p "${TMP}/streams"

"${CXX}" -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror "${SOURCE}" -o "${TMP}/typed_record_delta_reference"
"${TMP}/typed_record_delta_reference" --emit-dir "${TMP}/streams"

printf 'zstd_version='
"${ZSTD}" --version | head -n 1
printf 'zstd_level=%s\n' "${LEVEL}"
printf 'fixture,raw_bytes,state_delta_bytes,zstd_raw_bytes,zstd_state_delta_bytes,delta_plus_zstd_vs_raw_zstd_pct\n'

for name in sparse-updates bursty-updates noisy-updates mostly-idle; do
  raw="${TMP}/streams/${name}_raw.bin"
  delta="${TMP}/streams/${name}_state_delta.bin"
  raw_zst="${TMP}/${name}_raw.zst"
  delta_zst="${TMP}/${name}_state_delta.zst"

  "${ZSTD}" -q -"${LEVEL}" -f -c "${raw}" > "${raw_zst}"
  "${ZSTD}" -q -"${LEVEL}" -f -c "${delta}" > "${delta_zst}"

  # Both compressed representations must decode byte-for-byte to their inputs.
  "${ZSTD}" -q -d -c "${raw_zst}" | cmp - "${raw}"
  "${ZSTD}" -q -d -c "${delta_zst}" | cmp - "${delta}"

  raw_bytes="$(wc -c < "${raw}" | tr -d ' ')"
  delta_bytes="$(wc -c < "${delta}" | tr -d ' ')"
  zraw_bytes="$(wc -c < "${raw_zst}" | tr -d ' ')"
  zdelta_bytes="$(wc -c < "${delta_zst}" | tr -d ' ')"
  pct="$(awk -v a="${zraw_bytes}" -v b="${zdelta_bytes}" 'BEGIN { if (a == 0) print "0.00"; else printf "%.2f", (b/a - 1.0) * 100.0 }')"
  printf '%s,%s,%s,%s,%s,%s\n' "${name}" "${raw_bytes}" "${delta_bytes}" "${zraw_bytes}" "${zdelta_bytes}" "${pct}"
done
