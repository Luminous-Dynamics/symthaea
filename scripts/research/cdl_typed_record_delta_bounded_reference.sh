#!/usr/bin/env bash
set -euo pipefail

# Independent strict-warning C++ reference for the TRD1 state-delta grammar,
# followed by an apples-to-apples Zstandard comparison on identical records.
# This does not compile or execute the Rust CDL-Bench crate.

ROOT="$(CDPATH= cd -- "$(dirname -- "\${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="\${ROOT}/research/cdl-bench/reference/typed_record_delta_bounded_reference.cpp"
CXX="\${CXX:-g++}"
ZSTD="\${ZSTD:-zstd}"
LEVEL="\${ZSTD_LEVEL:-19}"

command -v "\${CXX}" >/dev/null 2>&1 || { echo "missing compiler: \${CXX}" >&2; exit 2; }
command -v "\${ZSTD}" >/dev/null 2>&1 || { echo "missing zstd: \${ZSTD}" >&2; exit 2; }
[[ -f "\${SOURCE}" ]] || { echo "missing source: \${SOURCE}" >&2; exit 2; }

TMP="$(mktemp -d)"
trap 'rm -rf "\${TMP}"' EXIT
mkdir -p "\${TMP}/streams"

"\${CXX}" -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  "\${SOURCE}" -o "\${TMP}/typed_record_delta_bounded_reference"

"\${TMP}/typed_record_delta_bounded_reference" --emit-dir "\${TMP}/streams"

printf 'zstd_version='
"\${ZSTD}" --version | head -n 1
printf 'zstd_level=%s\n' "\${LEVEL}"
printf 'fixture,records,raw_bytes,state_delta_bytes,zstd_raw_bytes,zstd_delta_bytes,delta_plus_zstd_vs_raw_zstd_pct\n'

for name in sparse bursty noisy idle; do
  raw="\${TMP}/streams/\${name}_raw.bin"
  delta="\${TMP}/streams/\${name}_state_delta.bin"
  raw_zst="\${TMP}/\${name}_raw.zst"
  delta_zst="\${TMP}/\${name}_delta.zst"

  "\${ZSTD}" -q -"\${LEVEL}" -f -c "\${raw}" > "\${raw_zst}"
  "\${ZSTD}" -q -"\${LEVEL}" -f -c "\${delta}" > "\${delta_zst}"

  # Verify each compressed representation restores its exact input.
  "\${ZSTD}" -q -d -c "\${raw_zst}" | cmp - "\${raw}"
  "\${ZSTD}" -q -d -c "\${delta_zst}" | cmp - "\${delta}"

  raw_bytes="$(wc -c < "\${raw}" | tr -d ' ')"
  delta_bytes="$(wc -c < "\${delta}" | tr -d ' ')"
  raw_zstd_bytes="$(wc -c < "\${raw_zst}" | tr -d ' ')"
  delta_zstd_bytes="$(wc -c < "\${delta_zst}" | tr -d ' ')"
  reduction="$(awk -v a="\${raw_zstd_bytes}" -v b="\${delta_zstd_bytes}" \
    'BEGIN { if (a == 0) print "NA"; else printf "%.2f%%", (b/a - 1.0) * 100.0 }')"
  printf '%s,65536,%s,%s,%s,%s,%s\n' \
    "\${name}" "\${raw_bytes}" "\${delta_bytes}" \
    "\${raw_zstd_bytes}" "\${delta_zstd_bytes}" "\${reduction}"
done
