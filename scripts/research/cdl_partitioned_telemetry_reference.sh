#!/usr/bin/env bash
set -euo pipefail

# Exact order-preserving per-series partition plus bit-packed interleave schedule.
# Independent C++ transform reference; not the Rust CDL codec.
ROOT="$(CDPATH= cd -- "$(dirname -- "\${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="\${ROOT}/research/cdl-bench/reference/typed_record_partitioned_runs_reference.cpp"
CXX="\${CXX:-g++}"
ZSTD="\${ZSTD:-zstd}"
LEVEL="\${ZSTD_LEVEL:-19}"
OUT_DIR="\${CDL_PARTITIONED_OUT:-\${ROOT}/target/cdl-partitioned-reference}"

command -v "\${CXX}" >/dev/null 2>&1 || { echo "missing compiler: \${CXX}" >&2; exit 2; }
command -v "\${ZSTD}" >/dev/null 2>&1 || { echo "missing zstd: \${ZSTD}" >&2; exit 2; }
[[ -f "\${SOURCE}" ]] || { echo "missing source: \${SOURCE}" >&2; exit 2; }
mkdir -p "\${OUT_DIR}/streams"

"\${CXX}" -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  "\${SOURCE}" -o "\${OUT_DIR}/typed_record_partitioned_runs_reference"
"\${OUT_DIR}/typed_record_partitioned_runs_reference" --emit-dir "\${OUT_DIR}/streams"

printf 'zstd_version='
"\${ZSTD}" --version | head -n 1
printf 'zstd_level=%s\n' "\${LEVEL}"
printf 'fixture,raw_bytes,global_state_delta_bytes,partitioned_frame_bytes,zstd_raw_bytes,zstd_global_delta_bytes,zstd_partitioned_bytes,partitioned_vs_global_delta_pct,partitioned_vs_raw_pct\n'

for name in sparse bursty idle noisy; do
  raw="\${OUT_DIR}/streams/\${name}_raw.bin"
  delta="\${OUT_DIR}/streams/\${name}_global_delta.bin"
  part="\${OUT_DIR}/streams/\${name}_partitioned.bin"
  raw_zst="\${OUT_DIR}/streams/\${name}_raw.zst"
  delta_zst="\${OUT_DIR}/streams/\${name}_global_delta.zst"
  part_zst="\${OUT_DIR}/streams/\${name}_partitioned.zst"

  "\${ZSTD}" -q -"\${LEVEL}" -f -c "\${raw}" > "\${raw_zst}"
  "\${ZSTD}" -q -"\${LEVEL}" -f -c "\${delta}" > "\${delta_zst}"
  "\${ZSTD}" -q -"\${LEVEL}" -f -c "\${part}" > "\${part_zst}"
  "\${ZSTD}" -q -d -c "\${raw_zst}" | cmp - "\${raw}"
  "\${ZSTD}" -q -d -c "\${delta_zst}" | cmp - "\${delta}"
  "\${ZSTD}" -q -d -c "\${part_zst}" | cmp - "\${part}"

  raw_bytes="$(wc -c < "\${raw}" | tr -d ' ')"
  delta_bytes="$(wc -c < "\${delta}" | tr -d ' ')"
  part_bytes="$(wc -c < "\${part}" | tr -d ' ')"
  raw_zst_bytes="$(wc -c < "\${raw_zst}" | tr -d ' ')"
  delta_zst_bytes="$(wc -c < "\${delta_zst}" | tr -d ' ')"
  part_zst_bytes="$(wc -c < "\${part_zst}" | tr -d ' ')"
  vs_delta="$(awk -v a="\${delta_zst_bytes}" -v b="\${part_zst_bytes}" 'BEGIN { printf "%.2f", (b/a-1)*100 }')"
  vs_raw="$(awk -v a="\${raw_zst_bytes}" -v b="\${part_zst_bytes}" 'BEGIN { printf "%.2f", (b/a-1)*100 }')"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "\${name}" "\${raw_bytes}" "\${delta_bytes}" "\${part_bytes}" \
    "\${raw_zst_bytes}" "\${delta_zst_bytes}" "\${part_zst_bytes}" "\${vs_delta}" "\${vs_raw}"
done

echo "all_zstd_decompressions=exact"
