#!/usr/bin/env bash
set -euo pipefail

# Per-series row-state, field-columnar and adaptive row/columnar exact-lossless layouts.
# Independent C++ reference; not the Rust CDL codec.
ROOT="$(CDPATH= cd -- "$(dirname -- "\${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="\${ROOT}/research/cdl-bench/reference/typed_record_partitioned_runs_reference.cpp"
CXX="\${CXX:-g++}"
ZSTD="\${ZSTD:-zstd}"
LEVEL="\${ZSTD_LEVEL:-19}"
OUT_DIR="\${CDL_PARTITIONED_OUT:-\${ROOT}/target/cdl-partitioned-reference}"

command -v "\${CXX}" >/dev/null 2>&1 || { echo "missing compiler: \${CXX}" >&2; exit 2; }
command -v "\${ZSTD}" >/dev/null 2>&1 || { echo "missing zstd: \${ZSTD}" >&2; exit 2; }
command -v pkg-config >/dev/null 2>&1 || { echo "missing pkg-config (required for libzstd)" >&2; exit 2; }
pkg-config --exists libzstd || { echo "missing libzstd development package" >&2; exit 2; }
[[ -f "\${SOURCE}" ]] || { echo "missing source: \${SOURCE}" >&2; exit 2; }
mkdir -p "\${OUT_DIR}/streams"

read -r -a ZSTD_CFLAGS <<< "$(pkg-config --cflags libzstd)"
read -r -a ZSTD_LIBS <<< "$(pkg-config --libs libzstd)"
"\${CXX}" -O2 -std=c++20 -Wall -Wextra -Wconversion -Wno-misleading-indentation -Werror \
  "\${ZSTD_CFLAGS[@]}" "\${SOURCE}" -o "\${OUT_DIR}/typed_record_partitioned_runs_reference" "\${ZSTD_LIBS[@]}"
"\${OUT_DIR}/typed_record_partitioned_runs_reference" --emit-dir "\${OUT_DIR}/streams"

printf 'zstd_version='
"\${ZSTD}" --version | head -n 1
printf 'zstd_level=%s\n' "\${LEVEL}"
printf 'fixture,raw_bytes,global_delta_bytes,row_partitioned_bytes,field_columnar_bytes,pax1_raw_selector_bytes,pzc1_child_zstd_bytes,zstd_raw_bytes,zstd_global_delta_bytes,zstd_row_partitioned_bytes,zstd_field_columnar_bytes,zstd_pax1_bytes,prn1_vs_global_delta_pct,pcl1_vs_global_delta_pct,pax1_vs_global_delta_pct,pzc1_vs_global_delta_pct\n'

for name in sparse bursty idle noisy; do
  raw="\${OUT_DIR}/streams/\${name}_raw.bin"
  delta="\${OUT_DIR}/streams/\${name}_global_delta.bin"
  part="\${OUT_DIR}/streams/\${name}_partitioned.bin"
  columnar="\${OUT_DIR}/streams/\${name}_columnar.bin"
  pax1="\${OUT_DIR}/streams/\${name}_adaptive.bin"
  pzc1="\${OUT_DIR}/streams/\${name}_adaptive_zstd.bin"
  raw_zst="\${OUT_DIR}/streams/\${name}_raw.zst"
  delta_zst="\${OUT_DIR}/streams/\${name}_global_delta.zst"
  part_zst="\${OUT_DIR}/streams/\${name}_partitioned.zst"
  columnar_zst="\${OUT_DIR}/streams/\${name}_columnar.zst"
  pax1_zst="\${OUT_DIR}/streams/\${name}_adaptive.zst"

  "\${ZSTD}" -q -"\${LEVEL}" -f -c "\${raw}" > "\${raw_zst}"
  "\${ZSTD}" -q -"\${LEVEL}" -f -c "\${delta}" > "\${delta_zst}"
  "\${ZSTD}" -q -"\${LEVEL}" -f -c "\${part}" > "\${part_zst}"
  "\${ZSTD}" -q -"\${LEVEL}" -f -c "\${columnar}" > "\${columnar_zst}"
  "\${ZSTD}" -q -"\${LEVEL}" -f -c "\${pax1}" > "\${pax1_zst}"

  "\${ZSTD}" -q -d -c "\${raw_zst}" | cmp - "\${raw}"
  "\${ZSTD}" -q -d -c "\${delta_zst}" | cmp - "\${delta}"
  "\${ZSTD}" -q -d -c "\${part_zst}" | cmp - "\${part}"
  "\${ZSTD}" -q -d -c "\${columnar_zst}" | cmp - "\${columnar}"
  "\${ZSTD}" -q -d -c "\${pax1_zst}" | cmp - "\${pax1}"

  raw_bytes="$(wc -c < "\${raw}" | tr -d ' ')"
  delta_bytes="$(wc -c < "\${delta}" | tr -d ' ')"
  part_bytes="$(wc -c < "\${part}" | tr -d ' ')"
  columnar_bytes="$(wc -c < "\${columnar}" | tr -d ' ')"
  pax1_bytes="$(wc -c < "\${pax1}" | tr -d ' ')"
  pzc1_bytes="$(wc -c < "\${pzc1}" | tr -d ' ')"
  raw_zst_bytes="$(wc -c < "\${raw_zst}" | tr -d ' ')"
  delta_zst_bytes="$(wc -c < "\${delta_zst}" | tr -d ' ')"
  part_zst_bytes="$(wc -c < "\${part_zst}" | tr -d ' ')"
  columnar_zst_bytes="$(wc -c < "\${columnar_zst}" | tr -d ' ')"
  pax1_zst_bytes="$(wc -c < "\${pax1_zst}" | tr -d ' ')"

  prn1_vs_delta="$(awk -v a="\${delta_zst_bytes}" -v b="\${part_zst_bytes}" 'BEGIN { printf "%.2f", (b/a-1)*100 }')"
  pcl1_vs_delta="$(awk -v a="\${delta_zst_bytes}" -v b="\${columnar_zst_bytes}" 'BEGIN { printf "%.2f", (b/a-1)*100 }')"
  pax1_vs_delta="$(awk -v a="\${delta_zst_bytes}" -v b="\${pax1_zst_bytes}" 'BEGIN { printf "%.2f", (b/a-1)*100 }')"
  pzc1_vs_delta="$(awk -v a="\${delta_zst_bytes}" -v b="\${pzc1_bytes}" 'BEGIN { printf "%.2f", (b/a-1)*100 }')"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "\${name}" "\${raw_bytes}" "\${delta_bytes}" "\${part_bytes}" "\${columnar_bytes}" \
    "\${pax1_bytes}" "\${pzc1_bytes}" "\${raw_zst_bytes}" "\${delta_zst_bytes}" \
    "\${part_zst_bytes}" "\${columnar_zst_bytes}" "\${pax1_zst_bytes}" \
    "\${prn1_vs_delta}" "\${pcl1_vs_delta}" "\${pax1_vs_delta}" "\${pzc1_vs_delta}"
done

echo "outer_zstd_decompressions=exact; PZC1 child-zstd round-trips=exact (C++ decoder)"
