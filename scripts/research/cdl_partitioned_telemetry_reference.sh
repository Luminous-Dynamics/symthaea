#!/usr/bin/env bash
set -euo pipefail

# Per-series row-state and field-columnar exact-lossless layouts with Zstd controls.
# Independent C++ transform reference; not the Rust CDL codec.
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${ROOT}/research/cdl-bench/reference/typed_record_partitioned_runs_reference.cpp"
CXX="${CXX:-g++}"
ZSTD="${ZSTD:-zstd}"
LEVEL="${ZSTD_LEVEL:-19}"
OUT_DIR="${CDL_PARTITIONED_OUT:-${ROOT}/target/cdl-partitioned-reference}"

command -v "${CXX}" >/dev/null 2>&1 || { echo "missing compiler: ${CXX}" >&2; exit 2; }
command -v "${ZSTD}" >/dev/null 2>&1 || { echo "missing zstd: ${ZSTD}" >&2; exit 2; }
command -v pkg-config >/dev/null 2>&1 || { echo "missing pkg-config (required for libzstd)" >&2; exit 2; }
pkg-config --exists libzstd || { echo "missing libzstd development package" >&2; exit 2; }
[[ -f "${SOURCE}" ]] || { echo "missing source: ${SOURCE}" >&2; exit 2; }
mkdir -p "${OUT_DIR}/streams"

read -r -a ZSTD_CFLAGS <<< "$(pkg-config --cflags libzstd)"
read -r -a ZSTD_LIBS <<< "$(pkg-config --libs libzstd)"
"${CXX}" -O2 -std=c++20 -Wall -Wextra -Wconversion -Wno-misleading-indentation -Werror \
  "${ZSTD_CFLAGS[@]}" "${SOURCE}" -o "${OUT_DIR}/typed_record_partitioned_runs_reference" "${ZSTD_LIBS[@]}"
"${OUT_DIR}/typed_record_partitioned_runs_reference" --emit-dir "${OUT_DIR}/streams"

printf 'zstd_version='
"${ZSTD}" --version | head -n 1
printf 'zstd_level=%s\n' "${LEVEL}"
printf 'fixture,raw_bytes,global_delta_bytes,row_partitioned_bytes,field_columnar_bytes,raw_adaptive_bytes,child_zstd_adaptive_bytes,zstd_raw_bytes,zstd_global_delta_bytes,zstd_row_partitioned_bytes,zstd_field_columnar_bytes,zstd_raw_adaptive_bytes,row_vs_global_delta_pct,columnar_vs_global_delta_pct,raw_adaptive_vs_global_delta_pct,child_zstd_adaptive_vs_global_delta_pct\n'

for name in sparse bursty idle noisy; do
  raw="${OUT_DIR}/streams/${name}_raw.bin"
  delta="${OUT_DIR}/streams/${name}_global_delta.bin"
  part="${OUT_DIR}/streams/${name}_partitioned.bin"
  columnar="${OUT_DIR}/streams/${name}_columnar.bin"
  adaptive="${OUT_DIR}/streams/${name}_adaptive.bin"
  adaptive_zstd="${OUT_DIR}/streams/${name}_adaptive_zstd.bin"
  raw_zst="${OUT_DIR}/streams/${name}_raw.zst"
  delta_zst="${OUT_DIR}/streams/${name}_global_delta.zst"
  part_zst="${OUT_DIR}/streams/${name}_partitioned.zst"
  columnar_zst="${OUT_DIR}/streams/${name}_columnar.zst"
  adaptive_zst="${OUT_DIR}/streams/${name}_adaptive.zst"

  "${ZSTD}" -q -"${LEVEL}" -f -c "${raw}" > "${raw_zst}"
  "${ZSTD}" -q -"${LEVEL}" -f -c "${delta}" > "${delta_zst}"
  "${ZSTD}" -q -"${LEVEL}" -f -c "${part}" > "${part_zst}"
  "${ZSTD}" -q -"${LEVEL}" -f -c "${columnar}" > "${columnar_zst}"
  "${ZSTD}" -q -"${LEVEL}" -f -c "${adaptive}" > "${adaptive_zst}"
  "${ZSTD}" -q -d -c "${raw_zst}" | cmp - "${raw}"
  "${ZSTD}" -q -d -c "${delta_zst}" | cmp - "${delta}"
  "${ZSTD}" -q -d -c "${part_zst}" | cmp - "${part}"
  "${ZSTD}" -q -d -c "${columnar_zst}" | cmp - "${columnar}"
  "${ZSTD}" -q -d -c "${adaptive_zst}" | cmp - "${adaptive}"

  raw_bytes="$(wc -c < "${raw}" | tr -d ' ')"
  delta_bytes="$(wc -c < "${delta}" | tr -d ' ')"
  part_bytes="$(wc -c < "${part}" | tr -d ' ')"
  columnar_bytes="$(wc -c < "${columnar}" | tr -d ' ')"
  adaptive_bytes="$(wc -c < "${adaptive}" | tr -d ' ')"
  adaptive_zstd_bytes="$(wc -c < "${adaptive_zstd}" | tr -d ' ')"
  raw_zst_bytes="$(wc -c < "${raw_zst}" | tr -d ' ')"
  delta_zst_bytes="$(wc -c < "${delta_zst}" | tr -d ' ')"
  part_zst_bytes="$(wc -c < "${part_zst}" | tr -d ' ')"
  columnar_zst_bytes="$(wc -c < "${columnar_zst}" | tr -d ' ')"
  adaptive_zst_frame_bytes="$(wc -c < "${adaptive_zst}" | tr -d ' ')"
  row_vs_delta="$(awk -v a="${delta_zst_bytes}" -v b="${part_zst_bytes}" 'BEGIN { printf "%.2f", (b/a-1)*100 }')"
  col_vs_delta="$(awk -v a="${delta_zst_bytes}" -v b="${columnar_zst_bytes}" 'BEGIN { printf "%.2f", (b/a-1)*100 }')"
  raw_adaptive_vs_delta="$(awk -v a="${delta_zst_bytes}" -v b="${adaptive_zst_bytes}" 'BEGIN { printf "%.2f", (b/a-1)*100 }')"
  child_zstd_vs_delta="$(awk -v a="${delta_zst_bytes}" -v b="${adaptive_zstd_frame_bytes}" 'BEGIN { printf "%.2f", (b/a-1)*100 }')"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${name}" "${raw_bytes}" "${delta_bytes}" "${part_bytes}" "${columnar_bytes}" \
    "${adaptive_bytes}" "${adaptive_zstd_bytes}" "${raw_zst_bytes}" "${delta_zst_bytes}" \
    "${part_zst_bytes}" "${columnar_zst_bytes}" "${adaptive_zst_bytes}" \
    "${row_vs_delta}" "${col_vs_delta}" "${raw_adaptive_vs_delta}" "${child_zstd_vs_delta}"
done

echo "all_outer_zstd_decompressions=exact; adaptive-child-zstd=exact-roundtrip-tested-by-C++"
