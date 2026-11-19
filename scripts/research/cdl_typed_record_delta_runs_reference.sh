#!/usr/bin/env bash
set -euo pipefail

# Independent C++20 transform reference: raw vs state-delta vs record-run state-delta.
# This is not the Rust CDL crate and has no arithmetic-coded TRD1 frame.

ROOT="$(CDPATH= cd -- "$(dirname -- "\${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="\${ROOT}/research/cdl-bench/reference/typed_record_delta_runs_reference.cpp"
CXX="\${CXX:-g++}"
ZSTD="\${ZSTD:-zstd}"
LEVEL="\${ZSTD_LEVEL:-19}"
OUT_DIR="\${CDL_RECORD_RUN_OUT:-\${ROOT}/target/cdl-record-run-reference}"

command -v "\${CXX}" >/dev/null 2>&1 || { echo "missing compiler: \${CXX}" >&2; exit 2; }
command -v "\${ZSTD}" >/dev/null 2>&1 || { echo "missing zstd: \${ZSTD}" >&2; exit 2; }
[[ -f "\${SOURCE}" ]] || { echo "missing source: \${SOURCE}" >&2; exit 2; }
mkdir -p "\${OUT_DIR}/streams"

"\${CXX}" -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  "\${SOURCE}" -o "\${OUT_DIR}/typed_record_delta_runs_reference"
"\${OUT_DIR}/typed_record_delta_runs_reference" --emit-dir "\${OUT_DIR}/streams"

printf 'zstd_version='
"\${ZSTD}" --version | head -n 1
printf 'zstd_level=%s\n' "\${LEVEL}"
printf 'fixture,raw_bytes,state_delta_bytes,state_delta_record_run_bytes,zstd_raw_bytes,zstd_state_delta_bytes,zstd_record_run_bytes,run_vs_state_delta_pct,zstd_run_vs_zstd_delta_pct,zstd_run_vs_zstd_raw_pct\n'

for name in sparse bursty noisy idle; do
  raw="\${OUT_DIR}/streams/\${name}_raw.bin"
  delta="\${OUT_DIR}/streams/\${name}_state_delta.bin"
  runs="\${OUT_DIR}/streams/\${name}_state_delta_record_runs.bin"
  raw_zst="\${OUT_DIR}/streams/\${name}_raw.zst"
  delta_zst="\${OUT_DIR}/streams/\${name}_state_delta.zst"
  runs_zst="\${OUT_DIR}/streams/\${name}_state_delta_record_runs.zst"

  "\${ZSTD}" -q -"\${LEVEL}" -f -c "\${raw}" > "\${raw_zst}"
  "\${ZSTD}" -q -"\${LEVEL}" -f -c "\${delta}" > "\${delta_zst}"
  "\${ZSTD}" -q -"\${LEVEL}" -f -c "\${runs}" > "\${runs_zst}"
  "\${ZSTD}" -q -d -c "\${raw_zst}" | cmp - "\${raw}"
  "\${ZSTD}" -q -d -c "\${delta_zst}" | cmp - "\${delta}"
  "\${ZSTD}" -q -d -c "\${runs_zst}" | cmp - "\${runs}"

  raw_bytes="$(wc -c < "\${raw}" | tr -d ' ')"
  delta_bytes="$(wc -c < "\${delta}" | tr -d ' ')"
  run_bytes="$(wc -c < "\${runs}" | tr -d ' ')"
  raw_zst_bytes="$(wc -c < "\${raw_zst}" | tr -d ' ')"
  delta_zst_bytes="$(wc -c < "\${delta_zst}" | tr -d ' ')"
  run_zst_bytes="$(wc -c < "\${runs_zst}" | tr -d ' ')"
  run_vs_delta="$(awk -v a="\${delta_bytes}" -v b="\${run_bytes}" 'BEGIN { printf "%.2f", (b/a-1)*100 }')"
  zst_run_vs_delta="$(awk -v a="\${delta_zst_bytes}" -v b="\${run_zst_bytes}" 'BEGIN { printf "%.2f", (b/a-1)*100 }')"
  zst_run_vs_raw="$(awk -v a="\${raw_zst_bytes}" -v b="\${run_zst_bytes}" 'BEGIN { printf "%.2f", (b/a-1)*100 }')"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "\${name}" "\${raw_bytes}" "\${delta_bytes}" "\${run_bytes}" \
    "\${raw_zst_bytes}" "\${delta_zst_bytes}" "\${run_zst_bytes}" \
    "\${run_vs_delta}" "\${zst_run_vs_delta}" "\${zst_run_vs_raw}"
done

echo "all_zstd_decompressions=exact"
