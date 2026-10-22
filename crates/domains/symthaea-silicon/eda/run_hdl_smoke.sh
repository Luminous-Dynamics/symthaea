#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Reproducible HDC XOR RTL smoke sweep. A PASS means only that each width
# passed this exact lint/build/simulation command, not that it is faster.

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
silicon_dir="$(cd -- "${script_dir}/.." && pwd)"
repo_root="$(git -C "${silicon_dir}" rev-parse --show-toplevel 2>/dev/null)" || repo_root=""
if [[ -z "${repo_root}" ]]; then
  repo_root="$(cd "${silicon_dir}/../../.." && pwd)"
fi

rtl="${silicon_dir}/rtl/hdc_bind_xor.sv"
testbench="${silicon_dir}/rtl/tb_hdc_bind_xor.sv"
stamp="$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || printf 'timestamp-unavailable')"
out_root="${HDL_OUT_DIR:-${repo_root}/target/hdl-qualification/hdc-bind-xor}"
out_dir="${out_root}/${stamp}-pid${BASHPID}"
mkdir -p "${out_dir}" || { printf 'cannot create output directory: %s\n' "${out_dir}" >&2; exit 2; }

missing=()
for tool in verilator jq sha256sum git make c++; do
  if ! command -v "${tool}" >/dev/null 2>&1; then missing+=("${tool}"); fi
done

rtl_sha="unavailable"
tb_sha="unavailable"
if command -v sha256sum >/dev/null 2>&1; then
  if [[ -f "${rtl}" ]]; then rtl_sha="$(sha256sum "${rtl}" | awk '{print $1}')"; fi
  if [[ -f "${testbench}" ]]; then tb_sha="$(sha256sum "${testbench}" | awk '{print $1}')"; fi
fi
git_head="unavailable"
if command -v git >/dev/null 2>&1; then
  git_head="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || printf 'unavailable')"
fi

if [[ ! -f "${rtl}" || ! -f "${testbench}" ]]; then
  printf 'Missing required source file(s): %s %s\n' "${rtl}" "${testbench}" >&2
  exit 2
fi

# jq is needed to emit the structured receipt. If missing, preserve a clear
# text blocker instead of emitting a hand-built or misleading JSON document.
if ! command -v jq >/dev/null 2>&1; then
  printf 'status=unqualified_tool_missing\nmissing_tool=jq\nrtl_sha256=%s\ntestbench_sha256=%s\n' "${rtl_sha}" "${tb_sha}" > "${out_dir}/receipt.txt"
  printf 'UNQUALIFIED: jq is missing; no JSON receipt was emitted. Output: %s\n' "${out_dir}" >&2
  exit 2
fi

missing_json="[]"
if (("${#missing[@]}" > 0)); then
  missing_json="$(printf '%s\n' "${missing[@]}" | jq -R . | jq -s .)"
fi

verilator_version="unavailable"
if command -v verilator >/dev/null 2>&1; then
  if verilator --version > "${out_dir}/verilator-version.txt" 2>&1; then
    verilator_version="$(head -n 1 "${out_dir}/verilator-version.txt")"
  else
    verilator_version="version-command-failed"
  fi
fi

if (("${#missing[@]}" > 0)); then
  jq -n \
    --arg status "unqualified_tool_missing" \
    --arg timestamp_utc "${stamp}" \
    --arg git_head "${git_head}" \
    --arg rtl_sha256 "${rtl_sha}" \
    --arg testbench_sha256 "${tb_sha}" \
    --arg verilator_version "${verilator_version}" \
    --argjson missing_tools "${missing_json}" \
    '{schema_version:"1.0", qualification:"hdc_bind_xor_width_sweep", status:$status, timestamp_utc:$timestamp_utc, git_head:$git_head, rtl_sha256:$rtl_sha256, testbench_sha256:$testbench_sha256, verilator_version:$verilator_version, missing_tools:$missing_tools, widths_tested:[], width_results:[], claim_ceiling:"No RTL simulation, formal proof, synthesis/PPA, FPGA observation, physical energy or SOTA claim."}' \
    > "${out_dir}/receipt.json"
  printf 'UNQUALIFIED: missing required tools: %s\nReceipt: %s/receipt.json\n' "${missing[*]}" "${out_dir}" >&2
  exit 2
fi

width_results="[]"
overall_status="simulation_smoke_pass"
for width in 64 128 256 512; do
  width_dir="${out_dir}/width-${width}"
  mkdir -p "${width_dir}" || { overall_status="failed"; continue; }

  lint_rc=125
  build_rc=125
  simulation_rc=125
  simulation_marker="false"

  verilator --lint-only --timing -Wall --top-module tb_hdc_bind_xor "-GDATA_W=${width}" \
    "${rtl}" "${testbench}" > "${width_dir}/lint.log" 2>&1
  lint_rc=$?

  if [[ "${lint_rc}" -eq 0 ]]; then
    verilator --binary --timing --assert -Wall --top-module tb_hdc_bind_xor "-GDATA_W=${width}" \
      --Mdir "${width_dir}/obj_dir" "${rtl}" "${testbench}" > "${width_dir}/build.log" 2>&1
    build_rc=$?
    if [[ "${build_rc}" -eq 0 ]]; then
      "${width_dir}/obj_dir/Vtb_hdc_bind_xor" > "${width_dir}/simulation.log" 2>&1
      simulation_rc=$?
      if [[ "${simulation_rc}" -eq 0 ]] && grep -Fq 'PASS: hdc_bind_xor smoke and backpressure tests completed' "${width_dir}/simulation.log"; then
        simulation_marker="true"
      fi
    fi
  fi

  width_status="failed"
  if [[ "${lint_rc}" -eq 0 && "${build_rc}" -eq 0 && "${simulation_rc}" -eq 0 && "${simulation_marker}" == "true" ]]; then
    width_status="simulation_smoke_pass"
  else
    overall_status="failed"
  fi

  for log_file in "${width_dir}"/*.log; do
    [[ -f "${log_file}" ]] || continue
    sha256sum "${log_file}" >> "${out_dir}/log-sha256.txt"
  done

  width_result="$(jq -n \
    --arg width "${width}" \
    --arg status "${width_status}" \
    --arg lint_rc "${lint_rc}" \
    --arg build_rc "${build_rc}" \
    --arg simulation_rc "${simulation_rc}" \
    --argjson simulation_marker "${simulation_marker}" \
    '{datapath_width_bits:($width|tonumber), status:$status, lint_exit_code:($lint_rc|tonumber), build_exit_code:($build_rc|tonumber), simulation_exit_code:($simulation_rc|tonumber), expected_pass_marker_seen:$simulation_marker}')"
  width_results="$(jq -cn --argjson existing "${width_results}" --argjson next "${width_result}" '$existing + [$next]')"
  printf 'width=%s status=%s lint=%s build=%s sim=%s\n' "${width}" "${width_status}" "${lint_rc}" "${build_rc}" "${simulation_rc}"
done

jq -n \
  --arg status "${overall_status}" \
  --arg timestamp_utc "${stamp}" \
  --arg git_head "${git_head}" \
  --arg rtl_sha256 "${rtl_sha}" \
  --arg testbench_sha256 "${tb_sha}" \
  --arg verilator_version "${verilator_version}" \
  --arg out_dir "${out_dir}" \
  --argjson width_results "${width_results}" \
  '{schema_version:"1.0", qualification:"hdc_bind_xor_width_sweep", status:$status, timestamp_utc:$timestamp_utc, git_head:$git_head, rtl_sha256:$rtl_sha256, testbench_sha256:$testbench_sha256, verilator_version:$verilator_version, widths_tested:[$width_results[].datapath_width_bits], width_results:$width_results, output_directory:$out_dir, claim_ceiling:"A pass qualifies only these word-level RTL/testbench widths under this simulator command. It is not formal proof, synthesis/PPA, FPGA observation, physical energy, throughput superiority or product/SOTA qualification."}' \
  > "${out_dir}/receipt.json"

printf 'status=%s\nreceipt=%s/receipt.json\n' "${overall_status}" "${out_dir}/receipt.json"
if [[ "${overall_status}" != "simulation_smoke_pass" ]]; then exit 1; fi
