#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Reproducible smoke qualification for the small HDC XOR RTL slice.
# A PASS only means this exact RTL/testbench passed this simulation command.

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
silicon_dir="$(cd -- "${script_dir}/.." && pwd)"
repo_root="$(git -C "${silicon_dir}" rev-parse --show-toplevel 2>/dev/null || cd "${silicon_dir}/../../../../.." && pwd)"
rtl="${silicon_dir}/rtl/hdc_bind_xor.sv"
testbench="${silicon_dir}/rtl/tb_hdc_bind_xor.sv"
stamp="$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || printf 'timestamp-unavailable')"
out_root="${HDL_OUT_DIR:-${repo_root}/target/hdl-qualification/hdc-bind-xor}"
out_dir="${out_root}/${stamp}"
mkdir -p "${out_dir}" || { printf 'cannot create output directory: %s\n' "${out_dir}" >&2; exit 2; }

missing=()
for tool in verilator jq sha256sum git make c++; do
  if ! command -v "${tool}" >/dev/null 2>&1; then missing+=("${tool}"); fi
done

rtl_sha="unavailable"
tb_sha="unavailable"
if command -v sha256sum >/dev/null 2>&1; then
  [[ -f "${rtl}" ]] && rtl_sha="$(sha256sum "${rtl}" | awk '{print $1}')"
  [[ -f "${testbench}" ]] && tb_sha="$(sha256sum "${testbench}" | awk '{print $1}')"
fi
git_head="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || printf 'unavailable')"

if [[ ! -f "${rtl}" || ! -f "${testbench}" ]]; then
  printf 'Missing required source file(s): %s %s\n' "${rtl}" "${testbench}" >&2
  exit 2
fi

# jq is required for the evidence receipt; record a plain-text blocker if the
# host lacks jq so the script fails visibly instead of inventing JSON.
if ! command -v jq >/dev/null 2>&1; then
  printf 'status=unqualified_tool_missing\nmissing_tool=jq\nrtl_sha256=%s\ntestbench_sha256=%s\n' "${rtl_sha}" "${tb_sha}" > "${out_dir}/receipt.txt"
  printf 'UNQUALIFIED: jq is missing; no JSON receipt was emitted. Output: %s\n' "${out_dir}" >&2
  exit 2
fi

missing_json="$(printf '%s\n' "${missing[@]}" | jq -R . | jq -s .)"
verilator_version="unavailable"
if command -v verilator >/dev/null 2>&1; then
  verilator --version > "${out_dir}/verilator-version.txt" 2>&1
  verilator_version="$(head -n 1 "${out_dir}/verilator-version.txt")"
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
    '{schema_version:"1.0", qualification:"hdc_bind_xor_simulation_smoke", status:$status, timestamp_utc:$timestamp_utc, git_head:$git_head, rtl_sha256:$rtl_sha256, testbench_sha256:$testbench_sha256, verilator_version:$verilator_version, missing_tools:$missing_tools, claim_ceiling:"No RTL simulation, formal proof, synthesis/PPA, FPGA observation, physical energy or SOTA claim."}' \
    > "${out_dir}/receipt.json"
  printf 'UNQUALIFIED: missing required tools: %s\nReceipt: %s/receipt.json\n' "${missing[*]}" "${out_dir}" >&2
  exit 2
fi

lint_rc=125
build_rc=125
simulation_rc=125
simulation_marker="false"

set +e
verilator --lint-only --timing -Wall --top-module tb_hdc_bind_xor "${rtl}" "${testbench}" > "${out_dir}/lint.log" 2>&1
lint_rc=$?
set -e 2>/dev/null || true

if [[ "${lint_rc}" -eq 0 ]]; then
  set +e
  verilator --binary --timing --assert -Wall --top-module tb_hdc_bind_xor \
    --Mdir "${out_dir}/obj_dir" "${rtl}" "${testbench}" > "${out_dir}/build.log" 2>&1
  build_rc=$?
  set -e 2>/dev/null || true
  if [[ "${build_rc}" -eq 0 ]]; then
    set +e
    "${out_dir}/obj_dir/Vtb_hdc_bind_xor" > "${out_dir}/simulation.log" 2>&1
    simulation_rc=$?
    set -e 2>/dev/null || true
    if [[ "${simulation_rc}" -eq 0 ]] && grep -Fq 'PASS: hdc_bind_xor smoke and backpressure tests completed' "${out_dir}/simulation.log"; then
      simulation_marker="true"
    fi
  fi
fi

status="failed"
if [[ "${lint_rc}" -eq 0 && "${build_rc}" -eq 0 && "${simulation_rc}" -eq 0 && "${simulation_marker}" == "true" ]]; then
  status="simulation_smoke_pass"
fi

# Hash logs after the process exits. Log hashes are provenance, not attestations.
for file in lint.log build.log simulation.log verilator-version.txt; do
  if [[ -f "${out_dir}/${file}" ]]; then
    sha256sum "${out_dir}/${file}" >> "${out_dir}/log-sha256.txt"
  fi
done

jq -n \
  --arg status "${status}" \
  --arg timestamp_utc "${stamp}" \
  --arg git_head "${git_head}" \
  --arg rtl_sha256 "${rtl_sha}" \
  --arg testbench_sha256 "${tb_sha}" \
  --arg verilator_version "${verilator_version}" \
  --arg out_dir "${out_dir}" \
  --arg lint_rc "${lint_rc}" \
  --arg build_rc "${build_rc}" \
  --arg simulation_rc "${simulation_rc}" \
  --argjson simulation_marker "${simulation_marker}" \
  '{schema_version:"1.0", qualification:"hdc_bind_xor_simulation_smoke", status:$status, timestamp_utc:$timestamp_utc, git_head:$git_head, rtl_sha256:$rtl_sha256, testbench_sha256:$testbench_sha256, verilator_version:$verilator_version, lint_exit_code:($lint_rc|tonumber), build_exit_code:($build_rc|tonumber), simulation_exit_code:($simulation_rc|tonumber), expected_pass_marker_seen:$simulation_marker, output_directory:$out_dir, claim_ceiling:"A pass qualifies only the exact word-level RTL/testbench and simulator command. It is not formal proof, synthesis/PPA, FPGA observation, physical energy, or product/SOTA qualification."}' \
  > "${out_dir}/receipt.json"

printf 'status=%s\nreceipt=%s/receipt.json\n' "${status}" "${out_dir}"
if [[ "${status}" != "simulation_smoke_pass" ]]; then exit 1; fi
