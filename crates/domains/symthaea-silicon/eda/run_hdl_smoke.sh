#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# HDC RTL correctness sweep only: XOR plus linear/tree Hamming count, four widths.
# A PASS is functional simulation evidence for these exact modules and tools;
# it is not a PPA, power, FPGA, ASIC or product/SOTA qualification.

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
silicon_dir="$(cd -- "${script_dir}/.." && pwd)"
repo_root="$(cd -- "${silicon_dir}/../../.." && pwd)"
rtl_xor="${silicon_dir}/rtl/hdc_bind_xor.sv"
tb_xor="${silicon_dir}/rtl/tb_hdc_bind_xor.sv"
rtl_hamming="${silicon_dir}/rtl/hdc_hamming_count.sv"
tb_hamming="${silicon_dir}/rtl/tb_hdc_hamming_count.sv"
manifest="${script_dir}/hdl_toolchain_manifest_v1.json"
receipt_schema="${script_dir}/hdl_run_receipt_v1.schema.json"
case_matrix="${script_dir}/hdl_expected_case_matrix_v1.json"
receipt_validator="${script_dir}/validate_hdl_receipt.sh"
flake_nix="${repo_root}/flake.nix"
flake_lock="${repo_root}/flake.lock"
stamp="$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || printf 'timestamp-unavailable')"
out_root="${HDL_OUT_DIR:-${repo_root}/target/hdl-qualification/hdc}"
out_dir="${out_root}/${stamp}-pid${BASHPID}"
mkdir -p "${out_dir}" || { printf 'cannot create output directory: %s\n' "${out_dir}" >&2; exit 2; }

missing=()
for tool in verilator jq sha256sum git make c++ head grep; do
  if ! command -v "${tool}" >/dev/null 2>&1; then missing+=("${tool}"); fi
done

hash_file() {
  if [[ -f "$1" ]] && command -v sha256sum >/dev/null 2>&1; then
    local hash_line
    hash_line="$(sha256sum "$1")" || { printf 'unavailable'; return; }
    printf '%s' "${hash_line%% *}"
  else
    printf 'unavailable'
  fi
}

rtl_xor_sha="$(hash_file "${rtl_xor}")"
tb_xor_sha="$(hash_file "${tb_xor}")"
rtl_hamming_sha="$(hash_file "${rtl_hamming}")"
tb_hamming_sha="$(hash_file "${tb_hamming}")"
runner_sha="$(hash_file "${script_dir}/run_hdl_smoke.sh")"
manifest_sha="$(hash_file "${manifest}")"
receipt_schema_sha="$(hash_file "${receipt_schema}")"
case_matrix_sha="$(hash_file "${case_matrix}")"
receipt_validator_sha="$(hash_file "${receipt_validator}")"
flake_sha="$(hash_file "${flake_nix}")"
flake_lock_sha="$(hash_file "${flake_lock}")"
git_head="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || printf 'unavailable')"

for required_file in "${rtl_xor}" "${tb_xor}" "${rtl_hamming}" "${tb_hamming}" "${manifest}" "${receipt_schema}" "${case_matrix}" "${receipt_validator}" "${flake_nix}" "${flake_lock}"; do
  if [[ ! -f "${required_file}" ]]; then
    printf 'Missing required file: %s\n' "${required_file}" >&2
    exit 2
  fi
done

if ! command -v jq >/dev/null 2>&1; then
  printf 'status=unqualified_tool_missing\nmissing_tool=jq\n' > "${out_dir}/receipt.txt"
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
    --arg out_dir "${out_dir}" \
    --arg timestamp_utc "${stamp}" \
    --arg git_head "${git_head}" \
    --arg verilator_version "${verilator_version}" \
    --argjson missing_tools "${missing_json}" \
    --arg rtl_xor_sha256 "${rtl_xor_sha}" \
    --arg tb_xor_sha256 "${tb_xor_sha}" \
    --arg rtl_hamming_sha256 "${rtl_hamming_sha}" \
    --arg tb_hamming_sha256 "${tb_hamming_sha}" \
    --arg runner_sha256 "${runner_sha}" \
    --arg manifest_sha256 "${manifest_sha}" \
    --arg receipt_schema_sha256 "${receipt_schema_sha}" \
    --arg case_matrix_sha256 "${case_matrix_sha}" \
    --arg receipt_validator_sha256 "${receipt_validator_sha}" \
    --arg flake_sha256 "${flake_sha}" \
    --arg flake_lock_sha256 "${flake_lock_sha}" \
    '{schema_version:"1.1", qualification:"hdc_rtl_correctness_sweep", status:$status, timestamp_utc:$timestamp_utc, git_head:$git_head, verilator_version:$verilator_version, missing_tools:$missing_tools, source_hashes:{rtl_bind_xor:$rtl_xor_sha256,tb_bind_xor:$tb_xor_sha256,rtl_hamming_count:$rtl_hamming_sha256,tb_hamming_count:$tb_hamming_sha256,runner:$runner_sha256,manifest:$manifest_sha256,receipt_schema:$receipt_schema_sha256,case_matrix:$case_matrix_sha256,receipt_validator:$receipt_validator_sha256,flake:$flake_sha256,flake_lock:$flake_lock_sha256}, widths_tested:[], architectures:["xor","linear_popcount","balanced_popcount"], case_results:[], output_directory:$out_dir, claim_ceiling:"No RTL simulation, formal proof, synthesis/PPA, FPGA observation, physical energy or SOTA claim."}' \
    > "${out_dir}/receipt.json"
  printf 'UNQUALIFIED: missing tools: %s\nReceipt: %s/receipt.json\n' "${missing[*]}" "${out_dir}" >&2
  exit 2
fi

case_results="[]"
overall_status="simulation_smoke_pass"
for width in 64 128 256 512; do
  for arch in xor linear_popcount balanced_popcount; do
    case "${arch}" in
      xor) module="hdc_bind_xor"; top="tb_hdc_bind_xor"; rtl="${rtl_xor}"; tb="${tb_xor}"; pass_marker="PASS: hdc_bind_xor smoke and backpressure tests completed";;
      linear_popcount) module="hdc_hamming_count"; top="tb_hdc_hamming_count"; rtl="${rtl_hamming}"; tb="${tb_hamming}"; pass_marker="PASS: hdc_hamming_count smoke and backpressure tests completed width=${width} balanced_tree=0";;
      balanced_popcount) module="hdc_hamming_count"; top="tb_hdc_hamming_count"; rtl="${rtl_hamming}"; tb="${tb_hamming}"; pass_marker="PASS: hdc_hamming_count smoke and backpressure tests completed width=${width} balanced_tree=1";;
      *) overall_status="failed"; continue;;
    esac

    if [[ "${arch}" == "balanced_popcount" ]]; then balanced=1; else balanced=0; fi
    case_id="${arch//_/-}-${width}"
    case_dir="${out_dir}/width-${width}/${arch}"
    mkdir -p "${case_dir}" || { overall_status="failed"; continue; }

    params=("-GDATA_W=${width}")
    if [[ "${module}" == "hdc_hamming_count" ]]; then params+=("-GBALANCED_TREE=${balanced}"); fi

    lint_rc=125
    build_rc=125
    simulation_rc=125
    simulation_marker="false"

    verilator --lint-only --timing -Wall --top-module "${top}" "${params[@]}" \
      "${rtl}" "${tb}" > "${case_dir}/lint.log" 2>&1
    lint_rc=$?

    if [[ "${lint_rc}" -eq 0 ]]; then
      verilator --binary --timing --assert -Wall --top-module "${top}" "${params[@]}" \
        --Mdir "${case_dir}/obj_dir" "${rtl}" "${tb}" > "${case_dir}/build.log" 2>&1
      build_rc=$?
      if [[ "${build_rc}" -eq 0 ]]; then
        "${case_dir}/obj_dir/V${top}" > "${case_dir}/simulation.log" 2>&1
        simulation_rc=$?
        if [[ "${simulation_rc}" -eq 0 ]] && grep -Fq "${pass_marker}" "${case_dir}/simulation.log"; then
          simulation_marker="true"
        fi
      fi
    fi

    case_status="failed"
    if [[ "${lint_rc}" -eq 0 && "${build_rc}" -eq 0 && "${simulation_rc}" -eq 0 && "${simulation_marker}" == "true" ]]; then
      case_status="simulation_smoke_pass"
    else
      overall_status="failed"
    fi

    for log_file in "${case_dir}"/*.log; do
      [[ -f "${log_file}" ]] || continue
      sha256sum "${log_file}" >> "${out_dir}/log-sha256.txt"
    done

    case_result="$(jq -n \
      --arg case_id "${case_id}" \
      --arg module "${module}" \
      --arg architecture "${arch}" \
      --arg width "${width}" \
      --arg status "${case_status}" \
      --arg lint_rc "${lint_rc}" \
      --arg build_rc "${build_rc}" \
      --arg simulation_rc "${simulation_rc}" \
      --argjson simulation_marker "${simulation_marker}" \
      '{case_id:$case_id, module:$module, architecture:$architecture, datapath_width_bits:($width|tonumber), status:$status, lint_exit_code:($lint_rc|tonumber), build_exit_code:($build_rc|tonumber), simulation_exit_code:($simulation_rc|tonumber), expected_pass_marker_seen:$simulation_marker}')"
    case_results="$(jq -cn --argjson existing "${case_results}" --argjson next "${case_result}" '$existing + [$next]')"
    printf 'width=%s architecture=%s status=%s lint=%s build=%s sim=%s\n' "${width}" "${arch}" "${case_status}" "${lint_rc}" "${build_rc}" "${simulation_rc}"
  done
done

jq -n \
  --arg status "${overall_status}" \
  --arg timestamp_utc "${stamp}" \
  --arg git_head "${git_head}" \
  --arg verilator_version "${verilator_version}" \
  --arg out_dir "${out_dir}" \
  --arg rtl_xor_sha256 "${rtl_xor_sha}" \
  --arg tb_xor_sha256 "${tb_xor_sha}" \
  --arg rtl_hamming_sha256 "${rtl_hamming_sha}" \
  --arg tb_hamming_sha256 "${tb_hamming_sha}" \
  --arg runner_sha256 "${runner_sha}" \
  --arg manifest_sha256 "${manifest_sha}" \
  --arg receipt_schema_sha256 "${receipt_schema_sha}" \
  --arg case_matrix_sha256 "${case_matrix_sha}" \
  --arg receipt_validator_sha256 "${receipt_validator_sha}" \
  --arg flake_sha256 "${flake_sha}" \
  --arg flake_lock_sha256 "${flake_lock_sha}" \
  --argjson missing_tools "${missing_json}" \
  --argjson case_results "${case_results}" \
  '{schema_version:"1.1", qualification:"hdc_rtl_correctness_sweep", status:$status, timestamp_utc:$timestamp_utc, git_head:$git_head, verilator_version:$verilator_version, widths_tested:[64,128,256,512], architectures:["xor","linear_popcount","balanced_popcount"], source_hashes:{rtl_bind_xor:$rtl_xor_sha256,tb_bind_xor:$tb_xor_sha256,rtl_hamming_count:$rtl_hamming_sha256,tb_hamming_count:$tb_hamming_sha256,runner:$runner_sha256,manifest:$manifest_sha256,receipt_schema:$receipt_schema_sha256,case_matrix:$case_matrix_sha256,receipt_validator:$receipt_validator_sha256,flake:$flake_sha256,flake_lock:$flake_lock_sha256}, missing_tools:$missing_tools, case_results:$case_results, output_directory:$out_dir, claim_ceiling:"A pass qualifies only these word-level RTL/testbench cases. It is not formal proof, synthesis/PPA, FPGA observation, physical energy, throughput superiority or product/SOTA qualification."}' \
  > "${out_dir}/receipt.json"

printf 'status=%s\nreceipt=%s/receipt.json\n' "${overall_status}" "${out_dir}/receipt.json"
if [[ "${overall_status}" != "simulation_smoke_pass" ]]; then exit 1; fi
