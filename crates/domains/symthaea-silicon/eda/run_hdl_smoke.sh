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
oracle_generator="${silicon_dir}/src/bin/hdc_rtl_oracle.rs"
silicon_cargo_manifest="${silicon_dir}/Cargo.toml"
canonical_binary_hv="${repo_root}/crates/core/symthaea-core/src/hdc/binary_hv.rs"
cargo_lock="${repo_root}/Cargo.lock"
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
for tool in verilator jq sha256sum git make c++ cargo rustc head grep; do
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
oracle_generator_sha="$(hash_file "${oracle_generator}")"
silicon_cargo_manifest_sha="$(hash_file "${silicon_cargo_manifest}")"
canonical_binary_hv_sha="$(hash_file "${canonical_binary_hv}")"
cargo_lock_sha="$(hash_file "${cargo_lock}")"
oracle_fixture_receipt_sha="unavailable"
git_head="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || printf 'unavailable')"

for required_file in "${rtl_xor}" "${tb_xor}" "${rtl_hamming}" "${tb_hamming}" "${oracle_generator}" "${silicon_cargo_manifest}" "${canonical_binary_hv}" "${cargo_lock}" "${manifest}" "${receipt_schema}" "${case_matrix}" "${receipt_validator}" "${flake_nix}" "${flake_lock}"; do
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
    --arg oracle_generator_sha256 "${oracle_generator_sha}" \
    --arg silicon_cargo_manifest_sha256 "${silicon_cargo_manifest_sha}" \
    --arg canonical_binary_hv_sha256 "${canonical_binary_hv_sha}" \
    --arg cargo_lock_sha256 "${cargo_lock_sha}" \
    --arg oracle_fixture_receipt_sha256 "${oracle_fixture_receipt_sha}" \
    '{schema_version:"1.1", qualification:"hdc_rtl_correctness_sweep", status:$status, timestamp_utc:$timestamp_utc, git_head:$git_head, verilator_version:$verilator_version, missing_tools:$missing_tools, source_hashes:{rtl_bind_xor:$rtl_xor_sha256,tb_bind_xor:$tb_xor_sha256,rtl_hamming_count:$rtl_hamming_sha256,tb_hamming_count:$tb_hamming_sha256,runner:$runner_sha256,manifest:$manifest_sha256,receipt_schema:$receipt_schema_sha256,case_matrix:$case_matrix_sha256,receipt_validator:$receipt_validator_sha256,oracle_generator:$oracle_generator_sha256,silicon_cargo_manifest:$silicon_cargo_manifest_sha256,canonical_binary_hv_source:$canonical_binary_hv_sha256,cargo_lock:$cargo_lock_sha256,oracle_fixture_receipt:$oracle_fixture_receipt_sha256,flake:$flake_sha256,flake_lock:$flake_lock_sha256, widths_tested:[], architectures:["xor","linear_popcount","balanced_popcount"], case_results:[], output_directory:$out_dir, claim_ceiling:"No RTL simulation, formal proof, synthesis/PPA, FPGA observation, physical energy or SOTA claim."}' \
    > "${out_dir}/receipt.json"
  printf 'UNQUALIFIED: missing tools: %s\nReceipt: %s/receipt.json\n' "${missing[*]}" "${out_dir}" >&2
  exit 2
fi

identity_failure=""
if [[ ! "${git_head}" =~ ^[a-f0-9]{40}$ ]]; then identity_failure="git_head"; fi
for digest in "${rtl_xor_sha}" "${tb_xor_sha}" "${rtl_hamming_sha}" "${tb_hamming_sha}" "${runner_sha}" "${manifest_sha}" "${receipt_schema_sha}" "${case_matrix_sha}" "${receipt_validator_sha}" "${flake_sha}" "${flake_lock_sha}" "${oracle_generator_sha}" "${silicon_cargo_manifest_sha}" "${canonical_binary_hv_sha}" "${cargo_lock_sha}"; do
  if [[ ! "${digest}" =~ ^[a-f0-9]{64}$ ]]; then identity_failure="source_hash"; break; fi
done
if [[ -n "${identity_failure}" ]]; then
  jq -n \
    --arg status "failed" \
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
    --arg oracle_generator_sha256 "${oracle_generator_sha}" \
    --arg silicon_cargo_manifest_sha256 "${silicon_cargo_manifest_sha}" \
    --arg canonical_binary_hv_sha256 "${canonical_binary_hv_sha}" \
    --arg cargo_lock_sha256 "${cargo_lock_sha}" \
    --arg oracle_fixture_receipt_sha256 "${oracle_fixture_receipt_sha}" \
    '{schema_version:"1.1", qualification:"hdc_rtl_correctness_sweep", status:$status, timestamp_utc:$timestamp_utc, git_head:$git_head, verilator_version:$verilator_version, missing_tools:[], source_hashes:{rtl_bind_xor:$rtl_xor_sha256,tb_bind_xor:$tb_xor_sha256,rtl_hamming_count:$rtl_hamming_sha256,tb_hamming_count:$tb_hamming_sha256,runner:$runner_sha256,manifest:$manifest_sha256,receipt_schema:$receipt_schema_sha256,case_matrix:$case_matrix_sha256,receipt_validator:$receipt_validator_sha256,oracle_generator:$oracle_generator_sha256,silicon_cargo_manifest:$silicon_cargo_manifest_sha256,canonical_binary_hv_source:$canonical_binary_hv_sha256,cargo_lock:$cargo_lock_sha256,oracle_fixture_receipt:$oracle_fixture_receipt_sha256,flake:$flake_sha256,flake_lock:$flake_lock_sha256, widths_tested:[], architectures:["xor","linear_popcount","balanced_popcount"], case_results:[], output_directory:$out_dir, claim_ceiling:"No test started: Git checkout identity or a required source/configuration hash is unavailable."}' \
    > "${out_dir}/receipt.json"
  printf 'UNQUALIFIED: Git checkout identity or source hash unavailable; receipt: %s/receipt.json\n' "${out_dir}" >&2
  exit 2
fi

# Generate expected words from canonical Symthaea BinaryHV before simulating
# the datapath. This keeps the oracle implementation outside the RTL testbench.
oracle_dir="${out_dir}/oracle"
mkdir -p "${oracle_dir}"
if ! (cd "${repo_root}" && cargo run --locked --offline -p symthaea-silicon --bin hdc_rtl_oracle -- --output-dir "${oracle_dir}" --expected-git-head "${git_head}") > "${out_dir}/oracle-generator.log" 2>&1; then
  oracle_fixture_receipt_sha="$(hash_file "${oracle_dir}/oracle_receipt.json")"
  jq -n \
    --arg status "failed" \
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
    --arg oracle_generator_sha256 "${oracle_generator_sha}" \
    --arg silicon_cargo_manifest_sha256 "${silicon_cargo_manifest_sha}" \
    --arg canonical_binary_hv_sha256 "${canonical_binary_hv_sha}" \
    --arg cargo_lock_sha256 "${cargo_lock_sha}" \
    --arg oracle_fixture_receipt_sha256 "${oracle_fixture_receipt_sha}" \
    --arg flake_sha256 "${flake_sha}" \
    --arg flake_lock_sha256 "${flake_lock_sha}" \
    '{schema_version:"1.1", qualification:"hdc_rtl_correctness_sweep", status:$status, timestamp_utc:$timestamp_utc, git_head:$git_head, verilator_version:$verilator_version, missing_tools:[], source_hashes:{rtl_bind_xor:$rtl_xor_sha256,tb_bind_xor:$tb_xor_sha256,rtl_hamming_count:$rtl_hamming_sha256,tb_hamming_count:$tb_hamming_sha256,runner:$runner_sha256,manifest:$manifest_sha256,receipt_schema:$receipt_schema_sha256,case_matrix:$case_matrix_sha256,receipt_validator:$receipt_validator_sha256,oracle_generator:$oracle_generator_sha256,silicon_cargo_manifest:$silicon_cargo_manifest_sha256,canonical_binary_hv_source:$canonical_binary_hv_sha256,cargo_lock:$cargo_lock_sha256,oracle_fixture_receipt:$oracle_fixture_receipt_sha256,flake:$flake_sha256,flake_lock:$flake_lock_sha256}, widths_tested:[], architectures:["xor","linear_popcount","balanced_popcount"], case_results:[], output_directory:$out_dir, claim_ceiling:"No RTL simulation started: canonical BinaryHV software-oracle fixture generation failed."}' \
    > "${out_dir}/receipt.json"
  sha256sum "${out_dir}/oracle-generator.log" >> "${out_dir}/log-sha256.txt"
  printf 'FAIL: canonical BinaryHV oracle generation failed; see %s/oracle-generator.log\n' "${out_dir}" >&2
  exit 1
fi

oracle_receipt="${oracle_dir}/oracle_receipt.json"
oracle_fixture_receipt_sha="$(hash_file "${oracle_receipt}")"
oracle_git_head="$(jq -r '.git_head // empty' "${oracle_receipt}" 2>/dev/null || true)"
oracle_source_sha="$(jq -r '.canonical_source_sha256 // empty' "${oracle_receipt}" 2>/dev/null || true)"
oracle_lock_sha="$(jq -r '.cargo_lock_sha256 // empty' "${oracle_receipt}" 2>/dev/null || true)"
oracle_valid=true
if [[ "${oracle_git_head}" != "${git_head}" || "${oracle_source_sha}" != "${canonical_binary_hv_sha}" || "${oracle_lock_sha}" != "${cargo_lock_sha}" || ! "${oracle_fixture_receipt_sha}" =~ ^[a-f0-9]{64}$ ]]; then
  oracle_valid=false
fi

declare -A oracle_word_counts
declare -A oracle_xor_files
declare -A oracle_hamming_files
for width in 64 128 256 512; do
  words="$(jq -r --argjson width "${width}" '.widths[] | select(.datapath_width_bits == $width) | .word_cases' "${oracle_receipt}" 2>/dev/null || true)"
  xor_name="$(jq -r --argjson width "${width}" '.widths[] | select(.datapath_width_bits == $width) | .xor_mem_file' "${oracle_receipt}" 2>/dev/null || true)"
  xor_expected_sha="$(jq -r --argjson width "${width}" '.widths[] | select(.datapath_width_bits == $width) | .xor_mem_sha256' "${oracle_receipt}" 2>/dev/null || true)"
  hamming_name="$(jq -r --argjson width "${width}" '.widths[] | select(.datapath_width_bits == $width) | .hamming_mem_file' "${oracle_receipt}" 2>/dev/null || true)"
  hamming_expected_sha="$(jq -r --argjson width "${width}" '.widths[] | select(.datapath_width_bits == $width) | .hamming_mem_sha256' "${oracle_receipt}" 2>/dev/null || true)"
  xor_actual_sha="$(hash_file "${oracle_dir}/${xor_name}")"
  hamming_actual_sha="$(hash_file "${oracle_dir}/${hamming_name}")"
  if [[ ! "${words}" =~ ^[1-9][0-9]*$ || "${xor_name}" != "hdc_oracle_xor_w${width}.mem" || "${hamming_name}" != "hdc_oracle_hamming_w${width}.mem" || "${xor_actual_sha}" != "${xor_expected_sha}" || "${hamming_actual_sha}" != "${hamming_expected_sha}" ]]; then
    oracle_valid=false
  else
    oracle_word_counts["${width}"]="${words}"
    oracle_xor_files["${width}"]="${oracle_dir}/${xor_name}"
    oracle_hamming_files["${width}"]="${oracle_dir}/${hamming_name}"
  fi
done

if [[ "${oracle_valid}" != "true" ]]; then
  printf 'FAIL: canonical oracle receipt, Git/source identity, or .mem hashes mismatch\n' > "${out_dir}/oracle-validation.log"
  jq -n \
    --arg status "failed" \
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
    --arg oracle_generator_sha256 "${oracle_generator_sha}" \
    --arg silicon_cargo_manifest_sha256 "${silicon_cargo_manifest_sha}" \
    --arg canonical_binary_hv_sha256 "${canonical_binary_hv_sha}" \
    --arg cargo_lock_sha256 "${cargo_lock_sha}" \
    --arg oracle_fixture_receipt_sha256 "${oracle_fixture_receipt_sha}" \
    --arg flake_sha256 "${flake_sha}" \
    --arg flake_lock_sha256 "${flake_lock_sha}" \
    '{schema_version:"1.1", qualification:"hdc_rtl_correctness_sweep", status:$status, timestamp_utc:$timestamp_utc, git_head:$git_head, verilator_version:$verilator_version, missing_tools:[], source_hashes:{rtl_bind_xor:$rtl_xor_sha256,tb_bind_xor:$tb_xor_sha256,rtl_hamming_count:$rtl_hamming_sha256,tb_hamming_count:$tb_hamming_sha256,runner:$runner_sha256,manifest:$manifest_sha256,receipt_schema:$receipt_schema_sha256,case_matrix:$case_matrix_sha256,receipt_validator:$receipt_validator_sha256,oracle_generator:$oracle_generator_sha256,silicon_cargo_manifest:$silicon_cargo_manifest_sha256,canonical_binary_hv_source:$canonical_binary_hv_sha256,cargo_lock:$cargo_lock_sha256,oracle_fixture_receipt:$oracle_fixture_receipt_sha256,flake:$flake_sha256,flake_lock:$flake_lock_sha256}, widths_tested:[], architectures:["xor","linear_popcount","balanced_popcount"], case_results:[], output_directory:$out_dir, claim_ceiling:"No RTL simulation started: software-oracle fixture receipt or file hashes were invalid."}' \
    > "${out_dir}/receipt.json"
  sha256sum "${out_dir}/oracle-validation.log" >> "${out_dir}/log-sha256.txt"
  printf 'FAIL: canonical oracle evidence did not validate; receipt: %s/receipt.json\n' "${out_dir}" >&2
  exit 1
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
  --arg oracle_generator_sha256 "${oracle_generator_sha}" \
  --arg silicon_cargo_manifest_sha256 "${silicon_cargo_manifest_sha}" \
  --arg canonical_binary_hv_sha256 "${canonical_binary_hv_sha}" \
  --arg cargo_lock_sha256 "${cargo_lock_sha}" \
  --arg oracle_fixture_receipt_sha256 "${oracle_fixture_receipt_sha}" \
  --argjson missing_tools "${missing_json}" \
  --argjson case_results "${case_results}" \
  '{schema_version:"1.1", qualification:"hdc_rtl_correctness_sweep", status:$status, timestamp_utc:$timestamp_utc, git_head:$git_head, verilator_version:$verilator_version, widths_tested:[64,128,256,512], architectures:["xor","linear_popcount","balanced_popcount"], source_hashes:{rtl_bind_xor:$rtl_xor_sha256,tb_bind_xor:$tb_xor_sha256,rtl_hamming_count:$rtl_hamming_sha256,tb_hamming_count:$tb_hamming_sha256,runner:$runner_sha256,manifest:$manifest_sha256,receipt_schema:$receipt_schema_sha256,case_matrix:$case_matrix_sha256,receipt_validator:$receipt_validator_sha256,oracle_generator:$oracle_generator_sha256,silicon_cargo_manifest:$silicon_cargo_manifest_sha256,canonical_binary_hv_source:$canonical_binary_hv_sha256,cargo_lock:$cargo_lock_sha256,oracle_fixture_receipt:$oracle_fixture_receipt_sha256,flake:$flake_sha256,flake_lock:$flake_lock_sha256, missing_tools:$missing_tools, case_results:$case_results, output_directory:$out_dir, claim_ceiling:"A pass qualifies only these word-level RTL/testbench cases. It is not formal proof, synthesis/PPA, FPGA observation, physical energy, throughput superiority or product/SOTA qualification."}' \
  > "${out_dir}/receipt.json"

printf 'status=%s\nreceipt=%s/receipt.json\n' "${overall_status}" "${out_dir}/receipt.json"
if [[ "${overall_status}" != "simulation_smoke_pass" ]]; then exit 1; fi
