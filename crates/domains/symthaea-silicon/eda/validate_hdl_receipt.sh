#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Independent fail-closed validator for a claimed passing HDC RTL sweep receipt.
# Checks expected case coverage and source/oracle identities; not hardware attestation.

set -uo pipefail

if [[ "${#}" -lt 1 || "${#}" -gt 2 ]]; then
  printf 'usage: %s <receipt.json> [expected-case-matrix.json]\n' "${0##*/}" >&2
  exit 2
fi

receipt="${1}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
silicon_dir="$(cd -- "${script_dir}/.." && pwd)"
repo_root="$(cd -- "${silicon_dir}/../../.." && pwd)"
matrix="${2:-${script_dir}/hdl_expected_case_matrix_v1.json}"
schema="${script_dir}/hdl_run_receipt_v1.schema.json"
canonical_binary_hv="${repo_root}/crates/core/symthaea-core/src/hdc/binary_hv.rs"
silicon_cargo_manifest="${silicon_dir}/Cargo.toml"
cargo_lock="${repo_root}/Cargo.lock"

for tool in jq sha256sum git; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    printf 'UNQUALIFIED: required validator tool missing: %s\n' "${tool}" >&2
    exit 2
  fi
done
if [[ ! -f "${receipt}" || ! -f "${matrix}" || ! -f "${schema}" ]]; then
  printf 'UNQUALIFIED: receipt, expected-case matrix or receipt schema not found\n' >&2
  exit 2
fi
for json_file in "${receipt}" "${matrix}" "${schema}"; do
  if ! jq -e . "${json_file}" >/dev/null 2>&1; then
    printf 'FAIL: invalid JSON: %s\n' "${json_file}" >&2
    exit 1
  fi
done

hash_file() {
  if [[ -f "$1" ]]; then
    local hash_line
    hash_line="$(sha256sum "$1")" || { printf 'unavailable'; return; }
    printf '%s' "${hash_line%% *}"
  else
    printf 'unavailable'
  fi
}

rtl_xor_sha="$(hash_file "${silicon_dir}/rtl/hdc_bind_xor.sv")"
tb_xor_sha="$(hash_file "${silicon_dir}/rtl/tb_hdc_bind_xor.sv")"
rtl_hamming_sha="$(hash_file "${silicon_dir}/rtl/hdc_hamming_count.sv")"
tb_hamming_sha="$(hash_file "${silicon_dir}/rtl/tb_hdc_hamming_count.sv")"
runner_sha="$(hash_file "${script_dir}/run_hdl_smoke.sh")"
manifest_sha="$(hash_file "${script_dir}/hdl_toolchain_manifest_v1.json")"
receipt_schema_sha="$(hash_file "${schema}")"
case_matrix_sha="$(hash_file "${matrix}")"
receipt_validator_sha="$(hash_file "${script_dir}/validate_hdl_receipt.sh")"
oracle_generator_sha="$(hash_file "${silicon_dir}/src/bin/hdc_rtl_oracle.rs")"
silicon_cargo_manifest_sha="$(hash_file "${silicon_cargo_manifest}")"
canonical_binary_hv_sha="$(hash_file "${canonical_binary_hv}")"
cargo_lock_sha="$(hash_file "${cargo_lock}")"
flake_sha="$(hash_file "${repo_root}/flake.nix")"
flake_lock_sha="$(hash_file "${repo_root}/flake.lock")"
git_head="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || printf 'unavailable')"

output_directory="$(jq -r '.output_directory // empty' "${receipt}")"
expected_output_prefix="${repo_root}/target/hdl-qualification/hdc/"
case "${output_directory}" in
  "${expected_output_prefix}"*) ;;
  *) printf 'FAIL: output_directory is outside the approved target path\n' >&2; exit 1 ;;
esac

oracle_receipt="${output_directory}/oracle/oracle_receipt.json"
oracle_receipt_sha="$(hash_file "${oracle_receipt}")"
if [[ ! "${oracle_receipt_sha}" =~ ^[a-f0-9]{64}$ ]]; then
  printf 'FAIL: software-oracle receipt missing or unreadable\n' >&2
  exit 1
fi
oracle_git_head="$(jq -r '.git_head // empty' "${oracle_receipt}" 2>/dev/null || true)"
oracle_source_sha="$(jq -r '.canonical_source_sha256 // empty' "${oracle_receipt}" 2>/dev/null || true)"
oracle_lock_sha="$(jq -r '.cargo_lock_sha256 // empty' "${oracle_receipt}" 2>/dev/null || true)"
if [[ "${oracle_git_head}" != "${git_head}" || "${oracle_source_sha}" != "${canonical_binary_hv_sha}" || "${oracle_lock_sha}" != "${cargo_lock_sha}" ]]; then
  printf 'FAIL: software-oracle Git/source/lock identity mismatch\n' >&2
  exit 1
fi

for width in 64 128 256 512; do
  oracle_width_count="$(jq -r --argjson width "${width}" '.widths[] | select(.datapath_width_bits == $width) | .word_cases' "${oracle_receipt}" 2>/dev/null || true)"
  xor_name="$(jq -r --argjson width "${width}" '.widths[] | select(.datapath_width_bits == $width) | .xor_mem_file' "${oracle_receipt}" 2>/dev/null || true)"
  xor_expected_sha="$(jq -r --argjson width "${width}" '.widths[] | select(.datapath_width_bits == $width) | .xor_mem_sha256' "${oracle_receipt}" 2>/dev/null || true)"
  hamming_name="$(jq -r --argjson width "${width}" '.widths[] | select(.datapath_width_bits == $width) | .hamming_mem_file' "${oracle_receipt}" 2>/dev/null || true)"
  hamming_expected_sha="$(jq -r --argjson width "${width}" '.widths[] | select(.datapath_width_bits == $width) | .hamming_mem_sha256' "${oracle_receipt}" 2>/dev/null || true)"
  if [[ "${oracle_width_count}" != "$((6 * (16384 / width)))" ||
        "${xor_name}" != "hdc_oracle_xor_w${width}.mem" ||
        "${hamming_name}" != "hdc_oracle_hamming_w${width}.mem" ||
        "$(hash_file "${output_directory}/oracle/${xor_name}")" != "${xor_expected_sha}" ||
        "$(hash_file "${output_directory}/oracle/${hamming_name}")" != "${hamming_expected_sha}" ]]; then
    printf 'FAIL: software-oracle fixture mismatch at width %s\n' "${width}" >&2
    exit 1
  fi
done

if jq -e --slurpfile matrix "${matrix}" \
  --arg git_head "${git_head}" \
  --arg rtl_xor_sha "${rtl_xor_sha}" \
  --arg tb_xor_sha "${tb_xor_sha}" \
  --arg rtl_hamming_sha "${rtl_hamming_sha}" \
  --arg tb_hamming_sha "${tb_hamming_sha}" \
  --arg runner_sha "${runner_sha}" \
  --arg manifest_sha "${manifest_sha}" \
  --arg receipt_schema_sha "${receipt_schema_sha}" \
  --arg case_matrix_sha "${case_matrix_sha}" \
  --arg receipt_validator_sha "${receipt_validator_sha}" \
  --arg oracle_generator_sha "${oracle_generator_sha}" \
  --arg silicon_cargo_manifest_sha "${silicon_cargo_manifest_sha}" \
  --arg canonical_binary_hv_sha "${canonical_binary_hv_sha}" \
  --arg cargo_lock_sha "${cargo_lock_sha}" \
  --arg oracle_receipt_sha "${oracle_receipt_sha}" \
  --arg flake_sha "${flake_sha}" \
  --arg flake_lock_sha "${flake_lock_sha}" '
  . as $r
  | $matrix[0] as $m
  | ($r.schema_version == "1.1")
    and ($r.qualification == "hdc_rtl_correctness_sweep")
    and ($r.status == "simulation_smoke_pass")
    and ($r.git_head == $git_head)
    and (($r.git_head | type) == "string" and ($r.git_head | test("^[a-f0-9]{40}$")))
    and (($r.verilator_version | type) == "string" and ($r.verilator_version | length) > 0 and $r.verilator_version != "unavailable")
    and ($r.missing_tools == [])
    and ($r.widths_tested == $m.widths_bits)
    and ($r.architectures == $m.architectures)
    and (($r.case_results | length) == $m.expected_case_count)
    and (($r.case_results | map(.case_id) | unique | length) == $m.expected_case_count)
    and all($m.cases[];
      . as $expected
      | [$r.case_results[] |
          select(.case_id == $expected.case_id
            and .module == $expected.module
            and .architecture == $expected.architecture
            and .datapath_width_bits == $expected.datapath_width_bits)]
        | length == 1)
    and all($r.case_results[];
      .status == "simulation_smoke_pass"
        and .lint_exit_code == 0
        and .build_exit_code == 0
        and .simulation_exit_code == 0
        and .expected_pass_marker_seen == true
        and .software_oracle_marker_seen == true)
    and ($r.source_hashes.rtl_bind_xor == $rtl_xor_sha)
    and ($r.source_hashes.tb_bind_xor == $tb_xor_sha)
    and ($r.source_hashes.rtl_hamming_count == $rtl_hamming_sha)
    and ($r.source_hashes.tb_hamming_count == $tb_hamming_sha)
    and ($r.source_hashes.runner == $runner_sha)
    and ($r.source_hashes.manifest == $manifest_sha)
    and ($r.source_hashes.receipt_schema == $receipt_schema_sha)
    and ($r.source_hashes.case_matrix == $case_matrix_sha)
    and ($r.source_hashes.receipt_validator == $receipt_validator_sha)
    and ($r.source_hashes.oracle_generator == $oracle_generator_sha)
    and ($r.source_hashes.silicon_cargo_manifest == $silicon_cargo_manifest_sha)
    and ($r.source_hashes.canonical_binary_hv_source == $canonical_binary_hv_sha)
    and ($r.source_hashes.cargo_lock == $cargo_lock_sha)
    and ($r.source_hashes.oracle_fixture_receipt == $oracle_receipt_sha)
    and ($r.source_hashes.flake == $flake_sha)
    and ($r.source_hashes.flake_lock == $flake_lock_sha)
    and (($r.output_directory | type) == "string" and ($r.output_directory | length) > 0)
  ' "${receipt}" >/dev/null; then
  printf 'PASS: receipt covers all %s expected passing cases and matches current source/configuration/software-oracle hashes.\n' "$(jq -r '.expected_case_count' "${matrix}")"
  printf 'Claim ceiling: this does not prove formal correctness, PPA, FPGA/ASIC behavior, measured energy or SOTA.\n'
  exit 0
else
  printf 'FAIL: receipt is missing, duplicate, mismatched or failed; it does not qualify the sweep.\n' >&2
  exit 1
fi
