#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Independent fail-closed validator for a claimed passing HDC RTL sweep receipt.
# It checks contract completeness; it does not authenticate tools or hardware.

set -uo pipefail

if [[ "$${#}" -lt 1 || "$${#}" -gt 2 ]]; then
  printf 'usage: %s <receipt.json> [expected-case-matrix.json]\n' "$${0##*/}" >&2
  exit 2
fi

receipt="$${1}"
script_dir="$(cd -- "$(dirname -- "$${BASH_SOURCE[0]}")" && pwd)"
matrix="${2:-${script_dir}/hdl_expected_case_matrix_v1.json}"

if ! command -v jq >/dev/null 2>&1; then
  printf 'UNQUALIFIED: jq is required to validate the receipt\n' >&2
  exit 2
fi
if [[ ! -f "$${receipt}" || ! -f "$${matrix}" ]]; then
  printf 'UNQUALIFIED: receipt or expected-case matrix not found\n' >&2
  exit 2
fi
if ! jq -e . "$${receipt}" >/dev/null 2>&1; then
  printf 'FAIL: receipt is not valid JSON\n' >&2
  exit 1
fi
if ! jq -e . "$${matrix}" >/dev/null 2>&1; then
  printf 'FAIL: expected-case matrix is not valid JSON\n' >&2
  exit 1
fi

if jq -e --slurpfile matrix "$${matrix}" '
  . as $r
  | $matrix[0] as $m
  | ($r.schema_version == "1.1")
    and ($r.qualification == "hdc_rtl_correctness_sweep")
    and ($r.status == "simulation_smoke_pass")
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
        and .expected_pass_marker_seen == true)
    and ([$r.source_hashes.rtl_bind_xor, $r.source_hashes.tb_bind_xor,
          $r.source_hashes.rtl_hamming_count, $r.source_hashes.tb_hamming_count,
          $r.source_hashes.runner, $r.source_hashes.manifest,
          $r.source_hashes.flake, $r.source_hashes.flake_lock]
         | all(.[]; type == "string" and test("^[a-f0-9]{64}$")))
    and (($r.output_directory | type) == "string" and ($r.output_directory | length) > 0)
  ' "$${receipt}" >/dev/null; then
  printf 'PASS: receipt structurally covers all %s expected passing cases.\n' "$(jq -r '.expected_case_count' "$${matrix}")"
  printf 'Claim ceiling: the receipt still does not prove formal correctness, PPA, FPGA/ASIC behavior, measured energy or SOTA.\n'
  exit 0
else
  printf 'FAIL: receipt is missing, duplicate, mismatched or failed; it does not qualify the sweep.\n' >&2
  exit 1
fi
