#!/usr/bin/env bash
# Copyright (C) 2026 Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Source-table consistency and anti-overclaim checks. Does not reproduce the
# USGS geological model and does not imply raw data-release ingestion.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_file="${1:-$repo_root/research/larsen-basin-repro/larsen_basin_assessment_v0.json}"

command -v jq >/dev/null 2>&1 || {
  echo "ERROR: jq is required to audit the Larsen Basin transcription" >&2
  exit 2
}
test -f "$data_file" || {
  echo "ERROR: missing assessment transcription: $data_file" >&2
  exit 2
}

audit_assessment() {
  local candidate="$1"
  jq -e '
    def valid_dist:
      (.F95 | type) == "number"
      and (.F50 | type) == "number"
      and (.F5 | type) == "number"
      and (.mean | type) == "number"
      and (.F95 <= .F50)
      and (.F50 <= .F5)
      and (.F95 <= .mean)
      and (.mean <= .F5);

    .schema_version == "usgs-larsen-basin-assessment-v0.1"
    and .assessment_id == "usgs-larsen-basin-reservoirs-au-fs20263063"
    and .status == "published_table_transcription_raw_release_not_yet_ingested"
    and .evidence_class == "published_geological_resource_assessment"
    and .sources.fact_sheet.publication_number == "Fact Sheet 2026-3063"
    and .sources.fact_sheet.publication_date == "2026-02-25"
    and .sources.fact_sheet.doi == "10.3133/fs20263063"
    and .sources.data_release.doi == "10.5066/P14LT9DY"
    and .sources.data_release.declared_license == "CC0 1.0 Universal"
    and .sources.data_release.raw_bytes_downloaded_by_this_workstream == false
    and .sources.data_release.raw_sha256 == null
    and .assessment_definition.assessment_unit == "Larsen Basin Reservoirs AU"
    and .assessment_definition.assessment_unit_probability == 0.9
    and .assessment_definition.resource_class == "undiscovered_technically_recoverable_conventional_resources"
    and .assessment_definition.estimates_are_fully_risked == true
    and .assessment_definition.proved_reserves == false
    and .assessment_definition.discoveries == false
    and .assessment_definition.economically_recoverable_reserves == false
    and .assessment_definition.commercial_viability_assessed == false
    and .assessment_definition.fractile_semantics.F95 == "95-percent chance of at least the amount tabulated"
    and .assessment_definition.fractile_semantics.F50 == "50-percent chance of at least the amount tabulated"
    and .assessment_definition.fractile_semantics.F5 == "5-percent chance of at least the amount tabulated"

    and .assessment_inputs.oil_field_count.minimum == 1
    and .assessment_inputs.oil_field_count.median == 20
    and .assessment_inputs.oil_field_count.maximum == 80
    and .assessment_inputs.oil_field_count.calculated_mean == 22.1
    and .assessment_inputs.gas_field_count.minimum == 1
    and .assessment_inputs.gas_field_count.median == 50
    and .assessment_inputs.gas_field_count.maximum == 200
    and .assessment_inputs.gas_field_count.calculated_mean == 55.1
    and .assessment_inputs.oil_field_size.minimum == 5
    and .assessment_inputs.oil_field_size.median == 8
    and .assessment_inputs.oil_field_size.maximum == 300
    and .assessment_inputs.oil_field_size.calculated_mean == 13.6
    and .assessment_inputs.gas_field_size.minimum == 30
    and .assessment_inputs.gas_field_size.median == 48
    and .assessment_inputs.gas_field_size.maximum == 30000
    and .assessment_inputs.gas_field_size.calculated_mean == 272.6

    and .results.oil_accumulations.oil.unit == "MMBO"
    and (.results.oil_accumulations.oil | valid_dist)
    and .results.oil_accumulations.oil.F95 == 0
    and .results.oil_accumulations.oil.F50 == 245
    and .results.oil_accumulations.oil.F5 == 591
    and .results.oil_accumulations.oil.mean == 269

    and .results.oil_accumulations.gas.unit == "BCFG"
    and (.results.oil_accumulations.gas | valid_dist)
    and .results.oil_accumulations.gas.F95 == 0
    and .results.oil_accumulations.gas.F50 == 637
    and .results.oil_accumulations.gas.F5 == 1538
    and .results.oil_accumulations.gas.mean == 698

    and .results.oil_accumulations.ngl.unit == "MMBNGL"
    and (.results.oil_accumulations.ngl | valid_dist)
    and .results.oil_accumulations.ngl.F95 == 0
    and .results.oil_accumulations.ngl.F50 == 29
    and .results.oil_accumulations.ngl.F5 == 71
    and .results.oil_accumulations.ngl.mean == 32

    and .results.gas_accumulations.oil == null
    and .results.gas_accumulations.gas.unit == "BCFG"
    and (.results.gas_accumulations.gas | valid_dist)
    and .results.gas_accumulations.gas.F95 == 0
    and .results.gas_accumulations.gas.F50 == 10702
    and .results.gas_accumulations.gas.F5 == 36266
    and .results.gas_accumulations.gas.mean == 13559
    and .results.gas_accumulations.ngl.unit == "MMBNGL"
    and (.results.gas_accumulations.ngl | valid_dist)
    and .results.gas_accumulations.ngl.F95 == 0
    and .results.gas_accumulations.ngl.F50 == 321
    and .results.gas_accumulations.ngl.F5 == 1089
    and .results.gas_accumulations.ngl.mean == 407

    and .results.total_undiscovered_conventional_resources.oil.unit == "MMBO"
    and (.results.total_undiscovered_conventional_resources.oil | valid_dist)
    and .results.total_undiscovered_conventional_resources.oil.F95 == 0
    and .results.total_undiscovered_conventional_resources.oil.F50 == 245
    and .results.total_undiscovered_conventional_resources.oil.F5 == 591
    and .results.total_undiscovered_conventional_resources.oil.mean == 269

    and .results.total_undiscovered_conventional_resources.gas.unit == "BCFG"
    and (.results.total_undiscovered_conventional_resources.gas | valid_dist)
    and .results.total_undiscovered_conventional_resources.gas.F95 == 0
    and .results.total_undiscovered_conventional_resources.gas.F50 == 11339
    and .results.total_undiscovered_conventional_resources.gas.F5 == 37804
    and .results.total_undiscovered_conventional_resources.gas.mean == 14257

    and .results.total_undiscovered_conventional_resources.ngl.unit == "MMBNGL"
    and (.results.total_undiscovered_conventional_resources.ngl | valid_dist)
    and .results.total_undiscovered_conventional_resources.ngl.F95 == 0
    and .results.total_undiscovered_conventional_resources.ngl.F50 == 350
    and .results.total_undiscovered_conventional_resources.ngl.F5 == 1160
    and .results.total_undiscovered_conventional_resources.ngl.mean == 439

    # Displayed table consistency. Quantile addition is checked only against
    # these specific published rows, not asserted as a general statistical law.
    and .results.total_undiscovered_conventional_resources.oil == .results.oil_accumulations.oil
    and .results.total_undiscovered_conventional_resources.gas.F95
        == (.results.oil_accumulations.gas.F95 + .results.gas_accumulations.gas.F95)
    and .results.total_undiscovered_conventional_resources.gas.F50
        == (.results.oil_accumulations.gas.F50 + .results.gas_accumulations.gas.F50)
    and .results.total_undiscovered_conventional_resources.gas.F5
        == (.results.oil_accumulations.gas.F5 + .results.gas_accumulations.gas.F5)
    and .results.total_undiscovered_conventional_resources.gas.mean
        == (.results.oil_accumulations.gas.mean + .results.gas_accumulations.gas.mean)
    and .results.total_undiscovered_conventional_resources.ngl.F95
        == (.results.oil_accumulations.ngl.F95 + .results.gas_accumulations.ngl.F95)
    and .results.total_undiscovered_conventional_resources.ngl.F50
        == (.results.oil_accumulations.ngl.F50 + .results.gas_accumulations.ngl.F50)
    and .results.total_undiscovered_conventional_resources.ngl.F5
        == (.results.oil_accumulations.ngl.F5 + .results.gas_accumulations.ngl.F5)
    and .results.total_undiscovered_conventional_resources.ngl.mean
        == (.results.oil_accumulations.ngl.mean + .results.gas_accumulations.ngl.mean)
  ' "$candidate" >/dev/null
}

expect_rejected() {
  local label="$1"
  local filter="$2"
  local mutated
  mutated="$(mktemp)"
  if ! jq "$filter" "$data_file" >"$mutated"; then
    rm -f "$mutated"
    echo "ERROR: could not construct Larsen negative control: $label" >&2
    return 1
  fi
  if audit_assessment "$mutated" >/dev/null 2>&1; then
    rm -f "$mutated"
    echo "ERROR: invalid Larsen assessment transcription was accepted: $label" >&2
    return 1
  fi
  rm -f "$mutated"
  echo "PASS: rejected negative control — $label"
}

audit_assessment "$data_file"
echo "PASS: USGS published-table transcription, units, risk/evidence semantics and displayed-table consistency."

expect_rejected "total gas mean tampered" '.results.total_undiscovered_conventional_resources.gas.mean = 14258'
expect_rejected "invalid F95/F50/F5 ordering" '.results.total_undiscovered_conventional_resources.gas.F95 = 20000'
expect_rejected "unit confusion" '.results.total_undiscovered_conventional_resources.oil.unit = "BCFG"'
expect_rejected "undiscovered resource relabeled as proved reserves" '.assessment_definition.proved_reserves = true'
expect_rejected "AU probability outside unit interval" '.assessment_definition.assessment_unit_probability = 1.5'
expect_rejected "raw data claimed as ingested without digest" '.sources.data_release.raw_bytes_downloaded_by_this_workstream = true'
expect_rejected "total gas row inconsistent with published components" '.results.total_undiscovered_conventional_resources.gas.F5 = 37805'
expect_rejected "source resource class changed to commercial reserves" '.assessment_definition.resource_class = "commercial_reserves"'
expect_rejected "oil fractile changed from source table" '.results.oil_accumulations.oil.F50 = 246'

echo "BOUNDARY: published table transcription and software semantics only; no independent geological model reproduction or extraction/commercial claim."
sha256sum "$data_file"
