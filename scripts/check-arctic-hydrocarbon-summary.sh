#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILE="$ROOT/research/arctic-hydrocarbon-atlas/resource-summary-v0.json"
SELF_TEST=0
if [[ "$#" -gt 0 ]]; then
  if [[ "$1" == "--self-test" ]]; then SELF_TEST=1; else FILE="$1"; fi
fi
command -v jq >/dev/null 2>&1 || { echo "ERROR: jq required" >&2; exit 2; }
[[ -r "$FILE" ]] || { echo "ERROR: cannot read $FILE" >&2; exit 2; }
jq -e '
 .schema_version=="0.1.0"
 and .artifact_kind=="arctic-hydrocarbon-summary-transcription"
 and .status=="source_cited_transcription"
 and .evidence_tier=="S1"
 and .claim_ceiling=="transcription_only"
 and .source_record.publication_year==2008
 and .source_record.source_bytes_downloaded_by_this_workstream==false
 and .source_record.raw_data_ingested==false
 and .source_record.raw_sha256==null
 and .scope.geographic_scope=="areas_north_of_arctic_circle"
 and .scope.north_pole_specific_estimate_present==false
 and .scope.resource_class=="undiscovered_conventional_resources"
 and (.reported_estimates|length==3)
 and all(.reported_estimates[]; .resource_class=="undiscovered_conventional_resource" and .statistic=="sum_of_mean_estimates")
 and ( [.reported_estimates[].resource] | sort ) == ["natural_gas","natural_gas_liquids","oil"]
 and all(.reported_qualifiers[]; .current_production_statistic == false)
 and (.reported_qualifiers[0].field == "share_expected_offshore")
 and (.reported_qualifiers[0].value_approx_percent >= 0 and .reported_qualifiers[0].value_approx_percent <= 100)
 and all(.limitations[]; type=="string" and length>5)
' "$FILE" >/dev/null || { echo "ERROR: summary record violates evidence contract" >&2; exit 1; }
if [[ "$SELF_TEST" -eq 0 ]]; then
 echo "S1 summary transcription checks passed; no raw source ingestion, reserve validation or economic assessment is asserted"
 exit 0
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
reject() {
 local name="$1" expr="$2"
 jq "$expr" "$ROOT/research/arctic-hydrocarbon-atlas/resource-summary-v0.json" > "$TMP/mutated.json"
 if "$0" "$TMP/mutated.json" >/dev/null 2>&1; then echo "ERROR: accepted mutation: $name" >&2; exit 1; fi
 echo "rejected: $name"
}
reject "S1-to-domain-validated claim escalation" '.claim_ceiling="domain_validated"'
reject "geographic scope narrowed to North Pole" '.scope.geographic_scope="geographic_north_pole"'
reject "resource promoted to reserves" '.reported_estimates[0].resource_class="proved_reserves"'
reject "source-ingestion claim inflation" '.source_record.raw_data_ingested=true'
reject "current production relabel" '.reported_qualifiers[0].current_production_statistic=true'
reject "assessment vintage changed silently" '.scope.publication_year=2026'
echo "6 negative controls rejected; this is transcription integrity only"
