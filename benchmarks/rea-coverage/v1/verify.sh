#!/bin/sh
# Contract-only verifier for normalized REA extraction-coverage reports.
# It does not invoke REA, external extraction tools, firmware, or hardware.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CASES="$ROOT/cases.json"
ORACLE="$ROOT/oracle.json"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

command -v jq >/dev/null 2>&1 || fail "required command unavailable: jq"
command -v sha256sum >/dev/null 2>&1 || fail "required command unavailable: sha256sum"
[ -f "$CASES" ] || fail "missing cases.json"
[ -f "$ORACLE" ] || fail "missing oracle.json"

jq -e '.schema_version == "symthaea.rea-coverage-manifest.v1" and .benchmark_id == "ENG-REA-003" and (.cases | length) == 15' "$CASES" >/dev/null ||
    fail "invalid case manifest or unexpected case count"
jq -e '.schema_version == "symthaea.rea-coverage-oracle.v1" and .benchmark_id == "ENG-REA-003" and (.expected_dispositions | length) == 15' "$ORACLE" >/dev/null ||
    fail "invalid oracle schema or unexpected case count"
case_ids=$(jq -er '[.cases[].id] | sort | join(",")' "$CASES")
oracle_ids=$(jq -er '.expected_dispositions | keys | sort | join(",")' "$ORACLE")
[ "$case_ids" = "$oracle_ids" ] || fail "case IDs do not match oracle IDs (missing, extra, or duplicate IDs)"

# Assessment implements only the declared byte-accounting contract:
# every byte in the selected input is either in a reported chunk or has one
# exact unreported-input-region gap; all gap root offsets map back to source.
assess() {
    case_json=$1
    jq -ner --argjson c "$case_json" '
      def complement($length; $chunks):
        ([$chunks[] | {start: .range.offset, end: (.range.offset + .range.length)}] | sort_by(.start, .end))
        | reduce .[] as $r ({cursor: 0, gaps: []};
            .gaps += (if .cursor < $r.start then [{offset: .cursor, length: ($r.start - .cursor)}] else [] end)
            | .cursor = (if $r.end > .cursor then $r.end else .cursor end)
          )
        | if .cursor < $length then .gaps + [{offset: .cursor, length: ($length - .cursor)}] else .gaps end;
      if ($c.selection_length | type) != "number" or $c.selection_length <= 0 or
         ($c.selection_offset | type) != "number" or $c.selection_offset < 0 then
        "invalid-input"
      elif any($c.chunks[]?;
          (.range.offset | type) != "number" or
          (.range.length | type) != "number" or
          .range.offset < 0 or .range.length <= 0 or
          (.range.offset + .range.length) > $c.selection_length) then
        "invalid-report"
      else
        (complement($c.selection_length; $c.chunks)) as $derived |
        ([$c.extraction_gaps[]? | select(.kind == "unreported-input-region") |
           {offset: .range.offset, length: .range.length,
            root_offset: .root_file_range.offset, root_length: .root_file_range.length}
         ] | sort_by(.offset, .length)) as $actual |
        ([$derived[] |
           {offset: .offset, length: .length,
            root_offset: ($c.selection_offset + .offset), root_length: .length}
         ] | sort_by(.offset, .length)) as $expected |
        (if ($derived | length) > 0 or
            ($c.partial_signals | length) > 0 or
            any($c.chunks[]?; .handler == null or .encrypted == true) or
            ($c.extraction_gaps | length) > 0
         then "partial" else "complete" end) as $coverage |
        if $actual == $expected and $c.coverage == $coverage then "accept" else "reject" end
      end
    '
}

accepted=0
rejected=0
invalid_input=0
invalid_report=0
while IFS= read -r case_json; do
    case_id=$(printf '%s' "$case_json" | jq -er '.id')
    expected=$(jq -er --arg id "$case_id" '.expected_dispositions[$id]' "$ORACLE")
    actual=$(assess "$case_json") || fail "$case_id: assessment crashed"
    [ "$actual" = "$expected" ] || fail "$case_id: expected $expected, got $actual"
    case "$actual" in
      accept) accepted=$((accepted + 1)) ;;
      reject) rejected=$((rejected + 1)) ;;
      invalid-input) invalid_input=$((invalid_input + 1)) ;;
      invalid-report) invalid_report=$((invalid_report + 1)) ;;
      *) fail "$case_id: unexpected disposition $actual" ;;
    esac
    printf 'PASS %s -> %s\n' "$case_id" "$actual"
done <<EOF_CASES
$(jq -c '.cases[]' "$CASES")
EOF_CASES

printf '\nPASS: ENG-REA-003 coverage contract (%s cases)\n' "$(jq '.cases | length' "$CASES")"
printf 'Accepted consistent reports: %s\nRejected inconsistent reports: %s\nInvalid inputs: %s\nMalformed reports: %s\n' \
  "$accepted" "$rejected" "$invalid_input" "$invalid_report"
printf 'cases.json SHA-256: %s\n' "$(sha256sum "$CASES" | awk '{print $1}')"
printf 'oracle.json SHA-256: %s\n' "$(sha256sum "$ORACLE" | awk '{print $1}')"
printf 'NOTE: contract-only synthetic verification; real REA/Unblob provider and hardware are not tested.\n'
