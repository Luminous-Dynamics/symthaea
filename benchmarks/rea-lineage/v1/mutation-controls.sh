#!/bin/sh
# Deliberate lineage-integrity mutations must be caught by the fixed oracle.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
command -v jq >/dev/null 2>&1 || { echo 'jq required' >&2; exit 1; }
work=$(mktemp -d "${TMPDIR:-/tmp}/rea-lineage-mutations.XXXXXX")
trap 'rm -rf "$work"' 0
fail(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }
expect_reject(){
 label=$1
 dir=$2
 if (cd "$dir" && sh ./verify.sh) >"$work/$label.out" 2>"$work/$label.err"; then fail "$label mutation unexpectedly passed"; fi
 printf 'PASS mutation rejected: %s\n' "$label"
}
prepare(){
 dir="$work/$1"; mkdir -p "$dir"
 cp "$ROOT/cases.json" "$ROOT/oracle.json" "$ROOT/verify.sh" "$dir/"
}
prepare fake-root-offset
jq '(.cases[]|select(.id=="LNG-001").chunks[]|select(.input_path=="$output/fw")).root_file_range={"offset":292,"length":16}' "$work/fake-root-offset/cases.json" >"$work/a.json"
mv "$work/a.json" "$work/fake-root-offset/cases.json"
expect_reject fake-root-offset "$work/fake-root-offset"
prepare dropped-edge
jq '(.cases[]|select(.id=="LNG-001").derivations)=[]' "$work/dropped-edge/cases.json" >"$work/a.json"
mv "$work/a.json" "$work/dropped-edge/cases.json"
expect_reject dropped-edge "$work/dropped-edge"
prepare null-to-zero
jq '(.cases[]|select(.id=="LNG-007").derivations[0].parent_file_range)={"offset":0,"length":1}' "$work/null-to-zero/cases.json" >"$work/a.json"
mv "$work/a.json" "$work/null-to-zero/cases.json"
expect_reject null-to-zero "$work/null-to-zero"
prepare missing-stage-kind
jq 'del(.cases[]|select(.id=="LNG-017").derivations[0].stage_kind)' "$work/missing-stage-kind/cases.json" >"$work/a.json"
mv "$work/a.json" "$work/missing-stage-kind/cases.json"
expect_reject missing-stage-kind "$work/missing-stage-kind"
prepare empty-provider-id
jq '(.cases[]|select(.id=="LNG-014").chunks[0].provider_id)=""' "$work/empty-provider-id/cases.json" >"$work/a.json"
mv "$work/a.json" "$work/empty-provider-id/cases.json"
expect_reject empty-provider-id "$work/empty-provider-id"
printf 'PASS all five lineage mutation controls\n'
printf 'NOTE: synthetic contract only; no provider or device executed.\n'
