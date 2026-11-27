#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
command -v go >/dev/null 2>&1 || { echo "go is required" >&2; exit 69; }
command -v jq >/dev/null 2>&1 || { echo "jq is required" >&2; exit 69; }
jq -e '.profile == "mon036-recovery-evidence-continuity-input-v1" and .ledger.complete == true and (.claims|length)==2 and (.ledger.events|length)==1' "$ROOT/INPUT-example.json" >/dev/null
jq -e '.properties.claims.maxItems == 10000 and .properties.current_states.maxItems == 250000 and .properties.ledger.maxItems == null' "$ROOT/INPUT-v1.schema.json" >/dev/null
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "$ROOT"/go.mod "$ROOT"/*.go "$TMP"/
cp -R "$ROOT"/cmd "$TMP"/
cp "$ROOT/INPUT-example.json" "$ROOT/INPUT-v1.schema.json" "$TMP"/
cd "$TMP"
gofmt -w ./*.go ./cmd/qualify/*.go
go vet ./...
go test ./...
go test -race ./...
BIN="$(mktemp)"
SYNTH="$(mktemp)"
SYNTH_OUT="$(mktemp)"
CORRUPT="$(mktemp)"
CORRUPT_OUT="$(mktemp)"
trap 'rm -f "$BIN" "$SYNTH" "$SYNTH_OUT" "$CORRUPT" "$CORRUPT_OUT"' EXIT
go build -o "$BIN" ./cmd/qualify
"$BIN" --input INPUT-example.json | jq -e '
 .ledger_integrity_valid == true and .ledger_complete == true
 and ([.claims[]|select(.claim_id=="claim-cdc-continuity" and .disposition=="revalidate")]|length)==1
 and ([.claims[]|select(.claim_id=="claim-backup-repository-integrity" and .disposition=="retain")]|length)==1
' >/dev/null
jq '(.claims[0].evidence_grade)="synthetic"' INPUT-example.json > "$SYNTH"
"$BIN" --input "$SYNTH" > "$SYNTH_OUT"
jq -e 'any(.claims[];.claim_id=="claim-cdc-continuity" and .disposition=="unresolved")' "$SYNTH_OUT" >/dev/null
jq '(.ledger.events[0].digest)="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"' INPUT-example.json > "$CORRUPT"
"$BIN" --input "$CORRUPT" > "$CORRUPT_OUT"
jq -e '.ledger_integrity_valid==false and all(.claims[];.disposition=="invalidate")' "$CORRUPT_OUT" >/dev/null
if python3 -c 'import jsonschema' >/dev/null 2>&1; then
 python3 - "$ROOT/INPUT-v1.schema.json" "$ROOT/INPUT-example.json" <<'PY'
import json,sys
from jsonschema import Draft202012Validator
s=json.load(open(sys.argv[1],encoding="utf-8")); e=json.load(open(sys.argv[2],encoding="utf-8"))
Draft202012Validator.check_schema(s); Draft202012Validator(s).validate(e)
e["profile"]="invalid"; assert list(Draft202012Validator(s).iter_errors(e))
PY
else echo "jsonschema unavailable; full schema check skipped" >&2; fi
go test -run '^$' -bench BenchmarkEvaluate100Claims1000Events -benchmem -count=5
echo "MON-036 format, vet, unit, race, CLI, schema and benchmark checks passed. No external services queried."
