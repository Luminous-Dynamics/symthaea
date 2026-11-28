#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
for cmd in go jq sha256sum; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd is required" >&2; exit 69; }
done

jq -e '
  .profile == "mon039-counterfactual-fault-scenarios-v1"
  and (.scenarios | length) >= 10
  and ([.scenarios[].id] | unique | length) == (.scenarios | length)
' "$ROOT/SCENARIOS-v1.json" >/dev/null

jq -e '
  .trial_id == "trial-039-001"
  and ([.observations[].trial_id] | unique) == [.trial_id]
  and ([.observations[].scenario_sha256] | unique) == [.scenario_sha256]
  and ([.observations[].recovery_boundary_sha256] | unique) == [.recovery_boundary_sha256]
  and ([.observations[].code_revision] | unique) == [.code_revision]
' "$ROOT/INPUT-example.json" >/dev/null
verify_manifest() {
  expected="$1"
  file="$2"
  actual="$(sha256sum "$ROOT/$file" | awk '{print $1}')"
  [ "$expected" = "$actual" ] || { echo "MON-039 fixture manifest mismatch: $file" >&2; exit 1; }
}
verify_manifest "$(jq -r '.scenario_sha256' "$ROOT/INPUT-example.json")" "fixtures/manifests/scenario.manifest"
verify_manifest "$(jq -r '.recovery_boundary_sha256' "$ROOT/INPUT-example.json")" "fixtures/manifests/recovery-boundary.manifest"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/cmd/minimize" "$TMP/cmd/tournament"
cp "$ROOT"/go.mod "$ROOT"/*.go "$TMP"/
cp "$ROOT"/cmd/minimize/main.go "$TMP/cmd/minimize/main.go"
cp "$ROOT"/cmd/tournament/main.go "$TMP/cmd/tournament/main.go"
cd "$TMP"

gofmt -w ./*.go ./cmd/minimize/*.go ./cmd/tournament/*.go
go vet ./...
go test ./...
go test -race ./...
go build -o "$TMP/mon039" ./cmd/minimize
go build -o "$TMP/mon039-tournament" ./cmd/tournament

if command -v check-jsonschema >/dev/null 2>&1; then
  check-jsonschema --schemafile "$ROOT/INPUT-v1.schema.json" "$ROOT/INPUT-example.json"
  check-jsonschema --schemafile "$ROOT/TOURNAMENT-INPUT-v1.schema.json" "$ROOT/TOURNAMENT-INPUT-example.json"
fi

set +e
"$TMP/mon039" --input "$ROOT/INPUT-example.json" > "$TMP/result.json"
STATUS=$?
set -e
[ "$STATUS" -eq 3 ] || { echo "offline fixture must exit indeterminate (3), got $STATUS" >&2; exit 1; }
jq -e '
  .full_fault_set_verdict == "blocked"
  and .minimal_fault_ids == ["fault-a","fault-c"]
  and .minimality_kind == "global_candidate_minimum"
  and .evidence_complete == true
  and .campaign_verdict == "indeterminate"
  and .production_action_permitted == false
  and .subset_evaluations == 7
' "$TMP/result.json" >/dev/null

jq '.observations |= map(select(.fault_ids | length > 0))' "$ROOT/INPUT-example.json" > "$TMP/missing.json"
set +e
"$TMP/mon039" --input "$TMP/missing.json" > "$TMP/missing-result.json"
MISSING_STATUS=$?
set -e
[ "$MISSING_STATUS" -eq 3 ] || { echo "missing subset must remain indeterminate" >&2; exit 1; }
jq -e '.minimality_kind == "not_established" and .evidence_complete == false' "$TMP/missing-result.json" >/dev/null

jq '.observations[0].complete = false' "$ROOT/INPUT-example.json" > "$TMP/receipt-tampered.json"
set +e
"$TMP/mon039" --input "$TMP/receipt-tampered.json" >/dev/null 2>&1
RECEIPT_STATUS=$?
set -e
[ "$RECEIPT_STATUS" -eq 2 ] || { echo "tampered observation receipt must block" >&2; exit 1; }

jq '.code_revision = "ffffffffffffffffffffffffffffffffffffffff"' "$ROOT/INPUT-example.json" > "$TMP/stale.json"
set +e
"$TMP/mon039" --input "$TMP/stale.json" >/dev/null 2>&1
STALE_STATUS=$?
set -e
[ "$STALE_STATUS" -eq 2 ] || { echo "stale code revision must block" >&2; exit 1; }

jq '.unexpected_field = true' "$ROOT/INPUT-example.json" > "$TMP/unknown.json"
set +e
"$TMP/mon039" --input "$TMP/unknown.json" >/dev/null 2>&1
UNKNOWN_STATUS=$?
set -e
[ "$UNKNOWN_STATUS" -eq 65 ] || { echo "unknown JSON field was not rejected" >&2; exit 1; }

set +e
"$TMP/mon039-tournament" --input "$ROOT/TOURNAMENT-INPUT-example.json" > "$TMP/tournament-result.json"
TOURNAMENT_STATUS=$?
set -e
[ "$TOURNAMENT_STATUS" -eq 0 ] || { echo "complete exhaustive tournament fixture must exit 0, got $TOURNAMENT_STATUS" >&2; exit 1; }
jq -e '.orders_attempted == 2 and (.orders|length) == 2 and .evidence_complete == true and .global_minimum_claimed == true and .campaign_verdict == "indeterminate" and .production_action_permitted == false' "$TMP/tournament-result.json" >/dev/null

go test -run '^
echo "MON-039 format, vet, unit, race, CLI, tournament, signature, completeness and benchmark checks passed."
 -bench 'Benchmark(ExhaustiveTwelveFaults|OneMinimalSixtyFourFaults|MinimizeTournamentEightOrders)
echo "MON-039 format, vet, unit, race, CLI, tournament, signature, completeness and benchmark checks passed."
 -benchmem -count=5
echo "MON-039 format, vet, unit, race, CLI, tournament, signature, completeness and benchmark checks passed."
