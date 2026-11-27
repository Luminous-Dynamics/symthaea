#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
for cmd in go jq sha256sum; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd is required" >&2; exit 69; }
done

jq -e '
  .profile == "mon040-replay-reproducibility-scenarios-v1"
  and (.scenarios | length) >= 10
  and ([.scenarios[].id] | unique | length) == (.scenarios | length)
' "$ROOT/SCENARIOS-v1.json" >/dev/null

jq -e '
  .profile == "mon040-replay-reproducibility-input-v1"
  and .trial_id == "trial-040-001"
  and .required_runs == 29
  and (.observations | length) == 29
  and ([.observations[].seed] | unique | length) == 29
  and ([.observations[].replay_id] | unique | length) == 29
  and ([.observations[].trial_id] | unique) == [.trial_id]
  and ([.observations[].fault_set_sha256] | unique) == [.fault_set_sha256]
  and ([.observations[].scenario_sha256] | unique) == [.scenario_sha256]
  and ([.observations[].recovery_boundary_sha256] | unique) == [.recovery_boundary_sha256]
  and ([.observations[].code_revision] | unique) == [.code_revision]
' "$ROOT/INPUT-example.json" >/dev/null

verify_manifest() {
  expected="$1"
  file="$2"
  actual="$(sha256sum "$ROOT/$file" | awk '{print $1}')"
  [ "$expected" = "$actual" ] || { echo "MON-040 fixture manifest mismatch: $file" >&2; exit 1; }
}
verify_manifest "$(jq -r '.scenario_sha256' "$ROOT/INPUT-example.json")" "fixtures/manifests/scenario.manifest"
verify_manifest "$(jq -r '.recovery_boundary_sha256' "$ROOT/INPUT-example.json")" "fixtures/manifests/recovery-boundary.manifest"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/cmd"
cp "$ROOT"/go.mod "$ROOT"/*.go "$TMP"/
cp "$ROOT"/cmd/qualify/main.go "$TMP/cmd/main.go"
cd "$TMP"

gofmt -w ./*.go ./cmd/*.go
go vet ./...
go test ./...
go test -race ./...
go build -o "$TMP/mon040" ./cmd

if command -v check-jsonschema >/dev/null 2>&1; then
  check-jsonschema --schemafile "$ROOT/INPUT-v1.schema.json" "$ROOT/INPUT-example.json"
fi

set +e
"$TMP/mon040" --input "$ROOT/INPUT-example.json" > "$TMP/result.json"
STATUS=$?
set -e
[ "$STATUS" -eq 3 ] || { echo "offline reproducibility fixture must exit indeterminate (3), got $STATUS" >&2; exit 1; }
jq -e '
  .reproducibility_verdict == "qualified"
  and .campaign_verdict == "indeterminate"
  and .production_use_permitted == false
  and .required_runs == 29
  and .observed_runs == 29
  and .target_signature_matches == 29
  and .one_sided_confidence_lower_bound >= 0.90
  and .one_sided_confidence_lower_bound < 0.903
  and .evidence_complete == true
' "$TMP/result.json" >/dev/null

jq '.observations |= .[0:28]' "$ROOT/INPUT-example.json" > "$TMP/missing.json"
set +e
"$TMP/mon040" --input "$TMP/missing.json" > "$TMP/missing-result.json"
MISSING_STATUS=$?
set -e
[ "$MISSING_STATUS" -eq 3 ] || { echo "missing preregistered replay must remain indeterminate" >&2; exit 1; }
jq -e '.reproducibility_verdict == "indeterminate" and .evidence_complete == false' "$TMP/missing-result.json" >/dev/null

jq '.observations[0].complete = false' "$ROOT/INPUT-example.json" > "$TMP/tampered.json"
set +e
"$TMP/mon040" --input "$TMP/tampered.json" >/dev/null 2>&1
TAMPER_STATUS=$?
set -e
[ "$TAMPER_STATUS" -eq 2 ] || { echo "receipt tampering must block" >&2; exit 1; }

jq '.observations[1].seed = .observations[0].seed' "$ROOT/INPUT-example.json" > "$TMP/duplicate-seed.json"
set +e
"$TMP/mon040" --input "$TMP/duplicate-seed.json" >/dev/null 2>&1
SEED_STATUS=$?
set -e
[ "$SEED_STATUS" -eq 2 ] || { echo "duplicate seed must block" >&2; exit 1; }

jq '.fault_ids = ["fault-a","fault-b"]' "$ROOT/INPUT-example.json" > "$TMP/wrong-fault-set.json"
set +e
"$TMP/mon040" --input "$TMP/wrong-fault-set.json" >/dev/null 2>&1
FAULT_STATUS=$?
set -e
[ "$FAULT_STATUS" -eq 2 ] || { echo "fault list/digest mismatch must block" >&2; exit 1; }

jq '.unexpected_field = true' "$ROOT/INPUT-example.json" > "$TMP/unknown.json"
set +e
"$TMP/mon040" --input "$TMP/unknown.json" >/dev/null 2>&1
UNKNOWN_STATUS=$?
set -e
[ "$UNKNOWN_STATUS" -eq 65 ] || { echo "unknown JSON field was not rejected" >&2; exit 1; }

go test -run '^$' -bench 'BenchmarkEvaluate(TwentyNine|TwoHundred)Replays$' -benchmem -count=5
echo "MON-040 format, vet, unit, race, CLI, schema, receipt, sample-size and confidence checks passed."
