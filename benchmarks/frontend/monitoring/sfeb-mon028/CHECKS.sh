#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE="$ROOT/benchmark"

for command in go jq; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "required command missing: $command" >&2
    exit 69
  }
done

go_version="$(go env GOVERSION)"
case "$go_version" in
  go1.23*|go1.2[4-9]*|go1.[3-9][0-9]*) ;;
  *) echo "Go 1.23 or newer is required; found $go_version" >&2; exit 69 ;;
esac

jq -e '.profile == "mon028-falsification-first-suite-v1" and (.cases | length >= 2)' "$ROOT/SUITE-example.json" >/dev/null
jq -e '.profile == "mon028-falsification-first-benchmark-v1" and ([.scenarios[].id] | unique | length == (.scenarios | length))' "$ROOT/SCENARIOS-v1.json" >/dev/null
jq -e '.properties.profile.const == "mon028-falsification-first-suite-v1"' "$ROOT/SUITE-v1.schema.json" >/dev/null
jq -e '.properties.profile.const == "mon028-falsification-first-results-v1"' "$ROOT/RESULT-v1.schema.json" >/dev/null
jq -e '.profile == "mon028-independent-ground-truth-oracle-v1" and ([.entries[].case_id] | unique | length == (.entries | length))' "$ROOT/ORACLE-v1.json" >/dev/null
jq -e '.profile == "mon028-workload-profile-v1"' "$ROOT/WORKLOAD-example.json" >/dev/null
jq -e '.profile == "mon028-environment-profile-v1"' "$ROOT/ENVIRONMENT-example.json" >/dev/null
jq -e '.profile == "mon028-resource-budget-profile-v1"' "$ROOT/RESOURCE-BUDGET-example.json" >/dev/null
jq -e '.properties.profile.const == "mon028-independent-ground-truth-oracle-v1"' "$ROOT/ORACLE-v1.schema.json" >/dev/null
jq -e '.profile == "mon028-falsification-first-suite-v1" and ([.cases[] | select(.expected == "qualified")] | length > 0) and ([.cases[] | select(.expected != "qualified")] | length > 0)' "$ROOT/SUITE-example.json" >/dev/null

cd "$MODULE"
find . -type f -name '*.go' -print0 | xargs -0 gofmt -w
go vet ./...
go test ./...
go test -race ./...
go test -run '^$' -fuzz FuzzPairKeyDeterministic -fuzztime=30s
go test -run '^$' -bench BenchmarkScoreRun -benchmem -count=10

echo "MON-028 local Go, race, vet, fuzz and benchmark checks passed."
echo "This script does not run PostgreSQL/Kafka/restore experiments."
