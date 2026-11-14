#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

unformatted="$(gofmt -l .)"
if [[ -n "$unformatted" ]]; then
  printf 'gofmt required:\n%s\n' "$unformatted" >&2
  exit 1
fi

go test -count=1 ./...
go test -count=1 -race ./...
go vet ./...

tmp_binary="$(mktemp)"
trap 'rm -f "$tmp_binary" "$tmp_binary.observerlab" "$tmp_binary.effectsvc"' EXIT
go build -o "$tmp_binary" ./cmd/mace-evalqual
go build -o "$tmp_binary.observerlab" ./cmd/mace-observerlab
go build -o "$tmp_binary.effectsvc" ./cmd/mace-effectsvc
"$tmp_binary" validate -corpus fixtures/corpus.json -manifest fixtures/manifest.json
"$tmp_binary" score -corpus fixtures/corpus.json -manifest fixtures/manifest.json -observations fixtures/observations.json > /dev/null
"$tmp_binary.observerlab" > /dev/null

if [[ -n "${TLA2TOOLS_JAR:-}" ]]; then
  (cd formal && TLA2TOOLS_JAR="$TLA2TOOLS_JAR" ./run-tlc.sh)
else
  echo "TLC NOT RUN: set TLA2TOOLS_JAR to the verified v1.7.4 tla2tools.jar" >&2
fi
