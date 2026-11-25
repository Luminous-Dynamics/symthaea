#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
fi

MON007="$ROOT/benchmarks/frontend/monitoring/sfeb-mon007"
MON008="$ROOT/benchmarks/frontend/monitoring/sfeb-mon008"

expand_gzip_tree() {
  local tree="$1"
  while IFS= read -r -d '' compressed; do
    local expanded="${compressed%.gz}"
    mkdir -p "$(dirname "$expanded")"
    gzip -t "$compressed"
    gzip -dc "$compressed" > "$expanded"
  done < <(find "$tree" -type f -name '*.gz' -print0)
}

echo "[1/5] Expanding committed MON-007 and MON-008 sources and fixtures"
expand_gzip_tree "$MON007"
expand_gzip_tree "$MON008"

echo "[2/5] MON-007 Go semantic + RFC 9162 suites"
(
  cd "$MON007/go"
  go test -count=1 ./...
  go vet ./...
)
(
  cd "$MON007"
  node --test sfeb-action-assurance-model-v1.test.mjs
)
(
  cd "$MON007/js"
  node --test rfc9162.test.mjs
)

echo "[3/5] MON-008 Go checkpoint and witness suite"
(
  cd "$MON008/go"
  go test -count=3 ./...
  go test -race -count=1 ./...
  go vet ./...
  go test -cover ./...
)

echo "[4/5] MON-008 Node cross-language verifier suite"
(
  cd "$MON008/js"
  node --test checkpoint.test.mjs
)

if [[ "${RUN_BENCHMARKS:-0}" == "1" ]]; then
  echo "[5/5] MON-008 verifier-only benchmark"
  (
    cd "$MON008/go"
    go test -run '^$' -bench 'BenchmarkVerifyCheckpointWitnessQuorum' -benchmem -count=5 ./checkpoint
  )
else
  echo "[5/5] Benchmark skipped (set RUN_BENCHMARKS=1 to run)"
fi

echo "All selected local preflight gates passed. This is not a GitHub CI, production, or end-to-end SOTA result."
