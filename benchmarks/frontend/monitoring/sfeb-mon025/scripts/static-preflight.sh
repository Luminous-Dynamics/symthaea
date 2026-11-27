#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
required=(bash jq sha256sum)
for cmd in "${required[@]}"; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "required command missing: $cmd" >&2; exit 69; }
done

jq -e '.profile == "mon025-version-matrix-v1" and (.source_profiles | length >= 2) and (.streaming_profiles | length >= 2)' "$ROOT/VERSIONS-v1.json" >/dev/null
jq -e '.profile == "mon025-live-recovery-challenge-v1" and (.scenarios | length >= 10) and ([.scenarios[].id] | unique | length == (.scenarios | length)) and all(.scenarios[]; (.id | test("^MON025-[PN][0-9]{3}$")) and (.expected_outcome == "qualified" or .expected_outcome == "blocked" or .expected_outcome == "indeterminate"))' "$ROOT/FAULT-MATRIX-v1.json" >/dev/null

for script in "$ROOT"/scripts/*.sh; do
  [ -f "$script" ] || continue
  bash -n "$script"
done
sh -n "$ROOT/compose/postgres/source-entrypoint.sh"
sh -n "$ROOT/compose/restore/restore-entrypoint.sh"

bash "$ROOT/scripts/test-transaction-metadata-order.sh"
bash "$ROOT/scripts/test-offset-target.sh"

echo "MON-025 JSON structure and shell syntax checks passed."

if command -v check-jsonschema >/dev/null 2>&1; then
  check-jsonschema --schemafile "$ROOT/VERSIONS-v1.schema.json" "$ROOT/VERSIONS-v1.json"
  check-jsonschema --schemafile "$ROOT/FAULT-MATRIX-v1.schema.json" "$ROOT/FAULT-MATRIX-v1.json"
  echo "JSON Schema validation passed."
else
  echo "check-jsonschema not installed; full JSON Schema validation skipped (jq structural checks passed)."
fi

if command -v shellcheck >/dev/null 2>&1; then
  shellcheck "$ROOT"/scripts/*.sh "$ROOT/compose/postgres/source-entrypoint.sh" "$ROOT/compose/restore/restore-entrypoint.sh"
  echo "ShellCheck passed."
else
  echo "ShellCheck not installed; shell static lint skipped."
fi

if command -v docker >/dev/null 2>&1 && [ -f "$ROOT/.env" ]; then
  docker compose --env-file "$ROOT/.env" -f "$ROOT/compose/compose.yaml" --project-directory "$ROOT/compose" config --quiet
  echo "Docker Compose configuration passed."
else
  echo "Docker Compose parse skipped: Docker or local .env unavailable. No container was started."
fi
