#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/compose/compose.yaml"
ENV_FILE="$ROOT/.env"
DC=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --project-directory "$ROOT/compose")

MAJOR="${1:-18}"
RUN_ID="${2:-}"
case "$MAJOR" in 18|19) ;; *) echo "usage: $0 18|19 RUN_ID" >&2; exit 64 ;; esac
case "$RUN_ID" in *[!A-Za-z0-9_-]*|'') echo "RUN_ID must use A-Z a-z 0-9 _ -" >&2; exit 64 ;; esac
[ -f "$ENV_FILE" ] || { echo ".env is required" >&2; exit 66; }
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
: "${MON025_DB_PASSWORD:?Set MON025_DB_PASSWORD in .env}"
export COMPOSE_PROFILES="pg$MAJOR"

OUT="$ROOT/out/$RUN_ID"
mkdir -p "$OUT"
[ ! -e "$OUT/run-manifest.json" ] || { echo "run manifest already exists; refusing to overwrite $OUT" >&2; exit 73; }

compose_digest="$(sha256sum "$COMPOSE_FILE" | awk '{print $1}')"
versions_digest="$(sha256sum "$ROOT/VERSIONS-v1.json" | awk '{print $1}')"
fault_digest="$(sha256sum "$ROOT/FAULT-MATRIX-v1.json" | awk '{print $1}')"
source_rev="${MON025_SOURCE_REV:-unknown}"
if git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
  source_rev="$(git -C "$ROOT" rev-parse HEAD)"
fi

images_json="$OUT/images.json"
: >"$images_json"
for service in "postgres$MAJOR" kafka connect; do
  container_id="$("${DC[@]}" ps -q "$service")"
  [ -n "$container_id" ] || { echo "service is not running: $service" >&2; exit 69; }
  docker inspect "$container_id" | jq -r '.[0] | {
    service: .Config.Hostname,
    configured_image: .Config.Image,
    image_id: .Image,
    repo_digests: (.RepoDigests // [])
  }' >>"$images_json"
done
jq -s '.' "$images_json" >"$OUT/images.normalized.json"
mv "$OUT/images.normalized.json" "$images_json"

cat >"$OUT/run-manifest.json" <<EOF
{
  "profile": "mon025-live-recovery-run-v1",
  "run_id": "$RUN_ID",
  "postgres_major": "$MAJOR",
  "source_revision": "$source_rev",
  "captured_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "host_kernel": "$(uname -sr)",
  "docker_compose_version": "$("${DC[@]}" version --short)",
  "compose_sha256": "$compose_digest",
  "versions_lock_sha256": "$versions_digest",
  "fault_matrix_sha256": "$fault_digest",
  "images_file": "images.json"
}
EOF

jq -e '.profile == "mon025-live-recovery-run-v1" and .compose_sha256 != ""' "$OUT/run-manifest.json" >/dev/null
echo "Run capsule: $OUT"
sha256sum "$OUT/run-manifest.json" "$OUT/images.json"
