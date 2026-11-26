#!/usr/bin/env sh
set -eu

: "${SFEB_PGURI:?Set SFEB_PGURI to an isolated disposable PostgreSQL database}"
if [ "${SFEB_ALLOW_DESTRUCTIVE_RESET:-}" != "yes" ]; then
  echo "Refusing to reset schema: set SFEB_ALLOW_DESTRUCTIVE_RESET=yes only for a disposable database" >&2
  exit 2
fi
command -v psql >/dev/null 2>&1 || { echo "psql is required for PostgreSQL integration" >&2; exit 127; }
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
psql "$SFEB_PGURI" -X -v ON_ERROR_STOP=1 -f "$root/schema.sql"
psql "$SFEB_PGURI" -X -v ON_ERROR_STOP=1 -f "$root/integration.sql"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM
cat > "$tmpdir/replay.sql" <<'SQL'
SET ROLE sfeb_writer;
SELECT sfeb.apply_action(
  'tenant-a', 'resource-1', '00000000-0000-0000-0000-000000000005'::uuid,
  convert_to('{"next_state":{"value":5}}', 'UTF8'),
  '{"next_state":{"value":5}}'::jsonb, '{"value":5}'::jsonb,
  4, 8, 2
)->>'status';
RESET ROLE;
SQL

pids=""
i=1
while [ "$i" -le 8 ]; do
  psql "$SFEB_PGURI" -X -q -v ON_ERROR_STOP=1 -f "$tmpdir/replay.sql" >/dev/null &
  pids="$pids $!"
  i=$((i + 1))
done
for pid in $pids; do wait "$pid"; done

psql "$SFEB_PGURI" -X -v ON_ERROR_STOP=1 <<'SQL'
DO $$
BEGIN
  IF (SELECT count(*) FROM sfeb.operation_ledger WHERE operation_id='00000000-0000-0000-0000-000000000005'::uuid) <> 1 THEN
    RAISE EXCEPTION 'concurrent same-ID delivery created multiple ledger entries';
  END IF;
  IF (SELECT revision FROM sfeb.resource_state WHERE tenant_id='tenant-a' AND resource_id='resource-1') <> 3 THEN
    RAISE EXCEPTION 'concurrent same-ID delivery duplicated the mutation';
  END IF;
END $$;
SQL

echo "MON-009 PostgreSQL integration checks passed. This validates the test database profile only; it does not establish independent administration, production key custody, SCITT interoperability, or external-world truth."
