#!/usr/bin/env sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -z "$MON013_TEST_DSN" ]; then
  echo "Set MON013_TEST_DSN to an EMPTY disposable PostgreSQL database DSN" >&2
  exit 2
fi
command -v psql >/dev/null 2>&1 || { echo "psql is required" >&2; exit 127; }
tracking=$(psql "$MON013_TEST_DSN" -X -A -t -v ON_ERROR_STOP=1 -c "SHOW track_commit_timestamp")
if [ "$tracking" != "on" ]; then
  echo "track_commit_timestamp=on is required for authority-at-commit qualification" >&2
  exit 2
fi
psql "$MON013_TEST_DSN" -X -v ON_ERROR_STOP=1 -f "$root/schema.sql"
psql "$MON013_TEST_DSN" -X -v ON_ERROR_STOP=1 -f "$root/commit-time-readback.sql"
psql "$MON013_TEST_DSN" -X -v ON_ERROR_STOP=1 -f "$root/integration.sql"
