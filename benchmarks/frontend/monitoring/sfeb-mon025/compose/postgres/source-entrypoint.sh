#!/bin/sh
set -eu

# Docker named volumes are commonly root-owned on first mount.
# Grant the database OS user access before the official image drops privileges.
if [ "$(id -u)" -eq 0 ]; then
    mkdir -p /backups
    chown postgres:postgres /backups
fi

exec /usr/local/bin/docker-entrypoint.sh "$@"
