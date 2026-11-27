#!/bin/sh
set -eu

: "${MON025_BACKUP_ID:?MON025_BACKUP_ID is required}"
: "${MON025_RECOVERY_TARGET_TIME:?MON025_RECOVERY_TARGET_TIME is required}"
: "${MON025_SOURCE_DATA_DIR:?MON025_SOURCE_DATA_DIR is required}"

case "$MON025_BACKUP_ID" in
    *[!A-Za-z0-9_-]*|'') echo "invalid backup ID" >&2; exit 64 ;;
esac
case "$MON025_RECOVERY_TARGET_TIME" in
    *[!0-9TtZz:+.\ -]*|'') echo "invalid recovery target timestamp characters" >&2; exit 64 ;;
esac
if ! date -u -d "$MON025_RECOVERY_TARGET_TIME" >/dev/null 2>&1; then
    echo "recovery target is not a parseable timestamp" >&2
    exit 64
fi

BACKUP_DIR="/backups/$MON025_BACKUP_ID"
mkdir -p "$PGDATA"
mkdir -p "$PGDATA"
if [ ! -s "$BACKUP_DIR/PG_VERSION" ]; then
    echo "backup directory is missing or has no PG_VERSION: $BACKUP_DIR" >&2
    exit 66
fi
if [ -s "$PGDATA/PG_VERSION" ]; then
    echo "restore target volume is already initialized; use an explicitly approved new volume" >&2
    exit 73
fi
OTHER_CONTENT="$(find "$PGDATA" -mindepth 1 -maxdepth 1 ! -name lost+found -print -quit)"
if [ -n "$OTHER_CONTENT" ]; then
    echo "restore target volume is not empty: $OTHER_CONTENT" >&2
    exit 73
fi

cp -a "$BACKUP_DIR/." "$PGDATA/"
chown -R postgres:postgres "$PGDATA"

cat >> "$PGDATA/postgresql.auto.conf" <<EOF
restore_command = 'cp $MON025_SOURCE_DATA_DIR/mon025-archive/%f %p'
recovery_target_time = '$MON025_RECOVERY_TARGET_TIME'
recovery_target_action = 'pause'
recovery_target_inclusive = 'true'
recovery_target_timeline = 'latest'
EOF

touch "$PGDATA/recovery.signal"
chown postgres:postgres "$PGDATA/postgresql.auto.conf" "$PGDATA/recovery.signal"

exec /usr/local/bin/docker-entrypoint.sh "$@"
