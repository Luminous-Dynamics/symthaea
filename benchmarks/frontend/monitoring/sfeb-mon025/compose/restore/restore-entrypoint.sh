#!/bin/sh
set -eu

: "${MON025_BACKUP_ID:?MON025_BACKUP_ID is required}"
: "${MON025_RECOVERY_TARGET_LSN:?MON025_RECOVERY_TARGET_LSN is required}"
: "${MON025_RECOVERY_TARGET_TIMELINE:?MON025_RECOVERY_TARGET_TIMELINE is required}"

case "$MON025_BACKUP_ID" in
    *[!A-Za-z0-9_-]*|'') echo "invalid backup ID" >&2; exit 64 ;;
esac
printf '%s\n' "$MON025_RECOVERY_TARGET_LSN" | grep -Eq '^[0-9A-Fa-f]{1,8}/[0-9A-Fa-f]{1,8}$' || {
    echo "invalid PostgreSQL LSN; expected HEX/HEX" >&2
    exit 64
}
case "$MON025_RECOVERY_TARGET_TIMELINE" in
    *[!0-9]*|'') echo "recovery timeline must be a positive integer" >&2; exit 64 ;;
esac
[ "$MON025_RECOVERY_TARGET_TIMELINE" -gt 0 ] || { echo "recovery timeline must be positive" >&2; exit 64; }

BACKUP_DIR="/backups/$MON025_BACKUP_ID"
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
restore_command = 'cp /wal-archive/%f %p'
recovery_target_lsn = '$MON025_RECOVERY_TARGET_LSN'
recovery_target_timeline = '$MON025_RECOVERY_TARGET_TIMELINE'
recovery_target_action = 'pause'
recovery_target_inclusive = 'true'
EOF

touch "$PGDATA/recovery.signal"
chown postgres:postgres "$PGDATA/postgresql.auto.conf" "$PGDATA/recovery.signal"

exec /usr/local/bin/docker-entrypoint.sh "$@"
