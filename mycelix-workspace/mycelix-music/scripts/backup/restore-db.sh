#!/usr/bin/env bash
# Database restore script for Mycelix Music
# Usage: ./restore-db.sh <backup_file>
#
# Environment variables:
#   DATABASE_URL - PostgreSQL connection string (required)
#   CONFIRM_RESTORE - Set to "yes" to skip confirmation prompt

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Validate arguments
if [[ $# -lt 1 ]]; then
    log_error "Usage: $0 <backup_file>"
    echo ""
    echo "Examples:"
    echo "  $0 ./backups/mycelix_backup_20240101_120000.sql.gz"
    echo "  $0 s3://my-bucket/backups/mycelix_backup_20240101_120000.sql.gz"
    exit 1
fi

BACKUP_FILE="$1"

# Validate environment
if [[ -z "${DATABASE_URL:-}" ]]; then
    log_error "DATABASE_URL environment variable is required"
    exit 1
fi

# Handle S3 paths
if [[ "${BACKUP_FILE}" == s3://* ]]; then
    log_info "Downloading backup from S3..."
    TEMP_FILE=$(mktemp)
    aws s3 cp "${BACKUP_FILE}" "${TEMP_FILE}"
    BACKUP_FILE="${TEMP_FILE}"
    CLEANUP_TEMP=true
else
    CLEANUP_TEMP=false
fi

# Validate backup file exists
if [[ ! -f "${BACKUP_FILE}" ]]; then
    log_error "Backup file not found: ${BACKUP_FILE}"
    exit 1
fi

# Verify checksum if available
CHECKSUM_FILE="${BACKUP_FILE}.sha256"
if [[ -f "${CHECKSUM_FILE}" ]]; then
    log_info "Verifying checksum..."
    if sha256sum -c "${CHECKSUM_FILE}" > /dev/null 2>&1; then
        log_info "Checksum verified"
    else
        log_error "Checksum verification failed!"
        exit 1
    fi
else
    log_warn "No checksum file found, skipping verification"
fi

# Parse DATABASE_URL
DB_HOST=$(echo "${DATABASE_URL}" | sed -E 's|.*@([^:/]+).*|\1|')
DB_PORT=$(echo "${DATABASE_URL}" | sed -E 's|.*:([0-9]+)/.*|\1|' || echo "5432")
DB_NAME=$(echo "${DATABASE_URL}" | sed -E 's|.*/([^?]+).*|\1|')
DB_USER=$(echo "${DATABASE_URL}" | sed -E 's|.*://([^:]+):.*|\1|')

# Confirmation prompt
log_warn "This will restore the database from: ${BACKUP_FILE}"
log_warn "Target database: ${DB_NAME} on ${DB_HOST}:${DB_PORT}"
log_warn "This will OVERWRITE existing data!"
echo ""

if [[ "${CONFIRM_RESTORE:-}" != "yes" ]]; then
    read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirm
    if [[ "${confirm}" != "yes" ]]; then
        log_info "Restore cancelled"
        exit 0
    fi
fi

log_info "Starting restore..."

# Restore from backup
PGPASSWORD=$(echo "${DATABASE_URL}" | sed -E 's|.*://[^:]+:([^@]+)@.*|\1|')

if [[ "${BACKUP_FILE}" == *.gz ]]; then
    log_info "Decompressing and restoring..."
    gunzip -c "${BACKUP_FILE}" | PGPASSWORD="${PGPASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" --single-transaction
else
    log_info "Restoring from uncompressed backup..."
    PGPASSWORD="${PGPASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" --single-transaction < "${BACKUP_FILE}"
fi

# Cleanup temp file if downloaded from S3
if [[ "${CLEANUP_TEMP}" == "true" ]]; then
    rm -f "${TEMP_FILE}"
fi

log_info "Database restore complete!"

# Show table counts
log_info "Verifying restored data..."
PGPASSWORD="${PGPASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -c "
SELECT 'songs' as table_name, COUNT(*) as row_count FROM songs
UNION ALL
SELECT 'plays', COUNT(*) FROM plays
UNION ALL
SELECT 'strategy_configs', COUNT(*) FROM strategy_configs
UNION ALL
SELECT 'indexer_state', COUNT(*) FROM indexer_state;
"
