#!/usr/bin/env bash
# Database backup script for Mycelix Music
# Usage: ./backup-db.sh [output_dir]
#
# Environment variables:
#   DATABASE_URL - PostgreSQL connection string (required)
#   BACKUP_RETENTION_DAYS - Number of days to keep backups (default: 7)
#   S3_BUCKET - Optional S3 bucket for remote backup storage
#   AWS_PROFILE - AWS profile to use for S3 uploads

set -euo pipefail

# Configuration
OUTPUT_DIR="${1:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="mycelix_backup_${TIMESTAMP}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Validate environment
if [[ -z "${DATABASE_URL:-}" ]]; then
    log_error "DATABASE_URL environment variable is required"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

log_info "Starting backup: ${BACKUP_NAME}"

# Parse DATABASE_URL for pg_dump
# Format: postgresql://user:pass@host:port/dbname
DB_HOST=$(echo "${DATABASE_URL}" | sed -E 's|.*@([^:/]+).*|\1|')
DB_PORT=$(echo "${DATABASE_URL}" | sed -E 's|.*:([0-9]+)/.*|\1|' || echo "5432")
DB_NAME=$(echo "${DATABASE_URL}" | sed -E 's|.*/([^?]+).*|\1|')
DB_USER=$(echo "${DATABASE_URL}" | sed -E 's|.*://([^:]+):.*|\1|')

# Create backup with pg_dump
BACKUP_FILE="${OUTPUT_DIR}/${BACKUP_NAME}.sql.gz"

log_info "Creating compressed backup to ${BACKUP_FILE}"

PGPASSWORD=$(echo "${DATABASE_URL}" | sed -E 's|.*://[^:]+:([^@]+)@.*|\1|') \
    pg_dump -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
    --format=plain \
    --no-owner \
    --no-acl \
    --clean \
    --if-exists \
    | gzip > "${BACKUP_FILE}"

BACKUP_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
log_info "Backup created: ${BACKUP_FILE} (${BACKUP_SIZE})"

# Create checksum
sha256sum "${BACKUP_FILE}" > "${BACKUP_FILE}.sha256"
log_info "Checksum created: ${BACKUP_FILE}.sha256"

# Upload to S3 if configured
if [[ -n "${S3_BUCKET:-}" ]]; then
    log_info "Uploading to S3: s3://${S3_BUCKET}/backups/${BACKUP_NAME}.sql.gz"

    AWS_ARGS=""
    if [[ -n "${AWS_PROFILE:-}" ]]; then
        AWS_ARGS="--profile ${AWS_PROFILE}"
    fi

    aws ${AWS_ARGS} s3 cp "${BACKUP_FILE}" "s3://${S3_BUCKET}/backups/${BACKUP_NAME}.sql.gz"
    aws ${AWS_ARGS} s3 cp "${BACKUP_FILE}.sha256" "s3://${S3_BUCKET}/backups/${BACKUP_NAME}.sql.gz.sha256"

    log_info "S3 upload complete"
fi

# Clean up old backups
log_info "Cleaning up backups older than ${RETENTION_DAYS} days"
find "${OUTPUT_DIR}" -name "mycelix_backup_*.sql.gz*" -mtime +${RETENTION_DAYS} -delete 2>/dev/null || true

# List remaining backups
BACKUP_COUNT=$(find "${OUTPUT_DIR}" -name "mycelix_backup_*.sql.gz" | wc -l)
log_info "Total backups in ${OUTPUT_DIR}: ${BACKUP_COUNT}"

log_info "Backup complete!"
echo ""
echo "To restore from this backup:"
echo "  ./restore-db.sh ${BACKUP_FILE}"
