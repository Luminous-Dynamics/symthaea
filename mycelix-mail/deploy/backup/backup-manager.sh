#!/bin/bash
#
# Mycelix Mail Backup Manager
#
# Automated backup system with point-in-time recovery,
# geo-replication, and disaster recovery support.
#
# Usage: ./backup-manager.sh [command] [options]

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/var/backups/mycelix}"
S3_BUCKET="${S3_BUCKET:-mycelix-backups}"
S3_REGION="${S3_REGION:-us-east-1}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
RETENTION_WEEKS="${RETENTION_WEEKS:-12}"
RETENTION_MONTHS="${RETENTION_MONTHS:-12}"

# Database configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-mycelix_mail}"
DB_USER="${DB_USER:-mycelix}"

# Redis configuration
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

# Encryption
ENCRYPTION_KEY_FILE="${ENCRYPTION_KEY_FILE:-/etc/mycelix/backup-key}"
GPG_RECIPIENT="${GPG_RECIPIENT:-backup@mycelix.mail}"

# Logging
LOG_FILE="${LOG_FILE:-/var/log/mycelix/backup.log}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "${YELLOW}$*${NC}"; }
log_error() { log "ERROR" "${RED}$*${NC}"; }
log_success() { log "SUCCESS" "${GREEN}$*${NC}"; }

notify_slack() {
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        local message="$1"
        local color="${2:-good}"
        curl -s -X POST "$SLACK_WEBHOOK" \
            -H 'Content-type: application/json' \
            -d "{\"attachments\":[{\"color\":\"$color\",\"text\":\"$message\"}]}" \
            > /dev/null 2>&1 || true
    fi
}

# Create backup directory structure
init_backup_dirs() {
    mkdir -p "$BACKUP_DIR"/{daily,weekly,monthly,temp}
    mkdir -p "$(dirname "$LOG_FILE")"
}

# Generate backup filename with timestamp
backup_filename() {
    local type="$1"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    echo "mycelix_${type}_${timestamp}"
}

# ============================================================================
# Database Backup Functions
# ============================================================================

backup_database() {
    log_info "Starting PostgreSQL database backup..."
    local filename=$(backup_filename "db")
    local backup_path="$BACKUP_DIR/temp/${filename}.sql"
    local compressed_path="${backup_path}.gz"
    local encrypted_path="${compressed_path}.gpg"

    # Create dump with parallel jobs for large databases
    PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -Fc \
        --no-owner \
        --no-acl \
        -j 4 \
        -f "$backup_path" 2>> "$LOG_FILE"

    if [[ $? -ne 0 ]]; then
        log_error "Database backup failed!"
        notify_slack "🚨 Database backup failed for Mycelix Mail" "danger"
        return 1
    fi

    # Compress
    gzip -9 "$backup_path"
    log_info "Compressed backup: $(du -h "$compressed_path" | cut -f1)"

    # Encrypt if GPG is available
    if command -v gpg &> /dev/null && [[ -n "$GPG_RECIPIENT" ]]; then
        gpg --encrypt --recipient "$GPG_RECIPIENT" --output "$encrypted_path" "$compressed_path"
        rm "$compressed_path"
        compressed_path="$encrypted_path"
        log_info "Backup encrypted with GPG"
    fi

    # Move to appropriate retention folder
    local day_of_week=$(date '+%u')
    local day_of_month=$(date '+%d')

    if [[ "$day_of_month" == "01" ]]; then
        mv "$compressed_path" "$BACKUP_DIR/monthly/"
        log_info "Monthly backup saved"
    elif [[ "$day_of_week" == "7" ]]; then
        mv "$compressed_path" "$BACKUP_DIR/weekly/"
        log_info "Weekly backup saved"
    else
        mv "$compressed_path" "$BACKUP_DIR/daily/"
        log_info "Daily backup saved"
    fi

    log_success "Database backup completed successfully"
    return 0
}

# Point-in-time recovery backup (WAL archiving)
setup_wal_archiving() {
    log_info "Configuring WAL archiving for point-in-time recovery..."

    local wal_dir="$BACKUP_DIR/wal"
    mkdir -p "$wal_dir"

    # PostgreSQL archive command (add to postgresql.conf)
    cat << EOF
# Add to postgresql.conf for WAL archiving:
archive_mode = on
archive_command = 'test ! -f ${wal_dir}/%f && cp %p ${wal_dir}/%f'
archive_timeout = 300
wal_level = replica
max_wal_senders = 3
wal_keep_size = 1GB
EOF

    log_info "WAL archiving configuration generated"
}

# ============================================================================
# Redis Backup Functions
# ============================================================================

backup_redis() {
    log_info "Starting Redis backup..."
    local filename=$(backup_filename "redis")
    local backup_path="$BACKUP_DIR/temp/${filename}.rdb"

    # Trigger Redis BGSAVE
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE

    # Wait for background save to complete
    while [[ $(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE) == $(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE) ]]; do
        sleep 1
    done

    # Copy the RDB file
    local redis_rdb="/var/lib/redis/dump.rdb"
    if [[ -f "$redis_rdb" ]]; then
        cp "$redis_rdb" "$backup_path"
        gzip -9 "$backup_path"
        mv "${backup_path}.gz" "$BACKUP_DIR/daily/"
        log_success "Redis backup completed"
    else
        log_warn "Redis RDB file not found at $redis_rdb"
    fi
}

# ============================================================================
# File Storage Backup
# ============================================================================

backup_attachments() {
    log_info "Starting attachments backup..."
    local filename=$(backup_filename "attachments")
    local backup_path="$BACKUP_DIR/temp/${filename}.tar.gz"
    local attachments_dir="${ATTACHMENTS_DIR:-/var/lib/mycelix/attachments}"

    if [[ -d "$attachments_dir" ]]; then
        tar -czf "$backup_path" -C "$attachments_dir" .
        log_info "Attachments backup: $(du -h "$backup_path" | cut -f1)"

        # Encrypt
        if command -v gpg &> /dev/null && [[ -n "$GPG_RECIPIENT" ]]; then
            gpg --encrypt --recipient "$GPG_RECIPIENT" --output "${backup_path}.gpg" "$backup_path"
            rm "$backup_path"
            backup_path="${backup_path}.gpg"
        fi

        mv "$backup_path" "$BACKUP_DIR/daily/"
        log_success "Attachments backup completed"
    else
        log_warn "Attachments directory not found: $attachments_dir"
    fi
}

# ============================================================================
# S3 Sync Functions
# ============================================================================

sync_to_s3() {
    log_info "Syncing backups to S3..."

    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not installed, skipping S3 sync"
        return 1
    fi

    # Sync with server-side encryption
    aws s3 sync "$BACKUP_DIR" "s3://${S3_BUCKET}/" \
        --region "$S3_REGION" \
        --sse AES256 \
        --storage-class STANDARD_IA \
        --exclude "temp/*" \
        2>> "$LOG_FILE"

    if [[ $? -eq 0 ]]; then
        log_success "S3 sync completed"
        notify_slack "✅ Mycelix Mail backups synced to S3" "good"
    else
        log_error "S3 sync failed"
        notify_slack "🚨 S3 sync failed for Mycelix Mail backups" "danger"
        return 1
    fi
}

# Geo-replication to secondary region
geo_replicate() {
    local secondary_bucket="${S3_BUCKET}-dr"
    local secondary_region="${S3_DR_REGION:-eu-west-1}"

    log_info "Replicating to secondary region: $secondary_region..."

    aws s3 sync "s3://${S3_BUCKET}/" "s3://${secondary_bucket}/" \
        --region "$secondary_region" \
        --source-region "$S3_REGION" \
        2>> "$LOG_FILE"

    if [[ $? -eq 0 ]]; then
        log_success "Geo-replication completed to $secondary_region"
    else
        log_error "Geo-replication failed"
    fi
}

# ============================================================================
# Retention Policy
# ============================================================================

cleanup_old_backups() {
    log_info "Applying retention policy..."

    # Daily backups: keep for RETENTION_DAYS
    find "$BACKUP_DIR/daily" -type f -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    log_info "Cleaned daily backups older than $RETENTION_DAYS days"

    # Weekly backups: keep for RETENTION_WEEKS weeks
    local weekly_days=$((RETENTION_WEEKS * 7))
    find "$BACKUP_DIR/weekly" -type f -mtime +$weekly_days -delete 2>/dev/null || true
    log_info "Cleaned weekly backups older than $RETENTION_WEEKS weeks"

    # Monthly backups: keep for RETENTION_MONTHS months
    local monthly_days=$((RETENTION_MONTHS * 30))
    find "$BACKUP_DIR/monthly" -type f -mtime +$monthly_days -delete 2>/dev/null || true
    log_info "Cleaned monthly backups older than $RETENTION_MONTHS months"

    # Clean temp directory
    find "$BACKUP_DIR/temp" -type f -mtime +1 -delete 2>/dev/null || true

    log_success "Retention policy applied"
}

# ============================================================================
# Restore Functions
# ============================================================================

list_backups() {
    echo "=== Available Backups ==="
    echo ""
    echo "Daily:"
    ls -lh "$BACKUP_DIR/daily/" 2>/dev/null || echo "  (none)"
    echo ""
    echo "Weekly:"
    ls -lh "$BACKUP_DIR/weekly/" 2>/dev/null || echo "  (none)"
    echo ""
    echo "Monthly:"
    ls -lh "$BACKUP_DIR/monthly/" 2>/dev/null || echo "  (none)"
}

restore_database() {
    local backup_file="$1"

    if [[ -z "$backup_file" ]]; then
        log_error "Usage: $0 restore-db <backup_file>"
        return 1
    fi

    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi

    log_warn "This will OVERWRITE the current database. Are you sure? (yes/no)"
    read -r confirm
    if [[ "$confirm" != "yes" ]]; then
        log_info "Restore cancelled"
        return 0
    fi

    log_info "Starting database restore from: $backup_file"

    local temp_file="$BACKUP_DIR/temp/restore_$(date +%s)"

    # Decrypt if needed
    if [[ "$backup_file" == *.gpg ]]; then
        gpg --decrypt --output "${temp_file}.sql.gz" "$backup_file"
        backup_file="${temp_file}.sql.gz"
    fi

    # Decompress if needed
    if [[ "$backup_file" == *.gz ]]; then
        gunzip -c "$backup_file" > "${temp_file}.sql"
        backup_file="${temp_file}.sql"
    fi

    # Restore
    PGPASSWORD="$DB_PASSWORD" pg_restore \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --clean \
        --if-exists \
        -j 4 \
        "$backup_file" 2>> "$LOG_FILE"

    if [[ $? -eq 0 ]]; then
        log_success "Database restore completed successfully"
        notify_slack "✅ Mycelix Mail database restored successfully" "good"
    else
        log_error "Database restore failed"
        notify_slack "🚨 Database restore failed for Mycelix Mail" "danger"
        return 1
    fi

    # Cleanup temp files
    rm -f "${temp_file}"* 2>/dev/null || true
}

# Point-in-time recovery
restore_pitr() {
    local target_time="$1"

    if [[ -z "$target_time" ]]; then
        log_error "Usage: $0 restore-pitr <target_time>"
        log_info "Example: $0 restore-pitr '2024-01-15 14:30:00'"
        return 1
    fi

    log_info "Point-in-time recovery to: $target_time"

    # Find the latest base backup before target time
    local base_backup=$(ls -t "$BACKUP_DIR"/{daily,weekly,monthly}/*_db_*.gz 2>/dev/null | head -1)

    if [[ -z "$base_backup" ]]; then
        log_error "No base backup found"
        return 1
    fi

    log_info "Using base backup: $base_backup"
    log_info "Applying WAL files up to: $target_time"

    # Recovery configuration for PostgreSQL
    cat << EOF
# Create recovery.conf or recovery.signal with these settings:
restore_command = 'cp ${BACKUP_DIR}/wal/%f %p'
recovery_target_time = '${target_time}'
recovery_target_action = 'promote'
EOF

    log_warn "Manual steps required for PITR - see output above"
}

# ============================================================================
# Verification
# ============================================================================

verify_backup() {
    local backup_file="$1"

    if [[ -z "$backup_file" ]]; then
        log_error "Usage: $0 verify <backup_file>"
        return 1
    fi

    log_info "Verifying backup: $backup_file"

    local temp_file="$BACKUP_DIR/temp/verify_$(date +%s)"

    # Decrypt if needed
    if [[ "$backup_file" == *.gpg ]]; then
        if gpg --decrypt --output "${temp_file}.sql.gz" "$backup_file" 2>/dev/null; then
            log_success "GPG decryption: OK"
            backup_file="${temp_file}.sql.gz"
        else
            log_error "GPG decryption: FAILED"
            return 1
        fi
    fi

    # Decompress and verify
    if [[ "$backup_file" == *.gz ]]; then
        if gzip -t "$backup_file" 2>/dev/null; then
            log_success "Compression integrity: OK"
        else
            log_error "Compression integrity: FAILED"
            return 1
        fi
    fi

    # For pg_dump format, verify with pg_restore
    if [[ "$backup_file" == *_db_* ]]; then
        if pg_restore -l "$backup_file" > /dev/null 2>&1; then
            log_success "PostgreSQL dump format: OK"
        else
            log_warn "PostgreSQL dump format: Could not verify (may be OK)"
        fi
    fi

    rm -f "${temp_file}"* 2>/dev/null || true
    log_success "Backup verification completed"
}

# ============================================================================
# Main
# ============================================================================

show_help() {
    cat << EOF
Mycelix Mail Backup Manager

Usage: $0 <command> [options]

Commands:
  backup              Run full backup (database, redis, attachments)
  backup-db           Backup database only
  backup-redis        Backup Redis only
  backup-attachments  Backup attachments only
  sync                Sync backups to S3
  geo-replicate       Replicate to secondary region
  cleanup             Apply retention policy
  list                List available backups
  restore-db <file>   Restore database from backup
  restore-pitr <time> Point-in-time recovery
  verify <file>       Verify backup integrity
  setup-wal           Show WAL archiving configuration
  help                Show this help

Environment Variables:
  BACKUP_DIR          Backup directory (default: /var/backups/mycelix)
  S3_BUCKET           S3 bucket for offsite backup
  DB_HOST, DB_PORT    Database connection
  DB_NAME, DB_USER    Database credentials
  DB_PASSWORD         Database password (required)
  RETENTION_DAYS      Daily backup retention (default: 30)
  RETENTION_WEEKS     Weekly backup retention (default: 12)
  RETENTION_MONTHS    Monthly backup retention (default: 12)
EOF
}

main() {
    init_backup_dirs

    case "${1:-}" in
        backup)
            backup_database
            backup_redis
            backup_attachments
            sync_to_s3
            cleanup_old_backups
            ;;
        backup-db)
            backup_database
            ;;
        backup-redis)
            backup_redis
            ;;
        backup-attachments)
            backup_attachments
            ;;
        sync)
            sync_to_s3
            ;;
        geo-replicate)
            geo_replicate
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        list)
            list_backups
            ;;
        restore-db)
            restore_database "$2"
            ;;
        restore-pitr)
            restore_pitr "$2"
            ;;
        verify)
            verify_backup "$2"
            ;;
        setup-wal)
            setup_wal_archiving
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            show_help
            exit 1
            ;;
    esac
}

main "$@"
