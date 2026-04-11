// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Backup Integrity Zome
//!
//! Secure backup and restore for Mycelix Mail data.
//! Supports encrypted exports, selective restore, and cross-device migration.

use hdi::prelude::*;

/// Backup manifest - describes a backup
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BackupManifest {
    /// Unique backup ID
    pub backup_id: String,
    /// Agent who created the backup
    pub agent: AgentPubKey,
    /// Backup type
    pub backup_type: BackupType,
    /// What's included
    pub contents: BackupContents,
    /// When created
    pub created_at: Timestamp,
    /// Backup size in bytes
    pub size_bytes: u64,
    /// Number of entries backed up
    pub entry_count: u32,
    /// Checksum for integrity verification
    pub checksum: Vec<u8>,
    /// Encryption info
    pub encryption: BackupEncryption,
    /// Backup status
    pub status: BackupStatus,
    /// Metadata
    pub metadata: BackupMetadata,
}

/// Type of backup
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum BackupType {
    /// Full backup of all data
    Full,
    /// Incremental since last backup
    Incremental { since_backup_id: String },
    /// Selective backup of specific data
    Selective,
    /// Migration export for new device
    Migration,
    /// Emergency recovery backup
    Emergency,
}

/// What's included in the backup
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Default)]
pub struct BackupContents {
    /// Include emails
    pub emails: bool,
    /// Include drafts
    pub drafts: bool,
    /// Include contacts
    pub contacts: bool,
    /// Include trust attestations
    pub trust_attestations: bool,
    /// Include folders
    pub folders: bool,
    /// Include labels
    pub labels: bool,
    /// Include settings
    pub settings: bool,
    /// Include keys (encrypted)
    pub keys: bool,
    /// Include search index
    pub search_index: bool,
    /// Custom filters
    pub filters: Option<BackupFilters>,
}

/// Filters for selective backup
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BackupFilters {
    /// Only emails from specific folders
    pub folder_whitelist: Option<Vec<ActionHash>>,
    /// Only emails with specific labels
    pub label_whitelist: Option<Vec<String>>,
    /// Only emails after date
    pub date_from: Option<Timestamp>,
    /// Only emails before date
    pub date_to: Option<Timestamp>,
    /// Only emails from specific senders
    pub sender_whitelist: Option<Vec<AgentPubKey>>,
    /// Exclude attachments over size (bytes)
    pub max_attachment_size: Option<u64>,
}

/// Backup encryption configuration
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BackupEncryption {
    /// Whether backup is encrypted
    pub is_encrypted: bool,
    /// Encryption algorithm
    pub algorithm: Option<String>,
    /// Key derivation function
    pub kdf: Option<String>,
    /// Salt for key derivation
    pub salt: Option<Vec<u8>>,
    /// Nonce/IV
    pub nonce: Option<Vec<u8>>,
    /// Key hint (for user to remember password)
    pub key_hint: Option<String>,
}

/// Backup status
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum BackupStatus {
    /// In progress
    InProgress { progress_percent: u8 },
    /// Completed successfully
    Completed,
    /// Failed
    Failed { error: String },
    /// Cancelled
    Cancelled,
    /// Expired (old backup)
    Expired,
}

/// Backup metadata
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Default)]
pub struct BackupMetadata {
    /// App version
    pub app_version: String,
    /// DNA version
    pub dna_version: String,
    /// Device identifier
    pub device_id: Option<String>,
    /// Device name
    pub device_name: Option<String>,
    /// Notes
    pub notes: Option<String>,
    /// Tags for organization
    pub tags: Vec<String>,
}

/// Backup chunk - for streaming large backups
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BackupChunk {
    /// Parent backup ID
    pub backup_id: String,
    /// Chunk index (0-based)
    pub chunk_index: u32,
    /// Total chunks
    pub total_chunks: u32,
    /// Encrypted chunk data
    pub data: Vec<u8>,
    /// Chunk checksum
    pub checksum: Vec<u8>,
}

/// Restore operation record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RestoreOperation {
    /// Restore ID
    pub restore_id: String,
    /// Source backup ID
    pub backup_id: String,
    /// Agent performing restore
    pub agent: AgentPubKey,
    /// What to restore
    pub restore_contents: BackupContents,
    /// Restore options
    pub options: RestoreOptions,
    /// Status
    pub status: RestoreStatus,
    /// Started at
    pub started_at: Timestamp,
    /// Completed at
    pub completed_at: Option<Timestamp>,
    /// Entries restored
    pub entries_restored: u32,
    /// Entries skipped
    pub entries_skipped: u32,
    /// Errors encountered
    pub errors: Vec<RestoreError>,
}

/// Restore options
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Default)]
pub struct RestoreOptions {
    /// Overwrite existing data
    pub overwrite_existing: bool,
    /// Skip conflicts
    pub skip_conflicts: bool,
    /// Merge with existing
    pub merge_mode: MergeMode,
    /// Restore to specific folder
    pub target_folder: Option<ActionHash>,
    /// Dry run (don't actually restore)
    pub dry_run: bool,
}

/// How to handle merging
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Default)]
pub enum MergeMode {
    /// Keep existing, skip backup version
    #[default]
    KeepExisting,
    /// Overwrite with backup version
    OverwriteWithBackup,
    /// Keep newer version
    KeepNewer,
    /// Keep both versions
    KeepBoth,
    /// Manual resolution required
    Manual,
}

/// Restore status
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum RestoreStatus {
    /// Pending start
    Pending,
    /// Validating backup
    Validating,
    /// Decrypting
    Decrypting,
    /// Restoring data
    Restoring { progress_percent: u8 },
    /// Rebuilding indexes
    Indexing,
    /// Completed
    Completed,
    /// Failed
    Failed { error: String },
    /// Cancelled
    Cancelled,
    /// Paused
    Paused,
}

/// Restore error
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RestoreError {
    /// Entry type that failed
    pub entry_type: String,
    /// Entry identifier
    pub entry_id: String,
    /// Error message
    pub error: String,
    /// Whether skipped
    pub skipped: bool,
}

/// Backup schedule for automatic backups
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BackupSchedule {
    /// Schedule ID
    pub schedule_id: String,
    /// Agent
    pub agent: AgentPubKey,
    /// Whether enabled
    pub enabled: bool,
    /// Frequency
    pub frequency: BackupFrequency,
    /// What to backup
    pub contents: BackupContents,
    /// Retention policy
    pub retention: RetentionPolicy,
    /// Last backup
    pub last_backup: Option<Timestamp>,
    /// Next scheduled
    pub next_backup: Option<Timestamp>,
}

/// Backup frequency
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum BackupFrequency {
    /// Every N hours
    Hourly { hours: u8 },
    /// Daily at specific time
    Daily { hour: u8, minute: u8 },
    /// Weekly on specific day
    Weekly { day: u8, hour: u8 },
    /// Monthly on specific day
    Monthly { day: u8, hour: u8 },
    /// Manual only
    Manual,
}

/// How long to keep backups
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RetentionPolicy {
    /// Keep last N backups
    pub keep_count: Option<u32>,
    /// Keep backups for N days
    pub keep_days: Option<u32>,
    /// Keep at least one full backup
    pub keep_one_full: bool,
    /// Keep monthly snapshots
    pub keep_monthly: bool,
}

/// Export format for external storage
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum ExportFormat {
    /// Native Holochain format
    HolochainNative,
    /// JSON export
    Json,
    /// MessagePack
    MessagePack,
    /// Custom format
    Custom(String),
}

/// Link types
#[hdk_link_types]
pub enum LinkTypes {
    /// Agent -> backups
    AgentToBackups,
    /// Backup -> chunks
    BackupToChunks,
    /// Agent -> restore operations
    AgentToRestores,
    /// Agent -> backup schedule
    AgentToSchedule,
    /// Backup -> validation result
    BackupToValidation,
}

/// Entry types
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(required_validations = 1)]
    BackupManifest(BackupManifest),
    #[entry_type(required_validations = 1)]
    BackupChunk(BackupChunk),
    #[entry_type(required_validations = 1)]
    RestoreOperation(RestoreOperation),
    #[entry_type(required_validations = 1)]
    BackupSchedule(BackupSchedule),
}

// ==================== VALIDATION ====================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                validate_create_entry(app_entry, action)
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_entry(
    entry: EntryTypes,
    action: Create,
) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::BackupManifest(manifest) => {
            if manifest.agent != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Backup agent must match author".to_string(),
                ));
            }
            if manifest.backup_id.is_empty() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Backup ID cannot be empty".to_string(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::BackupChunk(chunk) => {
            if chunk.data.is_empty() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Chunk data cannot be empty".to_string(),
                ));
            }
            if chunk.chunk_index >= chunk.total_chunks {
                return Ok(ValidateCallbackResult::Invalid(
                    "Invalid chunk index".to_string(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::RestoreOperation(restore) => {
            if restore.agent != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Restore agent must match author".to_string(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
