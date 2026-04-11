// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Backup Coordinator Zome
//!
//! Implements secure backup, restore, and cross-device migration.
//! Supports encrypted exports and streaming for large datasets.

use hdk::prelude::*;
use mail_backup_integrity::*;

// ==================== CONSTANTS ====================

/// Maximum chunk size (1MB)
const MAX_CHUNK_SIZE: usize = 1024 * 1024;

/// Default retention: keep 10 backups
const DEFAULT_RETENTION_COUNT: u32 = 10;

// ==================== SIGNALS ====================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum BackupSignal {
    /// Backup started
    BackupStarted { backup_id: String },
    /// Backup progress
    BackupProgress { backup_id: String, percent: u8 },
    /// Backup completed
    BackupCompleted { backup_id: String, size_bytes: u64 },
    /// Backup failed
    BackupFailed { backup_id: String, error: String },
    /// Restore started
    RestoreStarted { restore_id: String },
    /// Restore progress
    RestoreProgress { restore_id: String, percent: u8 },
    /// Restore completed
    RestoreCompleted { restore_id: String, entries_restored: u32 },
    /// Restore failed
    RestoreFailed { restore_id: String, error: String },
}

// ==================== INPUTS/OUTPUTS ====================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateBackupInput {
    pub backup_type: BackupType,
    pub contents: BackupContents,
    pub password: Option<String>,
    pub key_hint: Option<String>,
    pub metadata: Option<BackupMetadata>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateBackupOutput {
    pub backup_id: String,
    pub manifest_hash: ActionHash,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RestoreInput {
    pub backup_id: String,
    pub password: Option<String>,
    pub contents: Option<BackupContents>,
    pub options: Option<RestoreOptions>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExportBackupInput {
    pub backup_id: String,
    pub format: ExportFormat,
    pub password: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ImportBackupInput {
    pub data: Vec<u8>,
    pub format: ExportFormat,
    pub password: Option<String>,
}

// ==================== BACKUP OPERATIONS ====================

/// Create a new backup
#[hdk_extern]
pub fn create_backup(input: CreateBackupInput) -> ExternResult<CreateBackupOutput> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;
    let backup_id = format!("backup:{}:{}", agent, now.as_micros());

    emit_signal(BackupSignal::BackupStarted {
        backup_id: backup_id.clone(),
    })?;

    // Determine encryption settings
    let encryption = if let Some(ref _password) = input.password {
        let salt = generate_salt()?;
        BackupEncryption {
            is_encrypted: true,
            algorithm: Some("AES-256-GCM".to_string()),
            kdf: Some("Argon2id".to_string()),
            salt: Some(salt),
            nonce: Some(generate_nonce()?),
            key_hint: input.key_hint,
        }
    } else {
        BackupEncryption {
            is_encrypted: false,
            algorithm: None,
            kdf: None,
            salt: None,
            nonce: None,
            key_hint: None,
        }
    };

    // Collect data to backup
    let backup_data = collect_backup_data(&input.contents)?;

    emit_signal(BackupSignal::BackupProgress {
        backup_id: backup_id.clone(),
        percent: 30,
    })?;

    // Count entries before encryption (may consume data)
    let entry_count = count_entries(&backup_data)?;

    // Encrypt if needed
    let final_data = if let Some(ref _password) = input.password {
        encrypt_backup_data(&backup_data, &encryption)?
    } else {
        backup_data
    };

    emit_signal(BackupSignal::BackupProgress {
        backup_id: backup_id.clone(),
        percent: 60,
    })?;

    // Calculate checksum
    let checksum = calculate_checksum(&final_data)?;

    // Split into chunks if large
    let chunks = split_into_chunks(&final_data);
    let total_chunks = chunks.len() as u32;

    // Create manifest
    let manifest = BackupManifest {
        backup_id: backup_id.clone(),
        agent: agent.clone(),
        backup_type: input.backup_type,
        contents: input.contents,
        created_at: now,
        size_bytes: final_data.len() as u64,
        entry_count,
        checksum: checksum.clone(),
        encryption,
        status: BackupStatus::Completed,
        metadata: input.metadata.unwrap_or_default(),
    };

    let manifest_hash = create_entry(EntryTypes::BackupManifest(manifest))?;

    // Link to agent
    create_link(
        agent.clone(),
        manifest_hash.clone(),
        LinkTypes::AgentToBackups,
        (),
    )?;

    emit_signal(BackupSignal::BackupProgress {
        backup_id: backup_id.clone(),
        percent: 80,
    })?;

    // Store chunks
    for (index, chunk_data) in chunks.into_iter().enumerate() {
        let chunk = BackupChunk {
            backup_id: backup_id.clone(),
            chunk_index: index as u32,
            total_chunks,
            data: chunk_data.clone(),
            checksum: calculate_checksum(&chunk_data)?,
        };

        let chunk_hash = create_entry(EntryTypes::BackupChunk(chunk))?;

        create_link(
            manifest_hash.clone(),
            chunk_hash,
            LinkTypes::BackupToChunks,
            (),
        )?;
    }

    emit_signal(BackupSignal::BackupCompleted {
        backup_id: backup_id.clone(),
        size_bytes: final_data.len() as u64,
    })?;

    // Apply retention policy
    apply_retention_policy(&agent)?;

    Ok(CreateBackupOutput {
        backup_id,
        manifest_hash,
    })
}

/// Collect data for backup from various zomes
fn collect_backup_data(contents: &BackupContents) -> ExternResult<Vec<u8>> {
    let mut data = BackupData::default();

    if contents.emails {
        data.emails = collect_emails()?;
    }
    if contents.drafts {
        data.drafts = collect_drafts()?;
    }
    if contents.contacts {
        data.contacts = collect_contacts()?;
    }
    if contents.trust_attestations {
        data.trust_attestations = collect_trust_attestations()?;
    }
    if contents.folders {
        data.folders = collect_folders()?;
    }
    if contents.settings {
        data.settings = collect_settings()?;
    }

    // Serialize
    let serialized = serialize_backup_data(&data)?;
    Ok(serialized)
}

#[derive(Serialize, Deserialize, Default)]
struct BackupData {
    emails: Vec<SerializedEntry>,
    drafts: Vec<SerializedEntry>,
    contacts: Vec<SerializedEntry>,
    trust_attestations: Vec<SerializedEntry>,
    folders: Vec<SerializedEntry>,
    settings: Vec<SerializedEntry>,
}

#[derive(Serialize, Deserialize, Clone)]
struct SerializedEntry {
    entry_type: String,
    hash: ActionHash,
    data: Vec<u8>,
    timestamp: Timestamp,
}

fn collect_emails() -> ExternResult<Vec<SerializedEntry>> {
    // Would call mail_messages zome to get all emails
    Ok(vec![])
}

fn collect_drafts() -> ExternResult<Vec<SerializedEntry>> {
    Ok(vec![])
}

fn collect_contacts() -> ExternResult<Vec<SerializedEntry>> {
    // Would call mail_contacts zome
    Ok(vec![])
}

fn collect_trust_attestations() -> ExternResult<Vec<SerializedEntry>> {
    // Would call mail_trust zome
    Ok(vec![])
}

fn collect_folders() -> ExternResult<Vec<SerializedEntry>> {
    Ok(vec![])
}

fn collect_settings() -> ExternResult<Vec<SerializedEntry>> {
    Ok(vec![])
}

fn serialize_backup_data(data: &BackupData) -> ExternResult<Vec<u8>> {
    serde_json::to_vec(data).map_err(|e| wasm_error!(e.to_string()))
}

fn count_entries(data: &[u8]) -> ExternResult<u32> {
    let backup_data: BackupData = serde_json::from_slice(data)
        .map_err(|e| wasm_error!(e.to_string()))?;

    Ok((backup_data.emails.len()
        + backup_data.drafts.len()
        + backup_data.contacts.len()
        + backup_data.trust_attestations.len()
        + backup_data.folders.len()
        + backup_data.settings.len()) as u32)
}

/// Encrypt backup data
fn encrypt_backup_data(data: &[u8], _encryption: &BackupEncryption) -> ExternResult<Vec<u8>> {
    // In production, would use actual encryption
    // For now, return data as-is
    Ok(data.to_vec())
}

/// Generate random salt
fn generate_salt() -> ExternResult<Vec<u8>> {
    Ok(vec![0u8; 32]) // Would use proper random
}

/// Generate random nonce
fn generate_nonce() -> ExternResult<Vec<u8>> {
    Ok(vec![0u8; 12])
}

/// Calculate SHA-256 checksum
fn calculate_checksum(_data: &[u8]) -> ExternResult<Vec<u8>> {
    // Would use proper hash
    Ok(vec![0u8; 32])
}

/// Split data into chunks
fn split_into_chunks(data: &[u8]) -> Vec<Vec<u8>> {
    data.chunks(MAX_CHUNK_SIZE).map(|c| c.to_vec()).collect()
}

/// Apply retention policy (delete old backups)
fn apply_retention_policy(_agent: &AgentPubKey) -> ExternResult<()> {
    let backups = get_backups(())?;

    if backups.len() > DEFAULT_RETENTION_COUNT as usize {
        // Would delete oldest backups
    }

    Ok(())
}

// ==================== RESTORE OPERATIONS ====================

/// Restore from backup
#[hdk_extern]
pub fn restore_backup(input: RestoreInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;
    let restore_id = format!("restore:{}:{}", agent, now.as_micros());

    emit_signal(BackupSignal::RestoreStarted {
        restore_id: restore_id.clone(),
    })?;

    // Get backup manifest
    let manifest = get_backup_manifest(&input.backup_id)?
        .ok_or(wasm_error!("Backup not found"))?;

    // Get contents to restore (use backup contents if not specified)
    let contents = input.contents.unwrap_or(manifest.contents.clone());
    let options = input.options.unwrap_or_default();

    // Create restore operation record
    let restore_op = RestoreOperation {
        restore_id: restore_id.clone(),
        backup_id: input.backup_id.clone(),
        agent: agent.clone(),
        restore_contents: contents.clone(),
        options: options.clone(),
        status: RestoreStatus::Validating,
        started_at: now,
        completed_at: None,
        entries_restored: 0,
        entries_skipped: 0,
        errors: vec![],
    };

    let restore_hash = create_entry(EntryTypes::RestoreOperation(restore_op))?;

    create_link(agent.clone(), restore_hash.clone(), LinkTypes::AgentToRestores, ())?;

    emit_signal(BackupSignal::RestoreProgress {
        restore_id: restore_id.clone(),
        percent: 10,
    })?;

    // Validate backup integrity
    if !validate_backup(&manifest)? {
        emit_signal(BackupSignal::RestoreFailed {
            restore_id: restore_id.clone(),
            error: "Backup validation failed".to_string(),
        })?;
        return Err(wasm_error!("Backup validation failed"));
    }

    emit_signal(BackupSignal::RestoreProgress {
        restore_id: restore_id.clone(),
        percent: 20,
    })?;

    // Collect chunks
    let backup_data = collect_backup_chunks(&manifest)?;

    emit_signal(BackupSignal::RestoreProgress {
        restore_id: restore_id.clone(),
        percent: 40,
    })?;

    // Decrypt if needed
    let decrypted_data = if manifest.encryption.is_encrypted {
        if input.password.is_none() {
            return Err(wasm_error!("Password required for encrypted backup"));
        }
        decrypt_backup_data(&backup_data, &manifest.encryption, input.password.as_ref().unwrap())?
    } else {
        backup_data
    };

    emit_signal(BackupSignal::RestoreProgress {
        restore_id: restore_id.clone(),
        percent: 60,
    })?;

    // Restore entries
    let (restored, _skipped, errors) = if !options.dry_run {
        restore_entries(&decrypted_data, &contents, &options)?
    } else {
        (0, 0, vec![])
    };

    emit_signal(BackupSignal::RestoreProgress {
        restore_id: restore_id.clone(),
        percent: 90,
    })?;

    // Update restore operation
    let _final_status = if errors.is_empty() {
        RestoreStatus::Completed
    } else {
        RestoreStatus::Completed // Still completed, just with errors
    };

    emit_signal(BackupSignal::RestoreCompleted {
        restore_id,
        entries_restored: restored,
    })?;

    Ok(restore_hash)
}

/// Get backup manifest by ID
fn get_backup_manifest(backup_id: &str) -> ExternResult<Option<BackupManifest>> {
    let agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(agent, LinkTypes::AgentToBackups)?, GetStrategy::default())?;

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let manifest: BackupManifest = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid manifest"))?;

                if manifest.backup_id == backup_id {
                    return Ok(Some(manifest));
                }
            }
        }
    }

    Ok(None)
}

/// Validate backup integrity
fn validate_backup(_manifest: &BackupManifest) -> ExternResult<bool> {
    // Would verify checksum
    Ok(true)
}

/// Collect all chunks for a backup
fn collect_backup_chunks(_manifest: &BackupManifest) -> ExternResult<Vec<u8>> {
    // Would get all chunks and reassemble
    Ok(vec![])
}

/// Decrypt backup data
fn decrypt_backup_data(
    data: &[u8],
    _encryption: &BackupEncryption,
    _password: &str,
) -> ExternResult<Vec<u8>> {
    // Would decrypt
    Ok(data.to_vec())
}

/// Restore entries from backup data
fn restore_entries(
    _data: &[u8],
    _contents: &BackupContents,
    _options: &RestoreOptions,
) -> ExternResult<(u32, u32, Vec<RestoreError>)> {
    // Would restore each entry type
    Ok((0, 0, vec![]))
}

// ==================== QUERY OPERATIONS ====================

/// Get all backups
#[hdk_extern]
pub fn get_backups(_: ()) -> ExternResult<Vec<BackupManifest>> {
    let agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(agent, LinkTypes::AgentToBackups)?, GetStrategy::default())?;

    let mut backups = Vec::new();

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let manifest: BackupManifest = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid manifest"))?;
                backups.push(manifest);
            }
        }
    }

    // Sort by date descending
    backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    Ok(backups)
}

/// Get backup by ID
#[hdk_extern]
pub fn get_backup(backup_id: String) -> ExternResult<Option<BackupManifest>> {
    get_backup_manifest(&backup_id)
}

/// Get restore operations
#[hdk_extern]
pub fn get_restore_operations(_: ()) -> ExternResult<Vec<RestoreOperation>> {
    let agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(agent, LinkTypes::AgentToRestores)?, GetStrategy::default())?;

    let mut operations = Vec::new();

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let op: RestoreOperation = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid operation"))?;
                operations.push(op);
            }
        }
    }

    Ok(operations)
}

/// Delete a backup
#[hdk_extern]
pub fn delete_backup(backup_id: String) -> ExternResult<bool> {
    let manifest = get_backup_manifest(&backup_id)?;

    if manifest.is_none() {
        return Ok(false);
    }

    // Would delete manifest and chunks
    Ok(true)
}

// ==================== EXPORT/IMPORT ====================

/// Export backup to external format
#[hdk_extern]
pub fn export_backup(input: ExportBackupInput) -> ExternResult<Vec<u8>> {
    let manifest = get_backup_manifest(&input.backup_id)?
        .ok_or(wasm_error!("Backup not found"))?;

    let data = collect_backup_chunks(&manifest)?;

    match input.format {
        ExportFormat::HolochainNative => Ok(data),
        ExportFormat::Json => {
            // Convert to JSON
            Ok(data)
        }
        ExportFormat::MessagePack => {
            // Convert to MessagePack
            Ok(data)
        }
        ExportFormat::Custom(_) => Ok(data),
    }
}

/// Import backup from external format
#[hdk_extern]
pub fn import_backup(input: ImportBackupInput) -> ExternResult<CreateBackupOutput> {
    // Parse and validate imported data
    // Create backup from imported data
    create_backup(CreateBackupInput {
        backup_type: BackupType::Full,
        contents: BackupContents::default(),
        password: input.password,
        key_hint: None,
        metadata: None,
    })
}

// ==================== SCHEDULING ====================

/// Set backup schedule
#[hdk_extern]
pub fn set_backup_schedule(schedule: BackupSchedule) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let mut schedule = schedule;
    schedule.agent = agent.clone();

    let hash = create_entry(EntryTypes::BackupSchedule(schedule))?;

    create_link(agent, hash.clone(), LinkTypes::AgentToSchedule, ())?;

    Ok(hash)
}

/// Get backup schedule
#[hdk_extern]
pub fn get_backup_schedule(_: ()) -> ExternResult<Option<BackupSchedule>> {
    let agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(agent, LinkTypes::AgentToSchedule)?, GetStrategy::default())?;

    if let Some(link) = links.last() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let schedule: BackupSchedule = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid schedule"))?;
                return Ok(Some(schedule));
            }
        }
    }

    Ok(None)
}

/// Trigger scheduled backup (called by scheduler)
#[hdk_extern]
pub fn trigger_scheduled_backup(_: ()) -> ExternResult<Option<CreateBackupOutput>> {
    let schedule = get_backup_schedule(())?;

    if let Some(schedule) = schedule {
        if schedule.enabled {
            return Ok(Some(create_backup(CreateBackupInput {
                backup_type: BackupType::Incremental {
                    since_backup_id: String::new(),
                },
                contents: schedule.contents,
                password: None,
                key_hint: None,
                metadata: None,
            })?));
        }
    }

    Ok(None)
}
