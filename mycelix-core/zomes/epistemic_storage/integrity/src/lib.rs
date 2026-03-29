// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! UESS Epistemic Storage - Integrity Zome
//!
//! Defines entry types and validation rules for the DHT backend
//! of the Unified Epistemic Storage System.
//!
//! Entry Types:
//! - EpistemicEntry: The core storage unit with E/N/M classification
//! - StorageIndex: Key-to-entry mapping for fast lookups
//!
//! Link Types:
//! - KeyToEntry: Links storage keys to their entries
//! - CidToEntry: Links content IDs to their entries

use hdi::prelude::*;

// =============================================================================
// Entry Types
// =============================================================================

/// Epistemic classification for stored data
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EpistemicClassification {
    /// Empirical level (E0-E4): How verifiable is this data?
    /// E0: Subjective (personal experience)
    /// E1: Observable (witnessed by others)
    /// E2: Privately Verified (cryptographic proof, limited audience)
    /// E3: Cryptographic (publicly verifiable proof)
    /// E4: Reproducible (independently reproducible)
    pub empirical: u8,

    /// Normative level (N0-N3): Who has authority over this data?
    /// N0: Personal (owner only)
    /// N1: Communal (delegated access)
    /// N2: Network (public within network)
    /// N3: Axiomatic (universal truth)
    pub normative: u8,

    /// Materiality level (M0-M3): How long should this data persist?
    /// M0: Ephemeral (session-only)
    /// M1: Temporal (survives restart, not device change)
    /// M2: Persistent (survives node failure)
    /// M3: Foundational (survives network partition)
    pub materiality: u8,
}

/// Schema identity for stored data
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SchemaIdentity {
    /// Schema identifier (e.g., "user_profile", "transaction")
    pub id: String,
    /// Semantic version
    pub version: String,
    /// Optional schema family for alternative worldviews
    pub family: Option<String>,
}

/// Storage metadata for an entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StorageMetadata {
    /// Content identifier (hash of data)
    pub cid: String,
    /// Epistemic classification
    pub classification: EpistemicClassification,
    /// Schema identity
    pub schema: SchemaIdentity,
    /// When this entry was stored (Unix timestamp ms)
    pub stored_at: i64,
    /// When this entry was last modified (Unix timestamp ms)
    pub modified_at: Option<i64>,
    /// Version number for append-only entries
    pub version: u64,
    /// Expiration timestamp (Unix timestamp ms), if any
    pub expires_at: Option<i64>,
    /// Size of the data in bytes
    pub size_bytes: u64,
    /// Agent who created this entry
    pub created_by: String,
    /// Whether this entry has been tombstoned (soft deleted)
    pub tombstone: bool,
    /// If tombstoned, who retracted it
    pub retracted_by: Option<String>,
}

/// The core epistemic storage entry
#[hdk_entry_helper]
#[derive(Clone)]
pub struct EpistemicEntry {
    /// Storage key (unique identifier within namespace)
    pub key: String,
    /// Serialized data (JSON string)
    pub data: String,
    /// Storage metadata
    pub metadata: StorageMetadata,
}

/// Anchor entry for key lookups
#[hdk_entry_helper]
#[derive(Clone)]
pub struct KeyAnchor {
    /// The storage key
    pub key: String,
}

/// Anchor entry for CID lookups
#[hdk_entry_helper]
#[derive(Clone)]
pub struct CidAnchor {
    /// The content identifier
    pub cid: String,
}

// =============================================================================
// Entry Definitions
// =============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(name = "epistemic_entry")]
    EpistemicEntry(EpistemicEntry),

    #[entry_type(name = "key_anchor")]
    KeyAnchor(KeyAnchor),

    #[entry_type(name = "cid_anchor")]
    CidAnchor(CidAnchor),
}

// =============================================================================
// Link Types
// =============================================================================

#[hdk_link_types]
pub enum LinkTypes {
    /// Links a key anchor to an epistemic entry
    KeyToEntry,
    /// Links a CID anchor to an epistemic entry
    CidToEntry,
    /// Links to all entries (for listing)
    AllEntries,
}

// =============================================================================
// Validation
// =============================================================================

/// Validate entry creation
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => validate_create_entry(app_entry),
            OpEntry::UpdateEntry {
                app_entry,
                original_action_hash,
                ..
            } => validate_update_entry(app_entry, original_action_hash),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            ..
        } => validate_create_link(link_type, base_address, target_address),
        FlatOp::RegisterDeleteLink { link_type, .. } => {
            // Allow all link deletions (for tombstoning)
            validate_delete_link(link_type)
        }
        FlatOp::StoreRecord(store_record) => match store_record {
            OpRecord::CreateEntry { app_entry, .. } => validate_create_entry(app_entry),
            OpRecord::UpdateEntry {
                app_entry,
                original_action_hash,
                ..
            } => validate_update_entry(app_entry, original_action_hash),
            OpRecord::DeleteEntry { .. } => {
                // Allow deletions (will be tombstoned)
                Ok(ValidateCallbackResult::Valid)
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_entry(entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::EpistemicEntry(epistemic_entry) => {
            validate_epistemic_entry(&epistemic_entry)
        }
        EntryTypes::KeyAnchor(key_anchor) => validate_key_anchor(&key_anchor),
        EntryTypes::CidAnchor(cid_anchor) => validate_cid_anchor(&cid_anchor),
    }
}

fn validate_update_entry(
    entry: EntryTypes,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::EpistemicEntry(epistemic_entry) => {
            // For E3/E4 entries, updates should be blocked (immutable)
            if epistemic_entry.metadata.classification.empirical >= 3 {
                return Ok(ValidateCallbackResult::Invalid(
                    "E3+ entries are immutable and cannot be updated".to_string(),
                ));
            }
            validate_epistemic_entry(&epistemic_entry)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_epistemic_entry(entry: &EpistemicEntry) -> ExternResult<ValidateCallbackResult> {
    // Validate key is not empty
    if entry.key.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Storage key cannot be empty".to_string(),
        ));
    }

    // Validate key length (max 256 characters)
    if entry.key.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Storage key exceeds maximum length of 256 characters".to_string(),
        ));
    }

    // Validate CID format (should start with "cid:")
    if !entry.metadata.cid.starts_with("cid:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid CID format: must start with 'cid:'".to_string(),
        ));
    }

    // Validate empirical level (0-4)
    if entry.metadata.classification.empirical > 4 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid empirical level: must be 0-4".to_string(),
        ));
    }

    // Validate normative level (0-3)
    if entry.metadata.classification.normative > 3 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid normative level: must be 0-3".to_string(),
        ));
    }

    // Validate materiality level (0-3)
    if entry.metadata.classification.materiality > 3 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid materiality level: must be 0-3".to_string(),
        ));
    }

    // Validate schema ID is not empty
    if entry.metadata.schema.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Schema ID cannot be empty".to_string(),
        ));
    }

    // Validate version is not empty
    if entry.metadata.schema.version.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Schema version cannot be empty".to_string(),
        ));
    }

    // Validate version number is positive
    if entry.metadata.version == 0 && !entry.metadata.tombstone {
        return Ok(ValidateCallbackResult::Invalid(
            "Version must be at least 1 for non-tombstone entries".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_key_anchor(anchor: &KeyAnchor) -> ExternResult<ValidateCallbackResult> {
    if anchor.key.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Key anchor cannot have empty key".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_cid_anchor(anchor: &CidAnchor) -> ExternResult<ValidateCallbackResult> {
    if !anchor.cid.starts_with("cid:") {
        return Ok(ValidateCallbackResult::Invalid(
            "CID anchor must have valid CID format".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_link(
    link_type: LinkTypes,
    _base_address: AnyLinkableHash,
    _target_address: AnyLinkableHash,
) -> ExternResult<ValidateCallbackResult> {
    match link_type {
        LinkTypes::KeyToEntry => Ok(ValidateCallbackResult::Valid),
        LinkTypes::CidToEntry => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllEntries => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_delete_link(_link_type: LinkTypes) -> ExternResult<ValidateCallbackResult> {
    // Allow link deletions (for tombstoning)
    Ok(ValidateCallbackResult::Valid)
}
