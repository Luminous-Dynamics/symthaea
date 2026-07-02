// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! UESS Epistemic Storage - Coordinator Zome
//!
//! Implements the DHT backend API for the Unified Epistemic Storage System.
//! Provides CRUD operations for epistemic entries with E/N/M classification.
//!
//! API Functions:
//! - store_epistemic_entry: Store a new entry
//! - get_epistemic_entry: Retrieve by key
//! - has_epistemic_entry: Check existence
//! - delete_epistemic_entry: Soft delete (tombstone)
//! - list_epistemic_keys: List keys matching pattern
//! - get_storage_stats: Get storage statistics
//! - clear_epistemic_storage: Clear all entries (admin)
//! - get_by_cid: Retrieve by content ID
//! - get_replication_status: Check replication status
//! - ensure_replication: Request additional replication

use hdk::prelude::*;
use epistemic_storage_integrity::*;

// =============================================================================
// Input/Output Types
// =============================================================================

/// Input for storing an epistemic entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreInput {
    pub key: String,
    pub entry: StoreEntryData,
}

/// Entry data for storage
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreEntryData {
    pub data: String,
    pub metadata: StorageMetadata,
}

/// Input for getting an entry by key
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetInput {
    pub key: String,
}

/// Input for getting an entry by CID
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetByCidInput {
    pub cid: String,
}

/// Input for listing keys
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ListKeysInput {
    pub pattern: Option<String>,
}

/// Input for replication operations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicationInput {
    pub key: String,
    pub min_holders: Option<u32>,
}

/// Storage statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StorageStats {
    pub item_count: u64,
    pub total_size_bytes: u64,
    pub oldest_item: Option<i64>,
    pub newest_item: Option<i64>,
}

/// Replication status
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicationStatus {
    pub key: String,
    pub holder_count: u32,
    pub target_holders: u32,
    pub is_replicated: bool,
}

// =============================================================================
// Constants
// =============================================================================

/// Anchor tag for all entries
const ALL_ENTRIES_ANCHOR: &str = "all_epistemic_entries";

// =============================================================================
// Public API
// =============================================================================

/// Store an epistemic entry in the DHT
#[hdk_extern]
pub fn store_epistemic_entry(input: StoreInput) -> ExternResult<ActionHash> {
    // Create the entry
    let entry = EpistemicEntry {
        key: input.key.clone(),
        data: input.entry.data,
        metadata: input.entry.metadata,
    };

    // Create the entry in DHT
    let entry_hash = create_entry(EntryTypes::EpistemicEntry(entry.clone()))?;

    // Create key anchor for lookup
    let key_anchor = KeyAnchor {
        key: input.key.clone(),
    };
    let key_anchor_hash = create_entry(EntryTypes::KeyAnchor(key_anchor))?;

    // Link key anchor to entry
    create_link(
        key_anchor_hash.clone(),
        entry_hash.clone(),
        LinkTypes::KeyToEntry,
        (),
    )?;

    // Create CID anchor for content-addressed lookup
    let cid_anchor = CidAnchor {
        cid: entry.metadata.cid.clone(),
    };
    let cid_anchor_hash = create_entry(EntryTypes::CidAnchor(cid_anchor))?;

    // Link CID anchor to entry
    create_link(
        cid_anchor_hash,
        entry_hash.clone(),
        LinkTypes::CidToEntry,
        (),
    )?;

    // Link to all entries anchor for listing
    let all_anchor = anchor(LinkTypes::AllEntries, ALL_ENTRIES_ANCHOR.to_string())?;
    create_link(all_anchor, entry_hash.clone(), LinkTypes::AllEntries, ())?;

    Ok(entry_hash)
}

/// Get an epistemic entry by key
#[hdk_extern]
pub fn get_epistemic_entry(input: GetInput) -> ExternResult<Option<EpistemicEntry>> {
    // Find the key anchor
    let key_anchor = KeyAnchor {
        key: input.key.clone(),
    };

    // Hash the anchor to find it
    let key_anchor_hash = hash_entry(&key_anchor)?;

    // Get links from key anchor to entry
    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(key_anchor_hash),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::KeyToEntry as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get the most recent link (in case of updates)
    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);

    if let Some(link) = latest_link {
        let entry_hash = ActionHash::try_from(link.target).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to convert link target to ActionHash: {:?}",
                e
            )))
        })?;

        let record = get(entry_hash, GetOptions::default())?;

        if let Some(record) = record {
            let entry: EpistemicEntry = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest(
                        "Entry not found in record".to_string()
                    ))
                })?;

            // Check if tombstoned
            if entry.metadata.tombstone {
                return Ok(None);
            }

            // Check if expired
            if let Some(expires_at) = entry.metadata.expires_at {
                let now = sys_time()?.as_millis() as i64;
                if now > expires_at {
                    return Ok(None);
                }
            }

            return Ok(Some(entry));
        }
    }

    Ok(None)
}

/// Check if an entry exists by key
#[hdk_extern]
pub fn has_epistemic_entry(input: GetInput) -> ExternResult<bool> {
    let entry = get_epistemic_entry(input)?;
    Ok(entry.is_some())
}

/// Delete an epistemic entry (tombstone)
#[hdk_extern]
pub fn delete_epistemic_entry(input: GetInput) -> ExternResult<bool> {
    // Get the existing entry
    let existing = get_epistemic_entry(GetInput {
        key: input.key.clone(),
    })?;

    if existing.is_none() {
        return Ok(false);
    }

    let mut entry = existing.expect("existence verified by is_none check above");

    // Check if E3+ (immutable)
    if entry.metadata.classification.empirical >= 3 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot delete E3+ immutable entries".to_string()
        )));
    }

    // Mark as tombstoned
    entry.metadata.tombstone = true;
    entry.metadata.modified_at = Some(sys_time()?.as_millis() as i64);

    // Find the original entry to update
    let key_anchor = KeyAnchor {
        key: input.key.clone(),
    };
    let key_anchor_hash = hash_entry(&key_anchor)?;

    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(key_anchor_hash.clone()),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::KeyToEntry as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
        let original_hash = ActionHash::try_from(link.target).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to convert link target: {:?}",
                e
            )))
        })?;

        // Update the entry with tombstone
        let new_hash = update_entry(original_hash, EntryTypes::EpistemicEntry(entry))?;

        // Update link to point to new entry
        delete_link(link.create_link_hash, GetOptions::default())?;
        create_link(key_anchor_hash, new_hash, LinkTypes::KeyToEntry, ())?;
    }

    Ok(true)
}

/// List all keys matching a pattern
#[hdk_extern]
pub fn list_epistemic_keys(input: ListKeysInput) -> ExternResult<Vec<String>> {
    let all_anchor = anchor(LinkTypes::AllEntries, ALL_ENTRIES_ANCHOR.to_string())?;

    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(all_anchor),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AllEntries as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut keys: Vec<String> = Vec::new();

    for link in links {
        let entry_hash = ActionHash::try_from(link.target).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to convert link target: {:?}",
                e
            )))
        })?;

        if let Some(record) = get(entry_hash, GetOptions::default())? {
            if let Some(entry) = record
                .entry()
                .to_app_option::<EpistemicEntry>()
                .map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?
            {
                // Skip tombstoned entries
                if entry.metadata.tombstone {
                    continue;
                }

                // Skip expired entries
                if let Some(expires_at) = entry.metadata.expires_at {
                    let now = sys_time()?.as_millis() as i64;
                    if now > expires_at {
                        continue;
                    }
                }

                // Apply pattern filter if provided
                if let Some(ref pattern) = input.pattern {
                    if matches_pattern(&entry.key, pattern) {
                        keys.push(entry.key);
                    }
                } else {
                    keys.push(entry.key);
                }
            }
        }
    }

    Ok(keys)
}

/// Get storage statistics
#[hdk_extern]
pub fn get_storage_stats(_: ()) -> ExternResult<StorageStats> {
    let all_anchor = anchor(LinkTypes::AllEntries, ALL_ENTRIES_ANCHOR.to_string())?;

    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(all_anchor),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AllEntries as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut stats = StorageStats {
        item_count: 0,
        total_size_bytes: 0,
        oldest_item: None,
        newest_item: None,
    };

    for link in links {
        let entry_hash = ActionHash::try_from(link.target).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to convert link target: {:?}",
                e
            )))
        })?;

        if let Some(record) = get(entry_hash, GetOptions::default())? {
            if let Some(entry) = record
                .entry()
                .to_app_option::<EpistemicEntry>()
                .map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?
            {
                // Skip tombstoned entries
                if entry.metadata.tombstone {
                    continue;
                }

                // Skip expired entries
                if let Some(expires_at) = entry.metadata.expires_at {
                    let now = sys_time()?.as_millis() as i64;
                    if now > expires_at {
                        continue;
                    }
                }

                stats.item_count += 1;
                stats.total_size_bytes += entry.metadata.size_bytes;

                let stored_at = entry.metadata.stored_at;
                stats.oldest_item = Some(
                    stats
                        .oldest_item
                        .map_or(stored_at, |old| old.min(stored_at)),
                );
                stats.newest_item = Some(
                    stats
                        .newest_item
                        .map_or(stored_at, |old| old.max(stored_at)),
                );
            }
        }
    }

    Ok(stats)
}

/// Clear all epistemic storage (admin operation)
#[hdk_extern]
pub fn clear_epistemic_storage(_: ()) -> ExternResult<u64> {
    let all_anchor = anchor(LinkTypes::AllEntries, ALL_ENTRIES_ANCHOR.to_string())?;

    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(all_anchor),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AllEntries as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut deleted_count: u64 = 0;

    for link in links {
        // Delete the link to "unlist" the entry
        delete_link(link.create_link_hash, GetOptions::default())?;
        deleted_count += 1;
    }

    Ok(deleted_count)
}

/// Get an entry by CID (content identifier)
#[hdk_extern]
pub fn get_by_cid(input: GetByCidInput) -> ExternResult<Option<EpistemicEntry>> {
    let cid_anchor = CidAnchor {
        cid: input.cid.clone(),
    };

    let cid_anchor_hash = hash_entry(&cid_anchor)?;

    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(cid_anchor_hash),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::CidToEntry as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get the first matching entry (CIDs should be unique)
    if let Some(link) = links.into_iter().next() {
        let entry_hash = ActionHash::try_from(link.target).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to convert link target: {:?}",
                e
            )))
        })?;

        if let Some(record) = get(entry_hash, GetOptions::default())? {
            let entry: EpistemicEntry = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest(
                        "Entry not found in record".to_string()
                    ))
                })?;

            // Check if tombstoned
            if entry.metadata.tombstone {
                return Ok(None);
            }

            return Ok(Some(entry));
        }
    }

    Ok(None)
}

/// Get replication status for an entry
#[hdk_extern]
pub fn get_replication_status(input: GetInput) -> ExternResult<ReplicationStatus> {
    // Check if entry exists
    let exists = has_epistemic_entry(GetInput {
        key: input.key.clone(),
    })?;

    if !exists {
        return Ok(ReplicationStatus {
            key: input.key,
            holder_count: 0,
            target_holders: 0,
            is_replicated: false,
        });
    }

    // In Holochain, replication is automatic based on DHT topology
    // We return an estimate based on network participation
    // TODO: Use actual DHT authority information when available

    let _agent_info = agent_info()?;
    // In a full implementation, we would query actual DHT authority information

    // Estimate: assume at least 3 holders in a healthy network
    let estimated_holders = 3u32;
    let target_holders = 3u32; // Default target for M2

    Ok(ReplicationStatus {
        key: input.key,
        holder_count: estimated_holders,
        target_holders,
        is_replicated: estimated_holders >= target_holders,
    })
}

/// Request additional replication for an entry
#[hdk_extern]
pub fn ensure_replication(input: ReplicationInput) -> ExternResult<bool> {
    // Check if entry exists
    let entry = get_epistemic_entry(GetInput {
        key: input.key.clone(),
    })?;

    if entry.is_none() {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Entry not found: {}",
            input.key
        ))));
    }

    // In Holochain, replication is automatic
    // We can encourage gossip by re-requesting the entry
    // This is a simplified implementation

    // TODO: Implement explicit replication request when Holochain supports it
    // For now, return true to indicate the request was acknowledged

    Ok(true)
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Create an anchor entry and return its hash
fn anchor(_link_type: LinkTypes, anchor_text: String) -> ExternResult<EntryHash> {
    let anchor_entry = KeyAnchor { key: anchor_text };
    let anchor_hash = hash_entry(&anchor_entry)?;

    // Create the anchor if it doesn't exist (idempotent)
    if let Err(e) = create_entry(EntryTypes::KeyAnchor(anchor_entry)) { debug!("Anchor creation warning: {:?}", e); }

    Ok(anchor_hash)
}

/// Simple pattern matching (supports * wildcard)
fn matches_pattern(key: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    // Convert pattern to regex-like matching
    let parts: Vec<&str> = pattern.split('*').collect();

    if parts.len() == 1 {
        // No wildcard, exact match
        return key == pattern;
    }

    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }

        if i == 0 {
            // Must start with this part
            if !key.starts_with(*part) {
                return false;
            }
            pos = part.len();
        } else if i == parts.len() - 1 {
            // Must end with this part
            if !key.ends_with(*part) {
                return false;
            }
        } else {
            // Must contain this part after current position
            if let Some(found_pos) = key[pos..].find(*part) {
                pos = pos + found_pos + part.len();
            } else {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_pattern() {
        assert!(matches_pattern("foo", "foo"));
        assert!(matches_pattern("foo", "*"));
        assert!(matches_pattern("foo", "f*"));
        assert!(matches_pattern("foo", "*o"));
        assert!(matches_pattern("foo", "f*o"));
        assert!(matches_pattern("foobar", "foo*"));
        assert!(matches_pattern("foobar", "*bar"));
        assert!(matches_pattern("foobar", "f*b*r"));

        assert!(!matches_pattern("foo", "bar"));
        assert!(!matches_pattern("foo", "fo"));
        assert!(!matches_pattern("foo", "oo"));
    }
}
