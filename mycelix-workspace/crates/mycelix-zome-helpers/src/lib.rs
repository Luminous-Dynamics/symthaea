// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # mycelix-zome-helpers
//!
//! Shared helper functions extracted from Mycelix zome coordinators.
//!
//! These three functions were duplicated across 40+ coordinator zomes with
//! only trivial variation (bridge zome name). This crate provides canonical
//! implementations so coordinators can depend on a single source of truth.
//!
//! ## Migration
//!
//! Consumers can migrate incrementally. For each coordinator:
//!
//! 1. Add `mycelix-zome-helpers = { path = "..." }` to `Cargo.toml`
//! 2. Replace the local `anchor_hash` / `records_from_links` / `require_consciousness`
//!    with `use mycelix_zome_helpers::{anchor_hash, records_from_links, require_consciousness};`
//! 3. For `require_consciousness`, pass the cluster bridge zome name as the first argument.
//!
//! ### `anchor_hash` compatibility note
//!
//! This crate provides a **standalone** `anchor_hash` that uses `blake2b_256`
//! directly on the anchor string bytes. This produces a deterministic
//! `EntryHash` without requiring the zome's `EntryTypes` enum.
//!
//! Most existing coordinators use `hash_entry(&EntryTypes::Anchor(Anchor(s)))`,
//! which serializes the `Anchor` newtype wrapper before hashing — producing a
//! **different** hash. If a zome has existing data anchored with the
//! `hash_entry` approach, switching to this function would break link lookups.
//!
//! For new code or zomes that can re-anchor, use [`anchor_hash`]. For zomes
//! with existing data, keep the local `hash_entry`-based implementation or
//! use [`anchor_hash_of`] with a caller-provided serializable entry.

use hdk::prelude::*;
#[allow(deprecated)] // require_consciousness kept for backward compat
use mycelix_bridge_common::{
    GovernanceEligibility, GovernanceRequirement, gate_consciousness,
    sovereign_gate::{CivicRequirement, gate_civic},
};

// ============================================================================
// Anchor Hashing
// ============================================================================

/// Compute a deterministic `EntryHash` from an anchor string.
///
/// Uses `blake2b_256` directly on the UTF-8 bytes of `anchor_str`, producing a
/// standalone hash that does not depend on any zome's `EntryTypes` enum.
///
/// This matches the approach used in `justice-evidence` and other zomes that
/// need anchors without an `Anchor` entry type.
///
/// **Warning**: This produces a *different* hash than
/// `hash_entry(&EntryTypes::Anchor(Anchor(s)))`. See module docs for details.
pub fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let hash = holo_hash::blake2b_256(anchor_str.as_bytes());
    Ok(EntryHash::from_raw_32(hash.to_vec()))
}

/// Compute an `EntryHash` from a caller-provided entry value.
///
/// This is a thin wrapper around `hash_entry` that lets coordinators pass
/// their zome-specific `EntryTypes::Anchor(Anchor(s))` while still
/// centralising the calling pattern.
///
/// # Example
///
/// ```ignore
/// use mycelix_zome_helpers::anchor_hash_of;
/// let eh = anchor_hash_of(&EntryTypes::Anchor(Anchor("all_items".into())))?;
/// ```
pub fn anchor_hash_of<I, E>(entry: I) -> ExternResult<EntryHash>
where
    Entry: TryFrom<I, Error = E>,
    WasmError: From<E>,
{
    hash_entry(entry)
}

// ============================================================================
// Link Resolution
// ============================================================================

/// Follow the update chain to find the latest version of a record.
///
/// Given an initial `ActionHash`, fetches the record details and recursively
/// follows any updates to return the most recent version. Returns `None` if
/// the record has been deleted or not found.
///
/// This function is duplicated identically in 20+ coordinator zomes across
/// commons, civic, hearth, and other clusters.
pub fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update =
                    &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

/// Resolve a list of links into their corresponding records, following update chains.
///
/// Iterates over `links`, converts each target to an `ActionHash`, and fetches
/// the latest version of the record via [`get_latest_record`]. Deleted or
/// missing records are silently skipped.
///
/// This function is duplicated identically in 20+ coordinator zomes.
pub fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            records.push(record);
        }
    }
    Ok(records)
}

/// Resolve a list of links into their corresponding records, requiring all to exist.
///
/// Like [`records_from_links`], but returns an error if any link target is missing
/// or deleted. Use this when the caller expects all links to resolve (e.g., when
/// the link set was just queried and entries should not have been deleted between).
pub fn records_from_links_strict(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        let record = get_latest_record(action_hash.clone())?
            .ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Record not found for link target: {:?}",
                    action_hash
                )))
            })?;
        records.push(record);
    }
    Ok(records)
}

// ============================================================================
// Input Validation Helpers
// ============================================================================

/// Validate that a string field is not empty.
///
/// Returns a `WasmError::Guest` with the field name in the message.
///
/// # Example
///
/// ```ignore
/// use mycelix_zome_helpers::validate_non_empty;
/// validate_non_empty(&input.title, "title")?;
/// ```
pub fn validate_non_empty(value: &str, field_name: &str) -> ExternResult<()> {
    if value.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "{} cannot be empty",
            field_name
        ))));
    }
    Ok(())
}

/// Validate that a string field does not exceed a maximum byte length.
///
/// # Example
///
/// ```ignore
/// use mycelix_zome_helpers::validate_max_bytes;
/// validate_max_bytes(&input.description, 4096, "description")?;
/// ```
pub fn validate_max_bytes(value: &str, max: usize, field_name: &str) -> ExternResult<()> {
    if value.len() > max {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "{} too long ({} bytes, max {})",
            field_name,
            value.len(),
            max
        ))));
    }
    Ok(())
}

/// Validate that a collection's length is within a given range.
///
/// # Example
///
/// ```ignore
/// use mycelix_zome_helpers::validate_length_range;
/// validate_length_range(&input.outcomes, 2, 100, "outcomes")?;
/// ```
pub fn validate_length_range<T>(
    data: &[T],
    min: usize,
    max: usize,
    field_name: &str,
) -> ExternResult<()> {
    if data.len() < min {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "{} must have at least {} items, got {}",
            field_name, min,
            data.len()
        ))));
    }
    if data.len() > max {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "{} must have at most {} items, got {}",
            field_name, max,
            data.len()
        ))));
    }
    Ok(())
}

/// Shorthand macro for returning a `WasmError::Guest` error.
///
/// Reduces the boilerplate of `wasm_error!(WasmErrorInner::Guest(...))` which
/// is repeated hundreds of times across coordinator zomes.
///
/// # Examples
///
/// ```ignore
/// use mycelix_zome_helpers::bail_guest;
///
/// // Simple string
/// bail_guest!("Not found");
///
/// // With format args
/// bail_guest!("Market {} not found", market_id);
/// ```
#[macro_export]
macro_rules! bail_guest {
    ($msg:expr) => {
        return Err(::hdk::prelude::wasm_error!(
            ::hdk::prelude::WasmErrorInner::Guest($msg.into())
        ))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err(::hdk::prelude::wasm_error!(
            ::hdk::prelude::WasmErrorInner::Guest(format!($fmt, $($arg)*))
        ))
    };
}

// ============================================================================
// Consciousness Gating
// ============================================================================

/// Gate an action behind consciousness-level governance.
///
/// Delegates to [`mycelix_bridge_common::gate_consciousness`], passing the
/// cluster-specific bridge zome name. Each cluster uses a different bridge:
///
/// - Commons: `"commons_bridge"`
/// - Civic: `"civic_bridge"`
/// - Hearth: `"hearth_bridge"`
/// - Personal: `"personal_bridge"`
///
/// # Arguments
///
/// * `bridge_zome` - The bridge zome name for this cluster (e.g. `"commons_bridge"`)
/// * `requirement` - The governance requirement level for this action
/// * `action_name` - Human-readable name of the action being gated (for audit)
///
/// # Example
///
/// ```ignore
/// use mycelix_zome_helpers::require_consciousness;
/// use mycelix_bridge_common::requirement_for_basic;
///
/// let eligibility = require_consciousness("commons_bridge", &requirement_for_basic(), "register_source")?;
/// ```
pub fn require_consciousness(
    bridge_zome: &str,
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness(bridge_zome, requirement, action_name)
}

/// Civic gating — 8D replacement for `require_consciousness`.
///
/// Drop-in successor that accepts `CivicRequirement` directly, eliminating
/// the per-zome conversion shim. Delegates to `gate_civic()` which handles
/// backward compatibility internally.
pub fn require_civic(
    bridge_zome: &str,
    requirement: &CivicRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_civic(bridge_zome, requirement, action_name)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_hash_deterministic() {
        let h1 = holo_hash::blake2b_256("all_sources".as_bytes());
        let h2 = holo_hash::blake2b_256("all_sources".as_bytes());
        assert_eq!(h1, h2, "Same input must produce same hash");
    }

    #[test]
    fn anchor_hash_distinct_for_different_inputs() {
        let h1 = holo_hash::blake2b_256("all_sources".as_bytes());
        let h2 = holo_hash::blake2b_256("source_type:River".as_bytes());
        assert_ne!(h1, h2, "Different inputs must produce different hashes");
    }

    #[test]
    fn anchor_hash_produces_32_bytes() {
        let h = holo_hash::blake2b_256("test_anchor".as_bytes());
        assert_eq!(h.len(), 32, "blake2b_256 must produce 32 bytes");
    }

    #[test]
    fn anchor_hash_empty_string() {
        // Empty string is a valid input (edge case).
        let h = holo_hash::blake2b_256("".as_bytes());
        assert_eq!(h.len(), 32);
    }

    // NOTE: `records_from_links` and `require_consciousness` call HDK host
    // functions (`get`, `agent_info`, `call`, etc.) that are only available
    // inside a running Holochain conductor. They cannot be unit-tested outside
    // of a sweettest / tryorama harness. The functions are trivial wrappers,
    // so the risk of bugs is low. Integration testing should happen via
    // sweettest when migrating coordinators.

    #[test]
    fn records_from_links_empty_vec() {
        // An empty link list should always succeed and return an empty vec.
        // This does NOT call the HDK because the loop body is never entered.
        let result = records_from_links(vec![]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn records_from_links_strict_empty_vec() {
        let result = records_from_links_strict(vec![]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    // ---- Validation helpers ----

    #[test]
    fn validate_non_empty_accepts_content() {
        assert!(validate_non_empty("hello", "name").is_ok());
    }

    #[test]
    fn validate_non_empty_rejects_empty() {
        let err = validate_non_empty("", "title").unwrap_err();
        assert!(format!("{:?}", err).contains("title"));
    }

    #[test]
    fn validate_non_empty_rejects_whitespace() {
        let err = validate_non_empty("   ", "reason").unwrap_err();
        assert!(format!("{:?}", err).contains("reason"));
    }

    #[test]
    fn validate_max_bytes_accepts_within_limit() {
        assert!(validate_max_bytes("short", 100, "field").is_ok());
    }

    #[test]
    fn validate_max_bytes_accepts_at_limit() {
        let s = "x".repeat(256);
        assert!(validate_max_bytes(&s, 256, "field").is_ok());
    }

    #[test]
    fn validate_max_bytes_rejects_over_limit() {
        let s = "x".repeat(257);
        let err = validate_max_bytes(&s, 256, "description").unwrap_err();
        assert!(format!("{:?}", err).contains("description"));
        assert!(format!("{:?}", err).contains("257"));
    }

    #[test]
    fn validate_length_range_accepts_within() {
        let items = vec![1, 2, 3];
        assert!(validate_length_range(&items, 2, 5, "outcomes").is_ok());
    }

    #[test]
    fn validate_length_range_rejects_too_few() {
        let items = vec![1];
        let err = validate_length_range(&items, 2, 5, "outcomes").unwrap_err();
        assert!(format!("{:?}", err).contains("at least 2"));
    }

    #[test]
    fn validate_length_range_rejects_too_many() {
        let items = vec![1, 2, 3, 4, 5, 6];
        let err = validate_length_range(&items, 2, 5, "photos").unwrap_err();
        assert!(format!("{:?}", err).contains("at most 5"));
    }

    #[test]
    fn validate_length_range_at_boundaries() {
        let min_items = vec![1, 2];
        assert!(validate_length_range(&min_items, 2, 5, "f").is_ok());
        let max_items = vec![1, 2, 3, 4, 5];
        assert!(validate_length_range(&max_items, 2, 5, "f").is_ok());
    }
}
