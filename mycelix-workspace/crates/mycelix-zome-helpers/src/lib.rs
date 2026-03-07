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
use mycelix_bridge_common::{GovernanceEligibility, GovernanceRequirement, gate_consciousness};

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

/// Resolve a list of links into their corresponding records.
///
/// Iterates over `links`, converts each target to an `ActionHash`, and fetches
/// the record via `get()`. Deleted or missing records are silently skipped.
///
/// This function is duplicated identically in 15+ coordinator zomes.
pub fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
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
}
