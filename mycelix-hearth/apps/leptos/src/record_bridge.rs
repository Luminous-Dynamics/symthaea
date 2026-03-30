// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Record-to-View translation layer.
//!
//! Holochain zome calls return `Vec<Record>` where Record is:
//! ```ignore
//! {
//!     signed_action: { hashed: { hash: ActionHash, content: Action { ... } } },
//!     entry: { "Present": { "entry_type": "App", "entry": <msgpack bytes> } }
//! }
//! ```
//!
//! This module provides lightweight deserialization of the Record wire format
//! WITHOUT pulling in `hdk` or `holochain_integrity_types` (which are huge
//! and hostile to Leptos WASM builds).
//!
//! Strategy: use serde_json::Value as an intermediate representation,
//! since the conductor's MessagePack → our MessagePack decode → serde works
//! at the Value level. We extract the action hash and entry bytes, then
//! deserialize the entry into our View types.

use serde::{Deserialize, Serialize};
use hearth_leptos_types::*;

// ============================================================================
// Lightweight Record wire types (mirror of Holochain's Record, no hdk dep)
// ============================================================================

/// A Holochain Record as it appears on the MessagePack wire.
/// Uses Value for fields we don't need to fully parse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireRecord {
    pub signed_action: WireSignedAction,
    pub entry: WireRecordEntry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireSignedAction {
    pub hashed: WireHashedAction,
    #[serde(default)]
    pub signature: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireHashedAction {
    /// The ActionHash (39 bytes, usually base64-encoded in JSON context)
    pub hash: Vec<u8>,
    pub content: serde_json::Value,
}

/// The entry field of a Record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WireRecordEntry {
    Present { #[serde(rename = "Present")] present: WireEntry },
    Other(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireEntry {
    pub entry_type: serde_json::Value,
    /// The actual entry data as MessagePack bytes.
    pub entry: Vec<u8>,
}

impl WireRecord {
    /// Get the action hash as a hex string.
    pub fn action_hash_hex(&self) -> String {
        hex_encode(&self.hashed_hash())
    }

    /// Get the action hash as a base64 string (more common in Holochain).
    pub fn action_hash_b64(&self) -> String {
        base64_encode(&self.hashed_hash())
    }

    fn hashed_hash(&self) -> &[u8] {
        &self.signed_action.hashed.hash
    }

    /// Get the author AgentPubKey from the action content.
    pub fn author_b64(&self) -> Option<String> {
        self.signed_action.hashed.content.get("author")
            .and_then(|v| v.as_array())
            .map(|bytes| {
                let raw: Vec<u8> = bytes.iter().filter_map(|b| b.as_u64().map(|n| n as u8)).collect();
                base64_encode(&raw)
            })
    }

    /// Get the timestamp as microseconds (i64).
    pub fn timestamp_micros(&self) -> Option<i64> {
        self.signed_action.hashed.content.get("timestamp")
            .and_then(|v| v.as_i64())
    }

    /// Try to extract and deserialize the entry bytes.
    pub fn decode_entry<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        match &self.entry {
            WireRecordEntry::Present { present } => {
                rmp_serde::from_slice(&present.entry).ok()
            }
            _ => None,
        }
    }
}

// ============================================================================
// Domain-specific conversions: Record → View
// ============================================================================

/// Wire representation of HearthMembership entry (matches the zome's entry).
#[derive(Debug, Clone, Deserialize)]
pub struct WireMembership {
    pub hearth_hash: Vec<u8>,
    pub agent: Vec<u8>,
    pub role: MemberRole,
    pub status: MembershipStatus,
    pub display_name: String,
    pub joined_at: i64, // Timestamp as microseconds
}

/// Wire representation of KinshipBond entry.
#[derive(Debug, Clone, Deserialize)]
pub struct WireBond {
    pub hearth_hash: Vec<u8>,
    pub member_a: Vec<u8>,
    pub member_b: Vec<u8>,
    pub bond_type: BondType,
    pub strength_bp: u32,
    pub last_tended: i64,
    pub created_at: i64,
}

/// Wire representation of GratitudeExpression entry.
#[derive(Debug, Clone, Deserialize)]
pub struct WireGratitude {
    pub hearth_hash: Vec<u8>,
    pub from_agent: Vec<u8>,
    pub to_agent: Vec<u8>,
    pub message: String,
    pub gratitude_type: GratitudeType,
    pub visibility: HearthVisibility,
    pub created_at: i64,
}

/// Convert a vec of Records into MemberViews.
pub fn records_to_members(records: &[WireRecord]) -> Vec<MemberView> {
    records.iter().filter_map(|r| {
        let m: WireMembership = r.decode_entry()?;
        Some(MemberView {
            agent: base64_encode(&m.agent),
            display_name: m.display_name,
            role: m.role,
            status: m.status,
            joined_at: m.joined_at / 1_000_000, // micros → seconds
        })
    }).collect()
}

/// Convert a vec of Records into BondViews.
pub fn records_to_bonds(records: &[WireRecord]) -> Vec<BondView> {
    records.iter().filter_map(|r| {
        let b: WireBond = r.decode_entry()?;
        Some(BondView {
            hash: r.action_hash_b64(),
            member_a: base64_encode(&b.member_a),
            member_b: base64_encode(&b.member_b),
            bond_type: b.bond_type,
            strength_bp: b.strength_bp,
            last_tended: b.last_tended / 1_000_000,
            created_at: b.created_at / 1_000_000,
        })
    }).collect()
}

/// Convert a vec of Records into GratitudeExpressionViews.
pub fn records_to_gratitude(records: &[WireRecord]) -> Vec<GratitudeExpressionView> {
    records.iter().filter_map(|r| {
        let g: WireGratitude = r.decode_entry()?;
        Some(GratitudeExpressionView {
            hash: r.action_hash_b64(),
            from_agent: base64_encode(&g.from_agent),
            to_agent: base64_encode(&g.to_agent),
            message: g.message,
            gratitude_type: g.gratitude_type,
            visibility: g.visibility,
            created_at: g.created_at / 1_000_000,
        })
    }).collect()
}

// ============================================================================
// Helpers
// ============================================================================

fn base64_encode(bytes: &[u8]) -> String {
    // Simple base64 without pulling in a crate.
    // For production, use the `base64` crate.
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity(bytes.len() * 4 / 3 + 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((n >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((n >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(n & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_encodes_correctly() {
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"a"), "YQ==");
    }
}
