// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Anchor utilities for consistent indexing
//!
//! Uses blake2b_256 hashing for deterministic entry hashes,
//! extracted from mycelix-property/zomes/shared.

/// Create a deterministic entry hash for an anchor string
pub fn anchor_hash(anchor_str: &str) -> hdk::prelude::ExternResult<hdk::prelude::EntryHash> {
    let hash = holo_hash::blake2b_256(anchor_str.as_bytes());
    // from_raw_36 expects 32 hash bytes + 4 location bytes
    let mut raw = hash.to_vec();
    raw.extend_from_slice(&[0u8; 4]);
    Ok(hdk::prelude::EntryHash::from_raw_36(raw))
}
