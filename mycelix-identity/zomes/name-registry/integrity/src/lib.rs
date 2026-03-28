// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdi::prelude::*;

/// A registered mesh name binding.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct MeshNameEntry {
    /// Name segments (e.g., ["joburg", "water", "tank-7"]).
    pub segments: Vec<String>,
    /// Canonical form (e.g., "mycelix://joburg/water/tank-7").
    pub canonical: String,
    /// Endpoint type: "iroh", "lora", "holochain", "ip".
    pub endpoint_type: String,
    /// Endpoint data.
    pub endpoint_data: String,
    /// Registration timestamp (µs since epoch).
    pub registered_at: u64,
    /// Expiry timestamp (µs since epoch, default 1 year from registration).
    pub expires_at: u64,
}

/// Transfer of name ownership.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct NameTransfer {
    /// Action hash of the MeshNameEntry being transferred.
    pub name_hash: ActionHash,
    /// New owner agent.
    pub new_owner: AgentPubKey,
    /// Transfer timestamp.
    pub timestamp_us: u64,
}

/// Anchor entry for deterministic link bases.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(name = "MeshNameEntry", visibility = "public")]
    MeshNameEntry(MeshNameEntry),
    #[entry_type(name = "NameTransfer", visibility = "public")]
    NameTransfer(NameTransfer),
    #[entry_type(name = "Anchor", visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    NamePath,
    AgentToNames,
    NameToTransfers,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::MeshNameEntry(entry) => {
                        // Max depth 5
                        if entry.segments.len() > 5 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Name depth exceeds max 5".to_string(),
                            ));
                        }
                        // Validate segments
                        for seg in &entry.segments {
                            if seg.is_empty() || seg.len() > 63 {
                                return Ok(ValidateCallbackResult::Invalid(format!(
                                    "Segment '{}' invalid length",
                                    seg
                                )));
                            }
                            if !seg
                                .chars()
                                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
                            {
                                return Ok(ValidateCallbackResult::Invalid(format!(
                                    "Segment '{}' contains invalid chars",
                                    seg
                                )));
                            }
                            if seg.starts_with('-') || seg.ends_with('-') {
                                return Ok(ValidateCallbackResult::Invalid(format!(
                                    "Segment '{}' cannot start/end with hyphen",
                                    seg
                                )));
                            }
                        }
                        // Endpoint type validation
                        if !["iroh", "lora", "holochain", "ip"]
                            .contains(&entry.endpoint_type.as_str())
                        {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Invalid endpoint type".to_string(),
                            ));
                        }
                        Ok(ValidateCallbackResult::Valid)
                    }
                    EntryTypes::NameTransfer(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. }
        | FlatOp::RegisterDeleteLink { .. }
        | FlatOp::StoreRecord(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_)
        | FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
    }
}
