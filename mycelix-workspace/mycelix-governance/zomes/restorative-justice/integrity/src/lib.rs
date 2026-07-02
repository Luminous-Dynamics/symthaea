// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use hdi::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum WoundPhase {
    Inflammation,  // Initial detection/commitment
    Proliferation, // Active restorative labor
    Remodeling,    // Community evaluation
    Maturation,    // Final integration/restoration
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WoundRecord {
    pub agent: AgentPubKey,
    pub phase: WoundPhase,
    pub commitment_sap: u64,
    pub detail: String,
    pub created_at: Timestamp,
    pub resolution_score: f64, // Targets the Moral Resonance (Vector 2)
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    WoundRecord(WoundRecord),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToWound,
}
