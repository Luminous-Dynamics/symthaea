// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tool Library Integrity Zome
//!
//! Entry types and validation for a community tool lending library.
//! Tracks tools, their condition, availability, and lending history.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

// ============================================================================
// TOOL TYPES
// ============================================================================

/// Category of tool.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolCategory {
    HandTool,
    PowerTool,
    Garden,
    Kitchen,
    Workshop,
    Electronics,
    Other,
}

/// Physical condition of a tool.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolCondition {
    New,
    Good,
    Fair,
    NeedsRepair,
}

/// Geographic location (inline for WASM compatibility).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
}

/// A tool available for community lending.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Tool {
    /// Unique tool identifier.
    pub tool_id: String,
    /// Human-readable name.
    pub name: String,
    /// Description of the tool.
    pub description: String,
    /// Tool category.
    pub category: ToolCategory,
    /// Owner DID.
    pub owner_did: String,
    /// Optional geographic location.
    pub location: Option<GeoLocation>,
    /// Current condition.
    pub condition: ToolCondition,
    /// Whether the tool is currently available for borrowing.
    pub available: bool,
    /// Optional photo hash (entry hash or CID).
    pub photo_hash: Option<String>,
}

// ============================================================================
// LENDING TYPES
// ============================================================================

/// Status of a lending transaction.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum LendingStatus {
    Active,
    Returned,
    Overdue,
    Disputed,
}

/// A lending transaction for a tool.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Lending {
    /// Tool being lent (tool_id).
    pub tool_id: String,
    /// Borrower DID.
    pub borrower_did: String,
    /// Lender DID (tool owner).
    pub lender_did: String,
    /// When the tool was borrowed.
    pub borrowed_at: Timestamp,
    /// When the tool is due back.
    pub due_at: Timestamp,
    /// When the tool was actually returned (None if still out).
    pub returned_at: Option<Timestamp>,
    /// Condition notes on return.
    pub condition_on_return: Option<String>,
    /// Current status.
    pub status: LendingStatus,
}

/// Anchor entry for deterministic link bases.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Anchor(pub String);

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(name = "Tool", visibility = "public")]
    Tool(Tool),
    #[entry_type(name = "Lending", visibility = "public")]
    Lending(Lending),
    #[entry_type(name = "Anchor", visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllTools,
    ToolsByCategory,
    AgentToTool,
    AgentToLending,
    ToolToLending,
    GeoIndex,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::Tool(tool) => validate_tool(&tool),
                    EntryTypes::Lending(lending) => validate_lending(&lending),
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::RegisterCreateLink { .. }
        | FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

fn validate_tool(tool: &Tool) -> ExternResult<ValidateCallbackResult> {
    if tool.tool_id.is_empty() || tool.tool_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tool ID must be 1-256 chars".to_string(),
        ));
    }
    if tool.name.is_empty() || tool.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Name must be 1-256 chars".to_string(),
        ));
    }
    if tool.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description too long (max 4096)".to_string(),
        ));
    }
    if tool.owner_did.is_empty() || tool.owner_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Owner DID must be 1-256 chars".to_string(),
        ));
    }
    if let Some(ref loc) = tool.location {
        if !loc.latitude.is_finite() || loc.latitude < -90.0 || loc.latitude > 90.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Latitude must be in [-90, 90]".to_string(),
            ));
        }
        if !loc.longitude.is_finite() || loc.longitude < -180.0 || loc.longitude > 180.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Longitude must be in [-180, 180]".to_string(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_lending(lending: &Lending) -> ExternResult<ValidateCallbackResult> {
    if lending.tool_id.is_empty() || lending.tool_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tool ID must be 1-256 chars".to_string(),
        ));
    }
    if lending.borrower_did.is_empty() || lending.borrower_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Borrower DID must be 1-256 chars".to_string(),
        ));
    }
    if lending.lender_did.is_empty() || lending.lender_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Lender DID must be 1-256 chars".to_string(),
        ));
    }
    if lending.borrower_did == lending.lender_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot lend a tool to yourself".to_string(),
        ));
    }
    if lending.due_at <= lending.borrowed_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Due date must be after borrow date".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
