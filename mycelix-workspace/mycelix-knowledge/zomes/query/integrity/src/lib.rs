// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Query Integrity Zome
//! Defines entry types and validation for knowledge queries
//!
//! Updated to use HDI 0.7 patterns

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A saved query for reuse
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SavedQuery {
    pub id: String,
    pub name: String,
    pub description: String,
    pub query: String,
    pub parameters: Option<String>,
    pub creator: String,
    pub public: bool,
    pub created: Timestamp,
    pub updated: Timestamp,
}

/// Query execution result (for caching)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct QueryResult {
    pub id: String,
    pub query_id: String,
    pub parameters: Option<String>,
    pub count: u64,
    pub data: String,
    pub execution_time_ms: u64,
    pub executed_at: Timestamp,
    pub expires_at: Option<Timestamp>,
}

/// Query execution plan
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QueryPlan {
    pub steps: Vec<QueryStep>,
    pub estimated_cost: f64,
    pub use_cache: bool,
}

/// A step in the query execution plan
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QueryStep {
    pub step_type: QueryStepType,
    pub target: String,
    pub filters: Vec<QueryFilter>,
}

/// Types of query steps
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum QueryStepType {
    FullScan,
    TagLookup,
    AuthorLookup,
    GraphTraversal,
    EpistemicFilter,
    Join,
}

/// Filter conditions
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QueryFilter {
    pub field: String,
    pub operator: FilterOperator,
    pub value: String,
}

/// Filter operators
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FilterOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    Contains,
    StartsWith,
    EndsWith,
    In,
    Between,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    SavedQuery(SavedQuery),
    QueryResult(QueryResult),
}

#[hdk_link_types]
pub enum LinkTypes {
    CreatorToQuery,
    QueryToResult,
    PublicQueries,
}

/// HDI 0.7 single validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SavedQuery(query) => validate_create_query(action, query),
                EntryTypes::QueryResult(result) => validate_create_result(action, result),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SavedQuery(query) => validate_update_query(action, query, original_action_hash),
                EntryTypes::QueryResult(_) => Ok(ValidateCallbackResult::Invalid(
                    "Query results cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::CreatorToQuery => Ok(ValidateCallbackResult::Valid),
            LinkTypes::QueryToResult => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PublicQueries => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            // Only the original link creator can delete a link
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original link creator can delete a link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(op_update) => {
            // Only the original author can update an entry
            let update_action = match op_update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(update_action.original_action_address.clone())?;
            if update_action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can update an entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(op_delete) => {
            // Only the original author can delete an entry
            let original = must_get_action(op_delete.action.deletes_address.clone())?;
            if op_delete.action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete an entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

fn validate_create_query(_action: Create, query: SavedQuery) -> ExternResult<ValidateCallbackResult> {
    if query.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Query name cannot be empty".into(),
        ));
    }

    if query.query.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Query cannot be empty".into(),
        ));
    }

    if !query.creator.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Creator must be a valid DID".into(),
        ));
    }

    if let Some(ref params) = query.parameters {
        if serde_json::from_str::<serde_json::Value>(params).is_err() {
            return Ok(ValidateCallbackResult::Invalid(
                "Parameters must be valid JSON".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_query(
    _action: Update,
    query: SavedQuery,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_query: SavedQuery = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original query not found".into()
        )))?;

    // Cannot change creator
    if query.creator != original_query.creator {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change query creator".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_result(_action: Create, result: QueryResult) -> ExternResult<ValidateCallbackResult> {
    if serde_json::from_str::<serde_json::Value>(&result.data).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "Result data must be valid JSON".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
