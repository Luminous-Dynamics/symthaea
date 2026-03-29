// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gradient Storage Zome for ZeroTrustML
//!
//! Stores gradients in the Holochain DHT with metadata for Byzantine resistance

use hdk::prelude::*;
use serde::{Deserialize, Serialize};
use hdk::hash_path::anchor::anchor;

/// Gradient entry stored in the DHT
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GradientEntry {
    /// Node ID that generated this gradient
    pub node_id: u32,

    /// Training round number
    pub round_num: u32,

    /// Gradient data (base64 encoded)
    pub gradient_data: String,

    /// Shape of the gradient tensor
    pub gradient_shape: Vec<usize>,

    /// Data type (e.g., "float32", "float64")
    pub gradient_dtype: String,

    /// Reputation score at time of storage
    pub reputation_score: f32,

    /// Whether gradient passed validation
    pub validation_passed: bool,

    /// Proof of Gradient Quality score
    pub pogq_score: Option<f32>,

    /// Whether anomaly was detected
    pub anomaly_detected: bool,

    /// Whether node is blacklisted
    pub blacklisted: bool,

    /// Timestamp
    pub timestamp: Timestamp,

    /// Optional serialized edge proof (JSON)
    pub edge_proof: Option<String>,

    /// Optional serialized committee votes (JSON array)
    pub committee_votes: Option<String>,
}

/// Input for storing a gradient
#[derive(Serialize, Deserialize, Debug)]
pub struct StoreGradientInput {
    pub node_id: u32,
    pub round_num: u32,
    pub gradient_data: String,
    pub gradient_shape: Vec<usize>,
    pub gradient_dtype: String,
    pub reputation_score: f32,
    pub validation_passed: bool,
    pub pogq_score: Option<f32>,
    pub anomaly_detected: bool,
    pub blacklisted: bool,
    pub edge_proof: Option<String>,
    pub committee_votes: Option<String>,
}

/// Output from storing a gradient
#[derive(Serialize, Deserialize, Debug)]
pub struct StoreGradientOutput {
    pub entry_hash: EntryHash,
    pub action_hash: ActionHash,
}

/// Query parameters for retrieving gradients
#[derive(Serialize, Deserialize, Debug)]
pub struct QueryGradientsInput {
    pub node_id: Option<u32>,
    pub round_num: Option<u32>,
    pub validation_passed: Option<bool>,
    pub blacklisted: Option<bool>,
    pub limit: usize,
}

/// Input for audit trail query
#[derive(Serialize, Deserialize, Debug)]
pub struct AuditTrailInput {
    pub start_timestamp: Timestamp,
    pub end_timestamp: Timestamp,
}

/// Entry types for the zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    GradientEntry(GradientEntry),
}

/// Link types for the zome
#[hdk_link_types]
pub enum LinkTypes {
    NodeToGradient,
    RoundToGradient,
    ValidationStatus,
}

/// Initialize the zome
#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

/// Store a gradient in the DHT
#[hdk_extern]
pub fn store_gradient(input: StoreGradientInput) -> ExternResult<StoreGradientOutput> {
    // Create gradient entry
    let gradient_entry = GradientEntry {
        node_id: input.node_id,
        round_num: input.round_num,
        gradient_data: input.gradient_data,
        gradient_shape: input.gradient_shape,
        gradient_dtype: input.gradient_dtype,
        reputation_score: input.reputation_score,
        validation_passed: input.validation_passed,
        pogq_score: input.pogq_score,
        anomaly_detected: input.anomaly_detected,
        blacklisted: input.blacklisted,
        timestamp: sys_time()?,
        edge_proof: input.edge_proof,
        committee_votes: input.committee_votes,
    };

    // Create entry in DHT
    let action_hash = create_entry(EntryTypes::GradientEntry(gradient_entry.clone()))?;
    let entry_hash = hash_entry(&gradient_entry)?;

    // Create link from node ID to gradient
    let node_anchor = create_or_get_anchor(format!("node_{}", input.node_id))?;
    create_link(
        node_anchor,
        entry_hash.clone(),
        LinkTypes::NodeToGradient,
        (),
    )?;

    // Create link from round number to gradient
    let round_anchor = create_or_get_anchor(format!("round_{}", input.round_num))?;
    create_link(
        round_anchor,
        entry_hash.clone(),
        LinkTypes::RoundToGradient,
        (),
    )?;

    // Create link for validation status
    let validation_anchor = if input.validation_passed {
        create_or_get_anchor("validated".to_string())?
    } else {
        create_or_get_anchor("invalid".to_string())?
    };
    create_link(
        validation_anchor,
        entry_hash.clone(),
        LinkTypes::ValidationStatus,
        (),
    )?;

    Ok(StoreGradientOutput {
        entry_hash,
        action_hash,
    })
}

/// Retrieve a gradient by its hash
#[hdk_extern]
pub fn get_gradient(entry_hash: EntryHash) -> ExternResult<Option<GradientEntry>> {
    let maybe_record = get(entry_hash, GetOptions::default())?;

    match maybe_record {
        Some(record) => {
            let gradient: GradientEntry = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest("Entry not found".into())))?;
            Ok(Some(gradient))
        }
        None => Ok(None),
    }
}

/// Query gradients by node ID
#[hdk_extern]
pub fn get_gradients_by_node(node_id: u32) -> ExternResult<Vec<GradientEntry>> {
    let anchor = create_or_get_anchor(format!("node_{}", node_id))?;
    let links = get_links(GetLinksInputBuilder::try_new(anchor, LinkTypes::NodeToGradient)?.build())?;

    let mut gradients = Vec::new();
    for link in links {
        if let Some(entry_hash) = link.target.into_entry_hash() {
            if let Some(gradient) = get_gradient(entry_hash)? {
                gradients.push(gradient);
            }
        }
    }

    Ok(gradients)
}

/// Query gradients by round number
#[hdk_extern]
pub fn get_gradients_by_round(round_num: u32) -> ExternResult<Vec<GradientEntry>> {
    let anchor = create_or_get_anchor(format!("round_{}", round_num))?;
    let links = get_links(GetLinksInputBuilder::try_new(anchor, LinkTypes::RoundToGradient)?.build())?;

    let mut gradients = Vec::new();
    for link in links {
        if let Some(entry_hash) = link.target.into_entry_hash() {
            if let Some(gradient) = get_gradient(entry_hash)? {
                gradients.push(gradient);
            }
        }
    }

    Ok(gradients)
}

/// Get audit trail for a time range
#[hdk_extern]
pub fn get_audit_trail(input: AuditTrailInput) -> ExternResult<Vec<GradientEntry>> {
    let start_timestamp = input.start_timestamp;
    let end_timestamp = input.end_timestamp;
    // This is a simplified implementation
    // In production, you'd use more sophisticated querying

    // Get all validated gradients
    let validated_anchor = create_or_get_anchor("validated".to_string())?;
    let links = get_links(GetLinksInputBuilder::try_new(validated_anchor, LinkTypes::ValidationStatus)?.build())?;

    let mut trail = Vec::new();
    for link in links {
        if let Some(entry_hash) = link.target.into_entry_hash() {
            if let Some(gradient) = get_gradient(entry_hash)? {
                // Filter by timestamp
                if gradient.timestamp >= start_timestamp && gradient.timestamp <= end_timestamp {
                    trail.push(gradient);
                }
            }
        }
    }

    // Sort by timestamp
    trail.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    Ok(trail)
}

/// Get statistics about stored gradients
#[hdk_extern]
pub fn get_statistics(_: ()) -> ExternResult<GradientStatistics> {
    // Get counts from different anchors
    let validated_anchor = create_or_get_anchor("validated".to_string())?;
    let invalid_anchor = create_or_get_anchor("invalid".to_string())?;

    let validated_links = get_links(GetLinksInputBuilder::try_new(validated_anchor, LinkTypes::ValidationStatus)?.build())?;
    let invalid_links = get_links(GetLinksInputBuilder::try_new(invalid_anchor, LinkTypes::ValidationStatus)?.build())?;

    Ok(GradientStatistics {
        total_gradients: validated_links.len() + invalid_links.len(),
        valid_gradients: validated_links.len(),
        invalid_gradients: invalid_links.len(),
    })
}

/// Statistics about stored gradients
#[derive(Serialize, Deserialize, Debug)]
pub struct GradientStatistics {
    pub total_gradients: usize,
    pub valid_gradients: usize,
    pub invalid_gradients: usize,
}

/// Helper to create or get an anchor
fn create_or_get_anchor(anchor_text: String) -> ExternResult<EntryHash> {
    // Use anchor API to create anchor (HDK 0.4+)
    Ok(anchor(LinkTypes::NodeToGradient, "node".to_string(), anchor_text)?)
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Signal {
    GradientStored { entry_hash: EntryHash },
    ValidationStatusChanged { entry_hash: EntryHash, is_valid: bool },
}

/// Emit signal when gradient is stored
fn emit_gradient_stored_signal(entry_hash: EntryHash) -> ExternResult<()> {
    emit_signal(Signal::GradientStored { entry_hash })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_entry_creation() {
        let gradient = GradientEntry {
            node_id: 1,
            round_num: 1,
            gradient_data: "base64_encoded_data".to_string(),
            gradient_shape: vec![784, 128],
            gradient_dtype: "float32".to_string(),
            reputation_score: 0.8,
            validation_passed: true,
            pogq_score: Some(0.75),
            anomaly_detected: false,
            blacklisted: false,
            timestamp: Timestamp::now(),
        };

        assert_eq!(gradient.node_id, 1);
        assert!(gradient.validation_passed);
    }
}
