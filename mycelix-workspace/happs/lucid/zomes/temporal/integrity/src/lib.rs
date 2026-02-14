//! LUCID Temporal Integrity Zome
//!
//! Belief versioning and history tracking.
//! Answers: "What did I believe at time X?"

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A snapshot of a thought at a point in time
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BeliefVersion {
    /// The thought this version belongs to
    pub thought_id: String,
    /// Version number
    pub version: u32,
    /// Content at this version
    pub content: String,
    /// Confidence at this version
    pub confidence: f64,
    /// Epistemic classification code (E2N1M2H3)
    pub epistemic_code: String,
    /// Reason for change
    pub change_reason: Option<String>,
    /// Previous version action hash
    pub previous_version: Option<ActionHash>,
    /// Timestamp of this version
    pub timestamp: Timestamp,
}

/// A knowledge graph snapshot at a point in time
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GraphSnapshot {
    /// Unique identifier
    pub id: String,
    /// Snapshot name/label
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Timestamp of snapshot
    pub timestamp: Timestamp,
    /// Number of thoughts at this time
    pub thought_count: u32,
    /// Average confidence
    pub avg_confidence: f64,
    /// Tags for this snapshot
    pub tags: Vec<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    BeliefVersion(BeliefVersion),
    GraphSnapshot(GraphSnapshot),
}

#[hdk_link_types]
pub enum LinkTypes {
    ThoughtToVersions,
    VersionChain,
    AgentToSnapshots,
    SnapshotToThoughts,
    TimestampIndex,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::BeliefVersion(v) => {
                    if v.thought_id.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid("thought_id required".into()));
                    }
                    if v.confidence < 0.0 || v.confidence > 1.0 {
                        return Ok(ValidateCallbackResult::Invalid("confidence must be 0-1".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::GraphSnapshot(s) => {
                    if s.name.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid("name required".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
