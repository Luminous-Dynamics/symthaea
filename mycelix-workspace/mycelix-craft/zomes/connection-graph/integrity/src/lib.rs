#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Connection Graph Integrity Zome
//!
//! Craft network connections and recommendations.
//! Connection counts derived from links (never stored as mutable counters).

use hdi::prelude::*;

/// A connection request between two agents.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConnectionRequest {
    /// Agent who sent the request
    pub sender: AgentPubKey,
    /// Agent who receives the request
    pub target: AgentPubKey,
    /// Optional message with the request
    pub message: Option<String>,
    /// When the request was created
    pub created_at: Timestamp,
    /// Current status
    pub status: ConnectionStatus,
}

/// Status of a connection request
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionStatus {
    Pending,
    Accepted,
    Declined,
}

/// A professional recommendation from one agent to another.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Recommendation {
    /// Agent writing the recommendation
    pub recommender: AgentPubKey,
    /// Agent being recommended
    pub recommended_agent: AgentPubKey,
    /// Relationship context (e.g., "manager at X", "collaborated on Y")
    pub relationship: String,
    /// Recommendation text
    pub text: String,
    /// When written
    pub created_at: Timestamp,
    /// SAP staked on this recommendation (makes lying expensive).
    /// If set, references a CollateralStake in the Finance cluster.
    #[serde(default)]
    pub staked_sap: Option<u32>,
    /// Stake ID in the Finance cluster (for cross-cluster verification).
    #[serde(default)]
    pub stake_id: Option<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    ConnectionRequest(ConnectionRequest),
    #[entry_type(visibility = "public")]
    Recommendation(Recommendation),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Agent -> connection requests they sent
    AgentToOutgoingRequest,
    /// Agent -> connection requests they received
    AgentToIncomingRequest,
    /// Agent -> accepted connections (bidirectional, created on accept)
    AgentToConnection,
    /// Agent -> recommendations they wrote
    AgentToRecommendation,
    /// Agent -> recommendations they received
    RecommendedAgentToRecommendation,
}

pub fn validate_connection_request(req: &ConnectionRequest) -> ExternResult<ValidateCallbackResult> {
    if req.sender == req.target {
        return Ok(ValidateCallbackResult::Invalid("Cannot connect with yourself".into()));
    }
    if let Some(ref msg) = req.message {
        if msg.len() > 500 {
            return Ok(ValidateCallbackResult::Invalid("Connection message cannot exceed 500 characters".into()));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_recommendation(rec: &Recommendation) -> ExternResult<ValidateCallbackResult> {
    if rec.recommender == rec.recommended_agent {
        return Ok(ValidateCallbackResult::Invalid("Cannot recommend yourself".into()));
    }
    if rec.text.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Recommendation text cannot be empty".into()));
    }
    if rec.text.len() > 2_000 {
        return Ok(ValidateCallbackResult::Invalid("Recommendation text cannot exceed 2000 characters".into()));
    }
    if rec.relationship.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Relationship context cannot be empty".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::ConnectionRequest(req) => validate_connection_request(&req),
                    EntryTypes::Recommendation(rec) => validate_recommendation(&rec),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { original_action, action, .. } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid("Only the original author can delete this link".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. } | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. } | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid("Only the original author can update their entries".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid("Only the original author can delete their entries".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_agent(fill: u8) -> AgentPubKey {
        let mut bytes = vec![0x84, 0x20, 0x24]; // AgentPubKey prefix
        bytes.extend(std::iter::repeat(fill).take(36));
        AgentPubKey::from_raw_39(bytes)
    }

    #[test]
    fn test_self_connection_rejected() {
        let agent = test_agent(0);
        let req = ConnectionRequest {
            sender: agent.clone(),
            target: agent,
            message: None,
            created_at: Timestamp::from_micros(0),
            status: ConnectionStatus::Pending,
        };
        assert!(matches!(validate_connection_request(&req).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_self_recommendation_rejected() {
        let agent = test_agent(0);
        let rec = Recommendation {
            recommender: agent.clone(),
            recommended_agent: agent,
            relationship: "self".to_string(),
            text: "I recommend myself".to_string(),
            created_at: Timestamp::from_micros(0),
            staked_sap: None,
            stake_id: None,
        };
        assert!(matches!(validate_recommendation(&rec).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_empty_recommendation_text() {
        let a = test_agent(0);
        let b = test_agent(1);
        let rec = Recommendation {
            recommender: a,
            recommended_agent: b,
            relationship: "colleague".to_string(),
            text: "".to_string(),
            created_at: Timestamp::from_micros(0),
            staked_sap: None,
            stake_id: None,
        };
        assert!(matches!(validate_recommendation(&rec).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_message_too_long() {
        let a = test_agent(0);
        let b = test_agent(1);
        let req = ConnectionRequest {
            sender: a,
            target: b,
            message: Some("x".repeat(501)),
            created_at: Timestamp::from_micros(0),
            status: ConnectionStatus::Pending,
        };
        assert!(matches!(validate_connection_request(&req).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_valid_connection_request() {
        let a = test_agent(0);
        let b = test_agent(1);
        let req = ConnectionRequest {
            sender: a,
            target: b,
            message: Some("Let's connect".to_string()),
            created_at: Timestamp::from_micros(0),
            status: ConnectionStatus::Pending,
        };
        assert_eq!(validate_connection_request(&req).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_valid_recommendation() {
        let a = test_agent(0);
        let b = test_agent(1);
        let rec = Recommendation {
            recommender: a,
            recommended_agent: b,
            relationship: "Collaborated on Mycelix".to_string(),
            text: "Outstanding systems architect with deep Holochain expertise".to_string(),
            created_at: Timestamp::from_micros(0),
            staked_sap: None,
            stake_id: None,
        };
        assert_eq!(validate_recommendation(&rec).unwrap(), ValidateCallbackResult::Valid);
    }

    // ---- Supplementary tests ----

    #[test]
    fn test_connection_status_serde_roundtrip() {
        for s in [ConnectionStatus::Pending, ConnectionStatus::Accepted, ConnectionStatus::Declined] {
            let json = serde_json::to_string(&s).unwrap();
            let back: ConnectionStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn test_no_message_valid() {
        let a = test_agent(0);
        let b = test_agent(1);
        let req = ConnectionRequest {
            sender: a,
            target: b,
            message: None,
            created_at: Timestamp::from_micros(0),
            status: ConnectionStatus::Pending,
        };
        assert_eq!(validate_connection_request(&req).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_message_at_limit_valid() {
        let a = test_agent(0);
        let b = test_agent(1);
        let req = ConnectionRequest {
            sender: a,
            target: b,
            message: Some("x".repeat(500)), // exactly at limit
            created_at: Timestamp::from_micros(0),
            status: ConnectionStatus::Pending,
        };
        assert_eq!(validate_connection_request(&req).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_recommendation_text_too_long() {
        let a = test_agent(0);
        let b = test_agent(1);
        let rec = Recommendation {
            recommender: a,
            recommended_agent: b,
            relationship: "colleague".to_string(),
            text: "x".repeat(2001),
            created_at: Timestamp::from_micros(0),
            staked_sap: None,
            stake_id: None,
        };
        assert!(matches!(validate_recommendation(&rec).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_empty_relationship_rejected() {
        let a = test_agent(0);
        let b = test_agent(1);
        let rec = Recommendation {
            recommender: a,
            recommended_agent: b,
            relationship: "".to_string(),
            text: "Good collaborator".to_string(),
            created_at: Timestamp::from_micros(0),
            staked_sap: None,
            stake_id: None,
        };
        assert!(matches!(validate_recommendation(&rec).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_recommendation_with_staked_sap() {
        let a = test_agent(0);
        let b = test_agent(1);
        let rec = Recommendation {
            recommender: a,
            recommended_agent: b,
            relationship: "manager".to_string(),
            text: "Excellent team lead".to_string(),
            created_at: Timestamp::from_micros(0),
            staked_sap: Some(100),
            stake_id: None,
        };
        assert_eq!(validate_recommendation(&rec).unwrap(), ValidateCallbackResult::Valid);
    }
}
