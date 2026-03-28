// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Matching Integrity Zome
//! Defines entry types and validation for intelligent care matching.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Status of a care match
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MatchStatus {
    Suggested,
    Accepted,
    Declined,
    Completed,
}

/// Factors contributing to a match score
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MatchFactors {
    /// How close the provider is to the requester (0.0 - 1.0)
    pub proximity_score: f32,
    /// How well the provider's skills align with the request (0.0 - 1.0)
    pub skill_alignment: f32,
    /// How compatible the schedules are (0.0 - 1.0)
    pub schedule_compatibility: f32,
    /// Trust/reputation score between the agents (0.0 - 1.0)
    pub trust_score: f32,
}

/// A match between a service offer and a service request
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CareMatch {
    /// Hash of the ServiceOffer
    pub offer_hash: ActionHash,
    /// Hash of the ServiceRequest
    pub request_hash: ActionHash,
    /// Provider agent
    pub provider: AgentPubKey,
    /// Requester agent
    pub requester: AgentPubKey,
    /// Overall match score (0.0 - 1.0)
    pub score: f32,
    /// Breakdown of contributing factors
    pub factors: MatchFactors,
    /// Current status of the match
    pub status: MatchStatus,
    /// When the match was suggested
    pub created_at: Timestamp,
    /// When the match status was last updated
    pub updated_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    CareMatch(CareMatch),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Request to suggested matches
    RequestToMatch,
    /// Offer to suggested matches
    OfferToMatch,
    /// Agent to matches they are involved in (as provider)
    AgentProviderMatches,
    /// Agent to matches they are involved in (as requester)
    AgentRequesterMatches,
    /// All pending matches
    AllPendingMatches,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareMatch(care_match) => validate_create_match(action, care_match),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareMatch(care_match) => validate_update_match(care_match),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => match link_type {
            LinkTypes::RequestToMatch => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "RequestToMatch link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::OfferToMatch => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "OfferToMatch link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentProviderMatches => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentProviderMatches link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentRequesterMatches => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentRequesterMatches link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllPendingMatches => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllPendingMatches link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
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

fn validate_create_match(
    _action: Create,
    care_match: CareMatch,
) -> ExternResult<ValidateCallbackResult> {
    if !care_match.score.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Score must be a finite number".into(),
        ));
    }
    if care_match.score < 0.0 || care_match.score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Match score must be between 0.0 and 1.0".into(),
        ));
    }
    if !care_match.factors.proximity_score.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proximity score must be a finite number".into(),
        ));
    }
    if care_match.factors.proximity_score < 0.0 || care_match.factors.proximity_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Proximity score must be between 0.0 and 1.0".into(),
        ));
    }
    if !care_match.factors.skill_alignment.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Skill alignment must be a finite number".into(),
        ));
    }
    if care_match.factors.skill_alignment < 0.0 || care_match.factors.skill_alignment > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Skill alignment must be between 0.0 and 1.0".into(),
        ));
    }
    if !care_match.factors.schedule_compatibility.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Schedule compatibility must be a finite number".into(),
        ));
    }
    if care_match.factors.schedule_compatibility < 0.0
        || care_match.factors.schedule_compatibility > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Schedule compatibility must be between 0.0 and 1.0".into(),
        ));
    }
    if !care_match.factors.trust_score.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score must be a finite number".into(),
        ));
    }
    if care_match.factors.trust_score < 0.0 || care_match.factors.trust_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score must be between 0.0 and 1.0".into(),
        ));
    }
    if care_match.provider == care_match.requester {
        return Ok(ValidateCallbackResult::Invalid(
            "Provider and requester cannot be the same agent".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_match(care_match: CareMatch) -> ExternResult<ValidateCallbackResult> {
    if !care_match.score.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Score must be a finite number".into(),
        ));
    }
    if care_match.score < 0.0 || care_match.score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Match score must be between 0.0 and 1.0".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Factory functions for test data

    fn valid_factors() -> MatchFactors {
        MatchFactors {
            proximity_score: 0.8,
            skill_alignment: 0.9,
            schedule_compatibility: 0.7,
            trust_score: 0.85,
        }
    }

    fn agent_key_1() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xdb; 36])
    }

    fn agent_key_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xca; 36])
    }

    fn action_hash_1() -> ActionHash {
        ActionHash::from_raw_36(vec![0xa1; 36])
    }

    fn action_hash_2() -> ActionHash {
        ActionHash::from_raw_36(vec![0xa2; 36])
    }

    fn timestamp_now() -> Timestamp {
        Timestamp::from_micros(1_000_000) // 1 second in microseconds
    }

    fn valid_match() -> CareMatch {
        CareMatch {
            offer_hash: action_hash_1(),
            request_hash: action_hash_2(),
            provider: agent_key_1(),
            requester: agent_key_2(),
            score: 0.8,
            factors: valid_factors(),
            status: MatchStatus::Suggested,
            created_at: timestamp_now(),
            updated_at: timestamp_now(),
        }
    }

    fn create_action() -> Create {
        Create {
            author: agent_key_1(),
            timestamp: timestamp_now(),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 1.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        }
    }

    // Tests for MatchFactors structure

    #[test]
    fn test_match_factors_valid() {
        let factors = valid_factors();
        assert_eq!(factors.proximity_score, 0.8);
        assert_eq!(factors.skill_alignment, 0.9);
        assert_eq!(factors.schedule_compatibility, 0.7);
        assert_eq!(factors.trust_score, 0.85);
    }

    #[test]
    fn test_match_factors_serde_roundtrip() {
        let factors = valid_factors();
        let serialized = serde_json::to_string(&factors).unwrap();
        let deserialized: MatchFactors = serde_json::from_str(&serialized).unwrap();
        assert_eq!(factors, deserialized);
    }

    #[test]
    fn test_match_factors_all_zeros() {
        let factors = MatchFactors {
            proximity_score: 0.0,
            skill_alignment: 0.0,
            schedule_compatibility: 0.0,
            trust_score: 0.0,
        };
        assert_eq!(factors.proximity_score, 0.0);
    }

    #[test]
    fn test_match_factors_all_ones() {
        let factors = MatchFactors {
            proximity_score: 1.0,
            skill_alignment: 1.0,
            schedule_compatibility: 1.0,
            trust_score: 1.0,
        };
        assert_eq!(factors.proximity_score, 1.0);
    }

    // Tests for MatchStatus enum

    #[test]
    fn test_match_status_suggested() {
        let status = MatchStatus::Suggested;
        assert_eq!(status, MatchStatus::Suggested);
    }

    #[test]
    fn test_match_status_accepted() {
        let status = MatchStatus::Accepted;
        assert_eq!(status, MatchStatus::Accepted);
    }

    #[test]
    fn test_match_status_declined() {
        let status = MatchStatus::Declined;
        assert_eq!(status, MatchStatus::Declined);
    }

    #[test]
    fn test_match_status_completed() {
        let status = MatchStatus::Completed;
        assert_eq!(status, MatchStatus::Completed);
    }

    #[test]
    fn test_match_status_serde_suggested() {
        let status = MatchStatus::Suggested;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: MatchStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_match_status_serde_accepted() {
        let status = MatchStatus::Accepted;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: MatchStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_match_status_serde_declined() {
        let status = MatchStatus::Declined;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: MatchStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_match_status_serde_completed() {
        let status = MatchStatus::Completed;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: MatchStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    // Tests for CareMatch structure

    #[test]
    fn test_care_match_valid() {
        let care_match = valid_match();
        assert_eq!(care_match.provider, agent_key_1());
        assert_eq!(care_match.requester, agent_key_2());
        assert_eq!(care_match.score, 0.8);
    }

    #[test]
    fn test_care_match_serde_roundtrip() {
        let care_match = valid_match();
        let serialized = serde_json::to_string(&care_match).unwrap();
        let deserialized: CareMatch = serde_json::from_str(&serialized).unwrap();
        assert_eq!(care_match, deserialized);
    }

    // Tests for validate_create_match - Valid cases

    #[test]
    fn test_validate_create_match_valid() {
        let result = validate_create_match(create_action(), valid_match()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_create_match_score_zero() {
        let mut care_match = valid_match();
        care_match.score = 0.0;
        let result = validate_create_match(create_action(), care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_create_match_score_one() {
        let mut care_match = valid_match();
        care_match.score = 1.0;
        let result = validate_create_match(create_action(), care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_create_match_all_factors_zero() {
        let mut care_match = valid_match();
        care_match.factors.proximity_score = 0.0;
        care_match.factors.skill_alignment = 0.0;
        care_match.factors.schedule_compatibility = 0.0;
        care_match.factors.trust_score = 0.0;
        let result = validate_create_match(create_action(), care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_create_match_all_factors_one() {
        let mut care_match = valid_match();
        care_match.factors.proximity_score = 1.0;
        care_match.factors.skill_alignment = 1.0;
        care_match.factors.schedule_compatibility = 1.0;
        care_match.factors.trust_score = 1.0;
        let result = validate_create_match(create_action(), care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // Tests for validate_create_match - Invalid score

    #[test]
    fn test_validate_create_match_score_negative() {
        let mut care_match = valid_match();
        care_match.score = -0.001;
        let result = validate_create_match(create_action(), care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Match score must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_validate_create_match_score_too_high() {
        let mut care_match = valid_match();
        care_match.score = 1.001;
        let result = validate_create_match(create_action(), care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Match score must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    // Tests for validate_create_match - Invalid proximity_score

    #[test]
    fn test_validate_create_match_proximity_score_negative() {
        let mut care_match = valid_match();
        care_match.factors.proximity_score = -0.001;
        let result = validate_create_match(create_action(), care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Proximity score must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_validate_create_match_proximity_score_too_high() {
        let mut care_match = valid_match();
        care_match.factors.proximity_score = 1.001;
        let result = validate_create_match(create_action(), care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Proximity score must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_validate_create_match_proximity_score_zero() {
        let mut care_match = valid_match();
        care_match.factors.proximity_score = 0.0;
        let result = validate_create_match(create_action(), care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_create_match_proximity_score_one() {
        let mut care_match = valid_match();
        care_match.factors.proximity_score = 1.0;
        let result = validate_create_match(create_action(), care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // Tests for validate_create_match - Invalid skill_alignment

    #[test]
    fn test_validate_create_match_skill_alignment_negative() {
        let mut care_match = valid_match();
        care_match.factors.skill_alignment = -0.001;
        let result = validate_create_match(create_action(), care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Skill alignment must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_validate_create_match_skill_alignment_too_high() {
        let mut care_match = valid_match();
        care_match.factors.skill_alignment = 1.001;
        let result = validate_create_match(create_action(), care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Skill alignment must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_validate_create_match_skill_alignment_zero() {
        let mut care_match = valid_match();
        care_match.factors.skill_alignment = 0.0;
        let result = validate_create_match(create_action(), care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_create_match_skill_alignment_one() {
        let mut care_match = valid_match();
        care_match.factors.skill_alignment = 1.0;
        let result = validate_create_match(create_action(), care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // Tests for validate_create_match - Invalid schedule_compatibility

    #[test]
    fn test_validate_create_match_schedule_compatibility_negative() {
        let mut care_match = valid_match();
        care_match.factors.schedule_compatibility = -0.001;
        let result = validate_create_match(create_action(), care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Schedule compatibility must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_validate_create_match_schedule_compatibility_too_high() {
        let mut care_match = valid_match();
        care_match.factors.schedule_compatibility = 1.001;
        let result = validate_create_match(create_action(), care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Schedule compatibility must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_validate_create_match_schedule_compatibility_zero() {
        let mut care_match = valid_match();
        care_match.factors.schedule_compatibility = 0.0;
        let result = validate_create_match(create_action(), care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_create_match_schedule_compatibility_one() {
        let mut care_match = valid_match();
        care_match.factors.schedule_compatibility = 1.0;
        let result = validate_create_match(create_action(), care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // Tests for validate_create_match - Invalid trust_score

    #[test]
    fn test_validate_create_match_trust_score_negative() {
        let mut care_match = valid_match();
        care_match.factors.trust_score = -0.001;
        let result = validate_create_match(create_action(), care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Trust score must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_validate_create_match_trust_score_too_high() {
        let mut care_match = valid_match();
        care_match.factors.trust_score = 1.001;
        let result = validate_create_match(create_action(), care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Trust score must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_validate_create_match_trust_score_zero() {
        let mut care_match = valid_match();
        care_match.factors.trust_score = 0.0;
        let result = validate_create_match(create_action(), care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_create_match_trust_score_one() {
        let mut care_match = valid_match();
        care_match.factors.trust_score = 1.0;
        let result = validate_create_match(create_action(), care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // Tests for validate_create_match - Self-match prevention

    #[test]
    fn test_validate_create_match_self_match() {
        let mut care_match = valid_match();
        care_match.requester = care_match.provider.clone();
        let result = validate_create_match(create_action(), care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Provider and requester cannot be the same agent"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_validate_create_match_different_agents() {
        let care_match = valid_match();
        assert_ne!(care_match.provider, care_match.requester);
        let result = validate_create_match(create_action(), care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // Tests for validate_update_match - Valid cases

    #[test]
    fn test_validate_update_match_valid() {
        let result = validate_update_match(valid_match()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_update_match_score_zero() {
        let mut care_match = valid_match();
        care_match.score = 0.0;
        let result = validate_update_match(care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_update_match_score_one() {
        let mut care_match = valid_match();
        care_match.score = 1.0;
        let result = validate_update_match(care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_update_match_score_mid() {
        let mut care_match = valid_match();
        care_match.score = 0.5;
        let result = validate_update_match(care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // Tests for validate_update_match - Invalid score

    #[test]
    fn test_validate_update_match_score_negative() {
        let mut care_match = valid_match();
        care_match.score = -0.001;
        let result = validate_update_match(care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Match score must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_validate_update_match_score_too_high() {
        let mut care_match = valid_match();
        care_match.score = 1.001;
        let result = validate_update_match(care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Match score must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_validate_update_match_score_far_negative() {
        let mut care_match = valid_match();
        care_match.score = -1.0;
        let result = validate_update_match(care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Match score must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_validate_update_match_score_far_too_high() {
        let mut care_match = valid_match();
        care_match.score = 2.0;
        let result = validate_update_match(care_match).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Match score must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    // Tests for validate_update_match - Update doesn't validate factors or self-match

    #[test]
    fn test_validate_update_match_ignores_invalid_factors() {
        let mut care_match = valid_match();
        care_match.factors.proximity_score = -0.5; // Invalid, but not checked in update
        let result = validate_update_match(care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_update_match_ignores_self_match() {
        let mut care_match = valid_match();
        care_match.requester = care_match.provider.clone(); // Invalid, but not checked in update
        let result = validate_update_match(care_match).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // Edge case tests

    #[test]
    fn test_care_match_clone() {
        let care_match = valid_match();
        let cloned = care_match.clone();
        assert_eq!(care_match, cloned);
    }

    #[test]
    fn test_match_factors_clone() {
        let factors = valid_factors();
        let cloned = factors.clone();
        assert_eq!(factors, cloned);
    }

    // ============================================================================
    // Link Tag Validation Tests
    // ============================================================================

    fn validate_create_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::RequestToMatch => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "RequestToMatch link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::OfferToMatch => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "OfferToMatch link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentProviderMatches => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentProviderMatches link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentRequesterMatches => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentRequesterMatches link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllPendingMatches => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllPendingMatches link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn test_link_tag_request_to_match_at_limit() {
        let result = validate_create_link_tag(LinkTypes::RequestToMatch, vec![0u8; 256]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_request_to_match_over_limit() {
        let result = validate_create_link_tag(LinkTypes::RequestToMatch, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_agent_provider_matches_at_limit() {
        let result =
            validate_create_link_tag(LinkTypes::AgentProviderMatches, vec![0u8; 256]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_agent_provider_matches_over_limit() {
        let result =
            validate_create_link_tag(LinkTypes::AgentProviderMatches, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_all_pending_matches_at_limit() {
        let result =
            validate_create_link_tag(LinkTypes::AllPendingMatches, vec![0u8; 256]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_all_pending_matches_over_limit() {
        let result =
            validate_create_link_tag(LinkTypes::AllPendingMatches, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_empty_tag_valid() {
        let result = validate_create_link_tag(LinkTypes::RequestToMatch, vec![]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_anchor_creation() {
        let anchor = Anchor("test-anchor".to_string());
        assert_eq!(anchor.0, "test-anchor");
    }

    #[test]
    fn test_anchor_clone() {
        let anchor = Anchor("test-anchor".to_string());
        let cloned = anchor.clone();
        assert_eq!(anchor, cloned);
    }
}
