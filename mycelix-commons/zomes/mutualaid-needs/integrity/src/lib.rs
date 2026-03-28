// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Needs Integrity Zome
//!
//! This zome defines entry types and validation rules for needs matching
//! in the Mycelix Mutual Aid hApp. Supports needs, offers, matches, and fulfillments.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};
use mutualaid_common::*;

/// Entry types for the needs zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A need from a member
    #[entry_type(visibility = "public")]
    Need(Need),
    /// An offer from a member
    #[entry_type(visibility = "public")]
    Offer(Offer),
    /// A match between need and offer
    #[entry_type(visibility = "public")]
    Match(Match),
    /// Fulfillment record
    #[entry_type(visibility = "public")]
    Fulfillment(Fulfillment),
}

/// Link types for the needs zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from agent to their needs
    AgentToNeeds,
    /// Link from agent to their offers
    AgentToOffers,
    /// Link from category anchor to needs
    CategoryToNeeds,
    /// Link from category anchor to offers
    CategoryToOffers,
    /// Link from need to its matches
    NeedToMatches,
    /// Link from offer to its matches
    OfferToMatches,
    /// Link from match to fulfillment
    MatchToFulfillment,
    /// Link for all needs discovery
    AllNeeds,
    /// Link for all offers discovery
    AllOffers,
    /// Link for emergency needs
    EmergencyNeeds,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => validate_create_entry(app_entry),
            OpEntry::UpdateEntry { app_entry, .. } => validate_create_entry(app_entry),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            tag,
            ..
        } => validate_create_link(link_type, base_address, target_address, tag),
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
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
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate entry creation
fn validate_create_entry(entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::Need(need) => validate_need(need),
        EntryTypes::Offer(offer) => validate_offer(offer),
        EntryTypes::Match(m) => validate_match(m),
        EntryTypes::Fulfillment(fulfillment) => validate_fulfillment(fulfillment),
    }
}

/// Validate a need
fn validate_need(need: Need) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if need.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Need ID cannot be empty".to_string(),
        ));
    }

    // ID length limit
    if need.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Need ID exceeds 256 character limit".to_string(),
        ));
    }

    // Title must not be empty
    if need.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Need title cannot be empty".to_string(),
        ));
    }

    // Title length limit
    if need.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Need title exceeds 256 character limit".to_string(),
        ));
    }

    // Description length limit
    if need.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Need description exceeds 4096 character limit".to_string(),
        ));
    }

    // Reciprocity offers limit
    if need.reciprocity_offers.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 100 reciprocity offers".to_string(),
        ));
    }

    // Individual reciprocity offer length limit
    for offer in &need.reciprocity_offers {
        if offer.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Reciprocity offer exceeds 256 character limit".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate an offer
fn validate_offer(offer: Offer) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if offer.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer ID cannot be empty".to_string(),
        ));
    }

    // ID length limit
    if offer.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer ID exceeds 256 character limit".to_string(),
        ));
    }

    // Title must not be empty
    if offer.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer title cannot be empty".to_string(),
        ));
    }

    // Title length limit
    if offer.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer title exceeds 256 character limit".to_string(),
        ));
    }

    // Description length limit
    if offer.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Offer description exceeds 4096 character limit".to_string(),
        ));
    }

    // Asking for limit
    if offer.asking_for.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 100 asking for items".to_string(),
        ));
    }

    // Individual asking for item length limit
    for item in &offer.asking_for {
        if item.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Asking for item exceeds 256 character limit".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a match
fn validate_match(m: Match) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if m.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Match ID cannot be empty".to_string(),
        ));
    }

    // ID length limit
    if m.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Match ID exceeds 256 character limit".to_string(),
        ));
    }

    // Requester and offerer must be different
    if m.requester == m.offerer {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester and offerer must be different".to_string(),
        ));
    }

    // Notes length limit
    if m.notes.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Match notes exceeds 4096 character limit".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a fulfillment
fn validate_fulfillment(fulfillment: Fulfillment) -> ExternResult<ValidateCallbackResult> {
    // Notes length limit
    if fulfillment.notes.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Fulfillment notes exceeds 4096 character limit".to_string(),
        ));
    }

    // Gratitude message length limit
    if let Some(ref msg) = fulfillment.gratitude_message {
        if msg.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Gratitude message exceeds 4096 character limit".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate link creation
fn validate_create_link(
    link_type: LinkTypes,
    _base_address: AnyLinkableHash,
    _target_address: AnyLinkableHash,
    tag: LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    match link_type {
        LinkTypes::AgentToNeeds => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "AgentToNeeds link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::AgentToOffers => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "AgentToOffers link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::CategoryToNeeds => {
            // Category tags may carry category metadata
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "CategoryToNeeds link tag too long (max 512 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::CategoryToOffers => {
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "CategoryToOffers link tag too long (max 512 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::NeedToMatches => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "NeedToMatches link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::OfferToMatches => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "OfferToMatches link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::MatchToFulfillment => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "MatchToFulfillment link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::AllNeeds => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "AllNeeds link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::AllOffers => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "AllOffers link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::EmergencyNeeds => {
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "EmergencyNeeds link tag too long (max 512 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Factory Functions
    // =========================================================================

    fn agent_1() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xdb; 36])
    }

    fn agent_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xdc; 36])
    }

    fn action_hash_1() -> ActionHash {
        ActionHash::from_raw_36(vec![0xab; 36])
    }

    fn action_hash_2() -> ActionHash {
        ActionHash::from_raw_36(vec![0xac; 36])
    }

    fn valid_need() -> Need {
        Need {
            id: "need_001".to_string(),
            requester: agent_1(),
            category: NeedCategory::Food,
            title: "Need groceries".to_string(),
            description: "Looking for food assistance".to_string(),
            urgency: UrgencyLevel::Medium,
            emergency: false,
            quantity: Some(5),
            location: LocationConstraint::Remote,
            needed_by: None,
            reciprocity_offers: vec!["Can help with gardening".to_string()],
            status: NeedStatus::Open,
            created_at: Timestamp::from_micros(1000000),
        }
    }

    fn valid_offer() -> Offer {
        Offer {
            id: "offer_001".to_string(),
            offerer: agent_1(),
            category: NeedCategory::Food,
            title: "Offering groceries".to_string(),
            description: "Have extra produce to share".to_string(),
            quantity: Some(10),
            condition: Some("Fresh".to_string()),
            location: LocationConstraint::Remote,
            available_until: None,
            asking_for: vec!["Gardening help".to_string()],
            status: OfferStatus::Available,
            created_at: Timestamp::from_micros(1000000),
        }
    }

    fn valid_match() -> Match {
        Match {
            id: "match_001".to_string(),
            need_hash: action_hash_1(),
            offer_hash: action_hash_2(),
            requester: agent_1(),
            offerer: agent_2(),
            status: MatchStatus::Proposed,
            quantity: Some(5),
            notes: "Looks like a good match".to_string(),
            matched_at: Timestamp::from_micros(1000000),
            scheduled_handoff: None,
            handoff_location: None,
        }
    }

    fn valid_fulfillment() -> Fulfillment {
        Fulfillment {
            match_hash: action_hash_1(),
            quantity_given: Some(5),
            notes: "Successfully completed".to_string(),
            requester_confirmed: true,
            offerer_confirmed: true,
            fulfilled_at: Timestamp::from_micros(1000000),
            gratitude_message: Some("Thank you so much!".to_string()),
        }
    }

    // =========================================================================
    // Need Validation Tests
    // =========================================================================

    #[test]
    fn test_valid_need() {
        let need = valid_need();
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_need_empty_id() {
        let mut need = valid_need();
        need.id = "".to_string();
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_need_empty_title() {
        let mut need = valid_need();
        need.title = "".to_string();
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_need_title_at_boundary() {
        let mut need = valid_need();
        need.title = "a".repeat(256);
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_need_title_exceeds_limit() {
        let mut need = valid_need();
        need.title = "a".repeat(257);
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_need_title_one_char() {
        let mut need = valid_need();
        need.title = "a".to_string();
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_need_description_at_boundary() {
        let mut need = valid_need();
        need.description = "b".repeat(4096);
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_need_description_exceeds_limit() {
        let mut need = valid_need();
        need.description = "b".repeat(4097);
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_need_description_empty() {
        let mut need = valid_need();
        need.description = "".to_string();
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_need_reciprocity_offers_at_boundary() {
        let mut need = valid_need();
        need.reciprocity_offers = (0..100).map(|i| format!("offer_{}", i)).collect();
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_need_reciprocity_offers_exceeds_limit() {
        let mut need = valid_need();
        need.reciprocity_offers = (0..101).map(|i| format!("offer_{}", i)).collect();
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_need_reciprocity_offers_empty() {
        let mut need = valid_need();
        need.reciprocity_offers = vec![];
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_need_emergency_flag() {
        let mut need = valid_need();
        need.emergency = true;
        need.urgency = UrgencyLevel::Emergency;
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // =========================================================================
    // Offer Validation Tests
    // =========================================================================

    #[test]
    fn test_valid_offer() {
        let offer = valid_offer();
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_offer_empty_id() {
        let mut offer = valid_offer();
        offer.id = "".to_string();
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_offer_empty_title() {
        let mut offer = valid_offer();
        offer.title = "".to_string();
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_offer_title_at_boundary() {
        let mut offer = valid_offer();
        offer.title = "c".repeat(256);
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_offer_title_exceeds_limit() {
        let mut offer = valid_offer();
        offer.title = "c".repeat(257);
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_offer_title_one_char() {
        let mut offer = valid_offer();
        offer.title = "x".to_string();
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_offer_description_at_boundary() {
        let mut offer = valid_offer();
        offer.description = "d".repeat(4096);
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_offer_description_exceeds_limit() {
        let mut offer = valid_offer();
        offer.description = "d".repeat(4097);
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_offer_description_empty() {
        let mut offer = valid_offer();
        offer.description = "".to_string();
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_offer_asking_for_at_boundary() {
        let mut offer = valid_offer();
        offer.asking_for = (0..100).map(|i| format!("item_{}", i)).collect();
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_offer_asking_for_exceeds_limit() {
        let mut offer = valid_offer();
        offer.asking_for = (0..101).map(|i| format!("item_{}", i)).collect();
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_offer_asking_for_empty() {
        let mut offer = valid_offer();
        offer.asking_for = vec![];
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // =========================================================================
    // Match Validation Tests
    // =========================================================================

    #[test]
    fn test_valid_match() {
        let m = valid_match();
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_match_empty_id() {
        let mut m = valid_match();
        m.id = "".to_string();
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_match_same_requester_and_offerer() {
        let mut m = valid_match();
        m.requester = agent_1();
        m.offerer = agent_1();
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_match_different_requester_and_offerer() {
        let mut m = valid_match();
        m.requester = agent_1();
        m.offerer = agent_2();
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_match_notes_at_boundary() {
        let mut m = valid_match();
        m.notes = "e".repeat(4096);
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_match_notes_exceeds_limit() {
        let mut m = valid_match();
        m.notes = "e".repeat(4097);
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_match_notes_empty() {
        let mut m = valid_match();
        m.notes = "".to_string();
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // =========================================================================
    // Fulfillment Validation Tests
    // =========================================================================

    #[test]
    fn test_valid_fulfillment() {
        let fulfillment = valid_fulfillment();
        let result = validate_fulfillment(fulfillment);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fulfillment_notes_at_boundary() {
        let mut fulfillment = valid_fulfillment();
        fulfillment.notes = "f".repeat(4096);
        let result = validate_fulfillment(fulfillment);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fulfillment_notes_exceeds_limit() {
        let mut fulfillment = valid_fulfillment();
        fulfillment.notes = "f".repeat(4097);
        let result = validate_fulfillment(fulfillment);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fulfillment_notes_empty() {
        let mut fulfillment = valid_fulfillment();
        fulfillment.notes = "".to_string();
        let result = validate_fulfillment(fulfillment);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fulfillment_gratitude_message_at_boundary() {
        let mut fulfillment = valid_fulfillment();
        fulfillment.gratitude_message = Some("g".repeat(4096));
        let result = validate_fulfillment(fulfillment);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fulfillment_gratitude_message_exceeds_limit() {
        let mut fulfillment = valid_fulfillment();
        fulfillment.gratitude_message = Some("g".repeat(4097));
        let result = validate_fulfillment(fulfillment);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fulfillment_gratitude_message_none() {
        let mut fulfillment = valid_fulfillment();
        fulfillment.gratitude_message = None;
        let result = validate_fulfillment(fulfillment);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fulfillment_gratitude_message_empty_string() {
        let mut fulfillment = valid_fulfillment();
        fulfillment.gratitude_message = Some("".to_string());
        let result = validate_fulfillment(fulfillment);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // =========================================================================
    // Enum Serde Tests
    // =========================================================================

    #[test]
    fn test_need_category_serde_roundtrip() {
        let categories = vec![
            NeedCategory::Food,
            NeedCategory::Clothing,
            NeedCategory::Housing,
            NeedCategory::Healthcare,
            NeedCategory::Transportation,
            NeedCategory::Custom("CustomCategory".to_string()),
        ];

        for category in categories {
            let json = serde_json::to_string(&category).unwrap();
            let parsed: NeedCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(category, parsed);
        }
    }

    #[test]
    fn test_need_status_serde_roundtrip() {
        let statuses = vec![
            NeedStatus::Open,
            NeedStatus::PartiallyMet,
            NeedStatus::Matched,
            NeedStatus::Fulfilled,
            NeedStatus::Withdrawn,
            NeedStatus::Expired,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let parsed: NeedStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, parsed);
        }
    }

    #[test]
    fn test_offer_status_serde_roundtrip() {
        let statuses = vec![
            OfferStatus::Available,
            OfferStatus::Reserved,
            OfferStatus::Claimed,
            OfferStatus::Completed,
            OfferStatus::Withdrawn,
            OfferStatus::Expired,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let parsed: OfferStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, parsed);
        }
    }

    #[test]
    fn test_match_status_serde_roundtrip() {
        let statuses = vec![
            MatchStatus::Proposed,
            MatchStatus::Accepted,
            MatchStatus::Scheduled,
            MatchStatus::InProgress,
            MatchStatus::Completed,
            MatchStatus::Cancelled,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let parsed: MatchStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, parsed);
        }
    }

    #[test]
    fn test_urgency_level_serde_roundtrip() {
        let levels = vec![
            UrgencyLevel::Low,
            UrgencyLevel::Medium,
            UrgencyLevel::High,
            UrgencyLevel::Urgent,
            UrgencyLevel::Emergency,
        ];

        for level in levels {
            let json = serde_json::to_string(&level).unwrap();
            let parsed: UrgencyLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(level, parsed);
        }
    }

    // =========================================================================
    // Complex Scenario Tests
    // =========================================================================

    #[test]
    fn test_need_all_fields_at_boundaries() {
        let mut need = valid_need();
        need.id = "i".repeat(64);
        need.title = "t".repeat(256);
        need.description = "d".repeat(4096);
        need.reciprocity_offers = (0..100).map(|i| format!("offer_{}", i)).collect();
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_offer_all_fields_at_boundaries() {
        let mut offer = valid_offer();
        offer.id = "i".repeat(64);
        offer.title = "t".repeat(256);
        offer.description = "d".repeat(4096);
        offer.asking_for = (0..100).map(|i| format!("item_{}", i)).collect();
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_match_all_fields_at_boundaries() {
        let mut m = valid_match();
        m.id = "i".repeat(64);
        m.notes = "n".repeat(4096);
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fulfillment_all_fields_at_boundaries() {
        let mut fulfillment = valid_fulfillment();
        fulfillment.notes = "n".repeat(4096);
        fulfillment.gratitude_message = Some("g".repeat(4096));
        let result = validate_fulfillment(fulfillment);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_need_with_unicode_title() {
        let mut need = valid_need();
        need.title = "需要食物 🍞".to_string();
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_offer_with_unicode_description() {
        let mut offer = valid_offer();
        offer.description = "新鮮的蔬菜 🥕🥬".to_string();
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_match_with_unicode_notes() {
        let mut m = valid_match();
        m.notes = "良好的匹配 👍".to_string();
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fulfillment_with_unicode_gratitude() {
        let mut fulfillment = valid_fulfillment();
        fulfillment.gratitude_message = Some("非常感謝！🙏".to_string());
        let result = validate_fulfillment(fulfillment);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // =========================================================================
    // Validation Result Message Tests
    // =========================================================================

    #[test]
    fn test_need_empty_id_error_message() {
        let mut need = valid_need();
        need.id = "".to_string();
        if let Ok(ValidateCallbackResult::Invalid(msg)) = validate_need(need) {
            assert_eq!(msg, "Need ID cannot be empty");
        } else {
            panic!("Expected Invalid result");
        }
    }

    #[test]
    fn test_offer_title_exceeds_error_message() {
        let mut offer = valid_offer();
        offer.title = "x".repeat(257);
        if let Ok(ValidateCallbackResult::Invalid(msg)) = validate_offer(offer) {
            assert_eq!(msg, "Offer title exceeds 256 character limit");
        } else {
            panic!("Expected Invalid result");
        }
    }

    #[test]
    fn test_match_same_agents_error_message() {
        let mut m = valid_match();
        m.requester = agent_1();
        m.offerer = agent_1();
        if let Ok(ValidateCallbackResult::Invalid(msg)) = validate_match(m) {
            assert_eq!(msg, "Requester and offerer must be different");
        } else {
            panic!("Expected Invalid result");
        }
    }

    #[test]
    fn test_fulfillment_gratitude_exceeds_error_message() {
        let mut fulfillment = valid_fulfillment();
        fulfillment.gratitude_message = Some("g".repeat(4097));
        if let Ok(ValidateCallbackResult::Invalid(msg)) = validate_fulfillment(fulfillment) {
            assert_eq!(msg, "Gratitude message exceeds 4096 character limit");
        } else {
            panic!("Expected Invalid result");
        }
    }

    // =========================================================================
    // Additional Validation Tests
    // =========================================================================

    #[test]
    fn test_need_minimal_valid() {
        let need = Need {
            id: "n".to_string(),
            requester: agent_1(),
            category: NeedCategory::Food,
            title: "x".to_string(),
            description: "".to_string(),
            urgency: UrgencyLevel::Low,
            emergency: false,
            quantity: None,
            location: LocationConstraint::Remote,
            needed_by: None,
            reciprocity_offers: vec![],
            status: NeedStatus::Open,
            created_at: Timestamp::from_micros(1000000),
        };
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_offer_minimal_valid() {
        let offer = Offer {
            id: "o".to_string(),
            offerer: agent_1(),
            category: NeedCategory::Food,
            title: "x".to_string(),
            description: "".to_string(),
            quantity: None,
            condition: None,
            location: LocationConstraint::Remote,
            available_until: None,
            asking_for: vec![],
            status: OfferStatus::Available,
            created_at: Timestamp::from_micros(1000000),
        };
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_match_minimal_valid() {
        let m = Match {
            id: "m".to_string(),
            need_hash: action_hash_1(),
            offer_hash: action_hash_2(),
            requester: agent_1(),
            offerer: agent_2(),
            status: MatchStatus::Proposed,
            quantity: None,
            notes: "".to_string(),
            matched_at: Timestamp::from_micros(1000000),
            scheduled_handoff: None,
            handoff_location: None,
        };
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fulfillment_minimal_valid() {
        let fulfillment = Fulfillment {
            match_hash: action_hash_1(),
            quantity_given: None,
            notes: "".to_string(),
            requester_confirmed: false,
            offerer_confirmed: false,
            fulfilled_at: Timestamp::from_micros(1000000),
            gratitude_message: None,
        };
        let result = validate_fulfillment(fulfillment);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_need_with_all_custom_categories() {
        let mut need = valid_need();
        need.category = NeedCategory::Custom("SpecialNeed".to_string());
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_offer_with_custom_category() {
        let mut offer = valid_offer();
        offer.category = NeedCategory::Custom("SpecialOffer".to_string());
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_need_title_exactly_255_chars() {
        let mut need = valid_need();
        need.title = "x".repeat(255);
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_offer_title_exactly_255_chars() {
        let mut offer = valid_offer();
        offer.title = "y".repeat(255);
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_need_description_exactly_4095_chars() {
        let mut need = valid_need();
        need.description = "z".repeat(4095);
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_offer_description_exactly_4095_chars() {
        let mut offer = valid_offer();
        offer.description = "w".repeat(4095);
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_match_notes_exactly_4095_chars() {
        let mut m = valid_match();
        m.notes = "v".repeat(4095);
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fulfillment_notes_exactly_4095_chars() {
        let mut fulfillment = valid_fulfillment();
        fulfillment.notes = "u".repeat(4095);
        let result = validate_fulfillment(fulfillment);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fulfillment_gratitude_exactly_4095_chars() {
        let mut fulfillment = valid_fulfillment();
        fulfillment.gratitude_message = Some("t".repeat(4095));
        let result = validate_fulfillment(fulfillment);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_need_reciprocity_offers_exactly_99_items() {
        let mut need = valid_need();
        need.reciprocity_offers = (0..99).map(|i| format!("offer_{}", i)).collect();
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_offer_asking_for_exactly_99_items() {
        let mut offer = valid_offer();
        offer.asking_for = (0..99).map(|i| format!("item_{}", i)).collect();
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_need_description_exceeds_error_message() {
        let mut need = valid_need();
        need.description = "x".repeat(4097);
        if let Ok(ValidateCallbackResult::Invalid(msg)) = validate_need(need) {
            assert_eq!(msg, "Need description exceeds 4096 character limit");
        } else {
            panic!("Expected Invalid result");
        }
    }

    #[test]
    fn test_need_title_empty_error_message() {
        let mut need = valid_need();
        need.title = "".to_string();
        if let Ok(ValidateCallbackResult::Invalid(msg)) = validate_need(need) {
            assert_eq!(msg, "Need title cannot be empty");
        } else {
            panic!("Expected Invalid result");
        }
    }

    #[test]
    fn test_offer_empty_id_error_message() {
        let mut offer = valid_offer();
        offer.id = "".to_string();
        if let Ok(ValidateCallbackResult::Invalid(msg)) = validate_offer(offer) {
            assert_eq!(msg, "Offer ID cannot be empty");
        } else {
            panic!("Expected Invalid result");
        }
    }

    #[test]
    fn test_match_empty_id_error_message() {
        let mut m = valid_match();
        m.id = "".to_string();
        if let Ok(ValidateCallbackResult::Invalid(msg)) = validate_match(m) {
            assert_eq!(msg, "Match ID cannot be empty");
        } else {
            panic!("Expected Invalid result");
        }
    }

    #[test]
    fn test_match_notes_exceeds_error_message() {
        let mut m = valid_match();
        m.notes = "x".repeat(4097);
        if let Ok(ValidateCallbackResult::Invalid(msg)) = validate_match(m) {
            assert_eq!(msg, "Match notes exceeds 4096 character limit");
        } else {
            panic!("Expected Invalid result");
        }
    }

    #[test]
    fn test_fulfillment_notes_exceeds_error_message() {
        let mut fulfillment = valid_fulfillment();
        fulfillment.notes = "x".repeat(4097);
        if let Ok(ValidateCallbackResult::Invalid(msg)) = validate_fulfillment(fulfillment) {
            assert_eq!(msg, "Fulfillment notes exceeds 4096 character limit");
        } else {
            panic!("Expected Invalid result");
        }
    }

    #[test]
    fn test_need_reciprocity_exceeds_error_message() {
        let mut need = valid_need();
        need.reciprocity_offers = (0..101).map(|i| format!("offer_{}", i)).collect();
        if let Ok(ValidateCallbackResult::Invalid(msg)) = validate_need(need) {
            assert_eq!(msg, "Cannot have more than 100 reciprocity offers");
        } else {
            panic!("Expected Invalid result");
        }
    }

    #[test]
    fn test_offer_asking_for_exceeds_error_message() {
        let mut offer = valid_offer();
        offer.asking_for = (0..101).map(|i| format!("item_{}", i)).collect();
        if let Ok(ValidateCallbackResult::Invalid(msg)) = validate_offer(offer) {
            assert_eq!(msg, "Cannot have more than 100 asking for items");
        } else {
            panic!("Expected Invalid result");
        }
    }

    // =========================================================================
    // Location Constraint Tests
    // =========================================================================

    #[test]
    fn test_need_with_various_locations() {
        let locations = vec![
            LocationConstraint::Remote,
            LocationConstraint::FixedLocation("123 Main St".to_string()),
            LocationConstraint::WithinRadius {
                geohash: "u4pruyd".to_string(),
                radius_km: 5.0,
            },
            LocationConstraint::AtRequester,
            LocationConstraint::AtProvider,
            LocationConstraint::ToBeArranged,
        ];

        for location in locations {
            let mut need = valid_need();
            need.location = location;
            let result = validate_need(need);
            assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
        }
    }

    // =========================================================================
    // Link Tag Validation Tests
    // =========================================================================

    #[test]
    fn test_link_agent_to_needs_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::AgentToNeeds, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_agent_to_needs_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::AgentToNeeds, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_category_to_needs_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 512]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::CategoryToNeeds, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_category_to_needs_tag_too_long() {
        let tag = LinkTag(vec![0u8; 513]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::CategoryToNeeds, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_emergency_needs_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 512]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::EmergencyNeeds, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_emergency_needs_tag_too_long() {
        let tag = LinkTag(vec![0u8; 513]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::EmergencyNeeds, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_agent_to_offers_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::AgentToOffers, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_agent_to_offers_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::AgentToOffers, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_category_to_offers_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 512]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::CategoryToOffers, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_category_to_offers_tag_too_long() {
        let tag = LinkTag(vec![0u8; 513]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::CategoryToOffers, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_need_to_matches_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::NeedToMatches, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_need_to_matches_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::NeedToMatches, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_offer_to_matches_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::OfferToMatches, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_offer_to_matches_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::OfferToMatches, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_match_to_fulfillment_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::MatchToFulfillment, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_match_to_fulfillment_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::MatchToFulfillment, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_all_needs_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::AllNeeds, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_all_needs_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::AllNeeds, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_all_offers_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::AllOffers, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_all_offers_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::AllOffers, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // =========================================================================
    // STRING LENGTH LIMIT BOUNDARY TESTS
    // =========================================================================

    // -- Need ID --

    #[test]
    fn test_validate_need_id_at_limit() {
        let mut need = valid_need();
        need.id = "x".repeat(64);
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_validate_need_id_too_long() {
        let mut need = valid_need();
        need.id = "x".repeat(257);
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_validate_need_id_whitespace() {
        let mut need = valid_need();
        need.id = "   ".to_string();
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // -- Offer ID --

    #[test]
    fn test_validate_offer_id_at_limit() {
        let mut offer = valid_offer();
        offer.id = "x".repeat(64);
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_validate_offer_id_too_long() {
        let mut offer = valid_offer();
        offer.id = "x".repeat(257);
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_validate_offer_id_whitespace() {
        let mut offer = valid_offer();
        offer.id = "  \t ".to_string();
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // -- Match ID --

    #[test]
    fn test_validate_match_id_at_limit() {
        let mut m = valid_match();
        m.id = "x".repeat(64);
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_validate_match_id_too_long() {
        let mut m = valid_match();
        m.id = "x".repeat(257);
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_validate_match_id_whitespace() {
        let mut m = valid_match();
        m.id = "  \t  ".to_string();
        let result = validate_match(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // -- Need title whitespace --

    #[test]
    fn test_validate_need_title_whitespace() {
        let mut need = valid_need();
        need.title = "  \t ".to_string();
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // -- Offer title whitespace --

    #[test]
    fn test_validate_offer_title_whitespace() {
        let mut offer = valid_offer();
        offer.title = "  \t ".to_string();
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // -- Reciprocity offer item length --

    #[test]
    fn test_validate_need_reciprocity_item_at_limit() {
        let mut need = valid_need();
        need.reciprocity_offers = vec!["x".repeat(256)];
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_validate_need_reciprocity_item_too_long() {
        let mut need = valid_need();
        need.reciprocity_offers = vec!["x".repeat(257)];
        let result = validate_need(need);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // -- Asking for item length --

    #[test]
    fn test_validate_offer_asking_for_item_at_limit() {
        let mut offer = valid_offer();
        offer.asking_for = vec!["x".repeat(256)];
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_validate_offer_asking_for_item_too_long() {
        let mut offer = valid_offer();
        offer.asking_for = vec!["x".repeat(257)];
        let result = validate_offer(offer);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }
}
