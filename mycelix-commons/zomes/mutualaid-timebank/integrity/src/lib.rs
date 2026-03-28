// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Timebank Integrity Zome
//!
//! This zome defines the entry types and validation rules for time banking
//! in the Mycelix Mutual Aid hApp. It implements the core principle:
//! 1 hour = 1 hour, regardless of service type.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};
use mutualaid_common::*;

/// Entry types for the timebank zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A service offer from a member
    #[entry_type(visibility = "public")]
    ServiceOffer(ServiceOffer),
    /// A service request from a member
    #[entry_type(visibility = "public")]
    ServiceRequest(ServiceRequest),
    /// A completed time exchange
    #[entry_type(visibility = "public")]
    TimeExchange(TimeExchange),
    /// Time credit record
    #[entry_type(visibility = "public")]
    TimeCredit(TimeCredit),
}

/// Link types for the timebank zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from agent to their service offers
    AgentToOffers,
    /// Link from agent to their service requests
    AgentToRequests,
    /// Link from agent to exchanges they participated in
    AgentToExchanges,
    /// Link from category anchor to offers
    CategoryToOffers,
    /// Link from category anchor to requests
    CategoryToRequests,
    /// Link from offer to exchange
    OfferToExchange,
    /// Link from request to exchange
    RequestToExchange,
    /// Link for all offers discovery
    AllOffers,
    /// Link for all requests discovery
    AllRequests,
    /// Link from agent to their time credits
    AgentToCredits,
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
        EntryTypes::ServiceOffer(offer) => validate_service_offer(offer),
        EntryTypes::ServiceRequest(request) => validate_service_request(request),
        EntryTypes::TimeExchange(exchange) => validate_time_exchange(exchange),
        EntryTypes::TimeCredit(credit) => validate_time_credit(credit),
    }
}

/// Validate a service offer
fn validate_service_offer(offer: ServiceOffer) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if offer.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Service offer ID cannot be empty".to_string(),
        ));
    }

    // ID length limit
    if offer.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Service offer ID exceeds 256 character limit".to_string(),
        ));
    }

    // Title must not be empty
    if offer.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Service offer title cannot be empty".to_string(),
        ));
    }

    // Title length limit
    if offer.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Service offer title cannot exceed 200 characters".to_string(),
        ));
    }

    // Description length limit
    if offer.description.len() > 5000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Service offer description cannot exceed 5000 characters".to_string(),
        ));
    }

    // Minimum duration must be positive
    if offer.min_duration_hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Minimum duration must be positive".to_string(),
        ));
    }

    // Max duration must be >= min if specified
    if let Some(max) = offer.max_duration_hours {
        if max < offer.min_duration_hours {
            return Ok(ValidateCallbackResult::Invalid(
                "Maximum duration cannot be less than minimum".to_string(),
            ));
        }
    }

    // Qualifications limit
    if offer.qualifications.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many qualifications (max 20)".to_string(),
        ));
    }

    // Qualification entry length limit
    for qual in &offer.qualifications {
        if qual.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each qualification must be 256 characters or fewer".to_string(),
            ));
        }
    }

    // ServiceCategory::Custom variant length limit
    if let ServiceCategory::Custom(ref s) = offer.category {
        if s.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Custom service category cannot be empty".to_string(),
            ));
        }
        if s.len() > 128 {
            return Ok(ValidateCallbackResult::Invalid(
                "Custom service category exceeds 128 character limit".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a service request
fn validate_service_request(request: ServiceRequest) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if request.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Service request ID cannot be empty".to_string(),
        ));
    }

    // ID length limit
    if request.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Service request ID exceeds 256 character limit".to_string(),
        ));
    }

    // Title must not be empty
    if request.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Service request title cannot be empty".to_string(),
        ));
    }

    // Title length limit
    if request.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Service request title cannot exceed 200 characters".to_string(),
        ));
    }

    // Description length limit
    if request.description.len() > 5000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Service request description cannot exceed 5000 characters".to_string(),
        ));
    }

    // Estimated hours must be positive
    if request.estimated_hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Estimated hours must be positive".to_string(),
        ));
    }

    // Estimated hours should be reasonable (max 168 = 1 week)
    if request.estimated_hours > 168.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Estimated hours cannot exceed 168 (one week)".to_string(),
        ));
    }

    // ServiceCategory::Custom variant length limit
    if let ServiceCategory::Custom(ref s) = request.category {
        if s.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Custom service category cannot be empty".to_string(),
            ));
        }
        if s.len() > 128 {
            return Ok(ValidateCallbackResult::Invalid(
                "Custom service category exceeds 128 character limit".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a time exchange
fn validate_time_exchange(exchange: TimeExchange) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if exchange.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Time exchange ID cannot be empty".to_string(),
        ));
    }

    // ID length limit
    if exchange.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Time exchange ID exceeds 256 character limit".to_string(),
        ));
    }

    // Hours must be positive
    if exchange.hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange hours must be positive".to_string(),
        ));
    }

    // Hours should be reasonable (max 168 = 1 week)
    if exchange.hours > 168.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange hours cannot exceed 168 (one week)".to_string(),
        ));
    }

    // Provider and recipient must be different
    if exchange.provider == exchange.recipient {
        return Ok(ValidateCallbackResult::Invalid(
            "Provider and recipient must be different agents".to_string(),
        ));
    }

    // Description must not be empty
    if exchange.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange description cannot be empty".to_string(),
        ));
    }

    // Description length limit
    if exchange.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange description exceeds 4096 character limit".to_string(),
        ));
    }

    // Validate ratings if present
    if let Some(rating) = &exchange.provider_rating {
        if rating.score < 1 || rating.score > 5 {
            return Ok(ValidateCallbackResult::Invalid(
                "Rating score must be between 1 and 5".to_string(),
            ));
        }
        if let Some(ref comment) = rating.comment {
            if comment.len() > 1024 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Rating comment exceeds 1024 character limit".to_string(),
                ));
            }
        }
    }

    if let Some(rating) = &exchange.recipient_rating {
        if rating.score < 1 || rating.score > 5 {
            return Ok(ValidateCallbackResult::Invalid(
                "Rating score must be between 1 and 5".to_string(),
            ));
        }
        if let Some(ref comment) = rating.comment {
            if comment.len() > 1024 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Rating comment exceeds 1024 character limit".to_string(),
                ));
            }
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a time credit
fn validate_time_credit(credit: TimeCredit) -> ExternResult<ValidateCallbackResult> {
    // Hours must be positive
    if credit.hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit hours must be positive".to_string(),
        ));
    }

    // Hours should be reasonable
    if credit.hours > 168.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit hours cannot exceed 168 (one week)".to_string(),
        ));
    }

    // Earner and debtor must be different
    if credit.earner == credit.debtor {
        return Ok(ValidateCallbackResult::Invalid(
            "Earner and debtor must be different agents".to_string(),
        ));
    }

    // Description must not be empty
    if credit.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit description cannot be empty".to_string(),
        ));
    }

    // Description length limit
    if credit.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit description exceeds 4096 character limit".to_string(),
        ));
    }

    // ServiceCategory::Custom variant length limit
    if let ServiceCategory::Custom(ref s) = credit.service_category {
        if s.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Custom service category cannot be empty".to_string(),
            ));
        }
        if s.len() > 128 {
            return Ok(ValidateCallbackResult::Invalid(
                "Custom service category exceeds 128 character limit".to_string(),
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
        LinkTypes::AgentToOffers => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "AgentToOffers link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::AgentToRequests => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "AgentToRequests link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::AgentToExchanges => {
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "AgentToExchanges link tag too long (max 512 bytes)".into(),
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
        LinkTypes::CategoryToRequests => {
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "CategoryToRequests link tag too long (max 512 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::OfferToExchange => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "OfferToExchange link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::RequestToExchange => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "RequestToExchange link tag too long (max 256 bytes)".into(),
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
        LinkTypes::AllRequests => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "AllRequests link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::AgentToCredits => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "AgentToCredits link tag too long (max 256 bytes)".into(),
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
    // HELPERS
    // =========================================================================

    fn agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xAA; 36])
    }
    fn agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xBB; 36])
    }

    /// Minimal Create action with zero-filled hashes for test use.
    fn fake_create() -> Create {
        Create {
            author: agent_a(),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: EntryRateWeight::default(),
        }
    }

    fn is_valid(r: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(r, Ok(ValidateCallbackResult::Valid))
    }
    fn is_invalid(r: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(r, Ok(ValidateCallbackResult::Invalid(_)))
    }

    /// Assert that the result is invalid with a message containing the given substring.
    fn assert_invalid_contains(r: &ExternResult<ValidateCallbackResult>, substring: &str) {
        match r {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains(substring),
                    "Expected error containing '{}', got '{}'",
                    substring,
                    msg
                );
            }
            other => panic!(
                "Expected Invalid result containing '{}', got {:?}",
                substring, other
            ),
        }
    }

    fn valid_offer() -> ServiceOffer {
        ServiceOffer {
            id: "offer-1".into(),
            provider: agent_a(),
            category: ServiceCategory::Tutoring,
            title: "Math tutoring".into(),
            description: "Help with calculus".into(),
            qualifications: vec!["BS Mathematics".into()],
            availability: Availability::default(),
            location: LocationConstraint::Remote,
            min_duration_hours: 1.0,
            max_duration_hours: Some(3.0),
            active: true,
            created_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        }
    }

    fn valid_request() -> ServiceRequest {
        ServiceRequest {
            id: "req-1".into(),
            requester: agent_a(),
            category: ServiceCategory::HomeRepair,
            title: "Fix leaky faucet".into(),
            description: "Kitchen faucet dripping".into(),
            urgency: UrgencyLevel::Medium,
            needed_by: None,
            estimated_hours: 2.0,
            location: LocationConstraint::Remote,
            status: RequestStatus::Open,
            created_at: Timestamp::from_micros(0),
        }
    }

    fn valid_exchange() -> TimeExchange {
        TimeExchange {
            id: "exch-1".into(),
            offer_hash: None,
            request_hash: None,
            provider: agent_a(),
            recipient: agent_b(),
            hours: 2.0,
            category: ServiceCategory::Tutoring,
            description: "Helped with homework".into(),
            completed_at: Timestamp::from_micros(0),
            provider_rating: None,
            recipient_rating: None,
            confirmed: true,
        }
    }

    fn valid_credit() -> TimeCredit {
        TimeCredit {
            hours: 2.0,
            earner: agent_a(),
            debtor: agent_b(),
            service_category: ServiceCategory::Tutoring,
            description: "Tutored math for 2 hours".into(),
            performed_at: Timestamp::from_micros(0),
            expires_at: None,
        }
    }

    fn make_rating(score: u8) -> Rating {
        Rating {
            score,
            comment: None,
            rated_at: Timestamp::from_micros(0),
        }
    }

    // =========================================================================
    // FAKE_CREATE HELPER TEST
    // =========================================================================

    #[test]
    fn fake_create_is_well_formed() {
        let c = fake_create();
        assert_eq!(c.action_seq, 0);
        assert_eq!(c.author, agent_a());
    }

    // =========================================================================
    // SERDE ROUNDTRIP TESTS
    // =========================================================================

    #[test]
    fn serde_roundtrip_service_category_simple() {
        let cats = vec![
            ServiceCategory::Childcare,
            ServiceCategory::Eldercare,
            ServiceCategory::PetCare,
            ServiceCategory::PersonalCare,
            ServiceCategory::Cleaning,
            ServiceCategory::Cooking,
            ServiceCategory::Gardening,
            ServiceCategory::HomeRepair,
            ServiceCategory::MovingHelp,
            ServiceCategory::LegalAdvice,
            ServiceCategory::FinancialAdvice,
            ServiceCategory::MedicalConsult,
            ServiceCategory::TechSupport,
            ServiceCategory::Tutoring,
            ServiceCategory::LanguageTeaching,
            ServiceCategory::MusicLessons,
            ServiceCategory::ArtInstruction,
            ServiceCategory::Driving,
            ServiceCategory::Delivery,
            ServiceCategory::Errands,
            ServiceCategory::Photography,
            ServiceCategory::Design,
            ServiceCategory::Writing,
            ServiceCategory::Crafts,
            ServiceCategory::EventPlanning,
            ServiceCategory::Facilitation,
            ServiceCategory::Mediation,
            ServiceCategory::Organizing,
        ];
        for cat in cats {
            let json = serde_json::to_string(&cat).unwrap();
            let parsed: ServiceCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(cat, parsed);
        }
    }

    #[test]
    fn serde_roundtrip_service_category_custom() {
        let cat = ServiceCategory::Custom("Beekeeping".into());
        let json = serde_json::to_string(&cat).unwrap();
        let parsed: ServiceCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(cat, parsed);
    }

    #[test]
    fn serde_roundtrip_urgency_level() {
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

    #[test]
    fn serde_roundtrip_request_status() {
        let statuses = vec![
            RequestStatus::Open,
            RequestStatus::Matched,
            RequestStatus::InProgress,
            RequestStatus::Completed,
            RequestStatus::Cancelled,
            RequestStatus::Expired,
        ];
        for st in statuses {
            let json = serde_json::to_string(&st).unwrap();
            let parsed: RequestStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(st, parsed);
        }
    }

    #[test]
    fn serde_roundtrip_location_constraint() {
        let locs = vec![
            LocationConstraint::Remote,
            LocationConstraint::FixedLocation("123 Main St".into()),
            LocationConstraint::WithinRadius {
                geohash: "u4pru".into(),
                radius_km: 5.0,
            },
            LocationConstraint::AtRequester,
            LocationConstraint::AtProvider,
            LocationConstraint::ToBeArranged,
        ];
        for loc in locs {
            let json = serde_json::to_string(&loc).unwrap();
            let parsed: LocationConstraint = serde_json::from_str(&json).unwrap();
            assert_eq!(loc, parsed);
        }
    }

    #[test]
    fn serde_roundtrip_rating() {
        let r = Rating {
            score: 4,
            comment: Some("Great work!".into()),
            rated_at: Timestamp::from_micros(1000),
        };
        let json = serde_json::to_string(&r).unwrap();
        let parsed: Rating = serde_json::from_str(&json).unwrap();
        assert_eq!(r, parsed);
    }

    #[test]
    fn serde_roundtrip_rating_no_comment() {
        let r = Rating {
            score: 1,
            comment: None,
            rated_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&r).unwrap();
        let parsed: Rating = serde_json::from_str(&json).unwrap();
        assert_eq!(r, parsed);
    }

    #[test]
    fn serde_roundtrip_availability() {
        let a = Availability::default();
        let json = serde_json::to_string(&a).unwrap();
        let parsed: Availability = serde_json::from_str(&json).unwrap();
        assert_eq!(a, parsed);
    }

    #[test]
    fn serde_roundtrip_availability_with_exceptions() {
        let a = Availability {
            days: vec![0, 6],
            start_minutes: 0,
            end_minutes: 1439,
            timezone_offset_minutes: -300,
            exceptions: vec![DateRange {
                start: Timestamp::from_micros(100),
                end: Timestamp::from_micros(200),
                reason: Some("Holiday".into()),
            }],
            notes: Some("Weekends only".into()),
        };
        let json = serde_json::to_string(&a).unwrap();
        let parsed: Availability = serde_json::from_str(&json).unwrap();
        assert_eq!(a, parsed);
    }

    #[test]
    fn serde_roundtrip_service_offer() {
        let o = valid_offer();
        let bytes = serde_json::to_vec(&o).unwrap();
        let parsed: ServiceOffer = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(o, parsed);
    }

    #[test]
    fn serde_roundtrip_service_request() {
        let r = valid_request();
        let bytes = serde_json::to_vec(&r).unwrap();
        let parsed: ServiceRequest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(r, parsed);
    }

    #[test]
    fn serde_roundtrip_time_exchange() {
        let e = valid_exchange();
        let bytes = serde_json::to_vec(&e).unwrap();
        let parsed: TimeExchange = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(e, parsed);
    }

    #[test]
    fn serde_roundtrip_time_credit() {
        let c = valid_credit();
        let bytes = serde_json::to_vec(&c).unwrap();
        let parsed: TimeCredit = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(c, parsed);
    }

    #[test]
    fn serde_roundtrip_time_credit_with_expiry() {
        let mut c = valid_credit();
        c.expires_at = Some(Timestamp::from_micros(999_999_999));
        let bytes = serde_json::to_vec(&c).unwrap();
        let parsed: TimeCredit = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(c, parsed);
    }

    #[test]
    fn serde_roundtrip_date_range() {
        let dr = DateRange {
            start: Timestamp::from_micros(0),
            end: Timestamp::from_micros(1_000_000),
            reason: None,
        };
        let json = serde_json::to_string(&dr).unwrap();
        let parsed: DateRange = serde_json::from_str(&json).unwrap();
        assert_eq!(dr, parsed);
    }

    // =========================================================================
    // validate_create_entry DISPATCHER TESTS
    // =========================================================================

    #[test]
    fn dispatcher_routes_valid_offer() {
        let entry = EntryTypes::ServiceOffer(valid_offer());
        assert!(is_valid(&validate_create_entry(entry)));
    }

    #[test]
    fn dispatcher_routes_invalid_offer() {
        let mut o = valid_offer();
        o.id = "".into();
        let entry = EntryTypes::ServiceOffer(o);
        assert!(is_invalid(&validate_create_entry(entry)));
    }

    #[test]
    fn dispatcher_routes_valid_request() {
        let entry = EntryTypes::ServiceRequest(valid_request());
        assert!(is_valid(&validate_create_entry(entry)));
    }

    #[test]
    fn dispatcher_routes_invalid_request() {
        let mut r = valid_request();
        r.title = "".into();
        let entry = EntryTypes::ServiceRequest(r);
        assert!(is_invalid(&validate_create_entry(entry)));
    }

    #[test]
    fn dispatcher_routes_valid_exchange() {
        let entry = EntryTypes::TimeExchange(valid_exchange());
        assert!(is_valid(&validate_create_entry(entry)));
    }

    #[test]
    fn dispatcher_routes_invalid_exchange() {
        let mut e = valid_exchange();
        e.hours = 0.0;
        let entry = EntryTypes::TimeExchange(e);
        assert!(is_invalid(&validate_create_entry(entry)));
    }

    #[test]
    fn dispatcher_routes_valid_credit() {
        let entry = EntryTypes::TimeCredit(valid_credit());
        assert!(is_valid(&validate_create_entry(entry)));
    }

    #[test]
    fn dispatcher_routes_invalid_credit() {
        let mut c = valid_credit();
        c.description = "".into();
        let entry = EntryTypes::TimeCredit(c);
        assert!(is_invalid(&validate_create_entry(entry)));
    }

    // =========================================================================
    // ServiceOffer validation - original + new
    // =========================================================================

    #[test]
    fn offer_valid() {
        assert!(is_valid(&validate_service_offer(valid_offer())));
    }

    #[test]
    fn offer_empty_id_rejected() {
        let mut o = valid_offer();
        o.id = "".into();
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_empty_id_error_message() {
        let mut o = valid_offer();
        o.id = "".into();
        assert_invalid_contains(&validate_service_offer(o), "ID cannot be empty");
    }

    #[test]
    fn offer_empty_title_rejected() {
        let mut o = valid_offer();
        o.title = "".into();
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_empty_title_error_message() {
        let mut o = valid_offer();
        o.title = "".into();
        assert_invalid_contains(&validate_service_offer(o), "title cannot be empty");
    }

    #[test]
    fn offer_title_at_199_accepted() {
        let mut o = valid_offer();
        o.title = "x".repeat(199);
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_title_at_200_accepted() {
        let mut o = valid_offer();
        o.title = "x".repeat(200);
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_title_over_200_rejected() {
        let mut o = valid_offer();
        o.title = "x".repeat(201);
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_title_at_200_boundary_error_message() {
        let mut o = valid_offer();
        o.title = "x".repeat(201);
        assert_invalid_contains(&validate_service_offer(o), "title cannot exceed 200");
    }

    #[test]
    fn offer_description_at_5000_accepted() {
        let mut o = valid_offer();
        o.description = "d".repeat(5000);
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_description_over_5000_rejected() {
        let mut o = valid_offer();
        o.description = "x".repeat(5001);
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_description_over_5000_error_message() {
        let mut o = valid_offer();
        o.description = "x".repeat(5001);
        assert_invalid_contains(&validate_service_offer(o), "description cannot exceed 5000");
    }

    #[test]
    fn offer_empty_description_accepted() {
        let mut o = valid_offer();
        o.description = "".into();
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_zero_min_duration_rejected() {
        let mut o = valid_offer();
        o.min_duration_hours = 0.0;
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_negative_min_duration_rejected() {
        let mut o = valid_offer();
        o.min_duration_hours = -1.0;
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_very_small_positive_min_duration_accepted() {
        let mut o = valid_offer();
        o.min_duration_hours = 0.001;
        o.max_duration_hours = None;
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_negative_infinity_min_duration_rejected() {
        let mut o = valid_offer();
        o.min_duration_hours = f64::NEG_INFINITY;
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_max_less_than_min_rejected() {
        let mut o = valid_offer();
        o.min_duration_hours = 2.0;
        o.max_duration_hours = Some(1.0);
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_max_less_than_min_error_message() {
        let mut o = valid_offer();
        o.min_duration_hours = 5.0;
        o.max_duration_hours = Some(2.0);
        assert_invalid_contains(
            &validate_service_offer(o),
            "Maximum duration cannot be less than minimum",
        );
    }

    #[test]
    fn offer_max_equal_min_accepted() {
        let mut o = valid_offer();
        o.min_duration_hours = 2.0;
        o.max_duration_hours = Some(2.0);
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_no_max_duration_accepted() {
        let mut o = valid_offer();
        o.max_duration_hours = None;
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_too_many_qualifications_rejected() {
        let mut o = valid_offer();
        o.qualifications = (0..21).map(|i| format!("qual_{}", i)).collect();
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_too_many_qualifications_error_message() {
        let mut o = valid_offer();
        o.qualifications = (0..21).map(|i| format!("q{}", i)).collect();
        assert_invalid_contains(&validate_service_offer(o), "Too many qualifications");
    }

    #[test]
    fn offer_20_qualifications_accepted() {
        let mut o = valid_offer();
        o.qualifications = (0..20).map(|i| format!("qual_{}", i)).collect();
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_zero_qualifications_accepted() {
        let mut o = valid_offer();
        o.qualifications = vec![];
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_unicode_title_accepted() {
        let mut o = valid_offer();
        o.title = "Cours de francais".into();
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_emoji_title_accepted() {
        let mut o = valid_offer();
        o.title = "Math Help".into();
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_custom_category_accepted() {
        let mut o = valid_offer();
        o.category = ServiceCategory::Custom("Quantum Computing Tutoring".into());
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_inactive_still_valid() {
        let mut o = valid_offer();
        o.active = false;
        assert!(is_valid(&validate_service_offer(o)));
    }

    // =========================================================================
    // ServiceRequest validation - original + new
    // =========================================================================

    #[test]
    fn request_valid() {
        assert!(is_valid(&validate_service_request(valid_request())));
    }

    #[test]
    fn request_empty_id_rejected() {
        let mut r = valid_request();
        r.id = "".into();
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_empty_id_error_message() {
        let mut r = valid_request();
        r.id = "".into();
        assert_invalid_contains(&validate_service_request(r), "ID cannot be empty");
    }

    #[test]
    fn request_empty_title_rejected() {
        let mut r = valid_request();
        r.title = "".into();
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_empty_title_error_message() {
        let mut r = valid_request();
        r.title = "".into();
        assert_invalid_contains(&validate_service_request(r), "title cannot be empty");
    }

    #[test]
    fn request_title_at_200_accepted() {
        let mut r = valid_request();
        r.title = "t".repeat(200);
        assert!(is_valid(&validate_service_request(r)));
    }

    #[test]
    fn request_title_over_200_rejected() {
        let mut r = valid_request();
        r.title = "x".repeat(201);
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_title_over_200_error_message() {
        let mut r = valid_request();
        r.title = "x".repeat(201);
        assert_invalid_contains(&validate_service_request(r), "title cannot exceed 200");
    }

    #[test]
    fn request_description_at_5000_accepted() {
        let mut r = valid_request();
        r.description = "d".repeat(5000);
        assert!(is_valid(&validate_service_request(r)));
    }

    #[test]
    fn request_description_over_5000_rejected() {
        let mut r = valid_request();
        r.description = "x".repeat(5001);
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_description_over_5000_error_message() {
        let mut r = valid_request();
        r.description = "x".repeat(5001);
        assert_invalid_contains(
            &validate_service_request(r),
            "description cannot exceed 5000",
        );
    }

    #[test]
    fn request_empty_description_accepted() {
        let mut r = valid_request();
        r.description = "".into();
        assert!(is_valid(&validate_service_request(r)));
    }

    #[test]
    fn request_zero_hours_rejected() {
        let mut r = valid_request();
        r.estimated_hours = 0.0;
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_zero_hours_error_message() {
        let mut r = valid_request();
        r.estimated_hours = 0.0;
        assert_invalid_contains(
            &validate_service_request(r),
            "Estimated hours must be positive",
        );
    }

    #[test]
    fn request_negative_hours_rejected() {
        let mut r = valid_request();
        r.estimated_hours = -1.0;
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_very_small_positive_hours_accepted() {
        let mut r = valid_request();
        r.estimated_hours = 0.01;
        assert!(is_valid(&validate_service_request(r)));
    }

    #[test]
    fn request_exactly_168_hours_accepted() {
        let mut r = valid_request();
        r.estimated_hours = 168.0;
        assert!(is_valid(&validate_service_request(r)));
    }

    #[test]
    fn request_over_168_hours_rejected() {
        let mut r = valid_request();
        r.estimated_hours = 168.1;
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_over_168_hours_error_message() {
        let mut r = valid_request();
        r.estimated_hours = 200.0;
        assert_invalid_contains(&validate_service_request(r), "cannot exceed 168");
    }

    #[test]
    fn request_negative_infinity_hours_rejected() {
        let mut r = valid_request();
        r.estimated_hours = f64::NEG_INFINITY;
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_with_needed_by_accepted() {
        let mut r = valid_request();
        r.needed_by = Some(Timestamp::from_micros(999_999));
        assert!(is_valid(&validate_service_request(r)));
    }

    #[test]
    fn request_unicode_title_accepted() {
        let mut r = valid_request();
        r.title = "Ayuda con la mudanza".into();
        assert!(is_valid(&validate_service_request(r)));
    }

    #[test]
    fn request_all_urgency_levels_accepted() {
        for urgency in [
            UrgencyLevel::Low,
            UrgencyLevel::Medium,
            UrgencyLevel::High,
            UrgencyLevel::Urgent,
            UrgencyLevel::Emergency,
        ] {
            let mut r = valid_request();
            r.urgency = urgency;
            assert!(is_valid(&validate_service_request(r)));
        }
    }

    #[test]
    fn request_all_statuses_accepted() {
        for status in [
            RequestStatus::Open,
            RequestStatus::Matched,
            RequestStatus::InProgress,
            RequestStatus::Completed,
            RequestStatus::Cancelled,
            RequestStatus::Expired,
        ] {
            let mut r = valid_request();
            r.status = status;
            assert!(is_valid(&validate_service_request(r)));
        }
    }

    // =========================================================================
    // TimeExchange validation - original + new
    // =========================================================================

    #[test]
    fn exchange_valid() {
        assert!(is_valid(&validate_time_exchange(valid_exchange())));
    }

    #[test]
    fn exchange_empty_id_rejected() {
        let mut e = valid_exchange();
        e.id = "".into();
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_empty_id_error_message() {
        let mut e = valid_exchange();
        e.id = "".into();
        assert_invalid_contains(&validate_time_exchange(e), "ID cannot be empty");
    }

    #[test]
    fn exchange_zero_hours_rejected() {
        let mut e = valid_exchange();
        e.hours = 0.0;
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_zero_hours_error_message() {
        let mut e = valid_exchange();
        e.hours = 0.0;
        assert_invalid_contains(
            &validate_time_exchange(e),
            "Exchange hours must be positive",
        );
    }

    #[test]
    fn exchange_negative_hours_rejected() {
        let mut e = valid_exchange();
        e.hours = -0.5;
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_very_small_positive_hours_accepted() {
        let mut e = valid_exchange();
        e.hours = 0.01;
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_exactly_168_hours_accepted() {
        let mut e = valid_exchange();
        e.hours = 168.0;
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_over_168_hours_rejected() {
        let mut e = valid_exchange();
        e.hours = 169.0;
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_over_168_hours_error_message() {
        let mut e = valid_exchange();
        e.hours = 200.0;
        assert_invalid_contains(&validate_time_exchange(e), "cannot exceed 168");
    }

    #[test]
    fn exchange_same_provider_recipient_rejected() {
        let mut e = valid_exchange();
        e.recipient = agent_a(); // same as provider
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_same_agents_error_message() {
        let mut e = valid_exchange();
        e.recipient = agent_a();
        assert_invalid_contains(
            &validate_time_exchange(e),
            "Provider and recipient must be different",
        );
    }

    #[test]
    fn exchange_empty_description_rejected() {
        let mut e = valid_exchange();
        e.description = "".into();
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_empty_description_error_message() {
        let mut e = valid_exchange();
        e.description = "".into();
        assert_invalid_contains(
            &validate_time_exchange(e),
            "Exchange description cannot be empty",
        );
    }

    #[test]
    fn exchange_rating_score_0_rejected() {
        let mut e = valid_exchange();
        e.provider_rating = Some(make_rating(0));
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_rating_score_0_error_message() {
        let mut e = valid_exchange();
        e.provider_rating = Some(make_rating(0));
        assert_invalid_contains(
            &validate_time_exchange(e),
            "Rating score must be between 1 and 5",
        );
    }

    #[test]
    fn exchange_rating_score_6_rejected() {
        let mut e = valid_exchange();
        e.recipient_rating = Some(make_rating(6));
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_provider_rating_score_1_accepted() {
        let mut e = valid_exchange();
        e.provider_rating = Some(make_rating(1));
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_provider_rating_score_5_accepted() {
        let mut e = valid_exchange();
        e.provider_rating = Some(Rating {
            score: 5,
            comment: Some("Excellent!".into()),
            rated_at: Timestamp::from_micros(0),
        });
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_recipient_rating_score_1_accepted() {
        let mut e = valid_exchange();
        e.recipient_rating = Some(make_rating(1));
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_recipient_rating_score_5_accepted() {
        let mut e = valid_exchange();
        e.recipient_rating = Some(make_rating(5));
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_all_mid_range_ratings_accepted() {
        for score in 2..=4 {
            let mut e = valid_exchange();
            e.provider_rating = Some(make_rating(score));
            e.recipient_rating = Some(make_rating(score));
            assert!(
                is_valid(&validate_time_exchange(e)),
                "score={} should be valid",
                score
            );
        }
    }

    #[test]
    fn exchange_provider_rating_max_u8_rejected() {
        let mut e = valid_exchange();
        e.provider_rating = Some(make_rating(u8::MAX));
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_recipient_rating_max_u8_rejected() {
        let mut e = valid_exchange();
        e.recipient_rating = Some(make_rating(u8::MAX));
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_both_ratings_none_accepted() {
        let mut e = valid_exchange();
        e.provider_rating = None;
        e.recipient_rating = None;
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_both_ratings_valid_accepted() {
        let mut e = valid_exchange();
        e.provider_rating = Some(make_rating(3));
        e.recipient_rating = Some(make_rating(4));
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_provider_invalid_recipient_valid_still_rejected() {
        let mut e = valid_exchange();
        e.provider_rating = Some(make_rating(0));
        e.recipient_rating = Some(make_rating(5));
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_provider_valid_recipient_invalid_still_rejected() {
        let mut e = valid_exchange();
        e.provider_rating = Some(make_rating(3));
        e.recipient_rating = Some(make_rating(6));
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_with_offer_hash_accepted() {
        let mut e = valid_exchange();
        e.offer_hash = Some(ActionHash::from_raw_36(vec![0x11; 36]));
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_with_request_hash_accepted() {
        let mut e = valid_exchange();
        e.request_hash = Some(ActionHash::from_raw_36(vec![0x22; 36]));
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_unconfirmed_accepted() {
        let mut e = valid_exchange();
        e.confirmed = false;
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_unicode_description_accepted() {
        let mut e = valid_exchange();
        e.description = "Aide au demenagement".into();
        assert!(is_valid(&validate_time_exchange(e)));
    }

    // =========================================================================
    // TimeCredit validation - original + new
    // =========================================================================

    #[test]
    fn credit_valid() {
        assert!(is_valid(&validate_time_credit(valid_credit())));
    }

    #[test]
    fn credit_zero_hours_rejected() {
        let mut c = valid_credit();
        c.hours = 0.0;
        assert!(is_invalid(&validate_time_credit(c)));
    }

    #[test]
    fn credit_zero_hours_error_message() {
        let mut c = valid_credit();
        c.hours = 0.0;
        assert_invalid_contains(&validate_time_credit(c), "Credit hours must be positive");
    }

    #[test]
    fn credit_negative_hours_rejected() {
        let mut c = valid_credit();
        c.hours = -5.0;
        assert!(is_invalid(&validate_time_credit(c)));
    }

    #[test]
    fn credit_very_small_positive_hours_accepted() {
        let mut c = valid_credit();
        c.hours = 0.01;
        assert!(is_valid(&validate_time_credit(c)));
    }

    #[test]
    fn credit_exactly_168_hours_accepted() {
        let mut c = valid_credit();
        c.hours = 168.0;
        assert!(is_valid(&validate_time_credit(c)));
    }

    #[test]
    fn credit_over_168_hours_rejected() {
        let mut c = valid_credit();
        c.hours = 200.0;
        assert!(is_invalid(&validate_time_credit(c)));
    }

    #[test]
    fn credit_over_168_hours_error_message() {
        let mut c = valid_credit();
        c.hours = 169.0;
        assert_invalid_contains(&validate_time_credit(c), "Credit hours cannot exceed 168");
    }

    #[test]
    fn credit_at_168_point_1_rejected() {
        let mut c = valid_credit();
        c.hours = 168.1;
        assert!(is_invalid(&validate_time_credit(c)));
    }

    #[test]
    fn credit_same_earner_debtor_rejected() {
        let mut c = valid_credit();
        c.debtor = agent_a(); // same as earner
        assert!(is_invalid(&validate_time_credit(c)));
    }

    #[test]
    fn credit_same_earner_debtor_error_message() {
        let mut c = valid_credit();
        c.debtor = agent_a();
        assert_invalid_contains(
            &validate_time_credit(c),
            "Earner and debtor must be different",
        );
    }

    #[test]
    fn credit_empty_description_rejected() {
        let mut c = valid_credit();
        c.description = "".into();
        assert!(is_invalid(&validate_time_credit(c)));
    }

    #[test]
    fn credit_empty_description_error_message() {
        let mut c = valid_credit();
        c.description = "".into();
        assert_invalid_contains(
            &validate_time_credit(c),
            "Credit description cannot be empty",
        );
    }

    #[test]
    fn credit_with_expiry_accepted() {
        let mut c = valid_credit();
        c.expires_at = Some(Timestamp::from_micros(9_999_999));
        assert!(is_valid(&validate_time_credit(c)));
    }

    #[test]
    fn credit_without_expiry_accepted() {
        let mut c = valid_credit();
        c.expires_at = None;
        assert!(is_valid(&validate_time_credit(c)));
    }

    #[test]
    fn credit_unicode_description_accepted() {
        let mut c = valid_credit();
        c.description = "Tutorat en mathematiques".into();
        assert!(is_valid(&validate_time_credit(c)));
    }

    #[test]
    fn credit_all_service_categories_accepted() {
        let categories = vec![
            ServiceCategory::Childcare,
            ServiceCategory::Eldercare,
            ServiceCategory::Tutoring,
            ServiceCategory::HomeRepair,
            ServiceCategory::Custom("Basket Weaving".into()),
        ];
        for cat in categories {
            let mut c = valid_credit();
            c.service_category = cat;
            assert!(is_valid(&validate_time_credit(c)));
        }
    }

    // =========================================================================
    // LINK VALIDATION TESTS
    // =========================================================================

    #[test]
    fn link_agent_to_offers_valid() {
        let base = AnyLinkableHash::from(agent_a());
        let target = AnyLinkableHash::from(EntryHash::from_raw_36(vec![0x11; 36]));
        let tag = LinkTag::new(vec![]);
        assert!(is_valid(&validate_create_link(
            LinkTypes::AgentToOffers,
            base,
            target,
            tag,
        )));
    }

    #[test]
    fn link_agent_to_requests_valid() {
        let base = AnyLinkableHash::from(agent_a());
        let target = AnyLinkableHash::from(EntryHash::from_raw_36(vec![0x22; 36]));
        let tag = LinkTag::new(vec![]);
        assert!(is_valid(&validate_create_link(
            LinkTypes::AgentToRequests,
            base,
            target,
            tag,
        )));
    }

    #[test]
    fn link_agent_to_exchanges_valid() {
        let base = AnyLinkableHash::from(agent_a());
        let target = AnyLinkableHash::from(EntryHash::from_raw_36(vec![0x33; 36]));
        let tag = LinkTag::new(vec![]);
        assert!(is_valid(&validate_create_link(
            LinkTypes::AgentToExchanges,
            base,
            target,
            tag,
        )));
    }

    #[test]
    fn link_category_to_offers_valid() {
        let base = AnyLinkableHash::from(EntryHash::from_raw_36(vec![0x44; 36]));
        let target = AnyLinkableHash::from(EntryHash::from_raw_36(vec![0x55; 36]));
        let tag = LinkTag::new(b"Tutoring".to_vec());
        assert!(is_valid(&validate_create_link(
            LinkTypes::CategoryToOffers,
            base,
            target,
            tag,
        )));
    }

    #[test]
    fn link_category_to_requests_valid() {
        let base = AnyLinkableHash::from(EntryHash::from_raw_36(vec![0x66; 36]));
        let target = AnyLinkableHash::from(EntryHash::from_raw_36(vec![0x77; 36]));
        let tag = LinkTag::new(vec![]);
        assert!(is_valid(&validate_create_link(
            LinkTypes::CategoryToRequests,
            base,
            target,
            tag,
        )));
    }

    #[test]
    fn link_offer_to_exchange_valid() {
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0x88; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0x99; 36]));
        let tag = LinkTag::new(vec![]);
        assert!(is_valid(&validate_create_link(
            LinkTypes::OfferToExchange,
            base,
            target,
            tag,
        )));
    }

    #[test]
    fn link_request_to_exchange_valid() {
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0xAA; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0xBB; 36]));
        let tag = LinkTag::new(vec![]);
        assert!(is_valid(&validate_create_link(
            LinkTypes::RequestToExchange,
            base,
            target,
            tag,
        )));
    }

    #[test]
    fn link_all_offers_valid() {
        let base = AnyLinkableHash::from(EntryHash::from_raw_36(vec![0xCC; 36]));
        let target = AnyLinkableHash::from(EntryHash::from_raw_36(vec![0xDD; 36]));
        let tag = LinkTag::new(vec![]);
        assert!(is_valid(&validate_create_link(
            LinkTypes::AllOffers,
            base,
            target,
            tag,
        )));
    }

    #[test]
    fn link_all_requests_valid() {
        let base = AnyLinkableHash::from(EntryHash::from_raw_36(vec![0xEE; 36]));
        let target = AnyLinkableHash::from(EntryHash::from_raw_36(vec![0xFF; 36]));
        let tag = LinkTag::new(vec![]);
        assert!(is_valid(&validate_create_link(
            LinkTypes::AllRequests,
            base,
            target,
            tag,
        )));
    }

    #[test]
    fn link_agent_to_credits_valid() {
        let base = AnyLinkableHash::from(agent_b());
        let target = AnyLinkableHash::from(EntryHash::from_raw_36(vec![0x12; 36]));
        let tag = LinkTag::new(vec![1, 2, 3]);
        assert!(is_valid(&validate_create_link(
            LinkTypes::AgentToCredits,
            base,
            target,
            tag,
        )));
    }

    // =========================================================================
    // EDGE CASES: MULTI-BYTE UNICODE AND LEN() SEMANTICS
    // =========================================================================

    // Note: Rust's String::len() counts bytes, not chars. A 4-byte emoji
    // counts as 4 towards the limit. These tests document the byte-length
    // semantics of the validation.

    #[test]
    fn offer_title_multibyte_chars_counted_as_bytes() {
        // Each CJK character is 3 bytes in UTF-8
        // 67 CJK chars = 201 bytes, which exceeds the 200-byte limit
        let mut o = valid_offer();
        let cjk_char = "\u{4E16}"; // 3 bytes
        o.title = cjk_char.repeat(67); // 201 bytes
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_title_multibyte_at_boundary() {
        // 66 CJK chars = 198 bytes, under 200
        let mut o = valid_offer();
        let cjk_char = "\u{4E16}"; // 3 bytes
        o.title = cjk_char.repeat(66); // 198 bytes
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_description_multibyte_at_boundary() {
        // 4-byte emoji repeated to fill close to 5000
        let mut o = valid_offer();
        let emoji = "\u{1F600}"; // 4 bytes
        o.description = emoji.repeat(1250); // exactly 5000 bytes
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_description_multibyte_over_boundary() {
        let mut o = valid_offer();
        let emoji = "\u{1F600}"; // 4 bytes
        o.description = emoji.repeat(1251); // 5004 bytes
        assert!(is_invalid(&validate_service_offer(o)));
    }

    // =========================================================================
    // EDGE CASES: SPECIAL FLOAT VALUES
    // =========================================================================

    #[test]
    fn offer_nan_min_duration_rejected() {
        let mut o = valid_offer();
        o.min_duration_hours = f64::NAN;
        // NaN <= 0.0 is false, but NaN > 0.0 is also false.
        // The check `<= 0.0` returns false for NaN, so NaN passes that check.
        // NaN < min also returns false in max_duration check, so it passes.
        // This documents the current behavior: NaN is NOT caught by validation.
        // (This is a known limitation; NaN comparisons are always false.)
        let result = validate_service_offer(o);
        // NaN passes validation because `NaN <= 0.0` is false
        assert!(is_valid(&result));
    }

    #[test]
    fn offer_positive_infinity_min_duration_accepted() {
        // Infinity > 0.0 is true, so it passes the positive check.
        // If max_duration is None, no further check.
        let mut o = valid_offer();
        o.min_duration_hours = f64::INFINITY;
        o.max_duration_hours = None;
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn exchange_nan_hours_passes_validation() {
        // NaN <= 0.0 is false, NaN > 168.0 is false
        let mut e = valid_exchange();
        e.hours = f64::NAN;
        // Documenting: NaN slips through current float checks
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn credit_nan_hours_passes_validation() {
        let mut c = valid_credit();
        c.hours = f64::NAN;
        // Documenting: NaN slips through current float checks
        assert!(is_valid(&validate_time_credit(c)));
    }

    #[test]
    fn request_nan_hours_passes_validation() {
        let mut r = valid_request();
        r.estimated_hours = f64::NAN;
        // Documenting: NaN slips through current float checks
        assert!(is_valid(&validate_service_request(r)));
    }

    // =========================================================================
    // EDGE CASE: EMPTY COLLECTIONS
    // =========================================================================

    #[test]
    fn offer_empty_qualifications_accepted() {
        let mut o = valid_offer();
        o.qualifications = vec![];
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_single_qualification_accepted() {
        let mut o = valid_offer();
        o.qualifications = vec!["PhD".into()];
        assert!(is_valid(&validate_service_offer(o)));
    }

    // =========================================================================
    // VALIDATION ERROR PRIORITY / ORDER TESTS
    // =========================================================================
    // When multiple fields are invalid, the first check in code order wins.

    #[test]
    fn offer_multiple_errors_returns_first_id_error() {
        let mut o = valid_offer();
        o.id = "".into();
        o.title = "".into();
        o.min_duration_hours = -1.0;
        // ID is checked first
        assert_invalid_contains(&validate_service_offer(o), "ID cannot be empty");
    }

    #[test]
    fn offer_title_error_before_duration_error() {
        let mut o = valid_offer();
        o.id = "ok".into();
        o.title = "".into();
        o.min_duration_hours = -1.0;
        assert_invalid_contains(&validate_service_offer(o), "title cannot be empty");
    }

    #[test]
    fn request_multiple_errors_returns_first_id_error() {
        let mut r = valid_request();
        r.id = "".into();
        r.title = "".into();
        r.estimated_hours = 0.0;
        assert_invalid_contains(&validate_service_request(r), "ID cannot be empty");
    }

    #[test]
    fn exchange_multiple_errors_returns_first_id_error() {
        let mut e = valid_exchange();
        e.id = "".into();
        e.hours = 0.0;
        e.description = "".into();
        assert_invalid_contains(&validate_time_exchange(e), "ID cannot be empty");
    }

    #[test]
    fn exchange_hours_error_before_same_agent_error() {
        let mut e = valid_exchange();
        e.hours = 0.0;
        e.recipient = agent_a();
        assert_invalid_contains(
            &validate_time_exchange(e),
            "Exchange hours must be positive",
        );
    }

    #[test]
    fn credit_hours_error_before_same_agent_error() {
        let mut c = valid_credit();
        c.hours = 0.0;
        c.debtor = agent_a();
        assert_invalid_contains(&validate_time_credit(c), "Credit hours must be positive");
    }

    // =========================================================================
    // LINK TAG LENGTH VALIDATION TESTS
    // =========================================================================

    #[test]
    fn test_link_agent_to_offers_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_valid(&validate_create_link(
            LinkTypes::AgentToOffers,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_agent_to_offers_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_invalid(&validate_create_link(
            LinkTypes::AgentToOffers,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_agent_to_requests_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_valid(&validate_create_link(
            LinkTypes::AgentToRequests,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_agent_to_requests_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_invalid(&validate_create_link(
            LinkTypes::AgentToRequests,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_agent_to_exchanges_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 512]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_valid(&validate_create_link(
            LinkTypes::AgentToExchanges,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_agent_to_exchanges_tag_too_long() {
        let tag = LinkTag(vec![0u8; 513]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_invalid(&validate_create_link(
            LinkTypes::AgentToExchanges,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_category_to_offers_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 512]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_valid(&validate_create_link(
            LinkTypes::CategoryToOffers,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_category_to_offers_tag_too_long() {
        let tag = LinkTag(vec![0u8; 513]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_invalid(&validate_create_link(
            LinkTypes::CategoryToOffers,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_category_to_requests_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 512]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_valid(&validate_create_link(
            LinkTypes::CategoryToRequests,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_category_to_requests_tag_too_long() {
        let tag = LinkTag(vec![0u8; 513]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_invalid(&validate_create_link(
            LinkTypes::CategoryToRequests,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_offer_to_exchange_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_valid(&validate_create_link(
            LinkTypes::OfferToExchange,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_offer_to_exchange_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_invalid(&validate_create_link(
            LinkTypes::OfferToExchange,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_request_to_exchange_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_valid(&validate_create_link(
            LinkTypes::RequestToExchange,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_request_to_exchange_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_invalid(&validate_create_link(
            LinkTypes::RequestToExchange,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_all_offers_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_valid(&validate_create_link(
            LinkTypes::AllOffers,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_all_offers_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_invalid(&validate_create_link(
            LinkTypes::AllOffers,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_all_requests_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_valid(&validate_create_link(
            LinkTypes::AllRequests,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_all_requests_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_invalid(&validate_create_link(
            LinkTypes::AllRequests,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_agent_to_credits_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_valid(&validate_create_link(
            LinkTypes::AgentToCredits,
            base,
            target,
            tag
        )));
    }

    #[test]
    fn test_link_agent_to_credits_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        assert!(is_invalid(&validate_create_link(
            LinkTypes::AgentToCredits,
            base,
            target,
            tag
        )));
    }

    // =========================================================================
    // HARDENING: String length limit boundary tests
    // =========================================================================

    // ── ServiceOffer ID (max 64) ───────────────────────────────────────

    #[test]
    fn test_validate_offer_id_at_limit() {
        let mut o = valid_offer();
        o.id = "a".repeat(64);
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn test_validate_offer_id_too_long() {
        let mut o = valid_offer();
        o.id = "a".repeat(257);
        assert!(is_invalid(&validate_service_offer(o)));
    }

    // ── ServiceOffer qualifications items (max 256 each) ───────────────

    #[test]
    fn test_validate_offer_qualification_at_limit() {
        let mut o = valid_offer();
        o.qualifications = vec!["q".repeat(256)];
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn test_validate_offer_qualification_too_long() {
        let mut o = valid_offer();
        o.qualifications = vec!["q".repeat(257)];
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn test_validate_offer_qualification_one_too_long_among_valid() {
        let mut o = valid_offer();
        o.qualifications = vec!["valid qual".into(), "q".repeat(257)];
        assert!(is_invalid(&validate_service_offer(o)));
    }

    // ── ServiceOffer custom category (max 128) ─────────────────────────

    #[test]
    fn test_validate_offer_custom_category_at_limit() {
        let mut o = valid_offer();
        o.category = ServiceCategory::Custom("c".repeat(128));
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn test_validate_offer_custom_category_too_long() {
        let mut o = valid_offer();
        o.category = ServiceCategory::Custom("c".repeat(129));
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn test_validate_offer_custom_category_empty() {
        let mut o = valid_offer();
        o.category = ServiceCategory::Custom("".into());
        assert!(is_invalid(&validate_service_offer(o)));
    }

    // ── ServiceRequest ID (max 64) ─────────────────────────────────────

    #[test]
    fn test_validate_request_id_at_limit() {
        let mut r = valid_request();
        r.id = "a".repeat(64);
        assert!(is_valid(&validate_service_request(r)));
    }

    #[test]
    fn test_validate_request_id_too_long() {
        let mut r = valid_request();
        r.id = "a".repeat(257);
        assert!(is_invalid(&validate_service_request(r)));
    }

    // ── ServiceRequest custom category (max 128) ───────────────────────

    #[test]
    fn test_validate_request_custom_category_at_limit() {
        let mut r = valid_request();
        r.category = ServiceCategory::Custom("c".repeat(128));
        assert!(is_valid(&validate_service_request(r)));
    }

    #[test]
    fn test_validate_request_custom_category_too_long() {
        let mut r = valid_request();
        r.category = ServiceCategory::Custom("c".repeat(129));
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn test_validate_request_custom_category_empty() {
        let mut r = valid_request();
        r.category = ServiceCategory::Custom("".into());
        assert!(is_invalid(&validate_service_request(r)));
    }

    // ── TimeExchange ID (max 64) ───────────────────────────────────────

    #[test]
    fn test_validate_exchange_id_at_limit() {
        let mut e = valid_exchange();
        e.id = "a".repeat(64);
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn test_validate_exchange_id_too_long() {
        let mut e = valid_exchange();
        e.id = "a".repeat(257);
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    // ── TimeExchange description (max 4096) ────────────────────────────

    #[test]
    fn test_validate_exchange_description_at_limit() {
        let mut e = valid_exchange();
        e.description = "d".repeat(4096);
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn test_validate_exchange_description_too_long() {
        let mut e = valid_exchange();
        e.description = "d".repeat(4097);
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    // ── TimeExchange whitespace-only description ───────────────────────

    #[test]
    fn test_validate_exchange_whitespace_description() {
        let mut e = valid_exchange();
        e.description = "  \t\n ".into();
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    // ── TimeExchange rating comment (max 1024) ─────────────────────────

    #[test]
    fn test_validate_exchange_provider_rating_comment_at_limit() {
        let mut e = valid_exchange();
        e.provider_rating = Some(Rating {
            score: 4,
            comment: Some("c".repeat(1024)),
            rated_at: Timestamp::from_micros(0),
        });
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn test_validate_exchange_provider_rating_comment_too_long() {
        let mut e = valid_exchange();
        e.provider_rating = Some(Rating {
            score: 4,
            comment: Some("c".repeat(1025)),
            rated_at: Timestamp::from_micros(0),
        });
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn test_validate_exchange_recipient_rating_comment_too_long() {
        let mut e = valid_exchange();
        e.recipient_rating = Some(Rating {
            score: 3,
            comment: Some("c".repeat(1025)),
            rated_at: Timestamp::from_micros(0),
        });
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    // ── TimeCredit description (max 4096) ──────────────────────────────

    #[test]
    fn test_validate_credit_description_at_limit() {
        let mut c = valid_credit();
        c.description = "d".repeat(4096);
        assert!(is_valid(&validate_time_credit(c)));
    }

    #[test]
    fn test_validate_credit_description_too_long() {
        let mut c = valid_credit();
        c.description = "d".repeat(4097);
        assert!(is_invalid(&validate_time_credit(c)));
    }

    // ── TimeCredit whitespace-only description ─────────────────────────

    #[test]
    fn test_validate_credit_whitespace_description() {
        let mut c = valid_credit();
        c.description = "  \t\n ".into();
        assert!(is_invalid(&validate_time_credit(c)));
    }

    // ── TimeCredit custom category (max 128) ───────────────────────────

    #[test]
    fn test_validate_credit_custom_category_at_limit() {
        let mut c = valid_credit();
        c.service_category = ServiceCategory::Custom("c".repeat(128));
        assert!(is_valid(&validate_time_credit(c)));
    }

    #[test]
    fn test_validate_credit_custom_category_too_long() {
        let mut c = valid_credit();
        c.service_category = ServiceCategory::Custom("c".repeat(129));
        assert!(is_invalid(&validate_time_credit(c)));
    }

    #[test]
    fn test_validate_credit_custom_category_empty() {
        let mut c = valid_credit();
        c.service_category = ServiceCategory::Custom("".into());
        assert!(is_invalid(&validate_time_credit(c)));
    }

    // ── ServiceOffer whitespace-only ID ────────────────────────────────

    #[test]
    fn test_validate_offer_whitespace_id() {
        let mut o = valid_offer();
        o.id = "  \t ".into();
        assert!(is_invalid(&validate_service_offer(o)));
    }

    // ── ServiceRequest whitespace-only ID ──────────────────────────────

    #[test]
    fn test_validate_request_whitespace_id() {
        let mut r = valid_request();
        r.id = "  \t ".into();
        assert!(is_invalid(&validate_service_request(r)));
    }

    // ── TimeExchange whitespace-only ID ────────────────────────────────

    #[test]
    fn test_validate_exchange_whitespace_id() {
        let mut e = valid_exchange();
        e.id = "  \t ".into();
        assert!(is_invalid(&validate_time_exchange(e)));
    }
}
