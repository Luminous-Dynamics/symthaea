//! Timebank Integrity Zome
//!
//! This zome defines the entry types and validation rules for time banking
//! in the Mycelix Mutual Aid hApp. It implements the core principle:
//! 1 hour = 1 hour, regardless of service type.

use hdi::prelude::*;
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
        FlatOp::RegisterDeleteLink { link_type, .. } => {
            let _ = link_type;
            Ok(ValidateCallbackResult::Valid)
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
    if offer.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Service offer ID cannot be empty".to_string(),
        ));
    }

    // Title must not be empty
    if offer.title.is_empty() {
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

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a service request
fn validate_service_request(request: ServiceRequest) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if request.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Service request ID cannot be empty".to_string(),
        ));
    }

    // Title must not be empty
    if request.title.is_empty() {
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

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a time exchange
fn validate_time_exchange(exchange: TimeExchange) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if exchange.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Time exchange ID cannot be empty".to_string(),
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
    if exchange.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange description cannot be empty".to_string(),
        ));
    }

    // Validate ratings if present
    if let Some(rating) = &exchange.provider_rating {
        if rating.score < 1 || rating.score > 5 {
            return Ok(ValidateCallbackResult::Invalid(
                "Rating score must be between 1 and 5".to_string(),
            ));
        }
    }

    if let Some(rating) = &exchange.recipient_rating {
        if rating.score < 1 || rating.score > 5 {
            return Ok(ValidateCallbackResult::Invalid(
                "Rating score must be between 1 and 5".to_string(),
            ));
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
    if credit.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit description cannot be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate link creation
fn validate_create_link(
    link_type: LinkTypes,
    _base_address: AnyLinkableHash,
    _target_address: AnyLinkableHash,
    _tag: LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    match link_type {
        LinkTypes::AgentToOffers => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AgentToRequests => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AgentToExchanges => Ok(ValidateCallbackResult::Valid),
        LinkTypes::CategoryToOffers => Ok(ValidateCallbackResult::Valid),
        LinkTypes::CategoryToRequests => Ok(ValidateCallbackResult::Valid),
        LinkTypes::OfferToExchange => Ok(ValidateCallbackResult::Valid),
        LinkTypes::RequestToExchange => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllOffers => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllRequests => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AgentToCredits => Ok(ValidateCallbackResult::Valid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xAA; 36])
    }
    fn agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xBB; 36])
    }

    fn is_valid(r: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(r, Ok(ValidateCallbackResult::Valid))
    }
    fn is_invalid(r: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(r, Ok(ValidateCallbackResult::Invalid(_)))
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

    // ---- ServiceOffer validation ----

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
    fn offer_empty_title_rejected() {
        let mut o = valid_offer();
        o.title = "".into();
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_title_over_200_rejected() {
        let mut o = valid_offer();
        o.title = "x".repeat(201);
        assert!(is_invalid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_title_at_200_accepted() {
        let mut o = valid_offer();
        o.title = "x".repeat(200);
        assert!(is_valid(&validate_service_offer(o)));
    }

    #[test]
    fn offer_description_over_5000_rejected() {
        let mut o = valid_offer();
        o.description = "x".repeat(5001);
        assert!(is_invalid(&validate_service_offer(o)));
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
    fn offer_max_less_than_min_rejected() {
        let mut o = valid_offer();
        o.min_duration_hours = 2.0;
        o.max_duration_hours = Some(1.0);
        assert!(is_invalid(&validate_service_offer(o)));
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
    fn offer_20_qualifications_accepted() {
        let mut o = valid_offer();
        o.qualifications = (0..20).map(|i| format!("qual_{}", i)).collect();
        assert!(is_valid(&validate_service_offer(o)));
    }

    // ---- ServiceRequest validation ----

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
    fn request_empty_title_rejected() {
        let mut r = valid_request();
        r.title = "".into();
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_title_over_200_rejected() {
        let mut r = valid_request();
        r.title = "x".repeat(201);
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_description_over_5000_rejected() {
        let mut r = valid_request();
        r.description = "x".repeat(5001);
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_zero_hours_rejected() {
        let mut r = valid_request();
        r.estimated_hours = 0.0;
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_negative_hours_rejected() {
        let mut r = valid_request();
        r.estimated_hours = -1.0;
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_over_168_hours_rejected() {
        let mut r = valid_request();
        r.estimated_hours = 168.1;
        assert!(is_invalid(&validate_service_request(r)));
    }

    #[test]
    fn request_exactly_168_hours_accepted() {
        let mut r = valid_request();
        r.estimated_hours = 168.0;
        assert!(is_valid(&validate_service_request(r)));
    }

    // ---- TimeExchange validation ----

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
    fn exchange_zero_hours_rejected() {
        let mut e = valid_exchange();
        e.hours = 0.0;
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_over_168_hours_rejected() {
        let mut e = valid_exchange();
        e.hours = 169.0;
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_same_provider_recipient_rejected() {
        let mut e = valid_exchange();
        e.recipient = agent_a(); // same as provider
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_empty_description_rejected() {
        let mut e = valid_exchange();
        e.description = "".into();
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_rating_score_0_rejected() {
        let mut e = valid_exchange();
        e.provider_rating = Some(Rating {
            score: 0,
            comment: None,
            rated_at: Timestamp::from_micros(0),
        });
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_rating_score_6_rejected() {
        let mut e = valid_exchange();
        e.recipient_rating = Some(Rating {
            score: 6,
            comment: None,
            rated_at: Timestamp::from_micros(0),
        });
        assert!(is_invalid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_rating_score_1_accepted() {
        let mut e = valid_exchange();
        e.provider_rating = Some(Rating {
            score: 1,
            comment: None,
            rated_at: Timestamp::from_micros(0),
        });
        assert!(is_valid(&validate_time_exchange(e)));
    }

    #[test]
    fn exchange_rating_score_5_accepted() {
        let mut e = valid_exchange();
        e.recipient_rating = Some(Rating {
            score: 5,
            comment: Some("Excellent!".into()),
            rated_at: Timestamp::from_micros(0),
        });
        assert!(is_valid(&validate_time_exchange(e)));
    }

    // ---- TimeCredit validation ----

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
    fn credit_over_168_hours_rejected() {
        let mut c = valid_credit();
        c.hours = 200.0;
        assert!(is_invalid(&validate_time_credit(c)));
    }

    #[test]
    fn credit_same_earner_debtor_rejected() {
        let mut c = valid_credit();
        c.debtor = agent_a(); // same as earner
        assert!(is_invalid(&validate_time_credit(c)));
    }

    #[test]
    fn credit_empty_description_rejected() {
        let mut c = valid_credit();
        c.description = "".into();
        assert!(is_invalid(&validate_time_credit(c)));
    }
}
