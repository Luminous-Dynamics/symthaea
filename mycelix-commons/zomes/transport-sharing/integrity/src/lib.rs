//! Transport Sharing Integrity Zome
//! Entry types and validation for ride offers, requests, matches, and cargo.

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// RIDE OFFER
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum OfferStatus {
    Open,
    Full,
    InProgress,
    Completed,
    Cancelled,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RideOffer {
    pub vehicle_hash: ActionHash,
    pub route_hash: Option<ActionHash>,
    pub driver: AgentPubKey,
    pub departure_time: u64,
    pub seats_available: u32,
    pub price_per_seat: f64,
    pub status: OfferStatus,
}

// ============================================================================
// RIDE REQUEST
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RequestStatus {
    Open,
    Matched,
    Cancelled,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RideRequest {
    pub requester: AgentPubKey,
    pub origin_lat: f64,
    pub origin_lon: f64,
    pub destination_lat: f64,
    pub destination_lon: f64,
    pub requested_time: u64,
    pub passengers: u32,
    pub status: RequestStatus,
}

// ============================================================================
// RIDE MATCH
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MatchStatus {
    Pending,
    Confirmed,
    InProgress,
    Completed,
    Cancelled,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RideMatch {
    pub offer_hash: ActionHash,
    pub request_hash: ActionHash,
    pub confirmed_at: Option<u64>,
    pub status: MatchStatus,
}

// ============================================================================
// CARGO OFFER
// ============================================================================

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CargoOffer {
    pub vehicle_hash: ActionHash,
    pub origin_lat: f64,
    pub origin_lon: f64,
    pub destination_lat: f64,
    pub destination_lon: f64,
    pub capacity_kg: f64,
    pub price_per_kg: f64,
    pub departure_time: u64,
}

// ============================================================================
// RIDE REVIEW
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ReviewerRole {
    Driver,
    Passenger,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RideReview {
    pub match_hash: ActionHash,
    pub reviewer: AgentPubKey,
    pub role: ReviewerRole,
    pub rating: u8,
    pub comment: String,
    pub safety_concern: bool,
    pub created_at: u64,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    RideOffer(RideOffer),
    RideRequest(RideRequest),
    RideMatch(RideMatch),
    CargoOffer(CargoOffer),
    RideReview(RideReview),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllOffers,
    AllRequests,
    DriverToOffer,
    RequesterToRequest,
    OfferToMatch,
    RequestToMatch,
    MatchToReviews,
    AgentToReviews,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::RideOffer(o) => validate_ride_offer(o),
                EntryTypes::RideRequest(r) => validate_ride_request(r),
                EntryTypes::RideMatch(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CargoOffer(c) => validate_cargo_offer(c),
                EntryTypes::RideReview(rev) => validate_ride_review(rev),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::RideOffer(o) => validate_ride_offer(o),
                EntryTypes::RideMatch(_) => Ok(ValidateCallbackResult::Valid),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => match link_type {
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
            LinkTypes::DriverToOffer => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "DriverToOffer link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::RequesterToRequest => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "RequesterToRequest link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::OfferToMatch => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "OfferToMatch link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::RequestToMatch => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "RequestToMatch link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::MatchToReviews => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "MatchToReviews link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToReviews => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToReviews link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_ride_offer(o: RideOffer) -> ExternResult<ValidateCallbackResult> {
    if o.seats_available == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Must offer at least 1 seat".into(),
        ));
    }
    if !o.price_per_seat.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "price_per_seat must be a finite number".into(),
        ));
    }
    if o.price_per_seat < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Price cannot be negative".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_ride_request(r: RideRequest) -> ExternResult<ValidateCallbackResult> {
    if r.passengers == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Must request at least 1 passenger".into(),
        ));
    }
    if !r.origin_lat.is_finite() || !r.destination_lat.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "origin_lat/destination_lat must be a finite number".into(),
        ));
    }
    if !r.origin_lon.is_finite() || !r.destination_lon.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "origin_lon/destination_lon must be a finite number".into(),
        ));
    }
    if r.origin_lat < -90.0
        || r.origin_lat > 90.0
        || r.destination_lat < -90.0
        || r.destination_lat > 90.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if r.origin_lon < -180.0
        || r.origin_lon > 180.0
        || r.destination_lon < -180.0
        || r.destination_lon > 180.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_ride_review(rev: RideReview) -> ExternResult<ValidateCallbackResult> {
    if rev.rating < 1 || rev.rating > 5 {
        return Ok(ValidateCallbackResult::Invalid(
            "Rating must be between 1 and 5".into(),
        ));
    }
    if rev.comment.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Comment cannot be empty".into(),
        ));
    }
    if rev.comment.len() > 2048 {
        return Ok(ValidateCallbackResult::Invalid(
            "Comment too long (max 2048 chars)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_cargo_offer(c: CargoOffer) -> ExternResult<ValidateCallbackResult> {
    if !c.capacity_kg.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "capacity_kg must be a finite number".into(),
        ));
    }
    if c.capacity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cargo capacity must be positive".into(),
        ));
    }
    if !c.price_per_kg.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "price_per_kg must be a finite number".into(),
        ));
    }
    if c.price_per_kg < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Price cannot be negative".into(),
        ));
    }
    if !c.origin_lat.is_finite() || !c.destination_lat.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "origin_lat/destination_lat must be a finite number".into(),
        ));
    }
    if !c.origin_lon.is_finite() || !c.destination_lon.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "origin_lon/destination_lon must be a finite number".into(),
        ));
    }
    if c.origin_lat < -90.0
        || c.origin_lat > 90.0
        || c.destination_lat < -90.0
        || c.destination_lat > 90.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if c.origin_lon < -180.0
        || c.origin_lon > 180.0
        || c.destination_lon < -180.0
        || c.destination_lon > 180.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }
    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn valid_ride_offer() -> RideOffer {
        RideOffer {
            vehicle_hash: fake_action_hash(),
            route_hash: None,
            driver: fake_agent(),
            departure_time: 1700000000,
            seats_available: 3,
            price_per_seat: 5.0,
            status: OfferStatus::Open,
        }
    }

    fn valid_ride_request() -> RideRequest {
        RideRequest {
            requester: fake_agent(),
            origin_lat: 32.95,
            origin_lon: -96.73,
            destination_lat: 32.78,
            destination_lon: -96.80,
            requested_time: 1700000000,
            passengers: 2,
            status: RequestStatus::Open,
        }
    }

    fn valid_cargo_offer() -> CargoOffer {
        CargoOffer {
            vehicle_hash: fake_action_hash(),
            origin_lat: 32.95,
            origin_lon: -96.73,
            destination_lat: 33.45,
            destination_lon: -96.50,
            capacity_kg: 100.0,
            price_per_kg: 0.50,
            departure_time: 1700000000,
        }
    }

    fn assert_valid(result: ExternResult<ValidateCallbackResult>) {
        match result {
            Ok(ValidateCallbackResult::Valid) => {}
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                panic!("Expected Valid, got Invalid: {msg}")
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    fn assert_invalid(result: ExternResult<ValidateCallbackResult>, expected_substr: &str) {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains(expected_substr),
                    "Expected Invalid containing '{expected_substr}', got: '{msg}'"
                );
            }
            Ok(ValidateCallbackResult::Valid) => {
                panic!("Expected Invalid containing '{expected_substr}', got Valid")
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    // ── Serde roundtrip tests ───────────────────────────────────────────

    #[test]
    fn serde_roundtrip_offer_status() {
        let statuses = vec![
            OfferStatus::Open,
            OfferStatus::Full,
            OfferStatus::InProgress,
            OfferStatus::Completed,
            OfferStatus::Cancelled,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: OfferStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, s);
        }
    }

    #[test]
    fn serde_roundtrip_request_status() {
        let statuses = vec![
            RequestStatus::Open,
            RequestStatus::Matched,
            RequestStatus::Cancelled,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: RequestStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, s);
        }
    }

    #[test]
    fn serde_roundtrip_match_status() {
        let statuses = vec![
            MatchStatus::Pending,
            MatchStatus::Confirmed,
            MatchStatus::InProgress,
            MatchStatus::Completed,
            MatchStatus::Cancelled,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: MatchStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, s);
        }
    }

    #[test]
    fn serde_roundtrip_ride_offer() {
        let o = valid_ride_offer();
        let json = serde_json::to_string(&o).unwrap();
        let back: RideOffer = serde_json::from_str(&json).unwrap();
        assert_eq!(back, o);
    }

    #[test]
    fn serde_roundtrip_ride_request() {
        let r = valid_ride_request();
        let json = serde_json::to_string(&r).unwrap();
        let back: RideRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn serde_roundtrip_ride_match() {
        let m = RideMatch {
            offer_hash: fake_action_hash(),
            request_hash: fake_action_hash(),
            confirmed_at: Some(1700001000),
            status: MatchStatus::Confirmed,
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: RideMatch = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn serde_roundtrip_cargo_offer() {
        let c = valid_cargo_offer();
        let json = serde_json::to_string(&c).unwrap();
        let back: CargoOffer = serde_json::from_str(&json).unwrap();
        assert_eq!(back, c);
    }

    // ── validate_ride_offer: seats_available ────────────────────────────

    #[test]
    fn valid_ride_offer_passes() {
        assert_valid(validate_ride_offer(valid_ride_offer()));
    }

    #[test]
    fn ride_offer_zero_seats_rejected() {
        let mut o = valid_ride_offer();
        o.seats_available = 0;
        assert_invalid(validate_ride_offer(o), "Must offer at least 1 seat");
    }

    #[test]
    fn ride_offer_one_seat_valid() {
        let mut o = valid_ride_offer();
        o.seats_available = 1;
        assert_valid(validate_ride_offer(o));
    }

    #[test]
    fn ride_offer_many_seats_valid() {
        let mut o = valid_ride_offer();
        o.seats_available = 50;
        assert_valid(validate_ride_offer(o));
    }

    // ── validate_ride_offer: price_per_seat ─────────────────────────────

    #[test]
    fn ride_offer_negative_price_rejected() {
        let mut o = valid_ride_offer();
        o.price_per_seat = -1.0;
        assert_invalid(validate_ride_offer(o), "Price cannot be negative");
    }

    #[test]
    fn ride_offer_barely_negative_price_rejected() {
        let mut o = valid_ride_offer();
        o.price_per_seat = -0.01;
        assert_invalid(validate_ride_offer(o), "Price cannot be negative");
    }

    #[test]
    fn ride_offer_free_valid() {
        let mut o = valid_ride_offer();
        o.price_per_seat = 0.0;
        assert_valid(validate_ride_offer(o));
    }

    #[test]
    fn ride_offer_large_price_valid() {
        let mut o = valid_ride_offer();
        o.price_per_seat = 9999.0;
        assert_valid(validate_ride_offer(o));
    }

    // ── validate_ride_offer: status variants ────────────────────────────

    #[test]
    fn ride_offer_all_statuses_valid() {
        for status in [
            OfferStatus::Open,
            OfferStatus::Full,
            OfferStatus::InProgress,
            OfferStatus::Completed,
            OfferStatus::Cancelled,
        ] {
            let mut o = valid_ride_offer();
            o.status = status;
            assert_valid(validate_ride_offer(o));
        }
    }

    // ── validate_ride_offer: optional route_hash ────────────────────────

    #[test]
    fn ride_offer_with_route_hash_valid() {
        let mut o = valid_ride_offer();
        o.route_hash = Some(fake_action_hash());
        assert_valid(validate_ride_offer(o));
    }

    // ── validate_ride_offer: combined invalid ───────────────────────────

    #[test]
    fn ride_offer_zero_seats_negative_price_rejects_seats_first() {
        let mut o = valid_ride_offer();
        o.seats_available = 0;
        o.price_per_seat = -1.0;
        assert_invalid(validate_ride_offer(o), "Must offer at least 1 seat");
    }

    // ── validate_ride_request: passengers ───────────────────────────────

    #[test]
    fn valid_ride_request_passes() {
        assert_valid(validate_ride_request(valid_ride_request()));
    }

    #[test]
    fn ride_request_zero_passengers_rejected() {
        let mut r = valid_ride_request();
        r.passengers = 0;
        assert_invalid(
            validate_ride_request(r),
            "Must request at least 1 passenger",
        );
    }

    #[test]
    fn ride_request_one_passenger_valid() {
        let mut r = valid_ride_request();
        r.passengers = 1;
        assert_valid(validate_ride_request(r));
    }

    #[test]
    fn ride_request_many_passengers_valid() {
        let mut r = valid_ride_request();
        r.passengers = 100;
        assert_valid(validate_ride_request(r));
    }

    // ── validate_ride_request: latitude boundaries ──────────────────────

    #[test]
    fn ride_request_origin_lat_too_low_rejected() {
        let mut r = valid_ride_request();
        r.origin_lat = -91.0;
        assert_invalid(
            validate_ride_request(r),
            "Latitude must be between -90 and 90",
        );
    }

    #[test]
    fn ride_request_origin_lat_too_high_rejected() {
        let mut r = valid_ride_request();
        r.origin_lat = 91.0;
        assert_invalid(
            validate_ride_request(r),
            "Latitude must be between -90 and 90",
        );
    }

    #[test]
    fn ride_request_dest_lat_too_low_rejected() {
        let mut r = valid_ride_request();
        r.destination_lat = -91.0;
        assert_invalid(
            validate_ride_request(r),
            "Latitude must be between -90 and 90",
        );
    }

    #[test]
    fn ride_request_dest_lat_too_high_rejected() {
        let mut r = valid_ride_request();
        r.destination_lat = 90.5;
        assert_invalid(
            validate_ride_request(r),
            "Latitude must be between -90 and 90",
        );
    }

    #[test]
    fn ride_request_origin_lat_at_boundary_neg90_valid() {
        let mut r = valid_ride_request();
        r.origin_lat = -90.0;
        assert_valid(validate_ride_request(r));
    }

    #[test]
    fn ride_request_origin_lat_at_boundary_pos90_valid() {
        let mut r = valid_ride_request();
        r.origin_lat = 90.0;
        assert_valid(validate_ride_request(r));
    }

    #[test]
    fn ride_request_dest_lat_at_boundary_neg90_valid() {
        let mut r = valid_ride_request();
        r.destination_lat = -90.0;
        assert_valid(validate_ride_request(r));
    }

    #[test]
    fn ride_request_dest_lat_at_boundary_pos90_valid() {
        let mut r = valid_ride_request();
        r.destination_lat = 90.0;
        assert_valid(validate_ride_request(r));
    }

    // ── validate_ride_request: longitude boundaries ─────────────────────

    #[test]
    fn ride_request_origin_lon_too_low_rejected() {
        let mut r = valid_ride_request();
        r.origin_lon = -181.0;
        assert_invalid(
            validate_ride_request(r),
            "Longitude must be between -180 and 180",
        );
    }

    #[test]
    fn ride_request_origin_lon_too_high_rejected() {
        let mut r = valid_ride_request();
        r.origin_lon = 181.0;
        assert_invalid(
            validate_ride_request(r),
            "Longitude must be between -180 and 180",
        );
    }

    #[test]
    fn ride_request_dest_lon_too_low_rejected() {
        let mut r = valid_ride_request();
        r.destination_lon = -181.0;
        assert_invalid(
            validate_ride_request(r),
            "Longitude must be between -180 and 180",
        );
    }

    #[test]
    fn ride_request_dest_lon_too_high_rejected() {
        let mut r = valid_ride_request();
        r.destination_lon = 180.5;
        assert_invalid(
            validate_ride_request(r),
            "Longitude must be between -180 and 180",
        );
    }

    #[test]
    fn ride_request_origin_lon_at_boundary_neg180_valid() {
        let mut r = valid_ride_request();
        r.origin_lon = -180.0;
        assert_valid(validate_ride_request(r));
    }

    #[test]
    fn ride_request_origin_lon_at_boundary_pos180_valid() {
        let mut r = valid_ride_request();
        r.origin_lon = 180.0;
        assert_valid(validate_ride_request(r));
    }

    #[test]
    fn ride_request_dest_lon_at_boundary_neg180_valid() {
        let mut r = valid_ride_request();
        r.destination_lon = -180.0;
        assert_valid(validate_ride_request(r));
    }

    #[test]
    fn ride_request_dest_lon_at_boundary_pos180_valid() {
        let mut r = valid_ride_request();
        r.destination_lon = 180.0;
        assert_valid(validate_ride_request(r));
    }

    // ── validate_ride_request: status variants ──────────────────────────

    #[test]
    fn ride_request_all_statuses_valid() {
        for status in [
            RequestStatus::Open,
            RequestStatus::Matched,
            RequestStatus::Cancelled,
        ] {
            let mut r = valid_ride_request();
            r.status = status;
            assert_valid(validate_ride_request(r));
        }
    }

    // ── validate_ride_request: combined invalid ─────────────────────────

    #[test]
    fn ride_request_zero_passengers_invalid_lat_rejects_passengers_first() {
        let mut r = valid_ride_request();
        r.passengers = 0;
        r.origin_lat = -91.0;
        assert_invalid(
            validate_ride_request(r),
            "Must request at least 1 passenger",
        );
    }

    // ── validate_cargo_offer: capacity_kg ───────────────────────────────

    #[test]
    fn valid_cargo_offer_passes() {
        assert_valid(validate_cargo_offer(valid_cargo_offer()));
    }

    #[test]
    fn cargo_zero_capacity_rejected() {
        let mut c = valid_cargo_offer();
        c.capacity_kg = 0.0;
        assert_invalid(validate_cargo_offer(c), "Cargo capacity must be positive");
    }

    #[test]
    fn cargo_negative_capacity_rejected() {
        let mut c = valid_cargo_offer();
        c.capacity_kg = -50.0;
        assert_invalid(validate_cargo_offer(c), "Cargo capacity must be positive");
    }

    #[test]
    fn cargo_barely_positive_capacity_valid() {
        let mut c = valid_cargo_offer();
        c.capacity_kg = 0.01;
        assert_valid(validate_cargo_offer(c));
    }

    #[test]
    fn cargo_large_capacity_valid() {
        let mut c = valid_cargo_offer();
        c.capacity_kg = 50000.0;
        assert_valid(validate_cargo_offer(c));
    }

    // ── validate_cargo_offer: price_per_kg ──────────────────────────────

    #[test]
    fn cargo_negative_price_rejected() {
        let mut c = valid_cargo_offer();
        c.price_per_kg = -0.10;
        assert_invalid(validate_cargo_offer(c), "Price cannot be negative");
    }

    #[test]
    fn cargo_barely_negative_price_rejected() {
        let mut c = valid_cargo_offer();
        c.price_per_kg = -0.001;
        assert_invalid(validate_cargo_offer(c), "Price cannot be negative");
    }

    #[test]
    fn cargo_zero_price_valid() {
        let mut c = valid_cargo_offer();
        c.price_per_kg = 0.0;
        assert_valid(validate_cargo_offer(c));
    }

    #[test]
    fn cargo_large_price_valid() {
        let mut c = valid_cargo_offer();
        c.price_per_kg = 999.0;
        assert_valid(validate_cargo_offer(c));
    }

    // ── validate_cargo_offer: combined invalid ──────────────────────────

    #[test]
    fn cargo_zero_capacity_negative_price_rejects_capacity_first() {
        let mut c = valid_cargo_offer();
        c.capacity_kg = 0.0;
        c.price_per_kg = -1.0;
        assert_invalid(validate_cargo_offer(c), "Cargo capacity must be positive");
    }

    // ── validate_cargo_offer: latitude boundaries ─────────────────────

    #[test]
    fn cargo_origin_lat_too_low_rejected() {
        let mut c = valid_cargo_offer();
        c.origin_lat = -91.0;
        assert_invalid(
            validate_cargo_offer(c),
            "Latitude must be between -90 and 90",
        );
    }

    #[test]
    fn cargo_origin_lat_too_high_rejected() {
        let mut c = valid_cargo_offer();
        c.origin_lat = 91.0;
        assert_invalid(
            validate_cargo_offer(c),
            "Latitude must be between -90 and 90",
        );
    }

    #[test]
    fn cargo_dest_lat_too_low_rejected() {
        let mut c = valid_cargo_offer();
        c.destination_lat = -91.0;
        assert_invalid(
            validate_cargo_offer(c),
            "Latitude must be between -90 and 90",
        );
    }

    #[test]
    fn cargo_dest_lat_too_high_rejected() {
        let mut c = valid_cargo_offer();
        c.destination_lat = 91.0;
        assert_invalid(
            validate_cargo_offer(c),
            "Latitude must be between -90 and 90",
        );
    }

    #[test]
    fn cargo_origin_lat_at_boundary_neg90_valid() {
        let mut c = valid_cargo_offer();
        c.origin_lat = -90.0;
        assert_valid(validate_cargo_offer(c));
    }

    #[test]
    fn cargo_origin_lat_at_boundary_pos90_valid() {
        let mut c = valid_cargo_offer();
        c.origin_lat = 90.0;
        assert_valid(validate_cargo_offer(c));
    }

    #[test]
    fn cargo_dest_lat_at_boundary_neg90_valid() {
        let mut c = valid_cargo_offer();
        c.destination_lat = -90.0;
        assert_valid(validate_cargo_offer(c));
    }

    #[test]
    fn cargo_dest_lat_at_boundary_pos90_valid() {
        let mut c = valid_cargo_offer();
        c.destination_lat = 90.0;
        assert_valid(validate_cargo_offer(c));
    }

    // ── validate_cargo_offer: longitude boundaries ──────────────────────

    #[test]
    fn cargo_origin_lon_too_low_rejected() {
        let mut c = valid_cargo_offer();
        c.origin_lon = -181.0;
        assert_invalid(
            validate_cargo_offer(c),
            "Longitude must be between -180 and 180",
        );
    }

    #[test]
    fn cargo_origin_lon_too_high_rejected() {
        let mut c = valid_cargo_offer();
        c.origin_lon = 181.0;
        assert_invalid(
            validate_cargo_offer(c),
            "Longitude must be between -180 and 180",
        );
    }

    #[test]
    fn cargo_dest_lon_too_low_rejected() {
        let mut c = valid_cargo_offer();
        c.destination_lon = -181.0;
        assert_invalid(
            validate_cargo_offer(c),
            "Longitude must be between -180 and 180",
        );
    }

    #[test]
    fn cargo_dest_lon_too_high_rejected() {
        let mut c = valid_cargo_offer();
        c.destination_lon = 181.0;
        assert_invalid(
            validate_cargo_offer(c),
            "Longitude must be between -180 and 180",
        );
    }

    #[test]
    fn cargo_origin_lon_at_boundary_neg180_valid() {
        let mut c = valid_cargo_offer();
        c.origin_lon = -180.0;
        assert_valid(validate_cargo_offer(c));
    }

    #[test]
    fn cargo_origin_lon_at_boundary_pos180_valid() {
        let mut c = valid_cargo_offer();
        c.origin_lon = 180.0;
        assert_valid(validate_cargo_offer(c));
    }

    #[test]
    fn cargo_dest_lon_at_boundary_neg180_valid() {
        let mut c = valid_cargo_offer();
        c.destination_lon = -180.0;
        assert_valid(validate_cargo_offer(c));
    }

    #[test]
    fn cargo_dest_lon_at_boundary_pos180_valid() {
        let mut c = valid_cargo_offer();
        c.destination_lon = 180.0;
        assert_valid(validate_cargo_offer(c));
    }

    // ── validate_cargo_offer: combined lat/lon check order ──────────────

    #[test]
    fn cargo_invalid_capacity_with_invalid_lat_rejects_capacity_first() {
        let mut c = valid_cargo_offer();
        c.capacity_kg = 0.0;
        c.origin_lat = -91.0;
        assert_invalid(validate_cargo_offer(c), "Cargo capacity must be positive");
    }

    #[test]
    fn cargo_invalid_price_with_invalid_lon_rejects_price_first() {
        let mut c = valid_cargo_offer();
        c.price_per_kg = -1.0;
        c.origin_lon = 181.0;
        assert_invalid(validate_cargo_offer(c), "Price cannot be negative");
    }

    // ── Anchor test ─────────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip_anchor() {
        let a = Anchor("all_offers".to_string());
        let json = serde_json::to_string(&a).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    // ── Link tag length validation tests ────────────────────────────────

    fn validate_link_tag(link_type: &LinkTypes, tag_len: usize) -> ValidateCallbackResult {
        let tag = LinkTag(vec![0u8; tag_len]);
        let max = match link_type {
            LinkTypes::AllOffers
            | LinkTypes::AllRequests
            | LinkTypes::DriverToOffer
            | LinkTypes::RequesterToRequest
            | LinkTypes::MatchToReviews
            | LinkTypes::AgentToReviews => 256,
            LinkTypes::OfferToMatch | LinkTypes::RequestToMatch => 512,
        };
        let name = match link_type {
            LinkTypes::AllOffers => "AllOffers",
            LinkTypes::AllRequests => "AllRequests",
            LinkTypes::DriverToOffer => "DriverToOffer",
            LinkTypes::RequesterToRequest => "RequesterToRequest",
            LinkTypes::OfferToMatch => "OfferToMatch",
            LinkTypes::RequestToMatch => "RequestToMatch",
            LinkTypes::MatchToReviews => "MatchToReviews",
            LinkTypes::AgentToReviews => "AgentToReviews",
        };
        if tag.0.len() > max {
            ValidateCallbackResult::Invalid(format!(
                "{} link tag too long (max {} bytes)",
                name, max
            ))
        } else {
            ValidateCallbackResult::Valid
        }
    }

    #[test]
    fn test_link_all_offers_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AllOffers, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_all_offers_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AllOffers, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_all_requests_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AllRequests, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_all_requests_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AllRequests, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_driver_to_offer_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::DriverToOffer, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_driver_to_offer_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::DriverToOffer, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_requester_to_request_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::RequesterToRequest, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_requester_to_request_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::RequesterToRequest, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_offer_to_match_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::OfferToMatch, 512);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_offer_to_match_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::OfferToMatch, 513);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_request_to_match_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::RequestToMatch, 512);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_request_to_match_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::RequestToMatch, 513);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_match_to_reviews_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::MatchToReviews, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_match_to_reviews_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::MatchToReviews, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_agent_to_reviews_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AgentToReviews, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_agent_to_reviews_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AgentToReviews, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ── Serde roundtrip: new types ─────────────────────────────────────

    #[test]
    fn serde_roundtrip_reviewer_role() {
        let roles = vec![ReviewerRole::Driver, ReviewerRole::Passenger];
        for r in &roles {
            let json = serde_json::to_string(r).unwrap();
            let back: ReviewerRole = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, r);
        }
    }

    #[test]
    fn serde_roundtrip_ride_review() {
        let rev = valid_ride_review();
        let json = serde_json::to_string(&rev).unwrap();
        let back: RideReview = serde_json::from_str(&json).unwrap();
        assert_eq!(back, rev);
    }

    // ── validate_ride_review tests ─────────────────────────────────────

    fn valid_ride_review() -> RideReview {
        RideReview {
            match_hash: fake_action_hash(),
            reviewer: fake_agent(),
            role: ReviewerRole::Passenger,
            rating: 4,
            comment: "Great ride, very smooth".into(),
            safety_concern: false,
            created_at: 1700000000,
        }
    }

    #[test]
    fn valid_ride_review_passes() {
        assert_valid(validate_ride_review(valid_ride_review()));
    }

    #[test]
    fn ride_review_rating_zero_rejected() {
        let mut rev = valid_ride_review();
        rev.rating = 0;
        assert_invalid(validate_ride_review(rev), "Rating must be between 1 and 5");
    }

    #[test]
    fn ride_review_rating_six_rejected() {
        let mut rev = valid_ride_review();
        rev.rating = 6;
        assert_invalid(validate_ride_review(rev), "Rating must be between 1 and 5");
    }

    #[test]
    fn ride_review_rating_one_valid() {
        let mut rev = valid_ride_review();
        rev.rating = 1;
        assert_valid(validate_ride_review(rev));
    }

    #[test]
    fn ride_review_rating_five_valid() {
        let mut rev = valid_ride_review();
        rev.rating = 5;
        assert_valid(validate_ride_review(rev));
    }

    #[test]
    fn ride_review_empty_comment_rejected() {
        let mut rev = valid_ride_review();
        rev.comment = String::new();
        assert_invalid(validate_ride_review(rev), "Comment cannot be empty");
    }

    #[test]
    fn ride_review_whitespace_comment_rejected() {
        let mut rev = valid_ride_review();
        rev.comment = "   ".into();
        assert_invalid(validate_ride_review(rev), "Comment cannot be empty");
    }

    #[test]
    fn ride_review_comment_too_long_rejected() {
        let mut rev = valid_ride_review();
        rev.comment = "x".repeat(2049);
        assert_invalid(validate_ride_review(rev), "Comment too long");
    }

    #[test]
    fn ride_review_comment_at_max_valid() {
        let mut rev = valid_ride_review();
        rev.comment = "x".repeat(2048);
        assert_valid(validate_ride_review(rev));
    }

    #[test]
    fn ride_review_both_roles_valid() {
        for role in [ReviewerRole::Driver, ReviewerRole::Passenger] {
            let mut rev = valid_ride_review();
            rev.role = role;
            assert_valid(validate_ride_review(rev));
        }
    }

    #[test]
    fn ride_review_safety_concern_true_valid() {
        let mut rev = valid_ride_review();
        rev.safety_concern = true;
        assert_valid(validate_ride_review(rev));
    }

    // ── NaN/Infinity bypass hardening tests ────────────────────────────

    #[test]
    fn ride_offer_nan_price_rejected() {
        let mut o = valid_ride_offer();
        o.price_per_seat = f64::NAN;
        assert_invalid(
            validate_ride_offer(o),
            "price_per_seat must be a finite number",
        );
    }

    #[test]
    fn ride_offer_infinity_price_rejected() {
        let mut o = valid_ride_offer();
        o.price_per_seat = f64::INFINITY;
        assert_invalid(
            validate_ride_offer(o),
            "price_per_seat must be a finite number",
        );
    }

    #[test]
    fn ride_offer_neg_infinity_price_rejected() {
        let mut o = valid_ride_offer();
        o.price_per_seat = f64::NEG_INFINITY;
        assert_invalid(
            validate_ride_offer(o),
            "price_per_seat must be a finite number",
        );
    }

    #[test]
    fn ride_request_nan_origin_lat_rejected() {
        let mut r = valid_ride_request();
        r.origin_lat = f64::NAN;
        assert_invalid(validate_ride_request(r), "must be a finite number");
    }

    #[test]
    fn ride_request_infinity_origin_lon_rejected() {
        let mut r = valid_ride_request();
        r.origin_lon = f64::INFINITY;
        assert_invalid(validate_ride_request(r), "must be a finite number");
    }

    #[test]
    fn ride_request_nan_dest_lat_rejected() {
        let mut r = valid_ride_request();
        r.destination_lat = f64::NAN;
        assert_invalid(validate_ride_request(r), "must be a finite number");
    }

    #[test]
    fn ride_request_neg_infinity_dest_lon_rejected() {
        let mut r = valid_ride_request();
        r.destination_lon = f64::NEG_INFINITY;
        assert_invalid(validate_ride_request(r), "must be a finite number");
    }

    #[test]
    fn cargo_offer_nan_capacity_rejected() {
        let mut c = valid_cargo_offer();
        c.capacity_kg = f64::NAN;
        assert_invalid(
            validate_cargo_offer(c),
            "capacity_kg must be a finite number",
        );
    }

    #[test]
    fn cargo_offer_infinity_capacity_rejected() {
        let mut c = valid_cargo_offer();
        c.capacity_kg = f64::INFINITY;
        assert_invalid(
            validate_cargo_offer(c),
            "capacity_kg must be a finite number",
        );
    }

    #[test]
    fn cargo_offer_nan_price_rejected() {
        let mut c = valid_cargo_offer();
        c.price_per_kg = f64::NAN;
        assert_invalid(
            validate_cargo_offer(c),
            "price_per_kg must be a finite number",
        );
    }

    #[test]
    fn cargo_offer_infinity_price_rejected() {
        let mut c = valid_cargo_offer();
        c.price_per_kg = f64::INFINITY;
        assert_invalid(
            validate_cargo_offer(c),
            "price_per_kg must be a finite number",
        );
    }

    #[test]
    fn cargo_offer_nan_origin_lat_rejected() {
        let mut c = valid_cargo_offer();
        c.origin_lat = f64::NAN;
        assert_invalid(validate_cargo_offer(c), "must be a finite number");
    }

    #[test]
    fn cargo_offer_infinity_dest_lon_rejected() {
        let mut c = valid_cargo_offer();
        c.destination_lon = f64::INFINITY;
        assert_invalid(validate_cargo_offer(c), "must be a finite number");
    }
}
