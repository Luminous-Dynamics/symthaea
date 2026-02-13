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
}

#[hdk_link_types]
pub enum LinkTypes {
    AllOffers,
    AllRequests,
    DriverToOffer,
    RequesterToRequest,
    OfferToMatch,
    RequestToMatch,
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
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::RideOffer(o) => validate_ride_offer(o),
                EntryTypes::RideMatch(_) => Ok(ValidateCallbackResult::Valid),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_ride_offer(o: RideOffer) -> ExternResult<ValidateCallbackResult> {
    if o.seats_available == 0 {
        return Ok(ValidateCallbackResult::Invalid("Must offer at least 1 seat".into()));
    }
    if o.price_per_seat < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Price cannot be negative".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_ride_request(r: RideRequest) -> ExternResult<ValidateCallbackResult> {
    if r.passengers == 0 {
        return Ok(ValidateCallbackResult::Invalid("Must request at least 1 passenger".into()));
    }
    if r.origin_lat < -90.0 || r.origin_lat > 90.0 || r.destination_lat < -90.0 || r.destination_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid("Latitude must be between -90 and 90".into()));
    }
    if r.origin_lon < -180.0 || r.origin_lon > 180.0 || r.destination_lon < -180.0 || r.destination_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid("Longitude must be between -180 and 180".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_cargo_offer(c: CargoOffer) -> ExternResult<ValidateCallbackResult> {
    if c.capacity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Cargo capacity must be positive".into()));
    }
    if c.price_per_kg < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Price cannot be negative".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}
