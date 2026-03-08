//! Transport Sharing Coordinator Zome
//! Business logic for ride offers, requests, matching, and cargo coordination.

use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};
use transport_sharing_integrity::*;

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============================================================================
// RIDE OFFERS
// ============================================================================

#[hdk_extern]
pub fn post_ride_offer(offer: RideOffer) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "post_ride_offer")?;
    let action_hash = create_entry(&EntryTypes::RideOffer(offer.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_offers".to_string())))?;
    create_link(
        anchor_hash("all_offers")?,
        action_hash.clone(),
        LinkTypes::AllOffers,
        (),
    )?;
    create_link(
        offer.driver,
        action_hash.clone(),
        LinkTypes::DriverToOffer,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created offer".into()
    )))
}

#[hdk_extern]
pub fn get_available_rides(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_offers")?, LinkTypes::AllOffers)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// RIDE REQUESTS
// ============================================================================

#[hdk_extern]
pub fn request_ride(request: RideRequest) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "request_ride")?;
    let action_hash = create_entry(&EntryTypes::RideRequest(request.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_requests".to_string())))?;
    create_link(
        anchor_hash("all_requests")?,
        action_hash.clone(),
        LinkTypes::AllRequests,
        (),
    )?;
    create_link(
        request.requester,
        action_hash.clone(),
        LinkTypes::RequesterToRequest,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created request".into()
    )))
}

// ============================================================================
// MATCHING
// ============================================================================

#[hdk_extern]
pub fn match_ride(ride_match: RideMatch) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "match_ride")?;
    let _offer = get(ride_match.offer_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Offer not found".into())))?;
    let _request = get(ride_match.request_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Request not found".into())),
    )?;

    let action_hash = create_entry(&EntryTypes::RideMatch(ride_match.clone()))?;
    create_link(
        ride_match.offer_hash,
        action_hash.clone(),
        LinkTypes::OfferToMatch,
        (),
    )?;
    create_link(
        ride_match.request_hash,
        action_hash.clone(),
        LinkTypes::RequestToMatch,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created match".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateMatchStatusInput {
    pub match_hash: ActionHash,
    pub new_status: MatchStatus,
}

#[hdk_extern]
pub fn confirm_match(input: UpdateMatchStatusInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "confirm_match")?;
    let now = sys_time()?.as_micros() / 1_000_000;
    let record = get(input.match_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Match not found".into())))?;
    let mut ride_match: RideMatch = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid match entry".into()
        )))?;

    ride_match.status = MatchStatus::Confirmed;
    ride_match.confirmed_at = Some(now as u64);
    let new_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::RideMatch(ride_match),
    )?;
    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated match".into()
    )))
}

#[hdk_extern]
pub fn complete_ride(match_hash: ActionHash) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "complete_ride")?;
    let record = get(match_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Match not found".into())))?;
    let mut ride_match: RideMatch = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid match entry".into()
        )))?;

    ride_match.status = MatchStatus::Completed;
    let new_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::RideMatch(ride_match),
    )?;
    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated match".into()
    )))
}

#[hdk_extern]
pub fn cancel_ride(match_hash: ActionHash) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "cancel_ride")?;
    let record = get(match_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Match not found".into())))?;
    let mut ride_match: RideMatch = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid match entry".into()
        )))?;

    ride_match.status = MatchStatus::Cancelled;
    let new_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::RideMatch(ride_match),
    )?;
    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated match".into()
    )))
}

// ============================================================================
// CARGO
// ============================================================================

#[hdk_extern]
pub fn post_cargo_offer(cargo: CargoOffer) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "post_cargo_offer")?;
    let _vehicle = get(cargo.vehicle_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Vehicle not found".into())
    ))?;

    let action_hash = create_entry(&EntryTypes::CargoOffer(cargo))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_offers".to_string())))?;
    create_link(
        anchor_hash("all_offers")?,
        action_hash.clone(),
        LinkTypes::AllOffers,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created cargo offer".into()
    )))
}

// ============================================================================
// REVIEWS
// ============================================================================

#[hdk_extern]
pub fn review_ride(review: RideReview) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "review_ride")?;
    let action_hash = create_entry(&EntryTypes::RideReview(review.clone()))?;

    create_link(
        review.match_hash,
        action_hash.clone(),
        LinkTypes::MatchToReviews,
        (),
    )?;
    create_link(
        review.reviewer,
        action_hash.clone(),
        LinkTypes::AgentToReviews,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created review".into()
    )))
}

#[hdk_extern]
pub fn get_ride_reviews(match_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(match_hash, LinkTypes::MatchToReviews)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DriverRating {
    pub average_rating: f64,
    pub total_reviews: u32,
}

#[hdk_extern]
pub fn get_driver_rating(agent: AgentPubKey) -> ExternResult<DriverRating> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToReviews)?,
        GetStrategy::default(),
    )?;

    let mut total: u64 = 0;
    let mut count: u32 = 0;

    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Ok(Some(review)) = record.entry().to_app_option::<RideReview>() {
                total += review.rating as u64;
                count += 1;
            }
        }
    }

    let average = if count > 0 {
        total as f64 / count as f64
    } else {
        0.0
    };
    Ok(DriverRating {
        average_rating: average,
        total_reviews: count,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FindNearbyInput {
    pub origin_lat: f64,
    pub origin_lon: f64,
    pub radius_km: f64,
}

#[hdk_extern]
pub fn find_nearby_rides(input: FindNearbyInput) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_offers")?, LinkTypes::AllOffers)?,
        GetStrategy::default(),
    )?;

    let mut nearby = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
            // Check if it's a RideOffer — we can't filter by location on RideOffer
            // since it only has route_hash, but we can return all offers within
            // the system for now and let the client filter by route details.
            // For CargoOffer, we can filter by origin coordinates.
            if let Ok(Some(cargo)) = record.entry().to_app_option::<CargoOffer>() {
                let dist = haversine_km(
                    input.origin_lat,
                    input.origin_lon,
                    cargo.origin_lat,
                    cargo.origin_lon,
                );
                if dist <= input.radius_km {
                    nearby.push(record);
                }
            }
        }
    }
    Ok(nearby)
}

fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0; // Earth radius in km
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    r * c
}

// ============================================================================
// MY RIDES
// ============================================================================

#[hdk_extern]
pub fn get_my_rides(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let mut all = Vec::new();

    // Get rides as driver
    let driver_links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::DriverToOffer)?,
        GetStrategy::default(),
    )?;
    all.extend(records_from_links(driver_links)?);

    // Get rides as requester
    let requester_links = get_links(
        LinkQuery::try_new(agent, LinkTypes::RequesterToRequest)?,
        GetStrategy::default(),
    )?;
    all.extend(records_from_links(requester_links)?);

    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_match_status_input_serde_confirmed() {
        let input = UpdateMatchStatusInput {
            match_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            new_status: MatchStatus::Confirmed,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateMatchStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, MatchStatus::Confirmed);
    }

    #[test]
    fn update_match_status_input_serde_completed() {
        let input = UpdateMatchStatusInput {
            match_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            new_status: MatchStatus::Completed,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateMatchStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, MatchStatus::Completed);
    }

    #[test]
    fn match_status_all_variants_serialize() {
        let statuses = vec![
            MatchStatus::Pending,
            MatchStatus::Confirmed,
            MatchStatus::InProgress,
            MatchStatus::Completed,
            MatchStatus::Cancelled,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let decoded: MatchStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, status);
        }
    }

    #[test]
    fn offer_status_all_variants_serialize() {
        let statuses = vec![
            OfferStatus::Open,
            OfferStatus::Full,
            OfferStatus::InProgress,
            OfferStatus::Completed,
            OfferStatus::Cancelled,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let decoded: OfferStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, status);
        }
    }

    #[test]
    fn request_status_all_variants_serialize() {
        let statuses = vec![
            RequestStatus::Open,
            RequestStatus::Matched,
            RequestStatus::Cancelled,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let decoded: RequestStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, status);
        }
    }

    // ========================================================================
    // RideOffer struct serde roundtrip
    // ========================================================================

    #[test]
    fn ride_offer_serde_roundtrip_with_route() {
        let offer = RideOffer {
            vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            route_hash: Some(ActionHash::from_raw_36(vec![0xcd; 36])),
            driver: AgentPubKey::from_raw_36(vec![0xab; 36]),
            departure_time: 1700000000,
            seats_available: 3,
            price_per_seat: 5.0,
            status: OfferStatus::Open,
        };
        let json = serde_json::to_string(&offer).unwrap();
        let decoded: RideOffer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.departure_time, 1700000000);
        assert_eq!(decoded.seats_available, 3);
        assert_eq!(decoded.price_per_seat, 5.0);
        assert_eq!(decoded.status, OfferStatus::Open);
        assert!(decoded.route_hash.is_some());
    }

    #[test]
    fn ride_offer_serde_roundtrip_without_route() {
        let offer = RideOffer {
            vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            route_hash: None,
            driver: AgentPubKey::from_raw_36(vec![0xab; 36]),
            departure_time: 1700000000,
            seats_available: 1,
            price_per_seat: 0.0,
            status: OfferStatus::Full,
        };
        let json = serde_json::to_string(&offer).unwrap();
        let decoded: RideOffer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.route_hash, None);
        assert_eq!(decoded.seats_available, 1);
        assert_eq!(decoded.price_per_seat, 0.0);
        assert_eq!(decoded.status, OfferStatus::Full);
    }

    #[test]
    fn ride_offer_serde_all_statuses() {
        for status in [
            OfferStatus::Open,
            OfferStatus::Full,
            OfferStatus::InProgress,
            OfferStatus::Completed,
            OfferStatus::Cancelled,
        ] {
            let offer = RideOffer {
                vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
                route_hash: None,
                driver: AgentPubKey::from_raw_36(vec![0xab; 36]),
                departure_time: 1700000000,
                seats_available: 2,
                price_per_seat: 3.0,
                status: status.clone(),
            };
            let json = serde_json::to_string(&offer).unwrap();
            let decoded: RideOffer = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, status);
        }
    }

    // ========================================================================
    // RideRequest struct serde roundtrip
    // ========================================================================

    #[test]
    fn ride_request_serde_roundtrip() {
        let request = RideRequest {
            requester: AgentPubKey::from_raw_36(vec![0xab; 36]),
            origin_lat: 32.95,
            origin_lon: -96.73,
            destination_lat: 32.78,
            destination_lon: -96.80,
            requested_time: 1700000000,
            passengers: 2,
            status: RequestStatus::Open,
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: RideRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.origin_lat, 32.95);
        assert_eq!(decoded.origin_lon, -96.73);
        assert_eq!(decoded.destination_lat, 32.78);
        assert_eq!(decoded.destination_lon, -96.80);
        assert_eq!(decoded.requested_time, 1700000000);
        assert_eq!(decoded.passengers, 2);
        assert_eq!(decoded.status, RequestStatus::Open);
    }

    #[test]
    fn ride_request_serde_all_statuses() {
        for status in [
            RequestStatus::Open,
            RequestStatus::Matched,
            RequestStatus::Cancelled,
        ] {
            let request = RideRequest {
                requester: AgentPubKey::from_raw_36(vec![0xab; 36]),
                origin_lat: 0.0,
                origin_lon: 0.0,
                destination_lat: 1.0,
                destination_lon: 1.0,
                requested_time: 1700000000,
                passengers: 1,
                status: status.clone(),
            };
            let json = serde_json::to_string(&request).unwrap();
            let decoded: RideRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, status);
        }
    }

    #[test]
    fn ride_request_serde_single_passenger() {
        let request = RideRequest {
            requester: AgentPubKey::from_raw_36(vec![0xab; 36]),
            origin_lat: -33.87,
            origin_lon: 151.21,
            destination_lat: -33.85,
            destination_lon: 151.20,
            requested_time: 1705000000,
            passengers: 1,
            status: RequestStatus::Matched,
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: RideRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.passengers, 1);
        assert_eq!(decoded.origin_lat, -33.87);
    }

    // ========================================================================
    // RideMatch struct serde roundtrip
    // ========================================================================

    #[test]
    fn ride_match_serde_roundtrip_pending() {
        let ride_match = RideMatch {
            offer_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            request_hash: ActionHash::from_raw_36(vec![0xcd; 36]),
            confirmed_at: None,
            status: MatchStatus::Pending,
        };
        let json = serde_json::to_string(&ride_match).unwrap();
        let decoded: RideMatch = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.confirmed_at, None);
        assert_eq!(decoded.status, MatchStatus::Pending);
    }

    #[test]
    fn ride_match_serde_roundtrip_confirmed() {
        let ride_match = RideMatch {
            offer_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            request_hash: ActionHash::from_raw_36(vec![0xcd; 36]),
            confirmed_at: Some(1700050000),
            status: MatchStatus::Confirmed,
        };
        let json = serde_json::to_string(&ride_match).unwrap();
        let decoded: RideMatch = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.confirmed_at, Some(1700050000));
        assert_eq!(decoded.status, MatchStatus::Confirmed);
    }

    #[test]
    fn ride_match_serde_all_statuses() {
        for status in [
            MatchStatus::Pending,
            MatchStatus::Confirmed,
            MatchStatus::InProgress,
            MatchStatus::Completed,
            MatchStatus::Cancelled,
        ] {
            let ride_match = RideMatch {
                offer_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
                request_hash: ActionHash::from_raw_36(vec![0xcd; 36]),
                confirmed_at: None,
                status: status.clone(),
            };
            let json = serde_json::to_string(&ride_match).unwrap();
            let decoded: RideMatch = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, status);
        }
    }

    // ========================================================================
    // CargoOffer struct serde roundtrip
    // ========================================================================

    #[test]
    fn cargo_offer_serde_roundtrip() {
        let cargo = CargoOffer {
            vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            origin_lat: 32.95,
            origin_lon: -96.73,
            destination_lat: 33.45,
            destination_lon: -96.50,
            capacity_kg: 500.0,
            price_per_kg: 0.75,
            departure_time: 1700000000,
        };
        let json = serde_json::to_string(&cargo).unwrap();
        let decoded: CargoOffer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.origin_lat, 32.95);
        assert_eq!(decoded.origin_lon, -96.73);
        assert_eq!(decoded.destination_lat, 33.45);
        assert_eq!(decoded.destination_lon, -96.50);
        assert_eq!(decoded.capacity_kg, 500.0);
        assert_eq!(decoded.price_per_kg, 0.75);
        assert_eq!(decoded.departure_time, 1700000000);
    }

    #[test]
    fn cargo_offer_serde_free_delivery() {
        let cargo = CargoOffer {
            vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            origin_lat: 0.0,
            origin_lon: 0.0,
            destination_lat: 1.0,
            destination_lon: 1.0,
            capacity_kg: 100.0,
            price_per_kg: 0.0,
            departure_time: 1705000000,
        };
        let json = serde_json::to_string(&cargo).unwrap();
        let decoded: CargoOffer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.price_per_kg, 0.0);
        assert_eq!(decoded.capacity_kg, 100.0);
    }

    // ── Additional edge-case and boundary tests ─────────────────────────

    #[test]
    fn update_match_status_input_serde_all_statuses() {
        for status in [
            MatchStatus::Pending,
            MatchStatus::Confirmed,
            MatchStatus::InProgress,
            MatchStatus::Completed,
            MatchStatus::Cancelled,
        ] {
            let input = UpdateMatchStatusInput {
                match_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
                new_status: status.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: UpdateMatchStatusInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.new_status, status);
        }
    }

    #[test]
    fn ride_offer_max_seats_roundtrip() {
        let offer = RideOffer {
            vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            route_hash: Some(ActionHash::from_raw_36(vec![0xcd; 36])),
            driver: AgentPubKey::from_raw_36(vec![0xab; 36]),
            departure_time: u64::MAX,
            seats_available: u32::MAX,
            price_per_seat: 0.0,
            status: OfferStatus::Open,
        };
        let json = serde_json::to_string(&offer).unwrap();
        let decoded: RideOffer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.seats_available, u32::MAX);
        assert_eq!(decoded.departure_time, u64::MAX);
    }

    #[test]
    fn ride_request_extreme_coordinates_roundtrip() {
        let request = RideRequest {
            requester: AgentPubKey::from_raw_36(vec![0xab; 36]),
            origin_lat: -90.0,
            origin_lon: -180.0,
            destination_lat: 90.0,
            destination_lon: 180.0,
            requested_time: 0,
            passengers: 1,
            status: RequestStatus::Open,
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: RideRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.origin_lat, -90.0);
        assert_eq!(decoded.origin_lon, -180.0);
        assert_eq!(decoded.destination_lat, 90.0);
        assert_eq!(decoded.destination_lon, 180.0);
        assert_eq!(decoded.requested_time, 0);
    }

    #[test]
    fn ride_request_max_passengers_roundtrip() {
        let request = RideRequest {
            requester: AgentPubKey::from_raw_36(vec![0xab; 36]),
            origin_lat: 0.0,
            origin_lon: 0.0,
            destination_lat: 1.0,
            destination_lon: 1.0,
            requested_time: 1700000000,
            passengers: u32::MAX,
            status: RequestStatus::Cancelled,
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: RideRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.passengers, u32::MAX);
        assert_eq!(decoded.status, RequestStatus::Cancelled);
    }

    #[test]
    fn ride_match_confirmed_at_max_u64_roundtrip() {
        let ride_match = RideMatch {
            offer_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            request_hash: ActionHash::from_raw_36(vec![0xcd; 36]),
            confirmed_at: Some(u64::MAX),
            status: MatchStatus::Completed,
        };
        let json = serde_json::to_string(&ride_match).unwrap();
        let decoded: RideMatch = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.confirmed_at, Some(u64::MAX));
        assert_eq!(decoded.status, MatchStatus::Completed);
    }

    #[test]
    fn cargo_offer_extreme_coordinates_roundtrip() {
        let cargo = CargoOffer {
            vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            origin_lat: -90.0,
            origin_lon: -180.0,
            destination_lat: 90.0,
            destination_lon: 180.0,
            capacity_kg: 0.001,
            price_per_kg: 99999.99,
            departure_time: u64::MAX,
        };
        let json = serde_json::to_string(&cargo).unwrap();
        let decoded: CargoOffer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.origin_lat, -90.0);
        assert_eq!(decoded.destination_lon, 180.0);
        assert!((decoded.capacity_kg - 0.001).abs() < 1e-9);
        assert!((decoded.price_per_kg - 99999.99).abs() < 0.01);
    }

    #[test]
    fn cargo_offer_large_capacity_roundtrip() {
        let cargo = CargoOffer {
            vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            origin_lat: 32.95,
            origin_lon: -96.73,
            destination_lat: 33.45,
            destination_lon: -96.50,
            capacity_kg: 999_999_999.99,
            price_per_kg: 0.0,
            departure_time: 1700000000,
        };
        let json = serde_json::to_string(&cargo).unwrap();
        let decoded: CargoOffer = serde_json::from_str(&json).unwrap();
        assert!((decoded.capacity_kg - 999_999_999.99).abs() < 0.01);
    }

    #[test]
    fn ride_match_same_offer_request_hash_roundtrip() {
        let hash = ActionHash::from_raw_36(vec![0xdb; 36]);
        let ride_match = RideMatch {
            offer_hash: hash.clone(),
            request_hash: hash.clone(),
            confirmed_at: None,
            status: MatchStatus::Pending,
        };
        let json = serde_json::to_string(&ride_match).unwrap();
        let decoded: RideMatch = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.offer_hash, decoded.request_hash);
    }

    // ========================================================================
    // RideReview serde roundtrip
    // ========================================================================

    #[test]
    fn ride_review_serde_roundtrip() {
        let review = RideReview {
            match_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            reviewer: AgentPubKey::from_raw_36(vec![0xab; 36]),
            role: ReviewerRole::Passenger,
            rating: 5,
            comment: "Excellent ride!".to_string(),
            safety_concern: false,
            created_at: 1700000000,
        };
        let json = serde_json::to_string(&review).unwrap();
        let decoded: RideReview = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.rating, 5);
        assert_eq!(decoded.comment, "Excellent ride!");
        assert!(!decoded.safety_concern);
        assert_eq!(decoded.role, ReviewerRole::Passenger);
    }

    #[test]
    fn ride_review_serde_driver_role_with_safety_concern() {
        let review = RideReview {
            match_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            reviewer: AgentPubKey::from_raw_36(vec![0xab; 36]),
            role: ReviewerRole::Driver,
            rating: 2,
            comment: "Passenger was disruptive".to_string(),
            safety_concern: true,
            created_at: 1700000000,
        };
        let json = serde_json::to_string(&review).unwrap();
        let decoded: RideReview = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.role, ReviewerRole::Driver);
        assert!(decoded.safety_concern);
        assert_eq!(decoded.rating, 2);
    }

    #[test]
    fn reviewer_role_all_variants_serde() {
        for role in [ReviewerRole::Driver, ReviewerRole::Passenger] {
            let json = serde_json::to_string(&role).unwrap();
            let decoded: ReviewerRole = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, role);
        }
    }

    // ========================================================================
    // DriverRating serde roundtrip
    // ========================================================================

    #[test]
    fn driver_rating_serde_roundtrip() {
        let rating = DriverRating {
            average_rating: 4.5,
            total_reviews: 10,
        };
        let json = serde_json::to_string(&rating).unwrap();
        let decoded: DriverRating = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.average_rating, 4.5);
        assert_eq!(decoded.total_reviews, 10);
    }

    #[test]
    fn driver_rating_serde_no_reviews() {
        let rating = DriverRating {
            average_rating: 0.0,
            total_reviews: 0,
        };
        let json = serde_json::to_string(&rating).unwrap();
        let decoded: DriverRating = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.average_rating, 0.0);
        assert_eq!(decoded.total_reviews, 0);
    }

    // ========================================================================
    // FindNearbyInput serde roundtrip
    // ========================================================================

    #[test]
    fn find_nearby_input_serde_roundtrip() {
        let input = FindNearbyInput {
            origin_lat: 32.95,
            origin_lon: -96.73,
            radius_km: 10.0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FindNearbyInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.origin_lat, 32.95);
        assert_eq!(decoded.origin_lon, -96.73);
        assert_eq!(decoded.radius_km, 10.0);
    }

    // ========================================================================
    // Haversine pure function tests
    // ========================================================================

    #[test]
    fn haversine_same_point_is_zero() {
        let dist = haversine_km(32.95, -96.73, 32.95, -96.73);
        assert!(dist.abs() < 0.001);
    }

    #[test]
    fn haversine_known_distance() {
        // Richardson TX to Dallas TX ≈ ~20 km
        let dist = haversine_km(32.95, -96.73, 32.78, -96.80);
        assert!(dist > 15.0 && dist < 25.0, "Expected ~20km, got {dist}");
    }
}
