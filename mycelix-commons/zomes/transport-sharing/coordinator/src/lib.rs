//! Transport Sharing Coordinator Zome
//! Business logic for ride offers, requests, matching, and cargo coordination.

use transport_sharing_integrity::*;
use hdk::prelude::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
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
    let action_hash = create_entry(&EntryTypes::RideOffer(offer.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_offers".to_string())))?;
    create_link(anchor_hash("all_offers")?, action_hash.clone(), LinkTypes::AllOffers, ())?;
    create_link(offer.driver, action_hash.clone(), LinkTypes::DriverToOffer, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created offer".into())))
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
    let action_hash = create_entry(&EntryTypes::RideRequest(request.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_requests".to_string())))?;
    create_link(anchor_hash("all_requests")?, action_hash.clone(), LinkTypes::AllRequests, ())?;
    create_link(request.requester, action_hash.clone(), LinkTypes::RequesterToRequest, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created request".into())))
}

// ============================================================================
// MATCHING
// ============================================================================

#[hdk_extern]
pub fn match_ride(ride_match: RideMatch) -> ExternResult<Record> {
    let _offer = get(ride_match.offer_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Offer not found".into())))?;
    let _request = get(ride_match.request_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Request not found".into())))?;

    let action_hash = create_entry(&EntryTypes::RideMatch(ride_match.clone()))?;
    create_link(ride_match.offer_hash, action_hash.clone(), LinkTypes::OfferToMatch, ())?;
    create_link(ride_match.request_hash, action_hash.clone(), LinkTypes::RequestToMatch, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created match".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateMatchStatusInput {
    pub match_hash: ActionHash,
    pub new_status: MatchStatus,
}

#[hdk_extern]
pub fn confirm_match(input: UpdateMatchStatusInput) -> ExternResult<Record> {
    let now = sys_time()?.as_micros() / 1_000_000;
    let record = get(input.match_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Match not found".into())))?;
    let mut ride_match: RideMatch = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid match entry".into())))?;

    ride_match.status = MatchStatus::Confirmed;
    ride_match.confirmed_at = Some(now as u64);
    let new_hash = update_entry(record.action_address().clone(), &EntryTypes::RideMatch(ride_match))?;
    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find updated match".into())))
}

#[hdk_extern]
pub fn complete_ride(match_hash: ActionHash) -> ExternResult<Record> {
    let record = get(match_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Match not found".into())))?;
    let mut ride_match: RideMatch = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid match entry".into())))?;

    ride_match.status = MatchStatus::Completed;
    let new_hash = update_entry(record.action_address().clone(), &EntryTypes::RideMatch(ride_match))?;
    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find updated match".into())))
}

#[hdk_extern]
pub fn cancel_ride(match_hash: ActionHash) -> ExternResult<Record> {
    let record = get(match_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Match not found".into())))?;
    let mut ride_match: RideMatch = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid match entry".into())))?;

    ride_match.status = MatchStatus::Cancelled;
    let new_hash = update_entry(record.action_address().clone(), &EntryTypes::RideMatch(ride_match))?;
    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find updated match".into())))
}

// ============================================================================
// CARGO
// ============================================================================

#[hdk_extern]
pub fn post_cargo_offer(cargo: CargoOffer) -> ExternResult<Record> {
    let _vehicle = get(cargo.vehicle_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Vehicle not found".into())))?;

    let action_hash = create_entry(&EntryTypes::CargoOffer(cargo))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_offers".to_string())))?;
    create_link(anchor_hash("all_offers")?, action_hash.clone(), LinkTypes::AllOffers, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created cargo offer".into())))
}

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
}
