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
