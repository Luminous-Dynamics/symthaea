// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Timebank Coordinator Zome
//!
//! This zome provides the coordinator functions for time banking
//! in the Mycelix Mutual Aid hApp. Core principle: 1 hour = 1 hour.

use hdk::prelude::*;
use mutualaid_common::*;
use timebank_integrity::*;

// =============================================================================
// INPUT TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateOfferInput {
    pub title: String,
    pub description: String,
    pub category: ServiceCategory,
    pub qualifications: Vec<String>,
    pub availability: Availability,
    pub location: LocationConstraint,
    pub min_duration_hours: f64,
    pub max_duration_hours: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateRequestInput {
    pub title: String,
    pub description: String,
    pub category: ServiceCategory,
    pub urgency: UrgencyLevel,
    pub needed_by: Option<Timestamp>,
    pub estimated_hours: f64,
    pub location: LocationConstraint,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordExchangeInput {
    pub offer_hash: Option<ActionHash>,
    pub request_hash: Option<ActionHash>,
    pub provider: AgentPubKey,
    pub recipient: AgentPubKey,
    pub hours: f64,
    pub category: ServiceCategory,
    pub description: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RateExchangeInput {
    pub exchange_hash: ActionHash,
    pub score: u8,
    pub comment: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchOffersInput {
    pub category: Option<ServiceCategory>,
    pub query: Option<String>,
    pub limit: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchRequestsInput {
    pub category: Option<ServiceCategory>,
    pub urgency: Option<UrgencyLevel>,
    pub query: Option<String>,
    pub limit: Option<u32>,
}

// =============================================================================
// SERVICE OFFERS
// =============================================================================

/// Create a new service offer
#[hdk_extern]
pub fn create_service_offer(input: CreateOfferInput) -> ExternResult<Record> {
    let provider = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let offer = ServiceOffer {
        id: generate_id("offer"),
        provider: provider.clone(),
        category: input.category.clone(),
        title: input.title,
        description: input.description,
        qualifications: input.qualifications,
        availability: input.availability,
        location: input.location,
        min_duration_hours: input.min_duration_hours,
        max_duration_hours: input.max_duration_hours,
        active: true,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
        updated_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let action_hash = create_entry(EntryTypes::ServiceOffer(offer.clone()))?;

    // Link from agent to offer
    create_link(provider, action_hash.clone(), LinkTypes::AgentToOffers, ())?;

    // Link from category anchor to offer
    let cat_anchor = category_anchor(&input.category)?;
    create_link(
        cat_anchor,
        action_hash.clone(),
        LinkTypes::CategoryToOffers,
        (),
    )?;

    // Link to all offers
    let all_anchor = all_offers_anchor()?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllOffers, ())?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created offer".to_string()
    )))
}

/// Get a service offer by hash
#[hdk_extern]
pub fn get_service_offer(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all offers by an agent
#[hdk_extern]
pub fn get_my_offers(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    get_offers_by_agent(agent)
}

/// Get all offers by a specific agent
#[hdk_extern]
pub fn get_offers_by_agent(agent: AgentPubKey) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToOffers)?,
        GetStrategy::default(),
    )?;

    let mut offers = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                offers.push(record);
            }
        }
    }

    Ok(offers)
}

/// Search offers by category
#[hdk_extern]
pub fn search_offers(input: SearchOffersInput) -> ExternResult<Vec<Record>> {
    let anchor = all_offers_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllOffers)?,
        GetStrategy::default(),
    )?;

    let limit = input.limit.unwrap_or(100) as usize;
    let mut results = Vec::new();

    for link in links {
        if results.len() >= limit {
            break;
        }

        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(offer) = record
                    .entry()
                    .to_app_option::<ServiceOffer>()
                    .ok()
                    .flatten()
                {
                    // Only include active offers
                    if !offer.active {
                        continue;
                    }

                    // Apply filters
                    let mut matches = true;

                    if let Some(ref cat) = input.category {
                        if offer.category != *cat {
                            matches = false;
                        }
                    }

                    if let Some(ref q) = input.query {
                        let query_lower = q.to_lowercase();
                        if !offer.title.to_lowercase().contains(&query_lower)
                            && !offer.description.to_lowercase().contains(&query_lower)
                        {
                            matches = false;
                        }
                    }

                    if matches {
                        results.push(record);
                    }
                }
            }
        }
    }

    Ok(results)
}

/// Deactivate a service offer
#[hdk_extern]
pub fn deactivate_offer(hash: ActionHash) -> ExternResult<Record> {
    let record = get(hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Offer not found".to_string())
    ))?;

    let mut offer: ServiceOffer = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse offer".to_string()
        )))?;

    // Verify ownership
    let agent = agent_info()?.agent_initial_pubkey;
    if offer.provider != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the provider can deactivate their offer".to_string()
        )));
    }

    offer.active = false;
    offer.updated_at = Timestamp::from_micros(sys_time()?.as_micros() as i64);

    let new_hash = update_entry(hash, EntryTypes::ServiceOffer(offer))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated offer".to_string()
    )))
}

// =============================================================================
// SERVICE REQUESTS
// =============================================================================

/// Create a new service request
#[hdk_extern]
pub fn create_service_request(input: CreateRequestInput) -> ExternResult<Record> {
    let requester = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let request = ServiceRequest {
        id: generate_id("request"),
        requester: requester.clone(),
        category: input.category.clone(),
        title: input.title,
        description: input.description,
        urgency: input.urgency,
        needed_by: input.needed_by,
        estimated_hours: input.estimated_hours,
        location: input.location,
        status: RequestStatus::Open,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let action_hash = create_entry(EntryTypes::ServiceRequest(request.clone()))?;

    // Link from agent to request
    create_link(
        requester,
        action_hash.clone(),
        LinkTypes::AgentToRequests,
        (),
    )?;

    // Link from category anchor to request
    let cat_anchor = category_anchor(&input.category)?;
    create_link(
        cat_anchor,
        action_hash.clone(),
        LinkTypes::CategoryToRequests,
        (),
    )?;

    // Link to all requests
    let all_anchor = all_requests_anchor()?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllRequests, ())?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created request".to_string()
    )))
}

/// Get a service request by hash
#[hdk_extern]
pub fn get_service_request(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all requests by the current agent
#[hdk_extern]
pub fn get_my_requests(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    get_requests_by_agent(agent)
}

/// Get all requests by a specific agent
#[hdk_extern]
pub fn get_requests_by_agent(agent: AgentPubKey) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToRequests)?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                requests.push(record);
            }
        }
    }

    Ok(requests)
}

/// Search requests
#[hdk_extern]
pub fn search_requests(input: SearchRequestsInput) -> ExternResult<Vec<Record>> {
    let anchor = all_requests_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllRequests)?,
        GetStrategy::default(),
    )?;

    let limit = input.limit.unwrap_or(100) as usize;
    let mut results = Vec::new();

    for link in links {
        if results.len() >= limit {
            break;
        }

        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(request) = record
                    .entry()
                    .to_app_option::<ServiceRequest>()
                    .ok()
                    .flatten()
                {
                    // Only include open requests
                    if request.status != RequestStatus::Open {
                        continue;
                    }

                    let mut matches = true;

                    if let Some(ref cat) = input.category {
                        if request.category != *cat {
                            matches = false;
                        }
                    }

                    if let Some(ref urg) = input.urgency {
                        if request.urgency != *urg {
                            matches = false;
                        }
                    }

                    if let Some(ref q) = input.query {
                        let query_lower = q.to_lowercase();
                        if !request.title.to_lowercase().contains(&query_lower)
                            && !request.description.to_lowercase().contains(&query_lower)
                        {
                            matches = false;
                        }
                    }

                    if matches {
                        results.push(record);
                    }
                }
            }
        }
    }

    Ok(results)
}

// =============================================================================
// TIME EXCHANGES
// =============================================================================

/// Record a completed time exchange
#[hdk_extern]
pub fn record_exchange(input: RecordExchangeInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let exchange = TimeExchange {
        id: generate_id("exchange"),
        offer_hash: input.offer_hash.clone(),
        request_hash: input.request_hash.clone(),
        provider: input.provider.clone(),
        recipient: input.recipient.clone(),
        hours: input.hours,
        category: input.category,
        description: input.description,
        completed_at: Timestamp::from_micros(now.as_micros() as i64),
        provider_rating: None,
        recipient_rating: None,
        confirmed: false,
    };

    let action_hash = create_entry(EntryTypes::TimeExchange(exchange.clone()))?;

    // Link from both agents to the exchange
    create_link(
        input.provider.clone(),
        action_hash.clone(),
        LinkTypes::AgentToExchanges,
        (),
    )?;
    create_link(
        input.recipient.clone(),
        action_hash.clone(),
        LinkTypes::AgentToExchanges,
        (),
    )?;

    // Link from offer/request if applicable
    if let Some(offer_hash) = input.offer_hash {
        create_link(
            offer_hash,
            action_hash.clone(),
            LinkTypes::OfferToExchange,
            (),
        )?;
    }
    if let Some(request_hash) = input.request_hash {
        create_link(
            request_hash,
            action_hash.clone(),
            LinkTypes::RequestToExchange,
            (),
        )?;
    }

    // Create time credit entry
    let credit = TimeCredit {
        hours: input.hours,
        earner: input.provider.clone(),
        debtor: input.recipient,
        service_category: exchange.category.clone(),
        description: exchange.description.clone(),
        performed_at: exchange.completed_at,
        expires_at: None,
    };

    let credit_hash = create_entry(EntryTypes::TimeCredit(credit))?;
    create_link(input.provider, credit_hash, LinkTypes::AgentToCredits, ())?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created exchange".to_string()
    )))
}

/// Get an exchange by hash
#[hdk_extern]
pub fn get_exchange(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all exchanges for the current agent
#[hdk_extern]
pub fn get_my_exchanges(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    get_exchanges_by_agent(agent)
}

/// Get all exchanges for a specific agent
#[hdk_extern]
pub fn get_exchanges_by_agent(agent: AgentPubKey) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToExchanges)?,
        GetStrategy::default(),
    )?;

    let mut exchanges = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                exchanges.push(record);
            }
        }
    }

    Ok(exchanges)
}

/// Confirm an exchange (both parties must confirm)
#[hdk_extern]
pub fn confirm_exchange(hash: ActionHash) -> ExternResult<Record> {
    let record = get(hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Exchange not found".to_string())
    ))?;

    let mut exchange: TimeExchange = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse exchange".to_string()
        )))?;

    let agent = agent_info()?.agent_initial_pubkey;
    if agent != exchange.provider && agent != exchange.recipient {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only participants can confirm an exchange".to_string()
        )));
    }

    exchange.confirmed = true;

    let new_hash = update_entry(hash, EntryTypes::TimeExchange(exchange))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated exchange".to_string()
    )))
}

/// Rate an exchange
#[hdk_extern]
pub fn rate_exchange(input: RateExchangeInput) -> ExternResult<Record> {
    let record = get(input.exchange_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Exchange not found".to_string())
    ))?;

    let mut exchange: TimeExchange = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse exchange".to_string()
        )))?;

    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let rating = Rating {
        score: input.score,
        comment: input.comment,
        rated_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    if agent == exchange.provider {
        exchange.provider_rating = Some(rating);
    } else if agent == exchange.recipient {
        exchange.recipient_rating = Some(rating);
    } else {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only participants can rate an exchange".to_string()
        )));
    }

    let new_hash = update_entry(input.exchange_hash, EntryTypes::TimeExchange(exchange))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated exchange".to_string()
    )))
}

// =============================================================================
// BALANCE QUERIES
// =============================================================================

/// Get time credit balance for current agent
#[hdk_extern]
pub fn get_my_balance(_: ()) -> ExternResult<f64> {
    let agent = agent_info()?.agent_initial_pubkey;
    get_balance_for_agent(agent)
}

/// Get time credit balance for a specific agent
#[hdk_extern]
pub fn get_balance_for_agent(agent: AgentPubKey) -> ExternResult<f64> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToCredits)?,
        GetStrategy::default(),
    )?;

    let mut balance = 0.0;

    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(credit) = record.entry().to_app_option::<TimeCredit>().ok().flatten() {
                    balance += credit.hours;
                }
            }
        }
    }

    Ok(balance)
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Generate a unique ID
fn generate_id(prefix: &str) -> String {
    let now = sys_time().unwrap_or(Timestamp::from_micros(0));
    let agent = agent_info()
        .map(|info| info.agent_initial_pubkey.to_string())
        .unwrap_or_default();
    format!(
        "{}_{}_{}",
        prefix,
        now.as_micros(),
        &agent[..8.min(agent.len())]
    )
}

/// Simple anchor helper
fn make_anchor(name: &str) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("timebank_anchor:{}", name).into_bytes(),
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

/// Get anchor for a category
fn category_anchor(category: &ServiceCategory) -> ExternResult<EntryHash> {
    make_anchor(&format!("category_{:?}", category))
}

/// Get anchor for all offers
fn all_offers_anchor() -> ExternResult<EntryHash> {
    make_anchor("all_offers")
}

/// Get anchor for all requests
fn all_requests_anchor() -> ExternResult<EntryHash> {
    make_anchor("all_requests")
}
