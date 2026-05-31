// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Timebank Coordinator Zome
//! Business logic for service offers, requests, exchanges, and time credits.

use hdk::prelude::*;
use timebank_integrity::*;

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Helper to ensure an anchor entry exists and return its hash
fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

/// Collect records from links
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
// SERVICE OFFERS
// ============================================================================

/// Create a new service offer
#[hdk_extern]
pub fn create_service_offer(offer: ServiceOffer) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::ServiceOffer(offer.clone()))?;

    // Link agent to offer
    let agent_anchor = ensure_anchor(&format!("agent_offers:{}", offer.provider))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToOffer,
        (),
    )?;

    // Link category to offer
    let cat_anchor = ensure_anchor(&format!("cat_offers:{}", offer.category.anchor_key()))?;
    create_link(
        cat_anchor,
        action_hash.clone(),
        LinkTypes::CategoryToOffer,
        (),
    )?;

    // Link to all active offers
    if offer.active {
        let active_anchor = ensure_anchor("all_active_offers")?;
        create_link(
            active_anchor,
            action_hash.clone(),
            LinkTypes::AllActiveOffers,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created offer".into()
    )))
}

/// Get all offers in a given category
#[hdk_extern]
pub fn get_offers_by_category(category: ServiceCategory) -> ExternResult<Vec<Record>> {
    let cat_anchor = anchor_hash(&format!("cat_offers:{}", category.anchor_key()))?;
    let links = get_links(
        LinkQuery::try_new(cat_anchor, LinkTypes::CategoryToOffer)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all active offers
#[hdk_extern]
pub fn get_all_active_offers(_: ()) -> ExternResult<Vec<Record>> {
    let active_anchor = anchor_hash("all_active_offers")?;
    let links = get_links(
        LinkQuery::try_new(active_anchor, LinkTypes::AllActiveOffers)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get offers by a specific agent
#[hdk_extern]
pub fn get_my_offers(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_anchor = anchor_hash(&format!("agent_offers:{}", agent))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToOffer)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Search offers by keyword in title or description
#[hdk_extern]
pub fn search_offers(query: String) -> ExternResult<Vec<Record>> {
    let query_lower = query.to_lowercase();
    let all_offers = get_all_active_offers(())?;

    let mut results = Vec::new();
    for record in all_offers {
        if let Some(offer) = record
            .entry()
            .to_app_option::<ServiceOffer>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if offer.title.to_lowercase().contains(&query_lower)
                || offer.description.to_lowercase().contains(&query_lower)
                || offer
                    .skills_required
                    .iter()
                    .any(|s| s.to_lowercase().contains(&query_lower))
            {
                results.push(record);
            }
        }
    }

    Ok(results)
}

// ============================================================================
// SERVICE REQUESTS
// ============================================================================

/// Create a new service request
#[hdk_extern]
pub fn create_service_request(request: ServiceRequest) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::ServiceRequest(request.clone()))?;

    // Link agent to request
    let agent_anchor = ensure_anchor(&format!("agent_requests:{}", request.requester))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToRequest,
        (),
    )?;

    // Link category to request
    let cat_anchor = ensure_anchor(&format!("cat_requests:{}", request.category.anchor_key()))?;
    create_link(
        cat_anchor,
        action_hash.clone(),
        LinkTypes::CategoryToRequest,
        (),
    )?;

    // Link to all open requests
    if request.open {
        let open_anchor = ensure_anchor("all_open_requests")?;
        create_link(
            open_anchor,
            action_hash.clone(),
            LinkTypes::AllOpenRequests,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created request".into()
    )))
}

/// Get all open requests
#[hdk_extern]
pub fn get_all_open_requests(_: ()) -> ExternResult<Vec<Record>> {
    let open_anchor = anchor_hash("all_open_requests")?;
    let links = get_links(
        LinkQuery::try_new(open_anchor, LinkTypes::AllOpenRequests)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get requests by category
#[hdk_extern]
pub fn get_requests_by_category(category: ServiceCategory) -> ExternResult<Vec<Record>> {
    let cat_anchor = anchor_hash(&format!("cat_requests:{}", category.anchor_key()))?;
    let links = get_links(
        LinkQuery::try_new(cat_anchor, LinkTypes::CategoryToRequest)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get my requests
#[hdk_extern]
pub fn get_my_requests(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_anchor = anchor_hash(&format!("agent_requests:{}", agent))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToRequest)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// TIME EXCHANGES
// ============================================================================

/// Input for completing a service exchange
#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteExchangeInput {
    pub offer_id: ActionHash,
    pub request_id: ActionHash,
    pub provider: AgentPubKey,
    pub recipient: AgentPubKey,
    pub hours: f32,
    pub category: ServiceCategory,
    pub notes: String,
}

/// Complete a service exchange, creating a TimeExchange record and updating credits
#[hdk_extern]
pub fn complete_exchange(input: CompleteExchangeInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let exchange = TimeExchange {
        offer_id: input.offer_id.clone(),
        request_id: input.request_id.clone(),
        provider: input.provider.clone(),
        recipient: input.recipient.clone(),
        hours: input.hours,
        category: input.category,
        completed_at: now,
        rating_provider: None,
        rating_recipient: None,
        notes: input.notes,
    };

    let action_hash = create_entry(&EntryTypes::TimeExchange(exchange.clone()))?;

    // Link provider to exchange
    let provider_anchor = ensure_anchor(&format!("agent_exchanges:{}", input.provider))?;
    create_link(
        provider_anchor,
        action_hash.clone(),
        LinkTypes::AgentToExchange,
        (),
    )?;

    // Link recipient to exchange
    let recipient_anchor = ensure_anchor(&format!("agent_exchanges:{}", input.recipient))?;
    create_link(
        recipient_anchor,
        action_hash.clone(),
        LinkTypes::AgentToExchange,
        (),
    )?;

    // Link offer to exchange
    let offer_anchor = ensure_anchor(&format!("offer_exchanges:{}", input.offer_id))?;
    create_link(
        offer_anchor,
        action_hash.clone(),
        LinkTypes::OfferToExchange,
        (),
    )?;

    // Link request to exchange
    let request_anchor = ensure_anchor(&format!("request_exchanges:{}", input.request_id))?;
    create_link(
        request_anchor,
        action_hash.clone(),
        LinkTypes::RequestToExchange,
        (),
    )?;

    // Update provider credits (earned)
    update_agent_credit(&input.provider, input.hours as f64, true)?;

    // Update recipient credits (spent)
    update_agent_credit(&input.recipient, input.hours as f64, false)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created exchange".into()
    )))
}

/// Input for rating an exchange
#[derive(Serialize, Deserialize, Debug)]
pub struct RateExchangeInput {
    pub exchange_hash: ActionHash,
    pub rating: u8,
    pub is_provider_rating: bool,
}

/// Rate a completed exchange
#[hdk_extern]
pub fn rate_exchange(input: RateExchangeInput) -> ExternResult<Record> {
    if input.rating < 1 || input.rating > 5 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Rating must be between 1 and 5".into()
        )));
    }

    let record = get(input.exchange_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Exchange not found".into())
    ))?;

    let mut exchange: TimeExchange = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid exchange entry".into()
        )))?;

    let caller = agent_info()?.agent_initial_pubkey;

    if input.is_provider_rating {
        if caller != exchange.provider {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only the provider can set the provider rating".into()
            )));
        }
        exchange.rating_provider = Some(input.rating);
    } else {
        if caller != exchange.recipient {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only the recipient can set the recipient rating".into()
            )));
        }
        exchange.rating_recipient = Some(input.rating);
    }

    let updated_hash = update_entry(input.exchange_hash, &EntryTypes::TimeExchange(exchange))?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated exchange".into()
    )))
}

// ============================================================================
// TIME CREDITS
// ============================================================================

/// Get the caller's current time credit balance
#[hdk_extern]
pub fn get_my_balance(_: ()) -> ExternResult<TimeCredit> {
    let agent = agent_info()?.agent_initial_pubkey;
    get_or_create_credit(&agent)
}

/// Get or create a TimeCredit for an agent
fn get_or_create_credit(agent: &AgentPubKey) -> ExternResult<TimeCredit> {
    let agent_anchor = anchor_hash(&format!("agent_credit:{}", agent))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor.clone(), LinkTypes::AgentToCredit)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.into_iter().last() {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            let credit: TimeCredit = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid credit entry".into()
                )))?;
            return Ok(credit);
        }
    }

    // Create initial credit with starter balance of 5 hours
    let now = sys_time()?;
    let credit = TimeCredit {
        agent: agent.clone(),
        balance: 5.0,
        total_earned: 0.0,
        total_spent: 0.0,
        updated_at: now,
    };

    let action_hash = create_entry(&EntryTypes::TimeCredit(credit.clone()))?;
    let anchor = ensure_anchor(&format!("agent_credit:{}", agent))?;
    create_link(anchor, action_hash, LinkTypes::AgentToCredit, ())?;

    Ok(credit)
}

/// Update an agent's credit balance
fn update_agent_credit(agent: &AgentPubKey, hours: f64, is_earning: bool) -> ExternResult<()> {
    let current = get_or_create_credit(agent)?;
    let now = sys_time()?;

    let updated = if is_earning {
        TimeCredit {
            agent: agent.clone(),
            balance: current.balance + hours,
            total_earned: current.total_earned + hours,
            total_spent: current.total_spent,
            updated_at: now,
        }
    } else {
        TimeCredit {
            agent: agent.clone(),
            balance: current.balance - hours,
            total_earned: current.total_earned,
            total_spent: current.total_spent + hours,
            updated_at: now,
        }
    };

    // Find existing credit record to update
    let agent_anchor = anchor_hash(&format!("agent_credit:{}", agent))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor.clone(), LinkTypes::AgentToCredit)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.into_iter().last() {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        let new_hash = update_entry(action_hash, &EntryTypes::TimeCredit(updated))?;
        // Add link to new record
        let anchor = ensure_anchor(&format!("agent_credit:{}", agent))?;
        create_link(anchor, new_hash, LinkTypes::AgentToCredit, ())?;
    } else {
        // Should not happen since get_or_create_credit was called, but handle gracefully
        let action_hash = create_entry(&EntryTypes::TimeCredit(updated))?;
        let anchor = ensure_anchor(&format!("agent_credit:{}", agent))?;
        create_link(anchor, action_hash, LinkTypes::AgentToCredit, ())?;
    }

    Ok(())
}

/// Get all exchanges for the calling agent
#[hdk_extern]
pub fn get_my_exchanges(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_anchor = anchor_hash(&format!("agent_exchanges:{}", agent))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToExchange)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}
