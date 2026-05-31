// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! TEND (Time Exchange) Coordinator Zome
//!
//! Implements Commons Charter Article II, Section 2 - Time Exchange Module
//!
//! Key Features:
//! - Record time exchanges between community members
//! - Track balances (mutual credit: total always sums to zero)
//! - Enforce ±40 TEND balance limits
//! - Service listings and requests marketplace
//!
//! Philosophy: All hours are equal. A doctor's hour = a gardener's hour.
//! This radical equality is the foundation of time banking.

use hdk::prelude::*;

// Re-export integrity types for external use
pub use tend_integrity::*;

// =============================================================================
// INPUT/OUTPUT TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordExchangeInput {
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub service_category: ServiceCategory,
    pub cultural_alias: Option<String>,
    pub dao_did: String,
    pub service_date: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExchangeRecord {
    pub id: String,
    pub provider_did: String,
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub service_category: ServiceCategory,
    pub status: ExchangeStatus,
    pub timestamp: Timestamp,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BalanceInfo {
    pub member_did: String,
    pub dao_did: String,
    pub balance: i32,
    pub can_provide: bool, // Can still provide (balance < +40)
    pub can_receive: bool, // Can still receive (balance > -40)
    pub total_provided: f32,
    pub total_received: f32,
    pub exchange_count: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateListingInput {
    pub dao_did: String,
    pub title: String,
    pub description: String,
    pub category: ServiceCategory,
    pub estimated_hours: Option<f32>,
    pub availability: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateRequestInput {
    pub dao_did: String,
    pub title: String,
    pub description: String,
    pub category: ServiceCategory,
    pub estimated_hours: Option<f32>,
    pub urgency: Urgency,
}

// =============================================================================
// CORE EXCHANGE FUNCTIONS
// =============================================================================

/// Record a time exchange
///
/// Called by the PROVIDER after providing a service.
/// The exchange starts in "Proposed" status until the receiver confirms.
///
/// Effect on balances (after confirmation):
/// - Provider: +hours (credit)
/// - Receiver: -hours (debt)
#[hdk_extern]
pub fn record_exchange(input: RecordExchangeInput) -> ExternResult<ExchangeRecord> {
    if input.receiver_did.is_empty() || input.receiver_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Receiver DID must be 1-256 characters".into()
        )));
    }
    if input.hours <= 0.0 || input.hours > 168.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Hours must be between 0 and 168 (one week)".into()
        )));
    }
    if input.service_description.is_empty() || input.service_description.len() > 1024 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Service description must be 1-1024 characters".into()
        )));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let provider_did = format!("did:mycelix:{}", caller);
    let now = sys_time()?;

    // Validate not exchanging with self
    if provider_did == input.receiver_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot exchange time with yourself".into()
        )));
    }

    // Check provider's balance (can still earn if below +40)
    let provider_balance = get_or_create_balance(provider_did.clone(), input.dao_did.clone())?;
    let new_provider_balance = provider_balance.balance + (input.hours as i32);
    if new_provider_balance > BALANCE_LIMIT {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Exchange would exceed your credit limit of +{}. Current balance: {}",
            BALANCE_LIMIT, provider_balance.balance
        ))));
    }

    // Check receiver's balance (can still receive if above -40)
    let receiver_balance =
        get_or_create_balance(input.receiver_did.clone(), input.dao_did.clone())?;
    let new_receiver_balance = receiver_balance.balance - (input.hours as i32);
    if new_receiver_balance < -BALANCE_LIMIT {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Receiver would exceed debt limit of -{}. Their balance: {}",
            BALANCE_LIMIT, receiver_balance.balance
        ))));
    }

    // Create the exchange
    let exchange_id = format!(
        "tend:{}:{}:{}",
        provider_did,
        input.receiver_did,
        now.as_micros()
    );
    let exchange = TendExchange {
        id: exchange_id.clone(),
        provider_did: provider_did.clone(),
        receiver_did: input.receiver_did.clone(),
        hours: input.hours,
        service_description: input.service_description.clone(),
        service_category: input.service_category.clone(),
        cultural_alias: input.cultural_alias,
        dao_did: input.dao_did.clone(),
        timestamp: now,
        status: ExchangeStatus::Proposed,
        service_date: input.service_date,
    };

    let exchange_hash = create_entry(&EntryTypes::TendExchange(exchange.clone()))?;

    // Create links
    create_link(
        anchor_hash(&format!("provider:{}:{}", input.dao_did, provider_did))?,
        exchange_hash.clone(),
        LinkTypes::ProviderToExchanges,
        (),
    )?;

    create_link(
        anchor_hash(&format!(
            "receiver:{}:{}",
            input.dao_did, input.receiver_did
        ))?,
        exchange_hash.clone(),
        LinkTypes::ReceiverToExchanges,
        (),
    )?;

    create_link(
        anchor_hash(&format!("dao:{}", input.dao_did))?,
        exchange_hash.clone(),
        LinkTypes::DaoToExchanges,
        (),
    )?;

    // Create index link for lookup by exchange ID
    create_link(
        anchor_hash(&format!("exchange:{}", exchange_id))?,
        exchange_hash,
        LinkTypes::ExchangeIdToExchange,
        (),
    )?;

    Ok(ExchangeRecord {
        id: exchange_id,
        provider_did,
        receiver_did: input.receiver_did,
        hours: input.hours,
        service_description: input.service_description,
        service_category: input.service_category,
        status: ExchangeStatus::Proposed,
        timestamp: now,
    })
}

/// Confirm an exchange (called by receiver)
///
/// This finalizes the exchange and updates both balances.
#[hdk_extern]
pub fn confirm_exchange(exchange_id: String) -> ExternResult<ExchangeRecord> {
    if exchange_id.is_empty() || exchange_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Exchange ID must be 1-256 characters".into()
        )));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    // Find the exchange
    let exchange = find_exchange_by_id(&exchange_id)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Exchange not found".into()
    )))?;

    // Verify caller is the receiver
    if exchange.receiver_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the receiver can confirm an exchange".into()
        )));
    }

    // Verify status is Proposed
    if exchange.status != ExchangeStatus::Proposed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Exchange is not in Proposed status".into()
        )));
    }

    // Update balances
    update_balance_after_exchange(
        &exchange.provider_did,
        &exchange.dao_did,
        exchange.hours,
        true, // provider gains
    )?;

    update_balance_after_exchange(
        &exchange.receiver_did,
        &exchange.dao_did,
        exchange.hours,
        false, // receiver pays
    )?;

    // Update exchange status
    let updated_exchange = TendExchange {
        status: ExchangeStatus::Confirmed,
        ..exchange.clone()
    };

    // Find and update the entry
    update_exchange_entry(&exchange_id, &updated_exchange)?;

    Ok(ExchangeRecord {
        id: exchange.id,
        provider_did: exchange.provider_did,
        receiver_did: exchange.receiver_did,
        hours: exchange.hours,
        service_description: exchange.service_description,
        service_category: exchange.service_category,
        status: ExchangeStatus::Confirmed,
        timestamp: exchange.timestamp,
    })
}

/// Dispute an exchange
#[hdk_extern]
pub fn dispute_exchange(exchange_id: String) -> ExternResult<ExchangeRecord> {
    if exchange_id.is_empty() || exchange_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Exchange ID must be 1-256 characters".into()
        )));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    let exchange = find_exchange_by_id(&exchange_id)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Exchange not found".into()
    )))?;

    // Either party can dispute
    if exchange.provider_did != caller_did && exchange.receiver_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only exchange participants can dispute".into()
        )));
    }

    let updated_exchange = TendExchange {
        status: ExchangeStatus::Disputed,
        ..exchange.clone()
    };

    update_exchange_entry(&exchange_id, &updated_exchange)?;

    Ok(ExchangeRecord {
        id: exchange.id,
        provider_did: exchange.provider_did,
        receiver_did: exchange.receiver_did,
        hours: exchange.hours,
        service_description: exchange.service_description,
        service_category: exchange.service_category,
        status: ExchangeStatus::Disputed,
        timestamp: exchange.timestamp,
    })
}

/// Cancel an exchange (only if still Proposed)
#[hdk_extern]
pub fn cancel_exchange(exchange_id: String) -> ExternResult<ExchangeRecord> {
    if exchange_id.is_empty() || exchange_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Exchange ID must be 1-256 characters".into()
        )));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    let exchange = find_exchange_by_id(&exchange_id)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Exchange not found".into()
    )))?;

    // Only provider can cancel a proposed exchange
    if exchange.provider_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the provider can cancel a proposed exchange".into()
        )));
    }

    if exchange.status != ExchangeStatus::Proposed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only cancel exchanges in Proposed status".into()
        )));
    }

    let updated_exchange = TendExchange {
        status: ExchangeStatus::Cancelled,
        ..exchange.clone()
    };

    update_exchange_entry(&exchange_id, &updated_exchange)?;

    Ok(ExchangeRecord {
        id: exchange.id,
        provider_did: exchange.provider_did,
        receiver_did: exchange.receiver_did,
        hours: exchange.hours,
        service_description: exchange.service_description,
        service_category: exchange.service_category,
        status: ExchangeStatus::Cancelled,
        timestamp: exchange.timestamp,
    })
}

// =============================================================================
// BALANCE FUNCTIONS
// =============================================================================

/// Get or create balance for a member in a DAO (internal function)
fn get_or_create_balance(member_did: String, dao_did: String) -> ExternResult<BalanceInfo> {
    // Try to find existing balance
    if let Some(balance) = find_balance(&member_did, &dao_did)? {
        return Ok(balance_to_info(&balance));
    }

    // Create new balance (starts at 0)
    let now = sys_time()?;
    let balance = TendBalance {
        member_did: member_did.clone(),
        dao_did: dao_did.clone(),
        balance: 0,
        total_provided: 0.0,
        total_received: 0.0,
        exchange_count: 0,
        last_activity: now,
    };

    let action_hash = create_entry(&EntryTypes::TendBalance(balance.clone()))?;

    create_link(
        anchor_hash(&format!("balance:{}:{}", dao_did, member_did))?,
        action_hash,
        LinkTypes::MemberToBalance,
        (),
    )?;

    Ok(balance_to_info(&balance))
}

/// Get balance info for a member
#[hdk_extern]
pub fn get_balance(input: GetBalanceInput) -> ExternResult<BalanceInfo> {
    if input.member_did.is_empty() || input.member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    get_or_create_balance(input.member_did, input.dao_did)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetBalanceInput {
    pub member_did: String,
    pub dao_did: String,
}

/// Get all exchanges for a member in a DAO
#[hdk_extern]
pub fn get_my_exchanges(dao_did: String) -> ExternResult<Vec<ExchangeRecord>> {
    if dao_did.is_empty() || dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let member_did = format!("did:mycelix:{}", caller);

    let mut exchanges = Vec::new();

    // Get exchanges where member was provider
    let provider_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("provider:{}:{}", dao_did, member_did))?,
            LinkTypes::ProviderToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    for link in provider_links {
        if let Some(record) = get(
            link.target.into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(exchange) = record
                .entry()
                .to_app_option::<TendExchange>()
                .ok()
                .flatten()
            {
                exchanges.push(exchange_to_record(&exchange));
            }
        }
    }

    // Get exchanges where member was receiver
    let receiver_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("receiver:{}:{}", dao_did, member_did))?,
            LinkTypes::ReceiverToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    for link in receiver_links {
        if let Some(record) = get(
            link.target.into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(exchange) = record
                .entry()
                .to_app_option::<TendExchange>()
                .ok()
                .flatten()
            {
                // Avoid duplicates (shouldn't happen, but safety)
                if !exchanges.iter().any(|e| e.id == exchange.id) {
                    exchanges.push(exchange_to_record(&exchange));
                }
            }
        }
    }

    // Sort by timestamp (newest first)
    exchanges.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    Ok(exchanges)
}

// =============================================================================
// SERVICE MARKETPLACE FUNCTIONS
// =============================================================================

/// Create a service listing (offer to help)
#[hdk_extern]
pub fn create_listing(input: CreateListingInput) -> ExternResult<ServiceListing> {
    if input.title.is_empty() || input.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Title must be 1-256 characters".into()
        )));
    }
    if input.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be under 4096 characters".into()
        )));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    if let Some(hours) = input.estimated_hours {
        if hours <= 0.0 || hours > 168.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Estimated hours must be between 0 and 168".into()
            )));
        }
    }
    if let Some(ref avail) = input.availability {
        if avail.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Availability must be under 256 characters".into()
            )));
        }
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let provider_did = format!("did:mycelix:{}", caller);
    let now = sys_time()?;

    let listing_id = format!("listing:{}:{}", provider_did, now.as_micros());
    let listing = ServiceListing {
        id: listing_id,
        provider_did: provider_did.clone(),
        dao_did: input.dao_did.clone(),
        title: input.title,
        description: input.description,
        category: input.category.clone(),
        estimated_hours: input.estimated_hours,
        availability: input.availability,
        active: true,
        created: now,
    };

    let listing_hash = create_entry(&EntryTypes::ServiceListing(listing.clone()))?;

    // Link to DAO
    create_link(
        anchor_hash(&format!("listings:{}", input.dao_did))?,
        listing_hash.clone(),
        LinkTypes::DaoToListings,
        (),
    )?;

    // Link to provider
    create_link(
        anchor_hash(&format!("my_listings:{}", provider_did))?,
        listing_hash.clone(),
        LinkTypes::ProviderToListings,
        (),
    )?;

    // Link to category
    create_link(
        anchor_hash(&format!("category:{}:{:?}", input.dao_did, input.category))?,
        listing_hash,
        LinkTypes::CategoryToListings,
        (),
    )?;

    Ok(listing)
}

/// Get all active listings in a DAO
#[hdk_extern]
pub fn get_dao_listings(dao_did: String) -> ExternResult<Vec<ServiceListing>> {
    if dao_did.is_empty() || dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("listings:{}", dao_did))?,
            LinkTypes::DaoToListings,
        )?,
        GetStrategy::default(),
    )?;

    let mut listings = Vec::new();
    for link in links {
        if let Some(record) = get(
            link.target.into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(listing) = record
                .entry()
                .to_app_option::<ServiceListing>()
                .ok()
                .flatten()
            {
                if listing.active {
                    listings.push(listing);
                }
            }
        }
    }

    Ok(listings)
}

/// Create a service request (ask for help)
#[hdk_extern]
pub fn create_request(input: CreateRequestInput) -> ExternResult<ServiceRequest> {
    if input.title.is_empty() || input.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Title must be 1-256 characters".into()
        )));
    }
    if input.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be under 4096 characters".into()
        )));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    if let Some(hours) = input.estimated_hours {
        if hours <= 0.0 || hours > 168.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Estimated hours must be between 0 and 168".into()
            )));
        }
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let requester_did = format!("did:mycelix:{}", caller);
    let now = sys_time()?;

    let request_id = format!("request:{}:{}", requester_did, now.as_micros());
    let request = ServiceRequest {
        id: request_id,
        requester_did,
        dao_did: input.dao_did.clone(),
        title: input.title,
        description: input.description,
        category: input.category,
        estimated_hours: input.estimated_hours,
        urgency: input.urgency,
        open: true,
        created: now,
    };

    let request_hash = create_entry(&EntryTypes::ServiceRequest(request.clone()))?;

    // Link to DAO
    create_link(
        anchor_hash(&format!("requests:{}", input.dao_did))?,
        request_hash,
        LinkTypes::DaoToRequests,
        (),
    )?;

    Ok(request)
}

/// Get all open requests in a DAO
#[hdk_extern]
pub fn get_dao_requests(dao_did: String) -> ExternResult<Vec<ServiceRequest>> {
    if dao_did.is_empty() || dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("requests:{}", dao_did))?,
            LinkTypes::DaoToRequests,
        )?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        if let Some(record) = get(
            link.target.into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(request) = record
                .entry()
                .to_app_option::<ServiceRequest>()
                .ok()
                .flatten()
            {
                if request.open {
                    requests.push(request);
                }
            }
        }
    }

    Ok(requests)
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn anchor_hash(anchor: &str) -> ExternResult<EntryHash> {
    hash_entry(&Anchor(anchor.to_string()))
}

fn find_balance(member_did: &str, dao_did: &str) -> ExternResult<Option<TendBalance>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("balance:{}:{}", dao_did, member_did))?,
            LinkTypes::MemberToBalance,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(record) = get(
            link.target.clone().into_action_hash().unwrap(),
            GetOptions::default(),
        )? {
            return Ok(record.entry().to_app_option::<TendBalance>().ok().flatten());
        }
    }

    Ok(None)
}

fn update_balance_after_exchange(
    member_did: &str,
    dao_did: &str,
    hours: f32,
    is_provider: bool,
) -> ExternResult<()> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("balance:{}:{}", dao_did, member_did))?,
            LinkTypes::MemberToBalance,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        let action_hash =
            link.target.clone().into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?;
        if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
            if let Some(mut balance) = record.entry().to_app_option::<TendBalance>().ok().flatten()
            {
                let now = sys_time()?;

                if is_provider {
                    balance.balance += hours as i32;
                    balance.total_provided += hours;
                } else {
                    balance.balance -= hours as i32;
                    balance.total_received += hours;
                }
                balance.exchange_count += 1;
                balance.last_activity = now;

                update_entry(action_hash, &balance)?;
            }
        }
    }

    Ok(())
}

/// Find an exchange by its ID using the ExchangeIdToExchange index
fn find_exchange_by_id(exchange_id: &str) -> ExternResult<Option<TendExchange>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("exchange:{}", exchange_id))?,
            LinkTypes::ExchangeIdToExchange,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                return Ok(record
                    .entry()
                    .to_app_option::<TendExchange>()
                    .ok()
                    .flatten());
            }
        }
    }

    Ok(None)
}

/// Update an exchange entry by finding it via ID index and updating in place
fn update_exchange_entry(exchange_id: &str, exchange: &TendExchange) -> ExternResult<()> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("exchange:{}", exchange_id))?,
            LinkTypes::ExchangeIdToExchange,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            update_entry(action_hash, exchange)?;
            return Ok(());
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(format!(
        "Exchange not found for update: {}",
        exchange_id
    ))))
}

fn balance_to_info(balance: &TendBalance) -> BalanceInfo {
    BalanceInfo {
        member_did: balance.member_did.clone(),
        dao_did: balance.dao_did.clone(),
        balance: balance.balance,
        can_provide: balance.balance < BALANCE_LIMIT,
        can_receive: balance.balance > -BALANCE_LIMIT,
        total_provided: balance.total_provided,
        total_received: balance.total_received,
        exchange_count: balance.exchange_count,
    }
}

fn exchange_to_record(exchange: &TendExchange) -> ExchangeRecord {
    ExchangeRecord {
        id: exchange.id.clone(),
        provider_did: exchange.provider_did.clone(),
        receiver_did: exchange.receiver_did.clone(),
        hours: exchange.hours,
        service_description: exchange.service_description.clone(),
        service_category: exchange.service_category.clone(),
        status: exchange.status.clone(),
        timestamp: exchange.timestamp,
    }
}

// =============================================================================
// EXPORTS FOR OTHER ZOMES
// =============================================================================

/// Get TEND activity for reputation calculation (optional, max 5% weight per Commons Charter)
#[hdk_extern]
pub fn get_tend_reputation_input(input: GetBalanceInput) -> ExternResult<f32> {
    if input.member_did.is_empty() || input.member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    let balance = get_or_create_balance(input.member_did, input.dao_did)?;

    // Normalize based on exchange count (more exchanges = more active)
    // Cap at 50 exchanges for max score
    let activity_score = (balance.exchange_count as f32 / 50.0).min(1.0);

    // Apply max weight of 5%
    Ok(activity_score * 0.05)
}
