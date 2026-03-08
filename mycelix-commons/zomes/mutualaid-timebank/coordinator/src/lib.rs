//! Timebank Coordinator Zome
//!
//! This zome provides the coordinator functions for time banking
//! in the Mycelix Mutual Aid hApp. Core principle: 1 hour = 1 hour.

use hdk::prelude::*;
use mutualaid_common::*;
use mutualaid_timebank_integrity::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

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

#[hdk_extern]
pub fn create_service_offer(input: CreateOfferInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "create_service_offer")?;
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
            if let Some(record) = get_latest_record(hash)? {
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
            if let Some(record) = get_latest_record(hash)? {
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
    let _eligibility = require_consciousness(&requirement_for_proposal(), "deactivate_offer")?;
    let record = get_latest_record(hash.clone())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Offer not found".to_string()
    )))?;

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

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated offer".to_string()
    )))
}

// =============================================================================
// SERVICE REQUESTS
// =============================================================================

/// Create a new service request
#[hdk_extern]
pub fn create_service_request(input: CreateRequestInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "create_service_request")?;
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
            if let Some(record) = get_latest_record(hash)? {
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
            if let Some(record) = get_latest_record(hash)? {
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
    let _eligibility = require_consciousness(&requirement_for_basic(), "record_exchange")?;
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
            if let Some(record) = get_latest_record(hash)? {
                exchanges.push(record);
            }
        }
    }

    Ok(exchanges)
}

/// Confirm an exchange (both parties must confirm)
#[hdk_extern]
pub fn confirm_exchange(hash: ActionHash) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "confirm_exchange")?;
    let record = get_latest_record(hash.clone())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Exchange not found".to_string()
    )))?;

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

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated exchange".to_string()
    )))
}

/// Rate an exchange
#[hdk_extern]
pub fn rate_exchange(input: RateExchangeInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "rate_exchange")?;
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

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
            if let Some(record) = get_latest_record(hash)? {
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── Input struct serde roundtrip tests ─────────────────────────────

    #[test]
    fn create_offer_input_serde_roundtrip() {
        let input = CreateOfferInput {
            title: "Math Tutoring".to_string(),
            description: "Can help with algebra and calculus".to_string(),
            category: ServiceCategory::Tutoring,
            qualifications: vec![
                "BS in Mathematics".to_string(),
                "5 years experience".to_string(),
            ],
            availability: Availability::default(),
            location: LocationConstraint::Remote,
            min_duration_hours: 0.5,
            max_duration_hours: Some(3.0),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateOfferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.title, "Math Tutoring");
        assert_eq!(decoded.category, ServiceCategory::Tutoring);
        assert_eq!(decoded.qualifications.len(), 2);
        assert_eq!(decoded.min_duration_hours, 0.5);
        assert_eq!(decoded.max_duration_hours, Some(3.0));
    }

    #[test]
    fn create_offer_input_no_max_duration() {
        let input = CreateOfferInput {
            title: "Dog Walking".to_string(),
            description: "Happy to walk your dog".to_string(),
            category: ServiceCategory::PetCare,
            qualifications: vec![],
            availability: Availability::default(),
            location: LocationConstraint::AtRequester,
            min_duration_hours: 1.0,
            max_duration_hours: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateOfferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.max_duration_hours, None);
        assert!(decoded.qualifications.is_empty());
    }

    #[test]
    fn create_request_input_serde_roundtrip() {
        let input = CreateRequestInput {
            title: "Need plumbing help".to_string(),
            description: "Kitchen sink is leaking".to_string(),
            category: ServiceCategory::HomeRepair,
            urgency: UrgencyLevel::High,
            needed_by: Some(Timestamp::from_micros(1700000000000000)),
            estimated_hours: 2.5,
            location: LocationConstraint::AtRequester,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateRequestInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.title, "Need plumbing help");
        assert_eq!(decoded.category, ServiceCategory::HomeRepair);
        assert_eq!(decoded.urgency, UrgencyLevel::High);
        assert_eq!(decoded.estimated_hours, 2.5);
        assert!(decoded.needed_by.is_some());
    }

    #[test]
    fn create_request_input_no_needed_by() {
        let input = CreateRequestInput {
            title: "Language lessons".to_string(),
            description: "Want to learn Spanish".to_string(),
            category: ServiceCategory::LanguageTeaching,
            urgency: UrgencyLevel::Low,
            needed_by: None,
            estimated_hours: 10.0,
            location: LocationConstraint::Remote,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateRequestInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.needed_by, None);
        assert_eq!(decoded.estimated_hours, 10.0);
    }

    #[test]
    fn record_exchange_input_serde_roundtrip() {
        let provider = AgentPubKey::from_raw_36(vec![0xAA; 36]);
        let recipient = AgentPubKey::from_raw_36(vec![0xBB; 36]);
        let offer_hash = ActionHash::from_raw_36(vec![0xCC; 36]);
        let input = RecordExchangeInput {
            offer_hash: Some(offer_hash.clone()),
            request_hash: None,
            provider: provider.clone(),
            recipient: recipient.clone(),
            hours: 3.0,
            category: ServiceCategory::Gardening,
            description: "Helped with garden layout".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordExchangeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.hours, 3.0);
        assert_eq!(decoded.category, ServiceCategory::Gardening);
        assert_eq!(decoded.offer_hash, Some(offer_hash));
        assert_eq!(decoded.request_hash, None);
    }

    #[test]
    fn record_exchange_input_no_hashes() {
        let provider = AgentPubKey::from_raw_36(vec![0xAA; 36]);
        let recipient = AgentPubKey::from_raw_36(vec![0xBB; 36]);
        let input = RecordExchangeInput {
            offer_hash: None,
            request_hash: None,
            provider: provider.clone(),
            recipient: recipient.clone(),
            hours: 1.5,
            category: ServiceCategory::Cooking,
            description: "Cooked dinner together".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordExchangeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.offer_hash, None);
        assert_eq!(decoded.request_hash, None);
    }

    #[test]
    fn rate_exchange_input_serde_roundtrip() {
        let exchange_hash = ActionHash::from_raw_36(vec![0xDD; 36]);
        let input = RateExchangeInput {
            exchange_hash: exchange_hash.clone(),
            score: 5,
            comment: Some("Excellent service!".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RateExchangeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.exchange_hash, exchange_hash);
        assert_eq!(decoded.score, 5);
        assert_eq!(decoded.comment, Some("Excellent service!".to_string()));
    }

    #[test]
    fn rate_exchange_input_no_comment() {
        let exchange_hash = ActionHash::from_raw_36(vec![0xDD; 36]);
        let input = RateExchangeInput {
            exchange_hash: exchange_hash.clone(),
            score: 3,
            comment: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RateExchangeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.score, 3);
        assert_eq!(decoded.comment, None);
    }

    #[test]
    fn search_offers_input_serde_roundtrip() {
        let input = SearchOffersInput {
            category: Some(ServiceCategory::TechSupport),
            query: Some("computer".to_string()),
            limit: Some(20),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SearchOffersInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.category, Some(ServiceCategory::TechSupport));
        assert_eq!(decoded.query, Some("computer".to_string()));
        assert_eq!(decoded.limit, Some(20));
    }

    #[test]
    fn search_offers_input_all_none() {
        let input = SearchOffersInput {
            category: None,
            query: None,
            limit: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SearchOffersInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.category, None);
        assert_eq!(decoded.query, None);
        assert_eq!(decoded.limit, None);
    }

    #[test]
    fn search_requests_input_serde_roundtrip() {
        let input = SearchRequestsInput {
            category: Some(ServiceCategory::Eldercare),
            urgency: Some(UrgencyLevel::Urgent),
            query: Some("companion".to_string()),
            limit: Some(50),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SearchRequestsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.category, Some(ServiceCategory::Eldercare));
        assert_eq!(decoded.urgency, Some(UrgencyLevel::Urgent));
        assert_eq!(decoded.query, Some("companion".to_string()));
        assert_eq!(decoded.limit, Some(50));
    }

    #[test]
    fn search_requests_input_all_none() {
        let input = SearchRequestsInput {
            category: None,
            urgency: None,
            query: None,
            limit: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SearchRequestsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.category, None);
        assert_eq!(decoded.urgency, None);
    }

    // ── Integrity enum serde roundtrip tests ──────────────────────────

    #[test]
    fn service_category_all_variants_serde() {
        let variants = vec![
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
            ServiceCategory::Custom("Beekeeping".to_string()),
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: ServiceCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn urgency_level_all_variants_serde() {
        let variants = vec![
            UrgencyLevel::Low,
            UrgencyLevel::Medium,
            UrgencyLevel::High,
            UrgencyLevel::Urgent,
            UrgencyLevel::Emergency,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: UrgencyLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn request_status_all_variants_serde() {
        let variants = vec![
            RequestStatus::Open,
            RequestStatus::Matched,
            RequestStatus::InProgress,
            RequestStatus::Completed,
            RequestStatus::Cancelled,
            RequestStatus::Expired,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: RequestStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn rating_serde_roundtrip() {
        let rating = Rating {
            score: 4,
            comment: Some("Great help!".to_string()),
            rated_at: Timestamp::from_micros(1700000000000000),
        };
        let json = serde_json::to_string(&rating).unwrap();
        let decoded: Rating = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.score, 4);
        assert_eq!(decoded.comment, Some("Great help!".to_string()));
    }

    #[test]
    fn rating_no_comment_serde() {
        let rating = Rating {
            score: 5,
            comment: None,
            rated_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&rating).unwrap();
        let decoded: Rating = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.score, 5);
        assert_eq!(decoded.comment, None);
    }

    #[test]
    fn availability_default_serde_roundtrip() {
        let avail = Availability::default();
        let json = serde_json::to_string(&avail).unwrap();
        let decoded: Availability = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.days, vec![1, 2, 3, 4, 5]);
        assert_eq!(decoded.start_minutes, 540);
        assert_eq!(decoded.end_minutes, 1020);
        assert_eq!(decoded.timezone_offset_minutes, 0);
        assert!(decoded.exceptions.is_empty());
        assert_eq!(decoded.notes, None);
    }

    #[test]
    fn availability_with_exceptions_serde() {
        let avail = Availability {
            days: vec![0, 6],
            start_minutes: 600,
            end_minutes: 900,
            timezone_offset_minutes: -360,
            exceptions: vec![DateRange {
                start: Timestamp::from_micros(1700000000000000),
                end: Timestamp::from_micros(1700100000000000),
                reason: Some("Holiday".to_string()),
            }],
            notes: Some("Weekends only".to_string()),
        };
        let json = serde_json::to_string(&avail).unwrap();
        let decoded: Availability = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.days, vec![0, 6]);
        assert_eq!(decoded.timezone_offset_minutes, -360);
        assert_eq!(decoded.exceptions.len(), 1);
        assert_eq!(decoded.notes, Some("Weekends only".to_string()));
    }

    #[test]
    fn location_constraint_all_variants_serde() {
        let variants = vec![
            LocationConstraint::Remote,
            LocationConstraint::FixedLocation("456 Oak Ave".to_string()),
            LocationConstraint::WithinRadius {
                geohash: "9q8yy".to_string(),
                radius_km: 10.0,
            },
            LocationConstraint::AtRequester,
            LocationConstraint::AtProvider,
            LocationConstraint::ToBeArranged,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: LocationConstraint = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }
}
