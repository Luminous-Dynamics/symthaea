// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Timebank Coordinator Zome
//! Business logic for service offers, requests, exchanges, and time credits.

use care_timebank_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};
use mycelix_zome_helpers::records_from_links;

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


// ============================================================================
// SERVICE OFFERS
// ============================================================================

/// Create a new service offer
#[hdk_extern]
pub fn create_service_offer(offer: ServiceOffer) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_service_offer")?;
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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_service_request")?;
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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "complete_exchange")?;
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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "rate_exchange")?;
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![2u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_action_hash_b() -> ActionHash {
        ActionHash::from_raw_36(vec![10u8; 36])
    }

    // ── CompleteExchangeInput serde roundtrip ────────────────────────────

    #[test]
    fn complete_exchange_input_serde_roundtrip() {
        let input = CompleteExchangeInput {
            offer_id: fake_action_hash(),
            request_id: fake_action_hash_b(),
            provider: fake_agent(),
            recipient: fake_agent_b(),
            hours: 3.5,
            category: ServiceCategory::Tutoring,
            notes: "Math tutoring session".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CompleteExchangeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.offer_id, input.offer_id);
        assert_eq!(decoded.request_id, input.request_id);
        assert_eq!(decoded.provider, input.provider);
        assert_eq!(decoded.recipient, input.recipient);
        assert!((decoded.hours - 3.5).abs() < f32::EPSILON);
        assert_eq!(decoded.category, ServiceCategory::Tutoring);
        assert_eq!(decoded.notes, "Math tutoring session");
    }

    #[test]
    fn complete_exchange_input_empty_notes_roundtrip() {
        let input = CompleteExchangeInput {
            offer_id: fake_action_hash(),
            request_id: fake_action_hash_b(),
            provider: fake_agent(),
            recipient: fake_agent_b(),
            hours: 1.0,
            category: ServiceCategory::Cleaning,
            notes: String::new(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CompleteExchangeInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.notes.is_empty());
        assert_eq!(decoded.category, ServiceCategory::Cleaning);
    }

    // ── RateExchangeInput serde roundtrip ────────────────────────────────

    #[test]
    fn rate_exchange_input_serde_roundtrip_provider() {
        let input = RateExchangeInput {
            exchange_hash: fake_action_hash(),
            rating: 5,
            is_provider_rating: true,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RateExchangeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.exchange_hash, input.exchange_hash);
        assert_eq!(decoded.rating, 5);
        assert_eq!(decoded.is_provider_rating, true);
    }

    #[test]
    fn rate_exchange_input_serde_roundtrip_recipient() {
        let input = RateExchangeInput {
            exchange_hash: fake_action_hash(),
            rating: 3,
            is_provider_rating: false,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RateExchangeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.rating, 3);
        assert_eq!(decoded.is_provider_rating, false);
    }

    #[test]
    fn rate_exchange_input_serde_all_valid_ratings() {
        for rating in 1u8..=5 {
            let input = RateExchangeInput {
                exchange_hash: fake_action_hash(),
                rating,
                is_provider_rating: true,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: RateExchangeInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.rating, rating);
        }
    }

    // ── ServiceCategory serde roundtrip (all variants) ──────────────────

    #[test]
    fn service_category_serde_all_variants() {
        let cats = vec![
            ServiceCategory::Childcare,
            ServiceCategory::Eldercare,
            ServiceCategory::PetCare,
            ServiceCategory::Cooking,
            ServiceCategory::Cleaning,
            ServiceCategory::Gardening,
            ServiceCategory::Tutoring,
            ServiceCategory::TechSupport,
            ServiceCategory::Transportation,
            ServiceCategory::Companionship,
            ServiceCategory::HealthSupport,
            ServiceCategory::HomeRepair,
            ServiceCategory::LegalAdvice,
            ServiceCategory::Counseling,
            ServiceCategory::ArtMusic,
            ServiceCategory::LanguageHelp,
            ServiceCategory::Administrative,
            ServiceCategory::Other("Beekeeping".to_string()),
        ];
        for cat in &cats {
            let json = serde_json::to_string(cat).unwrap();
            let decoded: ServiceCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, cat);
        }
    }

    // ── ServiceCategory::anchor_key pure function tests ─────────────────

    #[test]
    fn service_category_anchor_key_known_variants() {
        assert_eq!(ServiceCategory::Childcare.anchor_key(), "childcare");
        assert_eq!(ServiceCategory::Eldercare.anchor_key(), "eldercare");
        assert_eq!(ServiceCategory::PetCare.anchor_key(), "petcare");
        assert_eq!(ServiceCategory::Cooking.anchor_key(), "cooking");
        assert_eq!(ServiceCategory::Cleaning.anchor_key(), "cleaning");
        assert_eq!(ServiceCategory::Gardening.anchor_key(), "gardening");
        assert_eq!(ServiceCategory::Tutoring.anchor_key(), "tutoring");
        assert_eq!(ServiceCategory::TechSupport.anchor_key(), "techsupport");
        assert_eq!(
            ServiceCategory::Transportation.anchor_key(),
            "transportation"
        );
        assert_eq!(ServiceCategory::Companionship.anchor_key(), "companionship");
        assert_eq!(ServiceCategory::HealthSupport.anchor_key(), "healthsupport");
        assert_eq!(ServiceCategory::HomeRepair.anchor_key(), "homerepair");
        assert_eq!(ServiceCategory::LegalAdvice.anchor_key(), "legaladvice");
        assert_eq!(ServiceCategory::Counseling.anchor_key(), "counseling");
        assert_eq!(ServiceCategory::ArtMusic.anchor_key(), "artmusic");
        assert_eq!(ServiceCategory::LanguageHelp.anchor_key(), "languagehelp");
        assert_eq!(
            ServiceCategory::Administrative.anchor_key(),
            "administrative"
        );
    }

    #[test]
    fn service_category_anchor_key_other_lowercases_and_replaces_spaces() {
        let cat = ServiceCategory::Other("Dog Walking Service".to_string());
        assert_eq!(cat.anchor_key(), "other_dog_walking_service");
    }

    #[test]
    fn service_category_anchor_key_other_empty_string() {
        let cat = ServiceCategory::Other(String::new());
        assert_eq!(cat.anchor_key(), "other_");
    }

    // ── UrgencyLevel serde roundtrip (all variants) ─────────────────────

    #[test]
    fn urgency_level_serde_all_variants() {
        let levels = vec![
            UrgencyLevel::Low,
            UrgencyLevel::Medium,
            UrgencyLevel::High,
            UrgencyLevel::Critical,
        ];
        for level in &levels {
            let json = serde_json::to_string(level).unwrap();
            let decoded: UrgencyLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, level);
        }
    }

    // ── ServiceOffer serde roundtrip ────────────────────────────────────

    #[test]
    fn service_offer_serde_roundtrip() {
        let offer = ServiceOffer {
            provider: fake_agent(),
            category: ServiceCategory::Gardening,
            title: "Garden help".to_string(),
            description: "I can help with your garden.".to_string(),
            hours_available: 8.0,
            availability: "Weekends".to_string(),
            location: "North side".to_string(),
            skills_required: vec!["Pruning".to_string()],
            active: true,
            created_at: Timestamp::from_micros(1000),
            updated_at: Timestamp::from_micros(2000),
        };
        let json = serde_json::to_string(&offer).unwrap();
        let decoded: ServiceOffer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, offer);
    }

    // ── ServiceRequest serde roundtrip ──────────────────────────────────

    #[test]
    fn service_request_serde_roundtrip() {
        let request = ServiceRequest {
            requester: fake_agent(),
            category: ServiceCategory::Companionship,
            title: "Need a friend".to_string(),
            description: "Weekly visit for elderly parent.".to_string(),
            hours_needed: 2.0,
            preferred_schedule: "Sunday afternoons".to_string(),
            location: "West end".to_string(),
            urgency: UrgencyLevel::High,
            open: true,
            created_at: Timestamp::from_micros(500),
            updated_at: Timestamp::from_micros(600),
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: ServiceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
    }

    // ── TimeExchange serde roundtrip ────────────────────────────────────

    #[test]
    fn time_exchange_serde_roundtrip() {
        let exchange = TimeExchange {
            offer_id: fake_action_hash(),
            request_id: fake_action_hash_b(),
            provider: fake_agent(),
            recipient: fake_agent_b(),
            hours: 2.5,
            category: ServiceCategory::TechSupport,
            completed_at: Timestamp::from_micros(9000),
            rating_provider: Some(4),
            rating_recipient: Some(5),
            notes: "Fixed the router".to_string(),
        };
        let json = serde_json::to_string(&exchange).unwrap();
        let decoded: TimeExchange = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, exchange);
    }

    #[test]
    fn time_exchange_no_ratings_serde_roundtrip() {
        let exchange = TimeExchange {
            offer_id: fake_action_hash(),
            request_id: fake_action_hash_b(),
            provider: fake_agent(),
            recipient: fake_agent_b(),
            hours: 1.0,
            category: ServiceCategory::Cooking,
            completed_at: Timestamp::from_micros(0),
            rating_provider: None,
            rating_recipient: None,
            notes: String::new(),
        };
        let json = serde_json::to_string(&exchange).unwrap();
        let decoded: TimeExchange = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.rating_provider, None);
        assert_eq!(decoded.rating_recipient, None);
    }

    // ── TimeCredit serde roundtrip ──────────────────────────────────────

    #[test]
    fn time_credit_serde_roundtrip() {
        let credit = TimeCredit {
            agent: fake_agent(),
            balance: 15.5,
            total_earned: 20.0,
            total_spent: 4.5,
            updated_at: Timestamp::from_micros(7000),
        };
        let json = serde_json::to_string(&credit).unwrap();
        let decoded: TimeCredit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, credit);
    }

    #[test]
    fn time_credit_negative_balance_roundtrip() {
        let credit = TimeCredit {
            agent: fake_agent(),
            balance: -3.0,
            total_earned: 2.0,
            total_spent: 5.0,
            updated_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&credit).unwrap();
        let decoded: TimeCredit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.balance, -3.0);
    }

    // ── Additional edge-case and boundary tests ─────────────────────────

    #[test]
    fn complete_exchange_input_zero_hours_roundtrip() {
        let input = CompleteExchangeInput {
            offer_id: fake_action_hash(),
            request_id: fake_action_hash_b(),
            provider: fake_agent(),
            recipient: fake_agent_b(),
            hours: 0.0,
            category: ServiceCategory::Other("Zero-hour test".to_string()),
            notes: "Edge case".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CompleteExchangeInput = serde_json::from_str(&json).unwrap();
        assert!((decoded.hours - 0.0).abs() < f32::EPSILON);
        assert_eq!(
            decoded.category,
            ServiceCategory::Other("Zero-hour test".to_string())
        );
    }

    #[test]
    fn rate_exchange_input_boundary_rating_one() {
        let input = RateExchangeInput {
            exchange_hash: fake_action_hash(),
            rating: 1,
            is_provider_rating: true,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RateExchangeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.rating, 1);
    }

    #[test]
    fn rate_exchange_input_zero_rating_serde_roundtrip() {
        // Zero is not valid for the business logic but serde must still roundtrip
        let input = RateExchangeInput {
            exchange_hash: fake_action_hash(),
            rating: 0,
            is_provider_rating: false,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RateExchangeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.rating, 0);
    }

    #[test]
    fn service_offer_inactive_with_empty_skills_roundtrip() {
        let offer = ServiceOffer {
            provider: fake_agent(),
            category: ServiceCategory::Administrative,
            title: "Filing".to_string(),
            description: "Help with paperwork.".to_string(),
            hours_available: 0.5,
            availability: String::new(),
            location: "Remote".to_string(),
            skills_required: vec![],
            active: false,
            created_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&offer).unwrap();
        let decoded: ServiceOffer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.active, false);
        assert!(decoded.skills_required.is_empty());
        assert!(decoded.availability.is_empty());
    }

    #[test]
    fn service_request_closed_critical_roundtrip() {
        let request = ServiceRequest {
            requester: fake_agent_b(),
            category: ServiceCategory::HealthSupport,
            title: "Urgent care".to_string(),
            description: "Medical appointment transport needed.".to_string(),
            hours_needed: 168.0,
            preferred_schedule: "Immediately".to_string(),
            location: "Hospital district".to_string(),
            urgency: UrgencyLevel::Critical,
            open: false,
            created_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: ServiceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.urgency, UrgencyLevel::Critical);
        assert_eq!(decoded.open, false);
        assert!((decoded.hours_needed - 168.0).abs() < f32::EPSILON);
    }

    #[test]
    fn time_exchange_mixed_rating_one_set_one_none() {
        let exchange = TimeExchange {
            offer_id: fake_action_hash(),
            request_id: fake_action_hash_b(),
            provider: fake_agent(),
            recipient: fake_agent_b(),
            hours: 0.5,
            category: ServiceCategory::PetCare,
            completed_at: Timestamp::from_micros(5000),
            rating_provider: Some(3),
            rating_recipient: None,
            notes: "Dog walking".to_string(),
        };
        let json = serde_json::to_string(&exchange).unwrap();
        let decoded: TimeExchange = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.rating_provider, Some(3));
        assert_eq!(decoded.rating_recipient, None);
    }

    #[test]
    fn time_credit_large_values_roundtrip() {
        let credit = TimeCredit {
            agent: fake_agent_b(),
            balance: 999_999.99,
            total_earned: 1_000_000.0,
            total_spent: 0.01,
            updated_at: Timestamp::from_micros(i64::MAX / 2),
        };
        let json = serde_json::to_string(&credit).unwrap();
        let decoded: TimeCredit = serde_json::from_str(&json).unwrap();
        assert!((decoded.balance - 999_999.99).abs() < 0.001);
        assert!((decoded.total_earned - 1_000_000.0).abs() < 0.001);
    }

    #[test]
    fn service_category_other_with_special_chars_anchor_key() {
        let cat = ServiceCategory::Other("Arts & Crafts!".to_string());
        let key = cat.anchor_key();
        assert_eq!(key, "other_arts_&_crafts!");
    }

    #[test]
    fn complete_exchange_input_same_offer_request_hash_roundtrip() {
        // Both hashes same -- serde should still work (business logic validates separately)
        let hash = fake_action_hash();
        let input = CompleteExchangeInput {
            offer_id: hash.clone(),
            request_id: hash.clone(),
            provider: fake_agent(),
            recipient: fake_agent_b(),
            hours: 1.0,
            category: ServiceCategory::Childcare,
            notes: "Same hash test".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CompleteExchangeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.offer_id, decoded.request_id);
    }
}
