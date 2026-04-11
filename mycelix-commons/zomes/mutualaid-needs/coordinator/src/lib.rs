// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Needs Coordinator Zome
//!
//! This zome provides coordinator functions for needs matching
//! in the Mycelix Mutual Aid hApp.

use hdk::prelude::*;
use mutualaid_common::*;
use mutualaid_needs_integrity::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};
use mycelix_zome_helpers::get_latest_record;


// =============================================================================
// INPUT TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateNeedInput {
    pub title: String,
    pub description: String,
    pub category: NeedCategory,
    pub urgency: UrgencyLevel,
    pub emergency: bool,
    pub quantity: Option<u32>,
    pub location: LocationConstraint,
    pub needed_by: Option<Timestamp>,
    pub reciprocity_offers: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateOfferInput {
    pub title: String,
    pub description: String,
    pub category: NeedCategory,
    pub quantity: Option<u32>,
    pub condition: Option<String>,
    pub location: LocationConstraint,
    pub available_until: Option<Timestamp>,
    pub asking_for: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProposeMatchInput {
    pub need_hash: ActionHash,
    pub offer_hash: ActionHash,
    pub quantity: Option<u32>,
    pub notes: String,
    pub scheduled_handoff: Option<Timestamp>,
    pub handoff_location: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FulfillMatchInput {
    pub match_hash: ActionHash,
    pub quantity_given: Option<u32>,
    pub notes: String,
    pub gratitude_message: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchNeedsInput {
    pub category: Option<NeedCategory>,
    pub urgency: Option<UrgencyLevel>,
    pub emergency_only: bool,
    pub query: Option<String>,
    pub limit: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchOffersInput {
    pub category: Option<NeedCategory>,
    pub query: Option<String>,
    pub limit: Option<u32>,
}

// =============================================================================
// NEEDS
// =============================================================================

/// Create a new need

#[hdk_extern]
pub fn create_need(input: CreateNeedInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_need")?;
    let requester = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let need = Need {
        id: generate_id("need"),
        requester: requester.clone(),
        category: input.category.clone(),
        title: input.title,
        description: input.description,
        urgency: input.urgency,
        emergency: input.emergency,
        quantity: input.quantity,
        location: input.location,
        needed_by: input.needed_by,
        reciprocity_offers: input.reciprocity_offers,
        status: NeedStatus::Open,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let action_hash = create_entry(EntryTypes::Need(need.clone()))?;

    // Link from agent to need
    create_link(requester, action_hash.clone(), LinkTypes::AgentToNeeds, ())?;

    // Link from category anchor to need
    let cat_anchor = category_anchor(&input.category)?;
    create_link(
        cat_anchor,
        action_hash.clone(),
        LinkTypes::CategoryToNeeds,
        (),
    )?;

    // Link to all needs
    let all_anchor = all_needs_anchor()?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllNeeds, ())?;

    // If emergency, add to emergency needs
    if input.emergency {
        let emergency_anchor = emergency_needs_anchor()?;
        create_link(
            emergency_anchor,
            action_hash.clone(),
            LinkTypes::EmergencyNeeds,
            (),
        )?;
    }

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created need".to_string()
    )))
}

/// Get a need by hash
#[hdk_extern]
pub fn get_need(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get my needs
#[hdk_extern]
pub fn get_my_needs(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToNeeds)?,
        GetStrategy::default(),
    )?;

    let mut needs = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(hash)? {
                needs.push(record);
            }
        }
    }

    Ok(needs)
}

/// Search needs
#[hdk_extern]
pub fn search_needs(input: SearchNeedsInput) -> ExternResult<Vec<Record>> {
    let anchor = if input.emergency_only {
        emergency_needs_anchor()?
    } else {
        all_needs_anchor()?
    };

    let link_type = if input.emergency_only {
        LinkTypes::EmergencyNeeds
    } else {
        LinkTypes::AllNeeds
    };

    let links = get_links(
        LinkQuery::try_new(anchor, link_type)?,
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
                if let Some(need) = record.entry().to_app_option::<Need>().ok().flatten() {
                    // Only include open needs
                    if need.status != NeedStatus::Open {
                        continue;
                    }

                    let mut matches = true;

                    if let Some(ref cat) = input.category {
                        if need.category != *cat {
                            matches = false;
                        }
                    }

                    if let Some(ref urg) = input.urgency {
                        if need.urgency != *urg {
                            matches = false;
                        }
                    }

                    if let Some(ref q) = input.query {
                        let query_lower = q.to_lowercase();
                        if !need.title.to_lowercase().contains(&query_lower)
                            && !need.description.to_lowercase().contains(&query_lower)
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

/// Get emergency needs
#[hdk_extern]
pub fn get_emergency_needs(_: ()) -> ExternResult<Vec<Record>> {
    search_needs(SearchNeedsInput {
        category: None,
        urgency: None,
        emergency_only: true,
        query: None,
        limit: Some(50),
    })
}

/// Withdraw a need
#[hdk_extern]
pub fn withdraw_need(hash: ActionHash) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "withdraw_need")?;
    let record = get_latest_record(hash.clone())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Need not found".to_string()
    )))?;

    let mut need: Need = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse need".to_string()
        )))?;

    // Verify ownership
    let agent = agent_info()?.agent_initial_pubkey;
    if need.requester != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the requester can withdraw their need".to_string()
        )));
    }

    need.status = NeedStatus::Withdrawn;

    let new_hash = update_entry(hash, EntryTypes::Need(need))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated need".to_string()
    )))
}

// =============================================================================
// OFFERS
// =============================================================================

/// Create a new offer
#[hdk_extern]
pub fn create_offer(input: CreateOfferInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_offer")?;
    let offerer = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let offer = Offer {
        id: generate_id("offer"),
        offerer: offerer.clone(),
        category: input.category.clone(),
        title: input.title,
        description: input.description,
        quantity: input.quantity,
        condition: input.condition,
        location: input.location,
        available_until: input.available_until,
        asking_for: input.asking_for,
        status: OfferStatus::Available,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let action_hash = create_entry(EntryTypes::Offer(offer.clone()))?;

    // Link from agent to offer
    create_link(offerer, action_hash.clone(), LinkTypes::AgentToOffers, ())?;

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

/// Get an offer by hash
#[hdk_extern]
pub fn get_offer(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get my offers
#[hdk_extern]
pub fn get_my_offers(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
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

/// Search offers
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
                if let Some(offer) = record.entry().to_app_option::<Offer>().ok().flatten() {
                    // Only include available offers
                    if offer.status != OfferStatus::Available {
                        continue;
                    }

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

/// Withdraw an offer
#[hdk_extern]
pub fn withdraw_offer(hash: ActionHash) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "withdraw_offer")?;
    let record = get_latest_record(hash.clone())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Offer not found".to_string()
    )))?;

    let mut offer: Offer = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse offer".to_string()
        )))?;

    // Verify ownership
    let agent = agent_info()?.agent_initial_pubkey;
    if offer.offerer != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the offerer can withdraw their offer".to_string()
        )));
    }

    offer.status = OfferStatus::Withdrawn;

    let new_hash = update_entry(hash, EntryTypes::Offer(offer))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated offer".to_string()
    )))
}

// =============================================================================
// MATCHING
// =============================================================================

/// Propose a match between a need and an offer
#[hdk_extern]
pub fn propose_match(input: ProposeMatchInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "propose_match")?;
    let now = sys_time()?;

    // Get need
    let need_record = get(input.need_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Need not found".to_string())
    ))?;

    let need: Need = need_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse need".to_string()
        )))?;

    // Get offer
    let offer_record = get(input.offer_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Offer not found".to_string())
    ))?;

    let offer: Offer = offer_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse offer".to_string()
        )))?;

    // Verify categories match
    if need.category != offer.category {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Need and offer categories must match".to_string()
        )));
    }

    let m = Match {
        id: generate_id("match"),
        need_hash: input.need_hash.clone(),
        offer_hash: input.offer_hash.clone(),
        requester: need.requester,
        offerer: offer.offerer,
        status: MatchStatus::Proposed,
        quantity: input.quantity,
        notes: input.notes,
        matched_at: Timestamp::from_micros(now.as_micros() as i64),
        scheduled_handoff: input.scheduled_handoff,
        handoff_location: input.handoff_location,
    };

    let action_hash = create_entry(EntryTypes::Match(m))?;

    // Link from need and offer to match
    create_link(
        input.need_hash,
        action_hash.clone(),
        LinkTypes::NeedToMatches,
        (),
    )?;
    create_link(
        input.offer_hash,
        action_hash.clone(),
        LinkTypes::OfferToMatches,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created match".to_string()
    )))
}

/// Accept a match
#[hdk_extern]
pub fn accept_match(match_hash: ActionHash) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "accept_match")?;
    let record = get_latest_record(match_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Match not found".to_string())
    ))?;

    let mut m: Match = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse match".to_string()
        )))?;

    // Verify the current agent is the requester or offerer
    let agent = agent_info()?.agent_initial_pubkey;
    if agent != m.requester && agent != m.offerer {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only participants can accept a match".to_string()
        )));
    }

    m.status = MatchStatus::Accepted;

    let new_hash = update_entry(match_hash, EntryTypes::Match(m))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated match".to_string()
    )))
}

/// Schedule handoff for a match
#[derive(Serialize, Deserialize, Debug)]
pub struct ScheduleHandoffInput {
    pub match_hash: ActionHash,
    pub scheduled_time: Timestamp,
    pub location: String,
}

#[hdk_extern]
pub fn schedule_handoff(input: ScheduleHandoffInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "schedule_handoff")?;
    let match_hash = input.match_hash;
    let scheduled_time = input.scheduled_time;
    let location = input.location;
    let record = get_latest_record(match_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Match not found".to_string())
    ))?;

    let mut m: Match = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse match".to_string()
        )))?;

    // Verify the current agent is a participant
    let agent = agent_info()?.agent_initial_pubkey;
    if agent != m.requester && agent != m.offerer {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only participants can schedule handoff".to_string()
        )));
    }

    m.status = MatchStatus::Scheduled;
    m.scheduled_handoff = Some(scheduled_time);
    m.handoff_location = Some(location);

    let new_hash = update_entry(match_hash, EntryTypes::Match(m))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated match".to_string()
    )))
}

/// Get matches for a need
#[hdk_extern]
pub fn get_need_matches(need_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(need_hash, LinkTypes::NeedToMatches)?,
        GetStrategy::default(),
    )?;

    let mut matches = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(hash)? {
                matches.push(record);
            }
        }
    }

    Ok(matches)
}

/// Get matches for an offer
#[hdk_extern]
pub fn get_offer_matches(offer_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(offer_hash, LinkTypes::OfferToMatches)?,
        GetStrategy::default(),
    )?;

    let mut matches = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(hash)? {
                matches.push(record);
            }
        }
    }

    Ok(matches)
}

// =============================================================================
// FULFILLMENT
// =============================================================================

/// Record fulfillment of a match
#[hdk_extern]
pub fn fulfill_match(input: FulfillMatchInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "fulfill_match")?;
    let now = sys_time()?;

    // Get match
    let match_record = get(input.match_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Match not found".to_string())
    ))?;

    let mut m: Match = match_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse match".to_string()
        )))?;

    let agent = agent_info()?.agent_initial_pubkey;
    let is_requester = agent == m.requester;
    let is_offerer = agent == m.offerer;

    if !is_requester && !is_offerer {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only participants can fulfill a match".to_string()
        )));
    }

    let fulfillment = Fulfillment {
        match_hash: input.match_hash.clone(),
        quantity_given: input.quantity_given,
        notes: input.notes,
        requester_confirmed: is_requester,
        offerer_confirmed: is_offerer,
        fulfilled_at: Timestamp::from_micros(now.as_micros() as i64),
        gratitude_message: input.gratitude_message,
    };

    let fulfillment_hash = create_entry(EntryTypes::Fulfillment(fulfillment))?;

    // Link match to fulfillment
    create_link(
        input.match_hash.clone(),
        fulfillment_hash.clone(),
        LinkTypes::MatchToFulfillment,
        (),
    )?;

    // Update match status
    m.status = MatchStatus::Completed;
    update_entry(input.match_hash.clone(), EntryTypes::Match(m.clone()))?;

    // Update need and offer statuses
    let mut need: Need = get(m.need_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Need not found".to_string()
        )))?
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse need".to_string()
        )))?;

    need.status = NeedStatus::Fulfilled;
    update_entry(m.need_hash, EntryTypes::Need(need))?;

    let mut offer: Offer = get(m.offer_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Offer not found".to_string()
        )))?
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse offer".to_string()
        )))?;

    offer.status = OfferStatus::Completed;
    update_entry(m.offer_hash, EntryTypes::Offer(offer))?;

    get_latest_record(fulfillment_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created fulfillment".to_string()
    )))
}

/// Confirm fulfillment (the other party confirms)
#[hdk_extern]
pub fn confirm_fulfillment(fulfillment_hash: ActionHash) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "confirm_fulfillment")?;
    let record = get_latest_record(fulfillment_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Fulfillment not found".to_string())
    ))?;

    let mut fulfillment: Fulfillment = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse fulfillment".to_string()
        )))?;

    // Get the match to verify the agent
    let match_record = get(fulfillment.match_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Match not found".to_string())),
    )?;

    let m: Match = match_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse match".to_string()
        )))?;

    let agent = agent_info()?.agent_initial_pubkey;

    if agent == m.requester {
        fulfillment.requester_confirmed = true;
    } else if agent == m.offerer {
        fulfillment.offerer_confirmed = true;
    } else {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only participants can confirm fulfillment".to_string()
        )));
    }

    let new_hash = update_entry(fulfillment_hash, EntryTypes::Fulfillment(fulfillment))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated fulfillment".to_string()
    )))
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
        format!("needs_anchor:{}", name).into_bytes(),
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

/// Get anchor for category
fn category_anchor(category: &NeedCategory) -> ExternResult<EntryHash> {
    make_anchor(&format!("category_{:?}", category))
}

/// Get anchor for all needs
fn all_needs_anchor() -> ExternResult<EntryHash> {
    make_anchor("all_needs")
}

/// Get anchor for all offers
fn all_offers_anchor() -> ExternResult<EntryHash> {
    make_anchor("all_offers")
}

/// Get anchor for emergency needs
fn emergency_needs_anchor() -> ExternResult<EntryHash> {
    make_anchor("emergency_needs")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Input struct serde roundtrip tests ─────────────────────────────

    #[test]
    fn create_need_input_serde_roundtrip() {
        let input = CreateNeedInput {
            title: "Winter coat needed".to_string(),
            description: "Size M, any color".to_string(),
            category: NeedCategory::Clothing,
            urgency: UrgencyLevel::High,
            emergency: false,
            quantity: Some(1),
            location: LocationConstraint::Remote,
            needed_by: None,
            reciprocity_offers: vec!["Can help with gardening".to_string()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateNeedInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.title, "Winter coat needed");
        assert_eq!(decoded.category, NeedCategory::Clothing);
        assert_eq!(decoded.urgency, UrgencyLevel::High);
        assert_eq!(decoded.emergency, false);
        assert_eq!(decoded.quantity, Some(1));
        assert_eq!(decoded.reciprocity_offers.len(), 1);
    }

    #[test]
    fn create_need_input_emergency_minimal() {
        let input = CreateNeedInput {
            title: "Emergency shelter".to_string(),
            description: "Family displaced by fire".to_string(),
            category: NeedCategory::Housing,
            urgency: UrgencyLevel::Emergency,
            emergency: true,
            quantity: None,
            location: LocationConstraint::ToBeArranged,
            needed_by: None,
            reciprocity_offers: vec![],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateNeedInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.emergency, true);
        assert_eq!(decoded.quantity, None);
        assert!(decoded.reciprocity_offers.is_empty());
    }

    #[test]
    fn create_offer_input_serde_roundtrip() {
        let input = CreateOfferInput {
            title: "Free winter clothes".to_string(),
            description: "Assorted coats and sweaters".to_string(),
            category: NeedCategory::Clothing,
            quantity: Some(5),
            condition: Some("Gently used".to_string()),
            location: LocationConstraint::FixedLocation("123 Main St".to_string()),
            available_until: None,
            asking_for: vec!["Nothing, happy to help".to_string()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateOfferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.title, "Free winter clothes");
        assert_eq!(decoded.quantity, Some(5));
        assert_eq!(decoded.condition, Some("Gently used".to_string()));
    }

    #[test]
    fn create_offer_input_no_optionals() {
        let input = CreateOfferInput {
            title: "Cooking help".to_string(),
            description: "Can cook meals".to_string(),
            category: NeedCategory::Food,
            quantity: None,
            condition: None,
            location: LocationConstraint::AtProvider,
            available_until: None,
            asking_for: vec![],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateOfferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.quantity, None);
        assert_eq!(decoded.condition, None);
        assert!(decoded.asking_for.is_empty());
    }

    #[test]
    fn propose_match_input_serde_roundtrip() {
        let need_hash = ActionHash::from_raw_36(vec![0xAA; 36]);
        let offer_hash = ActionHash::from_raw_36(vec![0xBB; 36]);
        let input = ProposeMatchInput {
            need_hash: need_hash.clone(),
            offer_hash: offer_hash.clone(),
            quantity: Some(3),
            notes: "Looks like a good match".to_string(),
            scheduled_handoff: None,
            handoff_location: Some("Community center".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ProposeMatchInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.need_hash, need_hash);
        assert_eq!(decoded.offer_hash, offer_hash);
        assert_eq!(decoded.quantity, Some(3));
        assert_eq!(
            decoded.handoff_location,
            Some("Community center".to_string())
        );
    }

    #[test]
    fn fulfill_match_input_serde_roundtrip() {
        let match_hash = ActionHash::from_raw_36(vec![0xCC; 36]);
        let input = FulfillMatchInput {
            match_hash: match_hash.clone(),
            quantity_given: Some(2),
            notes: "All went well".to_string(),
            gratitude_message: Some("Thank you so much!".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FulfillMatchInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.match_hash, match_hash);
        assert_eq!(decoded.quantity_given, Some(2));
        assert_eq!(
            decoded.gratitude_message,
            Some("Thank you so much!".to_string())
        );
    }

    #[test]
    fn fulfill_match_input_no_optionals() {
        let match_hash = ActionHash::from_raw_36(vec![0xCC; 36]);
        let input = FulfillMatchInput {
            match_hash: match_hash.clone(),
            quantity_given: None,
            notes: "Done".to_string(),
            gratitude_message: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FulfillMatchInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.quantity_given, None);
        assert_eq!(decoded.gratitude_message, None);
    }

    #[test]
    fn search_needs_input_serde_roundtrip() {
        let input = SearchNeedsInput {
            category: Some(NeedCategory::Food),
            urgency: Some(UrgencyLevel::Urgent),
            emergency_only: true,
            query: Some("rice".to_string()),
            limit: Some(25),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SearchNeedsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.category, Some(NeedCategory::Food));
        assert_eq!(decoded.urgency, Some(UrgencyLevel::Urgent));
        assert_eq!(decoded.emergency_only, true);
        assert_eq!(decoded.query, Some("rice".to_string()));
        assert_eq!(decoded.limit, Some(25));
    }

    #[test]
    fn search_needs_input_all_none_filters() {
        let input = SearchNeedsInput {
            category: None,
            urgency: None,
            emergency_only: false,
            query: None,
            limit: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SearchNeedsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.category, None);
        assert_eq!(decoded.emergency_only, false);
    }

    #[test]
    fn search_offers_input_serde_roundtrip() {
        let input = SearchOffersInput {
            category: Some(NeedCategory::BabyItems),
            query: Some("stroller".to_string()),
            limit: Some(10),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SearchOffersInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.category, Some(NeedCategory::BabyItems));
        assert_eq!(decoded.query, Some("stroller".to_string()));
        assert_eq!(decoded.limit, Some(10));
    }

    #[test]
    fn schedule_handoff_input_serde_roundtrip() {
        let match_hash = ActionHash::from_raw_36(vec![0xDD; 36]);
        let input = ScheduleHandoffInput {
            match_hash: match_hash.clone(),
            scheduled_time: Timestamp::from_micros(1700000000000000),
            location: "Main library entrance".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ScheduleHandoffInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.match_hash, match_hash);
        assert_eq!(decoded.location, "Main library entrance");
    }

    // ── Integrity enum serde roundtrip tests ──────────────────────────

    #[test]
    fn need_category_all_variants_serde() {
        let variants = vec![
            NeedCategory::Food,
            NeedCategory::Clothing,
            NeedCategory::Housing,
            NeedCategory::Healthcare,
            NeedCategory::Transportation,
            NeedCategory::Furniture,
            NeedCategory::Appliances,
            NeedCategory::Kitchenware,
            NeedCategory::Bedding,
            NeedCategory::BabyItems,
            NeedCategory::ChildrensItems,
            NeedCategory::PetSupplies,
            NeedCategory::SchoolSupplies,
            NeedCategory::Books,
            NeedCategory::Computers,
            NeedCategory::WorkClothes,
            NeedCategory::Tools,
            NeedCategory::Equipment,
            NeedCategory::Hygiene,
            NeedCategory::Medications,
            NeedCategory::Rides,
            NeedCategory::MovingHelp,
            NeedCategory::Childcare,
            NeedCategory::PetSitting,
            NeedCategory::Custom("Emergency Kit".to_string()),
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: NeedCategory = serde_json::from_str(&json).unwrap();
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
    fn need_status_all_variants_serde() {
        let variants = vec![
            NeedStatus::Open,
            NeedStatus::PartiallyMet,
            NeedStatus::Matched,
            NeedStatus::Fulfilled,
            NeedStatus::Withdrawn,
            NeedStatus::Expired,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: NeedStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn offer_status_all_variants_serde() {
        let variants = vec![
            OfferStatus::Available,
            OfferStatus::Reserved,
            OfferStatus::Claimed,
            OfferStatus::Completed,
            OfferStatus::Withdrawn,
            OfferStatus::Expired,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: OfferStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn match_status_all_variants_serde() {
        let variants = vec![
            MatchStatus::Proposed,
            MatchStatus::Accepted,
            MatchStatus::Scheduled,
            MatchStatus::InProgress,
            MatchStatus::Completed,
            MatchStatus::Cancelled,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: MatchStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn location_constraint_all_variants_serde() {
        let variants = vec![
            LocationConstraint::Remote,
            LocationConstraint::FixedLocation("123 Main St".to_string()),
            LocationConstraint::WithinRadius {
                geohash: "9q8yy".to_string(),
                radius_km: 5.0,
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

    // ── Boundary and edge case tests ─────────────────────────────────

    #[test]
    fn search_offers_input_all_none_filters_serde() {
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
    fn propose_match_input_all_none_optionals_serde() {
        let need_hash = ActionHash::from_raw_36(vec![0xAA; 36]);
        let offer_hash = ActionHash::from_raw_36(vec![0xBB; 36]);
        let input = ProposeMatchInput {
            need_hash: need_hash.clone(),
            offer_hash: offer_hash.clone(),
            quantity: None,
            notes: "".to_string(),
            scheduled_handoff: None,
            handoff_location: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ProposeMatchInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.need_hash, need_hash);
        assert_eq!(decoded.quantity, None);
        assert_eq!(decoded.scheduled_handoff, None);
        assert_eq!(decoded.handoff_location, None);
        assert_eq!(decoded.notes, "");
    }

    #[test]
    fn create_need_input_many_reciprocity_offers_serde() {
        let offers: Vec<String> = (0..50)
            .map(|i| format!("Reciprocity offer #{}", i))
            .collect();
        let input = CreateNeedInput {
            title: "Large request".to_string(),
            description: "Needs many things in return".to_string(),
            category: NeedCategory::Equipment,
            urgency: UrgencyLevel::Low,
            emergency: false,
            quantity: Some(u32::MAX),
            location: LocationConstraint::Remote,
            needed_by: Some(Timestamp::from_micros(1800000000000000)),
            reciprocity_offers: offers.clone(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateNeedInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.reciprocity_offers.len(), 50);
        assert_eq!(decoded.quantity, Some(u32::MAX));
    }

    #[test]
    fn create_offer_input_with_available_until_serde() {
        let input = CreateOfferInput {
            title: "Limited time offer".to_string(),
            description: "Available only this weekend".to_string(),
            category: NeedCategory::Furniture,
            quantity: Some(1),
            condition: Some("Must pick up".to_string()),
            location: LocationConstraint::FixedLocation("456 Elm St".to_string()),
            available_until: Some(Timestamp::from_micros(1700500000000000)),
            asking_for: vec!["Help moving".to_string(), "Baked goods".to_string()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateOfferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.available_until,
            Some(Timestamp::from_micros(1700500000000000))
        );
        assert_eq!(decoded.asking_for.len(), 2);
    }

    #[test]
    fn fulfill_match_input_empty_notes_serde() {
        let match_hash = ActionHash::from_raw_36(vec![0xEE; 36]);
        let input = FulfillMatchInput {
            match_hash: match_hash.clone(),
            quantity_given: Some(u32::MAX),
            notes: "".to_string(),
            gratitude_message: Some("".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FulfillMatchInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.match_hash, match_hash);
        assert_eq!(decoded.notes, "");
        assert_eq!(decoded.gratitude_message, Some("".to_string()));
        assert_eq!(decoded.quantity_given, Some(u32::MAX));
    }

    #[test]
    fn schedule_handoff_input_empty_location_serde() {
        let match_hash = ActionHash::from_raw_36(vec![0xFF; 36]);
        let input = ScheduleHandoffInput {
            match_hash: match_hash.clone(),
            scheduled_time: Timestamp::from_micros(0),
            location: "".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ScheduleHandoffInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.match_hash, match_hash);
        assert_eq!(decoded.location, "");
        assert_eq!(decoded.scheduled_time, Timestamp::from_micros(0));
    }
}
