// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Matching Coordinator Zome
//! Business logic for finding, suggesting, accepting, and declining care matches.

use hdk::prelude::*;
use matching_integrity::*;

// holochain_serialized_bytes is a dependency needed by the SerializedBytes derive macro
// on the local ServiceOffer/ServiceRequest structs below.

// ============================================================================
// Local deserialization-only copies of timebank types.
// These are plain serde structs (no HDI macros) so they don't export duplicate
// WASM callbacks. They must remain wire-compatible with the timebank_integrity
// originals.
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ServiceCategory {
    Childcare,
    Eldercare,
    PetCare,
    Cooking,
    Cleaning,
    Gardening,
    Tutoring,
    TechSupport,
    Transportation,
    Companionship,
    HealthSupport,
    HomeRepair,
    LegalAdvice,
    Counseling,
    ArtMusic,
    LanguageHelp,
    Administrative,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UrgencyLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Serialize, Deserialize, Debug, Clone, SerializedBytes)]
pub struct ServiceOffer {
    pub provider: AgentPubKey,
    pub category: ServiceCategory,
    pub title: String,
    pub description: String,
    pub hours_available: f32,
    pub availability: String,
    pub location: String,
    pub skills_required: Vec<String>,
    pub active: bool,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, SerializedBytes)]
pub struct ServiceRequest {
    pub requester: AgentPubKey,
    pub category: ServiceCategory,
    pub title: String,
    pub description: String,
    pub hours_needed: f32,
    pub preferred_schedule: String,
    pub location: String,
    pub urgency: UrgencyLevel,
    pub open: bool,
    pub created_at: Timestamp,
}

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
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

/// Input for finding matches for a request
#[derive(Serialize, Deserialize, Debug)]
pub struct FindMatchesInput {
    pub request_hash: ActionHash,
    /// Hashes of active ServiceOffer entries to consider for matching.
    /// The caller (UI or bridge zome) is responsible for gathering these
    /// from the timebank coordinator.
    pub offer_hashes: Vec<ActionHash>,
    pub max_results: u32,
}

/// Find potential matches for a service request by scoring the provided offers
#[hdk_extern]
pub fn find_matches_for_request(input: FindMatchesInput) -> ExternResult<Vec<Record>> {
    // Get the request
    let request_record = get(input.request_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Request not found".into())),
    )?;

    let request: ServiceRequest = request_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid request entry".into()
        )))?;

    let now = sys_time()?;
    let mut matches: Vec<(f32, Record)> = Vec::new();

    for offer_hash in input.offer_hashes {
        if let Some(offer_record) = get(offer_hash.clone(), GetOptions::default())? {
            if let Some(offer) = offer_record
                .entry()
                .to_app_option::<ServiceOffer>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                // Skip if same agent
                if offer.provider == request.requester {
                    continue;
                }

                // Skip inactive offers
                if !offer.active {
                    continue;
                }

                // Calculate match factors
                let factors = calculate_match_factors(&offer, &request);
                let score = compute_overall_score(&factors);

                // Only include matches above threshold
                if score >= 0.3 {
                    let care_match = CareMatch {
                        offer_hash: offer_hash.clone(),
                        request_hash: input.request_hash.clone(),
                        provider: offer.provider.clone(),
                        requester: request.requester.clone(),
                        score,
                        factors,
                        status: MatchStatus::Suggested,
                        created_at: now,
                        updated_at: now,
                    };

                    let match_hash = create_entry(&EntryTypes::CareMatch(care_match))?;

                    // Link request to match
                    let req_anchor =
                        ensure_anchor(&format!("request_matches:{}", input.request_hash))?;
                    create_link(
                        req_anchor,
                        match_hash.clone(),
                        LinkTypes::RequestToMatch,
                        (),
                    )?;

                    // Link offer to match
                    let offer_anchor_hash =
                        ensure_anchor(&format!("offer_matches:{}", offer_hash))?;
                    create_link(
                        offer_anchor_hash,
                        match_hash.clone(),
                        LinkTypes::OfferToMatch,
                        (),
                    )?;

                    // Link agents
                    let provider_anchor =
                        ensure_anchor(&format!("provider_matches:{}", offer.provider))?;
                    create_link(
                        provider_anchor,
                        match_hash.clone(),
                        LinkTypes::AgentProviderMatches,
                        (),
                    )?;

                    let requester_anchor =
                        ensure_anchor(&format!("requester_matches:{}", request.requester))?;
                    create_link(
                        requester_anchor,
                        match_hash.clone(),
                        LinkTypes::AgentRequesterMatches,
                        (),
                    )?;

                    if let Some(rec) = get(match_hash, GetOptions::default())? {
                        matches.push((score, rec));
                    }
                }
            }
        }
    }

    // Sort by score descending
    matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Limit results
    let max = input.max_results.min(50) as usize;
    Ok(matches.into_iter().take(max).map(|(_, r)| r).collect())
}

/// Suggest a specific match (manual matching by an organizer or system)
#[hdk_extern]
pub fn suggest_match(care_match: CareMatch) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::CareMatch(care_match.clone()))?;

    let req_anchor = ensure_anchor(&format!("request_matches:{}", care_match.request_hash))?;
    create_link(
        req_anchor,
        action_hash.clone(),
        LinkTypes::RequestToMatch,
        (),
    )?;

    let offer_anchor = ensure_anchor(&format!("offer_matches:{}", care_match.offer_hash))?;
    create_link(
        offer_anchor,
        action_hash.clone(),
        LinkTypes::OfferToMatch,
        (),
    )?;

    let provider_anchor = ensure_anchor(&format!("provider_matches:{}", care_match.provider))?;
    create_link(
        provider_anchor,
        action_hash.clone(),
        LinkTypes::AgentProviderMatches,
        (),
    )?;

    let requester_anchor = ensure_anchor(&format!("requester_matches:{}", care_match.requester))?;
    create_link(
        requester_anchor,
        action_hash.clone(),
        LinkTypes::AgentRequesterMatches,
        (),
    )?;

    let pending_anchor = ensure_anchor("all_pending_matches")?;
    create_link(
        pending_anchor,
        action_hash.clone(),
        LinkTypes::AllPendingMatches,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created match".into()
    )))
}

/// Accept a suggested match
#[hdk_extern]
pub fn accept_match(match_hash: ActionHash) -> ExternResult<Record> {
    update_match_status(match_hash, MatchStatus::Accepted)
}

/// Decline a suggested match
#[hdk_extern]
pub fn decline_match(match_hash: ActionHash) -> ExternResult<Record> {
    update_match_status(match_hash, MatchStatus::Declined)
}

/// Get matches for a specific request
#[hdk_extern]
pub fn get_matches_for_request(request_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let req_anchor = anchor_hash(&format!("request_matches:{}", request_hash))?;
    let links = get_links(
        LinkQuery::try_new(req_anchor, LinkTypes::RequestToMatch)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get matches where the caller is the provider
#[hdk_extern]
pub fn get_my_provider_matches(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let provider_anchor = anchor_hash(&format!("provider_matches:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(provider_anchor, LinkTypes::AgentProviderMatches)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get matches where the caller is the requester
#[hdk_extern]
pub fn get_my_requester_matches(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let requester_anchor = anchor_hash(&format!("requester_matches:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(requester_anchor, LinkTypes::AgentRequesterMatches)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// HELPERS
// ============================================================================

fn update_match_status(match_hash: ActionHash, new_status: MatchStatus) -> ExternResult<Record> {
    let record = get(match_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Match not found".into())))?;

    let mut care_match: CareMatch = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid match entry".into()
        )))?;

    let caller = agent_info()?.agent_initial_pubkey;

    // Only provider or requester can update match status
    if caller != care_match.provider && caller != care_match.requester {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the provider or requester can update match status".into()
        )));
    }

    let now = sys_time()?;
    care_match.status = new_status;
    care_match.updated_at = now;

    let updated_hash = update_entry(match_hash, &EntryTypes::CareMatch(care_match))?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated match".into()
    )))
}

/// Calculate match factors between an offer and a request
fn calculate_match_factors(offer: &ServiceOffer, request: &ServiceRequest) -> MatchFactors {
    // Category alignment: exact match = 1.0, otherwise 0.0
    let skill_alignment = if offer.category == request.category {
        1.0
    } else {
        0.0
    };

    // Location proximity: simple string comparison
    // In production this would use geospatial distance
    let proximity_score = if offer.location.to_lowercase() == request.location.to_lowercase() {
        1.0
    } else if offer
        .location
        .to_lowercase()
        .contains(&request.location.to_lowercase())
        || request
            .location
            .to_lowercase()
            .contains(&offer.location.to_lowercase())
    {
        0.6
    } else {
        0.2
    };

    // Schedule compatibility: simple heuristic based on availability text
    let schedule_compatibility = if offer.availability.to_lowercase() == "flexible"
        || request.preferred_schedule.to_lowercase() == "flexible"
    {
        0.9
    } else if offer
        .availability
        .to_lowercase()
        .contains(&request.preferred_schedule.to_lowercase())
        || request
            .preferred_schedule
            .to_lowercase()
            .contains(&offer.availability.to_lowercase())
    {
        0.7
    } else {
        0.3
    };

    // Hours compatibility: provider has enough hours
    let hours_factor = if offer.hours_available >= request.hours_needed {
        1.0
    } else {
        offer.hours_available / request.hours_needed
    };

    // Trust score: default baseline (would integrate with credentials zome in production)
    let trust_score = 0.5;

    MatchFactors {
        proximity_score,
        skill_alignment: skill_alignment * hours_factor,
        schedule_compatibility,
        trust_score,
    }
}

/// Compute overall match score from factors
fn compute_overall_score(factors: &MatchFactors) -> f32 {
    // Weighted average: skill alignment is most important
    let score = factors.skill_alignment * 0.35
        + factors.proximity_score * 0.25
        + factors.schedule_compatibility * 0.20
        + factors.trust_score * 0.20;
    score.clamp(0.0, 1.0)
}
