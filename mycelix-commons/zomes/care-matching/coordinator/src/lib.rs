// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Matching Coordinator Zome
//! Business logic for finding, suggesting, accepting, and declining care matches.

use care_matching_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};
use mycelix_zome_helpers::records_from_links;

// holochain_serialized_bytes is a dependency needed by the SerializedBytes derive macro
// on the local ServiceOffer/ServiceRequest structs below.

// ============================================================================
// Local deserialization-only copies of timebank types.
// These are plain serde structs (no HDI macros) so they don't export duplicate
// WASM callbacks. They must remain wire-compatible with the care_timebank_integrity
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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "suggest_match")?;
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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "accept_match")?;
    update_match_status(match_hash, MatchStatus::Accepted)
}

/// Decline a suggested match
#[hdk_extern]
pub fn decline_match(match_hash: ActionHash) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "decline_match")?;
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

// ============================================================================
// Cross-domain: Check mutual-aid resources for a care service
// ============================================================================

/// Wire-compatible copy of mutualaid ResourceType for serialization.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MutualAidResourceType {
    PowerTool,
    HandTool,
    GardenTool,
    CookingEquipment,
    CraftingSupplies,
    Car,
    Truck,
    Bicycle,
    Trailer,
    Boat,
    MeetingRoom,
    Workshop,
    Kitchen,
    GardenPlot,
    StorageSpace,
    ParkingSpot,
    CampingGear,
    SportsEquipment,
    MusicInstrument,
    Photography,
    Projector,
    Custom(String),
}

/// Wire-compatible copy of mutualaid SearchResourcesInput.
#[derive(Serialize, Deserialize, Debug)]
struct LocalSearchResourcesInput {
    pub resource_type: Option<MutualAidResourceType>,
    pub available_only: bool,
    pub query: Option<String>,
    pub limit: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CheckResourcesInput {
    pub category: ServiceCategory,
    pub location_hint: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResourceAvailabilityResult {
    pub category: ServiceCategory,
    pub resources_available: u32,
    pub has_resources: bool,
    pub error: Option<String>,
}

/// Check if mutual-aid physical resources are available to support a care service.
///
/// This is a concrete cross-domain call: care-matching queries
/// mutualaid_resources directly using `call(CallTargetCell::Local, ...)`.
/// For example, an eldercare match might need mobility aids or medical equipment.
#[hdk_extern]
pub fn check_resources_for_care_request(
    input: CheckResourcesInput,
) -> ExternResult<ResourceAvailabilityResult> {
    let resource_type = map_care_to_resource_type(&input.category);

    let search = LocalSearchResourcesInput {
        resource_type,
        available_only: true,
        query: input.location_hint,
        limit: Some(10),
    };

    let response = call(
        CallTargetCell::Local,
        ZomeName::from("mutualaid_resources"),
        FunctionName::from("search_resources"),
        None,
        search,
    );

    match &response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            let records: Vec<Record> = extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })?;
            let count = records.len() as u32;
            Ok(ResourceAvailabilityResult {
                category: input.category,
                resources_available: count,
                has_resources: count > 0,
                error: None,
            })
        }
        Ok(ZomeCallResponse::NetworkError(err)) => Ok(ResourceAvailabilityResult {
            category: input.category,
            resources_available: 0,
            has_resources: false,
            error: Some(format!("Network error: {}", err)),
        }),
        _ => Ok(ResourceAvailabilityResult {
            category: input.category,
            resources_available: 0,
            has_resources: false,
            error: Some("Failed to query mutualaid resources".into()),
        }),
    }
}

/// Map a care service category to a mutualaid resource type for search.
fn map_care_to_resource_type(category: &ServiceCategory) -> Option<MutualAidResourceType> {
    match category {
        ServiceCategory::Transportation => Some(MutualAidResourceType::Car),
        ServiceCategory::Gardening => Some(MutualAidResourceType::GardenTool),
        ServiceCategory::HomeRepair => Some(MutualAidResourceType::HandTool),
        ServiceCategory::Cooking => Some(MutualAidResourceType::CookingEquipment),
        ServiceCategory::ArtMusic => Some(MutualAidResourceType::MusicInstrument),
        // For categories without a direct resource mapping, return None
        // to search all available resources
        _ => None,
    }
}

// ============================================================================
// HELPERS
// ============================================================================

/// Compute overall match score from factors
fn compute_overall_score(factors: &MatchFactors) -> f32 {
    // Weighted average: skill alignment is most important
    let score = factors.skill_alignment * 0.35
        + factors.proximity_score * 0.25
        + factors.schedule_compatibility * 0.20
        + factors.trust_score * 0.20;
    score.clamp(0.0, 1.0)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn ts() -> Timestamp {
        Timestamp::from_micros(0)
    }

    fn make_offer(
        category: ServiceCategory,
        location: &str,
        availability: &str,
        hours: f32,
    ) -> ServiceOffer {
        ServiceOffer {
            provider: agent_a(),
            category,
            title: "Test offer".to_string(),
            description: "Test".to_string(),
            hours_available: hours,
            availability: availability.to_string(),
            location: location.to_string(),
            skills_required: vec![],
            active: true,
            created_at: ts(),
        }
    }

    fn make_request(
        category: ServiceCategory,
        location: &str,
        schedule: &str,
        hours: f32,
    ) -> ServiceRequest {
        ServiceRequest {
            requester: agent_b(),
            category,
            title: "Test request".to_string(),
            description: "Test".to_string(),
            hours_needed: hours,
            preferred_schedule: schedule.to_string(),
            location: location.to_string(),
            urgency: UrgencyLevel::Medium,
            open: true,
            created_at: ts(),
        }
    }

    // ========================================================================
    // map_care_to_resource_type
    // ========================================================================

    #[test]
    fn map_transportation_to_car() {
        assert!(matches!(
            map_care_to_resource_type(&ServiceCategory::Transportation),
            Some(MutualAidResourceType::Car)
        ));
    }

    #[test]
    fn map_gardening_to_garden_tool() {
        assert!(matches!(
            map_care_to_resource_type(&ServiceCategory::Gardening),
            Some(MutualAidResourceType::GardenTool)
        ));
    }

    #[test]
    fn map_home_repair_to_hand_tool() {
        assert!(matches!(
            map_care_to_resource_type(&ServiceCategory::HomeRepair),
            Some(MutualAidResourceType::HandTool)
        ));
    }

    #[test]
    fn map_cooking_to_cooking_equipment() {
        assert!(matches!(
            map_care_to_resource_type(&ServiceCategory::Cooking),
            Some(MutualAidResourceType::CookingEquipment)
        ));
    }

    #[test]
    fn map_art_music_to_music_instrument() {
        assert!(matches!(
            map_care_to_resource_type(&ServiceCategory::ArtMusic),
            Some(MutualAidResourceType::MusicInstrument)
        ));
    }

    #[test]
    fn map_childcare_returns_none() {
        assert!(map_care_to_resource_type(&ServiceCategory::Childcare).is_none());
    }

    #[test]
    fn map_eldercare_returns_none() {
        assert!(map_care_to_resource_type(&ServiceCategory::Eldercare).is_none());
    }

    #[test]
    fn map_tutoring_returns_none() {
        assert!(map_care_to_resource_type(&ServiceCategory::Tutoring).is_none());
    }

    #[test]
    fn map_other_returns_none() {
        assert!(map_care_to_resource_type(&ServiceCategory::Other("custom".into())).is_none());
    }

    // ========================================================================
    // calculate_match_factors — category alignment
    // ========================================================================

    #[test]
    fn exact_category_match_gives_full_skill_alignment() {
        let offer = make_offer(ServiceCategory::Childcare, "Austin", "flexible", 10.0);
        let request = make_request(ServiceCategory::Childcare, "Austin", "flexible", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        // skill_alignment = category_match(1.0) * hours_factor(1.0) = 1.0
        assert!((factors.skill_alignment - 1.0).abs() < 0.001);
    }

    #[test]
    fn different_category_gives_zero_skill_alignment() {
        let offer = make_offer(ServiceCategory::Childcare, "Austin", "flexible", 10.0);
        let request = make_request(ServiceCategory::Cooking, "Austin", "flexible", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        assert!((factors.skill_alignment - 0.0).abs() < 0.001);
    }

    #[test]
    fn insufficient_hours_reduces_skill_alignment() {
        let offer = make_offer(ServiceCategory::Childcare, "Austin", "flexible", 3.0);
        let request = make_request(ServiceCategory::Childcare, "Austin", "flexible", 10.0);
        let factors = calculate_match_factors(&offer, &request);
        // skill_alignment = 1.0 * (3.0/10.0) = 0.3
        assert!((factors.skill_alignment - 0.3).abs() < 0.001);
    }

    #[test]
    fn exact_hours_match_gives_full_hours_factor() {
        let offer = make_offer(ServiceCategory::Tutoring, "Dallas", "flexible", 5.0);
        let request = make_request(ServiceCategory::Tutoring, "Dallas", "flexible", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        assert!((factors.skill_alignment - 1.0).abs() < 0.001);
    }

    #[test]
    fn excess_hours_still_gives_full_factor() {
        let offer = make_offer(ServiceCategory::Tutoring, "Dallas", "flexible", 20.0);
        let request = make_request(ServiceCategory::Tutoring, "Dallas", "flexible", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        assert!((factors.skill_alignment - 1.0).abs() < 0.001);
    }

    // ========================================================================
    // calculate_match_factors — proximity
    // ========================================================================

    #[test]
    fn exact_location_match_gives_full_proximity() {
        let offer = make_offer(ServiceCategory::Childcare, "Austin, TX", "flexible", 10.0);
        let request = make_request(ServiceCategory::Childcare, "Austin, TX", "flexible", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        assert!((factors.proximity_score - 1.0).abs() < 0.001);
    }

    #[test]
    fn case_insensitive_location_match() {
        let offer = make_offer(ServiceCategory::Childcare, "AUSTIN, TX", "flexible", 10.0);
        let request = make_request(ServiceCategory::Childcare, "austin, tx", "flexible", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        assert!((factors.proximity_score - 1.0).abs() < 0.001);
    }

    #[test]
    fn partial_location_match_gives_0_6() {
        let offer = make_offer(
            ServiceCategory::Childcare,
            "Downtown Austin",
            "flexible",
            10.0,
        );
        let request = make_request(ServiceCategory::Childcare, "Austin", "flexible", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        assert!((factors.proximity_score - 0.6).abs() < 0.001);
    }

    #[test]
    fn no_location_match_gives_0_2() {
        let offer = make_offer(ServiceCategory::Childcare, "Dallas", "flexible", 10.0);
        let request = make_request(ServiceCategory::Childcare, "Houston", "flexible", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        assert!((factors.proximity_score - 0.2).abs() < 0.001);
    }

    // ========================================================================
    // calculate_match_factors — schedule compatibility
    // ========================================================================

    #[test]
    fn flexible_offer_gives_0_9_schedule() {
        let offer = make_offer(ServiceCategory::Childcare, "Austin", "flexible", 10.0);
        let request = make_request(ServiceCategory::Childcare, "Austin", "Monday mornings", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        assert!((factors.schedule_compatibility - 0.9).abs() < 0.001);
    }

    #[test]
    fn flexible_request_gives_0_9_schedule() {
        let offer = make_offer(ServiceCategory::Childcare, "Austin", "Weekdays", 10.0);
        let request = make_request(ServiceCategory::Childcare, "Austin", "flexible", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        assert!((factors.schedule_compatibility - 0.9).abs() < 0.001);
    }

    #[test]
    fn overlapping_schedule_gives_0_7() {
        let offer = make_offer(
            ServiceCategory::Childcare,
            "Austin",
            "Monday and Tuesday",
            10.0,
        );
        let request = make_request(ServiceCategory::Childcare, "Austin", "Monday", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        assert!((factors.schedule_compatibility - 0.7).abs() < 0.001);
    }

    #[test]
    fn no_schedule_overlap_gives_0_3() {
        let offer = make_offer(ServiceCategory::Childcare, "Austin", "Weekdays", 10.0);
        let request = make_request(ServiceCategory::Childcare, "Austin", "Weekends", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        assert!((factors.schedule_compatibility - 0.3).abs() < 0.001);
    }

    // ========================================================================
    // calculate_match_factors — trust score
    // ========================================================================

    #[test]
    fn trust_score_always_baseline() {
        let offer = make_offer(ServiceCategory::Childcare, "Austin", "flexible", 10.0);
        let request = make_request(ServiceCategory::Childcare, "Austin", "flexible", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        assert!((factors.trust_score - 0.5).abs() < 0.001);
    }

    // ========================================================================
    // compute_overall_score
    // ========================================================================

    #[test]
    fn perfect_match_score() {
        let factors = MatchFactors {
            skill_alignment: 1.0,
            proximity_score: 1.0,
            schedule_compatibility: 1.0,
            trust_score: 1.0,
        };
        let score = compute_overall_score(&factors);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn zero_match_score() {
        let factors = MatchFactors {
            skill_alignment: 0.0,
            proximity_score: 0.0,
            schedule_compatibility: 0.0,
            trust_score: 0.0,
        };
        let score = compute_overall_score(&factors);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn weighted_score_calculation() {
        // skill=1.0*0.35 + prox=0.0*0.25 + sched=0.0*0.20 + trust=0.0*0.20 = 0.35
        let factors = MatchFactors {
            skill_alignment: 1.0,
            proximity_score: 0.0,
            schedule_compatibility: 0.0,
            trust_score: 0.0,
        };
        let score = compute_overall_score(&factors);
        assert!((score - 0.35).abs() < 0.001);
    }

    #[test]
    fn proximity_only_score() {
        // skill=0 + prox=1.0*0.25 + sched=0 + trust=0 = 0.25
        let factors = MatchFactors {
            skill_alignment: 0.0,
            proximity_score: 1.0,
            schedule_compatibility: 0.0,
            trust_score: 0.0,
        };
        let score = compute_overall_score(&factors);
        assert!((score - 0.25).abs() < 0.001);
    }

    #[test]
    fn schedule_only_score() {
        // skill=0 + prox=0 + sched=1.0*0.20 + trust=0 = 0.20
        let factors = MatchFactors {
            skill_alignment: 0.0,
            proximity_score: 0.0,
            schedule_compatibility: 1.0,
            trust_score: 0.0,
        };
        let score = compute_overall_score(&factors);
        assert!((score - 0.20).abs() < 0.001);
    }

    #[test]
    fn trust_only_score() {
        // skill=0 + prox=0 + sched=0 + trust=1.0*0.20 = 0.20
        let factors = MatchFactors {
            skill_alignment: 0.0,
            proximity_score: 0.0,
            schedule_compatibility: 0.0,
            trust_score: 1.0,
        };
        let score = compute_overall_score(&factors);
        assert!((score - 0.20).abs() < 0.001);
    }

    #[test]
    fn score_clamped_to_max_1() {
        // Even with values > 1.0, clamp to 1.0
        let factors = MatchFactors {
            skill_alignment: 3.0,
            proximity_score: 3.0,
            schedule_compatibility: 3.0,
            trust_score: 3.0,
        };
        let score = compute_overall_score(&factors);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn score_clamped_to_min_0() {
        let factors = MatchFactors {
            skill_alignment: -1.0,
            proximity_score: -1.0,
            schedule_compatibility: -1.0,
            trust_score: -1.0,
        };
        let score = compute_overall_score(&factors);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn realistic_good_match_score() {
        // Exact category, exact location, flexible schedule, baseline trust
        let offer = make_offer(ServiceCategory::Childcare, "Austin", "flexible", 10.0);
        let request = make_request(ServiceCategory::Childcare, "Austin", "flexible", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        let score = compute_overall_score(&factors);
        // skill=1.0*0.35 + prox=1.0*0.25 + sched=0.9*0.20 + trust=0.5*0.20 = 0.35+0.25+0.18+0.10 = 0.88
        assert!((score - 0.88).abs() < 0.01);
    }

    #[test]
    fn realistic_poor_match_score() {
        // Different category, different location, no schedule overlap
        let offer = make_offer(ServiceCategory::Childcare, "Dallas", "Weekdays", 3.0);
        let request = make_request(ServiceCategory::Cooking, "Houston", "Weekends", 10.0);
        let factors = calculate_match_factors(&offer, &request);
        let score = compute_overall_score(&factors);
        // skill=0.0*0.35 + prox=0.2*0.25 + sched=0.3*0.20 + trust=0.5*0.20 = 0+0.05+0.06+0.10 = 0.21
        assert!((score - 0.21).abs() < 0.01);
    }

    #[test]
    fn match_above_threshold() {
        let offer = make_offer(ServiceCategory::Tutoring, "Austin", "flexible", 10.0);
        let request = make_request(ServiceCategory::Tutoring, "Austin", "flexible", 5.0);
        let factors = calculate_match_factors(&offer, &request);
        let score = compute_overall_score(&factors);
        assert!(score >= 0.3, "Good match should be above 0.3 threshold");
    }

    #[test]
    fn match_below_threshold() {
        let offer = make_offer(ServiceCategory::Childcare, "Dallas", "Weekdays", 1.0);
        let request = make_request(ServiceCategory::Cooking, "Houston", "Weekends", 10.0);
        let factors = calculate_match_factors(&offer, &request);
        let score = compute_overall_score(&factors);
        assert!(score < 0.3, "Poor match should be below 0.3 threshold");
    }

    // ========================================================================
    // Serde roundtrip tests
    // ========================================================================

    #[test]
    fn service_category_serde_roundtrip() {
        let categories = vec![
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
            ServiceCategory::Other("Plumbing".into()),
        ];
        for cat in categories {
            let json = serde_json::to_string(&cat).unwrap();
            let back: ServiceCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(cat, back);
        }
    }

    #[test]
    fn urgency_level_serde_roundtrip() {
        let levels = vec![
            UrgencyLevel::Low,
            UrgencyLevel::Medium,
            UrgencyLevel::High,
            UrgencyLevel::Critical,
        ];
        for level in levels {
            let json = serde_json::to_string(&level).unwrap();
            let back: UrgencyLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(level, back);
        }
    }

    #[test]
    fn match_factors_serde_roundtrip() {
        let factors = MatchFactors {
            proximity_score: 0.75,
            skill_alignment: 0.9,
            schedule_compatibility: 0.6,
            trust_score: 0.5,
        };
        let json = serde_json::to_string(&factors).unwrap();
        let back: MatchFactors = serde_json::from_str(&json).unwrap();
        assert_eq!(factors, back);
    }

    #[test]
    fn mutual_aid_resource_type_serde_roundtrip() {
        let types = vec![
            MutualAidResourceType::Car,
            MutualAidResourceType::GardenTool,
            MutualAidResourceType::HandTool,
            MutualAidResourceType::CookingEquipment,
            MutualAidResourceType::MusicInstrument,
            MutualAidResourceType::Custom("Wheelchair".into()),
        ];
        for t in types {
            let json = serde_json::to_string(&t).unwrap();
            let back: MutualAidResourceType = serde_json::from_str(&json).unwrap();
            // Custom comparison since no PartialEq derive
            assert_eq!(json, serde_json::to_string(&back).unwrap());
        }
    }
}
