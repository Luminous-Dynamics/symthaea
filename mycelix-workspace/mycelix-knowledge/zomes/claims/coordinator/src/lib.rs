// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Claims Coordinator Zome
//! Business logic for managing epistemic claims
//!
//! Updated to use HDK 0.6 patterns

use claims_integrity::*;
use hdk::prelude::*;

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Result of cascade update
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CascadeResult {
    pub updated_claims: Vec<String>,
    pub trigger_claim_id: String,
    pub depth_reached: u32,
    pub circular_detected: Vec<String>,
    pub processed_at: Timestamp,
    pub processing_time_ms: u64,
}

/// Submit a new claim to the knowledge graph
#[hdk_extern]
pub fn submit_claim(claim: Claim) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Claim(claim.clone()))?;

    // Create and link author anchor
    let author_anchor = format!("author:{}", claim.author);
    create_entry(&EntryTypes::Anchor(Anchor(author_anchor.clone())))?;
    create_link(
        anchor_hash(&author_anchor)?,
        action_hash.clone(),
        LinkTypes::AuthorToClaim,
        (),
    )?;

    // Create and link tag anchors
    for tag in &claim.tags {
        let tag_anchor = format!("tag:{}", tag);
        create_entry(&EntryTypes::Anchor(Anchor(tag_anchor.clone())))?;
        create_link(
            anchor_hash(&tag_anchor)?,
            action_hash.clone(),
            LinkTypes::TagToClaim,
            (),
        )?;
    }

    // Create and link type anchor
    let type_str = format!("type:{:?}", claim.claim_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_str.clone())))?;
    create_link(
        anchor_hash(&type_str)?,
        action_hash.clone(),
        LinkTypes::TypeToClaim,
        (),
    )?;

    // Index by claim ID for O(1) lookup
    let id_anchor = format!("claim_id:{}", claim.id);
    create_entry(&EntryTypes::Anchor(Anchor(id_anchor.clone())))?;
    create_link(
        anchor_hash(&id_anchor)?,
        action_hash.clone(),
        LinkTypes::ClaimIdToClaim,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created claim".into()
    )))
}

/// Get a claim by ID (O(1) via anchor index, works across agents)
#[hdk_extern]
pub fn get_claim(claim_id: String) -> ExternResult<Option<Record>> {
    // O(1) lookup via claim ID anchor index
    let id_anchor = format!("claim_id:{}", claim_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&id_anchor)?, LinkTypes::ClaimIdToClaim)?,
        GetStrategy::default(),
    )?;

    // Return the most recent link target (handles updates)
    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

/// Get claims by author
#[hdk_extern]
pub fn get_claims_by_author(author_did: String) -> ExternResult<Vec<Record>> {
    let author_anchor = format!("author:{}", author_did);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&author_anchor)?, LinkTypes::AuthorToClaim)?,
        GetStrategy::default(),
    )?;

    let mut claims = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            claims.push(record);
        }
    }

    Ok(claims)
}

/// Get claims by tag
#[hdk_extern]
pub fn get_claims_by_tag(tag: String) -> ExternResult<Vec<Record>> {
    let tag_anchor = format!("tag:{}", tag);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&tag_anchor)?, LinkTypes::TagToClaim)?,
        GetStrategy::default(),
    )?;

    let mut claims = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            claims.push(record);
        }
    }

    Ok(claims)
}

/// Pagination input for querying claims
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginatedQuery {
    /// The query key (author DID, tag name, claim ID, etc.)
    pub key: String,
    /// Maximum number of results to return (default: 50)
    pub limit: Option<u32>,
    /// Number of results to skip (default: 0)
    pub offset: Option<u32>,
}

/// Paginated result wrapper
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginatedResult {
    pub records: Vec<Record>,
    pub total: u32,
    pub offset: u32,
    pub limit: u32,
}

/// Helper: paginate a link list and resolve records
fn paginate_links(links: Vec<Link>, limit: u32, offset: u32) -> ExternResult<PaginatedResult> {
    let total = links.len() as u32;
    let start = (offset as usize).min(links.len());
    let end = (start + limit as usize).min(links.len());

    let mut records = Vec::new();
    for link in &links[start..end] {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(PaginatedResult {
        records,
        total,
        offset,
        limit,
    })
}

/// Get claims by author (paginated)
#[hdk_extern]
pub fn get_claims_by_author_paginated(input: PaginatedQuery) -> ExternResult<PaginatedResult> {
    let author_anchor = format!("author:{}", input.key);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&author_anchor)?, LinkTypes::AuthorToClaim)?,
        GetStrategy::default(),
    )?;
    paginate_links(links, input.limit.unwrap_or(50), input.offset.unwrap_or(0))
}

/// Get claims by tag (paginated)
#[hdk_extern]
pub fn get_claims_by_tag_paginated(input: PaginatedQuery) -> ExternResult<PaginatedResult> {
    let tag_anchor = format!("tag:{}", input.key);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&tag_anchor)?, LinkTypes::TagToClaim)?,
        GetStrategy::default(),
    )?;
    paginate_links(links, input.limit.unwrap_or(50), input.offset.unwrap_or(0))
}

/// Get evidence for a claim (paginated)
#[hdk_extern]
pub fn get_claim_evidence_paginated(input: PaginatedQuery) -> ExternResult<PaginatedResult> {
    let claim_anchor = format!("claim:{}", input.key);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&claim_anchor)?, LinkTypes::ClaimToEvidence)?,
        GetStrategy::default(),
    )?;
    paginate_links(links, input.limit.unwrap_or(50), input.offset.unwrap_or(0))
}

/// Get challenges for a claim (paginated)
#[hdk_extern]
pub fn get_claim_challenges_paginated(input: PaginatedQuery) -> ExternResult<PaginatedResult> {
    let claim_anchor = format!("claim:{}", input.key);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&claim_anchor)?, LinkTypes::ClaimToChallenge)?,
        GetStrategy::default(),
    )?;
    paginate_links(links, input.limit.unwrap_or(50), input.offset.unwrap_or(0))
}

/// Update a claim
#[hdk_extern]
pub fn update_claim(input: UpdateClaimInput) -> ExternResult<Record> {
    let current_record = get_claim(input.claim_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Claim not found".into())))?;

    let current_claim: Claim = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid claim entry".into()
        )))?;

    let now = sys_time()?;

    let updated_claim = Claim {
        id: current_claim.id,
        content: input.content.unwrap_or(current_claim.content),
        classification: input.classification.unwrap_or(current_claim.classification),
        author: current_claim.author,
        sources: input.sources.unwrap_or(current_claim.sources),
        tags: input.tags.unwrap_or(current_claim.tags),
        claim_type: current_claim.claim_type,
        confidence: input.confidence.unwrap_or(current_claim.confidence),
        expires: input.expires.or(current_claim.expires),
        created: current_claim.created,
        updated: now,
        version: current_claim.version + 1,
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::Claim(updated_claim),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated claim".into()
    )))
}

/// Input for updating a claim
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateClaimInput {
    pub claim_id: String,
    pub content: Option<String>,
    pub classification: Option<EpistemicPosition>,
    pub sources: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub confidence: Option<f64>,
    pub expires: Option<Timestamp>,
}

/// Add evidence to a claim
#[hdk_extern]
pub fn add_evidence(evidence: Evidence) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Evidence(evidence.clone()))?;

    // Create and link claim anchor to evidence
    let claim_anchor = format!("claim:{}", evidence.claim_id);
    create_entry(&EntryTypes::Anchor(Anchor(claim_anchor.clone())))?;
    create_link(
        anchor_hash(&claim_anchor)?,
        action_hash.clone(),
        LinkTypes::ClaimToEvidence,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find evidence".into()
    )))
}

/// Get evidence for a claim
#[hdk_extern]
pub fn get_claim_evidence(claim_id: String) -> ExternResult<Vec<Record>> {
    let claim_anchor = format!("claim:{}", claim_id);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&claim_anchor)?, LinkTypes::ClaimToEvidence)?,
        GetStrategy::default(),
    )?;

    let mut evidence = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            evidence.push(record);
        }
    }

    Ok(evidence)
}

/// Challenge a claim
#[hdk_extern]
pub fn challenge_claim(input: ChallengeClaimInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let challenge_id = format!("challenge:{}:{}", input.claim_id, now.as_micros());

    let challenge = ClaimChallenge {
        id: challenge_id,
        claim_id: input.claim_id.clone(),
        challenger: input.challenger_did,
        reason: input.reason,
        counter_evidence: input.counter_evidence,
        status: ChallengeStatus::Pending,
        created: now,
    };

    let action_hash = create_entry(&EntryTypes::ClaimChallenge(challenge))?;

    // Create and link claim anchor to challenge
    let claim_anchor = format!("claim:{}", input.claim_id);
    create_entry(&EntryTypes::Anchor(Anchor(claim_anchor.clone())))?;
    create_link(
        anchor_hash(&claim_anchor)?,
        action_hash.clone(),
        LinkTypes::ClaimToChallenge,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find challenge".into()
    )))
}

/// Input for challenging a claim
#[derive(Serialize, Deserialize, Debug)]
pub struct ChallengeClaimInput {
    pub claim_id: String,
    pub challenger_did: String,
    pub reason: String,
    pub counter_evidence: Vec<String>,
}

/// Get challenges for a claim
#[hdk_extern]
pub fn get_claim_challenges(claim_id: String) -> ExternResult<Vec<Record>> {
    let claim_anchor = format!("claim:{}", claim_id);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&claim_anchor)?, LinkTypes::ClaimToChallenge)?,
        GetStrategy::default(),
    )?;

    let mut challenges = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            challenges.push(record);
        }
    }

    Ok(challenges)
}

/// Search claims by epistemic position range
#[hdk_extern]
pub fn search_by_epistemic_range(input: EpistemicRangeSearch) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Claim,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    let matching: Vec<Record> = records
        .into_iter()
        .filter(|record| {
            if let Some(claim) = record.entry().to_app_option::<Claim>().ok().flatten() {
                let c = &claim.classification;
                c.empirical >= input.e_min
                    && c.empirical <= input.e_max
                    && c.normative >= input.n_min
                    && c.normative <= input.n_max
                    && c.mythic >= input.m_min
                    && c.mythic <= input.m_max
            } else {
                false
            }
        })
        .collect();

    Ok(matching)
}

/// Input for epistemic range search
#[derive(Serialize, Deserialize, Debug)]
pub struct EpistemicRangeSearch {
    pub e_min: f64,
    pub e_max: f64,
    pub n_min: f64,
    pub n_max: f64,
    pub m_min: f64,
    pub m_max: f64,
}

// ============================================================================
// MARKET INTEGRATION FUNCTIONS
// ============================================================================

/// Input for spawning a verification market
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpawnVerificationMarketInput {
    /// Claim to verify
    pub claim_id: String,

    /// Target epistemic level
    pub target_epistemic: EpistemicTarget,

    /// Question for the market
    pub question: String,

    /// Bounty in credits
    pub bounty: u64,

    /// Market deadline
    pub deadline: Timestamp,
}

/// Spawn a verification market for a claim
#[hdk_extern]
pub fn spawn_verification_market(input: SpawnVerificationMarketInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Verify the claim exists
    let _claim_record = get_claim(input.claim_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Claim not found".into())))?;

    // Create the market link entry
    let link_id = format!("cml_{}_{}", input.claim_id, now.as_micros());
    let market_link = ClaimMarketLink {
        id: link_id.clone(),
        claim_id: input.claim_id.clone(),
        market_id: String::new(), // Will be updated when market is created
        link_type: MarketLinkType::VerificationMarket,
        status: MarketVerificationStatus::Pending,
        target_epistemic: input.target_epistemic.clone(),
        requested_by: agent,
        created_at: now,
        resolved_at: None,
        resolution: None,
    };

    let link_hash = create_entry(EntryTypes::ClaimMarketLink(market_link))?;

    // Create links for indexing
    let claim_anchor = format!("claim:{}", input.claim_id);
    create_entry(&EntryTypes::Anchor(Anchor(claim_anchor.clone())))?;
    create_link(
        anchor_hash(&claim_anchor)?,
        link_hash.clone(),
        LinkTypes::ClaimToMarketLink,
        (),
    )?;

    // Index by verification status
    let status_anchor = "status:Pending".to_string();
    create_entry(&EntryTypes::Anchor(Anchor(status_anchor.clone())))?;
    create_link(
        anchor_hash(&status_anchor)?,
        link_hash.clone(),
        LinkTypes::VerificationStatusIndex,
        (),
    )?;

    Ok(link_hash)
}

/// Input for market resolution callback
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MarketResolvedInput {
    /// Market link ID
    pub link_id: String,

    /// Market ID from Epistemic Markets
    pub market_id: String,

    /// Resolution outcome
    pub resolution: MarketResolution,
}

/// Handle market resolution - update claim epistemic position
#[hdk_extern]
pub fn on_market_resolved(input: MarketResolvedInput) -> ExternResult<ActionHash> {
    // Find the market link
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::ClaimMarketLink,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    let mut found_record: Option<Record> = None;
    for record in records {
        if let Some(link) = record
            .entry()
            .to_app_option::<ClaimMarketLink>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if link.id == input.link_id {
                found_record = Some(record);
                break;
            }
        }
    }

    let original_record = found_record.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Market link not found".into()
    )))?;

    let original_link: ClaimMarketLink = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid market link".into()
        )))?;

    // Update the market link
    let now = sys_time()?;
    let updated_link = ClaimMarketLink {
        id: original_link.id,
        claim_id: original_link.claim_id.clone(),
        market_id: input.market_id,
        link_type: original_link.link_type,
        status: MarketVerificationStatus::Resolved,
        target_epistemic: original_link.target_epistemic,
        requested_by: original_link.requested_by,
        created_at: original_link.created_at,
        resolved_at: Some(now),
        resolution: Some(input.resolution.clone()),
    };

    let updated_hash = update_entry(
        original_record.action_address().clone(),
        &EntryTypes::ClaimMarketLink(updated_link),
    )?;

    // Update the claim's epistemic position based on resolution
    if input.resolution.target_achieved {
        // Get the claim and update its classification
        if let Some(claim_record) = get_claim(original_link.claim_id.clone())? {
            let mut claim: Claim = claim_record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid claim".into())))?;

            // Update to the new epistemic position from resolution
            claim.classification = input.resolution.new_epistemic;
            claim.confidence = input.resolution.matl_weighted_confidence;
            claim.version += 1;
            claim.updated = now;

            update_entry(
                claim_record.action_address().clone(),
                &EntryTypes::Claim(claim),
            )?;

            // Trigger cascade update for dependent claims
            match cascade_update(original_link.claim_id.clone()) {
                Ok(result) => {
                    debug!(
                        "Cascade update for claim {}: {} claims updated, depth={}, {} circular refs detected, {}ms",
                        original_link.claim_id,
                        result.updated_claims.len(),
                        result.depth_reached,
                        result.circular_detected.len(),
                        result.processing_time_ms,
                    );
                }
                Err(e) => {
                    debug!(
                        "Cascade update failed for claim {}: {:?}",
                        original_link.claim_id, e
                    );
                }
            }
        }
    }

    Ok(updated_hash)
}

/// Get claims pending verification above a minimum epistemic level
#[hdk_extern]
pub fn get_claims_pending_verification(min_e: f64) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Claim,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    let pending: Vec<Record> = records
        .into_iter()
        .filter(|record| {
            if let Some(claim) = record.entry().to_app_option::<Claim>().ok().flatten() {
                // Claims with high empirical potential but not yet verified
                claim.classification.empirical >= min_e
                    && claim.classification.empirical < 0.9 // Not yet at E4
                    && matches!(claim.claim_type, ClaimType::Fact | ClaimType::Hypothesis)
            } else {
                false
            }
        })
        .collect();

    Ok(pending)
}

// ============================================================================
// DEPENDENCY MANAGEMENT FUNCTIONS
// ============================================================================

/// Input for registering a dependency
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RegisterDependencyInput {
    /// Claim that depends on another
    pub dependent_claim_id: String,

    /// Claim being depended upon
    pub dependency_claim_id: String,

    /// Type of dependency
    pub dependency_type: DependencyType,

    /// Weight of dependency (0.0-1.0)
    pub weight: f64,

    /// Direction of influence
    pub influence: InfluenceDirection,

    /// Optional justification
    pub justification: Option<String>,
}

/// Register a dependency relationship between two claims
#[hdk_extern]
pub fn register_dependency(input: RegisterDependencyInput) -> ExternResult<Record> {
    // Validate both claims exist
    let _dependent = get_claim(input.dependent_claim_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Dependent claim not found".into())
    ))?;
    let _dependency = get_claim(input.dependency_claim_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Dependency claim not found".into())
    ))?;

    // Validate weight
    if input.weight < 0.0 || input.weight > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Dependency weight must be between 0.0 and 1.0".into()
        )));
    }

    // Check for circular dependency
    if would_create_cycle(&input.dependent_claim_id, &input.dependency_claim_id)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "This dependency would create a circular reference".into()
        )));
    }

    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let dep_id = format!(
        "dep_{}_{}_{}",
        input.dependent_claim_id,
        input.dependency_claim_id,
        now.as_micros()
    );

    let dependency = ClaimDependency {
        id: dep_id,
        dependent_claim_id: input.dependent_claim_id.clone(),
        dependency_claim_id: input.dependency_claim_id.clone(),
        dependency_type: input.dependency_type,
        weight: input.weight,
        influence: input.influence,
        established_by: agent,
        created_at: now,
        active: true,
        justification: input.justification,
    };

    let action_hash = create_entry(EntryTypes::ClaimDependency(dependency))?;

    // Create links for fast lookup
    let dependent_anchor = format!("claim:{}", input.dependent_claim_id);
    let dependency_anchor = format!("claim:{}", input.dependency_claim_id);
    create_entry(&EntryTypes::Anchor(Anchor(dependent_anchor.clone())))?;
    create_entry(&EntryTypes::Anchor(Anchor(dependency_anchor.clone())))?;

    // Dependent → its dependencies
    create_link(
        anchor_hash(&dependent_anchor)?,
        action_hash.clone(),
        LinkTypes::ClaimToDependency,
        (),
    )?;

    // Dependency → its dependents (reverse lookup)
    create_link(
        anchor_hash(&dependency_anchor)?,
        action_hash.clone(),
        LinkTypes::DependencyToClaim,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created dependency".into()
    )))
}

/// Check if adding a dependency would create a cycle
fn would_create_cycle(dependent_id: &str, dependency_id: &str) -> ExternResult<bool> {
    // Simple check: does the dependency depend (directly or indirectly) on the dependent?
    let mut visited: Vec<String> = vec![];
    let mut to_check: Vec<String> = vec![dependency_id.to_string()];

    while let Some(current) = to_check.pop() {
        if current == dependent_id {
            return Ok(true); // Cycle detected
        }

        if visited.contains(&current) {
            continue;
        }
        visited.push(current.clone());

        // Get dependencies of current
        let deps = get_claim_dependencies(current)?;
        for dep_record in deps {
            if let Some(dep) = dep_record
                .entry()
                .to_app_option::<ClaimDependency>()
                .ok()
                .flatten()
            {
                to_check.push(dep.dependency_claim_id);
            }
        }
    }

    Ok(false)
}

/// Get all dependencies of a claim
#[hdk_extern]
pub fn get_claim_dependencies(claim_id: String) -> ExternResult<Vec<Record>> {
    let claim_anchor = format!("claim:{}", claim_id);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&claim_anchor)?, LinkTypes::ClaimToDependency)?,
        GetStrategy::default(),
    )?;

    let mut deps = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            deps.push(record);
        }
    }

    Ok(deps)
}

/// Get all claims that depend on this claim
#[hdk_extern]
pub fn get_claim_dependents(claim_id: String) -> ExternResult<Vec<Record>> {
    let claim_anchor = format!("claim:{}", claim_id);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&claim_anchor)?, LinkTypes::DependencyToClaim)?,
        GetStrategy::default(),
    )?;

    let mut dependents = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            dependents.push(record);
        }
    }

    Ok(dependents)
}

/// Maximum depth for cascade propagation
const CASCADE_MAX_DEPTH: u32 = 10;
/// Maximum number of claims that can be updated in a single cascade
const CASCADE_MAX_UPDATES: usize = 100;

/// Cascade update - propagate changes through the dependency graph
///
/// Circuit breakers:
/// - Max depth: 10 levels of dependency traversal
/// - Max updates: 100 claims per cascade operation
/// - Cycle detection: visited set prevents infinite loops
#[hdk_extern]
pub fn cascade_update(claim_id: String) -> ExternResult<CascadeResult> {
    let start_time = sys_time()?;

    let mut updated_claims: Vec<String> = vec![];
    let mut circular_detected: Vec<String> = vec![];
    let mut max_depth: u32 = 0;

    // BFS through dependents
    let mut to_process: Vec<(String, u32)> = vec![(claim_id.clone(), 0)];
    let mut visited: Vec<String> = vec![];

    while let Some((current_id, depth)) = to_process.pop() {
        // Circuit breaker: max updates reached
        if updated_claims.len() >= CASCADE_MAX_UPDATES {
            debug!(
                "Cascade circuit breaker: max updates ({}) reached at depth {} for trigger claim {}",
                CASCADE_MAX_UPDATES, depth, claim_id
            );
            break;
        }

        if visited.contains(&current_id) {
            circular_detected.push(current_id.clone());
            continue;
        }
        visited.push(current_id.clone());

        if depth > max_depth {
            max_depth = depth;
        }

        // Get all claims that depend on this one
        let dependents = match get_claim_dependents(current_id.clone()) {
            Ok(deps) => deps,
            Err(e) => {
                debug!(
                    "Failed to get dependents for claim {} during cascade: {:?}",
                    current_id, e
                );
                continue;
            }
        };

        for dep_record in dependents {
            if let Some(dep) = dep_record
                .entry()
                .to_app_option::<ClaimDependency>()
                .ok()
                .flatten()
            {
                if !dep.active {
                    continue;
                }

                // Get the dependent claim and potentially update it
                if let Some(claim_record) = get_claim(dep.dependent_claim_id.clone())? {
                    let mut claim: Claim = match claim_record.entry().to_app_option() {
                        Ok(Some(c)) => c,
                        _ => continue,
                    };

                    // Recalculate the claim's epistemic position based on its dependencies
                    let new_classification = match recalculate_epistemic(&dep.dependent_claim_id) {
                        Ok(c) => c,
                        Err(e) => {
                            debug!(
                                "Failed to recalculate epistemic for claim {} during cascade: {:?}",
                                dep.dependent_claim_id, e
                            );
                            continue;
                        }
                    };

                    if claim.classification != new_classification {
                        let now = sys_time()?;
                        claim.classification = new_classification;
                        claim.version += 1;
                        claim.updated = now;

                        match update_entry(
                            claim_record.action_address().clone(),
                            &EntryTypes::Claim(claim),
                        ) {
                            Ok(_) => {
                                updated_claims.push(dep.dependent_claim_id.clone());
                            }
                            Err(e) => {
                                debug!(
                                    "Failed to update claim {} during cascade: {:?}",
                                    dep.dependent_claim_id, e
                                );
                                continue;
                            }
                        }
                    }

                    // Add to queue for further propagation (within depth limit)
                    if depth < CASCADE_MAX_DEPTH {
                        to_process.push((dep.dependent_claim_id.clone(), depth + 1));
                    }
                }
            }
        }
    }

    let end_time = sys_time()?;
    let processing_time_ms = (end_time.as_micros() - start_time.as_micros()) / 1000;

    Ok(CascadeResult {
        updated_claims,
        trigger_claim_id: claim_id,
        depth_reached: max_depth,
        circular_detected,
        processed_at: end_time,
        processing_time_ms: processing_time_ms as u64,
    })
}

/// Recalculate epistemic position based on dependencies
fn recalculate_epistemic(claim_id: &str) -> ExternResult<EpistemicPosition> {
    let deps = get_claim_dependencies(claim_id.to_string())?;

    if deps.is_empty() {
        // No dependencies, return current position
        if let Some(claim_record) = get_claim(claim_id.to_string())? {
            if let Some(claim) = claim_record.entry().to_app_option::<Claim>().ok().flatten() {
                return Ok(claim.classification);
            }
        }
        return Ok(EpistemicPosition::default());
    }

    let mut total_weight = 0.0;
    let mut weighted_e = 0.0;
    let mut weighted_n = 0.0;
    let mut weighted_m = 0.0;

    for dep_record in deps {
        if let Some(dep) = dep_record
            .entry()
            .to_app_option::<ClaimDependency>()
            .ok()
            .flatten()
        {
            if !dep.active {
                continue;
            }

            // Get the dependency claim's epistemic position
            if let Some(dep_claim_record) = get_claim(dep.dependency_claim_id)? {
                if let Some(dep_claim) = dep_claim_record
                    .entry()
                    .to_app_option::<Claim>()
                    .ok()
                    .flatten()
                {
                    let influence_factor = match dep.influence {
                        InfluenceDirection::Positive => 1.0,
                        InfluenceDirection::Negative => -1.0,
                        InfluenceDirection::Bidirectional => 0.5,
                    };

                    total_weight += dep.weight;
                    weighted_e +=
                        dep_claim.classification.empirical * dep.weight * influence_factor;
                    weighted_n +=
                        dep_claim.classification.normative * dep.weight * influence_factor;
                    weighted_m += dep_claim.classification.mythic * dep.weight * influence_factor;
                }
            }
        }
    }

    if total_weight == 0.0 {
        return Ok(EpistemicPosition::default());
    }

    // Normalize and clamp to [0.0, 1.0]
    Ok(EpistemicPosition {
        empirical: (weighted_e / total_weight).clamp(0.0, 1.0),
        normative: (weighted_n / total_weight).clamp(0.0, 1.0),
        mythic: (weighted_m / total_weight).clamp(0.0, 1.0),
    })
}

/// Get market links for a claim
#[hdk_extern]
pub fn get_claim_market_links(claim_id: String) -> ExternResult<Vec<Record>> {
    let claim_anchor = format!("claim:{}", claim_id);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&claim_anchor)?, LinkTypes::ClaimToMarketLink)?,
        GetStrategy::default(),
    )?;

    let mut market_links = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            market_links.push(record);
        }
    }

    Ok(market_links)
}
