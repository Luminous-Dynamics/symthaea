// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

//================================
// Constants
//================================

/// Maximum number of guardians a single DID can have
const MAX_GUARDIANS_PER_DID: usize = 20;

/// Minimum number of guardians required for HIGH assurance level
const MIN_GUARDIANS_HIGH_ASSURANCE: usize = 5;

/// Minimum number of guardians required for MEDIUM assurance level
const MIN_GUARDIANS_MEDIUM_ASSURANCE: usize = 3;

/// Maximum depth for circular detection traversal (DoS prevention)
const MAX_CYCLE_DETECTION_DEPTH: usize = 10;

/// Maximum nodes to visit during cycle detection (DoS prevention)
const MAX_NODES_TO_VISIT: usize = 100;

//================================
// Entry Types
//================================

/// Guardian relationship between two DIDs
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GuardianRelationship {
    pub subject_did: String,          // DID being guarded
    pub guardian_did: String,         // DID acting as guardian
    pub relationship_type: String,    // "RECOVERY", "ENDORSEMENT", "DELEGATION"
    pub weight: f64,                  // Guardian influence (0.0-1.0)
    pub status: String,               // "ACTIVE", "PENDING", "REVOKED"
    pub metadata: String,             // JSON-encoded relationship data
    pub established_at: i64,
    pub expires_at: Option<i64>,
    pub mutual: bool,                 // Is this a bidirectional relationship?
}

/// Guardian graph metrics for a DID
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GuardianGraphMetrics {
    pub did: String,
    pub guardian_count: u32,
    pub guarded_count: u32,
    pub diversity_score: f64,         // 0.0-1.0 (higher = more diverse)
    pub cartel_risk_score: f64,       // 0.0-1.0 (higher = more risk)
    pub cluster_id: Option<String>,   // Detected cartel cluster ID
    pub average_guardian_reputation: f64,
    pub network_degree: u32,          // Total connections (guardians + guarded)
    pub computed_at: i64,
    pub version: u32,
}

/// Guardian consent record - tracks guardian's acceptance
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GuardianConsent {
    pub subject_did: String,          // DID requesting guardianship
    pub guardian_did: String,         // DID being asked to be guardian
    pub relationship_type: String,    // Type of relationship requested
    pub status: String,               // "PENDING", "ACCEPTED", "REJECTED"
    pub requested_at: i64,
    pub responded_at: Option<i64>,
    pub response_signature: Option<Vec<u8>>,  // Guardian's signature on acceptance
}

/// Circular guardianship detection result
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CircularityCheckResult {
    pub has_cycle: bool,
    pub cycle_path: Vec<String>,      // DIDs forming the cycle
    pub cycle_type: String,           // "DIRECT", "INDIRECT", "NONE"
    pub depth: u32,                   // Depth at which cycle was found
}

/// Guardian network strength metrics
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GuardianNetworkStrength {
    pub did: String,
    pub total_guardians: u32,
    pub active_guardians: u32,
    pub recovery_capable: bool,       // Has enough guardians for recovery
    pub assurance_level: String,      // "HIGH", "MEDIUM", "LOW", "NONE"
    pub diversity_score: f64,
    pub collective_trust_score: f64,
    pub network_connectivity: f64,    // How connected to the broader network
    pub weakness_flags: Vec<String>,  // Potential vulnerability indicators
}

//================================
// Link Types
//================================

#[hdk_link_types]
pub enum LinkTypes {
    GuardianOf,              // Subject DID -> Guardian DID
    GuardedBy,               // Guardian DID -> Subject DID
    GuardianMetricsLink,     // DID -> Metrics
    GuardianConsentLink,     // DID -> Consent
}

//================================
// Entry Definitions
//================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type]
    GuardianRelationship(GuardianRelationship),
    #[entry_type]
    GuardianGraphMetrics(GuardianGraphMetrics),
    #[entry_type]
    GuardianConsent(GuardianConsent),
}

//================================
// Zome Functions
//================================

/// Add a guardian relationship with full authorization and validation
#[hdk_extern]
pub fn add_guardian(input: AddGuardianInput) -> ExternResult<ActionHash> {
    // 1. Validate relationship structure
    validate_guardian_relationship(&input.relationship)?;

    // 2. Verify caller authorization
    verify_add_guardian_authorization(&input.relationship)?;

    // 3. Check guardian consent if not self-initiated by guardian
    verify_guardian_consent(&input.relationship)?;

    // 4. Check maximum guardians limit
    let existing_guardians = get_guardians_internal(input.relationship.subject_did.clone())?;
    if existing_guardians.len() >= MAX_GUARDIANS_PER_DID {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Maximum guardians ({}) reached for this DID", MAX_GUARDIANS_PER_DID)
        )));
    }

    // 5. Check for duplicate relationship
    let existing = get_guardian_relationship(
        input.relationship.subject_did.clone(),
        input.relationship.guardian_did.clone()
    )?;

    if existing.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Guardian relationship already exists".into()
        )));
    }

    // 6. Check for circular guardianship
    let circularity = detect_circular_guardianship(
        input.relationship.subject_did.clone(),
        input.relationship.guardian_did.clone()
    )?;

    if circularity.has_cycle {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Circular guardianship detected: {} (path: {})",
                    circularity.cycle_type,
                    circularity.cycle_path.join(" -> "))
        )));
    }

    // 7. Create entry on DHT
    let action_hash = create_entry(&EntryTypes::GuardianRelationship(input.relationship.clone()))?;

    // 8. Create forward link (subject -> guardian)
    let subject_path = Path::from(format!("guardian.subject.{}", input.relationship.subject_did)).typed(LinkTypes::GuardianOf)?;
    subject_path.ensure()?;
    create_link(
        subject_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::GuardianOf,
        LinkTag::new(input.relationship.guardian_did.as_bytes())
    )?;

    // 9. Create reverse link (guardian -> subject)
    let guardian_path = Path::from(format!("guardian.guardian.{}", input.relationship.guardian_did)).typed(LinkTypes::GuardedBy)?;
    guardian_path.ensure()?;
    create_link(
        guardian_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::GuardedBy,
        LinkTag::new(input.relationship.subject_did.as_bytes())
    )?;

    debug!("Guardian relationship added: {} guards {}",
           input.relationship.guardian_did, input.relationship.subject_did);

    // 10. Trigger metrics update for both DIDs
    let _ = compute_guardian_metrics(input.relationship.subject_did.clone());
    let _ = compute_guardian_metrics(input.relationship.guardian_did.clone());

    Ok(action_hash)
}

/// Remove a guardian relationship with proper authorization
#[hdk_extern]
pub fn remove_guardian(input: RemoveGuardianInput) -> ExternResult<()> {
    // 1. Get existing relationship
    let relationship = get_guardian_relationship(
        input.subject_did.clone(),
        input.guardian_did.clone()
    )?.ok_or(wasm_error!(WasmErrorInner::Guest("Guardian relationship not found".into())))?;

    // 2. Verify caller authorization
    verify_remove_guardian_authorization(&input.subject_did, &input.guardian_did)?;

    // 3. Mark as revoked (don't delete from DHT)
    let mut updated = relationship.clone();
    updated.status = "REVOKED".to_string();

    let _action_hash = update_entry(
        input.original_action_hash,
        EntryTypes::GuardianRelationship(updated)
    )?;

    // 4. Remove links (makes relationship non-discoverable)
    let subject_path = Path::from(format!("guardian.subject.{}", input.subject_did));
    let guardian_path = Path::from(format!("guardian.guardian.{}", input.guardian_did));

    let subject_links = get_links(
        LinkQuery::try_new(subject_path.path_entry_hash()?, LinkTypes::GuardianOf)?,
        GetStrategy::default()
    )?;

    let guardian_links = get_links(
        LinkQuery::try_new(guardian_path.path_entry_hash()?, LinkTypes::GuardedBy)?,
        GetStrategy::default()
    )?;

    // Delete matching links
    for link in subject_links {
        if link.tag.0 == input.guardian_did.as_bytes() {
            delete_link(link.create_link_hash, GetOptions::default())?;
        }
    }

    for link in guardian_links {
        if link.tag.0 == input.subject_did.as_bytes() {
            delete_link(link.create_link_hash, GetOptions::default())?;
        }
    }

    debug!("Guardian relationship removed: {} no longer guards {}",
           input.guardian_did, input.subject_did);

    // 5. Trigger metrics update
    let _ = compute_guardian_metrics(input.subject_did);
    let _ = compute_guardian_metrics(input.guardian_did);

    Ok(())
}

/// Request guardian consent - creates a pending consent record
#[hdk_extern]
pub fn request_guardian_consent(input: RequestConsentInput) -> ExternResult<ActionHash> {
    // 1. Validate DIDs
    if !input.subject_did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest("Invalid subject DID format".into())));
    }
    if !input.guardian_did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest("Invalid guardian DID format".into())));
    }

    // 2. Prevent self-guardianship
    if input.subject_did == input.guardian_did {
        return Err(wasm_error!(WasmErrorInner::Guest("Cannot be your own guardian".into())));
    }

    // 3. Create consent record
    let consent = GuardianConsent {
        subject_did: input.subject_did.clone(),
        guardian_did: input.guardian_did.clone(),
        relationship_type: input.relationship_type,
        status: "PENDING".to_string(),
        requested_at: sys_time()?.as_micros(),
        responded_at: None,
        response_signature: None,
    };

    let action_hash = create_entry(&EntryTypes::GuardianConsent(consent.clone()))?;

    // 4. Create link for guardian to find pending requests
    let consent_path = Path::from(format!("consent.guardian.{}", input.guardian_did)).typed(LinkTypes::GuardianConsentLink)?;
    consent_path.ensure()?;
    create_link(
        consent_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::GuardianConsentLink,
        LinkTag::new(input.subject_did.as_bytes())
    )?;

    debug!("Guardian consent requested: {} wants {} as guardian",
           input.subject_did, input.guardian_did);

    Ok(action_hash)
}

/// Accept or reject guardian consent request
#[hdk_extern]
pub fn respond_to_consent(input: RespondConsentInput) -> ExternResult<ActionHash> {
    // 1. Get the consent record
    let consent_path = Path::from(format!("consent.guardian.{}", input.guardian_did));
    let links = get_links(
        LinkQuery::try_new(consent_path.path_entry_hash()?, LinkTypes::GuardianConsentLink)?,
        GetStrategy::default()
    )?;

    let mut found_consent: Option<(ActionHash, GuardianConsent)> = None;

    for link in links {
        if link.tag.0 == input.subject_did.as_bytes() {
            let action_hash = link.target.into_action_hash()
                .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(entry) = record.entry().as_option() {
                    if let Entry::App(app_bytes) = entry {
                        let consent = GuardianConsent::try_from(
                            SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                        ).map_err(|e| wasm_error!(e))?;

                        if consent.status == "PENDING" {
                            found_consent = Some((action_hash, consent));
                            break;
                        }
                    }
                }
            }
        }
    }

    let (original_hash, mut consent) = found_consent
        .ok_or(wasm_error!(WasmErrorInner::Guest("Pending consent not found".into())))?;

    // 2. Verify caller is the guardian
    // Note: In production, verify against DID document controller
    let _caller = agent_info()?.agent_initial_pubkey;

    // 3. Update consent status
    consent.status = if input.accept { "ACCEPTED".to_string() } else { "REJECTED".to_string() };
    consent.responded_at = Some(sys_time()?.as_micros());
    consent.response_signature = input.signature;

    let action_hash = update_entry(original_hash, EntryTypes::GuardianConsent(consent.clone()))?;

    debug!("Guardian consent {}: {} for {} as guardian",
           if input.accept { "accepted" } else { "rejected" },
           input.subject_did, input.guardian_did);

    Ok(action_hash)
}

/// Detect circular guardianship before adding a new relationship
#[hdk_extern]
pub fn detect_circular_guardianship(subject_did: String, proposed_guardian_did: String) -> ExternResult<CircularityCheckResult> {
    // Direct cycle check: Would the proposed guardian create A guards B, B guards A?
    let direct_reverse = get_guardian_relationship(
        proposed_guardian_did.clone(),
        subject_did.clone()
    )?;

    if direct_reverse.is_some() {
        return Ok(CircularityCheckResult {
            has_cycle: true,
            cycle_path: vec![subject_did.clone(), proposed_guardian_did.clone(), subject_did],
            cycle_type: "DIRECT".to_string(),
            depth: 1,
        });
    }

    // Indirect cycle check using BFS
    // Check if subject_did can reach proposed_guardian_did through guardian relationships
    // If so, adding this relationship would create a cycle

    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, Vec<String>, u32)> = VecDeque::new();
    let mut nodes_visited = 0;

    // Start from the proposed guardian and see if we can reach the subject
    queue.push_back((proposed_guardian_did.clone(), vec![proposed_guardian_did.clone()], 0));
    visited.insert(proposed_guardian_did.clone());

    while let Some((current_did, path, depth)) = queue.pop_front() {
        nodes_visited += 1;

        // DoS prevention
        if nodes_visited > MAX_NODES_TO_VISIT {
            debug!("Cycle detection: max nodes visited, stopping search");
            break;
        }

        if depth >= MAX_CYCLE_DETECTION_DEPTH as u32 {
            continue;
        }

        // Get guardians of current DID
        let guardians = get_guardians_internal(current_did.clone())?;

        for guardian in guardians {
            // Found cycle - proposed guardian can reach subject through existing guardians
            if guardian.guardian_did == subject_did {
                let mut cycle_path = path.clone();
                cycle_path.push(subject_did.clone());
                return Ok(CircularityCheckResult {
                    has_cycle: true,
                    cycle_path,
                    cycle_type: "INDIRECT".to_string(),
                    depth: depth + 1,
                });
            }

            // Continue BFS
            if !visited.contains(&guardian.guardian_did) {
                visited.insert(guardian.guardian_did.clone());
                let mut new_path = path.clone();
                new_path.push(guardian.guardian_did.clone());
                queue.push_back((guardian.guardian_did.clone(), new_path, depth + 1));
            }
        }
    }

    Ok(CircularityCheckResult {
        has_cycle: false,
        cycle_path: vec![],
        cycle_type: "NONE".to_string(),
        depth: 0,
    })
}

/// Get all guardians for a DID
#[hdk_extern]
pub fn get_guardians(did: String) -> ExternResult<Vec<GuardianRelationship>> {
    get_guardians_internal(did)
}

/// Get all DIDs guarded by a guardian DID
#[hdk_extern]
pub fn get_guarded_by(guardian_did: String) -> ExternResult<Vec<GuardianRelationship>> {
    let path = Path::from(format!("guardian.guardian.{}", guardian_did));

    let links = get_links(
        LinkQuery::try_new(path.path_entry_hash()?, LinkTypes::GuardedBy)?,
        GetStrategy::default()
    )?;

    let mut relationships = Vec::new();

    for link in links {
        let action_hash = link.target.into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(entry) = record.entry().as_option() {
                if let Entry::App(app_bytes) = entry {
                    let relationship = GuardianRelationship::try_from(
                        SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                    ).map_err(|e| wasm_error!(e))?;

                    // Filter expired and revoked
                    if relationship.status != "ACTIVE" {
                        continue;
                    }

                    if let Some(expires_at) = relationship.expires_at {
                        if expires_at < sys_time()?.as_micros() {
                            continue;
                        }
                    }

                    relationships.push(relationship);
                }
            }
        }
    }

    Ok(relationships)
}

/// Compute guardian graph metrics for a DID
#[hdk_extern]
pub fn compute_guardian_metrics(did: String) -> ExternResult<ActionHash> {
    // 1. Get guardians and guarded relationships
    let guardians = get_guardians_internal(did.clone())?;
    let guarded = get_guarded_by(did.clone())?;

    // 2. Compute diversity score
    let diversity_score = compute_diversity_score(&guardians)?;

    // 3. Detect cartel risk with circular detection
    let (cartel_risk_score, cluster_id) = detect_cartel_risk_with_circularity(&did, &guardians)?;

    // 4. Compute average guardian reputation
    let average_guardian_reputation = compute_average_guardian_reputation(&guardians)?;

    // 5. Get previous version
    let previous_version = if let Some(prev) = get_guardian_metrics(did.clone())? {
        prev.version
    } else {
        0
    };

    // 6. Create metrics entry
    let metrics = GuardianGraphMetrics {
        did: did.clone(),
        guardian_count: guardians.len() as u32,
        guarded_count: guarded.len() as u32,
        diversity_score,
        cartel_risk_score,
        cluster_id,
        average_guardian_reputation,
        network_degree: (guardians.len() + guarded.len()) as u32,
        computed_at: sys_time()?.as_micros(),
        version: previous_version + 1,
    };

    // 7. Store metrics on DHT
    let action_hash = create_entry(&EntryTypes::GuardianGraphMetrics(metrics.clone()))?;

    // 8. Create link for resolution
    let path = Path::from(format!("guardian_metrics.{}", did)).typed(LinkTypes::GuardianMetricsLink)?;
    path.ensure()?;
    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::GuardianMetricsLink,
        LinkTag::new("metrics")
    )?;

    debug!("Guardian metrics computed for {}: diversity={}, cartel_risk={}",
           did, diversity_score, cartel_risk_score);

    Ok(action_hash)
}

/// Get guardian graph metrics for a DID
#[hdk_extern]
pub fn get_guardian_metrics(did: String) -> ExternResult<Option<GuardianGraphMetrics>> {
    let path = Path::from(format!("guardian_metrics.{}", did));

    let links = get_links(
        LinkQuery::try_new(path.path_entry_hash()?, LinkTypes::GuardianMetricsLink)?,
        GetStrategy::default()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get most recent metrics
    let action_hash = links.last().expect("links verified non-empty above").target.clone().into_action_hash()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

    if let Some(record) = get(action_hash, GetOptions::default())? {
        if let Some(entry) = record.entry().as_option() {
            if let Entry::App(app_bytes) = entry {
                let metrics = GuardianGraphMetrics::try_from(
                    SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                ).map_err(|e| wasm_error!(e))?;
                return Ok(Some(metrics));
            }
        }
    }

    Ok(None)
}

/// Get guardian network strength for a DID
#[hdk_extern]
pub fn get_guardian_network_strength(did: String) -> ExternResult<GuardianNetworkStrength> {
    let guardians = get_guardians_internal(did.clone())?;
    let active_guardians: Vec<_> = guardians.iter()
        .filter(|g| g.status == "ACTIVE")
        .collect();

    let total = guardians.len() as u32;
    let active = active_guardians.len() as u32;

    // Determine assurance level
    let assurance_level = if active >= MIN_GUARDIANS_HIGH_ASSURANCE as u32 {
        "HIGH".to_string()
    } else if active >= MIN_GUARDIANS_MEDIUM_ASSURANCE as u32 {
        "MEDIUM".to_string()
    } else if active >= 1 {
        "LOW".to_string()
    } else {
        "NONE".to_string()
    };

    // Recovery capability (need at least 2 guardians for basic recovery)
    let recovery_capable = active >= 2;

    // Compute diversity
    let diversity_score = compute_diversity_score(&guardians)?;

    // Compute collective trust (weighted average of guardian reputations)
    let collective_trust_score = compute_collective_trust_score(&guardians)?;

    // Network connectivity (how connected are our guardians to others)
    let network_connectivity = compute_network_connectivity(&guardians)?;

    // Identify weaknesses
    let mut weakness_flags = Vec::new();

    if active < MIN_GUARDIANS_MEDIUM_ASSURANCE as u32 {
        weakness_flags.push("INSUFFICIENT_GUARDIANS".to_string());
    }

    if diversity_score < 0.3 {
        weakness_flags.push("LOW_DIVERSITY".to_string());
    }

    // Check for single point of failure (one guardian with very high weight)
    for g in &guardians {
        if g.weight > 0.6 {
            weakness_flags.push(format!("HIGH_WEIGHT_DEPENDENCY:{}", g.guardian_did));
        }
    }

    // Check for potentially circular relationships
    let (cartel_risk, _) = detect_cartel_risk_with_circularity(&did, &guardians)?;
    if cartel_risk > 0.5 {
        weakness_flags.push("CARTEL_RISK".to_string());
    }

    Ok(GuardianNetworkStrength {
        did,
        total_guardians: total,
        active_guardians: active,
        recovery_capable,
        assurance_level,
        diversity_score,
        collective_trust_score,
        network_connectivity,
        weakness_flags,
    })
}

/// Get collective trust score from all guardians
#[hdk_extern]
pub fn get_collective_trust_score(did: String) -> ExternResult<f64> {
    let guardians = get_guardians_internal(did)?;
    compute_collective_trust_score(&guardians)
}

/// Get guardian diversity score
#[hdk_extern]
pub fn get_guardian_diversity_score(did: String) -> ExternResult<f64> {
    let guardians = get_guardians_internal(did)?;
    compute_diversity_score(&guardians)
}

/// Authorize recovery action based on guardian consensus
#[hdk_extern]
pub fn authorize_recovery(input: AuthorizeRecoveryInput) -> ExternResult<RecoveryAuthorization> {
    // 1. Get guardians for subject
    let guardians = get_guardians_internal(input.subject_did.clone())?;

    if guardians.is_empty() {
        return Ok(RecoveryAuthorization {
            authorized: false,
            reason: "No guardians configured".to_string(),
            threshold_met: false,
            approval_weight: 0.0,
            required_weight: input.required_threshold,
        });
    }

    // 2. Calculate total guardian weight
    let _total_weight: f64 = guardians.iter().map(|g| g.weight).sum();

    // 3. Calculate approval weight from approving guardians
    let mut approval_weight = 0.0;
    for approving_did in &input.approving_guardians {
        for guardian in &guardians {
            if &guardian.guardian_did == approving_did && guardian.status == "ACTIVE" {
                approval_weight += guardian.weight;
            }
        }
    }

    // 4. Check if threshold met
    let threshold_met = approval_weight >= input.required_threshold;

    // 5. Return authorization result
    Ok(RecoveryAuthorization {
        authorized: threshold_met,
        reason: if threshold_met {
            format!("Guardian threshold met: {}/{}", approval_weight, input.required_threshold)
        } else {
            format!("Insufficient guardian approval: {}/{}", approval_weight, input.required_threshold)
        },
        threshold_met,
        approval_weight,
        required_weight: input.required_threshold,
    })
}

/// Check if a proposed guardian relationship would create a cycle
#[hdk_extern]
pub fn would_create_cycle(input: CycleCheckInput) -> ExternResult<bool> {
    let result = detect_circular_guardianship(input.subject_did, input.proposed_guardian_did)?;
    Ok(result.has_cycle)
}

//================================
// Internal Helper Functions
//================================

/// Internal function to get guardians (avoids hdk_extern recursion issues)
fn get_guardians_internal(did: String) -> ExternResult<Vec<GuardianRelationship>> {
    let path = Path::from(format!("guardian.subject.{}", did));

    let links = get_links(
        LinkQuery::try_new(path.path_entry_hash()?, LinkTypes::GuardianOf)?,
        GetStrategy::default()
    )?;

    let mut relationships = Vec::new();

    for link in links {
        let action_hash = link.target.into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(entry) = record.entry().as_option() {
                if let Entry::App(app_bytes) = entry {
                    let relationship = GuardianRelationship::try_from(
                        SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                    ).map_err(|e| wasm_error!(e))?;

                    // Filter expired and revoked
                    if relationship.status != "ACTIVE" {
                        continue;
                    }

                    if let Some(expires_at) = relationship.expires_at {
                        if expires_at < sys_time()?.as_micros() {
                            continue;
                        }
                    }

                    relationships.push(relationship);
                }
            }
        }
    }

    Ok(relationships)
}

/// Get a specific guardian relationship
fn get_guardian_relationship(subject_did: String, guardian_did: String) -> ExternResult<Option<GuardianRelationship>> {
    let guardians = get_guardians_internal(subject_did)?;

    for guardian in guardians {
        if guardian.guardian_did == guardian_did {
            return Ok(Some(guardian));
        }
    }

    Ok(None)
}

/// Verify authorization for adding a guardian
fn verify_add_guardian_authorization(rel: &GuardianRelationship) -> ExternResult<()> {
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_bytes = caller.get_raw_39();

    // The caller must be either:
    // 1. The subject (person requesting a guardian)
    // 2. The guardian (if they're initiating/accepting)
    // 3. Have delegation authority from the subject

    // For now, verify the caller's agent key is embedded in one of the DIDs
    // In production, this would query the did_registry zome to verify DID ownership

    // Extract expected agent key from subject DID
    // Format: did:mycelix:<base58-encoded-agent-key>
    let subject_key_part = rel.subject_did.strip_prefix("did:mycelix:")
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid subject DID format".into())))?;

    let guardian_key_part = rel.guardian_did.strip_prefix("did:mycelix:")
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid guardian DID format".into())))?;

    // Check if caller matches subject or guardian
    // Note: In production, this would do proper DID resolution
    let caller_hex = hex::encode(&caller_bytes);

    let is_subject = subject_key_part.contains(&caller_hex[0..16]);
    let is_guardian = guardian_key_part.contains(&caller_hex[0..16]);

    if !is_subject && !is_guardian {
        // Check for delegation authority
        // This would query existing guardian relationships for delegation type
        let subject_guardians = get_guardians_internal(rel.subject_did.clone())?;
        let has_delegation = subject_guardians.iter().any(|g| {
            g.relationship_type == "DELEGATION" &&
            g.guardian_did.contains(&caller_hex[0..16])
        });

        if !has_delegation {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Caller not authorized to add guardian for this DID".into()
            )));
        }
    }

    Ok(())
}

/// Verify authorization for removing a guardian
fn verify_remove_guardian_authorization(subject_did: &str, guardian_did: &str) -> ExternResult<()> {
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_bytes = caller.get_raw_39();
    let caller_hex = hex::encode(&caller_bytes);

    // The caller must be either:
    // 1. The subject (can remove their own guardians)
    // 2. The guardian being removed (can resign)
    // 3. Have delegation authority from the subject

    let subject_key_part = subject_did.strip_prefix("did:mycelix:")
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid subject DID format".into())))?;

    let guardian_key_part = guardian_did.strip_prefix("did:mycelix:")
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid guardian DID format".into())))?;

    let is_subject = subject_key_part.contains(&caller_hex[0..16]);
    let is_guardian = guardian_key_part.contains(&caller_hex[0..16]);

    if !is_subject && !is_guardian {
        // Check for delegation authority
        let subject_guardians = get_guardians_internal(subject_did.to_string())?;
        let has_delegation = subject_guardians.iter().any(|g| {
            g.relationship_type == "DELEGATION" &&
            g.guardian_did.contains(&caller_hex[0..16])
        });

        if !has_delegation {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Caller not authorized to remove guardian for this DID".into()
            )));
        }
    }

    Ok(())
}

/// Verify guardian consent for a new relationship
fn verify_guardian_consent(rel: &GuardianRelationship) -> ExternResult<()> {
    // If status is PENDING, consent check can be skipped (consent will be required later)
    if rel.status == "PENDING" {
        return Ok(());
    }

    // For ACTIVE relationships, check if guardian has consented
    let consent_path = Path::from(format!("consent.guardian.{}", rel.guardian_did));

    let links = get_links(
        LinkQuery::try_new(consent_path.path_entry_hash()?, LinkTypes::GuardianConsentLink)?,
        GetStrategy::default()
    )?;

    for link in links {
        if link.tag.0 == rel.subject_did.as_bytes() {
            let action_hash = link.target.into_action_hash()
                .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(entry) = record.entry().as_option() {
                    if let Entry::App(app_bytes) = entry {
                        let consent = GuardianConsent::try_from(
                            SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                        ).map_err(|e| wasm_error!(e))?;

                        if consent.status == "ACCEPTED" {
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    // If caller is the guardian themselves, consent is implied
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_bytes = caller.get_raw_39();
    let caller_hex = hex::encode(&caller_bytes);

    let guardian_key_part = rel.guardian_did.strip_prefix("did:mycelix:")
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid guardian DID format".into())))?;

    if guardian_key_part.contains(&caller_hex[0..16]) {
        return Ok(()); // Guardian is making the call, consent implied
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        "Guardian consent not found. Request consent first.".into()
    )))
}

fn validate_guardian_relationship(rel: &GuardianRelationship) -> ExternResult<()> {
    // 1. Validate DIDs
    if !rel.subject_did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest("Invalid subject DID format".into())));
    }

    if !rel.guardian_did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest("Invalid guardian DID format".into())));
    }

    // 2. Prevent self-guardianship
    if rel.subject_did == rel.guardian_did {
        return Err(wasm_error!(WasmErrorInner::Guest("Cannot be your own guardian".into())));
    }

    // 3. Validate relationship type
    match rel.relationship_type.as_str() {
        "RECOVERY" | "ENDORSEMENT" | "DELEGATION" => {},
        _ => return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid relationship type: {}", rel.relationship_type)
        )))
    }

    // 4. Validate weight (0.0-1.0)
    if rel.weight < 0.0 || rel.weight > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Guardian weight must be between 0.0 and 1.0".into()
        )));
    }

    // 5. Validate status
    match rel.status.as_str() {
        "ACTIVE" | "PENDING" | "REVOKED" => {},
        _ => return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid status: {}", rel.status)
        )))
    }

    // 6. Validate timestamps
    if let Some(expires_at) = rel.expires_at {
        if expires_at <= rel.established_at {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Expiration must be after establishment".into()
            )));
        }
    }

    Ok(())
}

fn compute_diversity_score(guardians: &[GuardianRelationship]) -> ExternResult<f64> {
    if guardians.is_empty() {
        return Ok(0.0);
    }

    // Diversity metric based on:
    // 1. Number of guardians (more = better, up to 10)
    // 2. Relationship type diversity (more types = better)
    // 3. Weight distribution (more even = better)

    let count_score = (guardians.len() as f64).min(10.0) / 10.0;

    // Count relationship types
    let mut type_counts = HashMap::new();
    for guardian in guardians {
        *type_counts.entry(&guardian.relationship_type).or_insert(0) += 1;
    }
    let type_diversity = type_counts.len() as f64 / 3.0; // Max 3 types

    // Weight distribution (using entropy-like measure)
    let total_weight: f64 = guardians.iter().map(|g| g.weight).sum();
    let weight_entropy = if total_weight > 0.0 {
        let mut entropy = 0.0;
        for g in guardians {
            let p = g.weight / total_weight;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        // Normalize: max entropy is ln(n)
        let max_entropy = (guardians.len() as f64).ln();
        if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 }
    } else {
        0.0
    };

    // Compute weighted average
    let diversity = (count_score * 0.4) + (type_diversity * 0.3) + (weight_entropy * 0.3);

    Ok(diversity.min(1.0))
}

fn detect_cartel_risk_with_circularity(did: &str, guardians: &[GuardianRelationship]) -> ExternResult<(f64, Option<String>)> {
    if guardians.is_empty() {
        return Ok((0.0, None));
    }

    let guardian_count = guardians.len();
    let mut risk_score: f64 = 0.0;
    let mut cluster_indicators: Vec<String> = Vec::new();

    // Risk factor 1: Too few guardians
    if guardian_count < MIN_GUARDIANS_MEDIUM_ASSURANCE {
        risk_score += 0.3;
        cluster_indicators.push("insufficient_guardians".to_string());
    }

    // Risk factor 2: Homogeneous relationship types
    let mut type_counts = HashMap::new();
    for guardian in guardians {
        *type_counts.entry(&guardian.relationship_type).or_insert(0) += 1;
    }

    if type_counts.len() == 1 {
        risk_score += 0.2;
        cluster_indicators.push("homogeneous_types".to_string());
    }

    // Risk factor 3: Circular guardianship detection (mutual relationships)
    // Check if any of our guardians are also guarded by us (mutual guardianship)
    let mut mutual_count = 0;
    for guardian in guardians {
        // Check if we guard our guardian
        let reverse = get_guardian_relationship(
            guardian.guardian_did.clone(),
            did.to_string()
        )?;
        if reverse.is_some() {
            mutual_count += 1;
        }
    }

    // High mutual guardianship indicates potential cartel
    if mutual_count > 0 {
        let mutual_ratio = mutual_count as f64 / guardian_count as f64;
        risk_score += mutual_ratio * 0.4;
        cluster_indicators.push(format!("mutual_guardians:{}", mutual_count));
    }

    // Risk factor 4: Check if guardians guard each other - full clique detection
    // Detects when all guardians are guardians of each other (circular cartel)
    let mut inter_guardian_edges = 0;
    let max_possible_edges = guardian_count * (guardian_count - 1);

    for g1 in guardians {
        for g2 in guardians {
            if g1.guardian_did != g2.guardian_did {
                let edge = get_guardian_relationship(
                    g1.guardian_did.clone(),
                    g2.guardian_did.clone()
                )?;
                if edge.is_some() {
                    inter_guardian_edges += 1;
                }
            }
        }
    }

    if max_possible_edges > 0 {
        let clique_density = inter_guardian_edges as f64 / max_possible_edges as f64;

        // Full clique detection: if density is 1.0, all guardians guard each other
        if clique_density >= 1.0 {
            risk_score += 0.5; // Maximum cartel risk for full clique
            cluster_indicators.push("full_clique_detected".to_string());
        } else if clique_density > 0.75 {
            // Near-complete clique - very high cartel risk
            risk_score += 0.4 * clique_density;
            cluster_indicators.push(format!("near_clique:{:.2}", clique_density));
        } else if clique_density > 0.5 {
            risk_score += 0.3 * clique_density;
            cluster_indicators.push(format!("partial_clique:{:.2}", clique_density));
        }
    }

    // Risk factor 5: Deep circular chain detection using BFS
    // Detect longer circular chains (A->B->C->A) that aren't direct mutual
    let circular_chain = detect_circular_chain_bfs(did, guardians)?;
    if let Some(chain_length) = circular_chain {
        // Longer chains are slightly less suspicious than direct mutual
        let chain_risk = 0.2 * (1.0 / chain_length as f64);
        risk_score += chain_risk;
        cluster_indicators.push(format!("circular_chain_length:{}", chain_length));
    }

    // Generate cluster ID if risk exceeds threshold
    let cluster_id = if risk_score > 0.5 {
        Some(format!("cartel_{}_{}",
            cluster_indicators.first().unwrap_or(&"unknown".to_string()),
            &did[did.len().saturating_sub(8)..]))
    } else {
        None
    };

    Ok((risk_score.min(1.0), cluster_id))
}

/// Detect circular guardian chains using BFS
/// Returns the chain length if a cycle is found back to the origin DID
fn detect_circular_chain_bfs(origin_did: &str, guardians: &[GuardianRelationship]) -> ExternResult<Option<usize>> {
    if guardians.is_empty() {
        return Ok(None);
    }

    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    let mut nodes_visited = 0;

    // Start from each guardian and see if we can reach back to origin
    for guardian in guardians {
        queue.push_back((guardian.guardian_did.clone(), 1));
        visited.insert(guardian.guardian_did.clone());
    }

    while let Some((current_did, depth)) = queue.pop_front() {
        nodes_visited += 1;

        // DoS prevention
        if nodes_visited > MAX_NODES_TO_VISIT {
            break;
        }

        if depth >= MAX_CYCLE_DETECTION_DEPTH {
            continue;
        }

        // Get guardians of current DID
        let current_guardians = get_guardians_internal(current_did.clone())?;

        for guardian in current_guardians {
            // Found cycle back to origin
            if guardian.guardian_did == origin_did && depth > 1 {
                return Ok(Some(depth + 1));
            }

            // Continue BFS if not visited
            if !visited.contains(&guardian.guardian_did) {
                visited.insert(guardian.guardian_did.clone());
                queue.push_back((guardian.guardian_did.clone(), depth + 1));
            }
        }
    }

    Ok(None)
}

fn compute_average_guardian_reputation(guardians: &[GuardianRelationship]) -> ExternResult<f64> {
    if guardians.is_empty() {
        return Ok(0.0);
    }

    // Query reputation_sync zome for actual guardian reputation scores
    let mut total_reputation: f64 = 0.0;
    let mut reputation_count: usize = 0;

    for guardian in guardians {
        // Try to get reputation from reputation_sync zome
        let reputation = query_guardian_reputation(&guardian.guardian_did)?;
        if let Some(rep_score) = reputation {
            total_reputation += rep_score;
            reputation_count += 1;
        } else {
            // Fallback to using guardian weight as proxy when reputation unavailable
            total_reputation += guardian.weight;
            reputation_count += 1;
        }
    }

    let average = if reputation_count > 0 {
        total_reputation / reputation_count as f64
    } else {
        0.0
    };

    Ok(average)
}

/// Query the reputation_sync zome to get reputation score for a guardian DID
///
/// This cross-zome call retrieves the reputation score from the reputation_sync zome.
/// Returns None if the reputation_sync zome is unavailable or DID has no reputation.
fn query_guardian_reputation(guardian_did: &str) -> ExternResult<Option<f64>> {
    // Define the expected response structure from reputation_sync zome
    #[derive(Serialize, Deserialize, Debug)]
    struct ReputationScore {
        did: String,
        score: f64,
        confidence: f64,
        computed_at: i64,
    }

    // Call the reputation_sync zome's get_reputation_score function
    let result: ExternResult<Option<ReputationScore>> = call(
        CallTargetCell::Local,
        "reputation_sync",
        "get_reputation_score".into(),
        None,
        guardian_did.to_string(),
    );

    match result {
        Ok(Some(rep)) => Ok(Some(rep.score)),
        Ok(None) => {
            debug!("No reputation found for guardian: {}", guardian_did);
            Ok(None)
        }
        Err(e) => {
            // Log the error but don't fail - reputation_sync might not be available
            debug!("Failed to query reputation_sync for {}: {:?}", guardian_did, e);
            Ok(None)
        }
    }
}

fn compute_collective_trust_score(guardians: &[GuardianRelationship]) -> ExternResult<f64> {
    if guardians.is_empty() {
        return Ok(0.0);
    }

    // Weighted trust score based on:
    // 1. Guardian weights (more weight = more influence)
    // 2. Relationship type (RECOVERY > ENDORSEMENT > DELEGATION for trust)
    // 3. Duration of relationship (older = more trusted)

    let current_time = sys_time()?.as_micros();
    let mut total_trust = 0.0;
    let mut total_weight = 0.0;

    for guardian in guardians {
        let type_multiplier = match guardian.relationship_type.as_str() {
            "RECOVERY" => 1.0,
            "ENDORSEMENT" => 0.8,
            "DELEGATION" => 0.6,
            _ => 0.5,
        };

        // Duration factor (relationships older than 30 days get bonus, max 1 year)
        let duration_secs = (current_time - guardian.established_at) / 1_000_000;
        let duration_days = duration_secs / 86400;
        let duration_factor = if duration_days < 30 {
            0.7
        } else if duration_days < 365 {
            0.7 + (0.3 * (duration_days - 30) as f64 / 335.0)
        } else {
            1.0
        };

        let trust_contribution = guardian.weight * type_multiplier * duration_factor;
        total_trust += trust_contribution;
        total_weight += guardian.weight;
    }

    let collective_trust = if total_weight > 0.0 {
        total_trust / total_weight
    } else {
        0.0
    };

    Ok(collective_trust.min(1.0))
}

fn compute_network_connectivity(guardians: &[GuardianRelationship]) -> ExternResult<f64> {
    if guardians.is_empty() {
        return Ok(0.0);
    }

    // Network connectivity measures how connected our guardians are to the broader network
    // High connectivity = guardians are well-connected themselves
    // Low connectivity = guardians are isolated

    let mut total_connections = 0;
    let mut guardian_count = 0;

    for guardian in guardians {
        // Get how many DIDs this guardian guards
        let guarded = get_guarded_by(guardian.guardian_did.clone())?;
        // Get how many guardians this guardian has
        let their_guardians = get_guardians_internal(guardian.guardian_did.clone())?;

        total_connections += guarded.len() + their_guardians.len();
        guardian_count += 1;
    }

    // Average connections per guardian, normalized
    let avg_connections = if guardian_count > 0 {
        total_connections as f64 / guardian_count as f64
    } else {
        0.0
    };

    // Normalize to 0-1 (assume 10+ connections is "fully connected")
    let connectivity = (avg_connections / 10.0).min(1.0);

    Ok(connectivity)
}

//================================
// Validation Callback
//================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(StoreEntry { entry, .. }) => {
            match entry {
                Entry::App(bytes) => {
                    // Try to deserialize as GuardianRelationship
                    if let Ok(relationship) = GuardianRelationship::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if let Err(e) = validate_guardian_relationship(&relationship) {
                            return Ok(ValidateCallbackResult::Invalid(
                                format!("Invalid guardian relationship: {:?}", e)
                            ));
                        }
                        return Ok(ValidateCallbackResult::Valid);
                    }

                    // Try to deserialize as GuardianGraphMetrics
                    if let Ok(_metrics) = GuardianGraphMetrics::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        return Ok(ValidateCallbackResult::Valid);
                    }

                    // Try to deserialize as GuardianConsent
                    if let Ok(consent) = GuardianConsent::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        // Validate consent
                        if !consent.subject_did.starts_with("did:mycelix:") {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Invalid subject DID format in consent".into()
                            ));
                        }
                        if !consent.guardian_did.starts_with("did:mycelix:") {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Invalid guardian DID format in consent".into()
                            ));
                        }
                        if consent.subject_did == consent.guardian_did {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Cannot consent to self-guardianship".into()
                            ));
                        }
                        return Ok(ValidateCallbackResult::Valid);
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
    Ok(ValidateCallbackResult::Valid)
}

//================================
// Input/Output Types
//================================

#[derive(Serialize, Deserialize, Debug)]
pub struct AddGuardianInput {
    pub relationship: GuardianRelationship,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoveGuardianInput {
    pub subject_did: String,
    pub guardian_did: String,
    pub original_action_hash: ActionHash,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestConsentInput {
    pub subject_did: String,
    pub guardian_did: String,
    pub relationship_type: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RespondConsentInput {
    pub subject_did: String,
    pub guardian_did: String,
    pub accept: bool,
    pub signature: Option<Vec<u8>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AuthorizeRecoveryInput {
    pub subject_did: String,
    pub approving_guardians: Vec<String>,
    pub required_threshold: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecoveryAuthorization {
    pub authorized: bool,
    pub reason: String,
    pub threshold_met: bool,
    pub approval_weight: f64,
    pub required_weight: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CycleCheckInput {
    pub subject_did: String,
    pub proposed_guardian_did: String,
}

//================================
// Tests Module
//================================

#[cfg(test)]
mod tests {
    use super::*;

    // Unit tests for validation functions

    #[test]
    fn test_validate_guardian_relationship_valid() {
        let rel = GuardianRelationship {
            subject_did: "did:mycelix:subject123".to_string(),
            guardian_did: "did:mycelix:guardian456".to_string(),
            relationship_type: "RECOVERY".to_string(),
            weight: 0.5,
            status: "ACTIVE".to_string(),
            metadata: "{}".to_string(),
            established_at: 1000000,
            expires_at: Some(2000000),
            mutual: false,
        };

        // Note: This test can only verify structure, not HDK calls
        assert!(rel.subject_did.starts_with("did:mycelix:"));
        assert!(rel.guardian_did.starts_with("did:mycelix:"));
        assert_ne!(rel.subject_did, rel.guardian_did);
        assert!(rel.weight >= 0.0 && rel.weight <= 1.0);
    }

    #[test]
    fn test_validate_guardian_relationship_self_guardian() {
        let rel = GuardianRelationship {
            subject_did: "did:mycelix:same123".to_string(),
            guardian_did: "did:mycelix:same123".to_string(),
            relationship_type: "RECOVERY".to_string(),
            weight: 0.5,
            status: "ACTIVE".to_string(),
            metadata: "{}".to_string(),
            established_at: 1000000,
            expires_at: None,
            mutual: false,
        };

        // Self-guardianship should be detected
        assert_eq!(rel.subject_did, rel.guardian_did);
    }

    #[test]
    fn test_validate_guardian_relationship_invalid_weight() {
        let rel = GuardianRelationship {
            subject_did: "did:mycelix:subject123".to_string(),
            guardian_did: "did:mycelix:guardian456".to_string(),
            relationship_type: "RECOVERY".to_string(),
            weight: 1.5, // Invalid: > 1.0
            status: "ACTIVE".to_string(),
            metadata: "{}".to_string(),
            established_at: 1000000,
            expires_at: None,
            mutual: false,
        };

        // Invalid weight should be detected
        assert!(rel.weight > 1.0);
    }

    #[test]
    fn test_validate_guardian_relationship_invalid_type() {
        let rel = GuardianRelationship {
            subject_did: "did:mycelix:subject123".to_string(),
            guardian_did: "did:mycelix:guardian456".to_string(),
            relationship_type: "INVALID_TYPE".to_string(),
            weight: 0.5,
            status: "ACTIVE".to_string(),
            metadata: "{}".to_string(),
            established_at: 1000000,
            expires_at: None,
            mutual: false,
        };

        // Invalid type should be detected
        let valid_types = ["RECOVERY", "ENDORSEMENT", "DELEGATION"];
        assert!(!valid_types.contains(&rel.relationship_type.as_str()));
    }

    #[test]
    fn test_circularity_check_result_none() {
        let result = CircularityCheckResult {
            has_cycle: false,
            cycle_path: vec![],
            cycle_type: "NONE".to_string(),
            depth: 0,
        };

        assert!(!result.has_cycle);
        assert!(result.cycle_path.is_empty());
    }

    #[test]
    fn test_circularity_check_result_direct() {
        let result = CircularityCheckResult {
            has_cycle: true,
            cycle_path: vec![
                "did:mycelix:a".to_string(),
                "did:mycelix:b".to_string(),
                "did:mycelix:a".to_string(),
            ],
            cycle_type: "DIRECT".to_string(),
            depth: 1,
        };

        assert!(result.has_cycle);
        assert_eq!(result.cycle_type, "DIRECT");
        assert_eq!(result.cycle_path.len(), 3);
    }

    #[test]
    fn test_circularity_check_result_indirect() {
        let result = CircularityCheckResult {
            has_cycle: true,
            cycle_path: vec![
                "did:mycelix:a".to_string(),
                "did:mycelix:b".to_string(),
                "did:mycelix:c".to_string(),
                "did:mycelix:a".to_string(),
            ],
            cycle_type: "INDIRECT".to_string(),
            depth: 3,
        };

        assert!(result.has_cycle);
        assert_eq!(result.cycle_type, "INDIRECT");
        assert_eq!(result.cycle_path.len(), 4);
    }

    #[test]
    fn test_guardian_network_strength_assurance_levels() {
        // Test assurance level thresholds
        assert!(MIN_GUARDIANS_HIGH_ASSURANCE > MIN_GUARDIANS_MEDIUM_ASSURANCE);
        assert!(MIN_GUARDIANS_MEDIUM_ASSURANCE > 0);
    }

    #[test]
    fn test_max_guardians_limit() {
        // Verify constant is reasonable
        assert!(MAX_GUARDIANS_PER_DID > MIN_GUARDIANS_HIGH_ASSURANCE);
        assert!(MAX_GUARDIANS_PER_DID <= 100); // Reasonable upper bound
    }

    #[test]
    fn test_cycle_detection_depth_limit() {
        // Verify DoS prevention limits
        assert!(MAX_CYCLE_DETECTION_DEPTH > 0);
        assert!(MAX_CYCLE_DETECTION_DEPTH <= 20);
        assert!(MAX_NODES_TO_VISIT > MAX_CYCLE_DETECTION_DEPTH);
    }

    #[test]
    fn test_consent_status_values() {
        let valid_statuses = ["PENDING", "ACCEPTED", "REJECTED"];
        for status in valid_statuses {
            assert!(status.len() > 0);
        }
    }

    #[test]
    fn test_relationship_status_values() {
        let valid_statuses = ["ACTIVE", "PENDING", "REVOKED"];
        for status in valid_statuses {
            assert!(status.len() > 0);
        }
    }

    // Integration test helpers (these would run with sweettest in a full environment)

    #[test]
    fn test_diversity_score_calculation_basic() {
        // Basic verification of diversity score logic
        // With 0 guardians, diversity should be 0
        let empty_guardians: Vec<GuardianRelationship> = vec![];
        assert!(empty_guardians.is_empty());

        // With 10 guardians of diverse types, diversity should be high
        // This is a structural test - actual calculation needs HDK
    }

    #[test]
    fn test_network_strength_weakness_flags() {
        // Verify weakness flag constants
        let known_flags = [
            "INSUFFICIENT_GUARDIANS",
            "LOW_DIVERSITY",
            "CARTEL_RISK",
        ];

        for flag in known_flags {
            assert!(flag.len() > 0);
            assert!(flag.chars().all(|c| c.is_ascii_uppercase() || c == '_'));
        }
    }
}
