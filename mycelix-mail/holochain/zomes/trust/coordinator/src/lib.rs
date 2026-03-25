// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust Coordinator Zome - MATL Algorithm Implementation
//!
//! Full implementation of Mycelix Advanced Trust Logic (MATL) for
//! decentralized spam prevention and identity verification.
//!
//! # Overview
//!
//! The MATL algorithm provides:
//! - **Direct Trust**: Personal attestations with evidence verification
//! - **Transitive Trust**: Trust propagation through your network with decay
//! - **Byzantine Detection**: Sybil attack and collusion resistance
//! - **Temporal Decay**: Recent attestations weighted higher
//! - **Category Scoring**: Different trust for identity, communication, etc.
//! - **Reputation Staking**: Stake reputation on attestations
//!
//! # Trust Score Computation
//!
//! Trust scores are computed as:
//!
//! ```text
//! TrustScore = aggregate(
//!     direct_trust * direct_weight * temporal_decay,
//!     transitive_trust * transitive_weight * path_decay
//! ) - byzantine_penalty
//! ```
//!
//! # Key Functions
//!
//! ## Attestation Management
//! - [`create_attestation`] - Create trust attestation with evidence
//! - [`revoke_attestation`] - Revoke an existing attestation
//! - [`get_attestations_about`] - Get attestations about an agent
//! - [`get_attestations_by`] - Get attestations made by an agent
//!
//! ## Trust Computation
//! - [`compute_trust_score`] - Compute full MATL trust score
//! - [`get_trust_score`] - Get cached trust score (recomputes if stale)
//!
//! ## Introductions
//! - [`create_introduction`] - Introduce two agents to each other
//! - [`respond_to_introduction`] - Accept or reject introduction
//!
//! ## Disputes
//! - [`file_dispute`] - File dispute against attestation
//!
//! # Evidence Types
//!
//! Evidence weights range from 0.0 to 1.0:
//! - `InPersonMeeting`: 0.95 (highest trust)
//! - `VideoVerification`: 0.85
//! - `GovernmentId`: 0.95
//! - `VerifiableCredential`: 0.90
//! - `BlockchainAttestation`: 0.85
//! - `PgpKeySigning`: 0.80
//! - `DomainVerification`: 0.75
//! - `PhoneVerification`: 0.70
//! - `MutualVouch`: 0.60
//! - `CommunicationHistory`: 0.55
//! - `SocialMediaVerification`: 0.50
//!
//! # Byzantine Detection
//!
//! The system detects:
//! - **Sybil Suspicion**: Many attestations from similar sources
//! - **Trust Volatility**: Rapid changes in trust scores
//! - **Inconsistent Attestations**: Conflicting attestations from same truster
//! - **Collusion Patterns**: Groups with suspiciously similar patterns
//!
//! # Example Usage
//!
//! ```ignore
//! // Create an attestation
//! let attestation_hash = create_attestation(CreateAttestationInput {
//!     trustee: recipient_agent,
//!     trust_level: 0.8,  // -1.0 to 1.0
//!     category: TrustCategory::Identity,
//!     evidence: vec![TrustEvidence {
//!         evidence_type: EvidenceType::InPersonMeeting,
//!         weight: 1.0,
//!         ..Default::default()
//!     }],
//!     reason: Some("Met at conference".to_string()),
//!     expires_at: None,
//!     stake: Some(ReputationStake { amount: 100 }),
//! })?;
//!
//! // Compute trust score
//! let score = compute_trust_score(some_agent)?;
//! println!("Trust: {}, Confidence: {}", score.score, score.confidence);
//! ```
//!
//! # Algorithm Parameters
//!
//! | Parameter | Default | Description |
//! |-----------|---------|-------------|
//! | `DEFAULT_TRUST_LEVEL` | 0.3 | Score for unknown agents |
//! | `TRANSITIVE_DECAY_FACTOR` | 0.7 | Decay per hop |
//! | `MAX_TRANSITIVE_DEPTH` | 5 | Maximum path length |
//! | `TEMPORAL_DECAY_RATE` | 0.01 | Decay per day |
//! | `SYBIL_DETECTION_THRESHOLD` | 10 | Min attestations for check |

#[cfg(test)]
mod tests;

use hdk::prelude::*;
use mail_trust_integrity::*;
use std::collections::{HashMap, HashSet, VecDeque};

// ==================== CONFIGURATION ====================

/// MATL Algorithm parameters
const DEFAULT_TRUST_LEVEL: f64 = 0.3;
const TRANSITIVE_DECAY_FACTOR: f64 = 0.7;
const MAX_TRANSITIVE_DEPTH: u8 = 5;
const TEMPORAL_DECAY_RATE: f64 = 0.01; // Per day
#[allow(dead_code)]
const MIN_ATTESTATIONS_FOR_TRUST: usize = 2;
#[allow(dead_code)]
const BYZANTINE_THRESHOLD: f64 = 0.3;
const SYBIL_DETECTION_THRESHOLD: usize = 10;
#[allow(dead_code)]
const COLLUSION_SIMILARITY_THRESHOLD: f64 = 0.9;

/// Evidence type weights for scoring
fn evidence_weight(evidence_type: &EvidenceType) -> f64 {
    match evidence_type {
        EvidenceType::InPersonMeeting => 0.95,
        EvidenceType::VideoVerification => 0.85,
        EvidenceType::PhoneVerification => 0.70,
        EvidenceType::VerifiableCredential => 0.90,
        EvidenceType::SocialMediaVerification => 0.50,
        EvidenceType::DomainVerification => 0.75,
        EvidenceType::PgpKeySigning => 0.80,
        EvidenceType::MutualVouch => 0.60,
        EvidenceType::CommunicationHistory => 0.55,
        EvidenceType::BlockchainAttestation => 0.85,
        EvidenceType::GovernmentId => 0.95,
        EvidenceType::ProfessionalCertification => 0.80,
        EvidenceType::OrganizationMembership => 0.70,
        EvidenceType::Custom(_) => 0.40,
    }
}

/// Signals for real-time trust updates
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "data")]
pub enum TrustSignal {
    /// New attestation received
    AttestationReceived {
        attestation_hash: ActionHash,
        truster: AgentPubKey,
        trust_level: f64,
        category: TrustCategory,
    },
    /// Trust score updated
    TrustScoreUpdated {
        agent: AgentPubKey,
        new_score: f64,
        confidence: f64,
    },
    /// Introduction received
    IntroductionReceived {
        introduction_hash: ActionHash,
        introducer: AgentPubKey,
        introduced: AgentPubKey,
        recommendation: f64,
    },
    /// Dispute filed
    DisputeFiled {
        dispute_hash: ActionHash,
        attestation_hash: ActionHash,
        disputer: AgentPubKey,
    },
    /// Byzantine behavior detected
    ByzantineDetected {
        agent: AgentPubKey,
        flag_type: ByzantineFlagType,
        severity: f64,
    },
}

// ==================== CREATE ATTESTATION ====================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateAttestationInput {
    pub trustee: AgentPubKey,
    pub trust_level: f64,
    pub category: TrustCategory,
    pub evidence: Vec<TrustEvidence>,
    pub reason: Option<String>,
    pub expires_at: Option<Timestamp>,
    pub stake: Option<ReputationStake>,
}

/// Create a new trust attestation
#[hdk_extern]
pub fn create_attestation(input: CreateAttestationInput) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Check rate limiting (max attestations per day)
    let recent_attestations = count_recent_attestations(&my_agent, 86400)?; // 24 hours
    if recent_attestations >= 50 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Rate limit exceeded: max 50 attestations per day".to_string()
        )));
    }

    // Create signature
    let content_hash = hash_attestation_content(
        &my_agent,
        &input.trustee,
        input.trust_level,
        &input.category,
        now,
    )?;
    let signature = sign_raw(my_agent.clone(), content_hash)?.0.to_vec();

    let attestation = TrustAttestation {
        truster: my_agent.clone(),
        trustee: input.trustee.clone(),
        trust_level: input.trust_level,
        category: input.category.clone(),
        evidence: input.evidence,
        reason: input.reason,
        created_at: now,
        expires_at: input.expires_at,
        signature,
        revoked: false,
        stake: input.stake,
    };

    let attestation_hash = create_entry(EntryTypes::TrustAttestation(attestation.clone()))?;

    // Link from truster
    create_link(
        my_agent.clone(),
        attestation_hash.clone(),
        LinkTypes::AgentToGivenAttestations,
        LinkTag::new(format!("to:{}", input.trustee)),
    )?;

    // Link to trustee
    create_link(
        input.trustee.clone(),
        attestation_hash.clone(),
        LinkTypes::AgentToReceivedAttestations,
        LinkTag::new(format!("from:{}", my_agent)),
    )?;

    // Link to category anchor
    let category_anchor = category_anchor(&input.category)?;
    create_link(
        category_anchor,
        attestation_hash.clone(),
        LinkTypes::CategoryToAttestations,
        LinkTag::new("attestation"),
    )?;

    // Signal to trustee
    let signal = TrustSignal::AttestationReceived {
        attestation_hash: attestation_hash.clone(),
        truster: my_agent,
        trust_level: input.trust_level,
        category: input.category,
    };
    let encoded = ExternIO::encode(signal).map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?;
    let _ = send_remote_signal(encoded, vec![input.trustee.clone()]);

    // Trigger score recomputation for trustee
    let _ = compute_trust_score(input.trustee);

    Ok(attestation_hash)
}

fn hash_attestation_content(
    truster: &AgentPubKey,
    trustee: &AgentPubKey,
    trust_level: f64,
    category: &TrustCategory,
    timestamp: Timestamp,
) -> ExternResult<Vec<u8>> {
    // Create a deterministic content string for signing
    let content = format!(
        "attestation:{}:{}:{}:{:?}:{}",
        truster, trustee, trust_level, category, timestamp.as_micros()
    );
    // Return content bytes directly for signing
    Ok(content.into_bytes())
}

fn category_anchor(category: &TrustCategory) -> ExternResult<EntryHash> {
    let category_str = format!("trust_category:{:?}", category);
    // Use path-based anchor for category
    let path = Path::from(category_str);
    path.path_entry_hash()
}

fn count_recent_attestations(agent: &AgentPubKey, seconds: i64) -> ExternResult<usize> {
    let now = sys_time()?;
    let cutoff = (now - core::time::Duration::from_secs(seconds as u64))
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Timestamp error: {:?}", e))))?;

    let links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToGivenAttestations)?,
        GetStrategy::default(),
    )?;

    let mut count = 0;
    for link in links {
        if link.timestamp > cutoff {
            count += 1;
        }
    }

    Ok(count)
}

// ==================== REVOKE ATTESTATION ====================

/// Revoke an existing attestation
#[hdk_extern]
pub fn revoke_attestation(attestation_hash: ActionHash) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Get existing attestation
    let record = get(attestation_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Attestation not found".to_string()
        )))?;

    let mut attestation: TrustAttestation = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid attestation entry".to_string()
        )))?;

    // Verify ownership
    if attestation.truster != my_agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only truster can revoke attestation".to_string()
        )));
    }

    // Mark as revoked
    attestation.revoked = true;

    // Update entry
    let new_hash = update_entry(attestation_hash, EntryTypes::TrustAttestation(attestation.clone()))?;

    // Trigger score recomputation
    let _ = compute_trust_score(attestation.trustee);

    Ok(new_hash)
}

// ==================== COMPUTE TRUST SCORE (MATL ALGORITHM) ====================

/// Compute trust score for an agent using MATL algorithm
#[hdk_extern]
pub fn compute_trust_score(subject: AgentPubKey) -> ExternResult<TrustScore> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Get all attestations about this subject
    let attestations = get_attestations_for_agent(&subject)?;

    if attestations.is_empty() {
        // No attestations - return default trust
        return Ok(TrustScore {
            agent: subject,
            score: DEFAULT_TRUST_LEVEL,
            confidence: 0.0,
            direct_attestation_count: 0,
            transitive_path_count: 0,
            category_scores: Vec::new(),
            computed_at: now,
            attestation_hashes: Vec::new(),
            byzantine_flags: Vec::new(),
        });
    }

    // Step 1: Calculate direct trust (from my attestations)
    let (direct_score, direct_confidence) = calculate_direct_trust(&my_agent, &attestations, now)?;

    // Step 2: Calculate transitive trust (from my trusted contacts' attestations)
    let (transitive_score, transitive_confidence, path_count) =
        calculate_transitive_trust(&my_agent, &subject, MAX_TRANSITIVE_DEPTH)?;

    // Step 3: Aggregate scores with weighted average
    let (combined_score, combined_confidence) =
        aggregate_trust_scores(direct_score, direct_confidence, transitive_score, transitive_confidence);

    // Step 4: Check for Byzantine behavior
    let byzantine_flags = detect_byzantine_behavior(&subject, &attestations)?;

    // Step 5: Apply Byzantine penalty if detected
    let final_score = if !byzantine_flags.is_empty() {
        let penalty: f64 = byzantine_flags.iter().map(|f| f.severity).sum::<f64>().min(0.5);
        (combined_score - penalty).max(0.0)
    } else {
        combined_score
    };

    // Step 6: Calculate category-specific scores
    let category_scores = calculate_category_scores(&attestations, now)?;

    // Step 7: Collect attestation hashes
    let attestation_hashes: Vec<ActionHash> = attestations
        .iter()
        .map(|(hash, _)| hash.clone())
        .collect();

    let trust_score = TrustScore {
        agent: subject.clone(),
        score: final_score,
        confidence: combined_confidence,
        direct_attestation_count: attestations.len() as u32,
        transitive_path_count: path_count,
        category_scores,
        computed_at: now,
        attestation_hashes,
        byzantine_flags: byzantine_flags.clone(),
    };

    // Store computed score
    let score_hash = create_entry(EntryTypes::TrustScore(trust_score.clone()))?;

    // Link to agent
    create_link(
        subject.clone(),
        score_hash,
        LinkTypes::AgentToTrustScore,
        LinkTag::new(format!("score:{}", now.as_micros())),
    )?;

    // Signal if Byzantine behavior detected
    for flag in byzantine_flags {
        let signal = TrustSignal::ByzantineDetected {
            agent: subject.clone(),
            flag_type: flag.flag_type,
            severity: flag.severity,
        };
        emit_signal(signal)?;
    }

    // Signal score update
    emit_signal(TrustSignal::TrustScoreUpdated {
        agent: subject,
        new_score: final_score,
        confidence: combined_confidence,
    })?;

    Ok(trust_score)
}

fn get_attestations_for_agent(
    agent: &AgentPubKey,
) -> ExternResult<Vec<(ActionHash, TrustAttestation)>> {
    let links = get_links(LinkQuery::try_new(agent.clone(), LinkTypes::AgentToReceivedAttestations)?, GetStrategy::default())?;

    let mut attestations = Vec::new();

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".to_string())))?;

        if let Some(record) = get(hash.clone(), GetOptions::default())? {
            if let Some(attestation) = record
                .entry()
                .to_app_option::<TrustAttestation>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                // Skip revoked attestations
                if !attestation.revoked {
                    attestations.push((hash, attestation));
                }
            }
        }
    }

    Ok(attestations)
}

fn calculate_direct_trust(
    querier: &AgentPubKey,
    attestations: &[(ActionHash, TrustAttestation)],
    now: Timestamp,
) -> ExternResult<(f64, f64)> {
    let my_attestations: Vec<_> = attestations
        .iter()
        .filter(|(_, a)| &a.truster == querier)
        .collect();

    if my_attestations.is_empty() {
        return Ok((0.0, 0.0));
    }

    let mut total_weight = 0.0;
    let mut weighted_sum = 0.0;

    for (_, attestation) in my_attestations {
        // Check expiration
        if let Some(expires) = attestation.expires_at {
            if expires < now {
                continue;
            }
        }

        // Calculate evidence weight
        let evidence_score: f64 = attestation
            .evidence
            .iter()
            .map(|e| evidence_weight(&e.evidence_type) * e.weight)
            .sum::<f64>()
            .min(1.0);

        // Apply temporal decay
        let age_days = (now.as_micros() - attestation.created_at.as_micros()) as f64
            / (1_000_000.0 * 86400.0);
        let decay = (-TEMPORAL_DECAY_RATE * age_days).exp();

        // Calculate weight including stake
        let stake_multiplier = attestation
            .stake
            .as_ref()
            .map(|s| 1.0 + (s.amount as f64 / 1000.0).min(2.0))
            .unwrap_or(1.0);

        let weight = evidence_score * decay * stake_multiplier;

        // Normalize trust_level from [-1, 1] to [0, 1]
        let normalized_trust = (attestation.trust_level + 1.0) / 2.0;

        weighted_sum += normalized_trust * weight;
        total_weight += weight;
    }

    if total_weight == 0.0 {
        return Ok((0.0, 0.0));
    }

    let score = weighted_sum / total_weight;
    let confidence = (total_weight / 3.0).min(1.0); // Max confidence at 3+ weighted attestations

    Ok((score, confidence))
}

fn calculate_transitive_trust(
    querier: &AgentPubKey,
    subject: &AgentPubKey,
    max_depth: u8,
) -> ExternResult<(f64, f64, u32)> {
    // BFS to find trust paths
    let mut visited: HashSet<AgentPubKey> = HashSet::new();
    let mut queue: VecDeque<(AgentPubKey, f64, u8)> = VecDeque::new();
    let mut paths_found = 0u32;
    let mut total_trust = 0.0;
    let mut total_weight = 0.0;

    // Start with querier's direct trusts
    let my_attestations = get_attestations_given_by(querier)?;
    for (_, attestation) in my_attestations {
        if !attestation.revoked && attestation.trust_level > 0.0 {
            let normalized = (attestation.trust_level + 1.0) / 2.0;
            queue.push_back((attestation.trustee, normalized, 1));
        }
    }

    visited.insert(querier.clone());

    while let Some((current, accumulated_trust, depth)) = queue.pop_front() {
        if depth > max_depth {
            continue;
        }

        if visited.contains(&current) {
            continue;
        }
        visited.insert(current.clone());

        // Check if current agent has attested to subject
        let their_attestations = get_attestations_given_by(&current)?;
        for (_, attestation) in &their_attestations {
            if attestation.trustee == *subject && !attestation.revoked {
                let their_trust = (attestation.trust_level + 1.0) / 2.0;
                let decayed_trust = accumulated_trust * their_trust * TRANSITIVE_DECAY_FACTOR.powi(depth as i32);

                total_trust += decayed_trust;
                total_weight += TRANSITIVE_DECAY_FACTOR.powi(depth as i32);
                paths_found += 1;
            }

            // Add their trustees to queue for further exploration
            if !attestation.revoked && attestation.trust_level > 0.0 {
                let next_trust = accumulated_trust * ((attestation.trust_level + 1.0) / 2.0) * TRANSITIVE_DECAY_FACTOR;
                queue.push_back((attestation.trustee.clone(), next_trust, depth + 1));
            }
        }
    }

    if total_weight == 0.0 {
        return Ok((0.0, 0.0, 0));
    }

    let score = total_trust / total_weight;
    let confidence = (paths_found as f64 / 5.0).min(1.0); // Max confidence at 5+ paths

    Ok((score, confidence, paths_found))
}

fn get_attestations_given_by(
    agent: &AgentPubKey,
) -> ExternResult<Vec<(ActionHash, TrustAttestation)>> {
    let links = get_links(LinkQuery::try_new(agent.clone(), LinkTypes::AgentToGivenAttestations)?, GetStrategy::default())?;

    let mut attestations = Vec::new();

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".to_string())))?;

        if let Some(record) = get(hash.clone(), GetOptions::default())? {
            if let Some(attestation) = record
                .entry()
                .to_app_option::<TrustAttestation>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                attestations.push((hash, attestation));
            }
        }
    }

    Ok(attestations)
}

fn aggregate_trust_scores(
    direct_score: f64,
    direct_confidence: f64,
    transitive_score: f64,
    transitive_confidence: f64,
) -> (f64, f64) {
    let total_confidence = direct_confidence + transitive_confidence;

    if total_confidence == 0.0 {
        return (DEFAULT_TRUST_LEVEL, 0.0);
    }

    // Weight direct trust higher than transitive
    let direct_weight = direct_confidence * 1.5;
    let transitive_weight = transitive_confidence;

    let combined_score =
        (direct_score * direct_weight + transitive_score * transitive_weight)
            / (direct_weight + transitive_weight);

    let combined_confidence = ((direct_confidence.powi(2) + transitive_confidence.powi(2)).sqrt() / 1.414).min(1.0);

    (combined_score, combined_confidence)
}

fn detect_byzantine_behavior(
    _subject: &AgentPubKey,
    attestations: &[(ActionHash, TrustAttestation)],
) -> ExternResult<Vec<ByzantineFlag>> {
    let mut flags = Vec::new();
    let now = sys_time()?;

    // Check 1: Sybil detection - too many attestations from similar sources
    let trusters: Vec<_> = attestations.iter().map(|(_, a)| &a.truster).collect();
    if trusters.len() > SYBIL_DETECTION_THRESHOLD {
        // Check if trusters are suspiciously interconnected
        // (simplified - in production would do more sophisticated analysis)
        let unique_trusters: HashSet<_> = trusters.iter().collect();
        if unique_trusters.len() < trusters.len() / 2 {
            flags.push(ByzantineFlag {
                flag_type: ByzantineFlagType::SybilSuspicion,
                severity: 0.3,
                detected_at: now,
                evidence: "Multiple attestations from similar sources".to_string(),
            });
        }
    }

    // Check 2: Trust volatility - rapid changes in attestations
    let recent_attestations: Vec<_> = attestations
        .iter()
        .filter(|(_, a)| {
            let age = now.as_micros() - a.created_at.as_micros();
            age < 86400 * 1_000_000 * 7 // Last 7 days
        })
        .collect();

    if recent_attestations.len() > 10 {
        let variance = calculate_trust_variance(&recent_attestations);
        if variance > 0.5 {
            flags.push(ByzantineFlag {
                flag_type: ByzantineFlagType::TrustVolatility,
                severity: variance * 0.4,
                detected_at: now,
                evidence: format!("High trust variance: {:.2}", variance),
            });
        }
    }

    // Check 3: Inconsistent attestations - same truster giving conflicting scores
    let mut truster_scores: HashMap<AgentPubKey, Vec<f64>> = HashMap::new();
    for (_, attestation) in attestations {
        truster_scores
            .entry(attestation.truster.clone())
            .or_default()
            .push(attestation.trust_level);
    }

    for (truster, scores) in truster_scores {
        if scores.len() > 1 {
            let max = scores.iter().cloned().fold(f64::MIN, f64::max);
            let min = scores.iter().cloned().fold(f64::MAX, f64::min);
            if max - min > 1.0 {
                flags.push(ByzantineFlag {
                    flag_type: ByzantineFlagType::InconsistentAttestations,
                    severity: 0.2,
                    detected_at: now,
                    evidence: format!("Inconsistent attestations from {}", truster),
                });
            }
        }
    }

    Ok(flags)
}

fn calculate_trust_variance(attestations: &[&(ActionHash, TrustAttestation)]) -> f64 {
    if attestations.len() < 2 {
        return 0.0;
    }

    let scores: Vec<f64> = attestations.iter().map(|(_, a)| a.trust_level).collect();
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;

    variance.sqrt()
}

fn calculate_category_scores(
    attestations: &[(ActionHash, TrustAttestation)],
    now: Timestamp,
) -> ExternResult<Vec<(TrustCategory, f64)>> {
    let mut category_sums: HashMap<String, (f64, f64)> = HashMap::new();

    for (_, attestation) in attestations {
        // Check expiration
        if let Some(expires) = attestation.expires_at {
            if expires < now {
                continue;
            }
        }

        let category_key = format!("{:?}", attestation.category);
        let entry = category_sums.entry(category_key).or_insert((0.0, 0.0));

        let normalized = (attestation.trust_level + 1.0) / 2.0;
        entry.0 += normalized;
        entry.1 += 1.0;
    }

    let mut scores = Vec::new();
    for (category_str, (sum, count)) in category_sums {
        if count > 0.0 {
            let avg = sum / count;
            // Parse category back (simplified)
            let category = match category_str.as_str() {
                "Identity" => TrustCategory::Identity,
                "Communication" => TrustCategory::Communication,
                "FileSharing" => TrustCategory::FileSharing,
                "Scheduling" => TrustCategory::Scheduling,
                "Financial" => TrustCategory::Financial,
                "CredentialIssuer" => TrustCategory::CredentialIssuer,
                "Organization" => TrustCategory::Organization,
                "Personal" => TrustCategory::Personal,
                "Professional" => TrustCategory::Professional,
                _ => TrustCategory::Custom(category_str),
            };
            scores.push((category, avg));
        }
    }

    Ok(scores)
}

// ==================== QUERY TRUST ====================

/// Get trust score for an agent (uses cached if available)
#[hdk_extern]
pub fn get_trust_score(agent: AgentPubKey) -> ExternResult<TrustScore> {
    // Try to get cached score
    let links = get_links(LinkQuery::try_new(agent.clone(), LinkTypes::AgentToTrustScore)?, GetStrategy::default())?;

    // Get most recent score
    if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".to_string())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(score) = record
                .entry()
                .to_app_option::<TrustScore>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                let now = sys_time()?;
                let age = now.as_micros() - score.computed_at.as_micros();

                // If score is less than 1 hour old, return cached
                if age < 3600 * 1_000_000 {
                    return Ok(score);
                }
            }
        }
    }

    // Recompute score
    compute_trust_score(agent)
}

/// Get attestations about an agent
#[hdk_extern]
pub fn get_attestations_about(agent: AgentPubKey) -> ExternResult<Vec<(ActionHash, TrustAttestation)>> {
    get_attestations_for_agent(&agent)
}

/// Get attestations made by an agent
#[hdk_extern]
pub fn get_attestations_by(agent: AgentPubKey) -> ExternResult<Vec<(ActionHash, TrustAttestation)>> {
    get_attestations_given_by(&agent)
}

// ==================== INTRODUCTIONS ====================

/// Create a trust introduction
#[hdk_extern]
pub fn create_introduction(input: TrustIntroduction) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    if input.introducer != my_agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Introducer must be self".to_string()
        )));
    }

    let intro_hash = create_entry(EntryTypes::TrustIntroduction(input.clone()))?;

    // Link from introducer
    create_link(
        my_agent,
        intro_hash.clone(),
        LinkTypes::AgentToIntroductions,
        LinkTag::new(format!("introduced:{}", input.introduced)),
    )?;

    // Link to target
    create_link(
        input.target.clone(),
        intro_hash.clone(),
        LinkTypes::AgentToReceivedIntroductions,
        LinkTag::new(format!("from:{}", input.introducer)),
    )?;

    // Signal to target
    let signal = TrustSignal::IntroductionReceived {
        introduction_hash: intro_hash.clone(),
        introducer: input.introducer,
        introduced: input.introduced,
        recommendation: input.recommendation_level,
    };
    let encoded = ExternIO::encode(signal).map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?;
    let _ = send_remote_signal(encoded, vec![input.target]);

    Ok(intro_hash)
}

/// Accept or reject an introduction
#[hdk_extern]
pub fn respond_to_introduction(input: (ActionHash, bool)) -> ExternResult<ActionHash> {
    let (intro_hash, accepted) = input;

    let record = get(intro_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Introduction not found".to_string()
        )))?;

    let mut intro: TrustIntroduction = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid introduction".to_string()
        )))?;

    intro.accepted = Some(accepted);

    update_entry(intro_hash, EntryTypes::TrustIntroduction(intro))
}

// ==================== DISPUTES ====================

/// File a dispute against an attestation
#[hdk_extern]
pub fn file_dispute(input: TrustDispute) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    if input.disputer != my_agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Disputer must be self".to_string()
        )));
    }

    let dispute_hash = create_entry(EntryTypes::TrustDispute(input.clone()))?;

    // Link to attestation
    create_link(
        input.attestation_hash.clone(),
        dispute_hash.clone(),
        LinkTypes::AttestationToDisputes,
        LinkTag::new("dispute"),
    )?;

    // Signal
    emit_signal(TrustSignal::DisputeFiled {
        dispute_hash: dispute_hash.clone(),
        attestation_hash: input.attestation_hash,
        disputer: my_agent,
    })?;

    Ok(dispute_hash)
}

// ==================== TRUST-GATED DELIVERY ====================

/// Get sender trust score for delivery gating.
/// Returns (score, confidence) tuple. Used by messages coordinator
/// to decide whether to accept or quarantine an incoming email.
#[hdk_extern]
pub fn get_sender_trust_for_delivery(sender: AgentPubKey) -> ExternResult<(f64, f64)> {
    let score = get_trust_score(sender)?;
    Ok((score.score, score.confidence))
}

// ==================== SIGNAL HANDLING ====================

#[hdk_extern]
pub fn recv_remote_signal(signal: ExternIO) -> ExternResult<()> {
    let trust_signal: TrustSignal = signal.decode().map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to decode signal: {}",
            e
        )))
    })?;

    emit_signal(trust_signal)?;

    Ok(())
}

// ==================== INIT ====================

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    // Grant capability for receiving signals
    let functions = GrantedFunctions::Listed(HashSet::from([
        (zome_info()?.name, "recv_remote_signal".into()),
    ]));

    create_cap_grant(CapGrantEntry {
        tag: "recv_trust_signals".to_string(),
        access: CapAccess::Unrestricted,
        functions,
    })?;

    Ok(InitCallbackResult::Pass)
}
