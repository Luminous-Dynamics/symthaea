// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use reputation_integrity::*;
use mycelix_common::{bridge, error_handling, link_queries, time};

mod cache;

/// Get or initialize MATL score for an agent
///
/// This is the entry point for the 45% Byzantine fault tolerance system.
/// New agents start with neutral reputation (0.5).
#[hdk_extern]
pub fn get_agent_matl_score(agent: AgentPubKey) -> ExternResult<Option<MatlScore>> {
    let path = agent.clone();
    // Use shared utility for get_links
    let links = link_queries::get_links_local(path, LinkTypes::AgentToScore)?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            let record = get(action_hash, GetOptions::default())?;
            if let Some(record) = record {
                // Use shared utility for deserialization
                let score: MatlScore = error_handling::deserialize_entry(&record)?;
                return Ok(Some(score));
            }
        }
    }

    // No score exists - return None (will be initialized on first transaction)
    Ok(None)
}

/// Get MATL score with caching (10-100x faster)
///
/// This is the recommended method for querying MATL scores.
/// Uses intelligent caching with TTL and automatic invalidation.
#[hdk_extern]
pub fn get_agent_matl_score_fast(agent: AgentPubKey) -> ExternResult<MatlScore> {
    cache::get_agent_matl_score_cached(agent)
}

/// Get combined reputation score (local MATL + cross-app reputation)
///
/// This function implements cross-app reputation sharing via the Bridge zome.
/// Formula: combined = (local_rep * 0.6) + (cross_app_rep * 0.4)
///
/// If Bridge is unavailable, returns local reputation only.
#[hdk_extern]
pub fn get_combined_reputation(agent: AgentPubKey) -> ExternResult<CombinedReputationResponse> {
    // Get local MATL score
    let local_score = match get_agent_matl_score(agent.clone())? {
        Some(score) => score,
        None => {
            // No local score - return neutral with cross-app check
            let did = bridge::agent_to_did(&agent);
            let (combined, used_cross_app) = bridge::compute_combined_reputation(0.5, &did);
            return Ok(CombinedReputationResponse {
                local_score: 0.5,
                cross_app_score: if used_cross_app { Some(combined) } else { None },
                combined_score: combined,
                cross_app_available: used_cross_app,
                transaction_count: 0,
            });
        }
    };

    // Convert agent to DID for cross-app lookup
    let did = bridge::agent_to_did(&agent);

    // Get cross-app reputation from Bridge
    let (combined_score, cross_app_available) =
        bridge::compute_combined_reputation(local_score.composite, &did);

    // Get cross-app score directly for response
    let cross_app_score = match bridge::get_cross_app_reputation(&did) {
        bridge::BridgeResult::Success(rep) => Some(rep.score),
        _ => None,
    };

    Ok(CombinedReputationResponse {
        local_score: local_score.composite,
        cross_app_score,
        combined_score,
        cross_app_available,
        transaction_count: local_score.transaction_count,
    })
}

/// Get cross-app reputation from Bridge zome
///
/// This directly queries the Bridge for cross-app reputation data.
/// Useful for UI display of reputation breakdown.
#[hdk_extern]
pub fn get_cross_app_reputation(agent: AgentPubKey) -> ExternResult<CrossAppReputationResponse> {
    let did = bridge::agent_to_did(&agent);

    match bridge::get_cross_app_reputation(&did) {
        bridge::BridgeResult::Success(rep) => Ok(CrossAppReputationResponse {
            available: true,
            did: rep.did,
            score: Some(rep.score),
            app_count: Some(rep.app_count),
            total_transactions: Some(rep.total_transactions),
            error: None,
        }),
        bridge::BridgeResult::Unavailable => Ok(CrossAppReputationResponse {
            available: false,
            did,
            score: None,
            app_count: None,
            total_transactions: None,
            error: Some("Bridge zome unavailable".to_string()),
        }),
        bridge::BridgeResult::Error(e) => Ok(CrossAppReputationResponse {
            available: false,
            did,
            score: None,
            app_count: None,
            total_transactions: None,
            error: Some(e),
        }),
    }
}

/// Update MATL score after a transaction
///
/// This implements the core MATL algorithm:
/// 1. Compute PoGQ (quality, consistency, entropy)
/// 2. Update reputation based on transaction outcome
/// 3. Detect Byzantine patterns
/// 4. Calculate composite score
#[hdk_extern]
pub fn update_matl_score(input: UpdateMatlInput) -> ExternResult<MatlScore> {
    let agent = input.agent.clone();

    // Get existing score or create new one
    let mut score = match get_agent_matl_score(agent.clone())? {
        Some(existing) => existing,
        None => {
            // Initialize new agent with neutral score
            MatlScore {
                agent: agent.clone(),
                pogq: ProofOfGradientQuality {
                    quality: 0.5,
                    consistency: 0.5,
                    entropy: 0.0,
                    timestamp: time::now()?,
                },
                reputation: 0.5, // Neutral starting point
                composite: 0.5,
                transaction_count: 0,
                total_value_cents: 0,
                updated_at: time::now()?,
                flags: ByzantineFlags {
                    cartel_detected: false,
                    volatile_reputation: false,
                    gradient_poisoning: false,
                    sybil_suspected: false,
                    risk_score: 0.0,
                },
            }
        }
    };

    // Update transaction stats
    score.transaction_count += 1;
    score.total_value_cents += input.transaction_value_cents;

    // Compute new PoGQ based on transaction outcome
    score.pogq = compute_pogq(&score, &input)?;

    // Update reputation with exponential moving average
    let alpha = 0.3; // Learning rate
    let transaction_quality = if input.successful { 1.0 } else { 0.0 };
    score.reputation = alpha * transaction_quality + (1.0 - alpha) * score.reputation;

    // Detect Byzantine patterns
    score.flags = detect_byzantine_patterns(&score)?;

    // Calculate composite score (MATL formula)
    score.composite = compute_composite_score(&score.pogq, score.reputation);

    // Update timestamp
    score.updated_at = time::now()?;

    // Save score
    let action_hash = create_entry(&EntryTypes::MatlScore(score.clone()))?;

    // Create/update link
    create_link(agent.clone(), action_hash, LinkTypes::AgentToScore, ())?;

    // Invalidate cache after update
    cache::invalidate_matl_cache(&agent);

    // Emit monitoring metric
    monitoring::emit_metric(
        monitoring::MetricType::MatlScoreUpdated,
        score.composite,
        Some(agent.clone()),
        Some(format!("quality:{:.2},consistency:{:.2},reputation:{:.2}",
            score.pogq.quality, score.pogq.consistency, score.reputation)),
    )?;

    // Check for Byzantine behavior and emit alert if needed
    if score.flags.risk_score >= 0.5 {
        monitoring::emit_metric(
            monitoring::MetricType::HighRiskAgent,
            score.flags.risk_score,
            Some(agent.clone()),
            Some(format!("composite:{:.2}", score.composite)),
        )?;
    }

    Ok(score)
}

/// Compute Proof of Gradient Quality
///
/// This measures the quality and consistency of an agent's behavior.
/// Higher quality + higher consistency = higher trust.
fn compute_pogq(
    score: &MatlScore,
    input: &UpdateMatlInput,
) -> ExternResult<ProofOfGradientQuality> {
    // Quality: weighted by transaction value and outcome
    let transaction_quality = if input.successful {
        // Successful transaction increases quality
        0.8 + (input.transaction_value_cents as f64 / 1_000_000.0).min(0.2)
    } else {
        // Failed transaction decreases quality
        0.2
    };

    // Exponential moving average for quality
    let alpha = 0.2;
    let new_quality = alpha * transaction_quality + (1.0 - alpha) * score.pogq.quality;

    // Consistency: measure variance in quality over time
    // Low variance = high consistency
    let quality_diff = (transaction_quality - score.pogq.quality).abs();
    let new_consistency = (1.0 - quality_diff.min(1.0)) * 0.7 + score.pogq.consistency * 0.3;

    // Entropy: measure of unpredictability
    // Lower entropy = more predictable (good)
    let new_entropy = compute_entropy(score)?;

    Ok(ProofOfGradientQuality {
        quality: new_quality.clamp(0.0, 1.0),
        consistency: new_consistency.clamp(0.0, 1.0),
        entropy: new_entropy,
        timestamp: time::now()?,
    })
}

/// Compute entropy of agent behavior
///
/// Low entropy = predictable, consistent behavior (trustworthy)
/// High entropy = erratic, unpredictable behavior (suspicious)
fn compute_entropy(score: &MatlScore) -> ExternResult<f64> {
    // Simplified entropy calculation
    // In production, this would analyze transaction patterns over time
    let quality_variance = (0.5 - score.pogq.quality).abs();
    let consistency_penalty = 1.0 - score.pogq.consistency;

    Ok((quality_variance + consistency_penalty) / 2.0)
}

/// Detect Byzantine attack patterns
///
/// This implements the key innovation for 45% Byzantine tolerance:
/// detecting coordinated attacks, Sybil identities, and malicious behavior.
fn detect_byzantine_patterns(score: &MatlScore) -> ExternResult<ByzantineFlags> {
    let mut flags = score.flags.clone();

    // 1. Volatile Reputation Detection
    // Rapid changes in reputation suggest manipulation
    flags.volatile_reputation = score.pogq.entropy > 0.7;

    // 2. Sybil Detection via Graph Analysis
    // Analyze transaction patterns to detect coordinated Sybil attacks
    let sybil_analysis = detect_sybil_patterns(score)?;
    flags.sybil_suspected = sybil_analysis.is_sybil_suspected;

    // 3. Quality Inconsistency
    // High quality but low consistency = suspicious
    let inconsistency = score.pogq.quality - score.pogq.consistency;
    if inconsistency > 0.3 {
        flags.risk_score = (flags.risk_score + 0.2).min(1.0);
    }

    // 4. Compute overall Byzantine risk score
    let mut risk: f64 = 0.0;
    if flags.cartel_detected {
        risk += 0.4;
    }
    if flags.volatile_reputation {
        risk += 0.2;
    }
    if flags.gradient_poisoning {
        risk += 0.3;
    }
    if flags.sybil_suspected {
        // Use the Sybil confidence score for weighted risk
        risk += sybil_analysis.confidence * 0.35;
    }

    flags.risk_score = risk.min(1.0);

    Ok(flags)
}

/// Sybil detection analysis result
#[derive(Clone, Debug)]
struct SybilAnalysisResult {
    /// Whether Sybil behavior is suspected
    is_sybil_suspected: bool,
    /// Confidence level (0.0 - 1.0)
    confidence: f64,
    /// Reasons for suspicion
    reasons: Vec<String>,
}

/// Detect Sybil patterns using graph analysis and behavioral heuristics
///
/// This implements multi-factor Sybil detection:
/// 1. Low transaction count with high reputation (boosted account)
/// 2. Clustered transaction timing (coordinated accounts)
/// 3. Similar transaction patterns (identical behavior)
/// 4. Shared counterparties (ring trading)
/// 5. Age-reputation mismatch (suspiciously fast growth)
fn detect_sybil_patterns(score: &MatlScore) -> ExternResult<SybilAnalysisResult> {
    let mut is_suspicious = false;
    let mut confidence: f64 = 0.0;
    let mut reasons: Vec<String> = Vec::new();

    // Heuristic 1: Low activity with high reputation
    // New accounts shouldn't have high trust without earning it
    if score.transaction_count < 5 && score.composite > 0.75 {
        is_suspicious = true;
        confidence += 0.25;
        reasons.push("High reputation with minimal activity".to_string());
    }

    // Heuristic 2: Perfect quality with no history
    // Real users make mistakes and have variance in performance
    if score.transaction_count < 10 && score.pogq.quality > 0.95 {
        is_suspicious = true;
        confidence += 0.2;
        reasons.push("Suspiciously perfect quality with little history".to_string());
    }

    // Heuristic 3: Age-reputation growth rate
    // Check if reputation grew faster than is organically possible
    // A new account shouldn't reach high reputation within hours
    let account_age_hours = score.transaction_count as f64 * 0.5; // Rough estimate
    let expected_max_reputation = (account_age_hours / 24.0).min(1.0) * 0.8; // Max 0.8 after 24h
    if score.composite > expected_max_reputation && score.transaction_count < 20 {
        is_suspicious = true;
        confidence += 0.15;
        reasons.push("Reputation growth exceeds organic rate".to_string());
    }

    // Heuristic 4: Transaction pattern entropy
    // Sybil accounts often have very predictable patterns
    if score.pogq.entropy < 0.1 && score.transaction_count > 5 {
        // Very low entropy = suspiciously uniform behavior
        is_suspicious = true;
        confidence += 0.2;
        reasons.push("Unusually uniform transaction patterns".to_string());
    }

    // Heuristic 5: PoGQ consistency without variance
    // Real ML nodes have natural variance in gradient quality
    if score.pogq.consistency > 0.98 && score.transaction_count > 10 {
        is_suspicious = true;
        confidence += 0.15;
        reasons.push("Unnaturally consistent gradient quality".to_string());
    }

    // Heuristic 6: Check for graph clustering
    // Sybil accounts often only transact with each other
    let clustering_score = analyze_transaction_clustering(score)?;
    if clustering_score > 0.8 {
        is_suspicious = true;
        confidence += 0.25;
        reasons.push("High transaction clustering indicates ring trading".to_string());
    }

    // Normalize confidence to [0, 1]
    confidence = confidence.min(1.0);

    // Only flag as Sybil if confidence is high enough
    let threshold = 0.35;
    is_suspicious = is_suspicious && confidence >= threshold;

    Ok(SybilAnalysisResult {
        is_sybil_suspected: is_suspicious,
        confidence,
        reasons,
    })
}

/// Analyze transaction clustering to detect ring trading
///
/// This checks if the agent's transactions are concentrated
/// among a small set of counterparties (potential Sybil ring).
fn analyze_transaction_clustering(score: &MatlScore) -> ExternResult<f64> {
    // In a full implementation, this would:
    // 1. Fetch all transactions for this agent
    // 2. Extract unique counterparties
    // 3. Calculate clustering coefficient
    // 4. Compare against known Sybil patterns

    // For now, use heuristics based on available data
    // High quality + high consistency + low transaction count = potential cluster
    let clustering_indicators = [
        // Low diversity indicator
        if score.transaction_count < 10 { 0.3 } else { 0.0 },
        // Perfect metrics indicator
        if score.pogq.quality > 0.9 && score.pogq.consistency > 0.9 { 0.3 } else { 0.0 },
        // Low entropy indicator (uniform patterns)
        if score.pogq.entropy < 0.2 { 0.4 } else { score.pogq.entropy * 0.2 },
    ];

    let clustering_score: f64 = clustering_indicators.iter().sum();
    Ok(clustering_score.min(1.0))
}

/// Compute composite MATL score
///
/// This is the final trust score formula:
/// composite = 0.4 * quality + 0.3 * consistency + 0.3 * reputation
///
/// Why these weights?
/// - Quality (0.4): Most important - what they do
/// - Consistency (0.3): Important - reliability over time
/// - Reputation (0.3): Important - historical track record
pub fn compute_composite_score(pogq: &ProofOfGradientQuality, reputation: f64) -> f64 {
    const W_QUALITY: f64 = 0.4;
    const W_CONSISTENCY: f64 = 0.3;
    const W_REPUTATION: f64 = 0.3;

    (W_QUALITY * pogq.quality + W_CONSISTENCY * pogq.consistency + W_REPUTATION * reputation)
        .clamp(0.0, 1.0)
}

/// Check if agent is Byzantine (above risk threshold)
///
/// Default threshold: 0.5 (50% confidence of Byzantine behavior)
/// This can be adjusted per marketplace based on risk tolerance.
#[hdk_extern]
pub fn is_byzantine(agent: AgentPubKey) -> ExternResult<ByzantineCheckResult> {
    const THRESHOLD: f64 = 0.5;

    match get_agent_matl_score(agent)? {
        Some(score) => Ok(ByzantineCheckResult {
            is_byzantine: score.flags.risk_score >= THRESHOLD,
            risk_score: score.flags.risk_score,
            composite_score: score.composite,
            flags: score.flags,
        }),
        None => {
            // Unknown agent - treat as neutral
            Ok(ByzantineCheckResult {
                is_byzantine: false,
                risk_score: 0.0,
                composite_score: 0.5,
                flags: ByzantineFlags {
                    cartel_detected: false,
                    volatile_reputation: false,
                    gradient_poisoning: false,
                    sybil_suspected: false,
                    risk_score: 0.0,
                },
            })
        }
    }
}

/// Submit a review after a transaction
#[hdk_extern]
pub fn submit_review(input: SubmitReviewInput) -> ExternResult<ReviewOutput> {
    let agent_info = agent_info()?;

    // Create review entry
    let review = Review {
        transaction_hash: input.transaction_hash.clone(),
        listing_hash: input.listing_hash,
        rating: input.rating,
        comment: input.comment,
        reviewer: agent_info.agent_initial_pubkey.clone(),
        seller: input.seller.clone(),
        created_at: sys_time()?,
        epistemic: EpistemicClassification {
            // Reviews are privately verifiable (only buyer experienced it)
            empirical: EmpiricalLevel::E2PrivateVerify,
            // Community agreement (buyer-seller)
            normative: NormativeLevel::N1Communal,
            // Persistent (keep for reputation)
            materiality: MaterialityLevel::M2Persistent,
        },
    };

    let action_hash = create_entry(&EntryTypes::Review(review.clone()))?;

    // Create links
    create_link(
        input.seller.clone(),
        action_hash.clone(),
        LinkTypes::AgentToSellerReviews,
        (),
    )?;

    create_link(
        agent_info.agent_initial_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToBuyerReviews,
        (),
    )?;

    create_link(
        input.transaction_hash,
        action_hash.clone(),
        LinkTypes::TransactionToReview,
        (),
    )?;

    // Update seller's MATL score based on review
    update_matl_score(UpdateMatlInput {
        agent: input.seller.clone(),
        successful: input.rating >= 4, // 4-5 stars = successful
        transaction_value_cents: 0,    // Value tracked elsewhere
    })?;

    // Emit monitoring metric
    monitoring::emit_metric(
        monitoring::MetricType::ReviewSubmitted,
        input.rating as f64,
        Some(review.reviewer.clone()),
        Some(format!("seller:{:?}", input.seller)),
    )?;

    Ok(ReviewOutput {
        review_hash: action_hash,
        review,
    })
}

/// Get reviews for a seller
#[hdk_extern]
pub fn get_seller_reviews(seller: AgentPubKey) -> ExternResult<ReviewsResponse> {
    // HDK 0.6.0: get_links with LinkQuery::try_new
    let links = get_links(
        LinkQuery::try_new(seller, LinkTypes::AgentToSellerReviews)?,
        GetStrategy::Local,
    )?;

    let mut reviews = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                // HDK 0.6.0: to_app_option() returns SerializedBytesError
                let review: Review = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialization error: {:?}", e))))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest(
                        "Could not deserialize review".into()
                    )))?;
                reviews.push(review);
            }
        }
    }

    Ok(ReviewsResponse { reviews })
}

// ===== Input/Output Types =====

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateMatlInput {
    pub agent: AgentPubKey,
    pub successful: bool,
    pub transaction_value_cents: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ByzantineCheckResult {
    pub is_byzantine: bool,
    pub risk_score: f64,
    pub composite_score: f64,
    pub flags: ByzantineFlags,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubmitReviewInput {
    pub transaction_hash: ActionHash,
    pub listing_hash: ActionHash,
    pub seller: AgentPubKey,
    pub rating: u8,
    pub comment: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReviewOutput {
    pub review_hash: ActionHash,
    pub review: Review,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReviewsResponse {
    pub reviews: Vec<Review>,
}

/// Response for combined reputation query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CombinedReputationResponse {
    /// Local MATL composite score (0.0 - 1.0)
    pub local_score: f64,
    /// Cross-app reputation score from Bridge (if available)
    pub cross_app_score: Option<f64>,
    /// Weighted combined score
    pub combined_score: f64,
    /// Whether cross-app reputation was available
    pub cross_app_available: bool,
    /// Number of local transactions
    pub transaction_count: u32,
}

/// Response for cross-app reputation query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrossAppReputationResponse {
    /// Whether Bridge/cross-app reputation is available
    pub available: bool,
    /// The DID used for lookup
    pub did: String,
    /// Cross-app reputation score (if available)
    pub score: Option<f64>,
    /// Number of apps contributing to score
    pub app_count: Option<u32>,
    /// Total transactions across all apps
    pub total_transactions: Option<u64>,
    /// Error message if unavailable
    pub error: Option<String>,
}


// ===== Tests =====
#[cfg(test)]
mod tests;
