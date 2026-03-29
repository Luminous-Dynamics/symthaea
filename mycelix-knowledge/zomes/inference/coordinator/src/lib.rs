// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Inference Coordinator Zome
//! Business logic for ML-powered knowledge inference
//!
//! Updated to use HDK 0.6 patterns (LinkQuery, GetStrategy, Anchor pattern)

use hdk::prelude::*;
use inference_integrity::*;

/// Helper to get an anchor entry hash for link bases
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Create an inference
#[hdk_extern]
pub fn create_inference(inference: Inference) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Inference(inference.clone()))?;

    // Index by inference ID for O(1) lookup
    let id_anchor = format!("inference_id:{}", inference.id);
    create_entry(&EntryTypes::Anchor(Anchor(id_anchor.clone())))?;
    create_link(
        anchor_hash(&id_anchor)?,
        action_hash.clone(),
        LinkTypes::InferenceIdToInference,
        (),
    )?;

    // Link each source claim to this inference using anchor pattern
    for claim_id in &inference.source_claims {
        let claim_anchor = format!("claim:{}", claim_id);
        create_entry(&EntryTypes::Anchor(Anchor(claim_anchor.clone())))?;
        create_link(
            anchor_hash(&claim_anchor)?,
            action_hash.clone(),
            LinkTypes::ClaimToInference,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find inference".into()
        )))
}

/// Get inferences for a claim
#[hdk_extern]
pub fn get_claim_inferences(claim_id: String) -> ExternResult<Vec<Record>> {
    let claim_anchor = format!("claim:{}", claim_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&claim_anchor)?, LinkTypes::ClaimToInference)?,
        GetStrategy::default(),
    )?;

    let mut inferences = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            inferences.push(record);
        }
    }

    Ok(inferences)
}

/// Assess credibility of a claim
#[hdk_extern]
pub fn assess_credibility(input: AssessCredibilityInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let score_id = format!("credibility:{}:{}", input.subject, now.as_micros());

    // Calculate component scores based on available data
    let components = calculate_credibility_components(&input.subject, &input.subject_type)?;

    // Calculate overall score as weighted average
    let overall = (components.accuracy * 0.25
        + components.consistency * 0.2
        + components.transparency * 0.2
        + components.track_record * 0.2
        + components.corroboration * 0.15)
        .min(1.0)
        .max(0.0);

    let score = CredibilityScore {
        id: score_id,
        subject: input.subject.clone(),
        subject_type: input.subject_type,
        score: overall,
        components,
        factors: vec![], // Would be populated by actual analysis
        assessed_at: now,
        expires_at: input.expires_at,
    };

    let action_hash = create_entry(&EntryTypes::CredibilityScore(score))?;

    // Link subject to credibility score using anchor pattern
    let subject_anchor = format!("subject:{}", input.subject);
    create_entry(&EntryTypes::Anchor(Anchor(subject_anchor.clone())))?;
    create_link(
        anchor_hash(&subject_anchor)?,
        action_hash.clone(),
        LinkTypes::SubjectToCredibility,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find credibility score".into()
        )))
}

/// Input for credibility assessment
#[derive(Serialize, Deserialize, Debug)]
pub struct AssessCredibilityInput {
    pub subject: String,
    pub subject_type: CredibilitySubjectType,
    pub expires_at: Option<Timestamp>,
}

/// Calculate credibility components (simplified implementation)
fn calculate_credibility_components(
    _subject: &str,
    _subject_type: &CredibilitySubjectType,
) -> ExternResult<CredibilityComponents> {
    // In a real implementation, this would:
    // - Check verification status
    // - Compare with other claims
    // - Analyze source transparency
    // - Look up author history
    // - Count corroborating sources

    // For now, return default scores
    Ok(CredibilityComponents {
        accuracy: 0.5,
        consistency: 0.5,
        transparency: 0.5,
        track_record: 0.5,
        corroboration: 0.5,
    })
}

/// Get credibility score for a subject
#[hdk_extern]
pub fn get_credibility(subject: String) -> ExternResult<Option<Record>> {
    let subject_anchor = format!("subject:{}", subject);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&subject_anchor)?, LinkTypes::SubjectToCredibility)?,
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get the most recent assessment
    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

/// Detect patterns in the knowledge graph
#[hdk_extern]
pub fn detect_patterns(input: DetectPatternsInput) -> ExternResult<Vec<Record>> {
    let now = sys_time()?;
    let mut patterns = Vec::new();

    // Pattern detection would involve:
    // - Clustering similar claims
    // - Detecting temporal trends
    // - Finding causal chains
    // - Identifying contradictions

    // For now, we'll create a simple cluster pattern if enough related claims exist
    if input.claims.len() >= 2 {
        let pattern_id = format!("pattern:cluster:{}", now.as_micros());

        let pattern = Pattern {
            id: pattern_id,
            pattern_type: PatternType::Cluster,
            description: format!("Cluster of {} related claims", input.claims.len()),
            claims: input.claims.clone(),
            strength: calculate_cluster_strength(&input.claims),
            detected_at: now,
        };

        let action_hash = create_entry(&EntryTypes::Pattern(pattern))?;

        // Link pattern type to pattern using anchor pattern
        let type_anchor = "pattern_type:Cluster".to_string();
        create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
        create_link(
            anchor_hash(&type_anchor)?,
            action_hash.clone(),
            LinkTypes::TypeToPattern,
            (),
        )?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            patterns.push(record);
        }
    }

    Ok(patterns)
}

/// Input for pattern detection
#[derive(Serialize, Deserialize, Debug)]
pub struct DetectPatternsInput {
    pub claims: Vec<String>,
    pub pattern_types: Option<Vec<PatternType>>,
}

/// Calculate cluster strength based on claims
fn calculate_cluster_strength(claims: &[String]) -> f64 {
    // Simple heuristic: more claims = stronger cluster
    let count = claims.len() as f64;
    (count / 10.0).min(1.0)
}

/// Find contradictions between claims
#[hdk_extern]
pub fn find_contradictions(claim_ids: Vec<String>) -> ExternResult<Vec<Record>> {
    let now = sys_time()?;
    let mut contradictions = Vec::new();

    // In a real implementation, this would:
    // - Parse claim content
    // - Use NLP to detect semantic contradictions
    // - Check logical consistency
    // - Verify against external sources

    // For demonstration, create a contradiction inference if claims exist
    if claim_ids.len() >= 2 {
        let inference_id = format!("inference:contradiction:{}", now.as_micros());

        let inference = Inference {
            id: inference_id,
            inference_type: InferenceType::Contradiction,
            source_claims: claim_ids,
            conclusion: "Potential contradiction detected between claims".to_string(),
            confidence: 0.5, // Low confidence without actual analysis
            reasoning: "Claims may contain contradictory statements".to_string(),
            model: "simple_contradiction_detector_v1".to_string(),
            created: now,
            verified: false,
        };

        let action_hash = create_entry(&EntryTypes::Inference(inference))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            contradictions.push(record);
        }
    }

    Ok(contradictions)
}

/// Verify an inference (human review)
#[hdk_extern]
pub fn verify_inference(input: VerifyInferenceInput) -> ExternResult<Record> {
    // O(1) lookup via inference ID anchor index
    let id_anchor = format!("inference_id:{}", input.inference_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&id_anchor)?, LinkTypes::InferenceIdToInference)?,
        GetStrategy::default(),
    )?;

    let current_record = if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        get(action_hash, GetOptions::default())?
    } else {
        None
    }
    .ok_or(wasm_error!(WasmErrorInner::Guest(
        "Inference not found".into()
    )))?;

    let current_inference: Inference = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid inference entry".into()
        )))?;

    // Update to verified
    let verified_inference = Inference {
        id: current_inference.id,
        inference_type: current_inference.inference_type,
        source_claims: current_inference.source_claims,
        conclusion: current_inference.conclusion,
        confidence: if input.is_correct {
            (current_inference.confidence + 0.2).min(1.0)
        } else {
            (current_inference.confidence - 0.3).max(0.0)
        },
        reasoning: current_inference.reasoning,
        model: current_inference.model,
        created: current_inference.created,
        verified: true,
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::Inference(verified_inference),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find verified inference".into()
        )))
}

/// Input for verifying an inference
#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyInferenceInput {
    pub inference_id: String,
    pub is_correct: bool,
    pub verifier_did: String,
    pub comment: Option<String>,
}

// ============================================================================
// MATL-ENHANCED CREDIBILITY FUNCTIONS
// ============================================================================

/// Input for enhanced credibility assessment
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EnhancedCredibilityInput {
    /// Subject to assess
    pub subject: String,

    /// Subject type
    pub subject_type: CredibilitySubjectType,

    /// Optional MATL data from markets
    pub matl_data: Option<MatlInputData>,

    /// Include evidence analysis
    pub include_evidence: bool,

    /// Include author reputation
    pub include_author: bool,

    /// Assessment expiration
    pub expires_at: Option<Timestamp>,
}

/// MATL data from external sources (Epistemic Markets)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MatlInputData {
    /// Quality score from MATL
    pub quality: f64,
    /// Consistency score from MATL
    pub consistency: f64,
    /// Reputation score from MATL
    pub reputation: f64,
    /// Stake-weighted score
    pub stake_weighted: f64,
    /// Oracle count
    pub oracle_count: u32,
    /// Oracle agreement level
    pub oracle_agreement: Option<f64>,
}

/// Calculate enhanced credibility with MATL integration
#[hdk_extern]
pub fn calculate_enhanced_credibility(input: EnhancedCredibilityInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let score_id = format!("enhanced_cred_{}_{}", input.subject, now.as_micros());

    // Calculate basic components
    let components = calculate_credibility_components(&input.subject, &input.subject_type)?;

    // Get MATL scores (from input or calculate defaults)
    let matl = if let Some(matl_data) = input.matl_data {
        let composite = (matl_data.quality * 0.3
            + matl_data.consistency * 0.25
            + matl_data.reputation * 0.25
            + matl_data.stake_weighted * 0.2)
            .clamp(0.0, 1.0);

        MatlCredibilityComponents {
            matl_composite: composite,
            matl_quality: matl_data.quality,
            matl_consistency: matl_data.consistency,
            matl_reputation: matl_data.reputation,
            matl_stake_weighted: matl_data.stake_weighted,
            oracle_agreement: matl_data.oracle_agreement,
            oracle_count: matl_data.oracle_count,
        }
    } else {
        MatlCredibilityComponents::default()
    };

    // Get evidence strength if requested
    let evidence_strength = if input.include_evidence {
        assess_evidence_strength_internal(&input.subject)?
    } else {
        EvidenceStrengthComponents::default()
    };

    // Get author reputation if requested
    let author_reputation = if input.include_author {
        // In a full implementation, we'd look up the claim's author
        // and retrieve or create their reputation
        None
    } else {
        None
    };

    // Calculate overall score with MATL weighting
    // MATL uses quadratic weighting: matl^2 for Byzantine resilience
    let matl_weight = matl.matl_composite * matl.matl_composite;
    let basic_weight = 1.0 - matl_weight.min(0.6);

    let basic_score = (components.accuracy * 0.25
        + components.consistency * 0.2
        + components.transparency * 0.2
        + components.track_record * 0.2
        + components.corroboration * 0.15)
        .clamp(0.0, 1.0);

    // Evidence bonus
    let evidence_bonus = (evidence_strength.total_evidence_count as f64 * 0.02)
        .min(0.1);

    let overall_score = (basic_score * basic_weight
        + matl.matl_composite * matl_weight
        + evidence_bonus)
        .clamp(0.0, 1.0);

    // Calculate assessment confidence based on data availability
    let assessment_confidence = calculate_assessment_confidence(
        &matl,
        &evidence_strength,
        author_reputation.is_some(),
    );

    let enhanced_score = EnhancedCredibilityScore {
        id: score_id,
        subject: input.subject.clone(),
        subject_type: input.subject_type,
        overall_score,
        components,
        matl,
        evidence_strength,
        author_reputation,
        factors: vec![], // Populated in full implementation
        assessed_at: now,
        expires_at: input.expires_at,
        assessment_model: "knowledge_v1_matl".to_string(),
        assessment_confidence,
    };

    let action_hash = create_entry(EntryTypes::EnhancedCredibilityScore(enhanced_score))?;

    // Create links for indexing using anchor pattern
    let subject_anchor = format!("subject:{}", input.subject);
    create_entry(&EntryTypes::Anchor(Anchor(subject_anchor.clone())))?;
    create_link(
        anchor_hash(&subject_anchor)?,
        action_hash.clone(),
        LinkTypes::SubjectToEnhancedCredibility,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find enhanced credibility score".into()
        )))
}

/// Calculate assessment confidence
fn calculate_assessment_confidence(
    matl: &MatlCredibilityComponents,
    evidence: &EvidenceStrengthComponents,
    has_author_rep: bool,
) -> f64 {
    let mut confidence = 0.5; // Base confidence

    // MATL oracle agreement boosts confidence
    if let Some(agreement) = matl.oracle_agreement {
        confidence += agreement * 0.2;
    }

    // More oracles = more confidence
    confidence += (matl.oracle_count as f64 * 0.01).min(0.15);

    // More evidence = more confidence
    confidence += (evidence.total_evidence_count as f64 * 0.02).min(0.1);

    // Evidence diversity boosts confidence
    confidence += evidence.evidence_diversity * 0.05;

    // Author reputation available boosts confidence
    if has_author_rep {
        confidence += 0.05;
    }

    confidence.clamp(0.0, 1.0)
}

/// Assess evidence strength for a claim
#[hdk_extern]
pub fn assess_evidence_strength(claim_id: String) -> ExternResult<EvidenceStrengthComponents> {
    assess_evidence_strength_internal(&claim_id)
}

/// Internal evidence strength assessment
fn assess_evidence_strength_internal(claim_id: &str) -> ExternResult<EvidenceStrengthComponents> {
    // In a full implementation, this would:
    // 1. Query the claims zome for evidence linked to this claim
    // 2. Categorize evidence by type
    // 3. Calculate average strengths
    // 4. Check for market resolutions

    // For now, return default values
    // A real implementation would use bridge calls to the claims zome:
    // call(CallTargetCell::Local, "claims", "get_claim_evidence", None, claim_id)?

    Ok(EvidenceStrengthComponents::default())
}

/// Get or create author reputation
#[hdk_extern]
pub fn get_author_reputation(author_did: String) -> ExternResult<Record> {
    // Check if reputation already exists
    let author_anchor = format!("author:{}", author_did);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&author_anchor)?, LinkTypes::AuthorToReputation)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(target) = link.target.clone().into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                return Ok(record);
            }
        }
    }

    // Create new reputation
    let now = sys_time()?;
    let reputation = AuthorReputation {
        id: format!("rep_{}_{}", author_did, now.as_micros()),
        author_did: author_did.clone(),
        overall_score: 0.5, // Default neutral reputation
        domain_scores: vec![],
        historical_accuracy: 0.5,
        claims_authored: 0,
        claims_verified_true: 0,
        claims_verified_false: 0,
        claims_pending: 0,
        average_epistemic_e: 0.5,
        matl_trust: 0.5,
        updated_at: now,
        reputation_age_days: 0,
    };

    let action_hash = create_entry(EntryTypes::AuthorReputation(reputation))?;
    create_entry(&EntryTypes::Anchor(Anchor(author_anchor.clone())))?;
    create_link(anchor_hash(&author_anchor)?, action_hash.clone(), LinkTypes::AuthorToReputation, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find author reputation".into()
        )))
}

/// Update author reputation based on claim verification
#[hdk_extern]
pub fn update_author_reputation(input: UpdateAuthorReputationInput) -> ExternResult<Record> {
    let author_anchor = format!("author:{}", input.author_did);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&author_anchor)?, LinkTypes::AuthorToReputation)?,
        GetStrategy::default(),
    )?;

    let current_record = if let Some(link) = links.first() {
        if let Some(target) = link.target.clone().into_action_hash() {
            get(target, GetOptions::default())?
        } else {
            None
        }
    } else {
        None
    };

    let now = sys_time()?;

    let (action_hash, updated_rep) = if let Some(record) = current_record {
        let mut rep: AuthorReputation = record
            .entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid reputation".into())))?;

        // Update statistics
        rep.claims_authored += input.claims_added.unwrap_or(0);
        rep.claims_verified_true += input.verified_true.unwrap_or(0);
        rep.claims_verified_false += input.verified_false.unwrap_or(0);

        // Recalculate accuracy
        let total_verified = rep.claims_verified_true + rep.claims_verified_false;
        if total_verified > 0 {
            rep.historical_accuracy = rep.claims_verified_true as f64 / total_verified as f64;
        }

        // Update MATL trust if provided
        if let Some(matl) = input.matl_trust_update {
            rep.matl_trust = (rep.matl_trust * 0.8 + matl * 0.2).clamp(0.0, 1.0);
        }

        // Recalculate overall score
        rep.overall_score = (rep.historical_accuracy * 0.4
            + rep.matl_trust * 0.4
            + (1.0 - (rep.claims_verified_false as f64 / (rep.claims_authored as f64 + 1.0))) * 0.2)
            .clamp(0.0, 1.0);

        rep.updated_at = now;

        let hash = update_entry(
            record.action_address().clone(),
            &EntryTypes::AuthorReputation(rep.clone()),
        )?;
        (hash, rep)
    } else {
        // Create new reputation
        let rep = AuthorReputation {
            id: format!("rep_{}_{}", input.author_did, now.as_micros()),
            author_did: input.author_did.clone(),
            overall_score: 0.5,
            domain_scores: vec![],
            historical_accuracy: 0.5,
            claims_authored: input.claims_added.unwrap_or(0),
            claims_verified_true: input.verified_true.unwrap_or(0),
            claims_verified_false: input.verified_false.unwrap_or(0),
            claims_pending: 0,
            average_epistemic_e: 0.5,
            matl_trust: input.matl_trust_update.unwrap_or(0.5),
            updated_at: now,
            reputation_age_days: 0,
        };

        let hash = create_entry(EntryTypes::AuthorReputation(rep.clone()))?;
        create_entry(&EntryTypes::Anchor(Anchor(author_anchor.clone())))?;
        create_link(anchor_hash(&author_anchor)?, hash.clone(), LinkTypes::AuthorToReputation, ())?;
        (hash, rep)
    };

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find updated reputation".into()
        )))
}

/// Input for updating author reputation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateAuthorReputationInput {
    pub author_did: String,
    pub claims_added: Option<u32>,
    pub verified_true: Option<u32>,
    pub verified_false: Option<u32>,
    pub matl_trust_update: Option<f64>,
}

/// Batch credibility assessment for multiple claims
#[hdk_extern]
pub fn batch_credibility_assessment(claim_ids: Vec<String>) -> ExternResult<BatchCredibilityResult> {
    let start_time = sys_time()?;

    let mut results: Vec<EnhancedCredibilitySummary> = vec![];
    let mut total_score = 0.0;
    let mut high_count = 0u32;
    let mut low_count = 0u32;

    for claim_id in &claim_ids {
        let input = EnhancedCredibilityInput {
            subject: claim_id.clone(),
            subject_type: CredibilitySubjectType::Claim,
            matl_data: None, // Would be fetched in full implementation
            include_evidence: true,
            include_author: false,
            expires_at: None,
        };

        if let Ok(record) = calculate_enhanced_credibility(input) {
            if let Some(score) = record
                .entry()
                .to_app_option::<EnhancedCredibilityScore>()
                .ok()
                .flatten()
            {
                total_score += score.overall_score;

                if score.overall_score > 0.7 {
                    high_count += 1;
                } else if score.overall_score < 0.3 {
                    low_count += 1;
                }

                results.push(EnhancedCredibilitySummary {
                    subject: claim_id.clone(),
                    overall_score: score.overall_score,
                    matl_composite: score.matl.matl_composite,
                    evidence_count: score.evidence_strength.total_evidence_count,
                    assessment_confidence: score.assessment_confidence,
                });
            }
        }
    }

    let end_time = sys_time()?;
    let processing_time_ms = (end_time.as_micros() - start_time.as_micros()) / 1000;

    let total_assessed = results.len() as u32;
    let average_score = if total_assessed > 0 {
        total_score / total_assessed as f64
    } else {
        0.0
    };

    Ok(BatchCredibilityResult {
        results,
        total_assessed,
        average_score,
        high_credibility_count: high_count,
        low_credibility_count: low_count,
        processing_time_ms: processing_time_ms as u64,
    })
}

/// Get enhanced credibility score for a subject
#[hdk_extern]
pub fn get_enhanced_credibility(subject: String) -> ExternResult<Option<Record>> {
    let subject_anchor = format!("subject:{}", subject);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&subject_anchor)?, LinkTypes::SubjectToEnhancedCredibility)?,
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get the most recent assessment
    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}
