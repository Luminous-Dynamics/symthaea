// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PoGQ Validation Coordinator Zome
//!
//! Provides zome functions for gradient submission and validation
//! in federated learning using PoGQ (Proof of Gradient Quality).

use hdk::prelude::*;
use hdk::prelude::HdkPathExt;
use pogq_validation_integrity::{
    AgentTrustRecord, EntryTypes, GradientSubmission, LinkTypes, ValidationResult,
    MAX_GRADIENT_NORM, MIN_GRADIENT_ELEMENTS, REQUIRED_EPSILON, REQUIRED_SIGMA,
};

/// Input for submitting a gradient contribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitGradientInput {
    pub round: u64,
    pub gradient_hash: String,
    pub l2_norm: f32,
    pub num_elements: u32,
    pub noise_sigma: f32,
    pub claimed_epsilon: f32,
    pub mean: f32,
    pub variance: f32,
    pub skewness: f32,
    pub kurtosis: f32,
}

/// Output from submitting a gradient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitGradientOutput {
    pub submission_hash: ActionHash,
    pub validation_hash: ActionHash,
    pub quality_score: f32,
    pub is_valid: bool,
}

/// PoGQ validation result (computed in-zome)
struct PoGQResult {
    dp_compliant: bool,
    bounds_valid: bool,
    quality_score: f32,
    error: Option<String>,
}

/// Get current timestamp as seconds
fn current_timestamp() -> ExternResult<i64> {
    Ok(sys_time()?.0 as i64 / 1_000_000)
}

/// Validate gradient contribution using PoGQ rules
fn validate_gradient_pogq(
    claimed_epsilon: f32,
    noise_sigma: f32,
    l2_norm: f32,
    num_elements: u32,
    skewness: f32,
    kurtosis: f32,
    variance: f32,
) -> PoGQResult {
    // 1. Check differential privacy compliance
    // Epsilon should be <= required (lower is more private)
    if claimed_epsilon > REQUIRED_EPSILON {
        return PoGQResult {
            dp_compliant: false,
            bounds_valid: false,
            quality_score: 0.0,
            error: Some(format!(
                "DP violation: epsilon {} > required {}",
                claimed_epsilon, REQUIRED_EPSILON
            )),
        };
    }

    // Sigma should be >= required (higher is more noise)
    let dp_compliant = noise_sigma >= REQUIRED_SIGMA;

    // 2. Check gradient bounds
    let norm_valid = l2_norm <= MAX_GRADIENT_NORM;
    let elements_valid = num_elements >= MIN_GRADIENT_ELEMENTS;
    let bounds_valid = norm_valid && elements_valid;

    // 3. Anomaly detection (Byzantine behavior indicators)
    let max_skewness = 1.0f32;
    let max_excess_kurtosis = 2.0f32;
    let min_variance = 0.001f32;
    let max_variance = 100.0f32;

    let skewness_ok = skewness.abs() < max_skewness;
    let kurtosis_ok = (kurtosis - 3.0).abs() < max_excess_kurtosis;
    let variance_ok = variance > min_variance && variance < max_variance;

    // Compute anomaly score (0 = no anomaly, 1 = definite anomaly)
    let mut anomaly_score = 0.0f32;
    if !skewness_ok {
        anomaly_score += 0.3;
    }
    if !kurtosis_ok {
        anomaly_score += 0.3;
    }
    if !variance_ok {
        anomaly_score += 0.4;
    }

    // 4. Compute quality score
    let quality_score = if !dp_compliant || !bounds_valid {
        0.0
    } else {
        let mut score = 1.0f32;

        // Penalize anomalies
        score -= anomaly_score * 0.5;

        // Bonus for stronger privacy (lower epsilon)
        if claimed_epsilon < REQUIRED_EPSILON {
            let privacy_bonus = (REQUIRED_EPSILON - claimed_epsilon) / REQUIRED_EPSILON;
            score += privacy_bonus * 0.1;
        }

        // Bonus for more noise (higher sigma)
        if noise_sigma > REQUIRED_SIGMA {
            let noise_bonus = (noise_sigma - REQUIRED_SIGMA) / REQUIRED_SIGMA;
            score += noise_bonus.min(0.1) * 0.1;
        }

        score.clamp(0.0, 1.0)
    };

    PoGQResult {
        dp_compliant,
        bounds_valid,
        quality_score,
        error: None,
    }
}

/// Submit a gradient contribution for validation
#[hdk_extern]
pub fn submit_gradient(input: SubmitGradientInput) -> ExternResult<SubmitGradientOutput> {
    let agent = agent_info()?.agent_initial_pubkey;
    let timestamp = current_timestamp()?;

    // Create the gradient submission entry
    let submission = GradientSubmission {
        agent: agent.to_string(),
        round: input.round,
        gradient_hash: input.gradient_hash.clone(),
        l2_norm: input.l2_norm,
        num_elements: input.num_elements,
        noise_sigma: input.noise_sigma,
        claimed_epsilon: input.claimed_epsilon,
        mean: input.mean,
        variance: input.variance,
        skewness: input.skewness,
        kurtosis: input.kurtosis,
        submitted_at: timestamp,
    };

    // Create the submission entry
    let submission_hash = create_entry(EntryTypes::GradientSubmission(submission.clone()))?;

    // Link from agent to submission
    create_link(
        agent.clone(),
        submission_hash.clone(),
        LinkTypes::AgentToSubmissions,
        (),
    )?;

    // Link from round to submission (using round number as path anchor)
    let round_path = Path::from(format!("rounds.{}", input.round));
    let typed_path = round_path.clone().typed(LinkTypes::RoundToSubmissions)?;
    typed_path.ensure()?;
    create_link(
        typed_path.path_entry_hash()?,
        submission_hash.clone(),
        LinkTypes::RoundToSubmissions,
        (),
    )?;

    // Validate the gradient using PoGQ rules
    let pogq_result = validate_gradient_pogq(
        input.claimed_epsilon,
        input.noise_sigma,
        input.l2_norm,
        input.num_elements,
        input.skewness,
        input.kurtosis,
        input.variance,
    );

    let is_valid = pogq_result.dp_compliant && pogq_result.bounds_valid && pogq_result.error.is_none();

    // Create validation result entry
    let result = ValidationResult {
        submission_hash: submission_hash.to_string(),
        agent: agent.to_string(),
        round: input.round,
        dp_compliant: pogq_result.dp_compliant,
        epsilon: input.claimed_epsilon,
        sigma: input.noise_sigma,
        l2_norm: input.l2_norm,
        bounds_valid: pogq_result.bounds_valid,
        quality_score: pogq_result.quality_score,
        is_valid,
        error_message: pogq_result.error,
        validator: agent.to_string(),
        validated_at: timestamp,
    };

    let validation_hash = create_entry(EntryTypes::ValidationResult(result))?;

    // Link from submission to validation
    create_link(
        submission_hash.clone(),
        validation_hash.clone(),
        LinkTypes::SubmissionToValidation,
        (),
    )?;

    // Update agent's trust record
    update_agent_trust_record(&agent, pogq_result.quality_score, is_valid)?;

    Ok(SubmitGradientOutput {
        submission_hash,
        validation_hash,
        quality_score: pogq_result.quality_score,
        is_valid,
    })
}

/// Get submissions for a specific round
#[hdk_extern]
pub fn get_round_submissions(round: u64) -> ExternResult<Vec<GradientSubmission>> {
    let round_path = Path::from(format!("rounds.{}", round));
    let typed_path = round_path.typed(LinkTypes::RoundToSubmissions)?;
    let path_hash = typed_path.path_entry_hash()?;

    let links = get_links(
        LinkQuery::new(
            path_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToSubmissions as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut submissions = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(entry) = record.entry().as_option() {
                    if let Entry::App(bytes) = entry {
                        if let Ok(submission) = GradientSubmission::try_from(
                            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                        ) {
                            submissions.push(submission);
                        }
                    }
                }
            }
        }
    }

    Ok(submissions)
}

/// Get an agent's submissions
#[hdk_extern]
pub fn get_agent_submissions(agent: AgentPubKey) -> ExternResult<Vec<GradientSubmission>> {
    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(agent),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AgentToSubmissions as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut submissions = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(entry) = record.entry().as_option() {
                    if let Entry::App(bytes) = entry {
                        if let Ok(submission) = GradientSubmission::try_from(
                            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                        ) {
                            submissions.push(submission);
                        }
                    }
                }
            }
        }
    }

    Ok(submissions)
}

/// Get validation result for a submission
#[hdk_extern]
pub fn get_validation_result(submission_hash: ActionHash) -> ExternResult<Option<ValidationResult>> {
    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(submission_hash),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::SubmissionToValidation as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(entry) = record.entry().as_option() {
                    if let Entry::App(bytes) = entry {
                        if let Ok(result) = ValidationResult::try_from(
                            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                        ) {
                            return Ok(Some(result));
                        }
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Get an agent's trust record
#[hdk_extern]
pub fn get_agent_trust_record(agent: AgentPubKey) -> ExternResult<Option<AgentTrustRecord>> {
    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(agent),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AgentToTrustRecord as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    // Get the most recent trust record (last link)
    if let Some(link) = links.last() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(entry) = record.entry().as_option() {
                    if let Entry::App(bytes) = entry {
                        if let Ok(trust_record) = AgentTrustRecord::try_from(
                            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                        ) {
                            return Ok(Some(trust_record));
                        }
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Get my trust record
#[hdk_extern]
pub fn get_my_trust_record(_: ()) -> ExternResult<Option<AgentTrustRecord>> {
    let agent = agent_info()?.agent_initial_pubkey;
    get_agent_trust_record(agent)
}

/// Input for flagging Byzantine behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlagByzantineInput {
    pub agent: AgentPubKey,
    pub evidence: String,
}

/// Flag an agent for Byzantine behavior
#[hdk_extern]
pub fn flag_byzantine(input: FlagByzantineInput) -> ExternResult<()> {
    let timestamp = current_timestamp()?;

    // Get current trust record or create new one
    let record = match get_agent_trust_record(input.agent.clone())? {
        Some(mut existing) => {
            existing.byzantine_flags += 1;
            existing.version += 1;
            existing.updated_at = timestamp;
            // Heavily penalize quality score
            existing.avg_quality_score *= 0.5;
            existing
        }
        None => AgentTrustRecord {
            agent: input.agent.to_string(),
            total_submissions: 0,
            valid_submissions: 0,
            avg_quality_score: 0.0,
            byzantine_flags: 1,
            updated_at: timestamp,
            version: 1,
        },
    };

    let hash = create_entry(EntryTypes::AgentTrustRecord(record))?;

    // Link to agent
    create_link(
        input.agent,
        hash,
        LinkTypes::AgentToTrustRecord,
        (),
    )?;

    Ok(())
}

/// Round statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundStats {
    pub round: u64,
    pub total_submissions: usize,
    pub valid_submissions: usize,
    pub avg_quality_score: f32,
    pub avg_l2_norm: f32,
}

/// Get statistics for a round
#[hdk_extern]
pub fn get_round_stats(round: u64) -> ExternResult<RoundStats> {
    let submissions = get_round_submissions(round)?;
    let total = submissions.len();

    if total == 0 {
        return Ok(RoundStats {
            round,
            total_submissions: 0,
            valid_submissions: 0,
            avg_quality_score: 0.0,
            avg_l2_norm: 0.0,
        });
    }

    let avg_l2_norm: f32 = submissions.iter().map(|s| s.l2_norm).sum::<f32>() / total as f32;

    // Count valid submissions
    let mut valid_count = 0;
    let mut quality_sum = 0.0f32;

    for submission in &submissions {
        let pogq = validate_gradient_pogq(
            submission.claimed_epsilon,
            submission.noise_sigma,
            submission.l2_norm,
            submission.num_elements,
            submission.skewness,
            submission.kurtosis,
            submission.variance,
        );

        if pogq.dp_compliant && pogq.bounds_valid && pogq.error.is_none() {
            valid_count += 1;
            quality_sum += pogq.quality_score;
        }
    }

    let avg_quality = if valid_count > 0 {
        quality_sum / valid_count as f32
    } else {
        0.0
    };

    Ok(RoundStats {
        round,
        total_submissions: total,
        valid_submissions: valid_count,
        avg_quality_score: avg_quality,
        avg_l2_norm,
    })
}

// Helper function to update agent trust record
fn update_agent_trust_record(
    agent: &AgentPubKey,
    quality_score: f32,
    is_valid: bool,
) -> ExternResult<()> {
    let timestamp = current_timestamp()?;

    let record = match get_agent_trust_record(agent.clone())? {
        Some(mut existing) => {
            existing.total_submissions += 1;
            if is_valid {
                existing.valid_submissions += 1;
            }
            // Update average quality score (exponential moving average)
            let alpha = 0.2f32;
            existing.avg_quality_score = (1.0 - alpha) * existing.avg_quality_score
                + alpha * quality_score;
            existing.updated_at = timestamp;
            existing.version += 1;
            existing
        }
        None => AgentTrustRecord {
            agent: agent.to_string(),
            total_submissions: 1,
            valid_submissions: if is_valid { 1 } else { 0 },
            avg_quality_score: quality_score,
            byzantine_flags: 0,
            updated_at: timestamp,
            version: 1,
        },
    };

    let hash = create_entry(EntryTypes::AgentTrustRecord(record))?;

    // Link to agent
    create_link(
        agent.clone(),
        hash,
        LinkTypes::AgentToTrustRecord,
        (),
    )?;

    Ok(())
}
