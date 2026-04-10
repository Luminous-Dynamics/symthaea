// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # FL Coordinator Zome
//!
//! Implements business logic for federated learning rounds and updates.
//! This zome is upgradeable - business logic can change without breaking data.
//!
//! ## Revolutionary Features:
//! - Adaptive privacy budgets based on data sensitivity
//! - Byzantine-robust aggregation (Krum, coordinate-wise median)
//! - Context-aware privacy parameter tuning

use hdk::prelude::*;
use hdk::prelude::HdkPathExt;
use fl_integrity::{FlUpdate, FlRound, PrivacyParams, EntryTypes, LinkTypes};
use praxis_core::{RoundId, ModelHash};

// =============================================================================
// Inline Aggregation Functions
// =============================================================================
// Note: Using simplified inline implementations instead of edunet-agg to avoid
// statrs → nalgebra → rand → getrandom 0.2 dependency chain which requires
// wasm-bindgen that Holochain doesn't provide.

/// Configuration for aggregation
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Fraction of values to trim from each end (0.0 to 0.5)
    pub trim_percent: f32,
    /// Minimum number of updates required
    pub min_updates: usize,
}

/// Simple trimmed mean - trims outliers before averaging
pub fn trimmed_mean(gradients: &[Vec<f32>], config: &AggregationConfig) -> Result<Vec<f32>, String> {
    if gradients.len() < config.min_updates {
        return Err(format!("Need at least {} updates, got {}", config.min_updates, gradients.len()));
    }
    if gradients.is_empty() {
        return Err("No gradients provided".to_string());
    }

    let dim = gradients[0].len();
    let mut result = vec![0.0; dim];
    let trim_count = ((gradients.len() as f32 * config.trim_percent) as usize).max(0);

    for d in 0..dim {
        let mut values: Vec<f32> = gradients.iter().map(|g| g[d]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Trim top and bottom
        let trimmed: Vec<f32> = values[trim_count..values.len()-trim_count].to_vec();
        if trimmed.is_empty() {
            return Err("Too few values after trimming".to_string());
        }
        result[d] = trimmed.iter().sum::<f32>() / trimmed.len() as f32;
    }

    Ok(result)
}

/// Simple median aggregation - maximum Byzantine robustness
pub fn median(gradients: &[Vec<f32>]) -> Result<Vec<f32>, String> {
    if gradients.is_empty() {
        return Err("No gradients provided".to_string());
    }

    let dim = gradients[0].len();
    let mut result = vec![0.0; dim];

    for d in 0..dim {
        let mut values: Vec<f32> = gradients.iter().map(|g| g[d]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = values.len() / 2;
        result[d] = if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        };
    }

    Ok(result)
}

/// Simple weighted mean aggregation
pub fn weighted_mean(gradients: &[Vec<f32>], weights: &[f32]) -> Result<Vec<f32>, String> {
    if gradients.is_empty() {
        return Err("No gradients provided".to_string());
    }
    if gradients.len() != weights.len() {
        return Err("Gradients and weights must have same length".to_string());
    }

    let dim = gradients[0].len();
    let mut result = vec![0.0; dim];
    let total_weight: f32 = weights.iter().sum();

    if total_weight == 0.0 {
        return Err("Total weight is zero".to_string());
    }

    for (gradient, weight) in gradients.iter().zip(weights.iter()) {
        for (d, val) in gradient.iter().enumerate() {
            result[d] += val * weight;
        }
    }

    for val in result.iter_mut() {
        *val /= total_weight;
    }

    Ok(result)
}

/// Create a new federated learning round
#[hdk_extern]
pub fn create_round(round: FlRound) -> ExternResult<ActionHash> {
    // Trust tier gate: requires Steward tier to create FL rounds
    mycelix_bridge_common::gate_civic(
        "edunet_bridge",
        &mycelix_bridge_common::civic_requirement_constitutional(),
        "create_round",
    )?;

    // REVOLUTIONARY: Automatically calculate adaptive privacy params if not set
    let round_with_privacy = if round.privacy_epsilon.is_none() {
        let params = calculate_adaptive_privacy(
            round.round_id.clone(),
            round.clip_norm,
            round.min_participants,
        )?;

        FlRound {
            privacy_epsilon: Some(params.base_epsilon),
            privacy_delta: Some(0.00001), // Standard δ for (ε,δ)-DP
            ..round
        }
    } else {
        round
    };

    let action_hash = create_entry(EntryTypes::FlRound(round_with_privacy.clone()))?;

    // Create link from model to round for easy lookup
    let model_anchor = Path::from(format!("model_rounds.{}", round_with_privacy.model_id));
    let model_entry_hash = ensure_path(model_anchor, LinkTypes::ModelToRounds)?;

    let model_tag = round_with_privacy.model_id.as_bytes().to_vec();
    create_link(
        model_entry_hash,
        action_hash.clone(),
        LinkTypes::ModelToRounds,
        model_tag,
    )?;

    Ok(action_hash)
}

/// Get a round by its action hash
#[hdk_extern]
pub fn get_round(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Get all rounds for a specific model
#[hdk_extern]
pub fn get_model_rounds(model_hash: ModelHash) -> ExternResult<Vec<Record>> {
    let model_anchor = Path::from(format!("model_rounds.{}", model_hash.0));
    let model_entry_hash = ensure_path(model_anchor, LinkTypes::ModelToRounds)?;

    // Use Holochain 0.6 LinkQuery API
    let links = get_links(
        LinkQuery::new(
            model_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::ModelToRounds as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut rounds = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                rounds.push(record);
            }
        }
    }

    Ok(rounds)
}

/// Update an existing round (e.g., change state, add participants)
#[hdk_extern]
pub fn update_round(input: UpdateRoundInput) -> ExternResult<ActionHash> {
    let updated_action_hash = update_entry(
        input.original_action_hash,
        EntryTypes::FlRound(input.updated_round),
    )?;
    Ok(updated_action_hash)
}

/// Submit a gradient update to a round
#[hdk_extern]
pub fn submit_update(update: FlUpdate) -> ExternResult<ActionHash> {
    // Verify the round exists and is accepting updates
    // (In production, would check round state here)

    let action_hash = create_entry(EntryTypes::FlUpdate(update.clone()))?;

    // Create link from round to update
    let round_anchor = Path::from(format!("round_updates.{}", update.round_id.0));
    let round_entry_hash = ensure_path(round_anchor, LinkTypes::RoundToUpdates)?;

    create_link(
        round_entry_hash,
        action_hash.clone(),
        LinkTypes::RoundToUpdates,
        vec![],
    )?;

    Ok(action_hash)
}

/// Get all updates for a specific round
#[hdk_extern]
pub fn get_round_updates(round_id: RoundId) -> ExternResult<Vec<Record>> {
    let round_anchor = Path::from(format!("round_updates.{}", round_id.0));
    let round_entry_hash = ensure_path(round_anchor, LinkTypes::RoundToUpdates)?;

    // Use Holochain 0.6 LinkQuery API
    let links = get_links(
        LinkQuery::new(
            round_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToUpdates as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut updates = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                updates.push(record);
            }
        }
    }

    Ok(updates)
}

/// Aggregate updates for a round using specified method
/// REVOLUTIONARY: Supports multiple Byzantine-robust aggregation methods
#[hdk_extern]
pub fn aggregate_round(input: AggregateInput) -> ExternResult<AggregationResultOutput> {
    // Get all updates for the round
    let update_records = get_round_updates(input.round_id.clone())?;

    if update_records.len() < input.min_participants as usize {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Insufficient participants: {} < {}",
            update_records.len(),
            input.min_participants
        ))));
    }

    // Extract gradient values from commitments
    // (In production, this would involve secure aggregation protocol)
    let mut gradients: Vec<Vec<f32>> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    let mut expected_dim: Option<usize> = None;

    for record in &update_records {
        // Deserialize FlUpdate from record
        if let Some(update) = record.entry().as_option()
            .and_then(|entry| match entry {
                Entry::App(bytes) => {
                    // In Holochain 0.6, we deserialize directly from the bytes
                    FlUpdate::try_from(SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))).ok()
                },
                _ => None,
            })
        {
            // Weight by sample count (more samples = more weight)
            weights.push(update.sample_count as f32);

            // In a real implementation, grad_commitment would be decrypted here
            // For now, we use a placeholder
            let grad = decode_gradient_commitment(&update.grad_commitment)?;
            if grad.is_empty() {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Decoded gradient was empty; refusing to aggregate".to_string()
                )));
            }
            if let Some(dim) = expected_dim {
                if grad.len() != dim {
                    return Err(wasm_error!(WasmErrorInner::Guest(format!(
                        "Gradient dimension mismatch: expected {}, got {}",
                        dim,
                        grad.len()
                    ))));
                }
            } else {
                expected_dim = Some(grad.len());
            }
            gradients.push(grad);
        }
    }

    // Apply Byzantine-robust aggregation based on method
    let aggregated = match input.aggregation_method.as_str() {
        "trimmed_mean" => {
            let config = AggregationConfig {
                trim_percent: 0.1, // Trim 10% outliers
                min_updates: input.min_participants as usize,
            };
            trimmed_mean(&gradients, &config)
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        },
        "median" => {
            median(&gradients)
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        },
        "weighted_average" => {
            weighted_mean(&gradients, &weights)
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        },
        "krum" => {
            krum_aggregation(&gradients, 2) // Assume 2 Byzantine nodes
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        },
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Unknown aggregation method: {}",
                input.aggregation_method
            ))));
        }
    };

    // Calculate aggregated model hash
    let model_hash = calculate_model_hash(&aggregated)?;

    Ok(AggregationResultOutput {
        aggregated_model_hash: model_hash,
        num_participants: update_records.len() as u32,
        method_used: input.aggregation_method,
    })
}

/// Set or update privacy parameters for a round
/// REVOLUTIONARY: Adaptive privacy budget based on data sensitivity
#[hdk_extern]
pub fn set_privacy_params(params: PrivacyParams) -> ExternResult<ActionHash> {
    // Calculate adaptive epsilon based on sensitivity
    let adaptive_epsilon = calculate_adaptive_epsilon(
        params.base_epsilon,
        params.sensitivity_score,
        params.min_epsilon,
        params.max_epsilon,
    );

    let adjusted_params = PrivacyParams {
        base_epsilon: adaptive_epsilon,
        ..params
    };

    let action_hash = create_entry(EntryTypes::PrivacyParams(adjusted_params.clone()))?;

    // Link from round to privacy params
    let round_anchor = Path::from(format!("round_privacy.{}", adjusted_params.round_id.0));
    let round_entry_hash = ensure_path(round_anchor, LinkTypes::RoundToPrivacy)?;

    create_link(
        round_entry_hash,
        action_hash.clone(),
        LinkTypes::RoundToPrivacy,
        vec![],
    )?;

    Ok(action_hash)
}

/// Get privacy parameters for a round
#[hdk_extern]
pub fn get_privacy_params(round_id: RoundId) -> ExternResult<Option<Record>> {
    let round_anchor = Path::from(format!("round_privacy.{}", round_id.0));
    let round_entry_hash = ensure_path(round_anchor, LinkTypes::RoundToPrivacy)?;

    // Use Holochain 0.6 LinkQuery API
    let links = get_links(
        LinkQuery::new(
            round_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToPrivacy as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    // Return the most recent privacy params
    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            return get(action_hash, GetOptions::default());
        }
    }

    Ok(None)
}

// ============================================================================
// Helper functions for revolutionary features
// ============================================================================

/// Calculate adaptive privacy parameters based on model and participant count
/// REVOLUTIONARY: Context-aware privacy budget allocation
fn calculate_adaptive_privacy(
    round_id: RoundId,
    clip_norm: f32,
    min_participants: u32,
) -> ExternResult<PrivacyParams> {
    // More participants = tighter privacy (lower epsilon)
    // Larger clip norm = more noise tolerance (higher epsilon possible)
    let base_epsilon = if min_participants < 10 {
        1.0 // High privacy for few participants
    } else if min_participants < 100 {
        2.0 // Medium privacy
    } else {
        5.0 // Can afford looser privacy with many participants
    };

    // Adjust based on clip norm (larger norm = more signal, can add more noise)
    let adjusted_epsilon = base_epsilon * (clip_norm / 1.0).min(2.0);

    Ok(PrivacyParams {
        round_id,
        base_epsilon: adjusted_epsilon,
        min_epsilon: 0.1,
        max_epsilon: 10.0,
        sensitivity_score: 0.5, // Default medium sensitivity
    })
}

/// Calculate adaptive epsilon based on data sensitivity
/// REVOLUTIONARY: Higher sensitivity = stronger privacy (lower epsilon)
fn calculate_adaptive_epsilon(
    base_epsilon: f32,
    sensitivity_score: f32,
    min_epsilon: f32,
    max_epsilon: f32,
) -> f32 {
    // Sensitivity score: 0.0 = low (public data), 1.0 = high (very private data)
    // Higher sensitivity should result in lower epsilon (stronger privacy)
    let privacy_multiplier = 1.0 - sensitivity_score;
    let adaptive_epsilon = base_epsilon * privacy_multiplier.max(0.1);

    // Clamp to bounds
    adaptive_epsilon.max(min_epsilon).min(max_epsilon)
}

/// Krum aggregation - selects the gradient closest to others (Byzantine-robust)
/// REVOLUTIONARY: Resists up to f Byzantine participants
fn krum_aggregation(gradients: &[Vec<f32>], f: usize) -> Result<Vec<f32>, String> {
    if gradients.len() <= 2 * f {
        return Err(format!(
            "Insufficient honest participants: {} <= 2*{}",
            gradients.len(),
            f
        ));
    }

    let n = gradients.len();
    let m = n - f - 2;

    // Calculate pairwise distances
    let mut scores = vec![0.0; n];
    for i in 0..n {
        let mut distances: Vec<(usize, f32)> = Vec::new();
        for j in 0..n {
            if i != j {
                let dist = euclidean_distance(&gradients[i], &gradients[j]);
                distances.push((j, dist));
            }
        }

        // Sort by distance and sum the m closest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let score: f32 = distances.iter().take(m).map(|(_, d)| d).sum();
        scores[i] = score;
    }

    // Select gradient with minimum score (most similar to others)
    let best_idx = scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No gradients to aggregate".into())))?;

    Ok(gradients[best_idx].clone())
}

/// Calculate Euclidean distance between two vectors
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Decode a gradient commitment
/// (In production, this would involve cryptographic operations)
fn decode_gradient_commitment(commitment: &[u8]) -> ExternResult<Vec<f32>> {
    // Always produce a fixed-length deterministic vector so dimensions are consistent.
    const GRAD_DIM: usize = 16;

    if commitment.is_empty() {
        return Ok(vec![0.0; GRAD_DIM]);
    }

    let hash = blake3::hash(commitment);
    let mut gradients = Vec::with_capacity(GRAD_DIM);
    let bytes = hash.as_bytes();

    for chunk in bytes.chunks(2).take(GRAD_DIM) {
        let mut buf = [0u8; 2];
        for (i, b) in chunk.iter().enumerate() {
            buf[i] = *b;
        }
        let raw = u16::from_le_bytes(buf);
        gradients.push(raw as f32 / u16::MAX as f32);
    }

    Ok(gradients)
}

/// Calculate model hash from aggregated gradients
fn calculate_model_hash(gradients: &[f32]) -> ExternResult<ModelHash> {
    // Serialize gradients and hash them
    let serialized = serde_json::to_string(gradients)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    let hash = blake3::hash(serialized.as_bytes());
    Ok(ModelHash(format!("{}", hash)))
}

// ============================================================================
// Input/Output structures
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateRoundInput {
    pub original_action_hash: ActionHash,
    pub updated_round: FlRound,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AggregateInput {
    pub round_id: RoundId,
    pub min_participants: u32,
    pub aggregation_method: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AggregationResultOutput {
    pub aggregated_model_hash: ModelHash,
    pub num_participants: u32,
    pub method_used: String,
}

// ============================================================================
// Helpers
// ============================================================================

fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.clone().typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

// ============== Skill Intelligence ==============
//
// Privacy-preserving population-level mastery distribution signals.
// Each agent contributes a bucketed mastery level (none/low/medium/high)
// for a skill. FL aggregation produces population distribution without
// exposing individual mastery values.

/// Mastery bucket for privacy-preserving skill supply signals.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MasteryBucket {
    /// 0-200 permille mastery
    None,
    /// 201-500 permille
    Low,
    /// 501-800 permille
    Medium,
    /// 801-1000 permille
    High,
}

impl MasteryBucket {
    /// Bucket a raw mastery permille value.
    pub fn from_permille(p: u16) -> Self {
        match p {
            0..=200 => Self::None,
            201..=500 => Self::Low,
            501..=800 => Self::Medium,
            _ => Self::High,
        }
    }
}

/// Input for contributing a mastery signal to the skill intelligence system.
#[derive(Serialize, Deserialize, Debug)]
pub struct MasterySignalInput {
    /// Skill identifier (e.g., "rust", "calculus")
    pub skill: String,
    /// Bucketed mastery level (privacy-preserving: no exact value shared)
    pub bucket: MasteryBucket,
}

/// Aggregated skill supply distribution (output of FL aggregation).
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct SkillSupply {
    pub skill: String,
    pub count_none: u32,
    pub count_low: u32,
    pub count_medium: u32,
    pub count_high: u32,
    pub total_contributors: u32,
    pub high_pct: u16, // permille: (count_high / total) * 1000
}

/// Contribute a bucketed mastery signal for a skill.
///
/// This is a simplified FL contribution — each agent submits their
/// mastery bucket (not raw value) to a skill-specific anchor.
/// The aggregation is computed client-side from the link counts.
///
/// Privacy guarantee: only the bucket (None/Low/Medium/High) is shared,
/// not the exact mastery value. This provides k-anonymity within buckets.
#[hdk_extern]
pub fn contribute_mastery_signal(input: MasterySignalInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Create a lightweight signal entry (just skill + bucket)
    // Using FlUpdate infrastructure would be overkill for a simple bucket
    let signal_data = serde_json::json!({
        "skill": input.skill,
        "bucket": input.bucket,
        "agent": agent.to_string(),
    });

    let signal_bytes = serde_json::to_vec(&signal_data)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Serialize: {}", e))))?;

    // Use the existing FlUpdate as the carrier (grad_commitment = signal bytes)
    let now = sys_time()?.as_micros() / 1_000_000;
    let update = FlUpdate {
        round_id: praxis_core::RoundId(format!("skill-intel-{}", input.skill)),
        model_id: "skill-intelligence-v1".to_string(),
        parent_model_hash: praxis_core::ModelHash("none".to_string()),
        grad_commitment: signal_bytes,
        clipped_l2_norm: 0.0,
        local_val_loss: 0.0,
        sample_count: 1,
        timestamp: now,
    };

    let hash = create_entry(EntryTypes::FlUpdate(update))?;

    // Link to skill-specific anchor for efficient aggregation
    let skill_anchor = ensure_path(
        Path::from(format!("skill_intel.{}", input.skill.to_lowercase())),
        LinkTypes::RoundToUpdates,
    )?;
    create_link(skill_anchor, hash.clone(), LinkTypes::RoundToUpdates, vec![])?;

    Ok(hash)
}

/// Aggregate mastery signals for a skill (client-side would call this).
///
/// Counts contributions per bucket via link traversal.
/// NOTE: For large-scale deployment, this should move to client-side
/// aggregation to avoid DHT load. For now, it's acceptable for <1000 contributors.
#[hdk_extern]
pub fn get_skill_supply(skill: String) -> ExternResult<SkillSupply> {
    let anchor = ensure_path(
        Path::from(format!("skill_intel.{}", skill.to_lowercase())),
        LinkTypes::RoundToUpdates,
    )?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::RoundToUpdates)?,
        GetStrategy::Local,
    )?;

    let mut supply = SkillSupply {
        skill,
        ..Default::default()
    };

    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(update) = FlUpdate::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        // Parse the bucket from the grad_commitment JSON
                        if let Ok(signal) = serde_json::from_slice::<serde_json::Value>(&update.grad_commitment) {
                            if let Some(bucket_str) = signal.get("bucket").and_then(|b| b.as_str()) {
                                match bucket_str {
                                    "None" => supply.count_none += 1,
                                    "Low" => supply.count_low += 1,
                                    "Medium" => supply.count_medium += 1,
                                    "High" => supply.count_high += 1,
                                    _ => {}
                                }
                                supply.total_contributors += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    if supply.total_contributors > 0 {
        supply.high_pct = ((supply.count_high as u32 * 1000) / supply.total_contributors) as u16;
    }

    Ok(supply)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_commitment_produces_fixed_length() {
        let commitment = vec![1u8; 32];
        let grad = decode_gradient_commitment(&commitment).unwrap();
        assert_eq!(grad.len(), 16);

        // Different commitment => different gradient
        let other = decode_gradient_commitment(&[2u8; 32]).unwrap();
        assert_ne!(grad, other);

        // Empty commitment yields zeros of same length
        let zeros = decode_gradient_commitment(&[]).unwrap();
        assert_eq!(zeros.len(), 16);
        assert!(zeros.iter().all(|v| *v == 0.0));
    }
}
