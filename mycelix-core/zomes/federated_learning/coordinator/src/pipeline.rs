//! Unified pipeline integration with mycelix-fl for decentralized aggregation.

use hdk::prelude::*;
use federated_learning_integrity::*;

use mycelix_fl::pipeline::{DecentralizedPipeline, PipelineConfig};
use mycelix_fl::types::{CompressedGradient, GradientMetadata as FlGradientMetadata};
use mycelix_fl::holochain::{
    ZomeAggregationCommitment, ZomeAggregationReveal, DetectionSummary as FlDetectionSummary,
    to_zome_commitment,
};
use mycelix_sdk::hyperfeel::HV16_BYTES;

use crate::signals::Signal;
use super::ensure_path;
use crate::hyperfeel::get_round_hypervectors;
use crate::consensus::get_active_validators_internal;

/// Result of running the validator pipeline locally
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ValidatorPipelineResult {
    /// SHA-256 commitment hash of the aggregated result
    pub commitment_hash: String,
    /// Aggregated HV16 data (2KB)
    pub aggregated_hv: Vec<u8>,
    /// Method used for aggregation
    pub method: String,
    /// Number of gradients included
    pub gradient_count: u32,
    /// Number excluded by Byzantine detection
    pub excluded_count: u32,
    /// IDs of excluded participants
    pub excluded_participants: Vec<String>,
    /// Detection summary (structured, from mycelix-fl bridge types)
    pub detection_summary: FlDetectionSummary,
}

// -------------------------------------------------------------------------
// Bridge type conversion helpers
// -------------------------------------------------------------------------

/// Convert a mycelix-fl ZomeAggregationCommitment to an integrity entry
pub(crate) fn bridge_commitment_to_entry(zc: &ZomeAggregationCommitment) -> AggregationCommitment {
    AggregationCommitment {
        round: zc.round,
        aggregator: zc.aggregator.clone(),
        commitment_hash: zc.commitment_hash.clone(),
        method: zc.method.clone(),
        gradient_count: zc.gradient_count,
        excluded_count: zc.excluded_count,
        committed_at: zc.committed_at,
        aggregator_trust_score: zc.aggregator_trust_score,
    }
}

/// Convert a mycelix-fl ZomeAggregationReveal to an integrity entry
pub(crate) fn bridge_reveal_to_entry(zr: &ZomeAggregationReveal) -> AggregationReveal {
    AggregationReveal {
        round: zr.round,
        aggregator: zr.aggregator.clone(),
        result_data: zr.result_data.clone(),
        result_hash: zr.result_hash.clone(),
        detection_summary_json: serde_json::to_string(&zr.detection_summary)
            .unwrap_or_default(),
        shapley_values_json: serde_json::to_string(&zr.shapley_values)
            .unwrap_or_default(),
        revealed_at: zr.revealed_at,
    }
}

/// Run the full mycelix-fl pipeline locally on a round's HV16 gradients
///
/// This is the key decentralization function: every validator calls this
/// independently on the same set of gradients. The result should be
/// deterministic -- all honest validators produce the same commitment hash.
///
/// Flow:
/// 1. Fetch all HV16 gradients for the round from DHT
/// 2. Build reputation map from on-chain data
/// 3. Run mycelix-fl DecentralizedPipeline::aggregate_compressed()
/// 4. Compute SHA-256 commitment hash
/// 5. Return result for the caller to submit as a commitment
#[hdk_extern]
pub fn run_validator_pipeline(round: u32) -> ExternResult<ValidatorPipelineResult> {
    // Step 1: Fetch all HV16 gradients for this round
    let hypervectors = get_round_hypervectors(round)?;
    if hypervectors.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "No hypervectors found for this round".to_string()
        )));
    }

    // Step 2: Build CompressedGradient objects + reputation map
    let mut compressed_gradients = Vec::with_capacity(hypervectors.len());
    let mut reputations = std::collections::HashMap::new();

    for (node_id, hv_data) in &hypervectors {
        // Build CompressedGradient from DHT data
        let gradient = CompressedGradient {
            participant_id: node_id.clone(),
            hv_data: hv_data.clone(),
            original_dimension: 0, // Unknown from compressed form
            quality_score: 1.0,    // Default; PoGQ score applied during detection
            metadata: FlGradientMetadata::new(round, 0.0),
        };
        compressed_gradients.push(gradient);

        // Look up reputation for this node
        let rep = match get_or_create_reputation(node_id) {
            Ok(node_rep) => node_rep.reputation_score as f32,
            Err(_) => 0.5, // Default reputation for unknown nodes
        };
        reputations.insert(node_id.clone(), rep);
    }

    // Step 3: Run the unified pipeline
    let pipeline = DecentralizedPipeline::new(PipelineConfig::default());
    let result = pipeline.aggregate_compressed(&compressed_gradients, &reputations)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(
            format!("Pipeline aggregation failed: {:?}", e)
        )))?;

    let aggregated_hv = result.aggregated_hv.ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(
            "Pipeline produced no aggregated HV".to_string()
        ))
    })?;

    // Step 4: Compute SHA-256 commitment hash
    let method = result.stats.method_used.clone();
    let commitment_hash = DecentralizedPipeline::commitment_hash(
        &aggregated_hv,
        &method,
        round as u64,
    );

    // Step 5: Build detection summary from pipeline result
    let detection_summary = FlDetectionSummary {
        flagged_nodes: result.stats.excluded_participants.iter()
            .map(|id| (id.clone(), 1.0_f32))
            .collect(),
        detection_layers_used: vec!["MultiSignal".to_string(), "HvCosineFilter".to_string()],
        total_checked: result.stats.total_contributions,
        total_flagged: result.stats.byzantine_detected,
    };

    // Step 6: Return result
    Ok(ValidatorPipelineResult {
        commitment_hash,
        aggregated_hv,
        method,
        gradient_count: result.stats.after_detection as u32,
        excluded_count: result.stats.byzantine_detected as u32,
        excluded_participants: result.stats.excluded_participants,
        detection_summary,
    })
}

/// Convenience: run pipeline + submit commitment in one call
///
/// Equivalent to calling run_validator_pipeline() then submit_aggregation_commitment().
/// Uses mycelix-fl bridge types (to_zome_commitment) for single-source-of-truth conversion.
#[hdk_extern]
pub fn run_and_commit(round: u32) -> ExternResult<ActionHash> {
    let pipeline_result = run_validator_pipeline(round)?;
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?.0 as i64 / 1_000_000;

    // Look up our trust score
    let validators = get_active_validators_internal()?;
    let trust_score = validators.iter()
        .find(|v| v.agent_pubkey == agent.to_string())
        .map(|v| v.trust_score)
        .unwrap_or(0.5);

    // Build bridge type using mycelix-fl helper
    let zome_commitment = to_zome_commitment(
        round as u64,
        &agent.to_string(),
        &pipeline_result.commitment_hash,
        &pipeline_result.method,
        pipeline_result.gradient_count,
        pipeline_result.excluded_count,
        trust_score,
        now,
    );

    // Convert bridge type to integrity entry
    let commitment = bridge_commitment_to_entry(&zome_commitment);
    let action_hash = create_entry(&EntryTypes::AggregationCommitment(commitment))?;

    // Link to round
    let round_path = Path::from(format!("rounds/{}/commitments", round));
    let round_hash = ensure_path(round_path, LinkTypes::RoundToCommitments)?;
    create_link(
        round_hash,
        action_hash.clone(),
        LinkTypes::RoundToCommitments,
        vec![],
    )?;

    // Emit gossip signal
    emit_signal(Signal::CommitReady {
        round: round as u64,
        validator_id: agent.to_string(),
        commitment_hash: zome_commitment.commitment_hash,
        source: Some(agent.to_string()),
        signature: None,
    })?;

    Ok(action_hash)
}

/// Convenience: reveal aggregation using stored pipeline result
///
/// Takes the pipeline result from run_validator_pipeline() and submits
/// it as a reveal entry. Uses mycelix-fl bridge types for conversion.
#[hdk_extern]
pub fn run_and_reveal(round: u32) -> ExternResult<ActionHash> {
    let pipeline_result = run_validator_pipeline(round)?;
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?.0 as i64 / 1_000_000;

    // Build the reveal using bridge type
    let zome_reveal = ZomeAggregationReveal {
        round: round as u64,
        aggregator: agent.to_string(),
        result_data: pipeline_result.aggregated_hv,
        result_hash: pipeline_result.commitment_hash,
        detection_summary: pipeline_result.detection_summary,
        shapley_values: pipeline_result.excluded_participants.iter()
            .map(|id| (id.clone(), 0.0_f32))
            .collect(),
        revealed_at: now,
    };

    // Convert bridge type to integrity entry and create
    let reveal = bridge_reveal_to_entry(&zome_reveal);
    let action_hash = create_entry(&EntryTypes::AggregationReveal(reveal))?;

    // Link to round
    let round_path = Path::from(format!("rounds/{}/reveals", round));
    let round_hash = ensure_path(round_path, LinkTypes::RoundToReveals)?;
    create_link(
        round_hash,
        action_hash.clone(),
        LinkTypes::RoundToReveals,
        vec![],
    )?;

    Ok(action_hash)
}

/// Compute similarity between two hypervectors
#[hdk_extern]
pub fn compute_hypervector_similarity(input: (Vec<u8>, Vec<u8>)) -> ExternResult<f32> {
    let (hv1, hv2) = input;

    if hv1.len() != HV16_BYTES || hv2.len() != HV16_BYTES {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid hypervector size".to_string()
        )));
    }

    // Cosine similarity for bipolar vectors
    let total_bits = hv1.len() * 8;
    let mut dot_product: i32 = 0;

    for (b1, b2) in hv1.iter().zip(hv2.iter()) {
        for i in 0..8 {
            let bit1 = (b1 >> (7 - i)) & 1;
            let bit2 = (b2 >> (7 - i)) & 1;
            let val1: i32 = if bit1 == 1 { 1 } else { -1 };
            let val2: i32 = if bit2 == 1 { 1 } else { -1 };
            dot_product += val1 * val2;
        }
    }

    Ok(dot_product as f32 / total_bits as f32)
}

/// Pure math: compute decayed reputation given elapsed seconds.
/// Separated from HDK for testability.
///
/// Formula: R_decayed = R_floor + (R_stored - R_floor) * decay^elapsed_intervals
/// where elapsed_intervals = elapsed_seconds / DECAY_INTERVAL_SECONDS
pub(crate) fn compute_decayed_reputation(stored_score: f32, elapsed_seconds: i64) -> f32 {
    use crate::config::{REPUTATION_DECAY_FACTOR, REPUTATION_DECAY_INTERVAL_SECONDS, REPUTATION_FLOOR};

    let elapsed_intervals = elapsed_seconds.max(0) as f64 / REPUTATION_DECAY_INTERVAL_SECONDS as f64;

    if elapsed_intervals < 0.01 {
        // Less than ~15 minutes — no meaningful decay
        return stored_score;
    }

    let floor = REPUTATION_FLOOR as f64;
    let stored = stored_score as f64;
    let decayed = floor + (stored - floor) * REPUTATION_DECAY_FACTOR.powf(elapsed_intervals);

    decayed.clamp(floor, 1.0) as f32
}

/// Apply time-based reputation decay lazily on retrieval.
///
/// This avoids background jobs (impossible in Holochain WASM) by computing
/// decay at read time. The stored value is NOT updated here — only callers
/// that write back (e.g. `update_node_reputation_positive`) persist the new value.
pub(crate) fn apply_reputation_decay(rep: &NodeReputation) -> ExternResult<f32> {
    let now_seconds = sys_time()?.0 as i64 / 1_000_000;
    let elapsed_seconds = now_seconds - rep.last_updated;
    Ok(compute_decayed_reputation(rep.reputation_score, elapsed_seconds))
}

/// Get reputation for a node, creating default if not exists.
/// Persists default reputation to DHT on first access for cross-session durability.
/// Uses `node_reputation.{node_id}` path + `NodeToReputation` links (consistent with
/// `update_reputation` and `get_reputation` extern functions).
///
/// Returns the stored reputation with time-based decay applied to the score.
pub(crate) fn get_or_create_reputation(node_id: &str) -> ExternResult<NodeReputation> {
    let rep_path = Path::from(format!("node_reputation.{}", node_id));
    let rep_hash = ensure_path(rep_path.clone(), LinkTypes::NodeToReputation)?;

    let links = get_links(
        LinkQuery::new(
            rep_hash.clone(),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::NodeToReputation as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    // Return existing reputation with decay applied
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(mut rep) = record
                    .entry()
                    .as_option()
                    .and_then(|entry| match entry {
                        Entry::App(bytes) => NodeReputation::try_from(
                            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                        ).ok(),
                        _ => None,
                    })
                {
                    // Apply time-based decay to the returned score
                    rep.reputation_score = apply_reputation_decay(&rep)?;
                    return Ok(rep);
                }
            }
        }
    }

    // Create and persist default reputation to DHT for cross-session persistence
    let reputation = NodeReputation {
        node_id: node_id.to_string(),
        successful_rounds: 0,
        failed_rounds: 0,
        reputation_score: 0.5, // Neutral starting reputation
        last_updated: (sys_time()?.0 as i64) / 1_000_000,
    };

    let hash = create_entry(&EntryTypes::NodeReputation(reputation.clone()))?;
    create_link(rep_hash, hash, LinkTypes::NodeToReputation, vec![])?;

    Ok(reputation)
}
