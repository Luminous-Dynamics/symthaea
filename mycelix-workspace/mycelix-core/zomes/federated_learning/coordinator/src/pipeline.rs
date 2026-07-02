// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified pipeline integration with mycelix-fl for decentralized aggregation.

use hdk::prelude::*;
use federated_learning_integrity::*;

use mycelix_fl::pipeline::{DecentralizedPipeline, PipelineConfig};
use mycelix_fl::types::{CompressedGradient, GradientMetadata as FlGradientMetadata};
use mycelix_fl::holochain::{
    ZomeAggregationCommitment, ZomeAggregationReveal, DetectionSummary as FlDetectionSummary,
    to_zome_commitment,
};
use mycelix_fl::fl_core::{ShapleyCalculator, ShapleyConfig, ReplayDetector, ReplayDetectorConfig,
    AdaptiveDefenseManager, AdaptiveDefenseConfig};
use mycelix_sdk::hyperfeel::HV16_BYTES;

use crate::signals::Signal;
use super::ensure_path;
use crate::hyperfeel::get_round_hypervectors;
use crate::consensus::get_active_validators_internal;

/// Convert HV16 binary bytes to bipolar f32: bit=1 → +1.0, bit=0 → -1.0
///
/// Used by Shapley computation, replay detection, and HV similarity comparison.
/// 2048 bytes (HV16) → 16384 f32 values.
pub(crate) fn hv16_to_bipolar(hv_bytes: &[u8]) -> Vec<f32> {
    hv_bytes.iter()
        .flat_map(|byte| (0..8).rev().map(move |i| {
            if (byte >> i) & 1 == 1 { 1.0f32 } else { -1.0f32 }
        }))
        .collect()
}

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

    // Step 2.5: Check for gradient replay attacks (cross-round + same-round)
    let mut replay_excluded = Vec::new();
    {
        let mut replay_detector = ReplayDetector::new(ReplayDetectorConfig::default());

        // Load prior round fingerprints from DHT for cross-round detection
        let lookback_rounds = 5u32;
        if let Ok(prior_fps) = load_gradient_fingerprints(
            round.saturating_sub(lookback_rounds),
            round.saturating_sub(1),
        ) {
            for fp in &prior_fps {
                // Reconstruct a synthetic gradient from statistics for near-replay detection
                // The hash-based detection works purely from the SHA-256 fingerprint
                replay_detector.record_submission(
                    &fp.node_id,
                    &[fp.l2_norm, fp.mean, fp.std_dev],
                    fp.round,
                );
            }
        }

        // Check current round's gradients against history + each other
        for gradient in &compressed_gradients {
            let gradient_f32 = hv16_to_bipolar(&gradient.hv_data);
            let check = replay_detector.check_replay(
                &gradient.participant_id,
                &gradient_f32,
                round as u64,
            );
            if check.is_replay {
                replay_excluded.push(gradient.participant_id.clone());
            }
        }

        // Persist this round's fingerprints for future rounds (best effort)
        for gradient in &compressed_gradients {
            if !replay_excluded.contains(&gradient.participant_id) {
                let gradient_f32 = hv16_to_bipolar(&gradient.hv_data);
                let fp = replay_detector.compute_fingerprint(&gradient_f32);
                let _ = store_gradient_fingerprint(
                    &gradient.participant_id,
                    round as u64,
                    &fp,
                );
            }
        }
    }
    // Remove replay-detected gradients before aggregation
    if !replay_excluded.is_empty() {
        let excluded_set: std::collections::HashSet<&str> = replay_excluded.iter()
            .map(|s| s.as_str()).collect();
        compressed_gradients.retain(|g| !excluded_set.contains(g.participant_id.as_str()));
    }

    if compressed_gradients.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "All gradients excluded by replay detection".to_string()
        )));
    }

    // Step 2.7: Adaptive defense — select aggregation strategy based on recent Byzantine rates
    let mut defense_manager = AdaptiveDefenseManager::new(AdaptiveDefenseConfig::default());
    // Feed historical Byzantine rates from recent rounds
    // Track DHT failures to apply fallback escalation
    let mut dht_history_failures = 0u32;
    let mut dht_history_loaded = 0u32;
    if round > 0 {
        let lookback = round.min(5);
        for prev_round in (round.saturating_sub(lookback))..round {
            match (
                crate::gradients::get_round_byzantine_records(prev_round),
                crate::hyperfeel::get_round_hypervectors(prev_round),
            ) {
                (Ok(byz_records), Ok(prev_hvs)) => {
                    defense_manager.record_round_result(byz_records.len(), prev_hvs.len());
                    dht_history_loaded += 1;
                }
                _ => {
                    dht_history_failures += 1;
                }
            }
        }
    }
    // Let adaptive defense escalate detection thresholds based on recent Byzantine rates
    let _adapted_method = defense_manager.adapt();
    let mut defense_stats = defense_manager.get_stats();

    // Fallback escalation: if DHT reads failed and we couldn't load history,
    // assume at least level 1 defense to avoid being lenient under uncertainty.
    if dht_history_failures > 0 && dht_history_loaded == 0 && defense_stats.escalation_level == 0 {
        defense_stats.escalation_level = 1;
    }

    // Step 3: Configure pipeline with defense-adjusted thresholds
    // Higher escalation → stricter cosine filtering, higher reputation bar
    let mut pipeline_config = PipelineConfig::default();
    match defense_stats.escalation_level {
        0 => {} // Level 0: defaults (cosine=0.1, rep=0.3, confidence=0.7)
        1 => {
            pipeline_config.cosine_threshold = 0.15;
            pipeline_config.method = format!("AdaptiveHV-L1-{:?}", defense_stats.current_method);
        }
        2 => {
            pipeline_config.cosine_threshold = 0.2;
            pipeline_config.reputation_threshold = 0.4;
            pipeline_config.method = format!("AdaptiveHV-L2-{:?}", defense_stats.current_method);
        }
        3 => {
            pipeline_config.cosine_threshold = 0.25;
            pipeline_config.reputation_threshold = 0.5;
            pipeline_config.confidence_threshold = 0.6;
            pipeline_config.method = format!("AdaptiveHV-L3-{:?}", defense_stats.current_method);
        }
        _ => {
            // Level 4+: maximum defense
            pipeline_config.cosine_threshold = 0.3;
            pipeline_config.reputation_threshold = 0.6;
            pipeline_config.confidence_threshold = 0.5;
            pipeline_config.max_byzantine_fraction = 0.34;
            pipeline_config.method = format!("AdaptiveHV-L{}-{:?}", defense_stats.escalation_level, defense_stats.current_method);
        }
    }
    let pipeline = DecentralizedPipeline::new(pipeline_config);
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

    // Step 5: Build detection summary from pipeline result (includes replay-excluded)
    let mut flagged_nodes: Vec<(String, f32)> = result.stats.excluded_participants.iter()
        .map(|id| (id.clone(), 1.0_f32))
        .collect();
    for replay_id in &replay_excluded {
        if !flagged_nodes.iter().any(|(id, _)| id == replay_id) {
            flagged_nodes.push((replay_id.clone(), 1.0_f32));
        }
    }
    let mut detection_layers = vec!["MultiSignal".to_string(), "HvCosineFilter".to_string()];
    if !replay_excluded.is_empty() {
        detection_layers.push("ReplayDetection".to_string());
    }
    let detection_summary = FlDetectionSummary {
        flagged_nodes,
        detection_layers_used: detection_layers,
        total_checked: result.stats.total_contributions + replay_excluded.len(),
        total_flagged: result.stats.byzantine_detected + replay_excluded.len(),
    };

    // Step 5.5: Record coherence data point to DHT + run anomaly detection
    {
        use mycelix_fl::coherence_series::{CoherenceTimeSeries, CoherenceTimeSeriesConfig};

        let total = result.stats.total_contributions + replay_excluded.len();
        let flagged = result.stats.byzantine_detected + replay_excluded.len();
        let coherence = if total > 0 {
            1.0 - (flagged as f32 / total as f32)
        } else {
            1.0
        };
        let now_ms = sys_time().map(|t| t.0 as i64 / 1_000).unwrap_or(0);

        // Persist this round's coherence to DHT
        let record = CoherenceRecord {
            round: round as u64,
            coherence_value: coherence,
            epistemic_confidence: 0.8,
            byzantine_count: flagged as u32,
            node_count: total as u32,
            defense_level: defense_stats.escalation_level as u32,
            recorded_at: now_ms,
        };
        if let Ok(hash) = create_entry(&EntryTypes::CoherenceRecord(record)) {
            let series_path = Path::from("coherence_time_series");
            if let Ok(series_hash) = ensure_path(series_path, LinkTypes::CoherenceTimeSeries) {
                let _ = create_link(series_hash, hash, LinkTypes::CoherenceTimeSeries, vec![]);
            }
        }

        // Load recent history from DHT and run anomaly detection
        let mut series = CoherenceTimeSeries::new(
            CoherenceTimeSeriesConfig::default().with_window_size(50)
        );
        if let Ok(history) = load_coherence_history(50) {
            for h in &history {
                series.record(
                    h.round, h.recorded_at, h.coherence_value,
                    h.epistemic_confidence, h.byzantine_count as usize, h.node_count as usize,
                );
            }
        }
        // Record current point into the series for anomaly detection
        series.record(round as u64, now_ms, coherence, 0.8, flagged, total);
        let _anomaly = series.detect_anomaly();
    }

    // Step 6: Return result
    let mut all_excluded = result.stats.excluded_participants;
    all_excluded.extend(replay_excluded);
    let total_excluded = all_excluded.len() as u32;

    Ok(ValidatorPipelineResult {
        commitment_hash,
        aggregated_hv,
        method,
        gradient_count: result.stats.after_detection as u32,
        excluded_count: total_excluded,
        excluded_participants: all_excluded,
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

    // Compute real Shapley values from pipeline participants
    let shapley_values = compute_reveal_shapley_values(
        round,
        &pipeline_result.excluded_participants,
    );

    // Build the reveal using bridge type
    let zome_reveal = ZomeAggregationReveal {
        round: round as u64,
        aggregator: agent.to_string(),
        result_data: pipeline_result.aggregated_hv,
        result_hash: pipeline_result.commitment_hash,
        detection_summary: pipeline_result.detection_summary,
        shapley_values,
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

/// Compute Shapley values for the reveal phase using HV16 hypervector data.
///
/// Fetches round hypervectors and runs ShapleyCalculator (MonteCarlo, 100 samples)
/// for n <= 20 valid participants. Falls back to equal shares for larger groups.
/// Excluded participants receive 0.0.
fn compute_reveal_shapley_values(
    round: u32,
    excluded_participants: &[String],
) -> Vec<(String, f32)> {
    let excluded_set: std::collections::HashSet<&str> = excluded_participants.iter()
        .map(|s| s.as_str()).collect();

    let Ok(hypervectors) = crate::hyperfeel::get_round_hypervectors(round) else {
        // Can't fetch HVs — return zeros for excluded only
        return excluded_participants.iter()
            .map(|id| (id.clone(), 0.0_f32))
            .collect();
    };

    let valid_hvs: Vec<_> = hypervectors.iter()
        .filter(|(id, _)| !excluded_set.contains(id.as_str()))
        .collect();

    if valid_hvs.is_empty() || valid_hvs.len() > 20 {
        // Fallback: equal share for valid, 0 for excluded
        let equal_share = if valid_hvs.is_empty() { 0.0 } else { 1.0 / valid_hvs.len() as f32 };
        let mut values: Vec<(String, f32)> = valid_hvs.iter()
            .map(|(id, _)| (id.clone(), equal_share))
            .collect();
        for id in excluded_participants {
            values.push((id.clone(), 0.0));
        }
        return values;
    }

    // Build gradient map from HV16 bipolar data
    let mut gradient_map: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::new();
    for (node_id, hv_bytes) in &valid_hvs {
        gradient_map.insert((*node_id).clone(), hv16_to_bipolar(hv_bytes));
    }

    // Compute aggregated reference (mean of all valid HVs)
    let dim = gradient_map.values().next().map_or(0, |v| v.len());
    let mut aggregated = vec![0.0f32; dim];
    let count = gradient_map.len() as f32;
    for vals in gradient_map.values() {
        for (i, v) in vals.iter().enumerate() {
            if i < dim {
                aggregated[i] += v / count;
            }
        }
    }

    let config = ShapleyConfig::monte_carlo(100).with_normalization();
    let mut calculator = ShapleyCalculator::new(config);
    let result = calculator.calculate_values(&gradient_map, &aggregated);

    let mut values: Vec<(String, f32)> = result.values.iter()
        .map(|(id, sv)| (id.clone(), *sv))
        .collect();
    for id in excluded_participants {
        if !values.iter().any(|(vid, _)| vid == id) {
            values.push((id.clone(), 0.0));
        }
    }
    values
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

// =============================================================================
// Gradient Fingerprint DHT Operations (Cross-Round Replay Detection)
// =============================================================================

/// Store a gradient fingerprint on DHT for future replay detection.
fn store_gradient_fingerprint(
    node_id: &str,
    round: u64,
    fp: &mycelix_fl::fl_core::GradientFingerprint,
) -> ExternResult<()> {
    let now_ms = sys_time().map(|t| t.0 as i64 / 1_000).unwrap_or(0);
    let hash_hex = fp.hash.iter().map(|b| format!("{:02x}", b)).collect::<String>();

    let entry = GradientFingerprint {
        node_id: node_id.to_string(),
        round,
        hash_hex,
        l2_norm: fp.norm,
        mean: fp.mean,
        std_dev: fp.std_dev,
        submitted_at: now_ms,
    };

    let action_hash = create_entry(&EntryTypes::GradientFingerprint(entry))?;
    let fp_path = Path::from(format!("gradient_fingerprints/{}", round));
    let fp_hash = ensure_path(fp_path, LinkTypes::GradientFingerprints)?;
    create_link(fp_hash, action_hash, LinkTypes::GradientFingerprints, vec![])?;
    Ok(())
}

/// Load gradient fingerprints from DHT for a range of rounds.
fn load_gradient_fingerprints(
    start_round: u32,
    end_round: u32,
) -> ExternResult<Vec<GradientFingerprint>> {
    let mut all_fps = Vec::new();

    for round in start_round..=end_round {
        let fp_path = Path::from(format!("gradient_fingerprints/{}", round));
        let typed = fp_path.typed(LinkTypes::GradientFingerprints)?;
        if !typed.exists()? {
            continue;
        }
        let fp_hash = typed.path_entry_hash()?;

        let links = get_links(
            LinkQuery::new(
                fp_hash,
                LinkTypeFilter::single_type(0.into(), (LinkTypes::GradientFingerprints as u8).into()),
            ),
            GetStrategy::default(),
        )?;

        for link in &links {
            if let Some(action_hash) = link.target.clone().into_action_hash() {
                if let Some(record) = get(action_hash, GetOptions::default())? {
                    if let Some(entry) = record.entry().as_option() {
                        if let Entry::App(bytes) = entry {
                            if let Ok(fp) = GradientFingerprint::try_from(
                                SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                            ) {
                                all_fps.push(fp);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(all_fps)
}

// =============================================================================
// Coherence Time-Series DHT Operations
// =============================================================================

/// Load recent coherence history from DHT, most recent last.
pub(crate) fn load_coherence_history(limit: usize) -> ExternResult<Vec<CoherenceRecord>> {
    let series_path = Path::from("coherence_time_series");
    let typed = series_path.typed(LinkTypes::CoherenceTimeSeries)?;
    if !typed.exists()? {
        return Ok(vec![]);
    }
    let series_hash = typed.path_entry_hash()?;

    let links = get_links(
        LinkQuery::new(
            series_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::CoherenceTimeSeries as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links.iter().rev() {
        if records.len() >= limit {
            break;
        }
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(entry) = record.entry().as_option() {
                    if let Entry::App(bytes) = entry {
                        if let Ok(cr) = CoherenceRecord::try_from(
                            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                        ) {
                            records.push(cr);
                        }
                    }
                }
            }
        }
    }
    // Sort by round ascending for chronological replay into CoherenceTimeSeries
    records.sort_by_key(|r| r.round);
    Ok(records)
}

/// Input for querying coherence time-series
#[derive(Serialize, Deserialize, Debug)]
pub struct GetCoherenceSeriesInput {
    pub start_round: Option<u64>,
    pub end_round: Option<u64>,
    pub limit: Option<usize>,
}

/// Query coherence time-series data from DHT
#[hdk_extern]
pub fn get_coherence_series(input: GetCoherenceSeriesInput) -> ExternResult<Vec<CoherenceRecord>> {
    let limit = input.limit.unwrap_or(100);
    let mut records = load_coherence_history(limit)?;

    // Apply round range filter
    if let Some(start) = input.start_round {
        records.retain(|r| r.round >= start);
    }
    if let Some(end) = input.end_round {
        records.retain(|r| r.round <= end);
    }

    Ok(records)
}

/// Coherence anomaly result returned to callers
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoherenceAnomalyResult {
    pub round: u64,
    pub expected_coherence: f32,
    pub actual_coherence: f32,
    pub z_score: f32,
    pub severity: String,
}

/// Check for coherence anomalies over a recent window
#[hdk_extern]
pub fn check_coherence_anomalies(window_size: u32) -> ExternResult<Vec<CoherenceAnomalyResult>> {
    use mycelix_fl::coherence_series::{CoherenceTimeSeries, CoherenceTimeSeriesConfig};

    let history = load_coherence_history(window_size as usize)?;
    if history.len() < 3 {
        return Ok(vec![]); // Need enough data for statistical significance
    }

    let config = CoherenceTimeSeriesConfig::default()
        .with_window_size(window_size as usize);
    let mut series = CoherenceTimeSeries::new(config);

    let mut anomalies = Vec::new();
    for (i, h) in history.iter().enumerate() {
        series.record(
            h.round, h.recorded_at, h.coherence_value,
            h.epistemic_confidence, h.byzantine_count as usize, h.node_count as usize,
        );

        // Only check anomalies after enough data points (at least 5)
        if i >= 4 {
            if let Some(anomaly) = series.detect_anomaly() {
                anomalies.push(CoherenceAnomalyResult {
                    round: anomaly.round,
                    expected_coherence: anomaly.expected_coherence,
                    actual_coherence: anomaly.actual_coherence,
                    z_score: anomaly.z_score,
                    severity: format!("{:?}", anomaly.severity),
                });
            }
        }
    }

    Ok(anomalies)
}
