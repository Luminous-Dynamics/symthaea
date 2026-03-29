// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MATL SDK Integration - Proof of Gradient Quality & Byzantine Detection.

use hdk::prelude::*;
use federated_learning_integrity::*;

use sha2::Digest;

use mycelix_sdk::matl::{
    ProofOfGradientQuality,
    ReputationScore,
    HierarchicalDetector,
    CartelDetector,
    DEFAULT_BYZANTINE_THRESHOLD,
    MAX_BYZANTINE_TOLERANCE,
    PoGQv41Config,
    PoGQv41Enhanced,
};

use crate::config::*;
use crate::auth::*;
use crate::detection::*;
use crate::signals::Signal;
use crate::gradients::{get_round_gradients, get_node_gradients, get_round_byzantine_records, UpdateReputationInput, get_reputation};
use super::ensure_path;

/// Input for gradient with PoGQ computation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GradientWithPoGQInput {
    pub node_id: String,
    pub round: u32,
    pub gradient_hash: String,
    pub cpu_usage: f32,
    pub memory_mb: f32,
    pub network_latency_ms: f32,
    /// Gradient quality score (0.0-1.0) - how good is this gradient
    pub quality: f64,
    /// Consistency with previous submissions (0.0-1.0)
    pub consistency: f64,
    /// Information entropy of the gradient
    pub entropy: f64,
    /// Optional model architecture hash for version validation
    #[serde(default)]
    pub architecture_hash: Option<String>,
}

/// Result of PoGQ evaluation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PoGQResult {
    pub action_hash: ActionHash,
    pub pogq: PoGQData,
    pub composite_score: f64,
    pub is_byzantine: bool,
}

/// Serializable PoGQ data
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PoGQData {
    pub quality: f64,
    pub consistency: f64,
    pub entropy: f64,
    pub timestamp: u64,
}

/// Submit a gradient with automatic PoGQ computation
///
/// This is the MATL-enhanced version of `submit_gradient` that:
/// 1. Computes Proof of Gradient Quality
/// 2. Looks up node reputation
/// 3. Calculates composite trust score
/// 4. Detects Byzantine behavior and REJECTS malicious gradients
#[hdk_extern]
pub fn submit_gradient_with_pogq(input: GradientWithPoGQInput) -> ExternResult<PoGQResult> {
    // F-01: Validate node_id
    validate_node_id(&input.node_id)?;
    // F-06: Check rate limit
    check_rate_limit("submit_gradient_with_pogq")?;

    // 1. Create PoGQ from input (use with_timestamp + sys_time since SystemTime::now is unavailable in WASM)
    let pogq_timestamp = sys_time()?.0 as u64 / 1_000_000; // microseconds -> seconds
    let pogq = ProofOfGradientQuality::with_timestamp(input.quality, input.consistency, input.entropy, pogq_timestamp);

    // 2. Get node reputation (or default to 0.5 for new nodes)
    let reputation_score = get_reputation(input.node_id.clone())?
        .map(|r| r.reputation_score as f64)
        .unwrap_or(0.5);

    // 3. Calculate composite trust score
    let composite_score = pogq.composite_score(reputation_score);

    // 4. Check for Byzantine behavior using PoGQ threshold
    let pogq_is_byzantine = pogq.is_byzantine(DEFAULT_BYZANTINE_THRESHOLD);

    // ==========================================================================
    // BYZANTINE DETECTION: Combined PoGQ + Hierarchical detection
    // ==========================================================================

    // Extract features for hierarchical detection
    let gradient_features = extract_pogq_gradient_features(&input)?;

    // Run hierarchical detection
    let (hierarchical_is_byzantine, hierarchical_confidence) = run_byzantine_detection(&gradient_features);

    // Combined detection: Byzantine if EITHER method flags it
    let combined_is_byzantine = pogq_is_byzantine || hierarchical_is_byzantine;

    // Calculate combined confidence
    let pogq_confidence = if pogq_is_byzantine {
        (DEFAULT_BYZANTINE_THRESHOLD - pogq.quality).abs() as f32
    } else {
        0.0
    };
    let combined_confidence = pogq_confidence.max(hierarchical_confidence);

    // If Byzantine detected with sufficient confidence, REJECT the gradient
    if combined_is_byzantine && combined_confidence >= BYZANTINE_REJECTION_THRESHOLD {
        // Compute evidence hash for audit trail
        let evidence_hash = compute_evidence_hash(&gradient_features, combined_confidence);

        // Record Byzantine behavior
        record_byzantine_internal(
            &input.node_id,
            input.round,
            "combined_pogq_hierarchical",
            combined_confidence,
            &evidence_hash,
        )?;

        // REJECT: Do not store the gradient
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Byzantine gradient detected and rejected: node={}, round={}, confidence={:.2}, pogq_quality={:.2}",
                input.node_id, input.round, combined_confidence, pogq.quality
            )
        )));
    }

    // ==========================================================================
    // GRADIENT ACCEPTED: Store as normal
    // ==========================================================================

    // 5. Create the gradient entry with computed trust
    let timestamp = sys_time()?;
    let gradient = ModelGradient {
        node_id: input.node_id.clone(),
        round: input.round,
        gradient_hash: input.gradient_hash,
        timestamp: timestamp.0 as i64 / 1_000_000,
        cpu_usage: input.cpu_usage,
        memory_mb: input.memory_mb,
        network_latency_ms: input.network_latency_ms,
        trust_score: Some(composite_score as f32),
    };

    let action_hash = create_entry(&EntryTypes::ModelGradient(gradient.clone()))?;

    // 6. Create links
    let round_path = Path::from(format!("round.{}", input.round));
    let round_entry_hash = ensure_path(round_path, LinkTypes::RoundToGradients)?;
    create_link(
        round_entry_hash,
        action_hash.clone(),
        LinkTypes::RoundToGradients,
        vec![],
    )?;

    let node_path = Path::from(format!("node.{}", input.node_id.clone()));
    let node_entry_hash = ensure_path(node_path, LinkTypes::NodeToGradients)?;
    create_link(
        node_entry_hash,
        action_hash.clone(),
        LinkTypes::NodeToGradients,
        vec![],
    )?;

    // 7. If Byzantine detected but below rejection threshold, record for monitoring
    // (This handles borderline cases that passed but should be tracked)
    if combined_is_byzantine && combined_confidence < BYZANTINE_REJECTION_THRESHOLD {
        let evidence = serde_json::json!({
            "pogq_quality": pogq.quality,
            "pogq_consistency": pogq.consistency,
            "composite_score": composite_score,
            "threshold": DEFAULT_BYZANTINE_THRESHOLD,
            "rejection_threshold": BYZANTINE_REJECTION_THRESHOLD,
            "confidence": combined_confidence,
            "accepted": true,  // Mark as accepted despite detection
        });

        let byzantine_record = ByzantineRecord {
            node_id: input.node_id.clone(),
            round: input.round,
            detection_method: "MATL_PoGQ_monitoring".to_string(),
            confidence: combined_confidence,
            evidence_hash: format!("{:x}", sha2::Sha256::digest(evidence.to_string().as_bytes())),
            detected_at: timestamp.0 as i64 / 1_000_000,
        };

        create_entry(&EntryTypes::ByzantineRecord(byzantine_record.clone()))?;

        emit_signal(Signal::ByzantineDetected {
            node_id: input.node_id.clone(),
            round: input.round,
            confidence: combined_confidence,
            source: None,
            signature: None,
        })?;
    }

    // 8. Emit success signal
    emit_signal(Signal::GradientSubmitted {
        node_id: input.node_id,
        round: input.round,
        action_hash: action_hash.clone(),
        source: None,
        signature: None,
    })?;

    Ok(PoGQResult {
        action_hash,
        pogq: PoGQData {
            quality: pogq.quality,
            consistency: pogq.consistency,
            entropy: pogq.entropy,
            timestamp: pogq.timestamp,
        },
        composite_score,
        is_byzantine: combined_is_byzantine,
    })
}

// =============================================================================
// PoGQ v4.1 Enhanced - Warm-up, Hysteresis, EMA Detection
// =============================================================================

/// Input for gradient with PoGQ v4.1 evaluation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GradientWithPoGQv41Input {
    pub node_id: String,
    pub round: u32,
    pub gradient_hash: String,
    pub cpu_usage: f32,
    pub memory_mb: f32,
    pub network_latency_ms: f32,
    /// Gradient quality score (0.0-1.0)
    pub quality: f32,
    /// Consistency with previous submissions (0.0-1.0)
    pub consistency: f32,
    /// Optional direction score (cosine similarity with reference)
    pub direction_score: Option<f32>,
    /// Optional custom configuration (uses defaults if None)
    pub config: Option<PoGQv41ConfigInput>,
    /// Optional model architecture hash for version validation
    #[serde(default)]
    pub architecture_hash: Option<String>,
}

/// Serializable PoGQ v4.1 configuration input
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PoGQv41ConfigInput {
    pub ema_beta: Option<f32>,
    pub warmup_rounds: Option<u32>,
    pub hysteresis_k: Option<u32>,
    pub hysteresis_m: Option<u32>,
    pub egregious_threshold: Option<f32>,
    pub byzantine_threshold: Option<f32>,
}

/// Result of PoGQ v4.1 evaluation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PoGQv41Result {
    pub action_hash: ActionHash,
    pub evaluation: PoGQv41EvaluationData,
    pub statistics: PoGQv41Stats,
}

/// Serializable evaluation data
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PoGQv41EvaluationData {
    pub client_id: String,
    pub round: u32,
    pub raw_score: f32,
    pub ema_score: f32,
    pub final_score: f32,
    pub is_byzantine: bool,
    pub is_quarantined: bool,
    pub in_warmup: bool,
    pub rejection_reason: Option<String>,
    pub confidence: f32,
}

/// Serializable detection statistics
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PoGQv41Stats {
    pub total_clients: usize,
    pub quarantined_clients: usize,
    pub clients_in_warmup: usize,
    pub average_score: f32,
    pub current_round: u32,
}

/// Submit a gradient with PoGQ v4.1 Enhanced evaluation
#[hdk_extern]
pub fn submit_gradient_with_pogq_v41(input: GradientWithPoGQv41Input) -> ExternResult<PoGQv41Result> {
    // F-01: Validate node_id
    validate_node_id(&input.node_id)?;
    // F-06: Check rate limit
    check_rate_limit("submit_gradient_with_pogq_v41")?;

    // Build configuration from input or use defaults
    let config = if let Some(cfg) = input.config {
        PoGQv41Config {
            ema_beta: cfg.ema_beta.unwrap_or(0.85),
            warmup_rounds: cfg.warmup_rounds.unwrap_or(3),
            hysteresis_k: cfg.hysteresis_k.unwrap_or(2),
            hysteresis_m: cfg.hysteresis_m.unwrap_or(3),
            egregious_threshold: cfg.egregious_threshold.unwrap_or(0.15),
            byzantine_threshold: cfg.byzantine_threshold.unwrap_or(0.5),
            min_acceptance_score: 0.3,
            direction_prefilter: input.direction_score.is_some(),
            direction_threshold: 0.0,
        }
    } else {
        PoGQv41Config::default()
    };

    // Create detector and set round
    let mut detector = PoGQv41Enhanced::new(config);
    detector.set_round(input.round);

    // Load historical client state from previous gradients
    if let Ok(client_gradients) = get_node_gradients(input.node_id.clone()) {
        for (_, grad) in client_gradients.iter() {
            let historical_quality = grad.trust_score.unwrap_or(0.5);
            let _ = detector.evaluate(
                &input.node_id,
                historical_quality,
                historical_quality,
                None,
            );
        }
    }

    // Evaluate the current gradient
    let evaluation = detector.evaluate(
        &input.node_id,
        input.quality,
        input.consistency,
        input.direction_score,
    );

    // If Byzantine/quarantined, reject the gradient
    if evaluation.is_byzantine || evaluation.is_quarantined {
        let evidence_hash = format!("{:x}", sha2::Sha256::digest(
            format!("pogq_v41:{}:{}:{}:{:.3}",
                input.node_id, input.round,
                evaluation.rejection_reason.as_deref().unwrap_or("quarantined"),
                evaluation.confidence
            ).as_bytes()
        ));

        record_byzantine_internal(
            &input.node_id,
            input.round,
            "pogq_v41_enhanced",
            evaluation.confidence,
            &evidence_hash,
        )?;

        if evaluation.is_quarantined {
            emit_signal(Signal::ByzantineDetected {
                node_id: input.node_id.clone(),
                round: input.round,
                confidence: evaluation.confidence,
                source: None,
                signature: None,
            })?;
        }

        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "PoGQ v4.1: Gradient rejected - node={}, round={}, reason={}, quarantined={}",
                input.node_id, input.round,
                evaluation.rejection_reason.as_deref().unwrap_or("policy violation"),
                evaluation.is_quarantined
            )
        )));
    }

    // Gradient accepted - create entry
    let timestamp = sys_time()?;
    let gradient = ModelGradient {
        node_id: input.node_id.clone(),
        round: input.round,
        gradient_hash: input.gradient_hash,
        timestamp: timestamp.0 as i64 / 1_000_000,
        cpu_usage: input.cpu_usage,
        memory_mb: input.memory_mb,
        network_latency_ms: input.network_latency_ms,
        trust_score: Some(evaluation.final_score),
    };

    let action_hash = create_entry(&EntryTypes::ModelGradient(gradient.clone()))?;

    // Create links
    let round_path = Path::from(format!("round.{}", input.round));
    let round_entry_hash = ensure_path(round_path, LinkTypes::RoundToGradients)?;
    create_link(
        round_entry_hash,
        action_hash.clone(),
        LinkTypes::RoundToGradients,
        vec![],
    )?;

    let node_path = Path::from(format!("node.{}", input.node_id.clone()));
    let node_entry_hash = ensure_path(node_path, LinkTypes::NodeToGradients)?;
    create_link(
        node_entry_hash,
        action_hash.clone(),
        LinkTypes::NodeToGradients,
        vec![],
    )?;

    // Emit success signal
    emit_signal(Signal::GradientSubmitted {
        node_id: input.node_id.clone(),
        round: input.round,
        action_hash: action_hash.clone(),
        source: None,
        signature: None,
    })?;

    // Get statistics
    let stats = detector.statistics();

    Ok(PoGQv41Result {
        action_hash,
        evaluation: PoGQv41EvaluationData {
            client_id: evaluation.client_id,
            round: evaluation.round,
            raw_score: evaluation.raw_score,
            ema_score: evaluation.ema_score,
            final_score: evaluation.final_score,
            is_byzantine: evaluation.is_byzantine,
            is_quarantined: evaluation.is_quarantined,
            in_warmup: evaluation.in_warmup,
            rejection_reason: evaluation.rejection_reason,
            confidence: evaluation.confidence,
        },
        statistics: PoGQv41Stats {
            total_clients: stats.total_clients,
            quarantined_clients: stats.quarantined_clients,
            clients_in_warmup: stats.clients_in_warmup,
            average_score: stats.average_score,
            current_round: stats.current_round,
        },
    })
}

/// Get PoGQ v4.1 detection statistics for a round
#[hdk_extern]
pub fn get_pogq_v41_statistics(round: u32) -> ExternResult<PoGQv41Stats> {
    let config = PoGQv41Config::default();
    let mut detector = PoGQv41Enhanced::new(config);
    detector.set_round(round);

    let gradients = get_round_gradients(round)?;
    for (_, gradient) in gradients {
        let quality = gradient.trust_score.unwrap_or(0.5);
        let _ = detector.evaluate(&gradient.node_id, quality, quality, None);
    }

    let stats = detector.statistics();
    Ok(PoGQv41Stats {
        total_clients: stats.total_clients,
        quarantined_clients: stats.quarantined_clients,
        clients_in_warmup: stats.clients_in_warmup,
        average_score: stats.average_score,
        current_round: stats.current_round,
    })
}

/// Input for hierarchical Byzantine detection
#[derive(Serialize, Deserialize, Debug)]
pub struct DetectByzantineInput {
    pub round: u32,
    /// Detection levels in hierarchy
    pub levels: usize,
    /// Minimum cluster size
    pub min_cluster_size: usize,
}

/// Result of hierarchical Byzantine detection
#[derive(Serialize, Deserialize, Debug)]
pub struct ByzantineDetectionResult {
    pub round: u32,
    pub suspected_nodes: Vec<String>,
    pub byzantine_fraction: f64,
    pub within_tolerance: bool,
    pub cluster_count: usize,
}

/// Detect Byzantine nodes using hierarchical clustering
#[hdk_extern]
pub fn detect_byzantine_hierarchical(input: DetectByzantineInput) -> ExternResult<ByzantineDetectionResult> {
    let gradients = get_round_gradients(input.round)?;
    let mut detector = HierarchicalDetector::new(input.levels, input.min_cluster_size);

    for (_, gradient) in &gradients {
        let score = gradient.trust_score.unwrap_or(0.5) as f64;
        detector.assign(&gradient.node_id, score);
    }

    let suspected_nodes = detector.get_suspected_byzantine();
    let byzantine_fraction = detector.byzantine_fraction();
    let within_tolerance = byzantine_fraction <= MAX_BYZANTINE_TOLERANCE;

    Ok(ByzantineDetectionResult {
        round: input.round,
        suspected_nodes,
        byzantine_fraction,
        within_tolerance,
        cluster_count: detector.cluster_count(),
    })
}

/// Input for cartel detection
#[derive(Serialize, Deserialize, Debug)]
pub struct DetectCartelInput {
    pub round: u32,
    pub correlation_threshold: f64,
    pub min_cartel_size: usize,
}

/// Result of cartel detection
#[derive(Serialize, Deserialize, Debug)]
pub struct CartelDetectionResult {
    pub round: u32,
    pub cartels: Vec<Vec<String>>,
    pub total_cartel_nodes: usize,
    pub cartel_fraction: f64,
}

/// Detect cartels (colluding nodes) in the network
#[hdk_extern]
pub fn detect_cartels(input: DetectCartelInput) -> ExternResult<CartelDetectionResult> {
    let gradients = get_round_gradients(input.round)?;
    let mut cartel_detector = CartelDetector::new(input.correlation_threshold, input.min_cartel_size);

    let gradient_vec: Vec<_> = gradients.iter().collect();
    for i in 0..gradient_vec.len() {
        for j in (i + 1)..gradient_vec.len() {
            let (_, g_i) = &gradient_vec[i];
            let (_, g_j) = &gradient_vec[j];

            let fp_i = [
                g_i.cpu_usage as f64,
                g_i.memory_mb as f64,
                g_i.network_latency_ms as f64,
                g_i.trust_score.unwrap_or(0.5) as f64,
            ];
            let fp_j = [
                g_j.cpu_usage as f64,
                g_j.memory_mb as f64,
                g_j.network_latency_ms as f64,
                g_j.trust_score.unwrap_or(0.5) as f64,
            ];

            let similarity = cosine_similarity(&fp_i, &fp_j);
            cartel_detector.record_similarity(&g_i.node_id, &g_j.node_id, similarity);
        }
    }

    let detected = cartel_detector.detect();
    let cartels: Vec<Vec<String>> = detected
        .iter()
        .map(|c| c.members.iter().cloned().collect())
        .collect();
    let total_cartel_nodes: usize = cartels.iter().map(|c| c.len()).sum();
    let cartel_fraction = if gradients.is_empty() {
        0.0
    } else {
        total_cartel_nodes as f64 / gradients.len() as f64
    };

    Ok(CartelDetectionResult {
        round: input.round,
        cartels,
        total_cartel_nodes,
        cartel_fraction,
    })
}

/// Compute cosine similarity between two vectors
pub(crate) fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

/// Compute reputation update based on round participation
#[hdk_extern]
pub fn update_reputation_with_matl(input: UpdateReputationInput) -> ExternResult<ActionHash> {
    // SECURITY: Only coordinators can update reputation scores
    require_coordinator_role()?;

    let mut rep = ReputationScore::new(&input.node_id, "federated_learning");

    for _ in 0..input.successful_rounds {
        rep.record_positive();
    }
    for _ in 0..input.failed_rounds {
        rep.record_negative();
    }

    let matl_score = rep.score as f32;

    let timestamp = sys_time()?;
    let node_path = Path::from(format!("node_reputation.{}", input.node_id.clone()));
    let node_entry_hash = ensure_path(node_path.clone(), LinkTypes::NodeToReputation)?;

    let existing_links = get_links(
        LinkQuery::new(
            node_entry_hash.clone(),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::NodeToReputation as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let reputation = NodeReputation {
        node_id: input.node_id.clone(),
        successful_rounds: input.successful_rounds,
        failed_rounds: input.failed_rounds,
        reputation_score: matl_score,
        last_updated: timestamp.0 as i64 / 1_000_000,
    };

    let action_hash = if let Some(link) = existing_links.first() {
        if let Some(old_hash) = link.target.clone().into_action_hash() {
            update_entry(old_hash, &EntryTypes::NodeReputation(reputation))?
        } else {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Invalid link target".to_string()
            )));
        }
    } else {
        let hash = create_entry(&EntryTypes::NodeReputation(reputation))?;
        create_link(node_entry_hash, hash.clone(), LinkTypes::NodeToReputation, vec![])?;
        hash
    };

    Ok(action_hash)
}

/// Get MATL statistics for a round
#[derive(Serialize, Deserialize, Debug)]
pub struct RoundMATLStats {
    pub round: u32,
    pub participant_count: usize,
    pub average_quality: f64,
    pub average_consistency: f64,
    pub average_composite: f64,
    pub byzantine_count: usize,
    pub byzantine_fraction: f64,
    pub within_tolerance: bool,
}

/// Get comprehensive MATL statistics for a training round
#[hdk_extern]
pub fn get_round_matl_stats(round: u32) -> ExternResult<RoundMATLStats> {
    let gradients = get_round_gradients(round)?;
    let byzantine_records = get_round_byzantine_records(round)?;

    if gradients.is_empty() {
        return Ok(RoundMATLStats {
            round,
            participant_count: 0,
            average_quality: 0.0,
            average_consistency: 0.0,
            average_composite: 0.0,
            byzantine_count: 0,
            byzantine_fraction: 0.0,
            within_tolerance: true,
        });
    }

    let total_composite: f64 = gradients.iter()
        .map(|(_, g)| g.trust_score.unwrap_or(0.5) as f64)
        .sum();

    let average_composite = total_composite / gradients.len() as f64;
    let byzantine_count = byzantine_records.len();
    let byzantine_fraction = byzantine_count as f64 / gradients.len() as f64;

    Ok(RoundMATLStats {
        round,
        participant_count: gradients.len(),
        average_quality: average_composite,
        average_consistency: average_composite,
        average_composite,
        byzantine_count,
        byzantine_fraction,
        within_tolerance: byzantine_fraction <= MAX_BYZANTINE_TOLERANCE,
    })
}
