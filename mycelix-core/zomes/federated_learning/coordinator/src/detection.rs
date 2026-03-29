// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Byzantine detection helpers: feature extraction and hierarchical detection.

use hdk::prelude::*;
use federated_learning_integrity::*;

use sha2::Digest;

use mycelix_sdk::matl::HierarchicalDetector;

use crate::config::*;
// auth items used transitively if needed
use crate::signals::Signal;
use crate::bridge::*;
use crate::pipeline::get_or_create_reputation;
use super::ensure_path;

/// Features extracted from gradient input for Byzantine detection.
/// Used by HierarchicalDetector to classify nodes.
#[derive(Debug, Clone)]
pub struct GradientFeatures {
    /// Node identifier
    pub node_id: String,
    /// Training round
    pub round: u32,
    /// Trust score (from input or computed)
    pub trust_score: f64,
    /// Resource utilization fingerprint (normalized 0-1)
    pub resource_fingerprint: f64,
    /// Network behavior score (based on latency)
    pub network_score: f64,
}

/// Extract features from a gradient input for Byzantine detection.
///
/// Creates a feature vector suitable for HierarchicalDetector analysis.
pub(crate) fn extract_gradient_features(input: &super::gradients::SubmitGradientInput) -> ExternResult<GradientFeatures> {
    // Compute trust score: use provided value or derive from resource metrics
    let trust_score = input.trust_score
        .map(|t| t as f64)
        .unwrap_or_else(|| {
            // Heuristic: combine resource metrics into a trust proxy
            // Normal nodes have: moderate CPU (20-80%), reasonable memory, low latency
            let cpu_score = if input.cpu_usage > 5.0 && input.cpu_usage < 95.0 {
                1.0 - (input.cpu_usage - 50.0).abs() as f64 / 50.0
            } else {
                0.2 // Extreme CPU usage is suspicious
            };

            let latency_score = if input.network_latency_ms < 1000.0 {
                1.0 - (input.network_latency_ms as f64 / 1000.0)
            } else {
                0.1 // High latency is suspicious
            };

            (cpu_score * 0.6 + latency_score * 0.4).max(0.0).min(1.0)
        });

    // Resource fingerprint: normalized combination of resource usage
    let resource_fingerprint = {
        let cpu_norm = (input.cpu_usage as f64 / 100.0).clamp(0.0, 1.0);
        let mem_norm = (input.memory_mb as f64 / 16384.0).clamp(0.0, 1.0); // Assume 16GB max
        let lat_norm = (input.network_latency_ms as f64 / 1000.0).clamp(0.0, 1.0);
        (cpu_norm + mem_norm + lat_norm) / 3.0
    };

    // Network score: lower latency = better
    let network_score = if input.network_latency_ms < 50.0 {
        1.0
    } else if input.network_latency_ms < 200.0 {
        0.8
    } else if input.network_latency_ms < 500.0 {
        0.5
    } else {
        0.2
    };

    Ok(GradientFeatures {
        node_id: input.node_id.clone(),
        round: input.round,
        trust_score,
        resource_fingerprint,
        network_score,
    })
}

/// Extract features from GradientWithPoGQInput for Byzantine detection.
pub(crate) fn extract_pogq_gradient_features(input: &super::matl::GradientWithPoGQInput) -> ExternResult<GradientFeatures> {
    // For PoGQ input, we have explicit quality metrics
    let trust_score = (input.quality * 0.4 + input.consistency * 0.4 + (1.0 - input.entropy.min(1.0)) * 0.2)
        .max(0.0)
        .min(1.0);

    let resource_fingerprint = {
        let cpu_norm = (input.cpu_usage as f64 / 100.0).clamp(0.0, 1.0);
        let mem_norm = (input.memory_mb as f64 / 16384.0).clamp(0.0, 1.0);
        let lat_norm = (input.network_latency_ms as f64 / 1000.0).clamp(0.0, 1.0);
        (cpu_norm + mem_norm + lat_norm) / 3.0
    };

    let network_score = if input.network_latency_ms < 50.0 {
        1.0
    } else if input.network_latency_ms < 200.0 {
        0.8
    } else if input.network_latency_ms < 500.0 {
        0.5
    } else {
        0.2
    };

    Ok(GradientFeatures {
        node_id: input.node_id.clone(),
        round: input.round,
        trust_score,
        resource_fingerprint,
        network_score,
    })
}

/// Extract features from SubmitCompressedGradientInput for Byzantine detection.
pub(crate) fn extract_compressed_gradient_features(input: &super::hyperfeel::SubmitCompressedGradientInput) -> ExternResult<GradientFeatures> {
    // Use quality_score as primary trust indicator
    let trust_score = input.quality_score as f64;

    let resource_fingerprint = {
        let cpu_norm = (input.cpu_usage as f64 / 100.0).clamp(0.0, 1.0);
        let mem_norm = (input.memory_mb as f64 / 16384.0).clamp(0.0, 1.0);
        let lat_norm = (input.network_latency_ms as f64 / 1000.0).clamp(0.0, 1.0);
        (cpu_norm + mem_norm + lat_norm) / 3.0
    };

    let network_score = if input.network_latency_ms < 50.0 {
        1.0
    } else if input.network_latency_ms < 200.0 {
        0.8
    } else if input.network_latency_ms < 500.0 {
        0.5
    } else {
        0.2
    };

    Ok(GradientFeatures {
        node_id: input.node_id.clone(),
        round: input.round,
        trust_score,
        resource_fingerprint,
        network_score,
    })
}

/// Compute evidence hash for audit trail of Byzantine detection.
///
/// Creates a cryptographic hash of the detection evidence for immutable recording.
pub(crate) fn compute_evidence_hash(features: &GradientFeatures, detection_confidence: f32) -> String {
    let evidence = serde_json::json!({
        "node_id": features.node_id,
        "round": features.round,
        "trust_score": features.trust_score,
        "resource_fingerprint": features.resource_fingerprint,
        "network_score": features.network_score,
        "detection_confidence": detection_confidence,
        "threshold": BYZANTINE_REJECTION_THRESHOLD,
    });

    format!("{:x}", sha2::Sha256::digest(evidence.to_string().as_bytes()))
}

/// Run Byzantine detection on gradient features.
///
/// Uses HierarchicalDetector to analyze the gradient and returns
/// (is_byzantine, confidence) tuple.
pub(crate) fn run_byzantine_detection(features: &GradientFeatures) -> (bool, f32) {
    // Create detector and analyze
    let mut detector = HierarchicalDetector::new(
        DETECTION_HIERARCHY_LEVELS,
        DETECTION_MIN_CLUSTER_SIZE,
    );

    // Assign this node based on its trust score
    detector.assign(&features.node_id, features.trust_score);

    // Check if node falls into a Byzantine cluster
    let is_in_byzantine_cluster = detector.is_in_byzantine_cluster(&features.node_id);

    // Compute confidence based on how far below threshold the trust score is
    let confidence = if features.trust_score < MIN_TRUST_SCORE_THRESHOLD {
        // Very low trust = high confidence it's Byzantine
        ((MIN_TRUST_SCORE_THRESHOLD - features.trust_score) / MIN_TRUST_SCORE_THRESHOLD) as f32
    } else if is_in_byzantine_cluster {
        // In Byzantine cluster but score not super low
        0.5
    } else {
        // Not detected as Byzantine
        0.0
    };

    // Consider Byzantine if trust is below threshold OR in Byzantine cluster with enough confidence
    let is_byzantine = features.trust_score < MIN_TRUST_SCORE_THRESHOLD
        || (is_in_byzantine_cluster && confidence >= BYZANTINE_REJECTION_THRESHOLD);

    (is_byzantine, confidence.min(1.0))
}

/// Record Byzantine behavior internally (without requiring detector role).
///
/// This is called automatically during gradient submission when Byzantine
/// behavior is detected, before rejecting the gradient.
pub(crate) fn record_byzantine_internal(
    node_id: &str,
    round: u32,
    detection_method: &str,
    confidence: f32,
    evidence_hash: &str,
) -> ExternResult<ActionHash> {
    let timestamp = sys_time()?;
    let detected_at = timestamp.0 as i64 / 1_000_000;

    let record = ByzantineRecord {
        node_id: node_id.to_string(),
        round,
        detection_method: detection_method.to_string(),
        confidence,
        evidence_hash: evidence_hash.to_string(),
        detected_at,
    };

    let action_hash = create_entry(&EntryTypes::ByzantineRecord(record))?;

    // Link to round for querying
    let round_path = Path::from(format!("round_byzantine.{}", round));
    let round_entry_hash = ensure_path(round_path, LinkTypes::RoundToByzantine)?;
    create_link(
        round_entry_hash,
        action_hash.clone(),
        LinkTypes::RoundToByzantine,
        vec![],
    )?;

    // Update reputation via Bridge (non-blocking)
    let negative_score = (1.0 - confidence).max(0.0).min(1.0) as f64;
    let current_rep = get_or_create_reputation(node_id).unwrap_or(NodeReputation {
        node_id: node_id.to_string(),
        successful_rounds: 0,
        failed_rounds: 0,
        reputation_score: 0.5,
        last_updated: detected_at,
    });

    let _ = call_bridge_record_reputation(
        node_id,
        negative_score,
        current_rep.successful_rounds as u64,
        current_rep.failed_rounds as u64 + 1,
        Some(evidence_hash.to_string()),
    );

    // Broadcast Byzantine detection event
    let event_payload = serde_json::json!({
        "node_id": node_id,
        "round": round,
        "detection_method": detection_method,
        "confidence": confidence,
        "evidence_hash": evidence_hash,
        "detected_at": detected_at,
        "rejected": true,
    });

    let _ = call_bridge_broadcast_event(
        EVENT_BYZANTINE_DETECTED,
        &event_payload,
        2, // Critical priority
    );

    // Emit local signal
    emit_signal(Signal::ByzantineDetected {
        node_id: node_id.to_string(),
        round,
        confidence,
        source: None,
        signature: None,
    })?;

    Ok(action_hash)
}
