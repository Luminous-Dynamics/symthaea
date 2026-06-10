// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! High-level MATL orchestrator.
//!
//! This module provides a small, opinionated façade over the core MATL
//! primitives so that callers can evaluate node contributions in a single
//! call without wiring up all of the adaptive + hierarchical components
//! themselves.
//!
//! The orchestrator is intentionally lightweight:
//! - it does **not** make strong guarantees about optimal thresholds,
//! - but it does keep behaviour explicit and testable.

use std::collections::HashMap;

use super::{
    AdaptiveByzantineThreshold, AdaptiveThresholdManager, CartelDetector, HierarchicalDetector,
    NetworkStatus, ProofOfGradientQuality,
};

/// Per-node evaluation result produced by [`MatlEngine`].
#[derive(Debug, Clone)]
pub struct NodeEvaluation {
    /// Node identifier.
    pub node_id: String,
    /// Composite PoGQ score including reputation.
    pub composite_score: f64,
    /// Adaptive per-node threshold.
    pub node_threshold: f64,
    /// Whether this contribution is anomalous for this node.
    pub is_node_anomalous: bool,
    /// Whether this node is currently in a suspicious cluster.
    pub is_in_suspicious_cluster: bool,
}

/// Network-level evaluation summary.
#[derive(Debug, Clone)]
pub struct NetworkEvaluation {
    /// Estimated Byzantine fraction from hierarchical detector.
    pub estimated_byzantine_fraction: f64,
    /// Current effective Byzantine tolerance threshold.
    pub adaptive_byzantine_threshold: f64,
    /// Network health status.
    pub status: NetworkStatus,
}

/// Minimal orchestration over adaptive thresholds, hierarchical detection,
/// and network-level Byzantine tolerance.
#[derive(Debug)]
pub struct MatlEngine {
    per_node_thresholds: AdaptiveThresholdManager,
    hierarchical: HierarchicalDetector,
    byzantine_threshold: AdaptiveByzantineThreshold,
    cartel_detector: CartelDetector,
    /// Cache of the last composite score per node so callers can inspect state.
    last_scores: HashMap<String, f64>,
}

impl MatlEngine {
    /// Create a new engine with reasonable defaults.
    ///
    /// - `window_size` controls how many observations are kept per node when
    ///   computing adaptive thresholds.
    /// - `hierarchy_levels` and `min_cluster_size` are passed through to the
    ///   hierarchical detector.
    pub fn new(window_size: usize, hierarchy_levels: usize, min_cluster_size: usize) -> Self {
        Self {
            per_node_thresholds: AdaptiveThresholdManager::new(window_size),
            hierarchical: HierarchicalDetector::new(hierarchy_levels, min_cluster_size),
            byzantine_threshold: AdaptiveByzantineThreshold::new(),
            cartel_detector: CartelDetector::new(0.9, 3),
            last_scores: HashMap::new(),
        }
    }

    /// Number of nodes currently tracked.
    pub fn node_count(&self) -> usize {
        self.per_node_thresholds.node_count()
    }

    /// Evaluate a single node contribution and update internal state.
    ///
    /// Callers supply:
    /// - `node_id`
    /// - `pogq` computed for this contribution
    /// - `reputation` in [0, 1]
    ///
    /// The engine:
    /// - updates per-node adaptive thresholds,
    /// - feeds the composite score into the hierarchical detector,
    /// - updates the adaptive Byzantine tolerance estimate.
    pub fn evaluate_node(
        &mut self,
        node_id: &str,
        pogq: &ProofOfGradientQuality,
        reputation: f64,
    ) -> (NodeEvaluation, NetworkEvaluation) {
        let composite = pogq.composite_score(reputation);
        self.per_node_thresholds.observe(node_id, composite);
        let node_threshold = self.per_node_thresholds.threshold(node_id);

        // Track latest score for potential cartel analysis.
        self.last_scores.insert(node_id.to_string(), composite);

        // Feed into hierarchical detector as a simple quality score.
        self.hierarchical.assign(node_id, composite);

        // Estimate Byzantine fraction from hierarchical detector.
        let estimated_frac = self.hierarchical.byzantine_fraction();

        // Simple attack confirmation heuristic: if estimated fraction exceeds
        // 80% of the current adaptive threshold, treat as an attack event.
        let attack_confirmed = {
            let current_threshold = self.byzantine_threshold.threshold();
            estimated_frac > current_threshold * 0.8
        };

        let network_size = self.node_count().max(1);
        self.byzantine_threshold
            .observe(estimated_frac, network_size, attack_confirmed);

        let recommendation = self.byzantine_threshold.recommendation();

        // A node is anomalous if:
        // 1. Score is below its adaptive threshold (anomaly for this node), OR
        // 2. Score is below a global minimum (inherently unacceptable), OR
        // 3. Node is in a suspicious cluster (hierarchical detection)
        const GLOBAL_MIN_SCORE: f64 = 0.3;
        let is_in_suspicious_cluster = self.hierarchical.is_in_byzantine_cluster(node_id);
        let is_node_anomalous =
            composite < node_threshold || composite < GLOBAL_MIN_SCORE || is_in_suspicious_cluster;

        let node_eval = NodeEvaluation {
            node_id: node_id.to_string(),
            composite_score: composite,
            node_threshold,
            is_node_anomalous,
            is_in_suspicious_cluster,
        };

        let net_eval = NetworkEvaluation {
            estimated_byzantine_fraction: estimated_frac,
            adaptive_byzantine_threshold: recommendation.current_threshold,
            status: recommendation.status,
        };

        (node_eval, net_eval)
    }

    /// Access the underlying hierarchical detector for advanced use-cases.
    pub fn hierarchical_detector(&self) -> &HierarchicalDetector {
        &self.hierarchical
    }

    /// Access the underlying adaptive Byzantine threshold manager.
    pub fn adaptive_byzantine(&self) -> &AdaptiveByzantineThreshold {
        &self.byzantine_threshold
    }

    /// Mutable access to the cartel detector so callers can feed in
    /// similarity signals between nodes.
    pub fn cartel_detector_mut(&mut self) -> &mut CartelDetector {
        &mut self.cartel_detector
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matl_engine_basic_flow_marks_honest_and_byzantine_nodes() {
        let mut engine = MatlEngine::new(20, 3, 2);

        // Three honest nodes with high composite scores.
        let pogq_honest = ProofOfGradientQuality::new(0.9, 0.9, 0.1);
        for id in &["h1", "h2", "h3"] {
            let (node_eval, _net_eval) = engine.evaluate_node(id, &pogq_honest, 0.8);
            assert!(
                !node_eval.is_node_anomalous,
                "Honest node {} should not be anomalous",
                id
            );
        }

        // Two obviously bad nodes with very low PoGQ.
        let pogq_bad = ProofOfGradientQuality::new(0.1, 0.2, 0.9);
        for id in &["b1", "b2"] {
            let (node_eval, _net_eval) = engine.evaluate_node(id, &pogq_bad, 0.2);
            assert!(
                node_eval.is_node_anomalous,
                "Byzantine-like node {} should be anomalous",
                id
            );
        }

        // At least some fraction should be considered Byzantine at the
        // hierarchical level once bad nodes are present.
        let net_summary = engine.adaptive_byzantine().recommendation();
        assert!(
            net_summary.current_threshold >= super::super::MIN_BYZANTINE_TOLERANCE
                && net_summary.current_threshold <= super::super::MAX_BYZANTINE_TOLERANCE,
            "Adaptive threshold should remain within allowed bounds",
        );
    }
}
