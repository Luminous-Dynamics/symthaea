// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Attractor classification.
//!
//! The [`AttractorClassifier`] inspects a time window of [`GraphMetrics`]
//! and determines which attractor regime the graph is in.
//!
//! ## Visual Color Mapping (per design doc §7.4)
//!
//! | Class | Color |
//! |-------|-------|
//! | [`AttractorClass::StableManifold`] | calm blue/amber |
//! | [`AttractorClass::OscillatoryAttractor`] | rhythmic violet rings |
//! | [`AttractorClass::StrangeAttractorRisk`] | jagged red/violet drift |
//! | [`AttractorClass::HairballExplosion`] | expanding noisy mesh |
//! | [`AttractorClass::StringCollapse`] | narrowing line |
//! | [`AttractorClass::Fragmentation`] | disconnected islands |
//! | [`AttractorClass::UsefulEmergentManifold`] | 🎯 target |

use serde::{Deserialize, Serialize};

use crate::metrics::GraphMetrics;

/// The classified attractor regime of a synthetic graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AttractorClass {
    #[default]
    /// Not yet classified (insufficient history).
    Unknown,
    /// Graph stabilized into a coherent, low-dimensional manifold.
    /// Estimated dimension ≈ 2.0 ± 0.5. Low entropy. Low churn.
    /// Visual: calm blue/amber.
    StableManifold,
    /// Graph oscillates between two or more states periodically.
    /// Edge churn is rhythmic. Entropy oscillates.
    /// Visual: rhythmic violet rings.
    OscillatoryAttractor,
    /// Graph exhibits drift with sensitive dependence on initial conditions.
    /// Holonomy drift increasing. Entropy growing irregularly.
    /// Visual: jagged red/violet drift.
    StrangeAttractorRisk,
    /// Degree distribution is exploding. Hubs forming unboundedly.
    /// max_degree growing. edge_count / node_count > threshold.
    /// Visual: expanding noisy mesh.
    HairballExplosion,
    /// Graph is collapsing to a 1D chain or near-linear structure.
    /// estimated_dimension < 1.2. diameter / sqrt(N) > threshold.
    /// Visual: narrowing line.
    StringCollapse,
    /// Graph has fragmented into many disconnected components.
    /// betti_0 >> 1. Many isolated nodes.
    /// Visual: disconnected islands.
    Fragmentation,
    /// 🎯 Target. Estimated dimension ≈ 2.0 ± 0.3. Stable.
    /// Low entropy. Low churn. Connected. Metric-like.
    /// Visual: gold — this is what we came for.
    UsefulEmergentManifold,
}

impl AttractorClass {
    /// Returns the recommended visualization color role (for downstream rendering).
    pub fn color_role(&self) -> &'static str {
        match self {
            AttractorClass::Unknown => "grey",
            AttractorClass::StableManifold => "blue_amber",
            AttractorClass::OscillatoryAttractor => "violet",
            AttractorClass::StrangeAttractorRisk => "red_violet",
            AttractorClass::HairballExplosion => "red_orange",
            AttractorClass::StringCollapse => "white_sterile",
            AttractorClass::Fragmentation => "grey_static",
            AttractorClass::UsefulEmergentManifold => "gold",
        }
    }

    /// True if this class should trigger quarantine (halt the run).
    pub fn should_quarantine(&self) -> bool {
        matches!(
            self,
            AttractorClass::HairballExplosion
                | AttractorClass::Fragmentation
                | AttractorClass::StrangeAttractorRisk
        )
    }
}

/// Classifies a time window of [`GraphMetrics`] into an [`AttractorClass`].
///
/// Classification is rule-based in v0.1. A learned classifier can replace
/// or augment these rules in Phase 4+.
#[derive(Debug)]
pub struct AttractorClassifier {
    /// Target estimated dimension for [`AttractorClass::UsefulEmergentManifold`].
    pub target_dimension: f64,
    /// Tolerance around target dimension.
    pub dimension_tolerance: f64,
    /// Minimum history length required for classification.
    pub min_history: usize,
}

impl AttractorClassifier {
    pub fn new(target_dimension: f64, dimension_tolerance: f64, min_history: usize) -> Self {
        Self {
            target_dimension,
            dimension_tolerance,
            min_history,
        }
    }

    /// Classify a slice of metrics (chronological, oldest first).
    ///
    /// Returns `(class, confidence)` where confidence ∈ [0, 1].
    pub fn classify(&self, history: &[&GraphMetrics]) -> AttractorClass {
        if history.len() < self.min_history.max(4) {
            return AttractorClass::Unknown;
        }

        let recent = &history[history.len().saturating_sub(16)..];

        // --- Fragmentation ---
        let avg_betti0 = mean(recent.iter().map(|m| m.betti_0 as f64));
        if avg_betti0 > 5.0 {
            return AttractorClass::Fragmentation;
        }

        // --- HairballExplosion ---
        let avg_dim = mean(recent.iter().map(|m| m.estimated_dimension));
        let avg_max_deg: f64 = mean(recent.iter().map(|m| m.max_degree as f64));
        if avg_dim > 6.0 || avg_max_deg > 20.0 {
            return AttractorClass::HairballExplosion;
        }

        // --- StringCollapse ---
        if avg_dim < 1.2 {
            return AttractorClass::StringCollapse;
        }

        // --- StrangeAttractorRisk: check for irregular entropy growth ---
        let entropies: Vec<f64> = recent.iter().map(|m| m.degree_entropy).collect();
        let entropy_variance = variance(&entropies);
        let entropy_trend = linear_trend(&entropies);
        if entropy_variance > 0.5 && entropy_trend > 0.02 {
            return AttractorClass::StrangeAttractorRisk;
        }

        // --- OscillatoryAttractor: check for periodic dimension oscillation ---
        let dims: Vec<f64> = recent.iter().map(|m| m.estimated_dimension).collect();
        let dim_variance = variance(&dims);
        let churn_values: Vec<f64> = recent.iter().map(|m| m.edge_churn as f64).collect();
        let churn_variance = variance(&churn_values);
        if dim_variance > 0.1 && churn_variance > 2.0 {
            return AttractorClass::OscillatoryAttractor;
        }

        // --- UsefulEmergentManifold ---
        let dim_near_target = (avg_dim - self.target_dimension).abs() <= self.dimension_tolerance;
        let low_entropy_variance = entropy_variance < 0.05;
        let low_churn = mean(churn_values.iter().copied()) < 5.0;
        let connected = avg_betti0 < 2.0;
        if dim_near_target && low_entropy_variance && low_churn && connected {
            return AttractorClass::UsefulEmergentManifold;
        }

        // --- StableManifold (stable but not at target dimension) ---
        if dim_variance < 0.1 && low_entropy_variance && connected {
            return AttractorClass::StableManifold;
        }

        AttractorClass::Unknown
    }
}

impl Default for AttractorClassifier {
    fn default() -> Self {
        Self {
            target_dimension: 2.0,
            dimension_tolerance: 0.3,
            min_history: 8,
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn mean(it: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<f64> = it.collect();
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / values.len() as f64
}

/// Simple linear trend (slope) via least-squares.
fn linear_trend(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;
    let num: f64 = values
        .iter()
        .enumerate()
        .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
        .sum();
    let den: f64 = values
        .iter()
        .enumerate()
        .map(|(i, _)| (i as f64 - x_mean).powi(2))
        .sum();
    if den == 0.0 { 0.0 } else { num / den }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(dim: f64, betti0: usize, entropy: f64, churn: usize) -> GraphMetrics {
        GraphMetrics {
            tick: 0,
            node_count: 64,
            edge_count: 100,
            max_degree: 8,
            avg_degree: 3.1,
            estimated_dimension: dim,
            graph_diameter: 8,
            clustering_coefficient: 0.3,
            spectral_gap_proxy: 5.0,
            edge_churn: churn,
            degree_entropy: entropy,
            structural_free_energy: 1.0,
            holonomy_drift: 0.0,
            betti_0: betti0,
            betti_1_proxy: 36,
            attractor_class: AttractorClass::Unknown,
            classifier_confidence: 0.0,
        }
    }

    #[test]
    fn classifies_useful_manifold() {
        let classifier = AttractorClassifier::default();
        let metrics: Vec<GraphMetrics> = (0..20).map(|_| make_metrics(2.05, 1, 1.2, 2)).collect();
        let refs: Vec<&GraphMetrics> = metrics.iter().collect();
        assert_eq!(
            classifier.classify(&refs),
            AttractorClass::UsefulEmergentManifold
        );
    }

    #[test]
    fn classifies_hairball() {
        let classifier = AttractorClassifier::default();
        let metrics: Vec<GraphMetrics> = (0..20).map(|_| make_metrics(9.0, 1, 2.5, 100)).collect();
        let refs: Vec<&GraphMetrics> = metrics.iter().collect();
        assert_eq!(
            classifier.classify(&refs),
            AttractorClass::HairballExplosion
        );
    }

    #[test]
    fn classifies_string_collapse() {
        let classifier = AttractorClassifier::default();
        let metrics: Vec<GraphMetrics> = (0..20).map(|_| make_metrics(0.8, 1, 0.5, 1)).collect();
        let refs: Vec<&GraphMetrics> = metrics.iter().collect();
        assert_eq!(classifier.classify(&refs), AttractorClass::StringCollapse);
    }

    #[test]
    fn classifies_fragmentation() {
        let classifier = AttractorClassifier::default();
        let metrics: Vec<GraphMetrics> = (0..20).map(|_| make_metrics(2.0, 12, 1.0, 3)).collect();
        let refs: Vec<&GraphMetrics> = metrics.iter().collect();
        assert_eq!(classifier.classify(&refs), AttractorClass::Fragmentation);
    }

    #[test]
    fn color_role_gold_for_target() {
        assert_eq!(AttractorClass::UsefulEmergentManifold.color_role(), "gold");
    }

    #[test]
    fn quarantine_classes_flagged() {
        assert!(AttractorClass::HairballExplosion.should_quarantine());
        assert!(AttractorClass::Fragmentation.should_quarantine());
        assert!(AttractorClass::StrangeAttractorRisk.should_quarantine());
        assert!(!AttractorClass::StableManifold.should_quarantine());
        assert!(!AttractorClass::UsefulEmergentManifold.should_quarantine());
    }
}
