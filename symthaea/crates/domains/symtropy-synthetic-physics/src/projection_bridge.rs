// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bridge from `symtropy-synthetic-physics` to `symthaea-projection`.
//!
//! Converts [`GraphMetrics`] into [`ProjectionFrameData`] — the wire format
//! that the Time-Waterfall renderer reads.
//!
//! This is a one-way bridge. The projection system reads lab output;
//! the lab does not depend on the projection system.

use std::collections::HashMap;

use crate::metrics::GraphMetrics;

/// A lightweight projection frame emitted by the synthetic physics lab.
///
/// This is intentionally minimal — it matches the `ProjectionFrame` schema
/// in `symthaea-projection` but does not import that crate (sealed lane rule).
#[derive(Debug, Clone)]
pub struct ProjectionFrameData {
    pub frame_id: u64,
    pub timestamp: f64,
    pub source: &'static str,
    pub projection_mode: &'static str,
    pub layer_id: &'static str,
    pub scalar_metrics: HashMap<String, f64>,
    pub confidence: f64,
    pub anomaly_tags: Vec<String>,
    pub attractor_class: String,
    pub attractor_color_role: String,
    pub halted_early: bool,
}

impl ProjectionFrameData {
    /// Build a projection frame from graph metrics.
    ///
    /// `age` is the index of this frame in the waterfall (0 = present, N = past).
    pub fn from_metrics(metrics: &GraphMetrics, age: usize, halted_early: bool) -> Self {
        let mut anomaly_tags = vec![];

        // Classify anomalies from metrics
        if metrics.betti_0 > 3 {
            anomaly_tags.push("fragmentation_risk".to_string());
        }
        if metrics.estimated_dimension > 5.0 {
            anomaly_tags.push("hairball_risk".to_string());
        }
        if metrics.estimated_dimension < 1.2 {
            anomaly_tags.push("string_collapse_risk".to_string());
        }
        if metrics.holonomy_drift > 0.3 {
            anomaly_tags.push("curvature_anomaly".to_string());
        }
        if halted_early {
            anomaly_tags.push("run_halted_early".to_string());
        }

        // Confidence: high when attractor is classified and stable
        let confidence = match metrics.attractor_class {
            crate::attractor::AttractorClass::UsefulEmergentManifold => 0.95,
            crate::attractor::AttractorClass::StableManifold => 0.80,
            crate::attractor::AttractorClass::OscillatoryAttractor => 0.65,
            crate::attractor::AttractorClass::Unknown => 0.3,
            _ => 0.4,
        };

        // Waterfall depth = age (0 = front/present plane)
        let depth_tag = format!("waterfall_age={age}");

        Self {
            frame_id: metrics.tick,
            timestamp: metrics.tick as f64,
            source: "symtropy-synthetic-physics",
            projection_mode: "time_waterfall",
            layer_id: "synthetic_physics_lab",
            scalar_metrics: metrics.to_scalar_map(),
            confidence,
            anomaly_tags: {
                let mut tags = anomaly_tags;
                tags.push(depth_tag);
                tags
            },
            attractor_class: format!("{:?}", metrics.attractor_class),
            attractor_color_role: metrics.attractor_class.color_role().to_string(),
            halted_early,
        }
    }

    /// Opacity for this frame based on its age (0 = opaque/present, N = faded/past).
    ///
    /// Uses exponential decay: `opacity = decay_rate ^ age`
    pub fn opacity(&self, age: usize, decay_rate: f64) -> f64 {
        decay_rate.powi(age as i32).max(0.05)
    }

    /// Depth position for 2.5D rendering (front = 0.0, rear = 1.0).
    pub fn depth_position(&self, age: usize, max_age: usize) -> f64 {
        if max_age == 0 {
            return 0.0;
        }
        age as f64 / max_age as f64
    }
}

/// Convert a full ring buffer history into a sequence of waterfall frames.
///
/// Returns frames ordered from present (age=0) to past (age=N-1).
pub fn metrics_to_waterfall(
    history: &[&GraphMetrics],
    halted_early: bool,
) -> Vec<ProjectionFrameData> {
    history
        .iter()
        .rev() // most recent first (age=0)
        .enumerate()
        .map(|(age, &m)| ProjectionFrameData::from_metrics(m, age, halted_early))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{attractor::AttractorClass, metrics::GraphMetrics};

    fn dummy_metrics(tick: u64, dim: f64, class: AttractorClass) -> GraphMetrics {
        GraphMetrics {
            tick,
            node_count: 32,
            edge_count: 80,
            max_degree: 6,
            avg_degree: 5.0,
            estimated_dimension: dim,
            graph_diameter: 6,
            clustering_coefficient: 0.4,
            spectral_gap_proxy: 5.5,
            edge_churn: 3,
            degree_entropy: 1.2,
            structural_free_energy: 0.8,
            holonomy_drift: 0.0,
            betti_0: 1,
            betti_1_proxy: 49,
            attractor_class: class,
            classifier_confidence: 0.9,
        }
    }

    #[test]
    fn frame_from_manifold_metrics() {
        let m = dummy_metrics(10, 2.1, AttractorClass::UsefulEmergentManifold);
        let frame = ProjectionFrameData::from_metrics(&m, 0, false);
        assert_eq!(frame.attractor_color_role, "gold");
        assert!((frame.confidence - 0.95).abs() < 1e-6);
        assert!(
            frame
                .anomaly_tags
                .iter()
                .any(|t| t.starts_with("waterfall_age="))
        );
    }

    #[test]
    fn opacity_decays_with_age() {
        let m = dummy_metrics(5, 2.0, AttractorClass::StableManifold);
        let frame = ProjectionFrameData::from_metrics(&m, 0, false);
        let op0 = frame.opacity(0, 0.95);
        let op10 = frame.opacity(10, 0.95);
        assert!(op0 > op10, "older frames must be more transparent");
    }

    #[test]
    fn waterfall_series_age_order() {
        let metrics: Vec<GraphMetrics> = (0..8)
            .map(|i| dummy_metrics(i, 2.0, AttractorClass::StableManifold))
            .collect();
        let refs: Vec<&GraphMetrics> = metrics.iter().collect();
        let frames = metrics_to_waterfall(&refs, false);
        assert_eq!(frames.len(), 8);
        // frame[0] is most recent (highest tick)
        assert_eq!(frames[0].frame_id, 7);
        // frame[7] is oldest
        assert_eq!(frames[7].frame_id, 0);
    }
}
