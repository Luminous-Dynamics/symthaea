// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! [`FrameBuilder`] — converts a [`MetricSnapshot`] into a [`ProjectionFrame`].
//!
//! This is the translation layer between raw telemetry and the projection data model.
//! It enforces the epistemic discipline rules:
//! - Frames with anomaly tags must have reduced confidence
//! - False-green suspects get the `FalseGreenSuspect` anomaly tag
//! - Perturbation events get amber Chronicle event refs

use symthaea_projection::{
    frame::{EventRef, ProjectionFrame, ProjectionMode, SourceSystem},
    layer::LayerId,
};

use crate::metric_snapshot::MetricSnapshot;

/// Converts [`MetricSnapshot`]s into [`ProjectionFrame`]s.
pub struct FrameBuilder {
    /// Which projection mode to use for emitted frames.
    pub mode: ProjectionMode,
    /// Chronicle durability threshold: anomaly_score above this → durable event candidate.
    pub chronicle_threshold: f64,
    /// Running frame counter.
    frame_counter: u64,
}

impl FrameBuilder {
    pub fn new(mode: ProjectionMode) -> Self {
        Self {
            mode,
            chronicle_threshold: 0.6,
            frame_counter: 0,
        }
    }

    /// Build a [`ProjectionFrame`] from a [`MetricSnapshot`].
    ///
    /// Enforces epistemic consistency:
    /// - If anomaly tags exist, confidence is reduced
    /// - False-green suspects are flagged
    /// - High-anomaly events become Chronicle references
    pub fn build(&mut self, snapshot: &MetricSnapshot) -> ProjectionFrame {
        self.frame_counter += 1;

        let anomaly_tags = snapshot.anomaly_tags();
        let mut confidence = snapshot.frame_confidence();

        // Epistemic rule: if anomaly tags exist, confidence cannot exceed 0.85
        if !anomaly_tags.is_empty() {
            confidence = confidence.min(0.85);
        }

        // Epistemic rule: false-green suspects get capped at 0.7
        if snapshot.is_false_green_suspect() {
            confidence = confidence.min(0.7);
        }

        let mut frame = ProjectionFrame::new(
            snapshot.frame_id,
            snapshot.timestamp,
            SourceSystem::Fep,
            self.mode,
            LayerId::Fep,
        );

        frame.confidence = confidence;
        frame.scalar_metrics = snapshot.to_scalar_map();
        frame.evidence_tags = snapshot.source_tags.clone();
        frame.is_anomalous = !anomaly_tags.is_empty();

        for tag in &anomaly_tags {
            frame.anomaly_tags.push(tag.clone());
        }

        // Chronicle durable event: high-anomaly events are recorded
        if snapshot.anomaly_score > self.chronicle_threshold || snapshot.is_perturbation_event {
            frame.durable_event_refs.push(EventRef {
                event_id: format!("evt_{}", snapshot.frame_id),
                timestamp: snapshot.timestamp,
                is_durable: snapshot.anomaly_score > self.chronicle_threshold,
                civic_authority: None,
            });
        }

        // Vector metrics: store the full metric vector for HDC/PCA projections
        frame.vector_metrics.insert(
            "canonical_7".to_string(),
            vec![
                snapshot.phi,
                snapshot.fep_prediction_error,
                snapshot.workspace_activation,
                snapshot.hot_confidence,
                snapshot.anomaly_score,
                snapshot.memory_pressure,
                snapshot.mip_instability,
            ],
        );

        // Topological metrics
        frame
            .topology_metrics
            .insert("anomaly_score".to_string(), snapshot.anomaly_score);
        frame
            .topology_metrics
            .insert("frame_confidence".to_string(), confidence);

        // Final epistemic check (logs warning if violated)
        if let Some(warning) = frame.epistemic_consistency_check() {
            tracing::warn!("{}", warning);
        }

        frame
    }

    /// Build a frame for each subsystem layer (for Stratified Stack mode).
    pub fn build_stack(&mut self, snapshot: &MetricSnapshot) -> Vec<ProjectionFrame> {
        vec![
            self.build_layer(snapshot, LayerId::Fep, SourceSystem::Fep),
            self.build_layer(snapshot, LayerId::Hdc, SourceSystem::Fep),
            self.build_layer(snapshot, LayerId::Workspace, SourceSystem::Workspace),
            self.build_layer(snapshot, LayerId::Hot, SourceSystem::ConsciousnessTopology),
            self.build_layer(snapshot, LayerId::IitPhi, SourceSystem::PhiOracle),
        ]
    }

    fn build_layer(
        &mut self,
        snapshot: &MetricSnapshot,
        layer: LayerId,
        source: SourceSystem,
    ) -> ProjectionFrame {
        self.frame_counter += 1;
        let mut frame = ProjectionFrame::new(
            self.frame_counter,
            snapshot.timestamp,
            source,
            ProjectionMode::StratifiedStack,
            layer,
        );
        frame.confidence = snapshot.frame_confidence();
        frame.scalar_metrics = snapshot.to_scalar_map();
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric_snapshot::MetricSnapshot;

    #[test]
    fn frame_built_from_baseline() {
        let mut builder = FrameBuilder::new(ProjectionMode::TimeWaterfall);
        let snap = MetricSnapshot::baseline(1);
        let frame = builder.build(&snap);
        // Baseline has the 7 canonical metrics
        assert!(frame.scalar_metrics.contains_key("phi"));
        assert!(frame.scalar_metrics.len() == 7);
        // Cold-start baseline has low (but non-zero) confidence — that's correct
        assert!(
            frame.confidence > 0.05,
            "confidence must be above the minimum floor"
        );
        assert!(
            frame.confidence < 0.8,
            "cold-start should not be highly confident"
        );
    }

    #[test]
    fn high_anomaly_reduces_confidence() {
        let mut builder = FrameBuilder::new(ProjectionMode::TimeWaterfall);
        let snap = MetricSnapshot::new(2, 0.0, 5.0, 0.0, 0.3, 3.0, 0.9);
        let frame = builder.build(&snap);
        assert!(
            frame.confidence < 0.85,
            "anomalous frame should have reduced confidence"
        );
        assert!(!frame.anomaly_tags.is_empty());
    }

    #[test]
    fn false_green_gets_tag() {
        let mut builder = FrameBuilder::new(ProjectionMode::TimeWaterfall);
        let snap = MetricSnapshot::new(3, 1.0, 0.001, 0.5, 0.999, 0.001, 0.001);
        let frame = builder.build(&snap);
        assert!(
            frame
                .anomaly_tags
                .iter()
                .any(|t| t == "false_green_suspect"),
            "false green should be tagged"
        );
    }

    #[test]
    fn perturbation_event_becomes_chronicle_ref() {
        let mut builder = FrameBuilder::new(ProjectionMode::TimeWaterfall);
        let snap = MetricSnapshot::new(4, 2.0, 1.0, 0.8, 0.7, 1.0, 0.3)
            .with_event("pump_contradiction_detected");
        let frame = builder.build(&snap);
        assert!(frame.has_durable_events());
    }

    #[test]
    fn stratified_stack_produces_five_frames() {
        let mut builder = FrameBuilder::new(ProjectionMode::StratifiedStack);
        let snap = MetricSnapshot::baseline(5);
        let frames = builder.build_stack(&snap);
        assert_eq!(frames.len(), 5);
    }
}
