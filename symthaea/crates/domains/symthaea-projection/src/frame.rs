// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! [`ProjectionFrame`] — one time step of projected state.
//!
//! This is the canonical data unit flowing through the projection system.
//! Every visualization reads from a sequence of `ProjectionFrame`s.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{grammar::DepthMeaning, layer::LayerId};

/// Which visualization mode this frame belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionMode {
    /// Depth = time. Present at front, past extruded backward.
    TimeWaterfall,
    /// Depth = abstraction layer (sensorimotor → civic).
    StratifiedStack,
    /// Multiple lenses through the same underlying state.
    HolographicCrossSection,
}

impl ProjectionMode {
    pub fn depth_meaning(&self) -> DepthMeaning {
        match self {
            ProjectionMode::TimeWaterfall => DepthMeaning::Time,
            ProjectionMode::StratifiedStack => DepthMeaning::AbstractionLayer,
            ProjectionMode::HolographicCrossSection => DepthMeaning::EvidenceChain,
        }
    }
}

/// Which Symthaea subsystem emitted this frame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceSystem {
    /// Free Energy Principle active inference.
    Fep,
    /// IIT/Phi integration measurement.
    PhiOracle,
    /// Global Workspace Theory broadcast state.
    Workspace,
    /// Probe stream telemetry.
    ProbeStream,
    /// Synthetic physics lab graph metrics.
    SyntheticPhysicsLab,
    /// Memory pressure / recall.
    Memory,
    /// Consciousness topology / HOT state.
    ConsciousnessTopology,
    /// Custom source (for extension).
    Custom(String),
}

/// A reference to a durable Chronicle event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRef {
    pub event_id: String,
    pub timestamp: f64,
    pub is_durable: bool,
    pub civic_authority: Option<String>,
}

/// A single frame in the projection system.
///
/// Represents the state of one source system at one moment in time,
/// projected through one visualization mode and one layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionFrame {
    /// Unique monotonic frame ID.
    pub frame_id: u64,
    /// Unix timestamp (seconds).
    pub timestamp: f64,
    /// Which system emitted this frame.
    pub source_system: SourceSystem,
    /// Which visualization mode applies.
    pub projection_mode: ProjectionMode,
    /// Which semantic layer this frame belongs to.
    pub layer_id: LayerId,
    /// Flat scalar metrics (name → value). Used by all visualization modes.
    ///
    /// Standard keys for Time-Waterfall:
    /// - `"phi"` — IIT integration score
    /// - `"fep_prediction_error"` — FEP surprise
    /// - `"workspace_activation"` — GWT broadcast strength
    /// - `"hot_confidence"` — HOT self-model confidence
    /// - `"anomaly_score"` — probe stream anomaly
    /// - `"memory_pressure"` — memory load
    /// - `"mip_instability"` — MIP boundary instability
    pub scalar_metrics: HashMap<String, f64>,
    /// High-dimensional vector metrics (name → values). For HDC/PCA projections.
    pub vector_metrics: HashMap<String, Vec<f64>>,
    /// Topological metrics (name → value).
    pub topology_metrics: HashMap<String, f64>,
    /// Overall confidence in this frame [0.0, 1.0].
    pub confidence: f64,
    /// Evidence tags (what justified this state).
    pub evidence_tags: Vec<String>,
    /// Anomaly tags (what looks wrong).
    pub anomaly_tags: Vec<String>,
    /// References to Chronicle durable events, if any.
    pub durable_event_refs: Vec<EventRef>,
    /// Whether this frame was produced during an anomalous halt.
    pub is_anomalous: bool,
}

impl ProjectionFrame {
    /// Create a minimal frame with just the required fields.
    pub fn new(
        frame_id: u64,
        timestamp: f64,
        source: SourceSystem,
        mode: ProjectionMode,
        layer: LayerId,
    ) -> Self {
        Self {
            frame_id,
            timestamp,
            source_system: source,
            projection_mode: mode,
            layer_id: layer,
            scalar_metrics: HashMap::new(),
            vector_metrics: HashMap::new(),
            topology_metrics: HashMap::new(),
            confidence: 1.0,
            evidence_tags: vec![],
            anomaly_tags: vec![],
            durable_event_refs: vec![],
            is_anomalous: false,
        }
    }

    /// Insert a scalar metric.
    pub fn with_scalar(mut self, key: impl Into<String>, value: f64) -> Self {
        self.scalar_metrics.insert(key.into(), value);
        self
    }

    /// Mark this frame as containing an anomaly.
    pub fn with_anomaly(mut self, tag: impl Into<String>) -> Self {
        self.anomaly_tags.push(tag.into());
        self.is_anomalous = true;
        self
    }

    /// True if confidence is below a threshold indicating uncertain data.
    pub fn is_uncertain(&self, threshold: f64) -> bool {
        self.confidence < threshold
    }

    /// Get a scalar metric by name.
    pub fn scalar(&self, key: &str) -> Option<f64> {
        self.scalar_metrics.get(key).copied()
    }

    /// True if this frame has any Chronicle durable event references.
    pub fn has_durable_events(&self) -> bool {
        !self.durable_event_refs.is_empty()
    }

    /// Visual doctrine check: warn if confidence is high but anomaly tags exist.
    ///
    /// The dashboard must NOT look more certain than the data.
    pub fn epistemic_consistency_check(&self) -> Option<String> {
        if self.confidence > 0.9 && !self.anomaly_tags.is_empty() {
            return Some(format!(
                "Frame {}: confidence={:.2} but anomaly_tags={:?} — reduce displayed confidence",
                self.frame_id, self.confidence, self.anomaly_tags
            ));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame() -> ProjectionFrame {
        ProjectionFrame::new(
            1,
            100.0,
            SourceSystem::Fep,
            ProjectionMode::TimeWaterfall,
            LayerId::Fep,
        )
        .with_scalar("phi", 3.5)
        .with_scalar("fep_prediction_error", 0.2)
    }

    #[test]
    fn scalar_retrieval() {
        let f = make_frame();
        assert!((f.scalar("phi").unwrap() - 3.5).abs() < 1e-9);
        assert!(f.scalar("nonexistent").is_none());
    }

    #[test]
    fn epistemic_consistency_high_confidence_no_anomaly() {
        let f = make_frame();
        assert!(f.epistemic_consistency_check().is_none());
    }

    #[test]
    fn epistemic_consistency_flags_inconsistency() {
        let f = make_frame().with_anomaly("false_green_detected");
        // confidence defaults to 1.0, anomaly present — should warn
        assert!(f.epistemic_consistency_check().is_some());
    }

    #[test]
    fn depth_meaning_correct() {
        assert_eq!(
            ProjectionMode::TimeWaterfall.depth_meaning(),
            DepthMeaning::Time
        );
        assert_eq!(
            ProjectionMode::StratifiedStack.depth_meaning(),
            DepthMeaning::AbstractionLayer
        );
    }
}
