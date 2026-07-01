// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Projection layers — semantic planes in the 2.5D visualization.

use serde::{Deserialize, Serialize};

use crate::grammar::{ColorRole, DepthMeaning, OpacityState};

/// Identifies a semantic layer in the projection system.
///
/// Layer ordering (low → high for Stratified Stack):
/// 1. Physical / sensorimotor (FEP)
/// 2. HDC representation
/// 3. GWT workspace
/// 4. HOT metacognition
/// 5. IIT / Φ integration
/// 6. Ethics / Governance
/// 7. Chronicle / durable memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerId {
    // Cognitive stack layers
    Fep,
    Hdc,
    Workspace,
    Hot,
    IitPhi,
    EthicsGovernance,
    Chronicle,
    // Synthetic Physics Lab
    SyntheticPhysicsLab,
    // Field Deck layers
    FieldDeckPhysical,
    FieldDeckDevice,
    FieldDeckEcological,
    FieldDeckCivic,
    FieldDeckForecast,
    // Mycelix / Xenia
    MycelixTrust,
    XeniaSession,
    // Custom
    Custom(u8),
}

impl LayerId {
    /// Canonical depth position [0.0, 1.0] in the Stratified Stack.
    /// 0.0 = bottom (physical), 1.0 = top (Chronicle/civic).
    pub fn stack_depth(&self) -> f64 {
        match self {
            LayerId::Fep => 0.0,
            LayerId::Hdc => 0.14,
            LayerId::Workspace => 0.28,
            LayerId::Hot => 0.42,
            LayerId::IitPhi => 0.57,
            LayerId::EthicsGovernance => 0.71,
            LayerId::Chronicle => 0.85,
            LayerId::SyntheticPhysicsLab => 0.5,
            LayerId::FieldDeckPhysical => 0.0,
            LayerId::FieldDeckDevice => 0.2,
            LayerId::FieldDeckEcological => 0.4,
            LayerId::FieldDeckCivic => 0.6,
            LayerId::FieldDeckForecast => 0.8,
            LayerId::MycelixTrust => 0.7,
            LayerId::XeniaSession => 0.6,
            LayerId::Custom(v) => (*v as f64) / 255.0,
        }
    }

    /// Default color role for this layer.
    pub fn default_color_role(&self) -> ColorRole {
        match self {
            LayerId::Fep => ColorRole::PhysicalSignal,
            LayerId::Hdc => ColorRole::Memory,
            LayerId::Workspace => ColorRole::PhysicalSignal,
            LayerId::Hot => ColorRole::Memory,
            LayerId::IitPhi => ColorRole::Chronicle,
            LayerId::EthicsGovernance => ColorRole::Chronicle,
            LayerId::Chronicle => ColorRole::Chronicle,
            LayerId::SyntheticPhysicsLab => ColorRole::PhysicalSignal,
            LayerId::FieldDeckPhysical => ColorRole::PhysicalSignal,
            LayerId::FieldDeckDevice => ColorRole::MachineTruth,
            LayerId::FieldDeckEcological => ColorRole::Ecology,
            LayerId::FieldDeckCivic => ColorRole::Chronicle,
            LayerId::FieldDeckForecast => ColorRole::Memory,
            LayerId::MycelixTrust => ColorRole::Ecology,
            LayerId::XeniaSession => ColorRole::Chronicle,
            LayerId::Custom(_) => ColorRole::Unknown,
        }
    }
}

/// The functional type of this layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    Cognitive,
    Physical,
    Ecological,
    Civic,
    Memory,
    Experimental,
}

/// Whether a layer is visible and interactive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisibilityState {
    Visible,
    Hidden,
    Pinned,    // always visible, cannot be hidden
    Collapsed, // visible but minimized
}

/// A semantic plane in the projection system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionLayer {
    pub layer_id: LayerId,
    pub layer_type: LayerType,
    /// Depth position [0.0, 1.0] — meaning depends on `ProjectionMode`.
    pub depth_position: f64,
    pub depth_meaning: DepthMeaning,
    pub color_role: ColorRole,
    pub visibility: VisibilityState,
    pub opacity: OpacityState,
    /// Ordered list of metric names active in this layer (for display priority).
    pub active_metrics: Vec<String>,
    /// Human-readable label shown in the UI.
    pub label: String,
    /// Brief description shown on hover.
    pub description: String,
}

impl ProjectionLayer {
    pub fn from_id(id: LayerId, depth_meaning: DepthMeaning) -> Self {
        Self {
            layer_id: id,
            layer_type: LayerType::Cognitive,
            depth_position: id.stack_depth(),
            depth_meaning,
            color_role: id.default_color_role(),
            visibility: VisibilityState::Visible,
            opacity: OpacityState::HighConfidence,
            active_metrics: vec![],
            label: format!("{id:?}"),
            description: String::new(),
        }
    }
}

/// Build the default 7-layer Symthaea cognitive stack.
pub fn default_cognitive_stack() -> Vec<ProjectionLayer> {
    use crate::grammar::DepthMeaning;
    let depth = DepthMeaning::AbstractionLayer;

    vec![
        ProjectionLayer {
            layer_id: LayerId::Fep,
            layer_type: LayerType::Physical,
            depth_position: 0.0,
            depth_meaning: depth,
            color_role: ColorRole::PhysicalSignal,
            visibility: VisibilityState::Visible,
            opacity: OpacityState::HighConfidence,
            active_metrics: vec![
                "fep_prediction_error".into(),
                "sensory_surprise".into(),
                "action_pressure".into(),
            ],
            label: "FEP / Sensorimotor".into(),
            description: "Prediction error, sensory surprise, action pressure, uncertainty".into(),
        },
        ProjectionLayer {
            layer_id: LayerId::Hdc,
            layer_type: LayerType::Cognitive,
            depth_position: 0.14,
            depth_meaning: depth,
            color_role: ColorRole::Memory,
            visibility: VisibilityState::Visible,
            opacity: OpacityState::HighConfidence,
            active_metrics: vec![
                "hdc_cluster_activation".into(),
                "similarity_resonance".into(),
                "memory_retrieval_strength".into(),
            ],
            label: "HDC Representation".into(),
            description: "Hypervector clusters, similarity resonance, memory binding".into(),
        },
        ProjectionLayer {
            layer_id: LayerId::Workspace,
            layer_type: LayerType::Cognitive,
            depth_position: 0.28,
            depth_meaning: depth,
            color_role: ColorRole::PhysicalSignal,
            visibility: VisibilityState::Visible,
            opacity: OpacityState::HighConfidence,
            active_metrics: vec![
                "workspace_activation".into(),
                "broadcast_strength".into(),
                "gwt_content_count".into(),
            ],
            label: "GWT Workspace".into(),
            description: "Global workspace access, broadcast strength, active content".into(),
        },
        ProjectionLayer {
            layer_id: LayerId::Hot,
            layer_type: LayerType::Cognitive,
            depth_position: 0.42,
            depth_meaning: depth,
            color_role: ColorRole::Memory,
            visibility: VisibilityState::Visible,
            opacity: OpacityState::HighConfidence,
            active_metrics: vec![
                "hot_confidence".into(),
                "self_model_uncertainty".into(),
                "reportability".into(),
            ],
            label: "HOT Metacognition".into(),
            description: "Self-model confidence, metacognitive assessment, reportability".into(),
        },
        ProjectionLayer {
            layer_id: LayerId::IitPhi,
            layer_type: LayerType::Cognitive,
            depth_position: 0.57,
            depth_meaning: depth,
            color_role: ColorRole::Chronicle,
            visibility: VisibilityState::Visible,
            opacity: OpacityState::HighConfidence,
            active_metrics: vec![
                "phi".into(),
                "mip_instability".into(),
                "integration_fragmentation".into(),
            ],
            label: "IIT / Φ Integration".into(),
            description: "Integration score, MIP boundary, fragmentation, re-integration".into(),
        },
        ProjectionLayer {
            layer_id: LayerId::EthicsGovernance,
            layer_type: LayerType::Civic,
            depth_position: 0.71,
            depth_meaning: depth,
            color_role: ColorRole::Chronicle,
            visibility: VisibilityState::Visible,
            opacity: OpacityState::HighConfidence,
            active_metrics: vec![
                "moral_gate_state".into(),
                "civic_authority_level".into(),
                "refusal_condition".into(),
            ],
            label: "Ethics / Governance".into(),
            description: "Moral gate state, civic authority, Mycelix trust, refusal conditions"
                .into(),
        },
        ProjectionLayer {
            layer_id: LayerId::Chronicle,
            layer_type: LayerType::Memory,
            depth_position: 0.85,
            depth_meaning: depth,
            color_role: ColorRole::Chronicle,
            visibility: VisibilityState::Pinned, // Chronicle is always visible
            opacity: OpacityState::HighConfidence,
            active_metrics: vec![
                "durable_event_count".into(),
                "chronicle_formation_rate".into(),
                "evidence_chain_depth".into(),
            ],
            label: "Chronicle / Memory".into(),
            description: "Durable event formation, evidence chains, civic witness records".into(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cognitive_stack_has_seven_layers() {
        let stack = default_cognitive_stack();
        assert_eq!(stack.len(), 7);
    }

    #[test]
    fn stack_depth_monotonic() {
        let stack = default_cognitive_stack();
        let depths: Vec<f64> = stack.iter().map(|l| l.depth_position).collect();
        for i in 1..depths.len() {
            assert!(
                depths[i] > depths[i - 1],
                "stack depths must be strictly monotonic"
            );
        }
    }

    #[test]
    fn chronicle_layer_pinned() {
        let stack = default_cognitive_stack();
        let chronicle = stack
            .iter()
            .find(|l| l.layer_id == LayerId::Chronicle)
            .unwrap();
        assert_eq!(chronicle.visibility, VisibilityState::Pinned);
    }

    #[test]
    fn layer_color_roles_match_doctrine() {
        assert_eq!(LayerId::Fep.default_color_role(), ColorRole::PhysicalSignal);
        assert_eq!(
            LayerId::Chronicle.default_color_role(),
            ColorRole::Chronicle
        );
        assert_eq!(
            LayerId::FieldDeckEcological.default_color_role(),
            ColorRole::Ecology
        );
    }
}
