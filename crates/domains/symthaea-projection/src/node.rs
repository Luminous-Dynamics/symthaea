// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! [`ProjectionNode`] — a visualized state element in the projection system.

use serde::{Deserialize, Serialize};

use crate::grammar::{ColorRole, LineStyle};

/// Semantic type of a projection node.
///
/// Used to drive visual grammar (color, size, line style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeSemanticType {
    /// Generic state node.
    Generic,
    /// High-integration node (IIT hub). → amber, larger.
    IntegrationHub,
    /// Memory retrieval node. → violet.
    MemoryRetrieval,
    /// Anomaly or contradiction node. → red.
    Anomaly,
    /// Chronicle/durable event node. → amber with ring.
    DurableEvent,
    /// Ecological signal node. → green/organic.
    EcologicalSignal,
    /// Machine diagnostic node. → white/clean.
    MachineDiagnostic,
    /// Suspicious false-green node (Null masking). → sterile white.
    FalseGreenSuspect,
}

impl NodeSemanticType {
    pub fn color_role(&self) -> ColorRole {
        match self {
            NodeSemanticType::Generic => ColorRole::Unknown,
            NodeSemanticType::IntegrationHub => ColorRole::Chronicle,
            NodeSemanticType::MemoryRetrieval => ColorRole::Memory,
            NodeSemanticType::Anomaly => ColorRole::Danger,
            NodeSemanticType::DurableEvent => ColorRole::Chronicle,
            NodeSemanticType::EcologicalSignal => ColorRole::Ecology,
            NodeSemanticType::MachineDiagnostic => ColorRole::MachineTruth,
            NodeSemanticType::FalseGreenSuspect => ColorRole::FalseGreen,
        }
    }

    pub fn border_style(&self) -> LineStyle {
        match self {
            NodeSemanticType::FalseGreenSuspect => LineStyle::TooSmooth,
            NodeSemanticType::Anomaly => LineStyle::Trembling,
            NodeSemanticType::DurableEvent => LineStyle::Braided,
            _ => LineStyle::Crisp,
        }
    }
}

/// A visualized state element in the projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionNode {
    pub node_id: String,
    pub semantic_type: NodeSemanticType,
    /// 2D position in the visualization plane [−1.0, 1.0] normalized.
    pub position_x: f32,
    pub position_y: f32,
    /// Depth in the 2.5D projection [0.0 = front, 1.0 = rear].
    pub depth: f32,
    /// Visual size multiplier [0.1, 3.0].
    pub size: f32,
    /// Confidence in this node [0.0, 1.0]. Drives opacity.
    pub confidence: f32,
    /// Display label (shown on hover or when pinned).
    pub label: String,
    /// Evidence references (what justifies this node's state).
    pub evidence_refs: Vec<String>,
    /// State tags (for filtering and search).
    pub state_tags: Vec<String>,
}
