// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! [`ProjectionEdge`] — a relationship or flow between projection nodes.

use serde::{Deserialize, Serialize};

use crate::grammar::LineStyle;

/// Type of relationship represented by this edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Causal relationship (A caused B).
    Causal,
    /// Data flow (A feeds B).
    DataFlow,
    /// Integration bridge (successful vertical integration in stack).
    IntegrationBridge,
    /// Failed broadcast (broken vertical bridge).
    FailedBroadcast,
    /// Trust relationship (Mycelix trust edge). → braided amber.
    Trust,
    /// Contested claim (two sources disagree). → split red/amber.
    ContestedClaim,
    /// Temporal succession (B followed A in time).
    Temporal,
    /// Evidence reference (this state was justified by that evidence).
    Evidence,
}

impl EdgeType {
    pub fn default_line_style(&self) -> LineStyle {
        match self {
            EdgeType::Causal => LineStyle::Crisp,
            EdgeType::DataFlow => LineStyle::Crisp,
            EdgeType::IntegrationBridge => LineStyle::Braided,
            EdgeType::FailedBroadcast => LineStyle::Broken,
            EdgeType::Trust => LineStyle::Braided,
            EdgeType::ContestedClaim => LineStyle::Diverging,
            EdgeType::Temporal => LineStyle::Dashed,
            EdgeType::Evidence => LineStyle::Dashed,
        }
    }
}

/// A relationship or flow between two projection nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionEdge {
    pub edge_id: String,
    pub source_node: String,
    pub target_node: String,
    pub edge_type: EdgeType,
    /// Weight [0.0, 1.0] — strength of the relationship.
    pub weight: f64,
    /// Confidence in this edge [0.0, 1.0].
    pub confidence: f64,
    pub line_style: LineStyle,
    /// Whether the edge is directed (arrow shown).
    pub directed: bool,
    /// Temporal phase [0.0, 1.0] for animated edges (e.g., ripple).
    pub temporal_phase: f64,
    /// Evidence references.
    pub evidence_refs: Vec<String>,
}

impl ProjectionEdge {
    pub fn new(
        id: impl Into<String>,
        source: impl Into<String>,
        target: impl Into<String>,
        edge_type: EdgeType,
    ) -> Self {
        let style = edge_type.default_line_style();
        Self {
            edge_id: id.into(),
            source_node: source.into(),
            target_node: target.into(),
            edge_type,
            weight: 1.0,
            confidence: 1.0,
            line_style: style,
            directed: true,
            temporal_phase: 0.0,
            evidence_refs: vec![],
        }
    }

    /// True if this edge represents a broken or uncertain relationship.
    pub fn is_anomalous(&self) -> bool {
        self.line_style.is_anomaly() || self.confidence < 0.4
    }
}
