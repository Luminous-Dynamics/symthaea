// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Butlin indicator report structures.

use serde::{Deserialize, Serialize};

/// Runtime consciousness data from the structural Phi engine.
///
/// When available, blends with static architectural scores to produce
/// theory-aligned indicator values.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeConsciousnessData {
    /// Micro-level Phi (within-cluster integration).
    pub micro_phi: f64,
    /// Meso-level Phi (inter-cluster integration).
    pub meso_phi: f64,
    /// Macro-level Phi (global integration).
    pub macro_phi: f64,
    /// Bottleneck score: gap between global and inter-cluster integration [0, 1].
    pub bottleneck_score: f64,
    /// Emergence ratio: macro / (micro + meso). > 1.0 means whole > sum of parts.
    pub emergence_ratio: f64,
    /// Number of detected clusters.
    pub num_clusters: usize,
}

impl RuntimeConsciousnessData {
    /// Construct from structural Phi fields (typically extracted from CycleMetadata).
    pub fn from_structural(
        micro_phi: f64,
        meso_phi: f64,
        macro_phi: f64,
        bottleneck_score: f64,
        emergence_ratio: f64,
        num_clusters: usize,
    ) -> Self {
        Self {
            micro_phi,
            meso_phi,
            macro_phi,
            bottleneck_score,
            emergence_ratio,
            num_clusters,
        }
    }
}

/// Status of a consciousness indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndicatorStatus {
    /// The architectural property is clearly present.
    Present,
    /// The property is partially implemented or ambiguous.
    Partial,
    /// The property is absent.
    Absent,
}

impl std::fmt::Display for IndicatorStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndicatorStatus::Present => write!(f, "PRESENT"),
            IndicatorStatus::Partial => write!(f, "PARTIAL"),
            IndicatorStatus::Absent => write!(f, "ABSENT"),
        }
    }
}

/// Evidence for a single consciousness indicator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorEvidence {
    /// Indicator ID (e.g., "RPT-1", "GWT-3").
    pub id: String,
    /// Theory of origin (e.g., "Recurrent Processing Theory").
    pub theory: String,
    /// Description of the indicator.
    pub description: String,
    /// Assessment status.
    pub status: IndicatorStatus,
    /// Detailed evidence string.
    pub evidence: String,
    /// Quantitative measure (if applicable, 0.0-1.0).
    pub score: Option<f64>,
}

/// Complete report of all consciousness indicators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ButlinIndicatorReport {
    /// All indicator evaluations.
    pub indicators: Vec<IndicatorEvidence>,
    /// Count of Present indicators.
    pub present_count: usize,
    /// Count of Partial indicators.
    pub partial_count: usize,
    /// Count of Absent indicators.
    pub absent_count: usize,
}

impl ButlinIndicatorReport {
    /// Build from a list of indicator evaluations.
    pub fn from_indicators(indicators: Vec<IndicatorEvidence>) -> Self {
        let present_count = indicators
            .iter()
            .filter(|i| i.status == IndicatorStatus::Present)
            .count();
        let partial_count = indicators
            .iter()
            .filter(|i| i.status == IndicatorStatus::Partial)
            .count();
        let absent_count = indicators
            .iter()
            .filter(|i| i.status == IndicatorStatus::Absent)
            .count();
        Self {
            indicators,
            present_count,
            partial_count,
            absent_count,
        }
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "=== Butlin et al. Consciousness Indicators ===".to_string(),
            format!(
                "  Present: {}, Partial: {}, Absent: {}",
                self.present_count, self.partial_count, self.absent_count
            ),
        ];
        for ind in &self.indicators {
            let score_str = ind
                .score
                .map(|s| format!(" ({:.2})", s))
                .unwrap_or_default();
            lines.push(format!(
                "  [{}] {} - {}: {}{}",
                ind.id, ind.status, ind.description, ind.evidence, score_str
            ));
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_consciousness_from_structural() {
        let data = RuntimeConsciousnessData::from_structural(0.1, 0.2, 0.3, 0.05, 1.5, 4);
        assert!((data.micro_phi - 0.1).abs() < f64::EPSILON);
        assert!((data.meso_phi - 0.2).abs() < f64::EPSILON);
        assert!((data.macro_phi - 0.3).abs() < f64::EPSILON);
        assert!((data.bottleneck_score - 0.05).abs() < f64::EPSILON);
        assert!((data.emergence_ratio - 1.5).abs() < f64::EPSILON);
        assert_eq!(data.num_clusters, 4);
    }
}
