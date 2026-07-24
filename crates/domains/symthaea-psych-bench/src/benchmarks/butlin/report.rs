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
    /// Real, mechanism-specific behavioral measurements from
    /// `ablation::measure_indicator` — the same probes the ablation matrix
    /// uses to prove a mechanism load-bearing. When present, these replace
    /// the structural-Phi-sigmoid proxy for the indicators they cover.
    #[serde(default)]
    pub behavioral: Option<BehavioralIndicatorSignals>,
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
            behavioral: None,
        }
    }

    /// Attach real behavioral measurements (see `ablation::measure_indicator`).
    pub fn with_behavioral(mut self, behavioral: BehavioralIndicatorSignals) -> Self {
        self.behavioral = Some(behavioral);
        self
    }
}

/// Real, mechanism-specific measurements for the 11 indicators the ablation
/// matrix already validates as load-bearing (or honestly not, for indicators
/// blocked on known separate bugs — see field docs) (see
/// `ablation::run_ablation_matrix`'s per-row causal effects). All fields are
/// the same probes `ablation::measure_indicator` computes, run here against
/// a live (non-ablated) service rather than a baseline-vs-ablated pair.
///
/// GWT-1 and IIT-1 are deliberately not fields here — GWT-1 is derived from
/// the other fields' aggregate in `indicators.rs`, and IIT-1's "is Phi
/// sensitive to ablation" claim can only be tested via a baseline-vs-ablated
/// comparison (see `ablation_specs`'s `disable_gwt_for_iit1` row), not a
/// single live snapshot, so it keeps using the structural-Phi proxy for live
/// scoring.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BehavioralIndicatorSignals {
    /// RPT-1: input-discrimination / temporal-coherence proxy (0-1).
    pub rpt1_temporal_coherence: f64,
    /// RPT-2: fraction of cycles with active cross-modal binding (0-1).
    pub rpt2_binding_activity: f64,
    /// GWT-2: fraction of cycles with a non-empty, bounded GWT coalition (0-1).
    pub gwt2_bounded_coalition: f64,
    /// GWT-3: fraction of cycles with an active GWT broadcast (0-1).
    pub gwt3_broadcast_activity: f64,
    /// GWT-4: mean deviation of phi_attention_weight from neutral (0-1).
    pub gwt4_state_dependent_attention: f64,
    /// HOT-1: variance-based signal for whether prediction_error actually
    /// differentiates across inputs (0-1) — honestly near-zero while PE is
    /// frozen (see memory/symthaea_prediction_error_frozen_investigation.md).
    pub hot1_prediction_differentiation: f64,
    /// HOT-2: metacognitive monitoring accuracy (0-1).
    pub hot2_meta_cognitive_accuracy: f64,
    /// HOT-3: effective learning rate actually applied this cycle (raw units;
    /// treated as a presence signal — see `indicators.rs`'s use site). Same
    /// underlying signal as PP-1, different Butlin theoretical claim.
    pub hot3_effective_lr: f64,
    /// PP-1: effective learning rate actually applied this cycle (raw units;
    /// treated as a presence signal — see `indicators.rs`'s use site).
    pub pp1_effective_lr: f64,
    /// PP-2: fraction of cycles with active hierarchical free-energy
    /// computation (0-1) — a module-engagement proxy, coarser than a true
    /// per-tau-level error trace (not currently surfaced on CycleMetadata).
    pub pp2_hierarchical_activity: f64,
    /// AST-1: attention-schema focus signal (0-1, non-zero fallback per
    /// `ablation::extract_indicator_score`).
    pub ast1_attention_focus: f64,
    /// HOT-4: fraction of near-zero output dimensions, averaged over several
    /// distinct inputs (0-1). Needs no cognitive-loop ablation at all — see
    /// `live_runner::CognitiveLoopBenchmarkRunner::measure_hot4_sparse_smooth_coding`.
    pub hot4_sparsity: f64,
    /// HOT-4: fraction of perturbation steps for which output dissimilarity
    /// grows non-decreasingly with perturbation size (0-1) — a genuinely
    /// smooth code shouldn't respond discontinuously to small changes.
    pub hot4_smoothness: f64,
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
