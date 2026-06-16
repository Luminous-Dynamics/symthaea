// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Anomaly Investigation (Path C)
//!
//! The central mystery: NASA observed ~10³ n/s while Gamow predicts ~10⁻⁵⁰.
//! This module provides tools to investigate potential explanations.

use crate::constants::*;
use crate::physics::GamowResult;
use serde::{Deserialize, Serialize};

/// Analysis of the gap between observed and predicted rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyGap {
    /// Observed neutron rate (n/s)
    pub observed_rate: f64,
    /// Predicted rate from standard Gamow physics (n/s)
    pub predicted_rate: f64,
    /// Gap factor (observed / predicted)
    pub gap_factor: f64,
    /// Log10 of gap factor
    pub gap_orders_of_magnitude: f64,
    /// Assessment of gap significance
    pub significance: GapSignificance,
}

/// Classification of gap significance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GapSignificance {
    /// Gap < 10×: within experimental/modeling uncertainty
    WithinUncertainty,
    /// Gap 10-1000×: significant, needs investigation
    Significant,
    /// Gap > 1000×: anomalous, suggests new physics or systematic error
    Anomalous,
    /// Gap > 10^10: extraordinary, likely measurement error or new physics
    Extraordinary,
}

/// Hypothesis for explaining the anomaly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyHypothesis {
    /// Hypothesis name
    pub name: String,
    /// Brief description
    pub description: String,
    /// Enhancement factor this hypothesis could provide
    pub potential_enhancement: f64,
    /// Whether this explains the full gap
    pub explains_gap: bool,
    /// Testable predictions
    pub predictions: Vec<String>,
    /// Experiments to validate
    pub validation_experiments: Vec<String>,
    /// Current status/evidence
    pub evidence_status: EvidenceStatus,
}

/// Evidence status for a hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceStatus {
    /// No experimental evidence
    Untested,
    /// Some supporting evidence
    PartialSupport,
    /// Strong supporting evidence
    StrongSupport,
    /// Contradicted by evidence
    Contradicted,
    /// Evidence is ambiguous
    Ambiguous,
}

/// Tools for anomaly analysis.
pub struct AnomalyAnalysis;

impl AnomalyAnalysis {
    /// Compute the gap between observed and predicted rates.
    pub fn compute_gap(observed: f64, predicted: f64) -> AnomalyGap {
        let gap_factor = if predicted > 0.0 {
            observed / predicted
        } else {
            f64::INFINITY
        };

        let gap_log = if gap_factor > 0.0 && gap_factor.is_finite() {
            gap_factor.log10()
        } else {
            f64::INFINITY
        };

        let significance = if gap_factor < 10.0 {
            GapSignificance::WithinUncertainty
        } else if gap_factor < 1000.0 {
            GapSignificance::Significant
        } else if gap_factor < 1e10 {
            GapSignificance::Anomalous
        } else {
            GapSignificance::Extraordinary
        };

        AnomalyGap {
            observed_rate: observed,
            predicted_rate: predicted,
            gap_factor,
            gap_orders_of_magnitude: gap_log,
            significance,
        }
    }

    /// Analyze NASA LCF results against standard physics.
    pub fn analyze_nasa_results(gamow: &GamowResult, volume_cm3: f64) -> AnomalyGap {
        // NASA conditions: ~1 cm² target, assume 25 μm depth
        let n_d = 0.7 * 12.02 * 6.022e23 / 106.42; // D atoms/cm³

        let predicted = gamow.to_neutron_rate(n_d, volume_cm3);

        Self::compute_gap(NASA_OBSERVED_RATE, predicted)
    }

    /// Generate standard hypotheses for the anomaly.
    pub fn standard_hypotheses() -> Vec<AnomalyHypothesis> {
        vec![
            // Hypothesis 1: Enhanced screening beyond Assenbaum
            AnomalyHypothesis {
                name: "Super-Screening".to_string(),
                description: "Screening enhancement beyond measured values due to \
                    coherent electronic effects in loaded lattice".to_string(),
                potential_enhancement: 1e3,
                explains_gap: false, // Even 1000× isn't enough
                predictions: vec![
                    "Screening should increase with D/Pd loading".to_string(),
                    "Should show temperature dependence different from Debye".to_string(),
                ],
                validation_experiments: vec![
                    "Measure screening at D/Pd > 0.7".to_string(),
                    "Map screening vs temperature precisely".to_string(),
                ],
                evidence_status: EvidenceStatus::PartialSupport,
            },
            // Hypothesis 2: Coherent multi-phonon enhancement
            AnomalyHypothesis {
                name: "Coherent Phonon Cascade".to_string(),
                description: "Multiple phonons coherently adding energy to D-D pairs, \
                    effectively raising CM energy by hundreds of meV".to_string(),
                potential_enhancement: 1e6,
                explains_gap: false,
                predictions: vec![
                    "Enhancement should correlate with phonon density of states".to_string(),
                    "Should show resonances at phonon energies".to_string(),
                ],
                validation_experiments: vec![
                    "Measure rate vs acoustic drive frequency".to_string(),
                    "Correlate with inelastic neutron scattering data".to_string(),
                ],
                evidence_status: EvidenceStatus::Untested,
            },
            // Hypothesis 3: Hot spots / localized heating
            AnomalyHypothesis {
                name: "Transient Hot Spots".to_string(),
                description: "X-ray absorption creates transient hot spots where \
                    local temperature briefly reaches thousands of K".to_string(),
                potential_enhancement: 1e20,
                explains_gap: true, // At 10000K, <σv> increases enormously
                predictions: vec![
                    "Should see rate increase with X-ray intensity".to_string(),
                    "Peak width should correlate with thermal diffusivity".to_string(),
                    "Should saturate at high intensity (thermal equilibration)".to_string(),
                ],
                validation_experiments: vec![
                    "Time-resolved X-ray diffraction during trigger".to_string(),
                    "Vary X-ray pulse duration and measure rate".to_string(),
                ],
                evidence_status: EvidenceStatus::Untested,
            },
            // Hypothesis 4: Novel nuclear physics
            AnomalyHypothesis {
                name: "Lattice-Modified Nuclear Interaction".to_string(),
                description: "The periodic potential of the lattice modifies the \
                    D-D nuclear interaction at short range".to_string(),
                potential_enhancement: 1e50,
                explains_gap: true,
                predictions: vec![
                    "Rate should depend on lattice structure (FCC vs BCC)".to_string(),
                    "D-D branching ratio might differ from free-space".to_string(),
                    "Neutron spectrum might show lattice effects".to_string(),
                ],
                validation_experiments: vec![
                    "Compare rates in Pd (FCC) vs Ti (HCP) vs V (BCC)".to_string(),
                    "Precision branching ratio measurement".to_string(),
                    "High-resolution neutron spectroscopy".to_string(),
                ],
                evidence_status: EvidenceStatus::Untested,
            },
            // Hypothesis 5: Measurement artifact
            AnomalyHypothesis {
                name: "Systematic Measurement Error".to_string(),
                description: "The observed signal is not from D-D fusion but from \
                    background, cosmic rays, or electronic noise".to_string(),
                potential_enhancement: 0.0, // Not enhancement, but error
                explains_gap: true,
                predictions: vec![
                    "Signal should not correlate with deuterium loading".to_string(),
                    "Should see signal without deuterium".to_string(),
                ],
                validation_experiments: vec![
                    "Run control with H instead of D".to_string(),
                    "Independent replication at different facility".to_string(),
                    "Coincidence measurement to verify fusion signature".to_string(),
                ],
                evidence_status: EvidenceStatus::Ambiguous,
            },
        ]
    }

    /// Rank hypotheses by plausibility given the gap.
    pub fn rank_hypotheses(
        hypotheses: &[AnomalyHypothesis],
        gap: &AnomalyGap,
    ) -> Vec<(usize, f64)> {
        let mut scores: Vec<(usize, f64)> = hypotheses
            .iter()
            .enumerate()
            .map(|(i, h)| {
                let mut score = 0.0;

                // Does it explain the gap?
                if h.explains_gap {
                    score += 50.0;
                } else if h.potential_enhancement > 0.0 {
                    // Partial credit based on how much it explains
                    let explained_log = h.potential_enhancement.log10();
                    let gap_log = gap.gap_orders_of_magnitude;
                    score += 50.0 * (explained_log / gap_log).min(1.0);
                }

                // Evidence status
                match h.evidence_status {
                    EvidenceStatus::StrongSupport => score += 30.0,
                    EvidenceStatus::PartialSupport => score += 15.0,
                    EvidenceStatus::Untested => score += 5.0,
                    EvidenceStatus::Ambiguous => score += 0.0,
                    EvidenceStatus::Contradicted => score -= 50.0,
                }

                // Testability (more predictions = better)
                score += (h.predictions.len() as f64) * 2.0;
                score += (h.validation_experiments.len() as f64) * 3.0;

                (i, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Suggest next experimental steps based on gap analysis.
    pub fn suggest_experiments(gap: &AnomalyGap) -> Vec<String> {
        let mut suggestions = vec![];

        match gap.significance {
            GapSignificance::WithinUncertainty => {
                suggestions.push("Gap is within uncertainty - improve measurement precision".to_string());
            }
            GapSignificance::Significant => {
                suggestions.push("1. Verify neutron spectrum matches D-D (2.45 MeV)".to_string());
                suggestions.push("2. Run H control to rule out non-fusion sources".to_string());
                suggestions.push("3. Map rate vs D/Pd loading".to_string());
            }
            GapSignificance::Anomalous => {
                suggestions.push("1. Independent replication at different lab".to_string());
                suggestions.push("2. Precision branching ratio measurement".to_string());
                suggestions.push("3. Time-resolved diagnostics during trigger".to_string());
                suggestions.push("4. Compare multiple host lattices (Pd, Ti, Ni)".to_string());
            }
            GapSignificance::Extraordinary => {
                suggestions.push("1. CRITICAL: Independent replication required".to_string());
                suggestions.push("2. Extensive background characterization".to_string());
                suggestions.push("3. Multiple detection methods (n, γ, charged particles)".to_string());
                suggestions.push("4. Theoretical engagement for new physics models".to_string());
            }
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gap_computation() {
        let gap = AnomalyAnalysis::compute_gap(1000.0, 1.0);
        assert_eq!(gap.gap_factor, 1000.0);
        assert!((gap.gap_orders_of_magnitude - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_gap_significance() {
        let small = AnomalyAnalysis::compute_gap(5.0, 1.0);
        assert_eq!(small.significance, GapSignificance::WithinUncertainty);

        let large = AnomalyAnalysis::compute_gap(1e20, 1.0);
        assert_eq!(large.significance, GapSignificance::Extraordinary);
    }

    #[test]
    fn test_standard_hypotheses() {
        let hypotheses = AnomalyAnalysis::standard_hypotheses();
        assert!(!hypotheses.is_empty());

        // At least one should explain the gap
        assert!(hypotheses.iter().any(|h| h.explains_gap));
    }

    #[test]
    fn test_hypothesis_ranking() {
        let hypotheses = AnomalyAnalysis::standard_hypotheses();
        let gap = AnomalyAnalysis::compute_gap(1e3, 1e-47);
        let ranked = AnomalyAnalysis::rank_hypotheses(&hypotheses, &gap);

        assert_eq!(ranked.len(), hypotheses.len());
        // Scores should be descending
        for i in 1..ranked.len() {
            assert!(ranked[i].1 <= ranked[i - 1].1);
        }
    }
}
