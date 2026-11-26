// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark the scientific-discovery loop as an epistemic process.
//!
//! The harness intentionally scores *process properties* (calibration,
//! discrimination, provenance, and negative-result retention), not whether a
//! scientific claim is true. It is deterministic and uses no external data.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationSummary {
    pub brier_score: f64,
    pub log_loss: f64,
}

pub fn brier_score(probabilities: &[f64], realized: usize) -> f64 {
    if probabilities.is_empty() || realized >= probabilities.len() { return f64::NAN; }
    probabilities.iter().enumerate()
        .map(|(i, p)| (p - if i == realized { 1.0 } else { 0.0 }).powi(2))
        .sum()
}

pub fn log_loss(probabilities: &[f64], realized: usize) -> f64 {
    if probabilities.is_empty() || realized >= probabilities.len() { return f64::NAN; }
    -(probabilities[realized].max(1e-15)).ln()
}

pub fn calibration(probabilities: &[f64], realized: usize) -> CalibrationSummary {
    CalibrationSummary {
        brier_score: brier_score(probabilities, realized),
        log_loss: log_loss(probabilities, realized),
    }
}

/// Fraction of lineage roots represented by the supplied model identifiers
/// that are unique. This is a structural independence diagnostic, not a
/// scientific evidence score.
pub fn lineage_diversity(lineages: &[String]) -> f64 {
    if lineages.is_empty() { return 0.0; }
    BTreeSet::<String>::from_iter(lineages.iter().cloned()).len() as f64 / lineages.len() as f64
}

/// Normalized reduction in entropy. Useful for evaluating whether a proposed
/// test actually discriminated models in a synthetic benchmark.
pub fn normalized_entropy_reduction(prior_entropy_bits: f64, posterior_entropy_bits: f64) -> f64 {
    if !prior_entropy_bits.is_finite() || prior_entropy_bits <= 0.0 { return 0.0; }
    ((prior_entropy_bits - posterior_entropy_bits) / prior_entropy_bits).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_prediction_has_zero_brier_and_log_loss() {
        let c = calibration(&[1.0, 0.0], 0);
        assert_eq!(c.brier_score, 0.0);
        assert_eq!(c.log_loss, 0.0);
    }

    #[test]
    fn calibration_penalizes_overconfidence() {
        let perfect = brier_score(&[1.0, 0.0], 0);
        let wrong = brier_score(&[1.0, 0.0], 1);
        assert!(wrong > perfect);
    }

    #[test]
    fn lineage_diversity_detects_shared_ancestry() {
        assert_eq!(lineage_diversity(&["a".into(), "a".into(), "b".into()]), 2.0 / 3.0);
        assert_eq!(lineage_diversity(&["a".into(), "b".into()]), 1.0);
    }

    #[test]
    fn entropy_reduction_is_bounded() {
        assert_eq!(normalized_entropy_reduction(2.0, 0.0), 1.0);
        assert_eq!(normalized_entropy_reduction(2.0, 3.0), 0.0);
    }
}
