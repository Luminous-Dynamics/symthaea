// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic Conflict Detector
//!
//! Detects 15 pairwise conflicts between 6 consciousness theories.
//! Each conflict is typed with a `ConflictKind` and produces a
//! recommended `EpistemicAction` for resolution.

use super::types::*;

/// Threshold for marking a conflict as chronic (persisted > N cycles).
const CHRONIC_THRESHOLD: usize = 20;

/// Conflict detector engine: computes the 15-element conflict matrix
/// from multi-theory metrics each cycle.
#[derive(Debug, Clone)]
pub struct ConflictDetector {
    /// Previous conflict scores for trend/chronic tracking.
    previous: Option<Vec<ConflictScore>>,
    /// Number of consecutive cycles each pair has been in conflict.
    chronic_counters: [usize; ConflictMatrix::NUM_PAIRS],
    /// Cycle counter.
    cycle_count: u64,
}

impl ConflictDetector {
    pub fn new() -> Self {
        Self {
            previous: None,
            chronic_counters: [0; ConflictMatrix::NUM_PAIRS],
            cycle_count: 0,
        }
    }

    /// Detect conflicts from multi-theory metrics.
    ///
    /// Returns a `ConflictMatrix` with 15 pairwise scores, entropy,
    /// and the dominant conflict.
    pub fn detect(&mut self, metrics: &MultiTheoryMetrics) -> ConflictMatrix {
        self.cycle_count += 1;
        let values = Self::extract_values(metrics);
        let mut scores = Vec::with_capacity(ConflictMatrix::NUM_PAIRS);

        let mut pair_idx = 0;
        for i in 0..6 {
            for j in (i + 1)..6 {
                let theory_a = TheoryId::ALL[i];
                let theory_b = TheoryId::ALL[j];
                let magnitude = (values[i] - values[j]).abs();
                let kind = Self::classify_conflict(theory_a, theory_b, values[i], values[j]);

                let mut score = if let Some(prev) = &self.previous {
                    let mut s = prev[pair_idx].clone();
                    s.update(magnitude);
                    s.kind = kind;
                    s.recommended_action = kind.recommended_action();
                    s
                } else {
                    ConflictScore::new(theory_a, theory_b, magnitude, kind)
                };

                // Track chronicity
                if magnitude > 0.3 {
                    self.chronic_counters[pair_idx] += 1;
                } else {
                    self.chronic_counters[pair_idx] =
                        self.chronic_counters[pair_idx].saturating_sub(1);
                }
                score.is_chronic = self.chronic_counters[pair_idx] > CHRONIC_THRESHOLD;

                // EVoI: higher when conflict is large and trending up
                score.evoi = magnitude * (1.0 + score.trend().max(0.0));

                scores.push(score);
                pair_idx += 1;
            }
        }

        self.previous = Some(scores.clone());
        ConflictMatrix::new(scores)
    }

    /// Extract the 6 theory metric values in canonical order.
    fn extract_values(metrics: &MultiTheoryMetrics) -> [f64; 6] {
        [
            metrics.phi,
            metrics.gwt,
            metrics.ast,
            metrics.pp,
            metrics.rpt,
            metrics.embodiment,
        ]
    }

    /// Classify which conflict kind applies between two theories,
    /// based on which theory is weaker.
    fn classify_conflict(
        theory_a: TheoryId,
        theory_b: TheoryId,
        value_a: f64,
        value_b: f64,
    ) -> ConflictKind {
        let weaker = if value_a < value_b {
            theory_a
        } else {
            theory_b
        };
        match weaker {
            TheoryId::IIT => ConflictKind::IntegrationCollapse,
            TheoryId::GWT => ConflictKind::NoBroadcast,
            TheoryId::AST => ConflictKind::AttentionalInstability,
            TheoryId::PP => ConflictKind::UnreliablePrediction,
            TheoryId::RPT => ConflictKind::ShallowRecurrence,
            TheoryId::FourE => ConflictKind::UngroundedRepresentation,
        }
    }
}

impl Default for ConflictDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(
        phi: f64,
        gwt: f64,
        ast: f64,
        pp: f64,
        rpt: f64,
        emb: f64,
    ) -> crate::consciousness::epistemic_conflict::MultiTheoryMetrics {
        crate::consciousness::epistemic_conflict::MultiTheoryMetrics {
            phi,
            gwt,
            ast,
            pp,
            rpt,
            embodiment: emb,
            unified: phi * 0.20 + gwt * 0.15 + ast * 0.15 + pp * 0.20 + rpt * 0.15 + emb * 0.15,
        }
    }

    #[test]
    fn test_detect_produces_15_conflicts() {
        let mut detector = ConflictDetector::new();
        let metrics = make_metrics(0.8, 0.7, 0.6, 0.5, 0.4, 0.3);
        let matrix = detector.detect(&metrics);
        assert_eq!(matrix.conflicts.len(), 15);
    }

    #[test]
    fn test_conflict_symmetry() {
        let mut detector = ConflictDetector::new();
        let metrics = make_metrics(0.9, 0.1, 0.5, 0.5, 0.5, 0.5);
        let matrix = detector.detect(&metrics);

        // IIT-GWT conflict should be the largest (0.9 - 0.1 = 0.8)
        let iit_gwt = matrix.get(TheoryId::IIT, TheoryId::GWT).unwrap();
        assert!((iit_gwt.magnitude - 0.8).abs() < 1e-10);

        // Symmetry: get(A,B) == get(B,A)
        let gwt_iit = matrix.get(TheoryId::GWT, TheoryId::IIT).unwrap();
        assert_eq!(iit_gwt.magnitude, gwt_iit.magnitude);
    }

    #[test]
    fn test_entropy_zero_when_uniform() {
        let mut detector = ConflictDetector::new();
        // All theories at same value → zero conflict
        let metrics = make_metrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
        let matrix = detector.detect(&metrics);
        assert!(
            matrix.total_entropy.abs() < 1e-10,
            "Entropy should be ~0 for uniform metrics"
        );
        assert!(matrix.max_magnitude() < 1e-10);
    }

    #[test]
    fn test_dominant_conflict_detected() {
        let mut detector = ConflictDetector::new();
        let metrics = make_metrics(0.9, 0.1, 0.5, 0.5, 0.5, 0.5);
        let matrix = detector.detect(&metrics);
        assert_eq!(matrix.dominant, Some((TheoryId::IIT, TheoryId::GWT)));
    }

    #[test]
    fn test_conflict_kind_classification() {
        let mut detector = ConflictDetector::new();
        // GWT is very low → NoBroadcast should appear
        let metrics = make_metrics(0.8, 0.1, 0.8, 0.8, 0.8, 0.8);
        let matrix = detector.detect(&metrics);
        let iit_gwt = matrix.get(TheoryId::IIT, TheoryId::GWT).unwrap();
        assert_eq!(iit_gwt.kind, ConflictKind::NoBroadcast);
    }

    #[test]
    fn test_chronic_detection() {
        let mut detector = ConflictDetector::new();
        let metrics = make_metrics(0.9, 0.1, 0.5, 0.5, 0.5, 0.5);

        // Run enough cycles for chronic threshold
        for _ in 0..(CHRONIC_THRESHOLD + 5) {
            detector.detect(&metrics);
        }

        let matrix = detector.detect(&metrics);
        let iit_gwt = matrix.get(TheoryId::IIT, TheoryId::GWT).unwrap();
        assert!(
            iit_gwt.is_chronic,
            "Should be marked chronic after {} cycles",
            CHRONIC_THRESHOLD
        );
    }
}
