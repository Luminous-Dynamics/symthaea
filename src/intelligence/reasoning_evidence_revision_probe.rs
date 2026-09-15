// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RQ-006D development probe for evidence-revision behavior.
//!
//! This module measures two narrow properties of the current epistemic-conflict substrate:
//! 1. whether reliability moves in the expected direction when evidence becomes contradictory
//!    and later resolves; and
//! 2. whether a deliberately weakened theory produces the intended typed epistemic action.
//!
//! This is a development probe, not evidence that answer-level confidence is calibrated.
//! `TheoryCalibrator::reliability` is used only as the subject signal whose revision direction is
//! measured. It is not re-labelled as a probability that an arbitrary answer is correct.

use crate::consciousness::epistemic_conflict::{
    AnchorKind, ConflictDetector, ConflictKind, EpistemicAction, MultiTheoryMetrics,
    TheoryCalibrator, TheoryId,
};
use serde::{Deserialize, Serialize};

use super::reasoning_metacognition::{
    ConfidenceRevisionDirection, ConfidenceRevisionObservation, MetacognitionEvaluationError,
    MetacognitionReport, evaluate_metacognition,
};

pub const EVIDENCE_REVISION_PROBE_VERSION: &str = "rq-006d-evidence-revision-v1";
const STABLE_TOLERANCE: f64 = 1.0e-12;

/// One mechanically specified conflict/action observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConflictActionObservation {
    pub weak_theory: TheoryId,
    pub expected_kind: ConflictKind,
    pub expected_action: EpistemicAction,
    /// Number of high-magnitude pairwise conflicts involving the weak theory.
    pub relevant_conflicts: usize,
    /// Number whose kind and action both match the predeclared expectation.
    pub matching_conflicts: usize,
}

/// Decomposed result of the public RQ-006D development probe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRevisionProbeReport {
    pub probe_version: String,
    pub aligned_reliability: f64,
    pub contradicted_reliability: f64,
    pub resolved_reliability: f64,
    pub uniformly_shifted_reliability: f64,
    pub revision: MetacognitionReport,
    pub conflict_actions: Vec<ConflictActionObservation>,
    pub conflict_action_accuracy: Option<f64>,
}

/// Execute the fixed public development probe.
///
/// Subject failures are returned as measured values in the report. The function only returns an
/// error when the RQ-006 evaluator rejects the measurement record itself.
pub fn run_evidence_revision_probe(
) -> Result<EvidenceRevisionProbeReport, MetacognitionEvaluationError> {
    let calibrator = TheoryCalibrator::new();

    // The first, third, and fourth fixtures have identical cross-theory agreement. The fourth
    // shifts every theory by the same amount, so a disagreement-based reliability signal should
    // remain invariant to that translation.
    let aligned = metrics([0.80, 0.80, 0.80, 0.80, 0.80, 0.80]);
    let contradicted = metrics([0.90, 0.10, 0.90, 0.10, 0.90, 0.10]);
    let resolved = metrics([0.70, 0.70, 0.70, 0.70, 0.70, 0.70]);
    let uniformly_shifted = metrics([0.20, 0.20, 0.20, 0.20, 0.20, 0.20]);

    let aligned_reliability = calibrator.reliability(&aligned);
    let contradicted_reliability = calibrator.reliability(&contradicted);
    let resolved_reliability = calibrator.reliability(&resolved);
    let uniformly_shifted_reliability = calibrator.reliability(&uniformly_shifted);

    let revisions = vec![
        ConfidenceRevisionObservation {
            pair_id: "aligned-to-contradicted".into(),
            before_episode_id: "aligned".into(),
            after_episode_id: "contradicted".into(),
            before_confidence: aligned_reliability,
            after_confidence: contradicted_reliability,
            expected_direction: ConfidenceRevisionDirection::Decrease,
            stable_tolerance: STABLE_TOLERANCE,
        },
        ConfidenceRevisionObservation {
            pair_id: "contradicted-to-resolved".into(),
            before_episode_id: "contradicted".into(),
            after_episode_id: "resolved".into(),
            before_confidence: contradicted_reliability,
            after_confidence: resolved_reliability,
            expected_direction: ConfidenceRevisionDirection::Increase,
            stable_tolerance: STABLE_TOLERANCE,
        },
        ConfidenceRevisionObservation {
            pair_id: "aligned-to-uniform-shift".into(),
            before_episode_id: "aligned".into(),
            after_episode_id: "uniform-shift".into(),
            before_confidence: aligned_reliability,
            after_confidence: uniformly_shifted_reliability,
            expected_direction: ConfidenceRevisionDirection::Stable,
            stable_tolerance: STABLE_TOLERANCE,
        },
    ];

    let revision = evaluate_metacognition(&[], &[], &revisions, 10, &[])?;
    let conflict_actions = conflict_action_observations();
    let relevant = conflict_actions
        .iter()
        .map(|observation| observation.relevant_conflicts)
        .sum::<usize>();
    let matching = conflict_actions
        .iter()
        .map(|observation| observation.matching_conflicts)
        .sum::<usize>();
    let conflict_action_accuracy = if relevant == 0 {
        None
    } else {
        Some(matching as f64 / relevant as f64)
    };

    Ok(EvidenceRevisionProbeReport {
        probe_version: EVIDENCE_REVISION_PROBE_VERSION.into(),
        aligned_reliability,
        contradicted_reliability,
        resolved_reliability,
        uniformly_shifted_reliability,
        revision,
        conflict_actions,
        conflict_action_accuracy,
    })
}

fn conflict_action_observations() -> Vec<ConflictActionObservation> {
    let cases = [
        (
            TheoryId::IIT,
            ConflictKind::IntegrationCollapse,
            EpistemicAction::Summarize,
        ),
        (
            TheoryId::GWT,
            ConflictKind::NoBroadcast,
            EpistemicAction::Verify(AnchorKind::ReadOnlyQuery),
        ),
        (
            TheoryId::AST,
            ConflictKind::AttentionalInstability,
            EpistemicAction::Ask,
        ),
        (
            TheoryId::PP,
            ConflictKind::UnreliablePrediction,
            EpistemicAction::Sense(AnchorKind::SensorMeasurement),
        ),
        (
            TheoryId::RPT,
            ConflictKind::ShallowRecurrence,
            EpistemicAction::Simulate,
        ),
        (
            TheoryId::FourE,
            ConflictKind::UngroundedRepresentation,
            EpistemicAction::Verify(AnchorKind::DeterministicProbe),
        ),
    ];

    cases
        .into_iter()
        .map(|(weak_theory, expected_kind, expected_action)| {
            let mut values = [0.80; 6];
            values[weak_theory.index()] = 0.10;
            let mut detector = ConflictDetector::new();
            let matrix = detector.detect(&metrics(values));
            let relevant: Vec<_> = matrix
                .conflicts
                .iter()
                .filter(|conflict| {
                    conflict.magnitude > 0.60
                        && (conflict.theory_a == weak_theory
                            || conflict.theory_b == weak_theory)
                })
                .collect();
            let matching_conflicts = relevant
                .iter()
                .filter(|conflict| {
                    conflict.kind == expected_kind
                        && conflict.recommended_action == expected_action
                })
                .count();

            ConflictActionObservation {
                weak_theory,
                expected_kind,
                expected_action,
                relevant_conflicts: relevant.len(),
                matching_conflicts,
            }
        })
        .collect()
}

fn metrics(values: [f64; 6]) -> MultiTheoryMetrics {
    MultiTheoryMetrics {
        phi: values[0],
        gwt: values[1],
        ast: values[2],
        pp: values[3],
        rpt: values[4],
        embodiment: values[5],
        unified: values.iter().sum::<f64>() / values.len() as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contradictory_evidence_reduces_and_resolution_restores_reliability() {
        let report = run_evidence_revision_probe().expect("development probe must be well formed");
        assert!(
            report.contradicted_reliability < report.aligned_reliability,
            "contradiction must reduce the measured epistemic reliability"
        );
        assert!(
            report.resolved_reliability > report.contradicted_reliability,
            "resolving contradiction must restore reliability"
        );
        assert_eq!(report.revision.revision_direction_accuracy, Some(1.0));
        assert_eq!(report.revision.confidence_revisions, 3);
    }

    #[test]
    fn uniform_evidence_shift_does_not_fake_a_revision() {
        let report = run_evidence_revision_probe().expect("development probe must be well formed");
        assert!(
            (report.aligned_reliability - report.uniformly_shifted_reliability).abs()
                <= STABLE_TOLERANCE,
            "uniform translation should not change disagreement-derived reliability"
        );
    }

    #[test]
    fn every_single_theory_weakness_maps_to_its_typed_epistemic_action() {
        let report = run_evidence_revision_probe().expect("development probe must be well formed");
        assert_eq!(report.conflict_actions.len(), 6);
        for observation in &report.conflict_actions {
            assert_eq!(
                observation.relevant_conflicts, 5,
                "one weak theory should create five high-magnitude pairwise conflicts"
            );
            assert_eq!(
                observation.matching_conflicts, observation.relevant_conflicts,
                "all conflicts involving the weak theory should preserve its typed remediation"
            );
        }
        assert_eq!(report.conflict_action_accuracy, Some(1.0));
    }
}
