//! Evaluator-side scoring helpers for ICCB.
//!
//! This package depends on the candidate-contract crate; the candidate crate
//! does not depend on this package. Keep hidden scenario truth and held-out data
//! in the evaluator process/storage, never in candidate-visible observations.
//!
//! Public development tests here are not a secret or held-out dataset. Package
//! separation is an architectural boundary for data flow, not a security sandbox.

use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateDependencyReport {
    pub scenario_id: String,
    pub detected_critical_dependency_ids: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluatorTruth {
    pub scenario_id: String,
    pub critical_dependency_ids: BTreeSet<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationError {
    ScenarioIdMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencyDetectionScore {
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub candidate_report_count: usize,
    pub ground_truth_count: usize,
}

impl DependencyDetectionScore {
    /// Undefined when neither the candidate nor the truth contains a positive.
    pub fn precision(&self) -> Option<f64> {
        let denominator = self.true_positives + self.false_positives;
        (denominator > 0).then(|| self.true_positives as f64 / denominator as f64)
    }

    /// Undefined when there are no critical dependencies in the truth.
    pub fn recall(&self) -> Option<f64> {
        (self.ground_truth_count > 0)
            .then(|| self.true_positives as f64 / self.ground_truth_count as f64)
    }

    /// F1 is a convenience view, never a replacement for the component counts.
    pub fn f1(&self) -> Option<f64> {
        let denominator = 2 * self.true_positives + self.false_positives + self.false_negatives;
        (denominator > 0).then(|| (2 * self.true_positives) as f64 / denominator as f64)
    }
}

/// Scores reported detections against evaluator-owned truth.
///
/// Returns an error instead of comparing unrelated scenarios. Candidate reports
/// contain only the candidate's declared detections; the truth object must be
/// supplied only inside the evaluator process.
pub fn score_dependency_detection(
    candidate: &CandidateDependencyReport,
    truth: &EvaluatorTruth,
) -> Result<DependencyDetectionScore, EvaluationError> {
    if candidate.scenario_id != truth.scenario_id {
        return Err(EvaluationError::ScenarioIdMismatch);
    }

    let true_positives = candidate
        .detected_critical_dependency_ids
        .intersection(&truth.critical_dependency_ids)
        .count();
    let false_positives = candidate
        .detected_critical_dependency_ids
        .difference(&truth.critical_dependency_ids)
        .count();
    let false_negatives = truth
        .critical_dependency_ids
        .difference(&candidate.detected_critical_dependency_ids)
        .count();

    Ok(DependencyDetectionScore {
        true_positives,
        false_positives,
        false_negatives,
        candidate_report_count: candidate.detected_critical_dependency_ids.len(),
        ground_truth_count: truth.critical_dependency_ids.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn exact_detection_has_perfect_component_metrics() {
        let candidate = CandidateDependencyReport {
            scenario_id: "fixture-1".into(),
            detected_critical_dependency_ids: set(&["grid-a", "water-b"]),
        };
        let truth = EvaluatorTruth {
            scenario_id: "fixture-1".into(),
            critical_dependency_ids: set(&["grid-a", "water-b"]),
        };

        let score = score_dependency_detection(&candidate, &truth).unwrap();
        assert_eq!(score.true_positives, 2);
        assert_eq!(score.false_positives, 0);
        assert_eq!(score.false_negatives, 0);
        assert_eq!(score.precision(), Some(1.0));
        assert_eq!(score.recall(), Some(1.0));
        assert_eq!(score.f1(), Some(1.0));
    }

    #[test]
    fn false_positive_and_false_negative_are_reported_separately() {
        let candidate = CandidateDependencyReport {
            scenario_id: "fixture-2".into(),
            detected_critical_dependency_ids: set(&["grid-a", "invented-route"]),
        };
        let truth = EvaluatorTruth {
            scenario_id: "fixture-2".into(),
            critical_dependency_ids: set(&["grid-a", "water-b"]),
        };

        let score = score_dependency_detection(&candidate, &truth).unwrap();
        assert_eq!(score.true_positives, 1);
        assert_eq!(score.false_positives, 1);
        assert_eq!(score.false_negatives, 1);
        assert_eq!(score.precision(), Some(0.5));
        assert_eq!(score.recall(), Some(0.5));
        assert_eq!(score.f1(), Some(0.5));
    }

    #[test]
    fn mismatched_scenario_ids_fail_instead_of_scoring() {
        let candidate = CandidateDependencyReport {
            scenario_id: "fixture-a".into(),
            detected_critical_dependency_ids: set(&["grid-a"]),
        };
        let truth = EvaluatorTruth {
            scenario_id: "fixture-b".into(),
            critical_dependency_ids: set(&["grid-a"]),
        };

        assert_eq!(
            score_dependency_detection(&candidate, &truth),
            Err(EvaluationError::ScenarioIdMismatch)
        );
    }

    #[test]
    fn empty_truth_does_not_make_recall_look_perfect() {
        let candidate = CandidateDependencyReport {
            scenario_id: "fixture-3".into(),
            detected_critical_dependency_ids: BTreeSet::new(),
        };
        let truth = EvaluatorTruth {
            scenario_id: "fixture-3".into(),
            critical_dependency_ids: BTreeSet::new(),
        };

        let score = score_dependency_detection(&candidate, &truth).unwrap();
        assert_eq!(score.precision(), None);
        assert_eq!(score.recall(), None);
        assert_eq!(score.f1(), None);
    }
}
