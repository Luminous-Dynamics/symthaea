// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact one-step industrial diagnosis / value-of-information benchmark.
//!
//! This module compares measurement selection by expected information gain
//! against selection by expected reduction in decision loss after accounting
//! for measurement cost. It is a synthetic decision-support benchmark, not a
//! diagnostic authority or a production control system.

use std::collections::BTreeSet;

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const METRIC_TOLERANCE: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceStatus {
    Current,
    Stale,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    pub id: &'static str,
    pub prior: f64,
    /// Loss for each decision in DecisionCase.decisions order.
    pub decision_losses: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionCase {
    pub schema_version: &'static str,
    pub case_id: &'static str,
    pub hypotheses: Vec<Hypothesis>,
    pub decisions: Vec<&'static str>,
    pub existing_evidence_origins: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    pub id: &'static str,
    pub cost_units: f64,
    /// P(positive | each hypothesis), in DecisionCase.hypotheses order.
    /// None means the measurement likelihood model is unknown.
    pub positive_likelihoods: Option<Vec<f64>>,
    pub calibration: EvidenceStatus,
    pub applicability: EvidenceStatus,
    pub authority: EvidenceStatus,
    pub evidence_origin: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaseInputFailure {
    UnsupportedSchema,
    EmptyCaseId,
    EmptyHypothesisSet,
    EmptyDecisionSet,
    EmptyHypothesisId,
    DuplicateHypothesisId { id: &'static str },
    EmptyDecisionId,
    DuplicateDecisionId { id: &'static str },
    InvalidPrior { hypothesis_id: &'static str },
    PriorMassNotOne,
    DecisionLossCountMismatch { hypothesis_id: &'static str },
    InvalidDecisionLoss { hypothesis_id: &'static str },
    DuplicateMeasurementId { id: &'static str },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeasurementBlock {
    EmptyMeasurementId,
    DuplicateMeasurementId { id: &'static str },
    EmptyEvidenceOrigin,
    InvalidCost,
    LikelihoodModelUnknown,
    LikelihoodCountMismatch,
    InvalidLikelihood,
    CalibrationNotCurrent,
    ApplicabilityNotCurrent,
    AuthorityNotCurrent,
    CorrelatedEvidenceOrigin,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionEstimate {
    pub decision_id: &'static str,
    pub expected_loss: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementDisposition {
    Eligible,
    Blocked,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementEvaluation {
    pub measurement_id: &'static str,
    pub disposition: MeasurementDisposition,
    pub block_reasons: Vec<MeasurementBlock>,
    pub probability_positive: Option<f64>,
    pub information_gain_bits: Option<f64>,
    pub expected_posterior_loss: Option<f64>,
    pub measurement_cost: Option<f64>,
    pub gross_value_of_information: Option<f64>,
    pub net_value_of_information: Option<f64>,
    pub positive_decision: Option<DecisionEstimate>,
    pub negative_decision: Option<DecisionEstimate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicySelection {
    pub policy_id: &'static str,
    pub measurement_id: Option<&'static str>,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosisBenchmark {
    pub baseline_decision: DecisionEstimate,
    pub measurements: Vec<MeasurementEvaluation>,
    pub information_gain_choice: PolicySelection,
    pub decision_value_choice: PolicySelection,
}

fn validate_case(case: &DecisionCase) -> Vec<CaseInputFailure> {
    let mut failures = Vec::new();
    if case.schema_version != "industrial-diagnosis-case/0.1" {
        failures.push(CaseInputFailure::UnsupportedSchema);
    }
    if case.case_id.trim().is_empty() {
        failures.push(CaseInputFailure::EmptyCaseId);
    }
    if case.hypotheses.is_empty() {
        failures.push(CaseInputFailure::EmptyHypothesisSet);
    }
    if case.decisions.is_empty() {
        failures.push(CaseInputFailure::EmptyDecisionSet);
    }

    let mut hypothesis_ids = BTreeSet::new();
    let mut prior_sum = 0.0;
    for hypothesis in &case.hypotheses {
        if hypothesis.id.trim().is_empty() {
            failures.push(CaseInputFailure::EmptyHypothesisId);
        } else if !hypothesis_ids.insert(hypothesis.id) {
            failures.push(CaseInputFailure::DuplicateHypothesisId { id: hypothesis.id });
        }
        if !hypothesis.prior.is_finite() || !(0.0..=1.0).contains(&hypothesis.prior) {
            failures.push(CaseInputFailure::InvalidPrior {
                hypothesis_id: hypothesis.id,
            });
        } else {
            prior_sum += hypothesis.prior;
        }
        if hypothesis.decision_losses.len() != case.decisions.len() {
            failures.push(CaseInputFailure::DecisionLossCountMismatch {
                hypothesis_id: hypothesis.id,
            });
        }
        if hypothesis
            .decision_losses
            .iter()
            .any(|loss| !loss.is_finite() || *loss < 0.0)
        {
            failures.push(CaseInputFailure::InvalidDecisionLoss {
                hypothesis_id: hypothesis.id,
            });
        }
    }
    if !case.hypotheses.is_empty() && (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        failures.push(CaseInputFailure::PriorMassNotOne);
    }

    let mut decision_ids = BTreeSet::new();
    for decision in &case.decisions {
        if decision.trim().is_empty() {
            failures.push(CaseInputFailure::EmptyDecisionId);
        } else if !decision_ids.insert(decision) {
            failures.push(CaseInputFailure::DuplicateDecisionId { id: decision });
        }
    }

    failures.sort_by_key(|failure| format!("{failure:?}"));
    failures.dedup();
    failures
}

fn validate_measurement(
    case: &DecisionCase,
    measurement: &Measurement,
    existing_ids: &BTreeSet<&'static str>,
) -> Vec<MeasurementBlock> {
    let mut reasons = Vec::new();
    if measurement.id.trim().is_empty() {
        reasons.push(MeasurementBlock::EmptyMeasurementId);
    }
    if measurement.evidence_origin.trim().is_empty() {
        reasons.push(MeasurementBlock::EmptyEvidenceOrigin);
    }
    if !measurement.cost_units.is_finite() || measurement.cost_units < 0.0 {
        reasons.push(MeasurementBlock::InvalidCost);
    }
    match &measurement.positive_likelihoods {
        None => reasons.push(MeasurementBlock::LikelihoodModelUnknown),
        Some(likelihoods) => {
            if likelihoods.len() != case.hypotheses.len() {
                reasons.push(MeasurementBlock::LikelihoodCountMismatch);
            }
            if likelihoods
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            {
                reasons.push(MeasurementBlock::InvalidLikelihood);
            }
        }
    }
    if measurement.calibration != EvidenceStatus::Current {
        reasons.push(MeasurementBlock::CalibrationNotCurrent);
    }
    if measurement.applicability != EvidenceStatus::Current {
        reasons.push(MeasurementBlock::ApplicabilityNotCurrent);
    }
    if measurement.authority != EvidenceStatus::Current {
        reasons.push(MeasurementBlock::AuthorityNotCurrent);
    }
    if existing_ids.contains(measurement.evidence_origin) {
        reasons.push(MeasurementBlock::CorrelatedEvidenceOrigin);
    }
    reasons.sort_by_key(|reason| format!("{reason:?}"));
    reasons.dedup();
    reasons
}

fn entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.log2())
        .sum()
}

fn minimum_expected_loss(case: &DecisionCase, posterior: &[f64]) -> DecisionEstimate {
    let mut best = DecisionEstimate {
        decision_id: case.decisions[0],
        expected_loss: f64::INFINITY,
    };
    for (decision_index, decision_id) in case.decisions.iter().enumerate() {
        let expected_loss: f64 = case
            .hypotheses
            .iter()
            .zip(posterior)
            .map(|(hypothesis, probability)| {
                probability * hypothesis.decision_losses[decision_index]
            })
            .sum();
        if expected_loss < best.expected_loss - METRIC_TOLERANCE {
            best = DecisionEstimate {
                decision_id,
                expected_loss,
            };
        }
    }
    best
}

fn posterior(
    prior: &[f64],
    likelihoods: &[f64],
    positive: bool,
    probability_outcome: f64,
) -> Option<Vec<f64>> {
    if probability_outcome <= 0.0 {
        return None;
    }
    let values: Vec<f64> = prior
        .iter()
        .zip(likelihoods)
        .map(|(p, likelihood)| {
            let outcome_probability = if positive {
                *likelihood
            } else {
                1.0 - *likelihood
            };
            p * outcome_probability / probability_outcome
        })
        .collect();
    Some(values)
}

/// Compute exact finite two-outcome posterior probabilities for one measurement.
/// "Exact" means complete enumeration of the supplied hypothesis set and both
/// outcomes; probabilities/losses are represented in f64, not symbolic rationals.
pub fn evaluate_measurement(
    case: &DecisionCase,
    measurement: &Measurement,
) -> MeasurementEvaluation {
    let existing_ids: BTreeSet<&'static str> =
        case.existing_evidence_origins.iter().copied().collect();
    let block_reasons = validate_measurement(case, measurement, &existing_ids);
    if !block_reasons.is_empty() {
        return MeasurementEvaluation {
            measurement_id: measurement.id,
            disposition: MeasurementDisposition::Blocked,
            block_reasons,
            probability_positive: None,
            information_gain_bits: None,
            expected_posterior_loss: None,
            measurement_cost: None,
            gross_value_of_information: None,
            net_value_of_information: None,
            positive_decision: None,
            negative_decision: None,
        };
    }

    let prior: Vec<f64> = case
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior)
        .collect();
    let likelihoods = measurement
        .positive_likelihoods
        .as_ref()
        .expect("validated likelihood model");
    let probability_positive: f64 = prior
        .iter()
        .zip(likelihoods)
        .map(|(p, likelihood)| p * likelihood)
        .sum();
    let probability_negative = (1.0 - probability_positive).max(0.0);

    let positive_posterior = posterior(&prior, likelihoods, true, probability_positive);
    let negative_posterior = posterior(&prior, likelihoods, false, probability_negative);
    let positive_decision = positive_posterior
        .as_ref()
        .map(|p| minimum_expected_loss(case, p));
    let negative_decision = negative_posterior
        .as_ref()
        .map(|p| minimum_expected_loss(case, p));

    let baseline_decision = minimum_expected_loss(case, &prior);
    let expected_posterior_loss = probability_positive
        * positive_decision.as_ref().map_or(0.0, |d| d.expected_loss)
        + probability_negative * negative_decision.as_ref().map_or(0.0, |d| d.expected_loss);
    let expected_posterior_entropy = probability_positive
        * positive_posterior.as_ref().map_or(0.0, |p| entropy_bits(p))
        + probability_negative * negative_posterior.as_ref().map_or(0.0, |p| entropy_bits(p));
    let information_gain_bits = (entropy_bits(&prior) - expected_posterior_entropy).max(0.0);
    let gross_value_of_information =
        (baseline_decision.expected_loss - expected_posterior_loss).max(0.0);
    let net_value_of_information =
        baseline_decision.expected_loss - expected_posterior_loss - measurement.cost_units;

    MeasurementEvaluation {
        measurement_id: measurement.id,
        disposition: MeasurementDisposition::Eligible,
        block_reasons: Vec::new(),
        probability_positive: Some(probability_positive),
        information_gain_bits: Some(information_gain_bits),
        expected_posterior_loss: Some(expected_posterior_loss),
        measurement_cost: Some(measurement.cost_units),
        gross_value_of_information: Some(gross_value_of_information),
        net_value_of_information: Some(net_value_of_information),
        positive_decision,
        negative_decision,
    }
}

fn selection(
    policy_id: &'static str,
    evaluations: &[MeasurementEvaluation],
    score: impl Fn(&MeasurementEvaluation) -> Option<f64>,
    require_positive: bool,
) -> PolicySelection {
    let best = evaluations
        .iter()
        .filter(|evaluation| evaluation.disposition == MeasurementDisposition::Eligible)
        .filter_map(|evaluation| score(evaluation).map(|value| (evaluation, value)))
        .filter(|(_, value)| !require_positive || *value > 0.0)
        .max_by(|(_, a), (_, b)| a.total_cmp(b));
    match best {
        Some((evaluation, value)) => PolicySelection {
            policy_id,
            measurement_id: Some(evaluation.measurement_id),
            score: Some(value),
        },
        None => PolicySelection {
            policy_id,
            measurement_id: None,
            score: None,
        },
    }
}

pub fn evaluate_case(
    case: &DecisionCase,
    measurements: &[Measurement],
) -> Result<DiagnosisBenchmark, Vec<CaseInputFailure>> {
    let failures = validate_case(case);
    if !failures.is_empty() {
        return Err(failures);
    }
    let mut measurement_ids = BTreeSet::new();
    let mut duplicate_failures = Vec::new();
    for measurement in measurements {
        if !measurement.id.trim().is_empty() && !measurement_ids.insert(measurement.id) {
            duplicate_failures
                .push(CaseInputFailure::DuplicateMeasurementId { id: measurement.id });
        }
    }
    if !duplicate_failures.is_empty() {
        return Err(duplicate_failures);
    }

    let prior: Vec<f64> = case
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior)
        .collect();
    let baseline_decision = minimum_expected_loss(case, &prior);
    let evaluations: Vec<MeasurementEvaluation> = measurements
        .iter()
        .map(|measurement| evaluate_measurement(case, measurement))
        .collect();
    let information_gain_choice = selection(
        "MAX_EXPECTED_INFORMATION_GAIN",
        &evaluations,
        |evaluation| evaluation.information_gain_bits,
        false,
    );
    let decision_value_choice = selection(
        "MAX_NET_VALUE_OF_INFORMATION",
        &evaluations,
        |evaluation| evaluation.net_value_of_information,
        true,
    );

    Ok(DiagnosisBenchmark {
        baseline_decision,
        measurements: evaluations,
        information_gain_choice,
        decision_value_choice,
    })
}

pub fn reference_case() -> DecisionCase {
    DecisionCase {
        schema_version: "industrial-diagnosis-case/0.1",
        case_id: "PUMP-VALVE-VOI-SYNTHETIC-001",
        hypotheses: vec![
            Hypothesis {
                id: "DANGEROUS_PROCESS_DEGRADATION",
                prior: 0.10,
                decision_losses: vec![100.0, 12.0, 60.0],
            },
            Hypothesis {
                id: "METROLOGY_CALIBRATION_DRIFT",
                prior: 0.30,
                decision_losses: vec![20.0, 12.0, 2.0],
            },
            Hypothesis {
                id: "NORMAL_OR_INNOCUOUS_VARIATION",
                prior: 0.60,
                decision_losses: vec![0.0, 12.0, 8.0],
            },
        ],
        decisions: vec!["CONTINUE", "STOP_AND_QUARANTINE", "RECALIBRATE_AND_RETEST"],
        existing_evidence_origins: vec!["LIVE-PRESSURE-TAG-07"],
    }
}

pub fn reference_measurements() -> Vec<Measurement> {
    let current = EvidenceStatus::Current;
    vec![
        Measurement {
            id: "CALIBRATION_REFERENCE_ARTIFACT",
            cost_units: 1.0,
            positive_likelihoods: Some(vec![0.10, 0.95, 0.05]),
            calibration: current,
            applicability: current,
            authority: current,
            evidence_origin: "INDEPENDENT-CAL-ARTIFACT",
        },
        Measurement {
            id: "INDEPENDENT_PROCESS_SAFETY_TEST",
            cost_units: 2.0,
            positive_likelihoods: Some(vec![0.90, 0.15, 0.10]),
            calibration: current,
            applicability: current,
            authority: current,
            evidence_origin: "INDEPENDENT-PROCESS-TEST",
        },
        Measurement {
            id: "GENERAL_VIBRATION_SCREEN",
            cost_units: 0.5,
            positive_likelihoods: Some(vec![0.55, 0.65, 0.35]),
            calibration: current,
            applicability: current,
            authority: current,
            evidence_origin: "INDEPENDENT-VIBRATION-SENSOR",
        },
    ]
}

pub fn hostile_measurements() -> Vec<Measurement> {
    let mut stale = reference_measurements()[0].clone();
    stale.id = "HOSTILE_STALE_CALIBRATION";
    stale.calibration = EvidenceStatus::Stale;

    let mut correlated = reference_measurements()[1].clone();
    correlated.id = "HOSTILE_DERIVED_DASHBOARD";
    correlated.evidence_origin = "LIVE-PRESSURE-TAG-07";

    let mut unknown_applicability = reference_measurements()[1].clone();
    unknown_applicability.id = "HOSTILE_OUT_OF_ENVELOPE";
    unknown_applicability.applicability = EvidenceStatus::Unknown;

    let mut unknown_likelihood = reference_measurements()[1].clone();
    unknown_likelihood.id = "HOSTILE_UNKNOWN_SENSITIVITY";
    unknown_likelihood.positive_likelihoods = None;

    let mut missing_authority = reference_measurements()[0].clone();
    missing_authority.id = "HOSTILE_NO_INSPECTOR_AUTHORITY";
    missing_authority.authority = EvidenceStatus::Unknown;

    vec![
        stale,
        correlated,
        unknown_applicability,
        unknown_likelihood,
        missing_authority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn baseline_recalibration_decision_has_expected_loss_11_4() {
        let case = reference_case();
        let benchmark = evaluate_case(&case, &reference_measurements()).unwrap();
        assert_eq!(
            benchmark.baseline_decision.decision_id,
            "RECALIBRATE_AND_RETEST"
        );
        close(benchmark.baseline_decision.expected_loss, 11.4);
    }

    #[test]
    fn information_gain_prefers_calibration_artifact() {
        let benchmark = evaluate_case(&reference_case(), &reference_measurements()).unwrap();
        assert_eq!(
            benchmark.information_gain_choice.measurement_id,
            Some("CALIBRATION_REFERENCE_ARTIFACT")
        );
        close(benchmark.information_gain_choice.score.unwrap(), 0.6050793);
    }

    #[test]
    fn decision_value_prefers_process_safety_test() {
        let benchmark = evaluate_case(&reference_case(), &reference_measurements()).unwrap();
        assert_eq!(
            benchmark.decision_value_choice.measurement_id,
            Some("INDEPENDENT_PROCESS_SAFETY_TEST")
        );
        close(benchmark.decision_value_choice.score.unwrap(), 1.63);
    }

    #[test]
    fn most_informative_test_is_not_the_highest_decision_value_test() {
        let benchmark = evaluate_case(&reference_case(), &reference_measurements()).unwrap();
        assert_ne!(
            benchmark.information_gain_choice.measurement_id,
            benchmark.decision_value_choice.measurement_id
        );
    }

    #[test]
    fn low_value_test_is_rejected_economically() {
        let benchmark = evaluate_case(&reference_case(), &reference_measurements()).unwrap();
        let general = benchmark
            .measurements
            .iter()
            .find(|item| item.measurement_id == "GENERAL_VIBRATION_SCREEN")
            .unwrap();
        close(general.net_value_of_information.unwrap(), -0.5);
        assert_eq!(
            benchmark.decision_value_choice.measurement_id,
            Some("INDEPENDENT_PROCESS_SAFETY_TEST")
        );
    }

    #[test]
    fn posterior_actions_change_after_test_outcomes() {
        let benchmark = evaluate_case(&reference_case(), &reference_measurements()).unwrap();
        let process = benchmark
            .measurements
            .iter()
            .find(|item| item.measurement_id == "INDEPENDENT_PROCESS_SAFETY_TEST")
            .unwrap();
        assert_eq!(
            process.positive_decision.as_ref().unwrap().decision_id,
            "STOP_AND_QUARANTINE"
        );
        assert_eq!(
            process.negative_decision.as_ref().unwrap().decision_id,
            "RECALIBRATE_AND_RETEST"
        );
    }

    #[test]
    fn stale_calibration_blocks_a_measurement() {
        let case = reference_case();
        let measurement = hostile_measurements().remove(0);
        let result = evaluate_measurement(&case, &measurement);
        assert_eq!(result.disposition, MeasurementDisposition::Blocked);
        assert!(
            result
                .block_reasons
                .contains(&MeasurementBlock::CalibrationNotCurrent)
        );
        assert!(result.net_value_of_information.is_none());
    }

    #[test]
    fn correlated_dashboard_is_not_counted_as_independent_evidence() {
        let case = reference_case();
        let measurement = hostile_measurements().remove(1);
        let result = evaluate_measurement(&case, &measurement);
        assert_eq!(result.disposition, MeasurementDisposition::Blocked);
        assert!(
            result
                .block_reasons
                .contains(&MeasurementBlock::CorrelatedEvidenceOrigin)
        );
    }

    #[test]
    fn out_of_envelope_measurement_is_blocked() {
        let case = reference_case();
        let measurement = hostile_measurements().remove(2);
        let result = evaluate_measurement(&case, &measurement);
        assert_eq!(result.disposition, MeasurementDisposition::Blocked);
        assert!(
            result
                .block_reasons
                .contains(&MeasurementBlock::ApplicabilityNotCurrent)
        );
    }

    #[test]
    fn unknown_likelihood_does_not_get_an_evsi_score() {
        let case = reference_case();
        let measurement = hostile_measurements().remove(3);
        let result = evaluate_measurement(&case, &measurement);
        assert_eq!(result.disposition, MeasurementDisposition::Blocked);
        assert!(
            result
                .block_reasons
                .contains(&MeasurementBlock::LikelihoodModelUnknown)
        );
        assert!(result.information_gain_bits.is_none());
    }

    #[test]
    fn missing_inspection_authority_blocks_measurement() {
        let case = reference_case();
        let measurement = hostile_measurements().remove(4);
        let result = evaluate_measurement(&case, &measurement);
        assert_eq!(result.disposition, MeasurementDisposition::Blocked);
        assert!(
            result
                .block_reasons
                .contains(&MeasurementBlock::AuthorityNotCurrent)
        );
    }

    #[test]
    fn duplicate_hypothesis_ids_invalidate_the_case() {
        let mut case = reference_case();
        case.hypotheses[1].id = case.hypotheses[0].id;
        assert!(evaluate_case(&case, &reference_measurements()).is_err());
    }

    #[test]
    fn prior_probabilities_must_sum_to_one() {
        let mut case = reference_case();
        case.hypotheses[0].prior += 0.1;
        assert!(evaluate_case(&case, &reference_measurements()).is_err());
    }

    #[test]
    fn duplicate_measurement_ids_are_rejected_as_duplicate_ids() {
        let mut measurements = reference_measurements();
        measurements[1].id = measurements[0].id;
        let result = evaluate_case(&reference_case(), &measurements);
        assert!(matches!(
            result,
            Err(failures) if failures.contains(&CaseInputFailure::DuplicateMeasurementId {
                id: "CALIBRATION_REFERENCE_ARTIFACT"
            })
        ));
    }

    #[test]
    fn measurement_likelihoods_must_match_hypothesis_count() {
        let case = reference_case();
        let mut measurements = reference_measurements();
        measurements[0].positive_likelihoods = Some(vec![0.5]);
        let result = evaluate_measurement(&case, &measurements[0]);
        assert_eq!(result.disposition, MeasurementDisposition::Blocked);
        assert!(
            result
                .block_reasons
                .contains(&MeasurementBlock::LikelihoodCountMismatch)
        );
    }

    #[test]
    fn no_positive_net_voi_returns_no_test_selection() {
        let case = reference_case();
        let mut measurement = reference_measurements()[2].clone();
        measurement.cost_units = 100.0;
        let choice = selection(
            "MAX_NET_VALUE_OF_INFORMATION",
            &[evaluate_measurement(&case, &measurement)],
            |evaluation| evaluation.net_value_of_information,
            true,
        );
        assert_eq!(choice.measurement_id, None);
    }

    #[test]
    fn repeated_evaluation_is_deterministic() {
        let first = evaluate_case(&reference_case(), &reference_measurements()).unwrap();
        let second = evaluate_case(&reference_case(), &reference_measurements()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn hostile_measurement_suite_is_complete_and_blocked() {
        let case = reference_case();
        let hostile = hostile_measurements();
        assert_eq!(hostile.len(), 5);
        assert!(hostile.iter().all(|measurement| {
            evaluate_measurement(&case, measurement).disposition == MeasurementDisposition::Blocked
        }));
    }
}
