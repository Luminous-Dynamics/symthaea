//! Selective decision metrics: risk, coverage and explicit fallback loss.
//!
//! These are descriptive finite-corpus metrics. They do not construct
//! confidence intervals, prove calibration, or certify safety under shift.
//! Cases must be independent evaluation units or must be aggregated by the
//! caller at the appropriate episode/trajectory level.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

/// One evaluator-owned outcome row for a selective decision policy.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectiveCase {
    /// Candidate-provided acceptance/ranking score in [0, 1].
    /// It need not be a calibrated probability.
    pub acceptance_score: f64,
    /// Realized loss if the candidate is allowed to execute its action.
    pub decision_loss: f64,
    /// Realized loss of the predeclared fallback if the candidate abstains.
    pub fallback_loss: f64,
    /// Whether executing the candidate action would expose this case to a
    /// critical event. Reported among accepted cases, not as a candidate-
    /// authored prediction.
    pub critical_event: bool,
    /// Evaluator-owned environment/group label for stratified reporting.
    pub environment: String,
}

/// Invalid input or numerical outcomes for selective-decision metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectiveError {
    EmptyCases,
    InvalidAcceptanceScore,
    InvalidThreshold,
    InvalidLoss(&'static str),
    EmptyEnvironment,
    NumericalFailure,
}

impl fmt::Display for SelectiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCases => write!(formatter, "selective evaluation requires at least one case"),
            Self::InvalidAcceptanceScore => {
                write!(formatter, "acceptance score must be finite and in [0, 1]")
            }
            Self::InvalidThreshold => {
                write!(formatter, "acceptance threshold must be finite and in [0, 1]")
            }
            Self::InvalidLoss(name) => {
                write!(formatter, "{name} must be finite and non-negative")
            }
            Self::EmptyEnvironment => write!(formatter, "environment label must not be empty"),
            Self::NumericalFailure => {
                write!(formatter, "selective metric accumulation produced a non-finite value")
            }
        }
    }
}

impl Error for SelectiveError {}

/// Descriptive metrics for one environment at one fixed threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentSelectiveMetrics {
    pub environment: String,
    pub total_cases: usize,
    pub accepted_cases: usize,
    pub abstained_cases: usize,
    pub coverage: f64,
    /// Undefined when this environment has no accepted cases.
    pub selective_risk: Option<f64>,
    /// Fraction of accepted cases whose candidate action would encounter a
    /// critical event. Undefined when this environment has no accepted cases.
    pub accepted_critical_event_rate: Option<f64>,
    /// Mean realized policy loss including the declared fallback on abstained
    /// cases, in the same units as decision_loss and fallback_loss.
    pub system_loss: f64,
}

/// Full selective-policy outcome at one threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectiveMetrics {
    pub total_cases: usize,
    pub accepted_cases: usize,
    pub abstained_cases: usize,
    pub coverage: f64,
    pub abstention_fraction: f64,
    /// Mean action loss conditional on the candidate being allowed to act.
    /// Undefined at zero coverage; never represented as zero.
    pub selective_risk: Option<f64>,
    /// Critical-event exposure conditional on the candidate being allowed to act.
    pub accepted_critical_event_rate: Option<f64>,
    /// Mean fallback loss among abstained cases; undefined if none abstained.
    pub mean_fallback_loss: Option<f64>,
    /// Mean realized loss for the deployed policy, including fallback loss.
    pub system_loss: f64,
    /// Maximum system loss over evaluator-declared environments.
    pub worst_environment_system_loss: f64,
    /// Maximum conditional action loss over environments with accepted cases.
    pub worst_environment_selective_risk: Option<f64>,
    /// Minimum coverage over evaluator-declared environments.
    pub minimum_environment_coverage: f64,
    pub environments: Vec<EnvironmentSelectiveMetrics>,
}

/// One point in the threshold risk-coverage curve.
///
/// threshold=None is the all-abstain endpoint. For a numeric threshold, every
/// case whose score is greater than or equal to that value is accepted.
/// Equal scores are always admitted or rejected together.
#[derive(Clone, Debug, PartialEq)]
pub struct RiskCoveragePoint {
    pub threshold: Option<f64>,
    pub metrics: SelectiveMetrics,
}

#[derive(Clone, Debug, Default)]
struct EnvironmentAccumulator {
    total: usize,
    accepted: usize,
    critical: usize,
    action_loss: f64,
    fallback_loss: f64,
    system_loss: f64,
}

fn validate_cases(cases: &[SelectiveCase]) -> Result<(), SelectiveError> {
    if cases.is_empty() {
        return Err(SelectiveError::EmptyCases);
    }
    for case in cases {
        if !case.acceptance_score.is_finite()
            || !(0.0..=1.0).contains(&case.acceptance_score)
        {
            return Err(SelectiveError::InvalidAcceptanceScore);
        }
        if !case.decision_loss.is_finite() || case.decision_loss < 0.0 {
            return Err(SelectiveError::InvalidLoss("decision loss"));
        }
        if !case.fallback_loss.is_finite() || case.fallback_loss < 0.0 {
            return Err(SelectiveError::InvalidLoss("fallback loss"));
        }
        if case.environment.trim().is_empty() {
            return Err(SelectiveError::EmptyEnvironment);
        }
    }
    Ok(())
}

fn validate_threshold(threshold: f64) -> Result<(), SelectiveError> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(SelectiveError::InvalidThreshold);
    }
    Ok(())
}

/// Evaluate a fixed selective policy.
///
/// Some(t) accepts cases with acceptance_score >= t. None abstains on every
/// case. All losses are evaluator-provided realized outcomes or declared
/// fallback costs; the acceptance score is only a ranking value.
pub fn evaluate_selective_policy(
    cases: &[SelectiveCase],
    threshold: Option<f64>,
) -> Result<SelectiveMetrics, SelectiveError> {
    validate_cases(cases)?;
    if let Some(value) = threshold {
        validate_threshold(value)?;
    }

    let mut total_accepted = 0usize;
    let mut total_critical = 0usize;
    let mut accepted_action_loss = 0.0;
    let mut abstained_fallback_loss = 0.0;
    let mut total_system_loss = 0.0;
    let mut groups: BTreeMap<String, EnvironmentAccumulator> = BTreeMap::new();

    for case in cases {
        let accepted = threshold.is_some_and(|value| case.acceptance_score >= value);
        let group = groups.entry(case.environment.clone()).or_default();
        group.total += 1;

        if accepted {
            total_accepted += 1;
            group.accepted += 1;
            accepted_action_loss += case.decision_loss;
            group.action_loss += case.decision_loss;
            if case.critical_event {
                total_critical += 1;
                group.critical += 1;
            }
            total_system_loss += case.decision_loss;
            group.system_loss += case.decision_loss;
        } else {
            abstained_fallback_loss += case.fallback_loss;
            group.fallback_loss += case.fallback_loss;
            total_system_loss += case.fallback_loss;
            group.system_loss += case.fallback_loss;
        }
    }

    if !accepted_action_loss.is_finite()
        || !abstained_fallback_loss.is_finite()
        || !total_system_loss.is_finite()
        || groups.values().any(|group| {
            !group.action_loss.is_finite()
                || !group.fallback_loss.is_finite()
                || !group.system_loss.is_finite()
        })
    {
        return Err(SelectiveError::NumericalFailure);
    }

    let total_cases = cases.len();
    let abstained_cases = total_cases - total_accepted;
    let total_as_f64 = total_cases as f64;
    let coverage = total_accepted as f64 / total_as_f64;
    let abstention_fraction = abstained_cases as f64 / total_as_f64;
    let selective_risk = if total_accepted > 0 {
        Some(accepted_action_loss / total_accepted as f64)
    } else {
        None
    };
    let accepted_critical_event_rate = if total_accepted > 0 {
        Some(total_critical as f64 / total_accepted as f64)
    } else {
        None
    };
    let mean_fallback_loss = if abstained_cases > 0 {
        Some(abstained_fallback_loss / abstained_cases as f64)
    } else {
        None
    };
    let system_loss = total_system_loss / total_as_f64;

    let mut environments = Vec::with_capacity(groups.len());
    let mut worst_environment_system_loss = 0.0_f64;
    let mut worst_environment_selective_risk: Option<f64> = None;
    let mut minimum_environment_coverage = 1.0_f64;

    for (environment, accumulator) in groups {
        let group_total = accumulator.total as f64;
        let group_coverage = accumulator.accepted as f64 / group_total;
        let group_selective_risk = if accumulator.accepted > 0 {
            Some(accumulator.action_loss / accumulator.accepted as f64)
        } else {
            None
        };
        let group_critical_rate = if accumulator.accepted > 0 {
            Some(accumulator.critical as f64 / accumulator.accepted as f64)
        } else {
            None
        };
        let group_system_loss = accumulator.system_loss / group_total;

        worst_environment_system_loss = worst_environment_system_loss.max(group_system_loss);
        if let Some(risk) = group_selective_risk {
            worst_environment_selective_risk = Some(
                worst_environment_selective_risk.map_or(risk, |current| current.max(risk)),
            );
        }
        minimum_environment_coverage = minimum_environment_coverage.min(group_coverage);

        environments.push(EnvironmentSelectiveMetrics {
            environment,
            total_cases: accumulator.total,
            accepted_cases: accumulator.accepted,
            abstained_cases: accumulator.total - accumulator.accepted,
            coverage: group_coverage,
            selective_risk: group_selective_risk,
            accepted_critical_event_rate: group_critical_rate,
            system_loss: group_system_loss,
        });
    }

    Ok(SelectiveMetrics {
        total_cases,
        accepted_cases: total_accepted,
        abstained_cases,
        coverage,
        abstention_fraction,
        selective_risk,
        accepted_critical_event_rate,
        mean_fallback_loss,
        system_loss,
        worst_environment_system_loss,
        worst_environment_selective_risk,
        minimum_environment_coverage,
        environments,
    })
}

/// Build the complete threshold curve using every distinct observed score.
///
/// Duplicate scores are never split, and the input row order cannot affect
/// threshold membership. The first point is the all-abstain endpoint, followed
/// by thresholds in descending score order.
pub fn risk_coverage_curve(
    cases: &[SelectiveCase],
) -> Result<Vec<RiskCoveragePoint>, SelectiveError> {
    validate_cases(cases)?;

    let mut thresholds: Vec<f64> = cases.iter().map(|case| case.acceptance_score).collect();
    thresholds.sort_by(f64::total_cmp);
    thresholds.dedup_by(|left, right| *left == *right);
    thresholds.reverse();

    let mut curve = Vec::with_capacity(thresholds.len() + 1);
    curve.push(RiskCoveragePoint {
        threshold: None,
        metrics: evaluate_selective_policy(cases, None)?,
    });

    for threshold in thresholds {
        curve.push(RiskCoveragePoint {
            threshold: Some(threshold),
            metrics: evaluate_selective_policy(cases, Some(threshold))?,
        });
    }
    Ok(curve)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(
        score: f64,
        decision_loss: f64,
        fallback_loss: f64,
        critical_event: bool,
        environment: &str,
    ) -> SelectiveCase {
        SelectiveCase {
            acceptance_score: score,
            decision_loss,
            fallback_loss,
            critical_event,
            environment: environment.to_owned(),
        }
    }

    #[test]
    fn threshold_reports_coverage_selective_risk_and_fallback_loss_separately() {
        let cases = [
            case(0.9, 1.0, 0.2, false, "nominal"),
            case(0.8, 3.0, 0.4, true, "nominal"),
            case(0.2, 10.0, 0.5, true, "nominal"),
            case(0.1, 8.0, 0.3, false, "nominal"),
        ];
        let metrics = evaluate_selective_policy(&cases, Some(0.8)).unwrap();

        assert_eq!(metrics.accepted_cases, 2);
        assert_eq!(metrics.abstained_cases, 2);
        assert_eq!(metrics.coverage, 0.5);
        assert_eq!(metrics.abstention_fraction, 0.5);
        assert_eq!(metrics.selective_risk, Some(2.0));
        assert_eq!(metrics.accepted_critical_event_rate, Some(0.5));
        assert_eq!(metrics.mean_fallback_loss, Some(0.4));
        assert_eq!(metrics.system_loss, 1.2);
    }

    #[test]
    fn all_abstain_has_undefined_selective_risk_and_explicit_system_loss() {
        let cases = [
            case(0.9, 100.0, 2.0, true, "a"),
            case(0.1, 50.0, 4.0, false, "b"),
        ];
        let metrics = evaluate_selective_policy(&cases, None).unwrap();

        assert_eq!(metrics.coverage, 0.0);
        assert_eq!(metrics.selective_risk, None);
        assert_eq!(metrics.accepted_critical_event_rate, None);
        assert_eq!(metrics.mean_fallback_loss, Some(3.0));
        assert_eq!(metrics.system_loss, 3.0);
        assert_eq!(metrics.minimum_environment_coverage, 0.0);
    }

    #[test]
    fn risk_coverage_curve_keeps_equal_scores_atomic() {
        let cases = [
            case(0.9, 1.0, 0.5, false, "a"),
            case(0.7, 2.0, 0.5, false, "a"),
            case(0.7, 9.0, 0.5, true, "a"),
            case(0.2, 10.0, 0.5, true, "b"),
        ];
        let curve = risk_coverage_curve(&cases).unwrap();

        assert_eq!(curve.len(), 4);
        assert_eq!(curve[0].threshold, None);
        assert_eq!(curve[0].metrics.coverage, 0.0);
        assert_eq!(curve[1].threshold, Some(0.9));
        assert_eq!(curve[1].metrics.accepted_cases, 1);
        assert_eq!(curve[2].threshold, Some(0.7));
        assert_eq!(curve[2].metrics.accepted_cases, 3);
        assert_eq!(curve[3].threshold, Some(0.2));
        assert_eq!(curve[3].metrics.accepted_cases, 4);
    }

    #[test]
    fn curve_is_input_order_invariant() {
        let first = vec![
            case(0.9, 1.0, 0.5, false, "a"),
            case(0.7, 2.0, 0.5, false, "b"),
            case(0.2, 10.0, 0.5, true, "b"),
        ];
        let second = vec![first[2].clone(), first[0].clone(), first[1].clone()];
        assert_eq!(
            risk_coverage_curve(&first).unwrap(),
            risk_coverage_curve(&second).unwrap()
        );
    }

    #[test]
    fn worst_environment_loss_is_not_hidden_by_pooled_average() {
        let cases = [
            case(0.9, 1.0, 0.5, false, "nominal"),
            case(0.9, 1.0, 0.5, false, "nominal"),
            case(0.9, 100.0, 0.5, true, "shifted"),
        ];
        let metrics = evaluate_selective_policy(&cases, Some(0.9)).unwrap();

        assert!((metrics.selective_risk.unwrap() - 34.0).abs() < 1e-12);
        assert_eq!(metrics.worst_environment_selective_risk, Some(100.0));
        assert_eq!(metrics.worst_environment_system_loss, 100.0);
        assert_eq!(metrics.minimum_environment_coverage, 1.0);
        assert_eq!(metrics.environments.len(), 2);
        assert_eq!(metrics.environments[1].environment, "shifted");
    }

    #[test]
    fn zero_coverage_in_one_environment_remains_visible() {
        let cases = [
            case(0.9, 1.0, 0.2, false, "nominal"),
            case(0.1, 20.0, 7.0, true, "shifted"),
        ];
        let metrics = evaluate_selective_policy(&cases, Some(0.5)).unwrap();

        let shifted = metrics
            .environments
            .iter()
            .find(|group| group.environment == "shifted")
            .unwrap();
        assert_eq!(shifted.coverage, 0.0);
        assert_eq!(shifted.selective_risk, None);
        assert_eq!(shifted.accepted_critical_event_rate, None);
        assert_eq!(shifted.system_loss, 7.0);
        assert_eq!(metrics.minimum_environment_coverage, 0.0);
    }

    #[test]
    fn rejects_invalid_scores_losses_thresholds_and_environment_labels() {
        assert_eq!(
            evaluate_selective_policy(&[], Some(0.5)),
            Err(SelectiveError::EmptyCases)
        );
        assert_eq!(
            evaluate_selective_policy(&[case(1.1, 1.0, 0.0, false, "a")], Some(0.5)),
            Err(SelectiveError::InvalidAcceptanceScore)
        );
        assert_eq!(
            evaluate_selective_policy(&[case(0.5, f64::NAN, 0.0, false, "a")], Some(0.5)),
            Err(SelectiveError::InvalidLoss("decision loss"))
        );
        assert_eq!(
            evaluate_selective_policy(&[case(0.5, 1.0, -1.0, false, "a")], Some(0.5)),
            Err(SelectiveError::InvalidLoss("fallback loss"))
        );
        assert_eq!(
            evaluate_selective_policy(&[case(0.5, 1.0, 0.0, false, "a")], Some(1.1)),
            Err(SelectiveError::InvalidThreshold)
        );
        assert_eq!(
            evaluate_selective_policy(&[case(0.5, 1.0, 0.0, false, "  ")], Some(0.5)),
            Err(SelectiveError::EmptyEnvironment)
        );
    }

    #[test]
    fn higher_threshold_does_not_assume_monotonic_risk_improvement() {
        let cases = [
            case(0.9, 8.0, 0.0, true, "a"),
            case(0.8, 1.0, 0.0, false, "a"),
            case(0.1, 2.0, 0.0, false, "a"),
        ];
        let curve = risk_coverage_curve(&cases).unwrap();

        assert_eq!(curve[1].metrics.selective_risk, Some(8.0));
        assert_eq!(curve[2].metrics.selective_risk, Some(4.5));
        assert_eq!(curve[3].metrics.selective_risk, Some(11.0 / 3.0));
    }
}
