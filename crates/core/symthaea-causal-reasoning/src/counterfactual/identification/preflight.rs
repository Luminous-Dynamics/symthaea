// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Structured preflight checks for causal-effect estimation.
//!
//! The legacy estimators predate evidence-bearing failure semantics and use
//! numeric zero sentinels in several degenerate cases. This module does not
//! change those APIs. Instead it provides an explicit readiness boundary so
//! callers can distinguish a defensible estimate request from insufficient or
//! malformed data before invoking a numerical estimator.

use super::{
    CausalDAG, CausalQuery, CausalQueryOutcome, CounterfactualReasoner, IdentificationMethod,
    ObservationalData, UnidentifiedReason,
};

const VARIANCE_EPSILON: f64 = 1e-10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimationFamily {
    /// Existing identification-selected estimator path.
    IdentifiedDefault,
    /// Regression/backdoor-style estimator.
    Regression,
    /// Binary-treatment inverse-probability weighting.
    InverseProbabilityWeighting,
    /// Binary-treatment doubly robust estimator.
    DoublyRobust,
    /// Existing multi-estimator robust comparison path.
    RobustComparison,
}

impl EstimationFamily {
    pub const fn minimum_samples(self) -> usize {
        match self {
            Self::IdentifiedDefault | Self::Regression => 2,
            Self::InverseProbabilityWeighting | Self::DoublyRobust | Self::RobustComparison => 10,
        }
    }

    pub const fn requires_binary_treatment(self) -> bool {
        matches!(
            self,
            Self::InverseProbabilityWeighting | Self::DoublyRobust | Self::RobustComparison
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EstimationPreflightFailure {
    QueryIndexOutOfBounds { index: usize, node_count: usize },
    VariableCountMismatch { dag_nodes: usize, data_variables: usize },
    RowWidthMismatch { row: usize, expected: usize, got: usize },
    NonFiniteData { row: usize, column: usize, value: f64 },
    InsufficientSamples { found: usize, required: usize },
    DegenerateTreatmentVariance { variance: f64 },
    TreatmentNotBinary { row: usize, value: f64 },
    MissingTreatmentArm { treated: usize, control: usize },
    Unidentified {
        reason: UnidentifiedReason,
        missing: Vec<String>,
        suggestions: Vec<String>,
    },
    AssumptionRequired { condition: String, testability: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum EstimationPreflightWarning {
    DegenerateAdjustmentVariable {
        node: usize,
        name: String,
        variance: f64,
    },
    ConstantOutcome {
        variance: f64,
    },
    LegacyZeroSentinelApi,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EstimationPreflightReport {
    pub family: EstimationFamily,
    pub sample_count: usize,
    pub treatment_variance: f64,
    pub outcome_variance: f64,
    pub identification_method: IdentificationMethod,
    pub identification_confidence: f64,
    pub adjustment_set: Vec<usize>,
    pub warnings: Vec<EstimationPreflightWarning>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EstimationPreflightOutcome {
    Ready(EstimationPreflightReport),
    NotReady(EstimationPreflightFailure),
}

impl EstimationPreflightOutcome {
    pub const fn is_ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }
}

fn invalid_query_index(query: &CausalQuery, node_count: usize) -> Option<usize> {
    std::iter::once(query.treatment)
        .chain(std::iter::once(query.outcome))
        .chain(query.conditioning.iter().copied())
        .find(|&index| index >= node_count)
}

fn validate_shape_and_finiteness(
    dag: &CausalDAG,
    data: &ObservationalData,
) -> Option<EstimationPreflightFailure> {
    if data.variables.len() != dag.num_nodes() {
        return Some(EstimationPreflightFailure::VariableCountMismatch {
            dag_nodes: dag.num_nodes(),
            data_variables: data.variables.len(),
        });
    }

    for (row_index, row) in data.observations.iter().enumerate() {
        if row.len() != data.variables.len() {
            return Some(EstimationPreflightFailure::RowWidthMismatch {
                row: row_index,
                expected: data.variables.len(),
                got: row.len(),
            });
        }
        for (column_index, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Some(EstimationPreflightFailure::NonFiniteData {
                    row: row_index,
                    column: column_index,
                    value,
                });
            }
        }
    }

    None
}

fn validate_binary_treatment(
    query: &CausalQuery,
    data: &ObservationalData,
) -> Option<EstimationPreflightFailure> {
    let mut treated = 0usize;
    let mut control = 0usize;

    for (row_index, row) in data.observations.iter().enumerate() {
        let value = row[query.treatment];
        if (value - 1.0).abs() <= 1e-12 {
            treated += 1;
        } else if value.abs() <= 1e-12 {
            control += 1;
        } else {
            return Some(EstimationPreflightFailure::TreatmentNotBinary {
                row: row_index,
                value,
            });
        }
    }

    if treated == 0 || control == 0 {
        return Some(EstimationPreflightFailure::MissingTreatmentArm { treated, control });
    }

    None
}

/// Validate whether an observational causal-effect request is ready for a
/// numerical estimator without conflating failure with an estimated zero.
///
/// This is intentionally a preflight, not a numerical estimator. A `Ready`
/// result means the obvious structural/data preconditions passed; it does not
/// guarantee the downstream estimator is statistically well specified.
pub fn preflight_estimation(
    dag: &CausalDAG,
    query: &CausalQuery,
    data: &ObservationalData,
    family: EstimationFamily,
) -> EstimationPreflightOutcome {
    let node_count = dag.num_nodes();
    if let Some(index) = invalid_query_index(query, node_count) {
        return EstimationPreflightOutcome::NotReady(
            EstimationPreflightFailure::QueryIndexOutOfBounds { index, node_count },
        );
    }

    if let Some(failure) = validate_shape_and_finiteness(dag, data) {
        return EstimationPreflightOutcome::NotReady(failure);
    }

    let required = family.minimum_samples();
    if data.n() < required {
        return EstimationPreflightOutcome::NotReady(
            EstimationPreflightFailure::InsufficientSamples {
                found: data.n(),
                required,
            },
        );
    }

    let identification = CounterfactualReasoner::new().query(dag, query);
    let (identification_method, identification_confidence, adjustment_set) = match identification {
        CausalQueryOutcome::Identified {
            estimand,
            method,
            confidence,
        } => (method, confidence, estimand.adjustment_set),
        CausalQueryOutcome::Unidentified {
            reason,
            missing,
            suggestions,
        } => {
            return EstimationPreflightOutcome::NotReady(
                EstimationPreflightFailure::Unidentified {
                    reason,
                    missing,
                    suggestions,
                },
            );
        }
        CausalQueryOutcome::AssumptionRequired {
            assumption,
            plausibility: _,
            estimand_if_assumed: _,
        } => {
            return EstimationPreflightOutcome::NotReady(
                EstimationPreflightFailure::AssumptionRequired {
                    condition: assumption.condition,
                    testability: assumption.testability,
                },
            );
        }
    };

    let treatment_variance = data.variance(query.treatment);
    if !treatment_variance.is_finite() || treatment_variance.abs() < VARIANCE_EPSILON {
        return EstimationPreflightOutcome::NotReady(
            EstimationPreflightFailure::DegenerateTreatmentVariance {
                variance: treatment_variance,
            },
        );
    }

    if family.requires_binary_treatment() {
        if let Some(failure) = validate_binary_treatment(query, data) {
            return EstimationPreflightOutcome::NotReady(failure);
        }
    }

    let outcome_variance = data.variance(query.outcome);
    let mut warnings = vec![EstimationPreflightWarning::LegacyZeroSentinelApi];
    if outcome_variance.abs() < VARIANCE_EPSILON {
        warnings.push(EstimationPreflightWarning::ConstantOutcome {
            variance: outcome_variance,
        });
    }

    for &node in &adjustment_set {
        let variance = data.variance(node);
        if variance.abs() < VARIANCE_EPSILON {
            warnings.push(EstimationPreflightWarning::DegenerateAdjustmentVariable {
                node,
                name: data.variables[node].clone(),
                variance,
            });
        }
    }

    EstimationPreflightOutcome::Ready(EstimationPreflightReport {
        family,
        sample_count: data.n(),
        treatment_variance,
        outcome_variance,
        identification_method,
        identification_confidence,
        adjustment_set,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_dag() -> CausalDAG {
        CausalDAG::new(vec!["X".into(), "Y".into()], vec![(0, 1)])
    }

    fn query() -> CausalQuery {
        CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        }
    }

    #[test]
    fn insufficient_samples_are_not_a_zero_effect() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        data.add_observation(vec![0.0, 1.0]);

        assert_eq!(
            preflight_estimation(&simple_dag(), &query(), &data, EstimationFamily::Regression),
            EstimationPreflightOutcome::NotReady(
                EstimationPreflightFailure::InsufficientSamples {
                    found: 1,
                    required: 2,
                }
            )
        );
    }

    #[test]
    fn degenerate_treatment_variance_is_not_a_zero_effect() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for y in [1.0, 2.0, 3.0] {
            data.add_observation(vec![1.0, y]);
        }

        assert!(matches!(
            preflight_estimation(&simple_dag(), &query(), &data, EstimationFamily::Regression),
            EstimationPreflightOutcome::NotReady(
                EstimationPreflightFailure::DegenerateTreatmentVariance { .. }
            )
        ));
    }

    #[test]
    fn ipw_requires_both_binary_treatment_arms() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..10 {
            data.add_observation(vec![1.0, i as f64]);
        }

        assert!(matches!(
            preflight_estimation(
                &simple_dag(),
                &query(),
                &data,
                EstimationFamily::InverseProbabilityWeighting,
            ),
            EstimationPreflightOutcome::NotReady(
                EstimationPreflightFailure::DegenerateTreatmentVariance { .. }
            )
        ));
    }

    #[test]
    fn ipw_rejects_non_binary_treatment() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..10 {
            let x = if i == 4 { 0.4 } else if i % 2 == 0 { 0.0 } else { 1.0 };
            data.add_observation(vec![x, i as f64]);
        }

        assert!(matches!(
            preflight_estimation(
                &simple_dag(),
                &query(),
                &data,
                EstimationFamily::InverseProbabilityWeighting,
            ),
            EstimationPreflightOutcome::NotReady(
                EstimationPreflightFailure::TreatmentNotBinary { row: 4, .. }
            )
        ));
    }

    #[test]
    fn ready_report_keeps_identification_separate_from_estimation() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..20 {
            let x = if i % 2 == 0 { 0.0 } else { 1.0 };
            data.add_observation(vec![x, 3.0 * x + i as f64 * 0.01]);
        }

        let outcome = preflight_estimation(
            &simple_dag(),
            &query(),
            &data,
            EstimationFamily::DoublyRobust,
        );
        let EstimationPreflightOutcome::Ready(report) = outcome else {
            panic!("expected ready preflight");
        };

        assert_eq!(report.sample_count, 20);
        assert!(report.identification_confidence.is_finite());
        assert!(report.treatment_variance > 0.0);
        assert!(report
            .warnings
            .contains(&EstimationPreflightWarning::LegacyZeroSentinelApi));
    }

    #[test]
    fn malformed_public_observation_rows_are_rejected_before_indexing() {
        let data = ObservationalData {
            variables: vec!["X".into(), "Y".into()],
            observations: vec![vec![0.0]],
        };

        assert_eq!(
            preflight_estimation(&simple_dag(), &query(), &data, EstimationFamily::Regression),
            EstimationPreflightOutcome::NotReady(
                EstimationPreflightFailure::RowWidthMismatch {
                    row: 0,
                    expected: 2,
                    got: 1,
                }
            )
        );
    }
}
