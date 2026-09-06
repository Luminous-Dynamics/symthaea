// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification-only measurement-uncertainty theorem for Economic Science.
//!
//! A reported point estimate is not automatically the exact realized economic
//! state. This test freezes the distinction before external outcomes can be used
//! for forecast scoring or claim evidence.

use symthaea_economics::{EconomicVariable, StateDomain, UnitId, VariableId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UncertaintySource {
    Sampling,
    MeasurementProcess,
    ModelBased,
    RevisionRange,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MeasurementUncertainty {
    ExactByConstruction {
        basis_id: String,
    },
    StandardError {
        standard_error_atoms: u128,
        source: UncertaintySource,
        method_id: String,
    },
    Interval {
        lower_atoms: i128,
        upper_atoms: i128,
        source: UncertaintySource,
        coverage_bps: Option<u16>,
        method_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum UncertaintyError {
    EmptyText(&'static str),
    UnitMismatch,
    ZeroStandardError,
    InvalidInterval,
    InvalidCoverage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MeasuredValue {
    variable_id: VariableId,
    unit: UnitId,
    point_atoms: i128,
    uncertainty: MeasurementUncertainty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QualifiedMeasuredValue {
    ExactObservation {
        variable_id: VariableId,
        unit: UnitId,
        value_atoms: i128,
        basis_id: String,
    },
    EstimatedObservation {
        variable_id: VariableId,
        unit: UnitId,
        point_atoms: i128,
        uncertainty: MeasurementUncertainty,
    },
}

impl MeasuredValue {
    fn new(
        variable: &EconomicVariable,
        unit: UnitId,
        point_atoms: i128,
        uncertainty: MeasurementUncertainty,
    ) -> Result<Self, UncertaintyError> {
        if &unit != variable.unit() {
            return Err(UncertaintyError::UnitMismatch);
        }
        validate_uncertainty(point_atoms, &uncertainty)?;
        Ok(Self {
            variable_id: variable.id().clone(),
            unit,
            point_atoms,
            uncertainty,
        })
    }

    fn qualify(self) -> QualifiedMeasuredValue {
        match self.uncertainty {
            MeasurementUncertainty::ExactByConstruction { basis_id } => {
                QualifiedMeasuredValue::ExactObservation {
                    variable_id: self.variable_id,
                    unit: self.unit,
                    value_atoms: self.point_atoms,
                    basis_id,
                }
            }
            uncertainty => QualifiedMeasuredValue::EstimatedObservation {
                variable_id: self.variable_id,
                unit: self.unit,
                point_atoms: self.point_atoms,
                uncertainty,
            },
        }
    }
}

fn validate_uncertainty(
    point_atoms: i128,
    uncertainty: &MeasurementUncertainty,
) -> Result<(), UncertaintyError> {
    let require_text = |field: &'static str, value: &str| {
        if value.trim().is_empty() {
            Err(UncertaintyError::EmptyText(field))
        } else {
            Ok(())
        }
    };

    match uncertainty {
        MeasurementUncertainty::ExactByConstruction { basis_id } => {
            require_text("exactness basis id", basis_id)
        }
        MeasurementUncertainty::StandardError {
            standard_error_atoms,
            method_id,
            ..
        } => {
            if *standard_error_atoms == 0 {
                return Err(UncertaintyError::ZeroStandardError);
            }
            require_text("standard-error method id", method_id)
        }
        MeasurementUncertainty::Interval {
            lower_atoms,
            upper_atoms,
            coverage_bps,
            method_id,
            ..
        } => {
            if lower_atoms >= upper_atoms || point_atoms < *lower_atoms || point_atoms > *upper_atoms {
                return Err(UncertaintyError::InvalidInterval);
            }
            if coverage_bps.is_some_and(|coverage| coverage == 0 || coverage >= 10_000) {
                return Err(UncertaintyError::InvalidCoverage);
            }
            require_text("interval method id", method_id)
        }
    }
}

fn unemployment_rate() -> EconomicVariable {
    EconomicVariable::new(
        VariableId::new("labor:unemployment_rate").unwrap(),
        StateDomain::Institutional,
        UnitId::new("basis_points").unwrap(),
        "Unemployment-rate construct encoded in fixed basis-point atoms.",
    )
    .unwrap()
}

#[test]
fn sampling_point_estimate_with_standard_error_remains_estimated() {
    let measured = MeasuredValue::new(
        &unemployment_rate(),
        UnitId::new("basis_points").unwrap(),
        430,
        MeasurementUncertainty::StandardError {
            standard_error_atoms: 12,
            source: UncertaintySource::Sampling,
            method_id: "variance:survey-linearization-v1".into(),
        },
    )
    .unwrap();

    assert_eq!(
        measured.qualify(),
        QualifiedMeasuredValue::EstimatedObservation {
            variable_id: VariableId::new("labor:unemployment_rate").unwrap(),
            unit: UnitId::new("basis_points").unwrap(),
            point_atoms: 430,
            uncertainty: MeasurementUncertainty::StandardError {
                standard_error_atoms: 12,
                source: UncertaintySource::Sampling,
                method_id: "variance:survey-linearization-v1".into(),
            },
        }
    );
}

#[test]
fn tiny_nonzero_uncertainty_does_not_upgrade_a_point_estimate_to_exact() {
    let qualified = MeasuredValue::new(
        &unemployment_rate(),
        UnitId::new("basis_points").unwrap(),
        430,
        MeasurementUncertainty::StandardError {
            standard_error_atoms: 1,
            source: UncertaintySource::MeasurementProcess,
            method_id: "variance:measurement-process-v1".into(),
        },
    )
    .unwrap()
    .qualify();

    assert!(matches!(
        qualified,
        QualifiedMeasuredValue::EstimatedObservation { .. }
    ));
}

#[test]
fn zero_standard_error_cannot_disguise_an_estimate_as_exact() {
    assert_eq!(
        MeasuredValue::new(
            &unemployment_rate(),
            UnitId::new("basis_points").unwrap(),
            430,
            MeasurementUncertainty::StandardError {
                standard_error_atoms: 0,
                source: UncertaintySource::Sampling,
                method_id: "variance:bad-zero-v1".into(),
            },
        ),
        Err(UncertaintyError::ZeroStandardError)
    );
}

#[test]
fn model_based_interval_retains_method_source_and_coverage() {
    let qualified = MeasuredValue::new(
        &unemployment_rate(),
        UnitId::new("basis_points").unwrap(),
        430,
        MeasurementUncertainty::Interval {
            lower_atoms: 400,
            upper_atoms: 460,
            source: UncertaintySource::ModelBased,
            coverage_bps: Some(9_500),
            method_id: "uncertainty:state-space-bootstrap-v2".into(),
        },
    )
    .unwrap()
    .qualify();

    assert_eq!(
        qualified,
        QualifiedMeasuredValue::EstimatedObservation {
            variable_id: VariableId::new("labor:unemployment_rate").unwrap(),
            unit: UnitId::new("basis_points").unwrap(),
            point_atoms: 430,
            uncertainty: MeasurementUncertainty::Interval {
                lower_atoms: 400,
                upper_atoms: 460,
                source: UncertaintySource::ModelBased,
                coverage_bps: Some(9_500),
                method_id: "uncertainty:state-space-bootstrap-v2".into(),
            },
        }
    );
}

#[test]
fn revision_range_without_probability_coverage_remains_estimated() {
    let qualified = MeasuredValue::new(
        &unemployment_rate(),
        UnitId::new("basis_points").unwrap(),
        430,
        MeasurementUncertainty::Interval {
            lower_atoms: 420,
            upper_atoms: 445,
            source: UncertaintySource::RevisionRange,
            coverage_bps: None,
            method_id: "revision-range:historical-vintages-v1".into(),
        },
    )
    .unwrap()
    .qualify();

    assert!(matches!(
        qualified,
        QualifiedMeasuredValue::EstimatedObservation {
            uncertainty: MeasurementUncertainty::Interval {
                source: UncertaintySource::RevisionRange,
                coverage_bps: None,
                ..
            },
            ..
        }
    ));
}

#[test]
fn interval_must_have_positive_width_and_contain_the_point_estimate() {
    let variable = unemployment_rate();
    for uncertainty in [
        MeasurementUncertainty::Interval {
            lower_atoms: 430,
            upper_atoms: 430,
            source: UncertaintySource::MeasurementProcess,
            coverage_bps: Some(9_500),
            method_id: "interval:zero-width-v1".into(),
        },
        MeasurementUncertainty::Interval {
            lower_atoms: 440,
            upper_atoms: 460,
            source: UncertaintySource::MeasurementProcess,
            coverage_bps: Some(9_500),
            method_id: "interval:misses-point-v1".into(),
        },
    ] {
        assert_eq!(
            MeasuredValue::new(
                &variable,
                UnitId::new("basis_points").unwrap(),
                430,
                uncertainty,
            ),
            Err(UncertaintyError::InvalidInterval)
        );
    }
}

#[test]
fn interval_coverage_must_be_strictly_between_zero_and_one() {
    let variable = unemployment_rate();
    for coverage in [0, 10_000] {
        assert_eq!(
            MeasuredValue::new(
                &variable,
                UnitId::new("basis_points").unwrap(),
                430,
                MeasurementUncertainty::Interval {
                    lower_atoms: 400,
                    upper_atoms: 460,
                    source: UncertaintySource::Sampling,
                    coverage_bps: Some(coverage),
                    method_id: "interval:coverage-v1".into(),
                },
            ),
            Err(UncertaintyError::InvalidCoverage)
        );
    }
}

#[test]
fn exact_observation_requires_an_explicit_exactness_basis() {
    assert_eq!(
        MeasuredValue::new(
            &unemployment_rate(),
            UnitId::new("basis_points").unwrap(),
            430,
            MeasurementUncertainty::ExactByConstruction {
                basis_id: "   ".into(),
            },
        ),
        Err(UncertaintyError::EmptyText("exactness basis id"))
    );

    let qualified = MeasuredValue::new(
        &unemployment_rate(),
        UnitId::new("basis_points").unwrap(),
        430,
        MeasurementUncertainty::ExactByConstruction {
            basis_id: "exactness:closed-ledger-count-v1".into(),
        },
    )
    .unwrap()
    .qualify();

    assert_eq!(
        qualified,
        QualifiedMeasuredValue::ExactObservation {
            variable_id: VariableId::new("labor:unemployment_rate").unwrap(),
            unit: UnitId::new("basis_points").unwrap(),
            value_atoms: 430,
            basis_id: "exactness:closed-ledger-count-v1".into(),
        }
    );
}

#[test]
fn measured_value_unit_must_match_the_etir_variable() {
    assert_eq!(
        MeasuredValue::new(
            &unemployment_rate(),
            UnitId::new("percent").unwrap(),
            43,
            MeasurementUncertainty::ExactByConstruction {
                basis_id: "exactness:test-v1".into(),
            },
        ),
        Err(UncertaintyError::UnitMismatch)
    );
}
