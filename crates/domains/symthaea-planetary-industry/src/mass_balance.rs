// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! PIE-001 interval-aware bulk-mass conservation.
//!
//! This module intentionally mirrors the independent
//! `scripts/pie-mass-balance-oracle.py` reference semantics. It proves only
//! whether bulk mass closure is exact, possible within declared uncertainty,
//! or impossible within the supplied bounds. It does not prove chemistry,
//! thermodynamics, process yield, equipment feasibility, or safety.

use serde::{Deserialize, Serialize};

use crate::{MassKg, OntologyError, ProcessDefinition};

/// Conservative classification of one process-basis mass balance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MassBalanceStatus {
    /// Every stream is point-valued and input/output totals agree within tolerance.
    ExactBalanced,
    /// Every stream is point-valued but totals disagree beyond tolerance.
    ExactUnbalanced,
    /// At least one stream is uncertain and closure is possible within the bounds.
    PossibleWithUncertainty,
    /// Input/output intervals remain disjoint even after the declared tolerance.
    ImpossibleWithinBounds,
}

/// Absolute + relative tolerance policy for bulk-mass closure.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MassBalanceTolerance {
    absolute_kg: MassKg,
    relative_fraction: f64,
}

impl MassBalanceTolerance {
    /// Construct a validated tolerance policy.
    pub fn new(absolute_kg: f64, relative_fraction: f64) -> Result<Self, OntologyError> {
        if !relative_fraction.is_finite() || relative_fraction < 0.0 {
            return Err(OntologyError::InvalidQuantity(
                "mass_balance_relative_tolerance",
            ));
        }
        Ok(Self {
            absolute_kg: MassKg::new(absolute_kg)?,
            relative_fraction,
        })
    }

    /// Exact-zero tolerance.
    pub fn zero() -> Self {
        Self {
            absolute_kg: MassKg::new(0.0).expect("zero mass is valid"),
            relative_fraction: 0.0,
        }
    }

    /// Absolute tolerance in kilograms.
    pub fn absolute_kg(self) -> f64 {
        self.absolute_kg.value()
    }

    /// Relative tolerance as a non-negative fraction.
    pub fn relative_fraction(self) -> f64 {
        self.relative_fraction
    }

    fn validate(self) -> Result<(), OntologyError> {
        self.absolute_kg.validate()?;
        if self.relative_fraction.is_finite() && self.relative_fraction >= 0.0 {
            Ok(())
        } else {
            Err(OntologyError::InvalidQuantity(
                "mass_balance_relative_tolerance",
            ))
        }
    }
}

impl Default for MassBalanceTolerance {
    fn default() -> Self {
        Self::zero()
    }
}

/// Mass-balance result for one declared process basis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MassBalanceReport {
    /// Conservative classification.
    pub status: MassBalanceStatus,
    /// Sum of lower input-mass bounds.
    pub input_min_kg: f64,
    /// Sum of upper input-mass bounds.
    pub input_max_kg: f64,
    /// Sum of lower output-mass bounds.
    pub output_min_kg: f64,
    /// Sum of upper output-mass bounds.
    pub output_max_kg: f64,
    /// Lower bound of `output - input` residual.
    pub residual_min_kg: f64,
    /// Upper bound of `output - input` residual.
    pub residual_max_kg: f64,
    /// Effective tolerance after combining absolute and relative policies.
    pub tolerance_kg: f64,
    /// Exact residual when every stream is point-valued.
    pub exact_residual_kg: Option<f64>,
    /// Whether all material streams are point-valued.
    pub all_streams_exact: bool,
}

/// Evaluate bulk-mass closure for one process definition.
///
/// Every `ProcessInput` and every `ProcessOutput` participates, regardless of
/// whether an output is a product, co-product, by-product, waste, or recycle
/// candidate. This is why matter is forbidden from the PIE utility channel.
pub fn evaluate_mass_balance(
    process: &ProcessDefinition,
    tolerance: MassBalanceTolerance,
) -> Result<MassBalanceReport, OntologyError> {
    process.validate()?;
    tolerance.validate()?;

    let input_min_kg: f64 = process
        .inputs
        .iter()
        .map(|stream| stream.mass_kg.min.value())
        .sum();
    let input_max_kg: f64 = process
        .inputs
        .iter()
        .map(|stream| stream.mass_kg.max.value())
        .sum();
    let output_min_kg: f64 = process
        .outputs
        .iter()
        .map(|stream| stream.mass_kg.min.value())
        .sum();
    let output_max_kg: f64 = process
        .outputs
        .iter()
        .map(|stream| stream.mass_kg.max.value())
        .sum();

    let reference_mass = input_max_kg.max(output_max_kg);
    let tolerance_kg = tolerance
        .absolute_kg()
        .max(tolerance.relative_fraction() * reference_mass);

    // Reference-oracle sign convention: output - input.
    let residual_min_kg = output_min_kg - input_max_kg;
    let residual_max_kg = output_max_kg - input_min_kg;

    let all_streams_exact = process
        .inputs
        .iter()
        .all(|stream| stream.mass_kg.min == stream.mass_kg.max)
        && process
            .outputs
            .iter()
            .all(|stream| stream.mass_kg.min == stream.mass_kg.max);

    if all_streams_exact {
        let exact_residual_kg = output_min_kg - input_min_kg;
        let status = if exact_residual_kg.abs() <= tolerance_kg {
            MassBalanceStatus::ExactBalanced
        } else {
            MassBalanceStatus::ExactUnbalanced
        };
        return Ok(MassBalanceReport {
            status,
            input_min_kg,
            input_max_kg,
            output_min_kg,
            output_max_kg,
            residual_min_kg,
            residual_max_kg,
            tolerance_kg,
            exact_residual_kg: Some(exact_residual_kg),
            all_streams_exact: true,
        });
    }

    // Overlap expanded by the declared tolerance establishes only that some
    // admissible realization could conserve mass. Uncertainty never upgrades a
    // result to `ExactBalanced`.
    let disjoint_beyond_tolerance = input_max_kg + tolerance_kg < output_min_kg
        || output_max_kg + tolerance_kg < input_min_kg;
    let status = if disjoint_beyond_tolerance {
        MassBalanceStatus::ImpossibleWithinBounds
    } else {
        MassBalanceStatus::PossibleWithUncertainty
    };

    Ok(MassBalanceReport {
        status,
        input_min_kg,
        input_max_kg,
        output_min_kg,
        output_max_kg,
        residual_min_kg,
        residual_max_kg,
        tolerance_kg,
        exact_residual_kg: None,
        all_streams_exact: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CelestialBody, DependencyCriticality, EquipmentRequirement, EvidenceClass, EvidenceRef,
        MaterialGrade, OutputDisposition, PhysicalForm, ProcessInput, ProcessInputRole,
        ProcessOutput, ProcessOutputRole,
    };

    fn grade() -> MaterialGrade {
        MaterialGrade {
            label: "synthetic".into(),
            specification_ref: None,
        }
    }

    fn hypothesis(id: &str) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.into(),
            class: EvidenceClass::Hypothesis,
            source: String::new(),
            note: Some("synthetic PIE-001 fixture".into()),
        }
    }

    fn process(inputs: &[(f64, f64)], outputs: &[(f64, f64)]) -> ProcessDefinition {
        ProcessDefinition {
            process_id: "mass-balance-fixture".into(),
            name: "Synthetic mass balance fixture".into(),
            inputs: inputs
                .iter()
                .enumerate()
                .map(|(i, (min, max))| ProcessInput {
                    material_key: format!("input-{i}"),
                    required_grade: None,
                    role: ProcessInputRole::Feedstock,
                    mass_kg: crate::MassRangeKg::new(*min, *max).unwrap(),
                })
                .collect(),
            outputs: outputs
                .iter()
                .enumerate()
                .map(|(i, (min, max))| ProcessOutput {
                    material_key: format!("output-{i}"),
                    grade: grade(),
                    form: PhysicalForm::Bulk,
                    role: match i {
                        0 => ProcessOutputRole::Product,
                        1 => ProcessOutputRole::Byproduct,
                        _ => ProcessOutputRole::Waste,
                    },
                    mass_kg: crate::MassRangeKg::new(*min, *max).unwrap(),
                    disposition: OutputDisposition::Inventory,
                })
                .collect(),
            utilities: vec![],
            equipment: vec![EquipmentRequirement {
                equipment_class: "synthetic-equipment".into(),
                quantity: 1,
                criticality: DependencyCriticality::Essential,
                evidence: vec![hypothesis("equip")],
            }],
            environment: vec![crate::EnvironmentConstraint::Body(CelestialBody::Moon)],
            evidence: vec![hypothesis("process")],
        }
    }

    #[test]
    fn exact_closure_matches_oracle() {
        let report = evaluate_mass_balance(
            &process(&[(10.0, 10.0)], &[(10.0, 10.0)]),
            MassBalanceTolerance::zero(),
        )
        .unwrap();
        assert_eq!(report.status, MassBalanceStatus::ExactBalanced);
        assert_eq!(report.exact_residual_kg, Some(0.0));
    }

    #[test]
    fn exact_mismatch_obeys_declared_tolerance_only() {
        let fixture = process(&[(10.0, 10.0)], &[(9.9, 9.9)]);
        let loose = evaluate_mass_balance(&fixture, MassBalanceTolerance::new(0.2, 0.0).unwrap())
            .unwrap();
        let tight = evaluate_mass_balance(&fixture, MassBalanceTolerance::new(0.01, 0.0).unwrap())
            .unwrap();
        assert_eq!(loose.status, MassBalanceStatus::ExactBalanced);
        assert_eq!(tight.status, MassBalanceStatus::ExactUnbalanced);
    }

    #[test]
    fn overlap_is_possible_never_exact() {
        let report = evaluate_mass_balance(
            &process(&[(9.0, 11.0)], &[(10.0, 12.0)]),
            MassBalanceTolerance::zero(),
        )
        .unwrap();
        assert_eq!(report.status, MassBalanceStatus::PossibleWithUncertainty);
        assert_eq!(report.exact_residual_kg, None);
    }

    #[test]
    fn disjoint_intervals_are_impossible() {
        let report = evaluate_mass_balance(
            &process(&[(9.0, 10.0)], &[(11.0, 12.0)]),
            MassBalanceTolerance::zero(),
        )
        .unwrap();
        assert_eq!(report.status, MassBalanceStatus::ImpossibleWithinBounds);
    }

    #[test]
    fn products_byproducts_and_waste_all_count() {
        let report = evaluate_mass_balance(
            &process(&[(10.0, 10.0)], &[(6.0, 6.0), (2.0, 2.0), (2.0, 2.0)]),
            MassBalanceTolerance::zero(),
        )
        .unwrap();
        assert_eq!(report.status, MassBalanceStatus::ExactBalanced);
        assert_eq!(report.output_min_kg, 10.0);
    }

    #[test]
    fn all_material_inputs_count() {
        let report = evaluate_mass_balance(
            &process(&[(8.0, 8.0), (2.0, 2.0)], &[(10.0, 10.0)]),
            MassBalanceTolerance::zero(),
        )
        .unwrap();
        assert_eq!(report.status, MassBalanceStatus::ExactBalanced);
    }

    #[test]
    fn widening_uncertainty_never_strengthens_to_exact_balance() {
        let exact_bad = evaluate_mass_balance(
            &process(&[(10.0, 10.0)], &[(9.0, 9.0)]),
            MassBalanceTolerance::zero(),
        )
        .unwrap();
        let widened = evaluate_mass_balance(
            &process(&[(9.0, 11.0)], &[(8.0, 10.0)]),
            MassBalanceTolerance::zero(),
        )
        .unwrap();
        assert_eq!(exact_bad.status, MassBalanceStatus::ExactUnbalanced);
        assert_eq!(widened.status, MassBalanceStatus::PossibleWithUncertainty);
        assert_ne!(widened.status, MassBalanceStatus::ExactBalanced);
    }

    #[test]
    fn relative_tolerance_uses_larger_conservative_total() {
        let report = evaluate_mass_balance(
            &process(&[(100.0, 100.0)], &[(99.0, 99.0)]),
            MassBalanceTolerance::new(0.0, 0.02).unwrap(),
        )
        .unwrap();
        assert_eq!(report.status, MassBalanceStatus::ExactBalanced);
        assert!((report.tolerance_kg - 2.0).abs() < 1e-12);
    }

    #[test]
    fn invalid_tolerance_fails_closed() {
        assert!(MassBalanceTolerance::new(-1.0, 0.0).is_err());
        assert!(MassBalanceTolerance::new(0.0, -0.1).is_err());
        assert!(MassBalanceTolerance::new(0.0, f64::NAN).is_err());
    }
}
