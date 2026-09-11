// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gauge-invariant spectroscopy-operator metadata and provenance.
//!
//! On a cubic lattice, continuum spin `J` is reduced to irreducible
//! representations of the octahedral group.  A lattice irrep is therefore not
//! a unique continuum-spin identification.  This module records the operator
//! construction needed for a variational basis without over-promoting that
//! representation label into a physical state assignment.

use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubicIrrep {
    A1,
    A2,
    E,
    T1,
    T2,
}

impl CubicIrrep {
    /// Continuum spins contributing to this cubic irrep for `J <= 4`.
    ///
    /// This is a low-spin subduction aid, not a state-identification rule.
    pub const fn low_spin_candidates(self) -> &'static [u8] {
        match self {
            Self::A1 => &[0, 4],
            Self::A2 => &[3],
            Self::E => &[2, 4],
            Self::T1 => &[1, 3, 4],
            Self::T2 => &[2, 3, 4],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Parity {
    Positive,
    Negative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChargeConjugation {
    Positive,
    Negative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LatticeQuantumNumbers {
    pub irrep: CubicIrrep,
    pub parity: Parity,
    pub charge_conjugation: ChargeConjugation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpatialStep {
    /// Cartesian spatial axis: 0=x, 1=y, 2=z.
    pub axis: u8,
    /// Unit step direction, exactly +1 or -1.
    pub direction: i8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosedLoopGeometry {
    pub steps: Vec<SpatialStep>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SmearingSpec {
    None,
    Ape { alpha: f64, iterations: usize },
    Stout { rho: f64, iterations: usize },
    GradientFlow { flow_time_lattice_units: f64 },
    Other { method: String, parameters: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlueballOperator {
    pub id: String,
    pub quantum_numbers: LatticeQuantumNumbers,
    pub loop_geometry: ClosedLoopGeometry,
    pub smearing: SmearingSpec,
    /// Stable description/hash/ID for the group-projection procedure.
    pub projection_lineage: String,
    /// Stable source/code/configuration lineage for construction.
    pub construction_lineage: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OperatorBasis {
    pub basis_id: String,
    pub operators: Vec<GlueballOperator>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperatorError {
    EmptyBasis,
    DuplicateOperatorId(String),
    InvalidStep,
    OpenLoop,
    InvalidSmearing,
    MissingLineage,
    MixedQuantumNumbers,
}

impl ClosedLoopGeometry {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.steps.is_empty() {
            return Err(OperatorError::OpenLoop);
        }
        let mut displacement = [0_i32; 3];
        for step in &self.steps {
            if step.axis > 2 || !matches!(step.direction, -1 | 1) {
                return Err(OperatorError::InvalidStep);
            }
            displacement[step.axis as usize] += step.direction as i32;
        }
        if displacement != [0, 0, 0] {
            return Err(OperatorError::OpenLoop);
        }
        Ok(())
    }
}

impl SmearingSpec {
    pub fn validate(&self) -> Result<(), OperatorError> {
        let valid = match self {
            Self::None => true,
            Self::Ape { alpha, iterations } => alpha.is_finite() && *alpha > 0.0 && *iterations > 0,
            Self::Stout { rho, iterations } => rho.is_finite() && *rho > 0.0 && *iterations > 0,
            Self::GradientFlow { flow_time_lattice_units } => {
                flow_time_lattice_units.is_finite() && *flow_time_lattice_units > 0.0
            }
            Self::Other { method, parameters } => {
                !method.trim().is_empty() && !parameters.trim().is_empty()
            }
        };
        if valid {
            Ok(())
        } else {
            Err(OperatorError::InvalidSmearing)
        }
    }
}

impl GlueballOperator {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.id.trim().is_empty()
            || self.projection_lineage.trim().is_empty()
            || self.construction_lineage.trim().is_empty()
        {
            return Err(OperatorError::MissingLineage);
        }
        self.loop_geometry.validate()?;
        self.smearing.validate()?;
        Ok(())
    }
}

impl OperatorBasis {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.basis_id.trim().is_empty() {
            return Err(OperatorError::MissingLineage);
        }
        if self.operators.is_empty() {
            return Err(OperatorError::EmptyBasis);
        }
        let expected = self.operators[0].quantum_numbers;
        let mut ids = HashSet::new();
        for operator in &self.operators {
            operator.validate()?;
            if operator.quantum_numbers != expected {
                return Err(OperatorError::MixedQuantumNumbers);
            }
            if !ids.insert(operator.id.clone()) {
                return Err(OperatorError::DuplicateOperatorId(operator.id.clone()));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plaquette() -> ClosedLoopGeometry {
        ClosedLoopGeometry {
            steps: vec![
                SpatialStep { axis: 0, direction: 1 },
                SpatialStep { axis: 1, direction: 1 },
                SpatialStep { axis: 0, direction: -1 },
                SpatialStep { axis: 1, direction: -1 },
            ],
        }
    }

    fn op(id: &str, irrep: CubicIrrep) -> GlueballOperator {
        GlueballOperator {
            id: id.into(),
            quantum_numbers: LatticeQuantumNumbers {
                irrep,
                parity: Parity::Positive,
                charge_conjugation: ChargeConjugation::Positive,
            },
            loop_geometry: plaquette(),
            smearing: SmearingSpec::Ape { alpha: 0.4, iterations: 8 },
            projection_lineage: "octahedral-projector-v1".into(),
            construction_lineage: "fixture-v1".into(),
        }
    }

    #[test]
    fn low_spin_subduction_does_not_make_irrep_unique_spin() {
        assert_eq!(CubicIrrep::A1.low_spin_candidates(), &[0, 4]);
        assert_eq!(CubicIrrep::E.low_spin_candidates(), &[2, 4]);
        assert_eq!(CubicIrrep::T2.low_spin_candidates(), &[2, 3, 4]);
    }

    #[test]
    fn closed_loop_validates() {
        assert!(plaquette().validate().is_ok());
    }

    #[test]
    fn open_loop_fails_closed() {
        let open = ClosedLoopGeometry {
            steps: vec![
                SpatialStep { axis: 0, direction: 1 },
                SpatialStep { axis: 1, direction: 1 },
            ],
        };
        assert_eq!(open.validate(), Err(OperatorError::OpenLoop));
    }

    #[test]
    fn basis_requires_one_lattice_channel() {
        let basis = OperatorBasis {
            basis_id: "mixed-fixture".into(),
            operators: vec![op("a", CubicIrrep::A1), op("b", CubicIrrep::E)],
        };
        assert_eq!(basis.validate(), Err(OperatorError::MixedQuantumNumbers));
    }

    #[test]
    fn valid_basis_retains_distinct_operator_constructions() {
        let mut second = op("rect", CubicIrrep::A1);
        second.smearing = SmearingSpec::GradientFlow { flow_time_lattice_units: 1.5 };
        let basis = OperatorBasis {
            basis_id: "a1pp-v1".into(),
            operators: vec![op("plaq", CubicIrrep::A1), second],
        };
        assert!(basis.validate().is_ok());
    }
}
