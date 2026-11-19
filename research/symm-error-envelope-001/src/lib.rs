// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Conservative finite-horizon error propagation for approximate symmetries.
//!
//! This crate propagates caller-supplied, independently qualified bounds. It
//! does not establish the equivariance defect, Lipschitz, domain-invariance,
//! objective, or constraint premises. It never authorizes quotient pruning.

use std::fmt;

const MAX_HORIZON: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalBoundV1 {
    pub numerator: u128,
    pub denominator: u128,
}

impl RationalBoundV1 {
    pub fn new(numerator: u128, denominator: u128) -> Result<Self, EnvelopeErrorV1> {
        if denominator == 0 {
            return Err(EnvelopeErrorV1::ZeroDenominator);
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepPremiseV1 {
    /// Lipschitz upper bound for the discrete step map on the certified domain.
    pub map_lipschitz: RationalBoundV1,
    /// Identifier for the independent derivation of the step-map Lipschitz bound.
    pub map_lipschitz_certificate_id: String,
    /// Upper bound on the one-step equivariance defect, in state units.
    pub equivariance_defect_upper: u128,
    /// Identifier for the independent derivation/measurement of this defect.
    pub defect_certificate_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomainEvidenceV1 {
    /// The caller asserts that the reference and transformed trajectories
    /// remain in the declared domain, with a certified reference-orbit
    /// boundary margin at every time index.
    Qualified {
        certificate_id: String,
        /// Distance from the reference orbit to the domain boundary, in state
        /// metric units, at t=0..=horizon.
        boundary_margin_by_step: Vec<u128>,
    },
    /// No domain-invariance certificate is available. The engine must abstain.
    NotEstablished,
    /// An upstream check has contradicted the domain-invariance premise.
    Contradicted { evidence_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveComparisonV1 {
    pub objective_unit_id: String,
    /// Lower bound on the nominal objective gap between the candidates.
    pub nominal_gap_lower: u128,
    /// Lipschitz bound from state error to objective error.
    pub objective_lipschitz: RationalBoundV1,
    pub objective_lipschitz_certificate_id: String,
    /// Direct objective equivariance/measurement error, in objective units.
    pub direct_defect_upper: u128,
    pub direct_defect_certificate_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintMarginV1 {
    pub constraint_unit_id: String,
    /// Lower bound on the nominal slack to the feasibility boundary.
    pub nominal_slack_lower: u128,
    /// Lipschitz bound from state error to constraint residual error.
    pub constraint_lipschitz: RationalBoundV1,
    pub constraint_lipschitz_certificate_id: String,
    /// Direct constraint equivariance/measurement error, in constraint units.
    pub direct_defect_upper: u128,
    pub direct_defect_certificate_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApproximateSymmetryProblemV1 {
    pub case_id: String,
    pub domain_id: String,
    pub state_unit_id: String,
    /// Fixed-point scale is encoded by the unit ID (e.g. "micrometre-int-v1").
    pub initial_state_error_upper: u128,
    pub state_error_tolerance: u128,
    /// One premise per discrete step. Defect bounds may vary by step.
    pub steps: Vec<StepPremiseV1>,
    pub domain_evidence: DomainEvidenceV1,
    pub objective: Option<ObjectiveComparisonV1>,
    pub constraint: Option<ConstraintMarginV1>,
}

impl ApproximateSymmetryProblemV1 {
    pub fn validate(&self) -> Result<(), EnvelopeErrorV1> {
        if self.case_id.trim().is_empty() {
            return Err(EnvelopeErrorV1::EmptyCaseId);
        }
        if self.domain_id.trim().is_empty() {
            return Err(EnvelopeErrorV1::EmptyDomainId);
        }
        if self.state_unit_id.trim().is_empty() {
            return Err(EnvelopeErrorV1::EmptyStateUnitId);
        }
        if self.steps.is_empty() {
            return Err(EnvelopeErrorV1::EmptyHorizon);
        }
        if self.steps.len() > MAX_HORIZON {
            return Err(EnvelopeErrorV1::HorizonTooLarge);
        }
        for step in &self.steps {
            if step.map_lipschitz.denominator == 0 {
                return Err(EnvelopeErrorV1::ZeroDenominator);
            }
            if step.map_lipschitz_certificate_id.trim().is_empty() {
                return Err(EnvelopeErrorV1::MissingMapLipschitzCertificate);
            }
            if step.defect_certificate_id.trim().is_empty() {
                return Err(EnvelopeErrorV1::MissingDefectCertificate);
            }
        }

        match &self.domain_evidence {
            DomainEvidenceV1::Qualified {
                certificate_id,
                boundary_margin_by_step,
            } => {
                if certificate_id.trim().is_empty() {
                    return Err(EnvelopeErrorV1::MissingDomainCertificate);
                }
                if boundary_margin_by_step.len() != self.steps.len() + 1 {
                    return Err(EnvelopeErrorV1::BoundaryMarginCountMismatch {
                        expected: self.steps.len() + 1,
                        actual: boundary_margin_by_step.len(),
                    });
                }
            }
            DomainEvidenceV1::Contradicted { evidence_id } if evidence_id.trim().is_empty() => {
                return Err(EnvelopeErrorV1::MissingContradictionEvidence);
            }
            DomainEvidenceV1::NotEstablished | DomainEvidenceV1::Contradicted { .. } => {}
        }

        if let Some(objective) = &self.objective {
            if objective.objective_unit_id.trim().is_empty() {
                return Err(EnvelopeErrorV1::EmptyObjectiveUnitId);
            }
            if objective.objective_lipschitz.denominator == 0 {
                return Err(EnvelopeErrorV1::ZeroDenominator);
            }
            if objective.objective_lipschitz_certificate_id.trim().is_empty()
                || objective.direct_defect_certificate_id.trim().is_empty()
            {
                return Err(EnvelopeErrorV1::MissingObjectiveCertificate);
            }
        }
        if let Some(constraint) = &self.constraint {
            if constraint.constraint_unit_id.trim().is_empty() {
                return Err(EnvelopeErrorV1::EmptyConstraintUnitId);
            }
            if constraint.constraint_lipschitz.denominator == 0 {
                return Err(EnvelopeErrorV1::ZeroDenominator);
            }
            if constraint.constraint_lipschitz_certificate_id.trim().is_empty()
                || constraint.direct_defect_certificate_id.trim().is_empty()
            {
                return Err(EnvelopeErrorV1::MissingConstraintCertificate);
            }
        }
        Ok(())
    }

    /// Canonical versioned bytes for deterministic receipt replay.
    /// These bytes are not a cryptographic digest or signature.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symm_error_envelope.v1\0".to_vec();
        append_string(&mut bytes, &self.case_id);
        append_string(&mut bytes, &self.domain_id);
        append_string(&mut bytes, &self.state_unit_id);
        bytes.extend_from_slice(&self.initial_state_error_upper.to_le_bytes());
        bytes.extend_from_slice(&self.state_error_tolerance.to_le_bytes());
        bytes.extend_from_slice(&(self.steps.len() as u64).to_le_bytes());
        for step in &self.steps {
            append_rational(&mut bytes, step.map_lipschitz);
            append_string(&mut bytes, &step.map_lipschitz_certificate_id);
            bytes.extend_from_slice(&step.equivariance_defect_upper.to_le_bytes());
            append_string(&mut bytes, &step.defect_certificate_id);
        }
        match &self.domain_evidence {
            DomainEvidenceV1::Qualified {
                certificate_id,
                boundary_margin_by_step,
            } => {
                bytes.push(0);
                append_string(&mut bytes, certificate_id);
                bytes.extend_from_slice(&(boundary_margin_by_step.len() as u64).to_le_bytes());
                for margin in boundary_margin_by_step {
                    bytes.extend_from_slice(&margin.to_le_bytes());
                }
            }
            DomainEvidenceV1::NotEstablished => bytes.push(1),
            DomainEvidenceV1::Contradicted { evidence_id } => {
                bytes.push(2);
                append_string(&mut bytes, evidence_id);
            }
        }
        match &self.objective {
            Some(objective) => {
                bytes.push(1);
                append_string(&mut bytes, &objective.objective_unit_id);
                bytes.extend_from_slice(&objective.nominal_gap_lower.to_le_bytes());
                append_rational(&mut bytes, objective.objective_lipschitz);
                append_string(&mut bytes, &objective.objective_lipschitz_certificate_id);
                bytes.extend_from_slice(&objective.direct_defect_upper.to_le_bytes());
                append_string(&mut bytes, &objective.direct_defect_certificate_id);
            }
            None => bytes.push(0),
        }
        match &self.constraint {
            Some(constraint) => {
                bytes.push(1);
                append_string(&mut bytes, &constraint.constraint_unit_id);
                bytes.extend_from_slice(&constraint.nominal_slack_lower.to_le_bytes());
                append_rational(&mut bytes, constraint.constraint_lipschitz);
                append_string(&mut bytes, &constraint.constraint_lipschitz_certificate_id);
                bytes.extend_from_slice(&constraint.direct_defect_upper.to_le_bytes());
                append_string(&mut bytes, &constraint.direct_defect_certificate_id);
            }
            None => bytes.push(0),
        }
        bytes
    }
}

fn append_string(target: &mut Vec<u8>, value: &str) {
    target.extend_from_slice(&(value.len() as u64).to_le_bytes());
    target.extend_from_slice(value.as_bytes());
}

fn append_rational(target: &mut Vec<u8>, value: RationalBoundV1) {
    target.extend_from_slice(&value.numerator.to_le_bytes());
    target.extend_from_slice(&value.denominator.to_le_bytes());
}

fn checked_ceil_mul_div(
    value: u128,
    numerator: u128,
    denominator: u128,
) -> Result<u128, EnvelopeErrorV1> {
    if denominator == 0 {
        return Err(EnvelopeErrorV1::ZeroDenominator);
    }
    let product = value
        .checked_mul(numerator)
        .ok_or(EnvelopeErrorV1::ArithmeticOverflow)?;
    let quotient = product / denominator;
    let remainder = product % denominator;
    if remainder == 0 {
        Ok(quotient)
    } else {
        quotient
            .checked_add(1)
            .ok_or(EnvelopeErrorV1::ArithmeticOverflow)
    }
}

fn checked_error_image(
    state_error: u128,
    lipschitz: RationalBoundV1,
    direct_defect: u128,
) -> Result<u128, EnvelopeErrorV1> {
    checked_ceil_mul_div(state_error, lipschitz.numerator, lipschitz.denominator)?
        .checked_add(direct_defect)
        .ok_or(EnvelopeErrorV1::ArithmeticOverflow)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeStatusV1 {
    CertifiedWithinTolerance,
    CertifiedBoundAboveTolerance,
    AbstainedDomainNotEstablished,
    AbstainedDomainContradicted,
    StoppedAtDomainBoundary { time_index: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateErrorBoundV1 {
    pub time_index: usize,
    pub error_upper_bound: u128,
    pub domain_boundary_margin: u128,
    pub within_domain_margin: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectiveAssessmentV1 {
    NotAssessed,
    RankingCertifiedStable { error_upper_bound: u128 },
    RankingUnresolved { error_upper_bound: u128 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintAssessmentV1 {
    NotAssessed,
    FeasibilityMarginCertified { error_upper_bound: u128 },
    FeasibilityUnresolved { error_upper_bound: u128 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorEnvelopeReceiptV1 {
    pub schema_version: u32,
    pub case_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub status: EnvelopeStatusV1,
    /// Includes t=0 and every propagated step up to the last valid bound.
    pub state_error_bounds: Vec<StateErrorBoundV1>,
    pub objective_assessment: ObjectiveAssessmentV1,
    pub constraint_assessment: ConstraintAssessmentV1,
    pub propagated_steps: usize,
    pub requested_horizon: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeErrorV1 {
    EmptyCaseId,
    EmptyDomainId,
    EmptyStateUnitId,
    EmptyHorizon,
    HorizonTooLarge,
    ZeroDenominator,
    MissingDefectCertificate,
    MissingMapLipschitzCertificate,
    MissingObjectiveCertificate,
    MissingConstraintCertificate,
    MissingDomainCertificate,
    MissingContradictionEvidence,
    BoundaryMarginCountMismatch { expected: usize, actual: usize },
    EmptyObjectiveUnitId,
    EmptyConstraintUnitId,
    ArithmeticOverflow,
    ReceiptMismatch,
}

impl fmt::Display for EnvelopeErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for EnvelopeErrorV1 {}

/// Propagate (e_{k+1} \le L_k e_k + \delta_k) with outward integer
/// rounding. Only an externally qualified domain certificate enables
/// propagation. A boundary-margin crossing stops the recurrence immediately.
pub fn propagate_error_envelope(
    problem: &ApproximateSymmetryProblemV1,
) -> Result<ErrorEnvelopeReceiptV1, EnvelopeErrorV1> {
    problem.validate()?;
    let mut state_error_bounds = Vec::new();

    let status = match &problem.domain_evidence {
        DomainEvidenceV1::NotEstablished => EnvelopeStatusV1::AbstainedDomainNotEstablished,
        DomainEvidenceV1::Contradicted { .. } => {
            EnvelopeStatusV1::AbstainedDomainContradicted
        }
        DomainEvidenceV1::Qualified {
            boundary_margin_by_step,
            ..
        } => {
            let initial_margin = boundary_margin_by_step[0];
            let initial_within = problem.initial_state_error_upper <= initial_margin;
            state_error_bounds.push(StateErrorBoundV1 {
                time_index: 0,
                error_upper_bound: problem.initial_state_error_upper,
                domain_boundary_margin: initial_margin,
                within_domain_margin: initial_within,
            });
            if !initial_within {
                EnvelopeStatusV1::StoppedAtDomainBoundary { time_index: 0 }
            } else {
                let mut current = problem.initial_state_error_upper;
                let mut stopped = None;
                for (step_index, step) in problem.steps.iter().enumerate() {
                    let next = checked_error_image(
                        current,
                        step.map_lipschitz,
                        step.equivariance_defect_upper,
                    )?;
                    let time_index = step_index + 1;
                    let margin = boundary_margin_by_step[time_index];
                    let within_domain_margin = next <= margin;
                    state_error_bounds.push(StateErrorBoundV1 {
                        time_index,
                        error_upper_bound: next,
                        domain_boundary_margin: margin,
                        within_domain_margin,
                    });
                    current = next;
                    if !within_domain_margin {
                        stopped = Some(time_index);
                        break;
                    }
                }
                if let Some(time_index) = stopped {
                    EnvelopeStatusV1::StoppedAtDomainBoundary { time_index }
                } else if current <= problem.state_error_tolerance {
                    EnvelopeStatusV1::CertifiedWithinTolerance
                } else {
                    EnvelopeStatusV1::CertifiedBoundAboveTolerance
                }
            }
        }
    };

    let propagated_steps = state_error_bounds.len().saturating_sub(1);
    let complete_horizon = matches!(
        &status,
        EnvelopeStatusV1::CertifiedWithinTolerance
            | EnvelopeStatusV1::CertifiedBoundAboveTolerance
    );

    let (objective_assessment, constraint_assessment) = if complete_horizon {
        let final_error = state_error_bounds
            .last()
            .map(|bound| bound.error_upper_bound)
            .ok_or(EnvelopeErrorV1::ReceiptMismatch)?;
        let objective_assessment = match &problem.objective {
            Some(objective) => {
                let error_upper_bound = checked_error_image(
                    final_error,
                    objective.objective_lipschitz,
                    objective.direct_defect_upper,
                )?;
                let two_sided_error = error_upper_bound
                    .checked_mul(2)
                    .ok_or(EnvelopeErrorV1::ArithmeticOverflow)?;
                if objective.nominal_gap_lower > two_sided_error {
                    ObjectiveAssessmentV1::RankingCertifiedStable { error_upper_bound }
                } else {
                    ObjectiveAssessmentV1::RankingUnresolved { error_upper_bound }
                }
            }
            None => ObjectiveAssessmentV1::NotAssessed,
        };
        let constraint_assessment = match &problem.constraint {
            Some(constraint) => {
                let error_upper_bound = checked_error_image(
                    final_error,
                    constraint.constraint_lipschitz,
                    constraint.direct_defect_upper,
                )?;
                if constraint.nominal_slack_lower > error_upper_bound {
                    ConstraintAssessmentV1::FeasibilityMarginCertified { error_upper_bound }
                } else {
                    ConstraintAssessmentV1::FeasibilityUnresolved { error_upper_bound }
                }
            }
            None => ConstraintAssessmentV1::NotAssessed,
        };
        (objective_assessment, constraint_assessment)
    } else {
        (
            ObjectiveAssessmentV1::NotAssessed,
            ConstraintAssessmentV1::NotAssessed,
        )
    };

    Ok(ErrorEnvelopeReceiptV1 {
        schema_version: 1,
        case_id: problem.case_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        status,
        state_error_bounds,
        objective_assessment,
        constraint_assessment,
        propagated_steps,
        requested_horizon: problem.steps.len(),
    })
}

/// Recompute every bound and decision from the frozen assumptions.
pub fn verify_error_envelope_receipt(
    problem: &ApproximateSymmetryProblemV1,
    receipt: &ErrorEnvelopeReceiptV1,
) -> Result<(), EnvelopeErrorV1> {
    if receipt.schema_version != 1
        || receipt.case_id != problem.case_id
        || receipt.problem_contract_bytes != problem.canonical_bytes()
    {
        return Err(EnvelopeErrorV1::ReceiptMismatch);
    }
    let replayed = propagate_error_envelope(problem)?;
    if *receipt != replayed {
        return Err(EnvelopeErrorV1::ReceiptMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qualified_fixture(
        lipschitz: RationalBoundV1,
        defect_bounds: &[u128],
    ) -> ApproximateSymmetryProblemV1 {
        ApproximateSymmetryProblemV1 {
            case_id: "scalar-toy-map-v1".to_owned(),
            domain_id: "closed-interval-domain-v1".to_owned(),
            state_unit_id: "micro-state-unit-v1".to_owned(),
            initial_state_error_upper: 0,
            state_error_tolerance: u128::MAX,
            steps: defect_bounds
                .iter()
                .enumerate()
                .map(|(index, defect)| StepPremiseV1 {
                    map_lipschitz: lipschitz,
                    map_lipschitz_certificate_id: format!("toy-lipschitz-{index}"),
                    equivariance_defect_upper: *defect,
                    defect_certificate_id: format!("toy-defect-{index}"),
                })
                .collect(),
            domain_evidence: DomainEvidenceV1::Qualified {
                certificate_id: "toy-domain-invariance-v1".to_owned(),
                boundary_margin_by_step: vec![u128::MAX; defect_bounds.len() + 1],
            },
            objective: None,
            constraint: None,
        }
    }

    #[test]
    fn contractive_neutral_and_expansive_recurrences_match_integer_oracle() {
        let contractive = qualified_fixture(RationalBoundV1::new(1, 2).unwrap(), &[100; 4]);
        let receipt = propagate_error_envelope(&contractive).unwrap();
        assert_eq!(
            receipt
                .state_error_bounds
                .iter()
                .map(|bound| bound.error_upper_bound)
                .collect::<Vec<_>>(),
            vec![0, 100, 150, 175, 188]
        );

        let neutral = qualified_fixture(RationalBoundV1::new(1, 1).unwrap(), &[100; 4]);
        let receipt = propagate_error_envelope(&neutral).unwrap();
        assert_eq!(
            receipt
                .state_error_bounds
                .iter()
                .map(|bound| bound.error_upper_bound)
                .collect::<Vec<_>>(),
            vec![0, 100, 200, 300, 400]
        );

        let expansive = qualified_fixture(RationalBoundV1::new(3, 2).unwrap(), &[100; 4]);
        let receipt = propagate_error_envelope(&expansive).unwrap();
        assert_eq!(
            receipt
                .state_error_bounds
                .iter()
                .map(|bound| bound.error_upper_bound)
                .collect::<Vec<_>>(),
            vec![0, 100, 250, 475, 813]
        );
    }

    #[test]
    fn time_varying_defect_envelopes_are_applied_at_their_own_steps() {
        let problem = qualified_fixture(RationalBoundV1::new(1, 1).unwrap(), &[100, 0, 50, 25]);
        let receipt = propagate_error_envelope(&problem).unwrap();
        assert_eq!(
            receipt
                .state_error_bounds
                .iter()
                .map(|bound| bound.error_upper_bound)
                .collect::<Vec<_>>(),
            vec![0, 100, 100, 150, 175]
        );
    }

    #[test]
    fn horizons_one_through_sixty_four_are_bounded_for_all_lipschitz_regimes() {
        for horizon in [1_usize, 2, 4, 8, 16, 32, 64] {
            for lipschitz in [
                RationalBoundV1::new(1, 2).unwrap(),
                RationalBoundV1::new(1, 1).unwrap(),
                RationalBoundV1::new(3, 2).unwrap(),
            ] {
                let problem = qualified_fixture(lipschitz, &vec![7; horizon]);
                let receipt = propagate_error_envelope(&problem).unwrap();
                assert_eq!(receipt.propagated_steps, horizon);
                assert_eq!(receipt.state_error_bounds.len(), horizon + 1);
            }
        }
    }

    #[test]
    fn missing_or_contradicted_domain_evidence_abstains_without_propagation() {
        let mut problem = qualified_fixture(RationalBoundV1::new(2, 1).unwrap(), &[100; 4]);
        problem.domain_evidence = DomainEvidenceV1::NotEstablished;
        let receipt = propagate_error_envelope(&problem).unwrap();
        assert_eq!(receipt.status, EnvelopeStatusV1::AbstainedDomainNotEstablished);
        assert!(receipt.state_error_bounds.is_empty());
        assert_eq!(receipt.propagated_steps, 0);

        problem.domain_evidence = DomainEvidenceV1::Contradicted {
            evidence_id: "domain-invariance-counterexample".to_owned(),
        };
        let receipt = propagate_error_envelope(&problem).unwrap();
        assert_eq!(receipt.status, EnvelopeStatusV1::AbstainedDomainContradicted);
        assert!(receipt.state_error_bounds.is_empty());
    }

    #[test]
    fn boundary_margin_crossing_stops_the_recurrence_at_the_first_unsafe_time() {
        let mut problem = qualified_fixture(RationalBoundV1::new(1, 1).unwrap(), &[40, 40, 40]);
        problem.domain_evidence = DomainEvidenceV1::Qualified {
            certificate_id: "narrow-domain-v1".to_owned(),
            boundary_margin_by_step: vec![100, 20, 100, 100],
        };
        let receipt = propagate_error_envelope(&problem).unwrap();
        assert_eq!(
            receipt.status,
            EnvelopeStatusV1::StoppedAtDomainBoundary { time_index: 1 }
        );
        assert_eq!(receipt.propagated_steps, 1);
        assert_eq!(receipt.state_error_bounds.len(), 2);
        assert_eq!(receipt.state_error_bounds[1].error_upper_bound, 40);
        assert!(!receipt.state_error_bounds[1].within_domain_margin);
        assert_eq!(receipt.objective_assessment, ObjectiveAssessmentV1::NotAssessed);
    }

    #[test]
    fn initial_error_outside_boundary_margin_prevents_any_step() {
        let mut problem = qualified_fixture(RationalBoundV1::new(1, 1).unwrap(), &[10, 10]);
        problem.initial_state_error_upper = 11;
        if let DomainEvidenceV1::Qualified {
            boundary_margin_by_step,
            ..
        } = &mut problem.domain_evidence
        {
            boundary_margin_by_step[0] = 10;
        }
        let receipt = propagate_error_envelope(&problem).unwrap();
        assert_eq!(
            receipt.status,
            EnvelopeStatusV1::StoppedAtDomainBoundary { time_index: 0 }
        );
        assert_eq!(receipt.propagated_steps, 0);
    }

    #[test]
    fn objective_and_constraint_margins_use_strict_safe_inequalities() {
        let mut problem = qualified_fixture(RationalBoundV1::new(1, 1).unwrap(), &[100]);
        problem.objective = Some(ObjectiveComparisonV1 {
            objective_unit_id: "micro-objective-unit-v1".to_owned(),
            nominal_gap_lower: 411,
            objective_lipschitz: RationalBoundV1::new(2, 1).unwrap(),
            objective_lipschitz_certificate_id: "objective-lipschitz-certificate".to_owned(),
            direct_defect_upper: 5,
            direct_defect_certificate_id: "objective-defect-certificate".to_owned(),
        });
        problem.constraint = Some(ConstraintMarginV1 {
            constraint_unit_id: "micro-constraint-unit-v1".to_owned(),
            nominal_slack_lower: 305,
            constraint_lipschitz: RationalBoundV1::new(3, 1).unwrap(),
            constraint_lipschitz_certificate_id: "constraint-lipschitz-certificate".to_owned(),
            direct_defect_upper: 4,
            direct_defect_certificate_id: "constraint-defect-certificate".to_owned(),
        });
        let receipt = propagate_error_envelope(&problem).unwrap();
        assert_eq!(
            receipt.objective_assessment,
            ObjectiveAssessmentV1::RankingCertifiedStable {
                error_upper_bound: 205
            }
        );
        assert_eq!(
            receipt.constraint_assessment,
            ConstraintAssessmentV1::FeasibilityMarginCertified {
                error_upper_bound: 304
            }
        );

        problem.objective.as_mut().unwrap().nominal_gap_lower = 410;
        problem.constraint.as_mut().unwrap().nominal_slack_lower = 304;
        let receipt = propagate_error_envelope(&problem).unwrap();
        assert_eq!(
            receipt.objective_assessment,
            ObjectiveAssessmentV1::RankingUnresolved {
                error_upper_bound: 205
            }
        );
        assert_eq!(
            receipt.constraint_assessment,
            ConstraintAssessmentV1::FeasibilityUnresolved {
                error_upper_bound: 304
            }
        );
    }

    #[test]
    fn zero_defect_and_zero_initial_error_remain_exactly_zero() {
        let problem = qualified_fixture(RationalBoundV1::new(7, 3).unwrap(), &[0; 64]);
        let receipt = propagate_error_envelope(&problem).unwrap();
        assert!(receipt
            .state_error_bounds
            .iter()
            .all(|bound| bound.error_upper_bound == 0));
        assert_eq!(receipt.status, EnvelopeStatusV1::CertifiedWithinTolerance);
    }

    #[test]
    fn receipt_replay_rejects_tampering_and_mutated_premises() {
        let problem = qualified_fixture(RationalBoundV1::new(1, 1).unwrap(), &[10, 20]);
        let receipt = propagate_error_envelope(&problem).unwrap();
        verify_error_envelope_receipt(&problem, &receipt).unwrap();

        let mut forged = receipt.clone();
        forged.state_error_bounds[1].error_upper_bound = 0;
        assert_eq!(
            verify_error_envelope_receipt(&problem, &forged),
            Err(EnvelopeErrorV1::ReceiptMismatch)
        );

        let mut changed = problem.clone();
        changed.steps[0].equivariance_defect_upper = 11;
        assert_eq!(
            verify_error_envelope_receipt(&changed, &receipt),
            Err(EnvelopeErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn missing_lipschitz_and_objective_constraint_provenance_fail_closed() {
        let mut problem = qualified_fixture(RationalBoundV1::new(1, 1).unwrap(), &[10]);
        problem.steps[0].map_lipschitz_certificate_id.clear();
        assert_eq!(
            problem.validate(),
            Err(EnvelopeErrorV1::MissingMapLipschitzCertificate)
        );

        let mut problem = qualified_fixture(RationalBoundV1::new(1, 1).unwrap(), &[10]);
        problem.objective = Some(ObjectiveComparisonV1 {
            objective_unit_id: "objective-unit".to_owned(),
            nominal_gap_lower: 100,
            objective_lipschitz: RationalBoundV1::new(1, 1).unwrap(),
            objective_lipschitz_certificate_id: String::new(),
            direct_defect_upper: 1,
            direct_defect_certificate_id: "objective-defect".to_owned(),
        });
        assert_eq!(
            problem.validate(),
            Err(EnvelopeErrorV1::MissingObjectiveCertificate)
        );

        let mut problem = qualified_fixture(RationalBoundV1::new(1, 1).unwrap(), &[10]);
        problem.constraint = Some(ConstraintMarginV1 {
            constraint_unit_id: "constraint-unit".to_owned(),
            nominal_slack_lower: 100,
            constraint_lipschitz: RationalBoundV1::new(1, 1).unwrap(),
            constraint_lipschitz_certificate_id: "constraint-lipschitz".to_owned(),
            direct_defect_upper: 1,
            direct_defect_certificate_id: String::new(),
        });
        assert_eq!(
            problem.validate(),
            Err(EnvelopeErrorV1::MissingConstraintCertificate)
        );
    }

    #[test]
    fn arithmetic_overflow_and_missing_identifiers_fail_closed() {
        let mut overflow = qualified_fixture(RationalBoundV1::new(2, 1).unwrap(), &[0]);
        overflow.initial_state_error_upper = u128::MAX;
        assert_eq!(
            propagate_error_envelope(&overflow),
            Err(EnvelopeErrorV1::ArithmeticOverflow)
        );

        let mut invalid = qualified_fixture(RationalBoundV1::new(1, 1).unwrap(), &[0]);
        invalid.steps[0].defect_certificate_id.clear();
        assert_eq!(
            invalid.validate(),
            Err(EnvelopeErrorV1::MissingDefectCertificate)
        );
        assert_eq!(
            RationalBoundV1::new(1, 0),
            Err(EnvelopeErrorV1::ZeroDenominator)
        );
    }
}
