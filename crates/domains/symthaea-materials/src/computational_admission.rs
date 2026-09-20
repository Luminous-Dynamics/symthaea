// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed computational-evidence admission above sealed MAT-011 evaluations.
//!
//! A well-formed or even sealed evaluation does not automatically establish a materials
//! evidence stage. Admission additionally requires that origin, evaluator capability,
//! observation method, applicability state, property identity, unit, and criterion match
//! the exact proposition being admitted.
//!
//! This first tranche deliberately supports only two propositions with existing contracts:
//! in-domain ML screening and in-domain local DFT negative formation energy. Convex-hull and
//! phonon promotion remain unsupported until their exact property/capability contracts are frozen.

use crate::authority_firewall::{AuthorityFirewallError, ScientificEvaluationRef};
use crate::conditioned_property::PropertyObservationMethod;
use crate::evaluation_seal::MultiFidelityEvaluationSeal;
use crate::evidence::{MaterialsEvidenceKind, MaterialsEvidenceStage};
use crate::multi_fidelity::{
    ApplicabilityState, EvaluationMethodClass, EvaluationOrigin, MultiFidelityEvaluation,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

const FORMATION_ENERGY_PROPERTY_ID: &str = "formation_energy";
const EV_PER_ATOM_UNIT: &str = "eV/atom";
const ML_SCREEN_CRITERION: &str = "sealed-in-domain-surrogate-screen-v1";
const DFT_FORMATION_CRITERION: &str = "sealed-in-domain-local-dft-negative-formation-energy-v1";

/// Verified eligibility for one narrow computational materials-evidence proposition.
///
/// Fields are private so ordinary Rust construction cannot mint admission. Deserialized
/// values remain untrusted until `validate_against` succeeds against the exact evaluation
/// and semantic seal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputationalEvidenceAdmission {
    stage: MaterialsEvidenceStage,
    kind: MaterialsEvidenceKind,
    scientific_evaluation: ScientificEvaluationRef,
    criterion_id: String,
}

impl ComputationalEvidenceAdmission {
    /// Evidence stage established by this exact admission proof.
    pub fn stage(&self) -> MaterialsEvidenceStage {
        self.stage
    }

    /// Evidence kind established by this exact admission proof.
    pub fn kind(&self) -> MaterialsEvidenceKind {
        self.kind
    }

    /// Exact sealed scientific evaluation supporting the admission.
    pub fn scientific_evaluation(&self) -> &ScientificEvaluationRef {
        &self.scientific_evaluation
    }

    /// Stable criterion contract used for this admission.
    pub fn criterion_id(&self) -> &str {
        &self.criterion_id
    }

    /// Revalidate a stored admission against the exact evaluation and seal.
    pub fn validate_against(
        &self,
        evaluation: &MultiFidelityEvaluation,
        seal: &MultiFidelityEvaluationSeal,
    ) -> Result<(), ComputationalAdmissionError> {
        let expected = match self.stage {
            MaterialsEvidenceStage::MlScreened => admit_ml_screening(evaluation, seal)?,
            MaterialsEvidenceStage::DftFormationStable => {
                admit_negative_dft_formation_energy(evaluation, seal)?
            }
            other => return Err(ComputationalAdmissionError::UnsupportedStage(other)),
        };
        if self != &expected {
            return Err(ComputationalAdmissionError::AdmissionMismatch);
        }
        Ok(())
    }
}

/// Admit an exact sealed surrogate result as in-domain ML screening evidence.
///
/// This establishes only that an explicitly in-domain surrogate model screened the exact
/// subject/property. It does not establish that the predicted value is physically correct.
pub fn admit_ml_screening(
    evaluation: &MultiFidelityEvaluation,
    seal: &MultiFidelityEvaluationSeal,
) -> Result<ComputationalEvidenceAdmission, ComputationalAdmissionError> {
    let scientific_evaluation = ScientificEvaluationRef::from_sealed_evaluation(evaluation, seal)?;

    if evaluation.origin != EvaluationOrigin::SurrogatePrediction {
        return Err(ComputationalAdmissionError::WrongOrigin {
            expected: EvaluationOrigin::SurrogatePrediction,
            actual: evaluation.origin,
        });
    }
    if evaluation.method_class != EvaluationMethodClass::Surrogate {
        return Err(ComputationalAdmissionError::WrongMethodClass);
    }
    if !matches!(
        &evaluation.observation.method,
        PropertyObservationMethod::ModelPrediction { .. }
    ) {
        return Err(ComputationalAdmissionError::WrongObservationMethod);
    }
    require_in_domain(evaluation)?;

    Ok(ComputationalEvidenceAdmission {
        stage: MaterialsEvidenceStage::MlScreened,
        kind: MaterialsEvidenceKind::MachineLearningModel,
        scientific_evaluation,
        criterion_id: ML_SCREEN_CRITERION.to_string(),
    })
}

/// Admit an exact sealed local DFT result as negative formation-energy evidence.
///
/// Negative formation energy is deliberately narrower than convex-hull stability. This
/// function must never be used to establish `ConvexHullScreened` or synthesizability.
pub fn admit_negative_dft_formation_energy(
    evaluation: &MultiFidelityEvaluation,
    seal: &MultiFidelityEvaluationSeal,
) -> Result<ComputationalEvidenceAdmission, ComputationalAdmissionError> {
    let scientific_evaluation = ScientificEvaluationRef::from_sealed_evaluation(evaluation, seal)?;

    if evaluation.origin != EvaluationOrigin::LocalReproduction {
        return Err(ComputationalAdmissionError::WrongOrigin {
            expected: EvaluationOrigin::LocalReproduction,
            actual: evaluation.origin,
        });
    }
    if evaluation.method_class != EvaluationMethodClass::Dft {
        return Err(ComputationalAdmissionError::WrongMethodClass);
    }
    if !matches!(
        &evaluation.observation.method,
        PropertyObservationMethod::Calculation { .. }
    ) {
        return Err(ComputationalAdmissionError::WrongObservationMethod);
    }
    require_in_domain(evaluation)?;
    if evaluation.observation.property_id != FORMATION_ENERGY_PROPERTY_ID {
        return Err(ComputationalAdmissionError::WrongProperty {
            expected: FORMATION_ENERGY_PROPERTY_ID,
            actual: evaluation.observation.property_id.clone(),
        });
    }
    if evaluation.observation.unit != EV_PER_ATOM_UNIT {
        return Err(ComputationalAdmissionError::WrongUnit {
            expected: EV_PER_ATOM_UNIT,
            actual: evaluation.observation.unit.clone(),
        });
    }
    if evaluation.observation.value >= 0.0 {
        return Err(ComputationalAdmissionError::FormationEnergyNotNegative(
            evaluation.observation.value,
        ));
    }

    Ok(ComputationalEvidenceAdmission {
        stage: MaterialsEvidenceStage::DftFormationStable,
        kind: MaterialsEvidenceKind::DftCalculation,
        scientific_evaluation,
        criterion_id: DFT_FORMATION_CRITERION.to_string(),
    })
}

fn require_in_domain(
    evaluation: &MultiFidelityEvaluation,
) -> Result<(), ComputationalAdmissionError> {
    if evaluation.applicability.state == ApplicabilityState::InDomain {
        Ok(())
    } else {
        Err(ComputationalAdmissionError::NotExplicitlyInDomain(
            evaluation.applicability.state,
        ))
    }
}

/// Failure to admit a sealed evaluation into a computational evidence proposition.
#[derive(Debug, Clone, PartialEq)]
pub enum ComputationalAdmissionError {
    /// Exact sealed-evaluation reference could not be established.
    Firewall(AuthorityFirewallError),
    /// Evaluation origin does not satisfy the requested proposition.
    WrongOrigin {
        /// Required origin.
        expected: EvaluationOrigin,
        /// Observed origin.
        actual: EvaluationOrigin,
    },
    /// MAT-011 evaluator family does not satisfy the requested proposition.
    WrongMethodClass,
    /// MAT-008 observation method does not satisfy the requested proposition.
    WrongObservationMethod,
    /// Result is not explicitly inside its declared applicability domain.
    NotExplicitlyInDomain(ApplicabilityState),
    /// Property identifier does not match the required proposition.
    WrongProperty {
        /// Required property ID.
        expected: &'static str,
        /// Observed property ID.
        actual: String,
    },
    /// Unit does not match the exact criterion contract.
    WrongUnit {
        /// Required unit.
        expected: &'static str,
        /// Observed unit.
        actual: String,
    },
    /// Local DFT formation energy was not strictly negative.
    FormationEnergyNotNegative(f64),
    /// This tranche has no frozen admission theorem for the requested stage.
    UnsupportedStage(MaterialsEvidenceStage),
    /// Stored admission no longer matches the exact evaluation/criterion projection.
    AdmissionMismatch,
}

impl fmt::Display for ComputationalAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Firewall(error) => write!(formatter, "sealed evaluation rejected: {error}"),
            Self::WrongOrigin { expected, actual } => {
                write!(formatter, "wrong evaluation origin: expected {expected:?}, got {actual:?}")
            }
            Self::WrongMethodClass => formatter.write_str("wrong evaluator method class"),
            Self::WrongObservationMethod => formatter.write_str("wrong observation method"),
            Self::NotExplicitlyInDomain(state) => {
                write!(formatter, "evaluation is not explicitly in-domain: {state:?}")
            }
            Self::WrongProperty { expected, actual } => {
                write!(formatter, "wrong property: expected {expected}, got {actual}")
            }
            Self::WrongUnit { expected, actual } => {
                write!(formatter, "wrong unit: expected {expected}, got {actual}")
            }
            Self::FormationEnergyNotNegative(value) => {
                write!(formatter, "formation energy is not negative: {value}")
            }
            Self::UnsupportedStage(stage) => {
                write!(formatter, "no computational admission contract for stage {stage:?}")
            }
            Self::AdmissionMismatch => {
                formatter.write_str("stored computational admission does not match exact evidence")
            }
        }
    }
}

impl Error for ComputationalAdmissionError {}

impl From<AuthorityFirewallError> for ComputationalAdmissionError {
    fn from(value: AuthorityFirewallError) -> Self {
        Self::Firewall(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conditioned_property::{
        ConditionedPropertyObservation, PropertyArtifactRef, PropertyConditions,
        PropertyUncertainty,
    };
    use crate::evaluation_seal::MultiFidelityEvaluationSeal;
    use crate::multi_fidelity::{
        ApplicabilityAssessment, EvaluationResourceCost, EvaluatorRef,
    };

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C64: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn artifact(source: &str, hash: &str) -> PropertyArtifactRef {
        PropertyArtifactRef {
            source_id: source.to_string(),
            artifact_sha256: hash.to_string(),
        }
    }

    fn surrogate(state: ApplicabilityState) -> MultiFidelityEvaluation {
        MultiFidelityEvaluation::new(
            EvaluatorRef {
                evaluator_id: "ml-fixture".to_string(),
                version: "1".to_string(),
                artifact_sha256: A64.to_string(),
            },
            EvaluationMethodClass::Surrogate,
            EvaluationOrigin::SurrogatePrediction,
            "ml-fixture".to_string(),
            ConditionedPropertyObservation {
                subject_identity: "material-subject:v1|fixture".to_string(),
                property_id: FORMATION_ENERGY_PROPERTY_ID.to_string(),
                value: -0.3,
                unit: EV_PER_ATOM_UNIT.to_string(),
                uncertainty: PropertyUncertainty::Unknown,
                conditions: PropertyConditions::default(),
                method: PropertyObservationMethod::ModelPrediction {
                    model_id: "ml-fixture".to_string(),
                    model_sha256: A64.to_string(),
                    applicability_domain_id: Some("ml-domain-v1".to_string()),
                },
                artifact: artifact("prediction", C64),
            },
            ApplicabilityAssessment {
                domain_id: "ml-domain-v1".to_string(),
                state,
                score: Some(0.1),
                artifact: Some(artifact("ood-basis", B64)),
            },
            None,
            vec![],
            EvaluationResourceCost::default(),
        )
        .unwrap()
    }

    fn local_dft(
        property_id: &str,
        unit: &str,
        value: f64,
        state: ApplicabilityState,
    ) -> MultiFidelityEvaluation {
        MultiFidelityEvaluation::new(
            EvaluatorRef {
                evaluator_id: "qe".to_string(),
                version: "7.x".to_string(),
                artifact_sha256: A64.to_string(),
            },
            EvaluationMethodClass::Dft,
            EvaluationOrigin::LocalReproduction,
            "dft-pbe".to_string(),
            ConditionedPropertyObservation {
                subject_identity: "material-subject:v1|fixture".to_string(),
                property_id: property_id.to_string(),
                value,
                unit: unit.to_string(),
                uncertainty: PropertyUncertainty::Unknown,
                conditions: PropertyConditions::default(),
                method: PropertyObservationMethod::Calculation {
                    method_id: "DFT-PBE".to_string(),
                    code_id: "qe".to_string(),
                    input_sha256: B64.to_string(),
                    output_sha256: C64.to_string(),
                },
                artifact: artifact("dft-result", C64),
            },
            ApplicabilityAssessment {
                domain_id: "dft-domain-v1".to_string(),
                state,
                score: None,
                artifact: None,
            },
            None,
            vec![],
            EvaluationResourceCost::default(),
        )
        .unwrap()
    }

    #[test]
    fn in_domain_surrogate_can_establish_ml_screened_only() {
        let evaluation = surrogate(ApplicabilityState::InDomain);
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        let admission = admit_ml_screening(&evaluation, &seal).unwrap();
        assert_eq!(admission.stage(), MaterialsEvidenceStage::MlScreened);
        assert_eq!(admission.kind(), MaterialsEvidenceKind::MachineLearningModel);
        admission.validate_against(&evaluation, &seal).unwrap();
    }

    #[test]
    fn out_of_domain_surrogate_cannot_establish_ml_screened() {
        let evaluation = surrogate(ApplicabilityState::OutOfDomain);
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        assert!(matches!(
            admit_ml_screening(&evaluation, &seal),
            Err(ComputationalAdmissionError::NotExplicitlyInDomain(
                ApplicabilityState::OutOfDomain
            ))
        ));
    }

    #[test]
    fn negative_in_domain_local_dft_formation_energy_can_be_admitted() {
        let evaluation = local_dft(
            FORMATION_ENERGY_PROPERTY_ID,
            EV_PER_ATOM_UNIT,
            -0.2,
            ApplicabilityState::InDomain,
        );
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        let admission = admit_negative_dft_formation_energy(&evaluation, &seal).unwrap();
        assert_eq!(
            admission.stage(),
            MaterialsEvidenceStage::DftFormationStable
        );
        assert_eq!(admission.kind(), MaterialsEvidenceKind::DftCalculation);
        admission.validate_against(&evaluation, &seal).unwrap();
    }

    #[test]
    fn out_of_domain_local_dft_cannot_advance_formation_stage() {
        let evaluation = local_dft(
            FORMATION_ENERGY_PROPERTY_ID,
            EV_PER_ATOM_UNIT,
            -0.2,
            ApplicabilityState::OutOfDomain,
        );
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        assert!(matches!(
            admit_negative_dft_formation_energy(&evaluation, &seal),
            Err(ComputationalAdmissionError::NotExplicitlyInDomain(
                ApplicabilityState::OutOfDomain
            ))
        ));
    }

    #[test]
    fn surrogate_with_negative_formation_number_cannot_impersonate_dft() {
        let evaluation = surrogate(ApplicabilityState::InDomain);
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        assert!(matches!(
            admit_negative_dft_formation_energy(&evaluation, &seal),
            Err(ComputationalAdmissionError::WrongOrigin { .. })
        ));
    }

    #[test]
    fn nonnegative_dft_formation_energy_is_refused() {
        let evaluation = local_dft(
            FORMATION_ENERGY_PROPERTY_ID,
            EV_PER_ATOM_UNIT,
            0.01,
            ApplicabilityState::InDomain,
        );
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        assert!(matches!(
            admit_negative_dft_formation_energy(&evaluation, &seal),
            Err(ComputationalAdmissionError::FormationEnergyNotNegative(_))
        ));
    }

    #[test]
    fn wrong_property_cannot_enter_dft_formation_stage() {
        let evaluation = local_dft("band_gap", "eV", -0.2, ApplicabilityState::InDomain);
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        assert!(matches!(
            admit_negative_dft_formation_energy(&evaluation, &seal),
            Err(ComputationalAdmissionError::WrongProperty { .. })
        ));
    }

    #[test]
    fn stored_admission_rejects_semantic_mutation() {
        let evaluation = local_dft(
            FORMATION_ENERGY_PROPERTY_ID,
            EV_PER_ATOM_UNIT,
            -0.2,
            ApplicabilityState::InDomain,
        );
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        let admission = admit_negative_dft_formation_energy(&evaluation, &seal).unwrap();
        let mut mutated = evaluation.clone();
        mutated.observation.value = -0.1;
        assert!(admission.validate_against(&mutated, &seal).is_err());
    }

    #[test]
    fn hull_and_phonon_stages_are_not_silently_supported() {
        let evaluation = local_dft(
            FORMATION_ENERGY_PROPERTY_ID,
            EV_PER_ATOM_UNIT,
            -0.2,
            ApplicabilityState::InDomain,
        );
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        let admission = admit_negative_dft_formation_energy(&evaluation, &seal).unwrap();

        let mut hull = admission.clone();
        hull.stage = MaterialsEvidenceStage::ConvexHullScreened;
        assert!(matches!(
            hull.validate_against(&evaluation, &seal),
            Err(ComputationalAdmissionError::UnsupportedStage(
                MaterialsEvidenceStage::ConvexHullScreened
            ))
        ));

        let mut phonon = admission;
        phonon.stage = MaterialsEvidenceStage::DynamicallyStable;
        assert!(matches!(
            phonon.validate_against(&evaluation, &seal),
            Err(ComputationalAdmissionError::UnsupportedStage(
                MaterialsEvidenceStage::DynamicallyStable
            ))
        ));
    }
}
