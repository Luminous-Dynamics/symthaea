// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Relativistic electronic-structure admission for Matter Observatory claims.
//!
//! Heavy-element chemistry cannot be qualified by the presence of a generic
//! quantum-chemistry result.  This module requires the claim to name the exact
//! relativistic treatment it needs and requires the solver profile to advertise
//! that exact capability.  No capability is inferred from atomic number, method
//! naming, solver reputation, or from a stronger-sounding neighboring flag.
//!
//! An admitted receipt is still a local computational claim.  It remains E1/N0/M0
//! under the canonical Matter Observatory constructor and cannot mint experiment
//! or independent-replication authority.

use std::collections::BTreeSet;
use std::fmt;

use symthaea_epistemic_types::{
    MatterClaim, MatterClaimError, MatterEvidenceRef, MatterScale, MatterSolverCapability,
    MatterSolverProfile, MatterValidationStage,
};

/// Exact relativistic physics required by one electronic-structure claim.
///
/// Requirements are intentionally not a strength ladder.  The admission rule
/// checks the corresponding capability token exactly; a four-component profile
/// does not silently satisfy `SpinOrbitResolved` unless it also explicitly
/// advertises `SpinOrbitElectronicStructure`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelativisticElectronicRequirement {
    ScalarRelativistic,
    SpinOrbitResolved,
    FourComponentDirac,
}

impl RelativisticElectronicRequirement {
    pub fn required_capability(self) -> MatterSolverCapability {
        match self {
            Self::ScalarRelativistic => {
                MatterSolverCapability::ScalarRelativisticElectronicStructure
            }
            Self::SpinOrbitResolved => MatterSolverCapability::SpinOrbitElectronicStructure,
            Self::FourComponentDirac => {
                MatterSolverCapability::FourComponentRelativisticElectronicStructure
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ScalarRelativistic => "scalar-relativistic",
            Self::SpinOrbitResolved => "spin-orbit-resolved",
            Self::FourComponentDirac => "four-component-Dirac",
        }
    }
}

/// Execution/provenance receipt supplied by a relativistic electronic-structure
/// producer.  This type records identity; it does not authenticate the producer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelativisticElectronicStructureReceipt {
    pub receipt_id: String,
    pub solver: MatterSolverProfile,
    pub requirement: RelativisticElectronicRequirement,
    /// Exact elements present in the target system, expressed as unique Z values.
    pub element_atomic_numbers: Vec<u16>,
    pub hamiltonian_id: String,
    pub basis_or_pseudopotential_id: String,
    pub input_identity: String,
    pub output_identity: String,
    pub validation_stage: MatterValidationStage,
    /// Required when the stage is `ReferenceBenchmarked`.
    pub reference_dataset_ids: Vec<String>,
    pub limitations: Vec<String>,
}

impl RelativisticElectronicStructureReceipt {
    pub fn validate(&self) -> Result<(), RelativisticAdmissionError> {
        if self.receipt_id.trim().is_empty() {
            return Err(RelativisticAdmissionError::EmptyReceiptId);
        }
        if self.input_identity.trim().is_empty() {
            return Err(RelativisticAdmissionError::EmptyInputIdentity);
        }
        if self.output_identity.trim().is_empty() {
            return Err(RelativisticAdmissionError::EmptyOutputIdentity);
        }
        if self.hamiltonian_id.trim().is_empty() {
            return Err(RelativisticAdmissionError::EmptyHamiltonianIdentity);
        }
        if self.basis_or_pseudopotential_id.trim().is_empty() {
            return Err(RelativisticAdmissionError::EmptyBasisIdentity);
        }
        if self.element_atomic_numbers.is_empty() {
            return Err(RelativisticAdmissionError::EmptyElementSet);
        }
        if self.element_atomic_numbers.iter().any(|z| *z == 0) {
            return Err(RelativisticAdmissionError::InvalidAtomicNumber);
        }

        let mut unique = BTreeSet::new();
        for z in &self.element_atomic_numbers {
            if !unique.insert(*z) {
                return Err(RelativisticAdmissionError::DuplicateAtomicNumber(*z));
            }
        }

        if !self.validation_stage.is_local_computational() {
            return Err(RelativisticAdmissionError::NonLocalValidationStage);
        }

        if self.validation_stage == MatterValidationStage::ReferenceBenchmarked
            && self.reference_dataset_ids.is_empty()
        {
            return Err(RelativisticAdmissionError::BenchmarkWithoutReference);
        }
        if self
            .reference_dataset_ids
            .iter()
            .any(|reference| reference.trim().is_empty())
        {
            return Err(RelativisticAdmissionError::EmptyReferenceIdentity);
        }

        let required = self.requirement.required_capability();
        if !self.solver.supports(required) {
            return Err(RelativisticAdmissionError::MissingRequiredCapability {
                required,
            });
        }

        Ok(())
    }
}

/// Result of admitting one receipt into the canonical Matter Observatory plane.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmittedRelativisticMatterClaim {
    pub receipt: RelativisticElectronicStructureReceipt,
    pub claim: MatterClaim,
}

/// Admit a local relativistic electronic-structure execution into a canonical
/// matter claim without upgrading its empirical authority.
pub fn admit_relativistic_electronic_claim(
    claim_id: impl Into<String>,
    statement: impl Into<String>,
    scale: MatterScale,
    receipt: RelativisticElectronicStructureReceipt,
) -> Result<AdmittedRelativisticMatterClaim, RelativisticAdmissionError> {
    receipt.validate()?;

    if !matches!(
        scale,
        MatterScale::AtomicElectronic | MatterScale::Molecular | MatterScale::Crystal
    ) {
        return Err(RelativisticAdmissionError::InvalidElectronicStructureScale);
    }

    let claim = MatterClaim::local_computational(
        claim_id,
        statement,
        scale,
        receipt.validation_stage,
        receipt.solver.clone(),
    )
    .map_err(RelativisticAdmissionError::MatterClaim)?
    .with_evidence(MatterEvidenceRef {
        evidence_id: receipt.receipt_id.clone(),
        run_id: None,
        config_hash: None,
        dataset_ids: receipt.reference_dataset_ids.clone(),
    })
    .with_falsifier(format!(
        "an independently reproduced {} calculation with the same Hamiltonian/basis target disagrees beyond a preregistered tolerance",
        receipt.requirement.label()
    ))
    .with_limitation(format!(
        "local {} execution is computational evidence, not experimental observation or independent replication",
        receipt.requirement.label()
    ))
    .with_limitation(format!(
        "input identity: {}; output identity: {}; Hamiltonian: {}; basis/pseudopotential: {}",
        receipt.input_identity,
        receipt.output_identity,
        receipt.hamiltonian_id,
        receipt.basis_or_pseudopotential_id
    ));

    let mut claim = claim;
    for limitation in receipt.solver.limitations.iter().chain(receipt.limitations.iter()) {
        claim = claim.with_limitation(limitation.clone());
    }

    Ok(AdmittedRelativisticMatterClaim { receipt, claim })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelativisticAdmissionError {
    EmptyReceiptId,
    EmptyInputIdentity,
    EmptyOutputIdentity,
    EmptyHamiltonianIdentity,
    EmptyBasisIdentity,
    EmptyElementSet,
    InvalidAtomicNumber,
    DuplicateAtomicNumber(u16),
    NonLocalValidationStage,
    BenchmarkWithoutReference,
    EmptyReferenceIdentity,
    MissingRequiredCapability {
        required: MatterSolverCapability,
    },
    InvalidElectronicStructureScale,
    MatterClaim(MatterClaimError),
}

impl fmt::Display for RelativisticAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyReceiptId => write!(f, "relativistic receipt id must not be empty"),
            Self::EmptyInputIdentity => write!(f, "relativistic input identity must not be empty"),
            Self::EmptyOutputIdentity => write!(f, "relativistic output identity must not be empty"),
            Self::EmptyHamiltonianIdentity => {
                write!(f, "relativistic Hamiltonian identity must not be empty")
            }
            Self::EmptyBasisIdentity => {
                write!(f, "basis/pseudopotential identity must not be empty")
            }
            Self::EmptyElementSet => write!(f, "electronic-structure element set must not be empty"),
            Self::InvalidAtomicNumber => write!(f, "atomic number zero is invalid"),
            Self::DuplicateAtomicNumber(z) => {
                write!(f, "element atomic number Z={z} appears more than once")
            }
            Self::NonLocalValidationStage => write!(
                f,
                "a local relativistic receipt cannot mint experimental or independent-replication status"
            ),
            Self::BenchmarkWithoutReference => write!(
                f,
                "ReferenceBenchmarked relativistic evidence requires an explicit reference dataset identity"
            ),
            Self::EmptyReferenceIdentity => {
                write!(f, "reference dataset identities must not be empty")
            }
            Self::MissingRequiredCapability { required } => write!(
                f,
                "solver profile does not explicitly advertise required capability {required:?}"
            ),
            Self::InvalidElectronicStructureScale => write!(
                f,
                "relativistic electronic-structure admission is limited to atomic/electronic, molecular, or crystal scales"
            ),
            Self::MatterClaim(error) => write!(f, "canonical matter claim rejected: {error}"),
        }
    }
}

impl std::error::Error for RelativisticAdmissionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_types::{EmpiricalLevel, MaterialityLevel, NormativeLevel};

    fn profile(capabilities: Vec<MatterSolverCapability>) -> MatterSolverProfile {
        MatterSolverProfile {
            name: "external-qc-test".to_string(),
            version: "1.0".to_string(),
            method: "explicit test electronic structure".to_string(),
            capabilities,
            limitations: vec!["test profile only".to_string()],
        }
    }

    fn receipt(
        requirement: RelativisticElectronicRequirement,
        capabilities: Vec<MatterSolverCapability>,
    ) -> RelativisticElectronicStructureReceipt {
        RelativisticElectronicStructureReceipt {
            receipt_id: "receipt:qc:og2".to_string(),
            solver: profile(capabilities),
            requirement,
            element_atomic_numbers: vec![8, 118],
            hamiltonian_id: "x2c-test-v1".to_string(),
            basis_or_pseudopotential_id: "basis:test-og-o".to_string(),
            input_identity: "sha256:input".to_string(),
            output_identity: "sha256:output".to_string(),
            validation_stage: MatterValidationStage::HighFidelitySimulated,
            reference_dataset_ids: Vec::new(),
            limitations: vec!["no experimental comparison".to_string()],
        }
    }

    #[test]
    fn nonrelativistic_profile_cannot_satisfy_scalar_requirement() {
        let candidate = receipt(
            RelativisticElectronicRequirement::ScalarRelativistic,
            vec![MatterSolverCapability::NonRelativisticElectronicStructure],
        );
        assert!(matches!(
            candidate.validate(),
            Err(RelativisticAdmissionError::MissingRequiredCapability {
                required: MatterSolverCapability::ScalarRelativisticElectronicStructure
            })
        ));
    }

    #[test]
    fn neighboring_relativistic_capability_is_not_inferred() {
        let candidate = receipt(
            RelativisticElectronicRequirement::SpinOrbitResolved,
            vec![MatterSolverCapability::FourComponentRelativisticElectronicStructure],
        );
        assert!(matches!(
            candidate.validate(),
            Err(RelativisticAdmissionError::MissingRequiredCapability {
                required: MatterSolverCapability::SpinOrbitElectronicStructure
            })
        ));
    }

    #[test]
    fn exact_relativistic_capability_admits_but_stays_e1_n0_m0() {
        let candidate = receipt(
            RelativisticElectronicRequirement::SpinOrbitResolved,
            vec![
                MatterSolverCapability::SpinOrbitElectronicStructure,
                MatterSolverCapability::DensityFunctionalTheory,
            ],
        );
        let admitted = admit_relativistic_electronic_claim(
            "qc:og2:spin-orbit",
            "spin-orbit-resolved electronic-structure calculation for an Og/O system",
            MatterScale::Molecular,
            candidate,
        )
        .unwrap();

        assert_eq!(admitted.claim.epistemic.empirical, EmpiricalLevel::E1Testimonial);
        assert_eq!(admitted.claim.epistemic.normative, NormativeLevel::N0Personal);
        assert_eq!(admitted.claim.epistemic.materiality, MaterialityLevel::M0Ephemeral);
        assert!(!admitted.claim.is_experimental());
        assert_eq!(admitted.claim.evidence.len(), 1);
    }

    #[test]
    fn reference_benchmark_requires_reference_identity() {
        let mut candidate = receipt(
            RelativisticElectronicRequirement::ScalarRelativistic,
            vec![MatterSolverCapability::ScalarRelativisticElectronicStructure],
        );
        candidate.validation_stage = MatterValidationStage::ReferenceBenchmarked;
        assert_eq!(
            candidate.validate().unwrap_err(),
            RelativisticAdmissionError::BenchmarkWithoutReference
        );
    }

    #[test]
    fn local_receipt_cannot_claim_experiment() {
        let mut candidate = receipt(
            RelativisticElectronicRequirement::ScalarRelativistic,
            vec![MatterSolverCapability::ScalarRelativisticElectronicStructure],
        );
        candidate.validation_stage = MatterValidationStage::ExperimentallyObserved;
        assert_eq!(
            candidate.validate().unwrap_err(),
            RelativisticAdmissionError::NonLocalValidationStage
        );
    }

    #[test]
    fn duplicate_element_identity_fails_closed() {
        let mut candidate = receipt(
            RelativisticElectronicRequirement::FourComponentDirac,
            vec![MatterSolverCapability::FourComponentRelativisticElectronicStructure],
        );
        candidate.element_atomic_numbers = vec![118, 118];
        assert_eq!(
            candidate.validate().unwrap_err(),
            RelativisticAdmissionError::DuplicateAtomicNumber(118)
        );
    }
}
