// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical epistemic contracts for multiscale matter research.
//!
//! These types deliberately separate three questions that are easy to conflate:
//!
//! 1. **What scale of matter is the claim about?** (`MatterScale`)
//! 2. **What scientific adjudication step has actually occurred?**
//!    (`MatterValidationStage`)
//! 3. **What is the canonical epistemic position of the claim?**
//!    (`EpistemicCoordinate`)
//!
//! `MatterValidationStage` is descriptive rather than an authority score. In
//! particular, a local simulation, a reference benchmark, and an experiment
//! are different evidence shapes; callers must not infer E/N/M coordinates by
//! ordinal comparison of stages.

use crate::{
    EmpiricalLevel, EpistemicContext, EpistemicCoordinate, EpistemicProvenance, MaterialityLevel,
    NormativeLevel,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Physical scale addressed by a matter claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatterScale {
    Nuclear,
    AtomicElectronic,
    Molecular,
    Crystal,
    DefectMicrostructure,
    Mesoscale,
    Metamaterial,
    Device,
    Multiscale,
}

/// Scientific adjudication step reached by a matter claim.
///
/// This is intentionally **not** a total evidence-strength ordering. A
/// `ReferenceBenchmarked` result and a `HighFidelitySimulated` result answer
/// different questions and neither silently upgrades the canonical E/N/M
/// coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatterValidationStage {
    Hypothesis,
    SurrogateSupported,
    PhysicsSimulated,
    CrossModelSupported,
    HighFidelitySimulated,
    ReferenceBenchmarked,
    ExperimentallyObserved,
    IndependentlyReplicated,
}

impl MatterValidationStage {
    /// Whether this stage can honestly be produced by a local computational
    /// solver without an experimental observation or independent replication.
    pub fn is_local_computational(self) -> bool {
        matches!(
            self,
            Self::Hypothesis
                | Self::SurrogateSupported
                | Self::PhysicsSimulated
                | Self::CrossModelSupported
                | Self::HighFidelitySimulated
                | Self::ReferenceBenchmarked
        )
    }
}

/// Physics that a solver explicitly advertises as implemented.
///
/// A missing capability is a hard absence, not an invitation for callers to
/// infer it from a nearby method name. This is especially important for heavy
/// elements, where non-relativistic electronic structure must not silently
/// stand in for scalar-relativistic, spin-orbit, or four-component treatment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatterSolverCapability {
    NuclearMass,
    NuclearSeparationEnergy,
    NuclearDecay,
    NuclearFission,
    NonRelativisticElectronicStructure,
    ScalarRelativisticElectronicStructure,
    SpinOrbitElectronicStructure,
    FourComponentRelativisticElectronicStructure,
    RestrictedHartreeFock,
    UnrestrictedHartreeFock,
    DensityFunctionalTheory,
    PostHartreeFockCorrelation,
    ExcitedStates,
    MolecularGradients,
    MolecularVibrations,
    PeriodicElectronicStructure,
    AtomisticDynamics,
    ContinuumMechanics,
    Electromagnetism,
    Acoustics,
    ThermalTransport,
    CoupledMultiphysics,
    Manufacturability,
    ExperimentalMeasurement,
}

/// Exact solver identity and the physics it claims to implement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatterSolverProfile {
    pub name: String,
    pub version: String,
    pub method: String,
    pub capabilities: Vec<MatterSolverCapability>,
    /// Known scope limitations and unresolved discrepancies.
    pub limitations: Vec<String>,
}

impl MatterSolverProfile {
    pub fn supports(&self, capability: MatterSolverCapability) -> bool {
        self.capabilities.contains(&capability)
    }

    pub fn supports_any_relativistic_electronic_structure(&self) -> bool {
        self.supports(MatterSolverCapability::ScalarRelativisticElectronicStructure)
            || self.supports(MatterSolverCapability::SpinOrbitElectronicStructure)
            || self.supports(MatterSolverCapability::FourComponentRelativisticElectronicStructure)
    }
}

/// Explicit uncertainty attached to one predicted quantity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatterUncertainty {
    pub value: f64,
    pub unit: String,
    pub method: String,
    /// True when the producer itself identifies the target as out-of-domain.
    pub out_of_domain: bool,
}

impl MatterUncertainty {
    pub fn new(
        value: f64,
        unit: impl Into<String>,
        method: impl Into<String>,
        out_of_domain: bool,
    ) -> Result<Self, MatterClaimError> {
        if !value.is_finite() || value < 0.0 {
            return Err(MatterClaimError::InvalidUncertainty);
        }
        Ok(Self {
            value,
            unit: unit.into(),
            method: method.into(),
            out_of_domain,
        })
    }
}

/// Reference to evidence retained outside the claim object itself.
///
/// The reference is deliberately generic so callers can point at a
/// `symthaea-evidence-plane` run, a frozen dataset snapshot, an external
/// solver receipt, or an experimental record without introducing a dependency
/// cycle from the canonical epistemic-types crate into those producers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatterEvidenceRef {
    pub evidence_id: String,
    pub run_id: Option<String>,
    pub config_hash: Option<String>,
    pub dataset_ids: Vec<String>,
}

/// Canonical multiscale matter claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatterClaim {
    pub claim_id: String,
    pub statement: String,
    pub scale: MatterScale,
    pub validation_stage: MatterValidationStage,
    pub epistemic: EpistemicCoordinate,
    pub provenance: EpistemicProvenance,
    pub solver: Option<MatterSolverProfile>,
    pub uncertainty: Option<MatterUncertainty>,
    pub evidence: Vec<MatterEvidenceRef>,
    /// Observations/calculations that would count against this claim.
    pub falsifiers: Vec<String>,
    /// Scope limits that must travel with the claim.
    pub limitations: Vec<String>,
}

impl MatterClaim {
    /// Construct a claim produced by one local computational method.
    ///
    /// This constructor intentionally fixes the canonical epistemic coordinate
    /// at E1/N0/M0 in scientific context. A high numerical confidence, a
    /// sophisticated solver, open-source code, or a reference comparison does
    /// not by itself establish independent reproduction.
    pub fn local_computational(
        claim_id: impl Into<String>,
        statement: impl Into<String>,
        scale: MatterScale,
        validation_stage: MatterValidationStage,
        solver: MatterSolverProfile,
    ) -> Result<Self, MatterClaimError> {
        let claim_id = claim_id.into();
        let statement = statement.into();
        if claim_id.trim().is_empty() {
            return Err(MatterClaimError::EmptyClaimId);
        }
        if statement.trim().is_empty() {
            return Err(MatterClaimError::EmptyStatement);
        }
        if !validation_stage.is_local_computational() {
            return Err(MatterClaimError::ObservationRequired);
        }

        Ok(Self {
            claim_id,
            statement,
            scale,
            validation_stage,
            epistemic: EpistemicCoordinate::new(
                EmpiricalLevel::E1Testimonial,
                NormativeLevel::N0Personal,
                MaterialityLevel::M0Ephemeral,
                EpistemicContext::Scientific,
            ),
            provenance: EpistemicProvenance::Inferred {
                method: solver.method.clone(),
                // Confidence here describes the deterministic production event,
                // not truth authority; E/N/M carries the epistemic boundary.
                confidence: 1.0,
            },
            solver: Some(solver),
            uncertainty: None,
            evidence: Vec::new(),
            falsifiers: Vec::new(),
            limitations: Vec::new(),
        })
    }

    pub fn with_uncertainty(mut self, uncertainty: MatterUncertainty) -> Self {
        self.uncertainty = Some(uncertainty);
        self
    }

    pub fn with_evidence(mut self, evidence: MatterEvidenceRef) -> Self {
        self.evidence.push(evidence);
        self
    }

    pub fn with_falsifier(mut self, falsifier: impl Into<String>) -> Self {
        self.falsifiers.push(falsifier.into());
        self
    }

    pub fn with_limitation(mut self, limitation: impl Into<String>) -> Self {
        self.limitations.push(limitation.into());
        self
    }

    pub fn is_experimental(&self) -> bool {
        matches!(
            self.validation_stage,
            MatterValidationStage::ExperimentallyObserved
                | MatterValidationStage::IndependentlyReplicated
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatterClaimError {
    EmptyClaimId,
    EmptyStatement,
    InvalidUncertainty,
    ObservationRequired,
}

impl fmt::Display for MatterClaimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyClaimId => write!(f, "matter claim id must not be empty"),
            Self::EmptyStatement => write!(f, "matter claim statement must not be empty"),
            Self::InvalidUncertainty => {
                write!(f, "matter uncertainty must be finite and non-negative")
            }
            Self::ObservationRequired => write!(
                f,
                "experimental/replication stages cannot be minted by a local computational constructor"
            ),
        }
    }
}

impl std::error::Error for MatterClaimError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn nonrel_solver() -> MatterSolverProfile {
        MatterSolverProfile {
            name: "test-qc".into(),
            version: "0.1".into(),
            method: "RHF/STO-3G".into(),
            capabilities: vec![
                MatterSolverCapability::NonRelativisticElectronicStructure,
                MatterSolverCapability::RestrictedHartreeFock,
            ],
            limitations: vec!["no relativistic treatment".into()],
        }
    }

    #[test]
    fn local_computation_cannot_mint_high_empirical_authority() {
        let claim = MatterClaim::local_computational(
            "qc:h2",
            "local H2 energy calculation",
            MatterScale::Molecular,
            MatterValidationStage::ReferenceBenchmarked,
            nonrel_solver(),
        )
        .unwrap();

        assert_eq!(claim.epistemic.empirical, EmpiricalLevel::E1Testimonial);
        assert_eq!(claim.epistemic.normative, NormativeLevel::N0Personal);
        assert_eq!(claim.epistemic.materiality, MaterialityLevel::M0Ephemeral);
        assert!(!claim.is_experimental());
    }

    #[test]
    fn local_computation_cannot_claim_experiment_or_replication() {
        for stage in [
            MatterValidationStage::ExperimentallyObserved,
            MatterValidationStage::IndependentlyReplicated,
        ] {
            assert_eq!(
                MatterClaim::local_computational(
                    "bad",
                    "should fail",
                    MatterScale::Molecular,
                    stage,
                    nonrel_solver(),
                )
                .unwrap_err(),
                MatterClaimError::ObservationRequired
            );
        }
    }

    #[test]
    fn heavy_element_capability_is_not_inferred_from_nonrelativistic_qc() {
        let solver = nonrel_solver();
        assert!(solver.supports(MatterSolverCapability::NonRelativisticElectronicStructure));
        assert!(!solver.supports_any_relativistic_electronic_structure());
        assert!(!solver.supports(MatterSolverCapability::SpinOrbitElectronicStructure));
    }

    #[test]
    fn uncertainty_fails_closed_on_nonfinite_or_negative_values() {
        assert_eq!(
            MatterUncertainty::new(f64::NAN, "MeV", "tree spread", true).unwrap_err(),
            MatterClaimError::InvalidUncertainty
        );
        assert_eq!(
            MatterUncertainty::new(-0.1, "MeV", "tree spread", false).unwrap_err(),
            MatterClaimError::InvalidUncertainty
        );
    }

    #[test]
    fn matter_claim_round_trips_through_json() {
        let claim = MatterClaim::local_computational(
            "nuclear:fe56",
            "DZ10 binding-energy prediction for Fe-56",
            MatterScale::Nuclear,
            MatterValidationStage::ReferenceBenchmarked,
            MatterSolverProfile {
                name: "symthaea-nuclear".into(),
                version: "0.1.0".into(),
                method: "DZ10".into(),
                capabilities: vec![MatterSolverCapability::NuclearMass],
                limitations: Vec::new(),
            },
        )
        .unwrap()
        .with_uncertainty(MatterUncertainty::new(0.67, "MeV", "CV RMS", false).unwrap())
        .with_falsifier("new mass measurement outside the declared error model");

        let json = serde_json::to_string(&claim).unwrap();
        let decoded: MatterClaim = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, claim);
    }
}
