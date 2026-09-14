// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Matter Observatory integration adapters.
//!
//! This module sits above domain solvers and translates their local outputs into
//! canonical epistemic matter claims. It deliberately does not live inside the
//! solver crates: the solver computes physics; this layer interprets what that
//! computation is allowed to claim.
//!
//! Quantum integration is feature-gated with the existing
//! `quantum-consciousness` root feature. The current quantum solver is explicitly
//! non-relativistic, so heavy/superheavy electronic-structure claims cannot use
//! this profile to satisfy relativistic capability requirements.

#[cfg(feature = "quantum-consciousness")]
use symthaea_epistemic_types::{
    MatterClaim, MatterClaimError, MatterScale, MatterSolverCapability, MatterSolverProfile,
    MatterValidationStage,
};
#[cfg(feature = "quantum-consciousness")]
use symthaea_quantum_chemistry::validation::ValidationResult;

/// Conservative capability profile for the currently shipped
/// `symthaea-quantum-chemistry` v0.1.0 implementation.
///
/// A capability appears here only when the domain crate exposes a real callable
/// implementation. Known incompleteness stays in `limitations`; it never becomes
/// an inferred capability merely because a related module or method name exists.
#[cfg(feature = "quantum-consciousness")]
pub fn current_quantum_solver_profile() -> MatterSolverProfile {
    MatterSolverProfile {
        name: "symthaea-quantum-chemistry".to_string(),
        version: "0.1.0".to_string(),
        method: "Gaussian-basis non-relativistic ab initio electronic structure".to_string(),
        capabilities: vec![
            MatterSolverCapability::NonRelativisticElectronicStructure,
            MatterSolverCapability::RestrictedHartreeFock,
            MatterSolverCapability::UnrestrictedHartreeFock,
            MatterSolverCapability::DensityFunctionalTheory,
            MatterSolverCapability::PostHartreeFockCorrelation,
            MatterSolverCapability::ExcitedStates,
            MatterSolverCapability::MolecularGradients,
            MatterSolverCapability::MolecularVibrations,
        ],
        limitations: vec![
            "no scalar-relativistic, spin-orbit, or four-component Dirac electronic-structure treatment"
                .to_string(),
            "Kohn-Sham SCF currently supports LDA only; PBE exchange is post-hoc/non-self-consistent and PBE correlation is not implemented"
                .to_string(),
            "CCD exists as experimental/non-production code and is not advertised as a supported solver capability"
                .to_string(),
            "known unresolved HF benchmark discrepancies remain for N2/STO-3G and H2O/CH4 with 6-31G"
                .to_string(),
            "scope is small-to-moderate molecules; periodic electronic structure is not implemented"
                .to_string(),
        ],
    }
}

/// Package one local published-reference Hartree-Fock comparison into the
/// canonical Matter Observatory contract.
///
/// The result remains `E1/N0/M0` because a local benchmark execution is not an
/// independent reproduction. A legacy pass flag is retained as an observation,
/// but a broad tolerance cannot erase a large absolute error.
#[cfg(feature = "quantum-consciousness")]
pub fn hf_reference_validation_claim(
    result: &ValidationResult,
) -> Result<MatterClaim, MatterClaimError> {
    let statement = format!(
        "HF reference comparison for {} / {}: computed {:.10} Ha, reference {:.10} Ha, error {:+.6} Ha ({:+.3} kcal/mol), tolerance {:.6} Ha, pass={}",
        result.molecule,
        result.basis,
        result.computed,
        result.reference,
        result.error_hartree,
        result.error_kcal,
        result.tolerance,
        result.pass,
    );

    let mut claim = MatterClaim::local_computational(
        format!("quantum:hf:{}:{}", result.basis, result.molecule),
        statement,
        MatterScale::Molecular,
        MatterValidationStage::ReferenceBenchmarked,
        current_quantum_solver_profile(),
    )?
    .with_falsifier(
        "an independently reproduced calculation with the same declared method/basis disagrees beyond the preregistered tolerance",
    )
    .with_limitation(
        "local comparison with a published reference is benchmark evidence, not independent reproduction",
    );

    if !result.pass {
        claim = claim.with_limitation("this benchmark exceeds its currently declared tolerance");
    }

    // The existing validation corpus intentionally contains broad historical
    // tolerances. Preserve `pass`, but never let it masquerade as chemical
    // accuracy when the absolute discrepancy itself is large.
    if result.error_kcal.abs() > 5.0 {
        claim = claim.with_limitation(format!(
            "absolute benchmark error is {:.3} kcal/mol; do not interpret the legacy pass flag as chemical-accuracy qualification",
            result.error_kcal.abs()
        ));
    }

    Ok(claim)
}

#[cfg(all(test, feature = "quantum-consciousness"))]
mod tests {
    use super::*;
    use symthaea_epistemic_types::{EmpiricalLevel, MatterValidationStage};

    #[test]
    fn current_profile_does_not_claim_relativistic_or_periodic_physics() {
        let profile = current_quantum_solver_profile();
        assert!(profile.supports(MatterSolverCapability::NonRelativisticElectronicStructure));
        assert!(!profile.supports_any_relativistic_electronic_structure());
        assert!(!profile.supports(MatterSolverCapability::PeriodicElectronicStructure));
        assert!(!profile.supports(MatterSolverCapability::ExperimentalMeasurement));
    }

    #[test]
    fn local_reference_benchmark_stays_e1_and_nonexperimental() {
        let result = ValidationResult {
            molecule: "H2".to_string(),
            basis: "STO-3G".to_string(),
            computed: -1.1170,
            reference: -1.1175,
            error_hartree: 0.0005,
            error_kcal: 0.3137545,
            pass: true,
            tolerance: 0.01,
        };

        let claim = hf_reference_validation_claim(&result).unwrap();
        assert_eq!(claim.validation_stage, MatterValidationStage::ReferenceBenchmarked);
        assert_eq!(claim.epistemic.empirical, EmpiricalLevel::E1Testimonial);
        assert!(!claim.is_experimental());
    }

    #[test]
    fn broad_legacy_tolerance_does_not_hide_large_error() {
        let result = ValidationResult {
            molecule: "N2".to_string(),
            basis: "STO-3G".to_string(),
            computed: -106.7664,
            reference: -107.4964,
            error_hartree: 0.73,
            error_kcal: 458.08157,
            pass: true,
            tolerance: 1.0,
        };

        let claim = hf_reference_validation_claim(&result).unwrap();
        assert!(claim.limitations.iter().any(|limit| {
            limit.contains("do not interpret the legacy pass flag as chemical-accuracy qualification")
        }));
    }
}
