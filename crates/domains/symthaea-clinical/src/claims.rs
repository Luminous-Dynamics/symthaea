// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Conservative vocabulary for medical and clinical claims.
//!
//! These types deliberately separate **what kind of claim is being made** from
//! **what stage of evidence currently supports it**. They do not confer clinical
//! authority, regulatory status, diagnostic validity, treatment authority, or
//! permission to present an output to a clinician or patient.
//!
//! In particular, this module encodes the non-equivalences:
//!
//! - candidate signal != biomarker;
//! - association != causation;
//! - prediction != diagnosis;
//! - simulation != patient estimate;
//! - research result != clinical authority.

use serde::{Deserialize, Serialize};

/// Version of the clinical-claim vocabulary encoded by this module.
pub const CLINICAL_CLAIM_VOCABULARY_VERSION: u16 = 1;

/// The epistemic kind of a medical or clinical claim.
///
/// Variants are intentionally non-ordered. A caller must not infer that moving
/// "down" this enum is an automatic promotion path.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ClinicalClaimKind {
    /// A hypothesis-generating pattern or anomaly that merits further study.
    CandidateSignal,
    /// A statistical or observational association without a causal claim.
    Association,
    /// A forecast of a future or unobserved state under a stated model.
    Prediction,
    /// A quantified estimate of risk for a defined population and horizon.
    RiskEstimate,
    /// A proposed causal explanation that remains to be tested.
    CausalHypothesis,
    /// A causal-effect estimate produced under an explicit causal design.
    CausalEffectEstimate,
    /// Information intended to support, but not replace, diagnostic judgment.
    DiagnosticSupport,
    /// Information intended to support, but not replace, treatment judgment.
    TreatmentSupport,
}

/// The stage of evidence supporting a clinical claim.
///
/// This is a provenance/maturity label, not a quality score. Two artifacts at
/// the same stage may have radically different quality, bias, applicability,
/// calibration, and clinical relevance.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ClinicalEvidenceStage {
    /// Mechanistic, theoretical, or literature-grounded hypothesis only.
    MechanisticHypothesis,
    /// Demonstrated in synthetic, simulated, or otherwise non-clinical data.
    SyntheticDemonstration,
    /// Evaluated retrospectively on data used within the originating program.
    RetrospectiveInternal,
    /// Evaluated retrospectively on an independently held-out external dataset.
    RetrospectiveExternal,
    /// Run prospectively in shadow mode without controlling clinical action.
    ProspectiveShadow,
    /// Evaluated prospectively in an explicitly designed clinical study.
    ProspectiveClinicalStudy,
    /// Replicated by independent evidence under a defined intended use.
    ReplicatedClinicalEvidence,
}

/// The population applicability asserted for a claim.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ClinicalApplicability {
    /// Applicability has not yet been established for a human population.
    Unestablished,
    /// Applicable only to the exact evaluated cohort or dataset.
    EvaluatedCohortOnly,
    /// Proposed for a defined target population, pending stronger validation.
    DefinedTargetPopulation,
    /// Evidence exists for a defined target population under a stated use.
    ValidatedTargetPopulation,
}

/// Whether the artifact is research-only or intended to support a human
/// clinical workflow.
///
/// This field is descriptive only. It does not itself authorize presentation or
/// action; downstream assurance systems must decide that separately.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ClinicalIntendedUseClass {
    ResearchOnly,
    ClinicalDecisionSupport,
}

/// Minimal semantic label attached to a Symthaea clinical/research output.
///
/// This type is deliberately small. Evidence identities, model execution
/// identity, subject binding, uncertainty, missingness, and authorization belong
/// in evidence-bearing envelopes and downstream assurance layers rather than in
/// this vocabulary object.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ClinicalClaimSemanticsV1 {
    pub schema_version: u16,
    pub claim_kind: ClinicalClaimKind,
    pub evidence_stage: ClinicalEvidenceStage,
    pub applicability: ClinicalApplicability,
    pub intended_use: ClinicalIntendedUseClass,
}

impl ClinicalClaimSemanticsV1 {
    /// Construct a v1 semantics label.
    #[must_use]
    pub const fn new(
        claim_kind: ClinicalClaimKind,
        evidence_stage: ClinicalEvidenceStage,
        applicability: ClinicalApplicability,
        intended_use: ClinicalIntendedUseClass,
    ) -> Self {
        Self {
            schema_version: CLINICAL_CLAIM_VOCABULARY_VERSION,
            claim_kind,
            evidence_stage,
            applicability,
            intended_use,
        }
    }

    /// Returns true when the claim is explicitly research-only.
    ///
    /// This convenience method must not be inverted into an authorization rule:
    /// `false` means only that the declared intended-use class is clinical
    /// decision support, not that the output is safe, validated, or permitted.
    #[must_use]
    pub const fn is_research_only(&self) -> bool {
        matches!(self.intended_use, ClinicalIntendedUseClass::ResearchOnly)
    }

    /// Validate only the schema-level invariants owned by this vocabulary.
    ///
    /// Clinical validity and authorization are intentionally out of scope.
    pub const fn validate_schema(&self) -> Result<(), ClinicalClaimVocabularyError> {
        if self.schema_version != CLINICAL_CLAIM_VOCABULARY_VERSION {
            return Err(ClinicalClaimVocabularyError::UnsupportedSchemaVersion {
                found: self.schema_version,
                expected: CLINICAL_CLAIM_VOCABULARY_VERSION,
            });
        }
        Ok(())
    }
}

/// Schema errors for [`ClinicalClaimSemanticsV1`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClinicalClaimVocabularyError {
    UnsupportedSchemaVersion { found: u16, expected: u16 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_pins_v1_schema() {
        let semantics = ClinicalClaimSemanticsV1::new(
            ClinicalClaimKind::CandidateSignal,
            ClinicalEvidenceStage::RetrospectiveInternal,
            ClinicalApplicability::EvaluatedCohortOnly,
            ClinicalIntendedUseClass::ResearchOnly,
        );

        assert_eq!(semantics.schema_version, CLINICAL_CLAIM_VOCABULARY_VERSION);
        assert!(semantics.validate_schema().is_ok());
        assert!(semantics.is_research_only());
    }

    #[test]
    fn unsupported_schema_fails_closed() {
        let semantics = ClinicalClaimSemanticsV1 {
            schema_version: CLINICAL_CLAIM_VOCABULARY_VERSION + 1,
            claim_kind: ClinicalClaimKind::Prediction,
            evidence_stage: ClinicalEvidenceStage::ProspectiveShadow,
            applicability: ClinicalApplicability::DefinedTargetPopulation,
            intended_use: ClinicalIntendedUseClass::ClinicalDecisionSupport,
        };

        assert_eq!(
            semantics.validate_schema(),
            Err(ClinicalClaimVocabularyError::UnsupportedSchemaVersion {
                found: CLINICAL_CLAIM_VOCABULARY_VERSION + 1,
                expected: CLINICAL_CLAIM_VOCABULARY_VERSION,
            })
        );
    }

    #[test]
    fn serde_round_trip_preserves_semantics() {
        let semantics = ClinicalClaimSemanticsV1::new(
            ClinicalClaimKind::CausalHypothesis,
            ClinicalEvidenceStage::RetrospectiveExternal,
            ClinicalApplicability::DefinedTargetPopulation,
            ClinicalIntendedUseClass::ResearchOnly,
        );

        let json = serde_json::to_string(&semantics).expect("serialize clinical claim semantics");
        let decoded: ClinicalClaimSemanticsV1 =
            serde_json::from_str(&json).expect("deserialize clinical claim semantics");

        assert_eq!(decoded, semantics);
    }

    #[test]
    fn vocabulary_does_not_encode_automatic_promotion_order() {
        let signal = ClinicalClaimKind::CandidateSignal;
        let causal = ClinicalClaimKind::CausalEffectEstimate;

        // Equality is available; ordering is intentionally not implemented.
        assert_ne!(signal, causal);
    }
}
