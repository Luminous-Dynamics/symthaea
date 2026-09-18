// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evidence-bound clinical inference envelopes for Symthaea.
//!
//! This module defines the upstream contract for clinical/research inferences.
//! It intentionally carries provenance, model/execution identity, evidence,
//! uncertainty, missingness, alternatives, and distribution-shift state while
//! carrying **no clinical authority**. Downstream systems such as Mycelix may
//! independently verify and promote an envelope under their own policies.

use crate::claims::{
    ClinicalClaimSemanticsV1, ClinicalClaimVocabularyError, ClinicalIntendedUseClass,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Version of the inference envelope wire contract.
pub const CLINICAL_INFERENCE_ENVELOPE_VERSION: u16 = 1;

/// Digest algorithm supported by the v1 envelope.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ClinicalDigestAlgorithmV1 {
    Blake3_256,
}

/// Fixed-width artifact identity used by the v1 envelope.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ClinicalDigestV1 {
    pub algorithm: ClinicalDigestAlgorithmV1,
    pub value: [u8; 32],
}

impl ClinicalDigestV1 {
    #[must_use]
    pub const fn blake3(value: [u8; 32]) -> Self {
        Self {
            algorithm: ClinicalDigestAlgorithmV1::Blake3_256,
            value,
        }
    }

    pub const fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeError> {
        if self.value == [0; 32] {
            return Err(ClinicalInferenceEnvelopeError::ZeroDigest);
        }
        Ok(())
    }
}

/// Named, versioned software or knowledge artifact.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClinicalArtifactIdentityV1 {
    pub name: String,
    pub version: String,
    pub digest: ClinicalDigestV1,
}

impl ClinicalArtifactIdentityV1 {
    fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeError> {
        if self.name.trim().is_empty() || self.version.trim().is_empty() {
            return Err(ClinicalInferenceEnvelopeError::IncompleteArtifactIdentity);
        }
        self.digest.validate()
    }
}

/// Exact model identity plus the lineages needed to interpret its output.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClinicalModelIdentityV1 {
    pub model: ClinicalArtifactIdentityV1,
    pub input_schema_digest: ClinicalDigestV1,
    pub output_schema_digest: ClinicalDigestV1,
    pub training_lineage_digest: Option<ClinicalDigestV1>,
    pub evaluation_lineage_digest: Option<ClinicalDigestV1>,
    pub calibration_evidence_digest: Option<ClinicalDigestV1>,
}

impl ClinicalModelIdentityV1 {
    fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeError> {
        self.model.validate()?;
        self.input_schema_digest.validate()?;
        self.output_schema_digest.validate()?;
        for digest in [
            self.training_lineage_digest,
            self.evaluation_lineage_digest,
            self.calibration_evidence_digest,
        ]
        .into_iter()
        .flatten()
        {
            digest.validate()?;
        }
        Ok(())
    }
}

/// Exact execution identity for one inference.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClinicalExecutionIdentityV1 {
    pub engine: ClinicalArtifactIdentityV1,
    pub model: ClinicalModelIdentityV1,
    pub runtime_digest: ClinicalDigestV1,
    pub configuration_digest: ClinicalDigestV1,
    pub input_evidence_digests: Vec<ClinicalDigestV1>,
    pub operation: String,
    pub executed_at_micros: i64,
    pub execution_nonce: [u8; 16],
}

impl ClinicalExecutionIdentityV1 {
    fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeError> {
        self.engine.validate()?;
        self.model.validate()?;
        self.runtime_digest.validate()?;
        self.configuration_digest.validate()?;
        if self.operation.trim().is_empty() {
            return Err(ClinicalInferenceEnvelopeError::IncompleteExecutionIdentity);
        }
        if self.execution_nonce == [0; 16] {
            return Err(ClinicalInferenceEnvelopeError::ZeroExecutionNonce);
        }
        if self.input_evidence_digests.is_empty() {
            return Err(ClinicalInferenceEnvelopeError::MissingExecutionInputs);
        }
        let mut seen = HashSet::new();
        for digest in &self.input_evidence_digests {
            digest.validate()?;
            if !seen.insert(*digest) {
                return Err(ClinicalInferenceEnvelopeError::DuplicateExecutionInput);
            }
        }
        Ok(())
    }
}

/// Subject identity is kept generic so the envelope can map to FHIR, OMOP, or
/// a privacy-preserving subject binding downstream.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClinicalSubjectBindingV1 {
    pub namespace: String,
    pub subject_id: String,
    pub binding_evidence_digest: ClinicalDigestV1,
}

impl ClinicalSubjectBindingV1 {
    fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeError> {
        if self.namespace.trim().is_empty() || self.subject_id.trim().is_empty() {
            return Err(ClinicalInferenceEnvelopeError::IncompleteSubjectBinding);
        }
        self.binding_evidence_digest.validate()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ClinicalEvidenceRoleV1 {
    Supports,
    Opposes,
    Context,
    Contraindication,
}

/// One exact evidence artifact used to support or challenge the inference.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClinicalEvidenceRefV1 {
    pub evidence_id: String,
    pub digest: ClinicalDigestV1,
    pub role: ClinicalEvidenceRoleV1,
}

impl ClinicalEvidenceRefV1 {
    fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeError> {
        if self.evidence_id.trim().is_empty() {
            return Err(ClinicalInferenceEnvelopeError::MissingEvidenceId);
        }
        self.digest.validate()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClinicalCalibrationStatusV1 {
    NotAssessed,
    Uncalibrated,
    Calibrated,
}

/// Uncertainty is decomposed instead of compressed into one confidence score.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClinicalUncertaintyV1 {
    pub epistemic: Option<f64>,
    pub aleatoric: Option<f64>,
    pub calibrated_probability: Option<f64>,
    pub calibration_status: ClinicalCalibrationStatusV1,
    pub calibration_evidence_digest: Option<ClinicalDigestV1>,
}

impl ClinicalUncertaintyV1 {
    fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeError> {
        for value in [self.epistemic, self.aleatoric].into_iter().flatten() {
            if !value.is_finite() || value < 0.0 {
                return Err(ClinicalInferenceEnvelopeError::InvalidUncertainty);
            }
        }
        if let Some(probability) = self.calibrated_probability {
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return Err(ClinicalInferenceEnvelopeError::InvalidCalibratedProbability);
            }
        }
        if let Some(digest) = self.calibration_evidence_digest {
            digest.validate()?;
        }
        if self.calibration_status == ClinicalCalibrationStatusV1::Calibrated
            && (self.calibrated_probability.is_none()
                || self.calibration_evidence_digest.is_none())
        {
            return Err(ClinicalInferenceEnvelopeError::IncompleteCalibrationEvidence);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClinicalDistributionStatusV1 {
    Unknown,
    InDistribution,
    OutOfDistribution,
}

/// Distribution-shift state is explicit and independently evidenced.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClinicalDistributionAssessmentV1 {
    pub status: ClinicalDistributionStatusV1,
    pub detector_evidence_digest: Option<ClinicalDigestV1>,
}

impl ClinicalDistributionAssessmentV1 {
    fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeError> {
        if let Some(digest) = self.detector_evidence_digest {
            digest.validate()?;
        }
        if self.status != ClinicalDistributionStatusV1::Unknown
            && self.detector_evidence_digest.is_none()
        {
            return Err(ClinicalInferenceEnvelopeError::MissingDistributionEvidence);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MissingEvidenceCriticalityV1 {
    Informational,
    Important,
    Critical,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MissingClinicalEvidenceV1 {
    pub requirement_id: String,
    pub description: String,
    pub criticality: MissingEvidenceCriticalityV1,
}

impl MissingClinicalEvidenceV1 {
    fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeError> {
        if self.requirement_id.trim().is_empty() || self.description.trim().is_empty() {
            return Err(ClinicalInferenceEnvelopeError::IncompleteMissingEvidence);
        }
        Ok(())
    }
}

/// Alternative interpretation preserved alongside the primary inference.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AlternativeClinicalHypothesisV1 {
    pub statement: String,
    pub semantics: ClinicalClaimSemanticsV1,
    pub rationale: String,
}

impl AlternativeClinicalHypothesisV1 {
    fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeError> {
        if self.statement.trim().is_empty() || self.rationale.trim().is_empty() {
            return Err(ClinicalInferenceEnvelopeError::IncompleteAlternative);
        }
        self.semantics
            .validate_schema()
            .map_err(ClinicalInferenceEnvelopeError::ClaimVocabulary)
    }
}

/// One evidence-bound Symthaea inference.
///
/// This artifact is evidence, not authority. It contains no field capable of
/// authorizing diagnosis, treatment, prescribing, dispensing, administration,
/// or autonomous therapeutic action.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClinicalInferenceEnvelopeV1 {
    pub schema_version: u16,
    pub semantics: ClinicalClaimSemanticsV1,
    pub subject: Option<ClinicalSubjectBindingV1>,
    pub statement: String,
    pub evidence: Vec<ClinicalEvidenceRefV1>,
    pub alternatives: Vec<AlternativeClinicalHypothesisV1>,
    pub missing_evidence: Vec<MissingClinicalEvidenceV1>,
    pub uncertainty: ClinicalUncertaintyV1,
    pub distribution: ClinicalDistributionAssessmentV1,
    pub execution: ClinicalExecutionIdentityV1,
    pub generated_at_micros: i64,
}

impl ClinicalInferenceEnvelopeV1 {
    pub fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeError> {
        if self.schema_version != CLINICAL_INFERENCE_ENVELOPE_VERSION {
            return Err(ClinicalInferenceEnvelopeError::UnsupportedSchemaVersion {
                found: self.schema_version,
                expected: CLINICAL_INFERENCE_ENVELOPE_VERSION,
            });
        }
        self.semantics
            .validate_schema()
            .map_err(ClinicalInferenceEnvelopeError::ClaimVocabulary)?;
        if self.statement.trim().is_empty() {
            return Err(ClinicalInferenceEnvelopeError::MissingStatement);
        }
        if self.semantics.intended_use == ClinicalIntendedUseClass::ClinicalDecisionSupport
            && self.subject.is_none()
        {
            return Err(ClinicalInferenceEnvelopeError::ClinicalUseWithoutSubjectBinding);
        }
        if let Some(subject) = &self.subject {
            subject.validate()?;
        }
        if self.evidence.is_empty() {
            return Err(ClinicalInferenceEnvelopeError::MissingEvidence);
        }
        let mut evidence_ids = HashSet::new();
        for evidence in &self.evidence {
            evidence.validate()?;
            if !evidence_ids.insert(evidence.evidence_id.as_str()) {
                return Err(ClinicalInferenceEnvelopeError::DuplicateEvidenceId);
            }
        }
        for alternative in &self.alternatives {
            alternative.validate()?;
        }
        for missing in &self.missing_evidence {
            missing.validate()?;
        }
        self.uncertainty.validate()?;
        self.distribution.validate()?;
        self.execution.validate()?;
        Ok(())
    }

    /// Whether critical evidence is explicitly missing.
    ///
    /// This is descriptive only; downstream clinical policy decides what to do
    /// with the result.
    #[must_use]
    pub fn has_critical_missing_evidence(&self) -> bool {
        self.missing_evidence
            .iter()
            .any(|item| item.criticality == MissingEvidenceCriticalityV1::Critical)
    }

    /// Whether the inference was explicitly identified as out-of-distribution.
    #[must_use]
    pub const fn is_out_of_distribution(&self) -> bool {
        self.distribution.status == ClinicalDistributionStatusV1::OutOfDistribution
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ClinicalInferenceEnvelopeError {
    UnsupportedSchemaVersion { found: u16, expected: u16 },
    ClaimVocabulary(ClinicalClaimVocabularyError),
    ZeroDigest,
    IncompleteArtifactIdentity,
    IncompleteExecutionIdentity,
    ZeroExecutionNonce,
    MissingExecutionInputs,
    DuplicateExecutionInput,
    IncompleteSubjectBinding,
    MissingEvidenceId,
    MissingEvidence,
    DuplicateEvidenceId,
    InvalidUncertainty,
    InvalidCalibratedProbability,
    IncompleteCalibrationEvidence,
    MissingDistributionEvidence,
    IncompleteMissingEvidence,
    IncompleteAlternative,
    MissingStatement,
    ClinicalUseWithoutSubjectBinding,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::{
        ClinicalApplicability, ClinicalClaimKind, ClinicalEvidenceStage,
        ClinicalIntendedUseClass,
    };

    fn digest(byte: u8) -> ClinicalDigestV1 {
        ClinicalDigestV1::blake3([byte; 32])
    }

    fn artifact(name: &str, version: &str, byte: u8) -> ClinicalArtifactIdentityV1 {
        ClinicalArtifactIdentityV1 {
            name: name.to_string(),
            version: version.to_string(),
            digest: digest(byte),
        }
    }

    fn execution() -> ClinicalExecutionIdentityV1 {
        ClinicalExecutionIdentityV1 {
            engine: artifact("symthaea", "0.1.0", 1),
            model: ClinicalModelIdentityV1 {
                model: artifact("clinical-model", "1.0.0", 2),
                input_schema_digest: digest(3),
                output_schema_digest: digest(4),
                training_lineage_digest: Some(digest(5)),
                evaluation_lineage_digest: Some(digest(6)),
                calibration_evidence_digest: None,
            },
            runtime_digest: digest(7),
            configuration_digest: digest(8),
            input_evidence_digests: vec![digest(9)],
            operation: "evaluate".to_string(),
            executed_at_micros: 1_000,
            execution_nonce: [1; 16],
        }
    }

    fn semantics(use_class: ClinicalIntendedUseClass) -> ClinicalClaimSemanticsV1 {
        ClinicalClaimSemanticsV1::new(
            ClinicalClaimKind::Prediction,
            ClinicalEvidenceStage::RetrospectiveExternal,
            ClinicalApplicability::DefinedTargetPopulation,
            use_class,
        )
    }

    fn uncertainty() -> ClinicalUncertaintyV1 {
        ClinicalUncertaintyV1 {
            epistemic: Some(0.2),
            aleatoric: Some(0.1),
            calibrated_probability: None,
            calibration_status: ClinicalCalibrationStatusV1::NotAssessed,
            calibration_evidence_digest: None,
        }
    }

    fn distribution() -> ClinicalDistributionAssessmentV1 {
        ClinicalDistributionAssessmentV1 {
            status: ClinicalDistributionStatusV1::Unknown,
            detector_evidence_digest: None,
        }
    }

    fn evidence() -> Vec<ClinicalEvidenceRefV1> {
        vec![ClinicalEvidenceRefV1 {
            evidence_id: "fact-1".to_string(),
            digest: digest(10),
            role: ClinicalEvidenceRoleV1::Supports,
        }]
    }

    #[test]
    fn research_only_envelope_can_omit_subject() {
        let envelope = ClinicalInferenceEnvelopeV1 {
            schema_version: CLINICAL_INFERENCE_ENVELOPE_VERSION,
            semantics: semantics(ClinicalIntendedUseClass::ResearchOnly),
            subject: None,
            statement: "candidate prediction".to_string(),
            evidence: evidence(),
            alternatives: vec![],
            missing_evidence: vec![],
            uncertainty: uncertainty(),
            distribution: distribution(),
            execution: execution(),
            generated_at_micros: 1_000,
        };

        assert!(envelope.validate().is_ok());
    }

    #[test]
    fn clinical_decision_support_requires_subject_binding() {
        let envelope = ClinicalInferenceEnvelopeV1 {
            schema_version: CLINICAL_INFERENCE_ENVELOPE_VERSION,
            semantics: semantics(ClinicalIntendedUseClass::ClinicalDecisionSupport),
            subject: None,
            statement: "clinical support prediction".to_string(),
            evidence: evidence(),
            alternatives: vec![],
            missing_evidence: vec![],
            uncertainty: uncertainty(),
            distribution: distribution(),
            execution: execution(),
            generated_at_micros: 1_000,
        };

        assert_eq!(
            envelope.validate(),
            Err(ClinicalInferenceEnvelopeError::ClinicalUseWithoutSubjectBinding)
        );
    }

    #[test]
    fn calibrated_uncertainty_requires_probability_and_evidence() {
        let uncertainty = ClinicalUncertaintyV1 {
            epistemic: Some(0.2),
            aleatoric: Some(0.1),
            calibrated_probability: None,
            calibration_status: ClinicalCalibrationStatusV1::Calibrated,
            calibration_evidence_digest: Some(digest(11)),
        };

        assert_eq!(
            uncertainty.validate(),
            Err(ClinicalInferenceEnvelopeError::IncompleteCalibrationEvidence)
        );
    }

    #[test]
    fn known_distribution_status_requires_detector_evidence() {
        let assessment = ClinicalDistributionAssessmentV1 {
            status: ClinicalDistributionStatusV1::OutOfDistribution,
            detector_evidence_digest: None,
        };

        assert_eq!(
            assessment.validate(),
            Err(ClinicalInferenceEnvelopeError::MissingDistributionEvidence)
        );
    }

    #[test]
    fn critical_missingness_is_machine_readable_but_not_authority() {
        let envelope = ClinicalInferenceEnvelopeV1 {
            schema_version: CLINICAL_INFERENCE_ENVELOPE_VERSION,
            semantics: semantics(ClinicalIntendedUseClass::ResearchOnly),
            subject: None,
            statement: "candidate prediction".to_string(),
            evidence: evidence(),
            alternatives: vec![],
            missing_evidence: vec![MissingClinicalEvidenceV1 {
                requirement_id: "lab".to_string(),
                description: "required lab unavailable".to_string(),
                criticality: MissingEvidenceCriticalityV1::Critical,
            }],
            uncertainty: uncertainty(),
            distribution: distribution(),
            execution: execution(),
            generated_at_micros: 1_000,
        };

        assert!(envelope.validate().is_ok());
        assert!(envelope.has_critical_missing_evidence());
    }

    #[test]
    fn zero_digest_fails_closed() {
        let zero = ClinicalDigestV1::blake3([0; 32]);
        assert_eq!(zero.validate(), Err(ClinicalInferenceEnvelopeError::ZeroDigest));
    }
}
