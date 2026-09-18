// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Typed-evidence clinical inference envelope v2.
//!
//! v1 bound exact evidence digests but deliberately left their semantic digest
//! namespace implicit. That is sufficient for internal experiments, but not for
//! long-lived external medical interoperability. v2 makes evidence namespace,
//! artifact identity, digest, and execution-input binding explicit while still
//! carrying **no clinical authority**.

use crate::claims::{
    ClinicalClaimSemanticsV1, ClinicalClaimVocabularyError, ClinicalIntendedUseClass,
};
use crate::inference::{
    AlternativeClinicalHypothesisV1, ClinicalArtifactIdentityV1, ClinicalCalibrationStatusV1,
    ClinicalDigestV1, ClinicalDistributionAssessmentV1, ClinicalDistributionStatusV1,
    ClinicalEvidenceRoleV1, ClinicalModelIdentityV1, ClinicalUncertaintyV1,
    MissingClinicalEvidenceV1, MissingEvidenceCriticalityV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const CLINICAL_INFERENCE_ENVELOPE_V2_VERSION: u16 = 2;
pub const CLINICAL_EVIDENCE_IDENTITY_V2_VERSION: u16 = 1;

const MAX_NAMESPACE_LEN: usize = 256;
const MAX_ARTIFACT_ID_LEN: usize = 512;

/// Typed identity of one exact evidence artifact.
///
/// `namespace` identifies the canonicalization/digest contract, not merely a
/// human source category. Example: `mycelix/clinical-fact-snapshot/v1`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct ClinicalEvidenceIdentityV2 {
    pub identity_version: u16,
    pub namespace: String,
    pub artifact_id: String,
    pub digest: ClinicalDigestV1,
}

impl ClinicalEvidenceIdentityV2 {
    pub fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeV2Error> {
        if self.identity_version != CLINICAL_EVIDENCE_IDENTITY_V2_VERSION {
            return Err(ClinicalInferenceEnvelopeV2Error::UnsupportedEvidenceIdentityVersion(
                self.identity_version,
            ));
        }
        validate_namespace(&self.namespace)?;
        validate_identifier(&self.artifact_id, "evidence artifact id", MAX_ARTIFACT_ID_LEN)?;
        validate_digest(&self.digest)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ClinicalEvidenceRefV2 {
    pub identity: ClinicalEvidenceIdentityV2,
    pub role: ClinicalEvidenceRoleV1,
}

/// Subject identity plus typed evidence proving the subject binding used by the
/// computation. Subject identifier namespace and binding-evidence namespace are
/// deliberately separate concepts.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ClinicalSubjectBindingV2 {
    pub subject_namespace: String,
    pub subject_id: String,
    pub binding_evidence: ClinicalEvidenceIdentityV2,
}

impl ClinicalSubjectBindingV2 {
    fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeV2Error> {
        validate_identifier(&self.subject_namespace, "subject namespace", MAX_NAMESPACE_LEN)?;
        validate_identifier(&self.subject_id, "subject id", MAX_ARTIFACT_ID_LEN)?;
        self.binding_evidence.validate()
    }
}

/// Exact execution identity whose inputs are themselves typed evidence identities.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ClinicalExecutionIdentityV2 {
    pub engine: ClinicalArtifactIdentityV1,
    pub model: ClinicalModelIdentityV1,
    pub runtime_digest: ClinicalDigestV1,
    pub configuration_digest: ClinicalDigestV1,
    pub input_evidence: Vec<ClinicalEvidenceIdentityV2>,
    pub operation: String,
    pub executed_at_micros: i64,
    pub execution_nonce: [u8; 16],
}

impl ClinicalExecutionIdentityV2 {
    fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeV2Error> {
        validate_artifact(&self.engine)?;
        validate_model(&self.model)?;
        validate_digest(&self.runtime_digest)?;
        validate_digest(&self.configuration_digest)?;
        validate_identifier(&self.operation, "execution operation", MAX_ARTIFACT_ID_LEN)?;
        if self.execution_nonce == [0u8; 16] {
            return Err(ClinicalInferenceEnvelopeV2Error::ZeroExecutionNonce);
        }
        if self.input_evidence.is_empty() {
            return Err(ClinicalInferenceEnvelopeV2Error::MissingExecutionInputs);
        }

        let mut seen = HashSet::new();
        for identity in &self.input_evidence {
            identity.validate()?;
            if !seen.insert(identity.clone()) {
                return Err(ClinicalInferenceEnvelopeV2Error::DuplicateExecutionInput);
            }
        }
        Ok(())
    }
}

/// External-interoperability form of an evidence-bound Symthaea inference.
///
/// This is still evidence, not authority. No field authorizes diagnosis,
/// treatment, prescribing, dispensing, administration, or autonomous action.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ClinicalInferenceEnvelopeV2 {
    pub schema_version: u16,
    pub semantics: ClinicalClaimSemanticsV1,
    pub subject: Option<ClinicalSubjectBindingV2>,
    pub statement: String,
    pub evidence: Vec<ClinicalEvidenceRefV2>,
    pub alternatives: Vec<AlternativeClinicalHypothesisV1>,
    pub missing_evidence: Vec<MissingClinicalEvidenceV1>,
    pub uncertainty: ClinicalUncertaintyV1,
    pub distribution: ClinicalDistributionAssessmentV1,
    pub execution: ClinicalExecutionIdentityV2,
    pub generated_at_micros: i64,
}

impl ClinicalInferenceEnvelopeV2 {
    pub fn validate(&self) -> Result<(), ClinicalInferenceEnvelopeV2Error> {
        if self.schema_version != CLINICAL_INFERENCE_ENVELOPE_V2_VERSION {
            return Err(ClinicalInferenceEnvelopeV2Error::UnsupportedSchemaVersion {
                found: self.schema_version,
                expected: CLINICAL_INFERENCE_ENVELOPE_V2_VERSION,
            });
        }
        self.semantics
            .validate_schema()
            .map_err(ClinicalInferenceEnvelopeV2Error::ClaimVocabulary)?;
        validate_identifier(&self.statement, "statement", usize::MAX)?;

        if self.semantics.intended_use == ClinicalIntendedUseClass::ClinicalDecisionSupport
            && self.subject.is_none()
        {
            return Err(ClinicalInferenceEnvelopeV2Error::ClinicalUseWithoutSubjectBinding);
        }

        self.execution.validate()?;

        if let Some(subject) = &self.subject {
            subject.validate()?;
            if !self.execution.input_evidence.contains(&subject.binding_evidence) {
                return Err(ClinicalInferenceEnvelopeV2Error::SubjectBindingNotExecutionBound);
            }
        }

        if self.evidence.is_empty() {
            return Err(ClinicalInferenceEnvelopeV2Error::MissingEvidence);
        }

        let mut named_evidence: HashMap<(&str, &str), &ClinicalDigestV1> = HashMap::new();
        for evidence in &self.evidence {
            evidence.identity.validate()?;
            let key = (
                evidence.identity.namespace.as_str(),
                evidence.identity.artifact_id.as_str(),
            );
            if named_evidence.insert(key, &evidence.identity.digest).is_some() {
                return Err(ClinicalInferenceEnvelopeV2Error::DuplicateEvidenceIdentity);
            }
            if !self.execution.input_evidence.contains(&evidence.identity) {
                return Err(ClinicalInferenceEnvelopeV2Error::EvidenceNotExecutionBound);
            }
        }

        for alternative in &self.alternatives {
            validate_identifier(&alternative.statement, "alternative statement", usize::MAX)?;
            validate_identifier(&alternative.rationale, "alternative rationale", usize::MAX)?;
            alternative
                .semantics
                .validate_schema()
                .map_err(ClinicalInferenceEnvelopeV2Error::ClaimVocabulary)?;
        }

        for missing in &self.missing_evidence {
            validate_identifier(&missing.requirement_id, "missing-evidence requirement id", MAX_ARTIFACT_ID_LEN)?;
            validate_identifier(&missing.description, "missing-evidence description", usize::MAX)?;
        }

        validate_uncertainty(&self.uncertainty)?;
        validate_distribution(&self.distribution)?;
        Ok(())
    }

    #[must_use]
    pub fn has_critical_missing_evidence(&self) -> bool {
        self.missing_evidence
            .iter()
            .any(|item| item.criticality == MissingEvidenceCriticalityV1::Critical)
    }

    #[must_use]
    pub fn is_out_of_distribution(&self) -> bool {
        self.distribution.status == ClinicalDistributionStatusV1::OutOfDistribution
    }
}

fn validate_namespace(value: &str) -> Result<(), ClinicalInferenceEnvelopeV2Error> {
    validate_identifier(value, "evidence namespace", MAX_NAMESPACE_LEN)?;
    if value.chars().any(char::is_whitespace) {
        return Err(ClinicalInferenceEnvelopeV2Error::InvalidEvidenceNamespace);
    }
    Ok(())
}

fn validate_identifier(
    value: &str,
    field: &'static str,
    max_len: usize,
) -> Result<(), ClinicalInferenceEnvelopeV2Error> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(ClinicalInferenceEnvelopeV2Error::InvalidIdentifier(field));
    }
    if value.len() > max_len {
        return Err(ClinicalInferenceEnvelopeV2Error::IdentifierTooLong(field));
    }
    Ok(())
}

fn validate_digest(digest: &ClinicalDigestV1) -> Result<(), ClinicalInferenceEnvelopeV2Error> {
    if digest.value == [0u8; 32] {
        return Err(ClinicalInferenceEnvelopeV2Error::ZeroDigest);
    }
    Ok(())
}

fn validate_artifact(
    artifact: &ClinicalArtifactIdentityV1,
) -> Result<(), ClinicalInferenceEnvelopeV2Error> {
    validate_identifier(&artifact.name, "artifact name", MAX_ARTIFACT_ID_LEN)?;
    validate_identifier(&artifact.version, "artifact version", MAX_ARTIFACT_ID_LEN)?;
    validate_digest(&artifact.digest)
}

fn validate_model(model: &ClinicalModelIdentityV1) -> Result<(), ClinicalInferenceEnvelopeV2Error> {
    validate_artifact(&model.model)?;
    validate_digest(&model.input_schema_digest)?;
    validate_digest(&model.output_schema_digest)?;
    for digest in [
        model.training_lineage_digest,
        model.evaluation_lineage_digest,
        model.calibration_evidence_digest,
    ]
    .into_iter()
    .flatten()
    {
        validate_digest(&digest)?;
    }
    Ok(())
}

fn validate_uncertainty(
    uncertainty: &ClinicalUncertaintyV1,
) -> Result<(), ClinicalInferenceEnvelopeV2Error> {
    for value in [uncertainty.epistemic, uncertainty.aleatoric]
        .into_iter()
        .flatten()
    {
        if !value.is_finite() || value < 0.0 {
            return Err(ClinicalInferenceEnvelopeV2Error::InvalidUncertainty);
        }
    }
    if let Some(probability) = uncertainty.calibrated_probability {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(ClinicalInferenceEnvelopeV2Error::InvalidCalibratedProbability);
        }
    }
    if let Some(digest) = uncertainty.calibration_evidence_digest {
        validate_digest(&digest)?;
    }
    if uncertainty.calibration_status == ClinicalCalibrationStatusV1::Calibrated
        && (uncertainty.calibrated_probability.is_none()
            || uncertainty.calibration_evidence_digest.is_none())
    {
        return Err(ClinicalInferenceEnvelopeV2Error::IncompleteCalibrationEvidence);
    }
    Ok(())
}

fn validate_distribution(
    distribution: &ClinicalDistributionAssessmentV1,
) -> Result<(), ClinicalInferenceEnvelopeV2Error> {
    if let Some(digest) = distribution.detector_evidence_digest {
        validate_digest(&digest)?;
    }
    if distribution.status != ClinicalDistributionStatusV1::Unknown
        && distribution.detector_evidence_digest.is_none()
    {
        return Err(ClinicalInferenceEnvelopeV2Error::MissingDistributionEvidence);
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClinicalInferenceEnvelopeV2Error {
    UnsupportedSchemaVersion { found: u16, expected: u16 },
    UnsupportedEvidenceIdentityVersion(u16),
    ClaimVocabulary(ClinicalClaimVocabularyError),
    InvalidIdentifier(&'static str),
    IdentifierTooLong(&'static str),
    InvalidEvidenceNamespace,
    ZeroDigest,
    ZeroExecutionNonce,
    MissingExecutionInputs,
    DuplicateExecutionInput,
    ClinicalUseWithoutSubjectBinding,
    SubjectBindingNotExecutionBound,
    MissingEvidence,
    DuplicateEvidenceIdentity,
    EvidenceNotExecutionBound,
    InvalidUncertainty,
    InvalidCalibratedProbability,
    IncompleteCalibrationEvidence,
    MissingDistributionEvidence,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::{
        ClinicalApplicability, ClinicalClaimKind, ClinicalEvidenceStage,
    };

    fn digest(byte: u8) -> ClinicalDigestV1 {
        ClinicalDigestV1::blake3([byte; 32])
    }

    fn artifact(name: &str, version: &str, byte: u8) -> ClinicalArtifactIdentityV1 {
        ClinicalArtifactIdentityV1 {
            name: name.into(),
            version: version.into(),
            digest: digest(byte),
        }
    }

    fn evidence(namespace: &str, artifact_id: &str, byte: u8) -> ClinicalEvidenceIdentityV2 {
        ClinicalEvidenceIdentityV2 {
            identity_version: CLINICAL_EVIDENCE_IDENTITY_V2_VERSION,
            namespace: namespace.into(),
            artifact_id: artifact_id.into(),
            digest: digest(byte),
        }
    }

    fn envelope() -> ClinicalInferenceEnvelopeV2 {
        let fact = evidence("mycelix/clinical-fact-snapshot/v1", "fact-1", 11);
        let subject_binding = evidence(
            "mycelix/patient-subject-binding-evidence/v1",
            "binding-1",
            10,
        );
        ClinicalInferenceEnvelopeV2 {
            schema_version: CLINICAL_INFERENCE_ENVELOPE_V2_VERSION,
            semantics: ClinicalClaimSemanticsV1::new(
                ClinicalClaimKind::Prediction,
                ClinicalEvidenceStage::RetrospectiveExternal,
                ClinicalApplicability::DefinedTargetPopulation,
                ClinicalIntendedUseClass::ClinicalDecisionSupport,
            ),
            subject: Some(ClinicalSubjectBindingV2 {
                subject_namespace: "fhir/Patient".into(),
                subject_id: "patient-a".into(),
                binding_evidence: subject_binding.clone(),
            }),
            statement: "Candidate risk prediction".into(),
            evidence: vec![ClinicalEvidenceRefV2 {
                identity: fact.clone(),
                role: ClinicalEvidenceRoleV1::Supports,
            }],
            alternatives: Vec::new(),
            missing_evidence: Vec::new(),
            uncertainty: ClinicalUncertaintyV1 {
                epistemic: Some(0.2),
                aleatoric: Some(0.1),
                calibrated_probability: Some(0.7),
                calibration_status: ClinicalCalibrationStatusV1::Calibrated,
                calibration_evidence_digest: Some(digest(12)),
            },
            distribution: ClinicalDistributionAssessmentV1 {
                status: ClinicalDistributionStatusV1::InDistribution,
                detector_evidence_digest: Some(digest(13)),
            },
            execution: ClinicalExecutionIdentityV2 {
                engine: artifact("symthaea", "0.1.0", 1),
                model: ClinicalModelIdentityV1 {
                    model: artifact("clinical-model", "1.0.0", 2),
                    input_schema_digest: digest(3),
                    output_schema_digest: digest(4),
                    training_lineage_digest: Some(digest(5)),
                    evaluation_lineage_digest: Some(digest(6)),
                    calibration_evidence_digest: Some(digest(12)),
                },
                runtime_digest: digest(7),
                configuration_digest: digest(8),
                input_evidence: vec![fact, subject_binding],
                operation: "evaluate".into(),
                executed_at_micros: 1_000,
                execution_nonce: [1u8; 16],
            },
            generated_at_micros: 1_001,
        }
    }

    #[test]
    fn typed_evidence_envelope_validates() {
        assert_eq!(envelope().validate(), Ok(()));
    }

    #[test]
    fn same_raw_digest_in_different_namespaces_is_not_same_identity() {
        let left = evidence("mycelix/clinical-fact-snapshot/v1", "fact-1", 42);
        let right = evidence("hl7/fhir-r4/resource-canonical/v1", "fact-1", 42);
        assert_ne!(left, right);
    }

    #[test]
    fn evidence_namespace_substitution_breaks_execution_binding() {
        let mut candidate = envelope();
        candidate.evidence[0].identity.namespace = "hl7/fhir-r4/resource-canonical/v1".into();
        assert!(matches!(
            candidate.validate(),
            Err(ClinicalInferenceEnvelopeV2Error::EvidenceNotExecutionBound)
        ));
    }

    #[test]
    fn subject_binding_namespace_substitution_breaks_execution_binding() {
        let mut candidate = envelope();
        candidate
            .subject
            .as_mut()
            .unwrap()
            .binding_evidence
            .namespace = "hl7/fhir-r4/resource-canonical/v1".into();
        assert!(matches!(
            candidate.validate(),
            Err(ClinicalInferenceEnvelopeV2Error::SubjectBindingNotExecutionBound)
        ));
    }

    #[test]
    fn duplicate_named_evidence_is_rejected_even_with_different_digest() {
        let mut candidate = envelope();
        let mut duplicate = candidate.evidence[0].clone();
        duplicate.identity.digest = digest(99);
        candidate.execution.input_evidence.push(duplicate.identity.clone());
        candidate.evidence.push(duplicate);
        assert!(matches!(
            candidate.validate(),
            Err(ClinicalInferenceEnvelopeV2Error::DuplicateEvidenceIdentity)
        ));
    }

    #[test]
    fn clinical_use_still_requires_subject_binding() {
        let mut candidate = envelope();
        candidate.subject = None;
        assert!(matches!(
            candidate.validate(),
            Err(ClinicalInferenceEnvelopeV2Error::ClinicalUseWithoutSubjectBinding)
        ));
    }

    #[test]
    fn critical_missing_evidence_remains_descriptive_not_authorizing() {
        let mut candidate = envelope();
        candidate.missing_evidence.push(MissingClinicalEvidenceV1 {
            requirement_id: "renal-function".into(),
            description: "renal function is required".into(),
            criticality: MissingEvidenceCriticalityV1::Critical,
        });
        assert!(candidate.validate().is_ok());
        assert!(candidate.has_critical_missing_evidence());
    }
}
