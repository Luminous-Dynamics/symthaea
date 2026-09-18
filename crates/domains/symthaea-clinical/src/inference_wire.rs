// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Stable wire identity for evidence-bound clinical inference envelopes.
//!
//! A downstream system must not treat "the current Rust type from Symthaea" as
//! clinical trust. This module gives the v1 inference envelope an explicit,
//! domain-separated wire identity that can be pinned by an independent admission
//! policy.
//!
//! The wire digest proves only exact artifact identity. It does not prove that
//! the inference is scientifically correct, clinically valid, in-distribution,
//! trusted by Mycelix, or authorized for presentation/action.

use crate::inference::{
    ClinicalDigestV1, ClinicalInferenceEnvelopeError, ClinicalInferenceEnvelopeV1,
};

/// Explicit version of the wire-identity framing. This is separate from the
/// envelope schema version so a future encoding migration cannot silently reuse
/// the same identity domain.
pub const CLINICAL_INFERENCE_WIRE_IDENTITY_VERSION: u16 = 1;

const DERIVE_KEY_CONTEXT: &str = "symthaea.clinical.inference-envelope-wire.v1";
const SCHEMA_TAG: &[u8] = b"symthaea/clinical-inference-envelope/v1";

/// Validate and serialize the exact v1 envelope representation used for wire
/// identity. JSON is used here as the explicit v1 encoding contract; changing
/// encoding requires a new wire-identity version/domain rather than silent reuse.
pub fn clinical_inference_wire_bytes(
    envelope: &ClinicalInferenceEnvelopeV1,
) -> Result<Vec<u8>, ClinicalInferenceWireError> {
    envelope
        .validate()
        .map_err(ClinicalInferenceWireError::InvalidEnvelope)?;
    serde_json::to_vec(envelope).map_err(|_| ClinicalInferenceWireError::SerializationFailure)
}

/// Compute the exact v1 domain-separated identity of one validated inference
/// envelope.
pub fn clinical_inference_wire_digest(
    envelope: &ClinicalInferenceEnvelopeV1,
) -> Result<ClinicalDigestV1, ClinicalInferenceWireError> {
    let bytes = clinical_inference_wire_bytes(envelope)?;
    let mut hasher = blake3::Hasher::new_derive_key(DERIVE_KEY_CONTEXT);
    hasher.update(&CLINICAL_INFERENCE_WIRE_IDENTITY_VERSION.to_be_bytes());
    hasher.update(&(SCHEMA_TAG.len() as u16).to_be_bytes());
    hasher.update(SCHEMA_TAG);
    hasher.update(&(bytes.len() as u64).to_be_bytes());
    hasher.update(&bytes);
    Ok(ClinicalDigestV1::blake3(*hasher.finalize().as_bytes()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClinicalInferenceWireError {
    InvalidEnvelope(ClinicalInferenceEnvelopeError),
    SerializationFailure,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::{
        ClinicalApplicability, ClinicalClaimKind, ClinicalClaimSemanticsV1,
        ClinicalEvidenceStage, ClinicalIntendedUseClass,
    };
    use crate::inference::{
        AlternativeClinicalHypothesisV1, ClinicalArtifactIdentityV1,
        ClinicalCalibrationStatusV1, ClinicalDistributionAssessmentV1,
        ClinicalDistributionStatusV1, ClinicalEvidenceRefV1, ClinicalEvidenceRoleV1,
        ClinicalExecutionIdentityV1, ClinicalModelIdentityV1, ClinicalSubjectBindingV1,
        ClinicalUncertaintyV1, MissingClinicalEvidenceV1, CLINICAL_INFERENCE_ENVELOPE_VERSION,
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

    fn envelope() -> ClinicalInferenceEnvelopeV1 {
        ClinicalInferenceEnvelopeV1 {
            schema_version: CLINICAL_INFERENCE_ENVELOPE_VERSION,
            semantics: ClinicalClaimSemanticsV1::new(
                ClinicalClaimKind::Prediction,
                ClinicalEvidenceStage::RetrospectiveExternal,
                ClinicalApplicability::DefinedTargetPopulation,
                ClinicalIntendedUseClass::ClinicalDecisionSupport,
            ),
            subject: Some(ClinicalSubjectBindingV1 {
                namespace: "fhir/Patient".into(),
                subject_id: "patient-a".into(),
                binding_evidence_digest: digest(10),
            }),
            statement: "Candidate risk prediction".into(),
            evidence: vec![ClinicalEvidenceRefV1 {
                evidence_id: "fact-1".into(),
                digest: digest(11),
                role: ClinicalEvidenceRoleV1::Supports,
            }],
            alternatives: vec![AlternativeClinicalHypothesisV1 {
                statement: "Alternative explanation".into(),
                semantics: ClinicalClaimSemanticsV1::new(
                    ClinicalClaimKind::CausalHypothesis,
                    ClinicalEvidenceStage::MechanisticHypothesis,
                    ClinicalApplicability::Unestablished,
                    ClinicalIntendedUseClass::ResearchOnly,
                ),
                rationale: "Preserve competing hypothesis".into(),
            }],
            missing_evidence: Vec::<MissingClinicalEvidenceV1>::new(),
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
            execution: ClinicalExecutionIdentityV1 {
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
                input_evidence_digests: vec![digest(11)],
                operation: "evaluate".into(),
                executed_at_micros: 1_000,
                execution_nonce: [1; 16],
            },
            generated_at_micros: 1_001,
        }
    }

    #[test]
    fn identical_valid_envelopes_have_identical_wire_identity() {
        let a = envelope();
        let b = a.clone();
        assert_eq!(
            clinical_inference_wire_digest(&a).unwrap(),
            clinical_inference_wire_digest(&b).unwrap()
        );
    }

    #[test]
    fn model_substitution_changes_wire_identity() {
        let a = envelope();
        let mut b = a.clone();
        b.execution.model.model.version = "2.0.0".into();
        assert_ne!(
            clinical_inference_wire_digest(&a).unwrap(),
            clinical_inference_wire_digest(&b).unwrap()
        );
    }

    #[test]
    fn evidence_substitution_changes_wire_identity() {
        let a = envelope();
        let mut b = a.clone();
        b.evidence[0].digest = digest(42);
        assert_ne!(
            clinical_inference_wire_digest(&a).unwrap(),
            clinical_inference_wire_digest(&b).unwrap()
        );
    }

    #[test]
    fn execution_nonce_changes_wire_identity() {
        let a = envelope();
        let mut b = a.clone();
        b.execution.execution_nonce = [2; 16];
        assert_ne!(
            clinical_inference_wire_digest(&a).unwrap(),
            clinical_inference_wire_digest(&b).unwrap()
        );
    }

    #[test]
    fn invalid_envelope_cannot_obtain_wire_identity() {
        let mut invalid = envelope();
        invalid.execution.execution_nonce = [0; 16];
        assert!(matches!(
            clinical_inference_wire_digest(&invalid),
            Err(ClinicalInferenceWireError::InvalidEnvelope(_))
        ));
    }
}
