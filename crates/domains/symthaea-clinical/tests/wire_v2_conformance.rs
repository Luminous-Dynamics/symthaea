use symthaea_clinical::{
    clinical_inference_wire_v2_bytes, clinical_inference_wire_v2_digest_from_bytes,
    parse_clinical_inference_wire_v2, AlternativeClinicalHypothesisV1, ClinicalApplicability,
    ClinicalArtifactIdentityV1, ClinicalCalibrationStatusV1, ClinicalClaimKind,
    ClinicalClaimSemanticsV1, ClinicalDigestV1, ClinicalDistributionAssessmentV2,
    ClinicalDistributionStatusV1, ClinicalEvidenceIdentityV2, ClinicalEvidenceRefV2,
    ClinicalEvidenceRoleV1, ClinicalEvidenceStage, ClinicalExecutionIdentityV2,
    ClinicalInferenceEnvelopeV2, ClinicalIntendedUseClass, ClinicalModelIdentityV2,
    ClinicalSubjectBindingV2, ClinicalUncertaintyV2, CLINICAL_EVIDENCE_IDENTITY_V2_VERSION,
    CLINICAL_INFERENCE_ENVELOPE_V2_VERSION,
};

const VECTOR_HEX: &str = include_str!("../fixtures/clinical_inference_wire_v2.hex");

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

fn canonical_envelope() -> ClinicalInferenceEnvelopeV2 {
    let fact = evidence("mycelix/clinical-fact-snapshot/v1", "fact-1", 11);
    let subject_binding = evidence(
        "mycelix/patient-subject-binding-evidence/v1",
        "binding-1",
        10,
    );
    let training = evidence("symthaea/model-training-lineage/v1", "train-1", 5);
    let evaluation = evidence("symthaea/model-evaluation-lineage/v1", "eval-1", 6);
    let calibration = evidence("symthaea/model-calibration-evidence/v1", "cal-1", 12);
    let detector = evidence("symthaea/ood-detector-evidence/v1", "ood-1", 13);

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
        missing_evidence: vec![],
        uncertainty: ClinicalUncertaintyV2 {
            epistemic: Some(0.2),
            aleatoric: Some(0.1),
            calibrated_probability: Some(0.7),
            calibration_status: ClinicalCalibrationStatusV1::Calibrated,
            calibration_evidence: Some(calibration.clone()),
        },
        distribution: ClinicalDistributionAssessmentV2 {
            status: ClinicalDistributionStatusV1::InDistribution,
            detector_evidence: Some(detector),
        },
        execution: ClinicalExecutionIdentityV2 {
            engine: artifact("symthaea", "0.1.0", 1),
            model: ClinicalModelIdentityV2 {
                model: artifact("clinical-model", "1.0.0", 2),
                input_schema_digest: digest(3),
                output_schema_digest: digest(4),
                training_lineage: Some(training),
                evaluation_lineage: Some(evaluation),
                calibration_evidence: Some(calibration),
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

fn decode_hex(input: &str) -> Vec<u8> {
    let compact: Vec<u8> = input.bytes().filter(|byte| !byte.is_ascii_whitespace()).collect();
    assert_eq!(compact.len() % 2, 0, "fixture hex must contain full octets");
    compact
        .chunks_exact(2)
        .map(|pair| {
            let high = hex_nibble(pair[0]);
            let low = hex_nibble(pair[1]);
            (high << 4) | low
        })
        .collect()
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => panic!("invalid fixture hex digit"),
    }
}

#[test]
fn encoder_matches_frozen_cross_repository_vector() {
    let fixture = decode_hex(VECTOR_HEX);
    assert_eq!(fixture.len(), 1_260);
    assert_eq!(clinical_inference_wire_v2_bytes(&canonical_envelope()).unwrap(), fixture);
}

#[test]
fn frozen_vector_decodes_and_reencodes_exactly() {
    let fixture = decode_hex(VECTOR_HEX);
    let decoded = parse_clinical_inference_wire_v2(&fixture).unwrap();
    assert_eq!(decoded, canonical_envelope());
    assert_eq!(clinical_inference_wire_v2_bytes(&decoded).unwrap(), fixture);
}

#[test]
fn frozen_vector_has_valid_domain_separated_identity() {
    let fixture = decode_hex(VECTOR_HEX);
    let left = clinical_inference_wire_v2_digest_from_bytes(&fixture).unwrap();
    let right = clinical_inference_wire_v2_digest_from_bytes(&fixture).unwrap();
    assert_eq!(left, right);
    assert_ne!(left.into_bytes(), [0u8; 32]);
}
