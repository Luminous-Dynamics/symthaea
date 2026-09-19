use symthaea_clinical::{
    clinical_inference_wire_bytes, clinical_inference_wire_digest_from_bytes,
    parse_clinical_inference_wire_bytes,
};

const VECTOR_V1: &[u8] = include_bytes!("../fixtures/clinical_inference_wire_v1.json");

#[test]
fn canonical_cross_repo_vector_is_accepted_by_symthaea() {
    let envelope = parse_clinical_inference_wire_bytes(VECTOR_V1)
        .expect("canonical v1 conformance vector must parse");
    let regenerated = clinical_inference_wire_bytes(&envelope)
        .expect("validated vector must reserialize canonically");
    assert_eq!(regenerated.as_slice(), VECTOR_V1);
    assert!(clinical_inference_wire_digest_from_bytes(VECTOR_V1).is_ok());
}

#[test]
fn conformance_vector_remains_bound_to_expected_model_and_subject() {
    let envelope = parse_clinical_inference_wire_bytes(VECTOR_V1)
        .expect("canonical v1 conformance vector must parse");
    assert_eq!(envelope.execution.model.model.name, "clinical-model");
    assert_eq!(envelope.execution.model.model.version, "1.0.0");
    let subject = envelope.subject.expect("clinical vector requires a subject");
    assert_eq!(subject.namespace, "fhir/Patient");
    assert_eq!(subject.subject_id, "patient-a");
}
