use serde_json::Value;
use std::fs;

#[test]
fn released_likelihood_is_not_misclassified_as_independent_implementation() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/references/de001a_likelihood_provenance_v1.json"
    );
    let value: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();

    assert_eq!(value["status"], "provenance-correction-frozen");
    assert_eq!(value["desi_attestation"]["likelihood_identical"], true);
    assert_eq!(
        value["desi_attestation"]["measurement_files_byte_identical"],
        true
    );
    assert_eq!(
        value["desi_attestation"]["covariance_files_byte_identical"],
        true
    );
    assert_eq!(
        value["methodological_classification"]["independent_likelihood_implementation"],
        false
    );
    assert_eq!(
        value["methodological_classification"]["true_independent_implementation_gate"],
        "DE-001I"
    );
    assert_eq!(value["scientific_claim"], "NONE");
}
