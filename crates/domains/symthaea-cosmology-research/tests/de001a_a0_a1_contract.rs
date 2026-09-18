use serde_json::Value;
use std::fs;

#[test]
fn a0_a1_manifest_is_fail_closed() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/references/de001a_a0_a1_execution_contract_v1.json"
    );
    let raw = fs::read_to_string(path).unwrap();
    let value: Value = serde_json::from_str(&raw).unwrap();

    assert_eq!(value["status"], "contract-frozen-blocked");
    assert_eq!(value["authority"], "reproduction-sanity-only");
    assert_eq!(value["a0"]["hash_mismatch_class"], "INVALID");
    assert_eq!(value["a1"]["mode"], "single-fixed-point-likelihood-evaluation");
    assert_eq!(value["a1"]["required_evaluation_count"], 1);
    assert_eq!(value["a1"]["minimizer"], "forbidden");
    assert_eq!(value["a1"]["sampler"], "forbidden");
    assert_eq!(value["a1"]["parameter_mutation"], "forbidden");
    assert!(value["a1"]["parameter_point"]["sha256"].is_null());
    assert_eq!(value["promotion_rule"]["scientific_claim"], "NONE");

    let blockers = value["blockers"].as_array().unwrap();
    assert!(blockers
        .iter()
        .any(|b| b == "environment-closure-not-yet-qualified"));
    assert!(blockers
        .iter()
        .any(|b| b == "complete-published-parameter-point-artifact-not-yet-frozen"));
}

#[test]
fn known_answer_bindings_are_exact_git_blobs() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/references/de001a_a0_a1_execution_contract_v1.json"
    );
    let value: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();

    assert_eq!(
        value["bindings"]["known_answer_manifest"]["git_blob"],
        "38421dcaaa29f7e50dde5a3acb7075ea7bda2c25"
    );
    assert_eq!(
        value["bindings"]["closure_manifest"]["git_blob"],
        "3f3157b18cd4b994abe389561b25b9d9b715abee"
    );
}
