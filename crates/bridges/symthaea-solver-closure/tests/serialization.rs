use symthaea_solver_closure::{
    AmbientDiscoveryPolicy, ContentDigest, DigestAlgorithm, SolverInputArtifact,
    SolverInputClosure, SolverInputRole,
};

fn sha256(fill: char) -> ContentDigest {
    ContentDigest::new(DigestAlgorithm::Sha256, fill.to_string().repeat(64)).unwrap()
}

fn fixture() -> SolverInputClosure {
    let primary = SolverInputArtifact::new("case", SolverInputRole::Primary, sha256('a'))
        .with_locator("case/input.dat");

    let solver = SolverInputArtifact::new("solver", SolverInputRole::SolverExecutable, sha256('b'))
        .with_locator("/nix/store/example/bin/solver")
        .with_reported_identity("solver 1.2.3");

    let material = SolverInputArtifact::new("material", SolverInputRole::Referenced, sha256('c'))
        .with_parent("case")
        .with_locator("materials/steel.dat");

    SolverInputClosure::new(
        "serialization-fixture-v1",
        AmbientDiscoveryPolicy::Prohibited,
        vec![primary, solver, material],
        vec!["ambient user configuration disabled".into()],
    )
    .unwrap()
}

#[test]
fn json_round_trip_preserves_closure_identity() {
    let original = fixture();
    let original_id = original.closure_id().unwrap();
    let json = serde_json::to_string_pretty(&original).unwrap();
    let decoded: SolverInputClosure = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded.closure_id().unwrap(), original_id);
    assert_eq!(decoded, original);
}

#[test]
fn deserialization_does_not_bypass_validation() {
    let original = fixture();
    let mut value = serde_json::to_value(original).unwrap();
    value["artifacts"][0]["digest"]["hex"] = serde_json::Value::String("not-a-digest".into());

    let decoded: SolverInputClosure = serde_json::from_value(value).unwrap();
    assert!(decoded.validate().is_err());
    assert!(decoded.closure_id().is_err());
}
