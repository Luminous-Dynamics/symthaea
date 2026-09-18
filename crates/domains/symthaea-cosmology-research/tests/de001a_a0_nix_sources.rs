use serde_json::Value;
use std::collections::BTreeMap;

const A0: &str = include_str!("../references/de001a_a0_artifacts_v1.json");
const NIX: &str = include_str!("../references/de001a_a0_nix_sources_v1.json");

#[test]
fn nix_fixed_outputs_match_the_a0_integrity_contract() {
    let a0: Value = serde_json::from_str(A0).unwrap();
    let nix: Value = serde_json::from_str(NIX).unwrap();
    assert_eq!(nix["schema_version"], 1);
    assert_eq!(nix["scientific_claim"], "NONE");

    let by_role = |value: &Value| -> BTreeMap<String, (u64, String)> {
        value["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|artifact| {
                (
                    artifact["role"].as_str().unwrap().to_owned(),
                    (
                        artifact["expected_size"].as_u64().unwrap(),
                        artifact["sha256"].as_str().unwrap().to_owned(),
                    ),
                )
            })
            .collect()
    };

    assert_eq!(by_role(&a0), by_role(&nix));

    for artifact in nix["artifacts"].as_array().unwrap() {
        assert!(artifact["url"].as_str().unwrap().starts_with("https://"));
        assert!(artifact["nix_sri"].as_str().unwrap().starts_with("sha256-"));
    }
}
