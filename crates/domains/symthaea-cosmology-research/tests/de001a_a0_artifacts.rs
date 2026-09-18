use serde_json::Value;
use std::collections::BTreeSet;

const MANIFEST: &str = include_str!("../references/de001a_a0_artifacts_v1.json");

#[test]
fn a0_manifest_is_complete_and_fail_closed() {
    let value: Value = serde_json::from_str(MANIFEST).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["protocol"], "DE-001A0-BYTE-INTEGRITY-v1");
    assert_eq!(value["scientific_claim"], "NONE");

    let artifacts = value["artifacts"].as_array().unwrap();
    assert_eq!(artifacts.len(), 8);
    let roles: BTreeSet<_> = artifacts
        .iter()
        .map(|artifact| artifact["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles.len(), artifacts.len());

    for artifact in artifacts {
        assert!(artifact["expected_size"].as_u64().unwrap() > 0);
        let digest = artifact["sha256"].as_str().unwrap();
        assert_eq!(digest.len(), 64);
        assert!(digest.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
        assert!(!artifact["authority"].as_str().unwrap().is_empty());
        assert!(!artifact["locator"].as_str().unwrap().is_empty());
    }
}

#[test]
fn official_reference_sizes_are_pinned() {
    let value: Value = serde_json::from_str(MANIFEST).unwrap();
    let artifacts = value["artifacts"].as_array().unwrap();
    let size = |role: &str| {
        artifacts
            .iter()
            .find(|artifact| artifact["role"] == role)
            .unwrap()["expected_size"]
            .as_u64()
            .unwrap()
    };
    assert_eq!(size("reference-input-configuration"), 2381);
    assert_eq!(size("reference-expanded-configuration"), 3969);
    assert_eq!(size("reference-minimizer-configuration"), 2484);
    assert_eq!(size("reference-bestfit-text"), 902);
    assert_eq!(size("reference-bestfit-getdist"), 3940);
}
