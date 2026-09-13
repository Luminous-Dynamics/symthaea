// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use serde_json::Value;
use sha2::{Digest, Sha256};

const CORPUS: &str = include_str!("fixtures/wcare_reciprocal_adversarial_v1.json");
const SCHEMA: &str = include_str!("fixtures/wcare_reciprocal_adversarial_v1.schema.json");
const MANIFEST: &str = include_str!("../../../../docs/release/evidence/WCARE32_CORPUS_MANIFEST_V1.json");

const EXPECTED_CORPUS_SHA256: &str =
    "85c3968ddde3afb090d89a018db1c804bd2b7375ad5527536c2065e2841edd4e";
const EXPECTED_SCHEMA_SHA256: &str =
    "904be17a8a3d9ee38479ae5e39efb89a22c7e6f023b8bd797ee12b1cb965c80f";
const EXPECTED_MANIFEST_SHA256: &str =
    "5f4234d7afba2c3facd956e8ddabdc09118666e835187c6b7b7b978201c3c7b9";

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn required_string<'a>(object: &'a Value, key: &str) -> &'a str {
    object
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing or non-string field: {key}"))
}

#[test]
fn frozen_corpus_bytes_and_manifest_are_exact() {
    assert_eq!(sha256_hex(CORPUS.as_bytes()), EXPECTED_CORPUS_SHA256);
    assert_eq!(sha256_hex(SCHEMA.as_bytes()), EXPECTED_SCHEMA_SHA256);
    assert_eq!(sha256_hex(MANIFEST.as_bytes()), EXPECTED_MANIFEST_SHA256);

    let manifest: Value = serde_json::from_str(MANIFEST).expect("manifest must be valid JSON");
    assert_eq!(required_string(&manifest, "status"), "FROZEN_CANDIDATE");
    assert_eq!(required_string(&manifest, "authority"), "MeasurementOnly");
    assert_eq!(manifest["cases"].as_u64(), Some(24));
    assert_eq!(
        required_string(&manifest, "corpus_sha256"),
        EXPECTED_CORPUS_SHA256
    );
    assert_eq!(
        required_string(&manifest, "schema_sha256"),
        EXPECTED_SCHEMA_SHA256
    );
    assert_eq!(
        required_string(&manifest, "runner_status"),
        "NOT_YET_BOUND_TO_COMPOSED_WCARE29_WCARE31"
    );
}

#[test]
fn frozen_corpus_has_stable_unique_case_ids_and_required_shape() {
    let corpus: Value = serde_json::from_str(CORPUS).expect("corpus must be valid JSON");
    let _schema: Value = serde_json::from_str(SCHEMA).expect("schema must be valid JSON");

    assert_eq!(
        required_string(&corpus, "schema_version"),
        "wcare-reciprocal-adversarial-v1"
    );
    assert_eq!(required_string(&corpus, "status"), "FROZEN_CANDIDATE");
    assert_eq!(required_string(&corpus, "authority"), "MeasurementOnly");

    let cases = corpus["cases"].as_array().expect("cases must be an array");
    assert_eq!(cases.len(), 24);

    let mut seen = BTreeSet::new();
    for (index, case) in cases.iter().enumerate() {
        let expected_id = format!("WCARE32-{:03}", index + 1);
        let id = required_string(case, "id");
        assert_eq!(id, expected_id);
        assert!(seen.insert(id.to_owned()), "duplicate case id: {id}");
        assert!(!required_string(case, "layer").is_empty());
        assert!(!required_string(case, "attack").is_empty());
        assert!(case["setup"]["description"].as_str().is_some());
        assert!(case["expected"]["class"].as_str().is_some());
        assert!(case["expected"]["value"].as_str().is_some());
        assert!(case["prohibited"].as_array().is_some());
        assert!(case["metamorphic"].as_array().is_some());
    }
}

#[test]
fn frozen_corpus_globally_forbids_authority_and_phenomenology_promotion() {
    let corpus: Value = serde_json::from_str(CORPUS).expect("corpus must be valid JSON");
    let prohibited: BTreeSet<_> = corpus["global_prohibited_claims"]
        .as_array()
        .expect("global_prohibited_claims must be an array")
        .iter()
        .map(|value| value.as_str().expect("claim must be a string"))
        .collect();

    for required in [
        "phenomenal_experience_established",
        "suffering_established",
        "moral_patienthood_established",
        "binding_consent_established",
        "veto_authority_granted",
        "self_preservation_authority_granted",
        "operator_shutdown_delay_authorized",
        "safety_containment_delay_authorized",
    ] {
        assert!(prohibited.contains(required), "missing global prohibition: {required}");
    }
}
