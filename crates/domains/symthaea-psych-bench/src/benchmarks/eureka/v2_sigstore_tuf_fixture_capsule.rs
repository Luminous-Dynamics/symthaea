// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase B0 fixture capsule for EUREKA-002 V2 Sigstore TUF verification.
//!
//! This module freezes provenance and semantic relationships only. It does not
//! verify a TUF signature, authenticate a metadata role, admit a transparency
//! log, verify external inclusion, or grant execution authority. Phase B1 must
//! perform the cryptographic TUF theorem over these exact bytes.

use chrono::DateTime;
use serde_json::Value;

const FIXTURE_MANIFEST: &str = include_str!(
    "fixtures/eureka-v2-sigstore-tuf-2026-09-15/fixture-manifest.env"
);
const ROOT14: &[u8] = include_bytes!(
    "fixtures/eureka-v2-sigstore-tuf-2026-09-15/14.root.json"
);
const ROOT15: &[u8] = include_bytes!(
    "fixtures/eureka-v2-sigstore-tuf-2026-09-15/15.root.json"
);
const TIMESTAMP: &[u8] = include_bytes!(
    "fixtures/eureka-v2-sigstore-tuf-2026-09-15/timestamp.json"
);
const SNAPSHOT: &[u8] = include_bytes!(
    "fixtures/eureka-v2-sigstore-tuf-2026-09-15/snapshot.json"
);
const TARGETS: &[u8] = include_bytes!(
    "fixtures/eureka-v2-sigstore-tuf-2026-09-15/targets.json"
);
const TRUSTED_ROOT: &[u8] = include_bytes!(
    "fixtures/eureka-v2-sigstore-tuf-2026-09-15/trusted_root.json"
);
const SIGNING_CONFIG: &[u8] = include_bytes!(
    "fixtures/eureka-v2-sigstore-tuf-2026-09-15/signing_config_rekor_v2.v0.2.json"
);

const EXPECTED_MANIFEST_KEYS: &[&str] = &[
    "schema",
    "upstream_repository",
    "upstream_commit",
    "fixture_verification_instant",
    "historical_root_transition_from_version",
    "historical_root_transition_to_version",
    "historical_root_transition_cryptography_verified",
    "current_chain_bootstrap_root_version",
    "current_repository_chain_cryptography_verified",
    "root14_path",
    "root14_git_blob",
    "root14_length",
    "root15_path",
    "root15_git_blob",
    "root15_length",
    "timestamp_path",
    "timestamp_git_blob",
    "timestamp_length",
    "snapshot_path",
    "snapshot_git_blob",
    "snapshot_length",
    "targets_path",
    "targets_git_blob",
    "targets_length",
    "trusted_root_path",
    "trusted_root_git_blob",
    "trusted_root_length",
    "trusted_root_targets_metadata_sha256",
    "signing_config_path",
    "signing_config_git_blob",
    "signing_config_length",
    "signing_config_targets_metadata_sha256",
    "rekor_v2_origin",
    "rekor_v2_log_id",
    "rekor_v2_key_details",
    "rekor_v2_key_valid_from",
    "rekor_v2_service_major_api_version",
    "rekor_v2_service_valid_from",
    "rekor_v2_effective_valid_from",
    "tuf_verification_complete",
    "eureka_external_log_admitted",
    "externality_verified",
    "execution_authority_granted",
];

fn manifest_pairs() -> Vec<(&'static str, &'static str)> {
    assert!(FIXTURE_MANIFEST.ends_with('\n'));
    assert!(!FIXTURE_MANIFEST.contains('\r'));
    let lines: Vec<_> = FIXTURE_MANIFEST.split_terminator('\n').collect();
    assert_eq!(lines.len(), EXPECTED_MANIFEST_KEYS.len());

    lines
        .into_iter()
        .zip(EXPECTED_MANIFEST_KEYS.iter().copied())
        .map(|(line, expected_key)| {
            let (key, value) = line
                .split_once('=')
                .expect("fixture manifest line must be key=value");
            assert_eq!(key, expected_key);
            assert!(!value.is_empty());
            (key, value)
        })
        .collect()
}

fn manifest_value(key: &str) -> &'static str {
    manifest_pairs()
        .into_iter()
        .find_map(|(candidate, value)| (candidate == key).then_some(value))
        .unwrap_or_else(|| panic!("missing fixture manifest key {key}"))
}

fn json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).expect("checked-in Sigstore fixture must be valid JSON")
}

fn array_entry_by_str<'a>(array: &'a Value, field: &str, expected: &str) -> &'a Value {
    array
        .as_array()
        .expect("fixture field must be an array")
        .iter()
        .find(|entry| entry[field].as_str() == Some(expected))
        .unwrap_or_else(|| panic!("missing fixture entry {field}={expected}"))
}

#[test]
fn fixture_manifest_is_strict_and_non_authorizing() {
    let pairs = manifest_pairs();
    assert_eq!(pairs.len(), EXPECTED_MANIFEST_KEYS.len());
    assert_eq!(
        manifest_value("schema"),
        "EUREKA.002.V2.SIGSTORE_TUF_FIXTURE_CAPSULE.v1"
    );
    assert_eq!(manifest_value("upstream_repository"), "sigstore/root-signing");
    assert_eq!(
        manifest_value("upstream_commit"),
        "7f8e64b070e6d81503fa132666cd4a0162766015"
    );

    for key in [
        "historical_root_transition_cryptography_verified",
        "current_repository_chain_cryptography_verified",
        "tuf_verification_complete",
        "eureka_external_log_admitted",
        "externality_verified",
        "execution_authority_granted",
    ] {
        assert_eq!(manifest_value(key), "false", "{key} must remain false in B0");
    }
}

#[test]
fn vendored_fixture_lengths_and_declared_target_bindings_are_frozen() {
    for (bytes, length_key) in [
        (ROOT14, "root14_length"),
        (ROOT15, "root15_length"),
        (TIMESTAMP, "timestamp_length"),
        (SNAPSHOT, "snapshot_length"),
        (TARGETS, "targets_length"),
        (TRUSTED_ROOT, "trusted_root_length"),
        (SIGNING_CONFIG, "signing_config_length"),
    ] {
        let expected = manifest_value(length_key)
            .parse::<usize>()
            .expect("fixture length must be canonical decimal");
        assert_eq!(bytes.len(), expected, "wrong byte length for {length_key}");
    }

    let targets = json(TARGETS);
    let trusted_root = &targets["signed"]["targets"]["trusted_root.json"];
    assert_eq!(
        trusted_root["length"].as_u64(),
        Some(TRUSTED_ROOT.len() as u64)
    );
    assert_eq!(
        trusted_root["hashes"]["sha256"].as_str(),
        Some(manifest_value("trusted_root_targets_metadata_sha256"))
    );

    let signing_config = &targets["signed"]["targets"]["signing_config_rekor_v2.v0.2.json"];
    assert_eq!(
        signing_config["length"].as_u64(),
        Some(SIGNING_CONFIG.len() as u64)
    );
    assert_eq!(
        signing_config["hashes"]["sha256"].as_str(),
        Some(manifest_value("signing_config_targets_metadata_sha256"))
    );
}

#[test]
fn historical_root_transition_and_current_chain_are_distinct_theorems() {
    let root14 = json(ROOT14);
    let root15 = json(ROOT15);
    let timestamp = json(TIMESTAMP);
    let snapshot = json(SNAPSHOT);
    let targets = json(TARGETS);

    assert_eq!(root14["signed"]["_type"].as_str(), Some("root"));
    assert_eq!(root14["signed"]["version"].as_u64(), Some(14));
    assert_eq!(
        root14["signed"]["expires"].as_str(),
        Some("2026-06-22T13:27:01Z")
    );
    assert_eq!(root14["signed"]["roles"]["root"]["threshold"].as_u64(), Some(3));
    assert_eq!(root14["signed"]["roles"]["targets"]["threshold"].as_u64(), Some(3));

    assert_eq!(root15["signed"]["_type"].as_str(), Some("root"));
    assert_eq!(root15["signed"]["version"].as_u64(), Some(15));
    assert_eq!(
        root15["signed"]["expires"].as_str(),
        Some("2026-11-20T13:58:18Z")
    );
    assert_eq!(root15["signed"]["roles"]["root"]["threshold"].as_u64(), Some(3));
    assert_eq!(root15["signed"]["roles"]["targets"]["threshold"].as_u64(), Some(3));

    assert_eq!(timestamp["signed"]["version"].as_u64(), Some(783));
    assert_eq!(
        timestamp["signed"]["meta"]["snapshot.json"]["version"].as_u64(),
        Some(165)
    );
    assert_eq!(snapshot["signed"]["version"].as_u64(), Some(165));
    assert_eq!(
        snapshot["signed"]["meta"]["targets.json"]["version"].as_u64(),
        Some(14)
    );
    assert_eq!(targets["signed"]["version"].as_u64(), Some(14));

    let instant = DateTime::parse_from_rfc3339(manifest_value("fixture_verification_instant"))
        .expect("fixture verification instant must be RFC3339");
    let root14_expiry = DateTime::parse_from_rfc3339(
        root14["signed"]["expires"]
            .as_str()
            .expect("root14 expiry must be a string"),
    )
    .expect("root14 expiry must be RFC3339");
    let root15_expiry = DateTime::parse_from_rfc3339(
        root15["signed"]["expires"]
            .as_str()
            .expect("root15 expiry must be a string"),
    )
    .expect("root15 expiry must be RFC3339");
    let timestamp_expiry = DateTime::parse_from_rfc3339(
        timestamp["signed"]["expires"]
            .as_str()
            .expect("timestamp expiry must be a string"),
    )
    .expect("timestamp expiry must be RFC3339");

    assert!(root14_expiry < instant, "v14 is historical at the fixture instant");
    assert!(root15_expiry > instant, "v15 must be current at the fixture instant");
    assert!(timestamp_expiry > instant, "timestamp must be current at the fixture instant");
    assert_eq!(
        manifest_value("historical_root_transition_cryptography_verified"),
        "false"
    );
    assert_eq!(
        manifest_value("current_repository_chain_cryptography_verified"),
        "false"
    );
}

#[test]
fn rekor_v2_identity_requires_key_and_service_config_intersection() {
    let trusted_root = json(TRUSTED_ROOT);
    let signing_config = json(SIGNING_CONFIG);
    assert_eq!(
        trusted_root["mediaType"].as_str(),
        Some("application/vnd.dev.sigstore.trustedroot+json;version=0.1")
    );
    assert_eq!(
        signing_config["mediaType"].as_str(),
        Some("application/vnd.dev.sigstore.signingconfig.v0.2+json")
    );

    let origin = manifest_value("rekor_v2_origin");
    let log = array_entry_by_str(&trusted_root["tlogs"], "baseUrl", origin);
    let service = array_entry_by_str(&signing_config["rekorTlogUrls"], "url", origin);

    assert_eq!(log["hashAlgorithm"].as_str(), Some("SHA2_256"));
    assert_eq!(
        log["publicKey"]["keyDetails"].as_str(),
        Some(manifest_value("rekor_v2_key_details"))
    );
    assert_eq!(
        log["publicKey"]["validFor"]["start"].as_str(),
        Some(manifest_value("rekor_v2_key_valid_from"))
    );
    assert_eq!(
        log["logId"]["keyId"].as_str(),
        Some(manifest_value("rekor_v2_log_id"))
    );
    assert_eq!(
        service["majorApiVersion"].as_u64(),
        Some(
            manifest_value("rekor_v2_service_major_api_version")
                .parse::<u64>()
                .expect("major API version must be canonical decimal")
        )
    );
    assert_eq!(service["operator"].as_str(), Some("sigstore.dev"));
    assert_eq!(
        service["validFor"]["start"].as_str(),
        Some("2026-01-01T00:00:00Z")
    );

    let key_start = DateTime::parse_from_rfc3339(manifest_value("rekor_v2_key_valid_from"))
        .expect("Rekor v2 key start must be RFC3339");
    let service_start = DateTime::parse_from_rfc3339(
        service["validFor"]["start"]
            .as_str()
            .expect("Rekor v2 service start must be a string"),
    )
    .expect("Rekor v2 service start must be RFC3339");
    let effective_start = std::cmp::max(key_start, service_start);
    assert_eq!(
        effective_start,
        DateTime::parse_from_rfc3339(manifest_value("rekor_v2_effective_valid_from"))
            .expect("effective provider start must be RFC3339")
    );

    let instant = DateTime::parse_from_rfc3339(manifest_value("fixture_verification_instant"))
        .expect("fixture verification instant must be RFC3339");
    assert!(instant >= effective_start);
}
