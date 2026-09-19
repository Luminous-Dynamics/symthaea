// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use sha2::{Digest, Sha256};

const SPEC: &str = include_str!("../references/de001a_a2g_config_guard_v1.json");
const SCRIPT: &[u8] = include_bytes!("../scripts/de001a_a2g_config_guard.py");

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn a2g_resolution_guard_is_fail_closed() {
    let value: Value = serde_json::from_str(SPEC).expect("A2G specification must parse");

    assert_eq!(value["schema_version"], 1);
    assert_eq!(
        value["protocol"],
        "DE-001A2G-CONFIG-RESOLUTION-GUARD-v1"
    );
    assert_eq!(value["status"], "implementation-frozen-unqualified");
    assert_eq!(value["scientific_claim"], "NONE");
    assert_eq!(
        value["authority"],
        "configuration-resolution-determinism-only"
    );

    let expected_script_sha = value["script_sha256"]
        .as_str()
        .expect("script_sha256 must be string");
    assert_eq!(sha256_hex(SCRIPT), expected_script_sha);

    let markers = &value["forbidden_ambient_resolution_markers"];
    assert_eq!(markers["defaults_directive"], "!defaults");
    assert_eq!(markers["path_directive"], "!path");
    assert_eq!(markers["environment_placeholder"], "${");

    let policy = &value["policy"];
    for key in [
        "all_three_configurations_must_be_exact_a0_bytes",
        "all_forbidden_markers_must_be_absent",
        "a2r_must_bind_same_a0_receipt",
        "a2r_sampled_coordinate_order_must_be_nonempty_unique",
        "a2m_requires_a2g_pass",
        "raw_a2r_alone_must_not_authorize_a2m",
    ] {
        assert_eq!(policy[key], true, "{key} must remain true");
    }

    let forbidden = &value["forbidden_operations"];
    for key in [
        "yaml_parsing",
        "defaults_resolution",
        "path_expansion",
        "environment_expansion",
        "network",
        "likelihood_evaluation",
        "sampler_execution",
        "optimization",
        "configuration_mutation",
    ] {
        assert_eq!(forbidden[key], true, "{key} must remain forbidden");
    }

    assert_eq!(value["promotion"]["a2m_execution_authorized"], false);
    assert_eq!(value["promotion"]["a2_execution_authorized"], false);
}

#[test]
fn a2g_exact_configuration_hashes_remain_frozen() {
    let value: Value = serde_json::from_str(SPEC).expect("A2G specification must parse");
    let configs = &value["required_configurations"];

    let expected = [
        (
            "reference-input-configuration",
            2381_u64,
            "34499cb78ecaec78db44da9f06f61cd9c9ee497dc5c541b72b48cda54091c6ef",
        ),
        (
            "reference-expanded-configuration",
            3969_u64,
            "c4c23032d1695635aaea6eb47fabd909006ff32df0a27eadbf64b36c89f31ba1",
        ),
        (
            "reference-minimizer-configuration",
            2484_u64,
            "6b51048b359e4b9d646de09379ca273a8d6a1bc1b5a88f585a09d8f2e61f290c",
        ),
    ];

    for (role, size, sha) in expected {
        assert_eq!(configs[role]["size"].as_u64(), Some(size));
        assert_eq!(configs[role]["sha256"].as_str(), Some(sha));
    }
}
