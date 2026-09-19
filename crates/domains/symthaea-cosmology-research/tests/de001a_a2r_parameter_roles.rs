// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use sha2::{Digest, Sha256};

const SPEC: &str = include_str!("../references/de001a_a2r_parameter_roles_v1.json");
const SCRIPT: &[u8] = include_bytes!("../scripts/de001a_a2r_parameter_roles.py");

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn a2r_parameter_role_contract_is_fail_closed() {
    let value: Value = serde_json::from_str(SPEC).expect("A2R specification must parse");

    assert_eq!(value["schema_version"], 1);
    assert_eq!(
        value["protocol"],
        "DE-001A2R-PARAMETER-ROLE-BINDING-v1"
    );
    assert_eq!(value["status"], "implementation-frozen-unqualified");
    assert_eq!(value["scientific_claim"], "NONE");
    assert_eq!(
        value["authority"],
        "optimizer-parameter-role-binding-only"
    );
    assert_eq!(value["runtime"]["cobaya_version"], "3.6.2");
    assert_eq!(
        value["runtime"]["cobaya_source_commit"],
        "899f30a49f85de610dac321e91a1af50018e56aa"
    );
    assert_eq!(
        value["runtime"]["role_engine"],
        "cobaya.parameterization.Parameterization"
    );

    let expected_script_sha = value["script_sha256"]
        .as_str()
        .expect("script_sha256 must be string");
    assert_eq!(sha256_hex(SCRIPT), expected_script_sha);

    let policy = &value["role_policy"];
    assert_eq!(
        policy["sampled_coordinates_source"],
        "complete ordered Parameterization.sampled_params() from the exact expanded configuration"
    );
    for key in [
        "manual_coordinate_additions_forbidden",
        "manual_coordinate_deletions_forbidden",
        "manual_coordinate_renames_forbidden",
        "manual_coordinate_reordering_forbidden",
        "every_sampled_coordinate_must_exist_in_bestfit_table",
        "named_review_does_not_prejudge_roles",
    ] {
        assert_eq!(policy[key], true, "{key} must remain true");
    }
    assert_eq!(
        policy["named_review_only"],
        serde_json::json!(["omm", "omegam", "hrdrag"])
    );

    let forbidden = &value["forbidden_operations"];
    for key in [
        "optimization",
        "sampler_execution",
        "likelihood_evaluation",
        "network",
        "configuration_mutation",
        "coordinate_selection_by_name",
    ] {
        assert_eq!(forbidden[key], true, "{key} must remain forbidden");
    }

    assert_eq!(value["promotion"]["a2_execution_authorized"], false);
}

#[test]
fn a2r_exact_configuration_hashes_remain_frozen() {
    let value: Value = serde_json::from_str(SPEC).expect("A2R specification must parse");
    let artifacts = &value["required_a0_artifacts"];

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
        (
            "reference-bestfit-text",
            902_u64,
            "bf8e35e2380ef35b137a77645dcb351af2ed2a93ca8da16c1fd71cb5dd7a1358",
        ),
    ];

    for (role, size, sha) in expected {
        assert_eq!(artifacts[role]["size"].as_u64(), Some(size));
        assert_eq!(artifacts[role]["sha256"].as_str(), Some(sha));
    }
}
