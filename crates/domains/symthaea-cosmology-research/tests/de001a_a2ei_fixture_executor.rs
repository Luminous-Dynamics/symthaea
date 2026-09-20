// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Static authority ratchets for the DE-001A2EI fixture executor tranche.

use serde_json::Value;

const SPEC: &str = include_str!("../references/de001a_a2ei_fixture_executor_v1.json");
const EXECUTOR: &str = include_str!("../src/bin/de001a-a2ei-executor.rs");
const FIXTURE: &str = include_str!("../src/bin/de001a-a2ei-fixture.rs");

#[test]
fn fixture_executor_manifest_remains_non_authoritative() {
    let value: Value = serde_json::from_str(SPEC).expect("fixture executor spec JSON");
    assert_eq!(value["schema_version"].as_u64(), Some(1));
    assert_eq!(
        value["protocol"].as_str(),
        Some("DE-001A2EI-FIXTURE-EXECUTOR-v1")
    );
    assert_eq!(value["status"].as_str(), Some("implementation-frozen-unqualified"));
    assert_eq!(value["scientific_claim"].as_str(), Some("NONE"));
    assert_eq!(
        value["authority"].as_str(),
        Some("fixture-executor-implementation-only")
    );
    for key in [
        "executor_implementation_qualified",
        "optimizer_execution_authorized",
        "a2_execution_authorized",
        "real_optimizer_execution_authorized",
        "a2q_execution_authorized",
    ] {
        assert_eq!(value[key].as_bool(), Some(false), "{key} drifted");
    }
    assert_eq!(value["reproduction_verdict"].as_str(), Some("UNASSESSED"));
    assert_eq!(
        value["process_policy"]["process_invocations_per_executor_process"].as_u64(),
        Some(1)
    );
    assert_eq!(
        value["process_policy"]["automatic_retry_allowed"].as_bool(),
        Some(false)
    );
    assert_eq!(
        value["promotion_boundary"]["fixture_execution_establishes_network_isolation"]
            .as_bool(),
        Some(false)
    );
}

#[test]
fn executor_has_one_spawn_path_and_no_scientific_target_constants() {
    assert_eq!(EXECUTOR.matches(".spawn()").count(), 1);
    assert!(EXECUTOR.contains(".env_clear()"));
    assert!(EXECUTOR.contains("reproduction_verdict: \"UNASSESSED\""));
    assert!(EXECUTOR.contains("network_isolation_proven: false"));
    assert!(!EXECUTOR.contains("Command::output"));

    for forbidden in [
        "10.282299",
        "0.29717936",
        "101.54786",
        "chi2__BAO",
        "omegam",
        "hrdrag",
    ] {
        assert!(
            !EXECUTOR.contains(forbidden),
            "fixture executor embedded forbidden scientific target {forbidden:?}"
        );
    }
}

#[test]
fn inert_fixture_contains_no_cosmology_or_optimizer_surface() {
    for forbidden in [
        "cobaya",
        "iminuit",
        "likelihood",
        "chi2__BAO",
        "omegam",
        "hrdrag",
        "DESI",
    ] {
        assert!(
            !FIXTURE.contains(forbidden),
            "inert fixture contains forbidden surface {forbidden:?}"
        );
    }
    assert!(FIXTURE.contains("process::exit(23)"));
    assert!(FIXTURE.contains("fixture-result-success-v1"));
    assert!(FIXTURE.contains("fixture-result-nonzero-v1"));
}
