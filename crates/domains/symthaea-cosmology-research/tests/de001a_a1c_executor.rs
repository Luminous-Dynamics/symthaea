// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn a1c_executor_contract_is_frozen_and_self_consistent() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let spec_path = root.join("references/de001a_a1c_executor_v1.json");
    let spec_bytes = fs::read(&spec_path).expect("read executor spec");
    let spec: Value = serde_json::from_slice(&spec_bytes).expect("parse executor spec");

    assert_eq!(spec["schema_version"], 1);
    assert_eq!(spec["protocol"], "DE-001A1C-COBAYA-EXECUTOR-v1");
    assert_eq!(spec["status"], "implementation-frozen-unqualified");
    assert_eq!(spec["scientific_claim"], "NONE");
    assert_eq!(
        spec["authority"],
        "released-likelihood-fixed-point-reproduction-only"
    );
    assert_eq!(spec["result_protocol"], "DE-001A1C-COBAYA-RESULT-v1");

    let script_rel = spec["script_path"].as_str().expect("script path");
    let workspace_root = root.ancestors().nth(3).expect("workspace root");
    let script_path = workspace_root.join(script_rel);
    let script_bytes = fs::read(&script_path).expect("read executor script");
    assert_eq!(
        sha256_hex(&script_bytes),
        spec["script_sha256"].as_str().expect("script sha256")
    );

    assert_eq!(spec["authorization"]["required_verdict"], "PASS");
    assert_eq!(
        spec["authorization"]["required_authority"],
        "fixed-point-execution-authorization-only"
    );
    assert!(
        spec["authorization"]["required_a1c_execution_authorized"]
            .as_bool()
            .expect("authorization boolean")
    );

    assert_eq!(spec["subject"]["omega_m"], 0.29717787);
    assert_eq!(spec["subject"]["h_r_d_mpc"], 101.54786);
    assert_eq!(spec["subject"]["rdrag_gauge_mpc"], 100.0);
    assert_eq!(spec["subject"]["reference_chi2_bao"], 10.282299);
    assert_eq!(spec["subject"]["absolute_tolerance"], 0.01);
    assert_eq!(spec["subject"]["mean_size"], 472);
    assert_eq!(spec["subject"]["covariance_size"], 2547);
    assert_eq!(
        spec["subject"]["mean_sha256"],
        "9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585"
    );
    assert_eq!(
        spec["subject"]["covariance_sha256"],
        "252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509"
    );

    assert_eq!(spec["released_cobaya"]["version"], "3.6.2");
    assert_eq!(
        spec["released_cobaya"]["source_commit"],
        "899f30a49f85de610dac321e91a1af50018e56aa"
    );
    assert_eq!(
        spec["released_cobaya"]["likelihood_yaml_sha256"],
        "fd7e9bf2dcf5ffee90a9a30b18227f4337d6d5c1978782c63513cbe0d8280daa"
    );

    assert_eq!(spec["provider"]["integration"], "scipy.integrate.quad");
    assert!(
        spec["provider"]["independent_from_a1r_quadrature"]
            .as_bool()
            .expect("quadrature independence boolean")
    );
    assert_eq!(spec["execution_policy"]["likelihood_calls"], 1);
    for key in [
        "sampler_forbidden",
        "minimizer_forbidden",
        "optimization_forbidden",
        "parameter_mutation_forbidden",
        "network_forbidden",
        "camb_forbidden",
        "runtime_package_installation_forbidden",
        "tracked_worktree_must_be_clean_before_and_after",
    ] {
        assert!(
            spec["execution_policy"][key].as_bool().unwrap_or(false),
            "{key}"
        );
    }
    assert_eq!(spec["internal_chi2_tolerance"], 1.0e-10);
}
