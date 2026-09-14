// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent GEOM D0A3 golden-vector qualification.
//!
//! Expected values in the fixture were derived by the dependency-free Python
//! oracle in `scripts/oracles/geom_run_seal_v1_oracle.py` before this Rust test
//! was executed. This test intentionally uses only the public run-seal API.

use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use symthaea_geometric_run_seal::{
    RunEnvironmentSnapshot, canonical_json_commitment, inventory_evidence_directory,
    process_environment_commitment_from_entries, seal_run_environment,
};

const FIXTURE: &str = include_str!(
    "../../../../docs/research/vectors/GEOM_RUN_SEAL_V1.json"
);
static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("valid frozen GEOM run-seal fixture")
}

fn temp_root() -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "symthaea_geom_run_seal_oracle_{}_{}",
        std::process::id(),
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).expect("create isolated evidence root");
    root
}

fn decode_hex(value: &str) -> Vec<u8> {
    assert_eq!(value.len() % 2, 0, "hex input must have even length");
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).expect("fixture hex is ASCII");
            u8::from_str_radix(text, 16).expect("fixture hex byte")
        })
        .collect()
}

#[test]
fn independent_python_vectors_match_public_run_seal_api() {
    let fixture = fixture();

    // 1. Full process-environment commitment. Input order is deliberately B,A.
    let process = &fixture["process_environment"];
    let entries: Vec<(String, String)> = process["entries"]
        .as_array()
        .expect("entries array")
        .iter()
        .map(|entry| {
            let pair = entry.as_array().expect("environment pair");
            (
                pair[0].as_str().expect("environment key").to_string(),
                pair[1].as_str().expect("environment value").to_string(),
            )
        })
        .collect();
    let observed_process = process_environment_commitment_from_entries(entries);
    assert_eq!(
        observed_process,
        process["expected_commitment"]
            .as_str()
            .expect("process commitment")
    );

    // 2. Local canonical-JSON profile. Fixture source order is deliberately z,a.
    let canonical = &fixture["canonical_json"];
    let observed_json = canonical_json_commitment(
        canonical["domain"].as_str().expect("canonical domain"),
        &canonical["value"],
    )
    .expect("canonical JSON commitment");
    assert_eq!(
        observed_json,
        canonical["expected_commitment"]
            .as_str()
            .expect("canonical commitment")
    );

    // 3. Complete run-environment commitment, including feature normalization.
    let run = &fixture["run_environment"];
    let snapshot: RunEnvironmentSnapshot =
        serde_json::from_value(run["snapshot"].clone()).expect("run snapshot fixture");
    let sealed = seal_run_environment(snapshot, None).expect("seal frozen run snapshot");
    assert_eq!(
        sealed.commitment,
        run["expected_commitment"]
            .as_str()
            .expect("run commitment")
    );

    // 4. Real filesystem evidence inventory through the public API.
    let root = temp_root();
    let evidence = &fixture["evidence"];
    let mut expected_file_hashes = BTreeMap::new();
    for file in evidence["files"].as_array().expect("evidence files") {
        let relative = file["path"].as_str().expect("evidence relative path");
        let target = root.join(relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).expect("create evidence parent");
        }
        let content = decode_hex(file["content_hex"].as_str().expect("content hex"));
        fs::write(&target, content).expect("write evidence fixture");
        expected_file_hashes.insert(
            relative.to_string(),
            file["expected_blake3"]
                .as_str()
                .expect("file digest")
                .to_string(),
        );
    }

    let inventory = inventory_evidence_directory(&root).expect("inventory fixture evidence");
    assert_eq!(
        inventory.commitment,
        evidence["expected_inventory_commitment"]
            .as_str()
            .expect("inventory commitment")
    );
    assert_eq!(inventory.artifacts.len(), expected_file_hashes.len());
    for artifact in &inventory.artifacts {
        assert_eq!(
            expected_file_hashes.get(&artifact.relative_path),
            Some(&artifact.blake3),
            "independent file digest mismatch for {}",
            artifact.relative_path
        );
    }

    fs::remove_dir_all(root).ok();
}
