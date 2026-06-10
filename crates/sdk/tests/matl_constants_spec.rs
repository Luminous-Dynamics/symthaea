// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MATL constant consistency tests
//!
//! Ensures the Rust SDK constants match the canonical values
//! defined in `shared/matl_constants.toml` at the workspace root.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use mycelix_sdk::matl::{
    DEFAULT_BYZANTINE_THRESHOLD, DEFAULT_CONSISTENCY_WEIGHT, DEFAULT_QUALITY_WEIGHT,
    DEFAULT_REPUTATION_WEIGHT, MAX_BYZANTINE_TOLERANCE,
};

fn load_spec() -> HashMap<String, f64> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path: PathBuf = [manifest_dir, "..", "shared", "matl_constants.toml"]
        .iter()
        .collect();

    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));

    let mut values = HashMap::new();

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.splitn(2, '=');
        let key = parts
            .next()
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .unwrap_or_else(|| panic!("Invalid line in spec: {}", line));
        let value_str = parts
            .next()
            .map(str::trim)
            .unwrap_or_else(|| panic!("Missing value in spec line: {}", line));

        let value: f64 = value_str
            .parse()
            .unwrap_or_else(|e| panic!("Failed to parse value for {}: {}", key, e));

        values.insert(key.to_string(), value);
    }

    values
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < 1e-9,
        "Mismatch for {}: expected {}, got {}, diff {}",
        label,
        expected,
        actual,
        diff
    );
}

#[test]
fn matl_constants_match_shared_spec() {
    let spec = load_spec();

    assert_close(
        DEFAULT_QUALITY_WEIGHT,
        spec["default_quality_weight"],
        "DEFAULT_QUALITY_WEIGHT",
    );
    assert_close(
        DEFAULT_CONSISTENCY_WEIGHT,
        spec["default_consistency_weight"],
        "DEFAULT_CONSISTENCY_WEIGHT",
    );
    assert_close(
        DEFAULT_REPUTATION_WEIGHT,
        spec["default_reputation_weight"],
        "DEFAULT_REPUTATION_WEIGHT",
    );
    assert_close(
        MAX_BYZANTINE_TOLERANCE,
        spec["max_byzantine_tolerance"],
        "MAX_BYZANTINE_TOLERANCE",
    );
    assert_close(
        DEFAULT_BYZANTINE_THRESHOLD,
        spec["default_byzantine_threshold"],
        "DEFAULT_BYZANTINE_THRESHOLD",
    );
}
