// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic classification consistency tests
//!
//! Verifies that the Rust SDK produces the canonical
//! classification codes from `shared/epistemic_spec.toml`.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use mycelix_sdk::epistemic::{
    EmpiricalLevel, EpistemicClassification, MaterialityLevel, NormativeLevel,
};

fn load_spec() -> HashMap<String, String> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path: PathBuf = [manifest_dir, "..", "shared", "epistemic_spec.toml"]
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
        let key = parts.next().map(str::trim).unwrap_or_default();
        let value = parts
            .next()
            .map(str::trim)
            .unwrap_or_default()
            .trim_matches('"');

        if key.is_empty() || value.is_empty() {
            continue;
        }

        values.insert(key.to_string(), value.to_string());
    }

    values
}

#[test]
fn epistemic_codes_match_shared_spec() {
    let spec = load_spec();

    let basic = EpistemicClassification::new(
        EmpiricalLevel::E1Testimonial,
        NormativeLevel::N1Communal,
        MaterialityLevel::M1Temporal,
    );
    assert_eq!(basic.to_code(), spec["basic"]);

    let financial = EpistemicClassification::new(
        EmpiricalLevel::E3Cryptographic,
        NormativeLevel::N2Network,
        MaterialityLevel::M2Persistent,
    );
    assert_eq!(financial.to_code(), spec["financial"]);

    let foundational = EpistemicClassification::new(
        EmpiricalLevel::E4PublicRepro,
        NormativeLevel::N3Axiomatic,
        MaterialityLevel::M3Foundational,
    );
    assert_eq!(foundational.to_code(), spec["foundational"]);
}
