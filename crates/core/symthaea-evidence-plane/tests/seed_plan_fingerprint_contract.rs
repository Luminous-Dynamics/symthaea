// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent conformance check for the public SeedPlan fingerprint-v1 byte contract.
//!
//! This test deliberately reconstructs the canonical byte stream without calling
//! the implementation's private framing helpers. It is not an independent BLAKE3
//! implementation; it proves that `SeedPlan::fingerprint()` agrees with the
//! documented v1 domain/version/length/ordering contract.

use symthaea_evidence_plane::seed_plan::{SeedPlan, SEED_PLAN_FINGERPRINT_VERSION};

#[test]
fn seed_plan_fingerprint_matches_v1_reference_framing() {
    let plan = SeedPlan::register(
        vec![107, 100, 105, 102, 101, 106, 104, 103],
        vec![3, 1, 2],
    )
    .expect("reference seed plan must be valid");

    // Reconstruct the documented v1 stream independently of fingerprint().
    let mut reference = Vec::new();
    reference.extend_from_slice(b"symthaea-seed-plan");
    reference.push(SEED_PLAN_FINGERPRINT_VERSION);

    let confirmatory = [100_u64, 101, 102, 103, 104, 105, 106, 107];
    reference.extend_from_slice(&(confirmatory.len() as u64).to_le_bytes());
    for seed in confirmatory {
        reference.extend_from_slice(&seed.to_le_bytes());
    }

    let development = [1_u64, 2, 3];
    reference.extend_from_slice(&(development.len() as u64).to_le_bytes());
    for seed in development {
        reference.extend_from_slice(&seed.to_le_bytes());
    }

    let expected = format!("blake3:{}", blake3::hash(&reference).to_hex());
    assert_eq!(plan.fingerprint(), expected);
}

#[test]
fn seed_plan_fingerprint_algorithm_prefix_is_not_optional_metadata() {
    let plan = SeedPlan::register((100..108).collect(), vec![1, 2, 3]).expect("valid");
    let fingerprint = plan.fingerprint();

    assert!(
        fingerprint.starts_with("blake3:"),
        "persisted durable identity must declare the digest algorithm"
    );
    assert_eq!(fingerprint.len(), "blake3:".len() + 64);
}
