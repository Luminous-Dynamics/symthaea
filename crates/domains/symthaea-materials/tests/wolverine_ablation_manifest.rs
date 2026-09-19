// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_materials::{
    WolverineAblationManifest, generate_wolverine_ablation_manifest,
};

const FROZEN_MANIFEST: &str = include_str!("../data/wolverine_ablation_v1.json");

#[test]
fn frozen_manifest_matches_generator_exactly() {
    let frozen: WolverineAblationManifest =
        serde_json::from_str(FROZEN_MANIFEST).expect("frozen MAT-004 manifest must remain valid JSON");
    let generated = generate_wolverine_ablation_manifest()
        .expect("MAT-004 generator must produce its frozen study subjects");
    assert_eq!(frozen, generated);
}

#[test]
fn frozen_manifest_contains_no_untracked_subjects() {
    let frozen: WolverineAblationManifest =
        serde_json::from_str(FROZEN_MANIFEST).expect("frozen MAT-004 manifest must remain valid JSON");
    assert_eq!(frozen.schema_version, 1);
    assert_eq!(frozen.generator_version, "mat-004-generator-v1");
    assert_eq!(frozen.candidates.len(), 8);
}
