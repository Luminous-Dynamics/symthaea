// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Validate the version-pinned music benchmark registry profile.
//!
//! Run:
//!   cargo run -p symthaea-muse --example mel_benchmark_registry
//!
//! This validates a synthetic registry profile only. It does not run a model,
//! score an external benchmark, or emit an SOTA claim.

use serde_json::json;
use symthaea_muse::benchmark_registry::{
    validate_music_benchmark_profile_v1, MusicBenchmarkProfileV1,
};

const PROFILE_JSON: &str = include_str!(
    "../../../../benchmarks/mel/registry/internal-synthetic-profile-v1.json"
);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profile: MusicBenchmarkProfileV1 = serde_json::from_str(PROFILE_JSON)?;
    let issues = validate_music_benchmark_profile_v1(&profile);
    if !issues.is_empty() {
        return Err(std::io::Error::other(format!(
            "benchmark registry profile is invalid: {issues:?}"
        ))
        .into());
    }
    let profile_sha256 = profile.expected_profile_sha256()?;
    let report = json!({
        "schema_version": "mel-music-benchmark-profile-check-v1",
        "benchmark_id": profile.benchmark_id,
        "benchmark_version": profile.benchmark_version,
        "native_track_id": profile.native_track_id,
        "comparison_scope_id": profile.comparison_scope_id,
        "intended_use": profile.intended_use,
        "profile_sha256": profile_sha256,
        "valid": true,
        "model_run_performed": false,
        "benchmark_score_emitted": false
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
