use mel_benchmark_registry_harness::{
    validate_music_benchmark_profile_v1, MusicBenchmarkProfileV1,
};
use serde_json::json;
use std::error::Error;

const PROFILE_JSON: &str = include_str!("../fixtures/internal-synthetic-profile-v1.json");

fn main() -> Result<(), Box<dyn Error>> {
    let profile: MusicBenchmarkProfileV1 = serde_json::from_str(PROFILE_JSON)?;
    let issues = validate_music_benchmark_profile_v1(&profile);
    if !issues.is_empty() {
        return Err(std::io::Error::other(format!(
            "benchmark registry profile is invalid: {issues:?}"
        ))
        .into());
    }
    let profile_sha256 = profile
        .expected_profile_sha256()
        .map_err(std::io::Error::other)?;
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
