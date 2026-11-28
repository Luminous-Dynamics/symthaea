#![forbid(unsafe_code)]

//! Contract-only validation for the first MEL-HEAR-011 auditory evidence fixture.
//!
//! This crate does not implement an audio front end. It verifies provenance,
//! sample metadata, representation declarations and claim-state boundaries.

pub mod dsp;

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const SCHEMA: &str =
    include_str!("../../../benchmarks/mel/schemas/mel-hear-011-auditory-evidence-v0-1.schema.json");

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FixtureSummary {
    pub record_id: String,
    pub source_frames: usize,
    pub view_count: usize,
    pub materialized_view_count: usize,
    pub evidence_status: String,
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, String> {
    value
        .get(name)
        .ok_or_else(|| format!("missing required field: {name}"))
}

fn string<'a>(value: &'a Value, name: &str) -> Result<&'a str, String> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| format!("field {name} must be a string"))
}

fn positive_usize(value: &Value, name: &str) -> Result<usize, String> {
    let number = field(value, name)?
        .as_u64()
        .ok_or_else(|| format!("field {name} must be a positive integer"))?;
    usize::try_from(number)
        .ok()
        .filter(|number| *number > 0)
        .ok_or_else(|| format!("field {name} must fit a positive usize"))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_source_asset(source: &Value, asset: &[u8]) -> Result<usize, String> {
    let expected_digest = string(source, "artifact_sha256")?;
    if !is_sha256(expected_digest) {
        return Err("source artifact_sha256 must be lowercase SHA-256".to_owned());
    }
    if sha256_hex(asset) != expected_digest {
        return Err("source artifact digest does not match supplied bytes".to_owned());
    }
    if string(source, "sample_format")? != "f32le_hex_ascii_v1" {
        return Err("v0 fixture validator accepts only f32le_hex_ascii_v1".to_owned());
    }

    let sample_count = asset
        .split_ascii_whitespace()
        .try_fold(0usize, |count, sample| {
            if sample.len() != 8 || !sample.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err("each f32le hex sample must be exactly eight hex digits".to_owned());
            }
            count
                .checked_add(1)
                .ok_or_else(|| "sample count overflow".to_owned())
        })?;
    let frame_count = positive_usize(source, "frame_count")?;
    let channel_count = positive_usize(source, "channel_count")?;
    let expected_samples = frame_count
        .checked_mul(channel_count)
        .ok_or_else(|| "source sample count overflow".to_owned())?;
    if sample_count != expected_samples {
        return Err("source frame_count/channel_count do not match sample asset".to_owned());
    }

    let calibration = string(source, "calibration_state")?;
    let amplitude_unit = string(source, "amplitude_unit")?;
    if calibration == "uncalibrated_digital" && amplitude_unit != "digital_full_scale_ratio" {
        return Err("uncalibrated digital samples cannot claim SPL or pressure units".to_owned());
    }
    Ok(frame_count)
}

fn validate_views(record: &Value) -> Result<(usize, usize), String> {
    let views = field(record, "views")?
        .as_array()
        .ok_or_else(|| "views must be an array".to_owned())?;
    if views.len() < 3 {
        return Err("at least three independent representation views are required".to_owned());
    }

    let mut ids = BTreeSet::new();
    let mut kinds = BTreeSet::new();
    let mut materialized = 0usize;

    for view in views {
        let view_id = string(view, "view_id")?;
        let kind = string(view, "kind")?;
        let status = string(view, "status")?;
        if !ids.insert(view_id.to_owned()) {
            return Err("duplicate view_id".to_owned());
        }
        if !kinds.insert(kind.to_owned()) {
            return Err("duplicate representation kind".to_owned());
        }

        match status {
            "planned" | "not_estimable" => {
                if view.get("artifact_sha256").is_some()
                    || view.get("artifact_locator").is_some()
                    || view.get("shape").is_some()
                {
                    return Err("unmaterialized views cannot carry artifact claims".to_owned());
                }
            }
            "materialized" => {
                let digest = string(view, "artifact_sha256")?;
                if !is_sha256(digest) {
                    return Err("materialized view digest is not lowercase SHA-256".to_owned());
                }
                let shape = field(view, "shape")?
                    .as_array()
                    .ok_or_else(|| "materialized view shape must be an array".to_owned())?;
                if shape.is_empty()
                    || shape
                        .iter()
                        .any(|dimension| dimension.as_u64().is_none_or(|value| value == 0))
                {
                    return Err("materialized view dimensions must be positive integers".to_owned());
                }
                string(view, "artifact_locator")?;
                materialized += 1;
            }
            _ => return Err("unknown view status".to_owned()),
        }

        match kind {
            "complex_stft" if string(view, "phase_policy")? != "complex_retained" => {
                return Err("complex STFT view must declare phase retention".to_owned());
            }
            "erb_band_energy" if string(view, "frequency_scale")? != "erb" => {
                return Err("ERB energy view must declare its ERB frequency scale".to_owned());
            }
            "constant_q" | "log_frequency_energy"
                if string(view, "frequency_scale")? != "log_frequency" =>
            {
                return Err("log-frequency views must declare log-frequency scale".to_owned());
            }
            _ => {}
        }
    }

    for required in ["waveform", "complex_stft", "erb_band_energy", "log_frequency_energy"] {
        if !kinds.contains(required) {
            return Err(format!(
                "required representation view is absent: {required}"
            ));
        }
    }
    Ok((views.len(), materialized))
}

/// Validate the metadata-only known-signal fixture and its exact source artifact.
pub fn validate_fixture(record: &Value, source_asset: &[u8]) -> Result<FixtureSummary, String> {
    let schema: Value = serde_json::from_str(SCHEMA)
        .map_err(|error| format!("committed schema is invalid JSON: {error}"))?;
    if string(&schema, "$schema")? != "https://json-schema.org/draft/2020-12/schema" {
        return Err("unsupported JSON Schema dialect".to_owned());
    }

    if string(record, "schema_version")? != "mel-hear-011-auditory-evidence-v0.1" {
        return Err("unsupported auditory evidence schema version".to_owned());
    }
    let record_id = string(record, "record_id")?.to_owned();
    let evidence_status = string(record, "evidence_status")?.to_owned();
    let source = field(record, "source")?;
    let source_frames = validate_source_asset(source, source_asset)?;

    let analysis = field(record, "analysis")?;
    let analysis_rate = positive_usize(analysis, "sample_rate_hz")?;
    let source_rate = positive_usize(source, "sample_rate_hz")?;
    if analysis_rate != source_rate {
        return Err("analysis and source sample rates differ".to_owned());
    }
    let window = positive_usize(analysis, "window_samples")?;
    let hop = positive_usize(analysis, "hop_samples")?;
    let fft_size = positive_usize(analysis, "fft_size")?;
    if window > source_frames || hop > window || fft_size < window {
        return Err("analysis window/hop/FFT geometry is invalid for the source".to_owned());
    }

    let (view_count, materialized_view_count) = validate_views(record)?;
    let events = field(record, "event_hypotheses")?
        .as_array()
        .ok_or_else(|| "event_hypotheses must be an array".to_owned())?;
    let scenes = field(record, "scene_hypotheses")?
        .as_array()
        .ok_or_else(|| "scene_hypotheses must be an array".to_owned())?;
    let claims = field(record, "query_claims")?
        .as_array()
        .ok_or_else(|| "query_claims must be an array".to_owned())?;
    let causal = field(record, "causal_factor_assessment")?;
    let identifiability = field(record, "identifiability")?;

    if evidence_status == "metadata_fixture_only" {
        if !events.is_empty() || !scenes.is_empty() || !claims.is_empty() {
            return Err(
                "metadata-only fixture cannot make event, scene or query claims".to_owned(),
            );
        }
        if materialized_view_count != 0 {
            return Err("metadata-only fixture cannot claim materialized feature views".to_owned());
        }
        if string(causal, "status")? != "not_assessed"
            || string(identifiability, "status")? != "not_assessed"
        {
            return Err("metadata-only fixture cannot claim causal identification".to_owned());
        }
    }

    Ok(FixtureSummary {
        record_id,
        source_frames,
        view_count,
        materialized_view_count,
        evidence_status,
    })
}

#[cfg(test)]
mod tests {
    use super::{FixtureSummary, validate_fixture};
    use serde_json::{Value, json};

    const FIXTURE: &str = include_str!("../fixtures/auditory-evidence-v0.1.json");
    const SOURCE_ASSET: &[u8] = include_bytes!("../fixtures/synthetic-tone-f32le-hex.txt");

    fn fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("fixture JSON must parse")
    }

    #[test]
    fn metadata_fixture_validates_without_claiming_signal_analysis() {
        let report = validate_fixture(&fixture(), SOURCE_ASSET).unwrap();
        assert_eq!(
            report,
            FixtureSummary {
                record_id: "synthetic-tone-001".to_owned(),
                source_frames: 16,
                view_count: 4,
                materialized_view_count: 0,
                evidence_status: "metadata_fixture_only".to_owned(),
            }
        );
    }

    #[test]
    fn changed_source_bytes_fail_the_digest_check() {
        let changed = b"tampered source bytes";
        assert!(
            validate_fixture(&fixture(), changed)
                .unwrap_err()
                .contains("digest")
        );
    }

    #[test]
    fn declared_frame_count_must_match_the_sample_asset() {
        let mut record = fixture();
        record["source"]["frame_count"] = json!(15);
        assert!(
            validate_fixture(&record, SOURCE_ASSET)
                .unwrap_err()
                .contains("frame_count")
        );
    }

    #[test]
    fn uncalibrated_digital_values_cannot_claim_sound_pressure() {
        let mut record = fixture();
        record["source"]["amplitude_unit"] = json!("dB_SPL");
        assert!(
            validate_fixture(&record, SOURCE_ASSET)
                .unwrap_err()
                .contains("cannot claim SPL")
        );
    }

    #[test]
    fn complex_stft_must_preserve_phase_by_contract() {
        let mut record = fixture();
        record["views"][1]["phase_policy"] = json!("magnitude_only");
        assert!(
            validate_fixture(&record, SOURCE_ASSET)
                .unwrap_err()
                .contains("phase retention")
        );
    }

    #[test]
    fn metadata_only_records_cannot_claim_events_or_scenes() {
        let mut record = fixture();
        record["event_hypotheses"] = json!([{"event_id":"invented"}]);
        assert!(
            validate_fixture(&record, SOURCE_ASSET)
                .unwrap_err()
                .contains("cannot make event")
        );
    }

    #[test]
    fn planned_views_cannot_smuggle_unverified_artifacts() {
        let mut record = fixture();
        record["views"][0]["artifact_sha256"] = json!("0".repeat(64));
        assert!(
            validate_fixture(&record, SOURCE_ASSET)
                .unwrap_err()
                .contains("unmaterialized views")
        );
    }

    #[test]
    fn duplicate_representation_kinds_are_rejected() {
        let mut record = fixture();
        record["views"][3]["kind"] = json!("erb_band_energy");
        assert!(
            validate_fixture(&record, SOURCE_ASSET)
                .unwrap_err()
                .contains("duplicate representation kind")
        );
    }
}
