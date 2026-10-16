// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate a generated Muse SFZ compatibility fixture pack against the
//! current SFZ parser and round-robin selection primitive.
//!
//! Example:
//!   cargo run --manifest-path crates/domains/symthaea-muse/Cargo.toml --bin sfz_compat_fixture_validate -- --pack-dir /tmp/muse-sfz-compat-v1
//!
//! This checks fixture/source/asset integrity and declared parser/selection
//! expectations. It does not render audio, invoke a reference SFZ player, or
//! convert an authored expectation into evidence of universal SFZ compatibility.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env,
    error::Error,
    fs,
    io,
    path::{Component, Path, PathBuf},
    process::ExitCode,
};
use symthaea_muse::sfz_manifest::{
    inspect_sample_reference, parse_sfz_mapping, SfzMappingManifestV1,
    SfzRoundRobinSequencerV1, SfzZoneSelectionV1,
};

const EXPECTED_PACK_SCHEMA: &str = "muse-sfz-compat-fixture-pack-v1";

#[derive(Debug, Deserialize)]
struct FixturePackV1 {
    schema: String,
    fixtures: Vec<FixtureRecordV1>,
}

#[derive(Debug, Deserialize)]
struct FixtureRecordV1 {
    fixture_id: String,
    sfz_file: String,
    sfz_sha256: String,
    sfz_byte_len: usize,
    parser_expectation: String,
    sample_assets: Vec<AssetRecordV1>,
    timeline: Vec<TimelineExpectationV1>,
    negative_control: bool,
    unsupported_semantics: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AssetRecordV1 {
    path: String,
    sha256: String,
    byte_len: u64,
    sample_rate_hz: u32,
    channels: u16,
    frames: usize,
    encoding: String,
}

#[derive(Debug, Deserialize)]
struct TimelineExpectationV1 {
    ordinal: usize,
    event: String,
    stream_id: String,
    midi_key: u8,
    velocity: u8,
    expected_selection_decision: String,
    expected_region_index: Option<usize>,
    expected_sample_path: Option<String>,
    expected_render_outcome: String,
}

#[derive(Debug, Serialize)]
struct ValidationReportV1 {
    schema: &'static str,
    pack_schema: String,
    fixture_pack_digest: String,
    fixture_count: usize,
    fixtures_passed: usize,
    fixtures_failed: usize,
    negative_controls_seen: usize,
    negative_controls_passed: usize,
    passed: bool,
    evidence_boundary: &'static str,
    fixtures: Vec<FixtureValidationV1>,
    failures: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FixtureValidationV1 {
    fixture_id: String,
    negative_control: bool,
    sfz_sha256_verified: bool,
    parser_outcome: String,
    unhandled_opcodes: Vec<String>,
    asset_checks: Vec<AssetValidationV1>,
    timeline_checks: Vec<TimelineValidationV1>,
    passed: bool,
    failures: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AssetValidationV1 {
    path: String,
    sha256_verified: bool,
    byte_len_verified: bool,
    metadata_verified: bool,
    safe_regular_file: bool,
    passed: bool,
    failure: Option<String>,
}

#[derive(Debug, Serialize)]
struct TimelineValidationV1 {
    ordinal: usize,
    event: String,
    stream_id: String,
    expected_decision: String,
    actual_decision: Option<String>,
    expected_region_index: Option<usize>,
    actual_region_index: Option<usize>,
    expected_sample_path: Option<String>,
    actual_sample_path: Option<String>,
    matched: bool,
    failure: Option<String>,
}

fn main() -> ExitCode {
    match run() {
        Ok(report) => {
            let passed = report.passed;
            match serde_json::to_string_pretty(&report) {
                Ok(json) => println!("{json}"),
                Err(error) => {
                    eprintln!("sfz_compat_fixture_validate: cannot serialize receipt: {error}");
                    return ExitCode::from(2);
                }
            }
            if passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(error) => {
            eprintln!("sfz_compat_fixture_validate: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ValidationReportV1, Box<dyn Error>> {
    let pack_dir = parse_pack_dir()?;
    let root_meta = fs::symlink_metadata(&pack_dir)?;
    if root_meta.file_type().is_symlink() || !root_meta.file_type().is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "pack directory must be a real directory, not a symlink",
        )
        .into());
    }
    let root = fs::canonicalize(&pack_dir)?;
    let manifest_path = safe_join(&root, "fixtures.json")?;
    let pack_bytes = fs::read(&manifest_path)?;
    let pack: FixturePackV1 = serde_json::from_slice(&pack_bytes)?;
    if pack.schema != EXPECTED_PACK_SCHEMA {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported fixture pack schema {:?}; expected {:?}",
                pack.schema, EXPECTED_PACK_SCHEMA
            ),
        )
        .into());
    }

    let mut root_failures = Vec::new();
    let computed_digest = pack_tree_sha256(&root)?;
    let digest_path = safe_join(&root, "pack.sha256")?;
    let declared_digest = match fs::read_to_string(&digest_path) {
        Ok(value) => value.split_whitespace().next().unwrap_or("").to_owned(),
        Err(error) => {
            root_failures.push(format!("cannot read pack.sha256: {error}"));
            String::new()
        }
    };
    if declared_digest.is_empty() || declared_digest != computed_digest {
        root_failures.push(format!(
            "fixture pack tree digest mismatch: declared {:?}, computed {}",
            declared_digest, computed_digest
        ));
    }

    let mut seen_fixture_ids = BTreeSet::new();
    let mut fixtures = Vec::with_capacity(pack.fixtures.len());
    for fixture in &pack.fixtures {
        if !seen_fixture_ids.insert(fixture.fixture_id.clone()) {
            root_failures.push(format!("duplicate fixture ID {:?}", fixture.fixture_id));
        }
        fixtures.push(validate_fixture(&root, fixture)?);
    }

    let fixtures_passed = fixtures.iter().filter(|fixture| fixture.passed).count();
    let fixtures_failed = fixtures.len() - fixtures_passed;
    let negative_controls_seen = fixtures.iter().filter(|fixture| fixture.negative_control).count();
    let negative_controls_passed = fixtures
        .iter()
        .filter(|fixture| fixture.negative_control && fixture.passed)
        .count();
    let passed = root_failures.is_empty() && fixtures_failed == 0;
    Ok(ValidationReportV1 {
        schema: "muse-sfz-compat-fixture-validation-report-v1",
        pack_schema: pack.schema,
        fixture_pack_digest: computed_digest,
        fixture_count: fixtures.len(),
        fixtures_passed,
        fixtures_failed,
        negative_controls_seen,
        negative_controls_passed,
        passed,
        evidence_boundary: "validates generated source/asset hashes and authored parser/selection expectations against this Muse revision; does not render audio or establish reference-engine compatibility",
        fixtures,
        failures: root_failures,
    })
}

fn parse_pack_dir() -> Result<PathBuf, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let flag = args.next();
    let path = args.next();
    if flag.as_deref() == Some("--pack-dir") {
        if let Some(path) = path {
            if args.next().is_none() {
                return Ok(PathBuf::from(path));
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "usage: sfz_compat_fixture_validate --pack-dir FIXTURE_PACK_DIRECTORY",
    )
    .into())
}

fn safe_join(root: &Path, relative: &str) -> Result<PathBuf, Box<dyn Error>> {
    let candidate = Path::new(relative);
    if relative.is_empty()
        || relative.contains('\\')
        || candidate.is_absolute()
        || candidate
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsafe relative pack path {relative:?}"),
        )
        .into());
    }
    Ok(root.join(candidate))
}

fn validate_fixture(
    root: &Path,
    fixture: &FixtureRecordV1,
) -> Result<FixtureValidationV1, Box<dyn Error>> {
    let mut failures = Vec::new();
    let sfz_path = safe_join(root, &fixture.sfz_file)?;
    let sfz_bytes = fs::read(&sfz_path)?;
    let actual_sfz_sha = sha256_hex(&sfz_bytes);
    let sfz_sha256_verified = actual_sfz_sha == fixture.sfz_sha256;
    if !sfz_sha256_verified {
        failures.push(format!(
            "SFZ hash mismatch: expected {}, got {}",
            fixture.sfz_sha256, actual_sfz_sha
        ));
    }
    if sfz_bytes.len() != fixture.sfz_byte_len {
        failures.push(format!(
            "SFZ byte length mismatch: expected {}, got {}",
            fixture.sfz_byte_len,
            sfz_bytes.len()
        ));
    }
    let sfz_source = std::str::from_utf8(&sfz_bytes)?;
    let parse_result = parse_sfz_mapping(sfz_source);
    if let Ok(parsed_manifest) = &parse_result {
        if parsed_manifest.sfz_sha256 != fixture.sfz_sha256 {
            failures.push(format!(
                "parser manifest SFZ hash mismatch: expected {}, got {}",
                fixture.sfz_sha256, parsed_manifest.sfz_sha256
            ));
        }
    }
    let parser_outcome = match (&fixture.parser_expectation[..], &parse_result) {
        ("parse_success", Ok(_)) => "parse_success",
        ("parse_success_with_unhandled_opcode", Ok(manifest))
            if !manifest.unhandled_opcodes.is_empty() =>
        {
            "parse_success_with_unhandled_opcode"
        }
        ("parse_success_with_unhandled_opcodes", Ok(manifest))
            if !manifest.unhandled_opcodes.is_empty() =>
        {
            "parse_success_with_unhandled_opcodes"
        }
        ("parse_rejected_path_escape", Err(error))
            if error.to_string().contains("escapes the SFZ/sample root") =>
        {
            "parse_rejected_path_escape"
        }
        (_, Ok(_)) => {
            failures.push(format!(
                "parser expectation {:?} did not match successful parsing",
                fixture.parser_expectation
            ));
            "unexpected_parse_success"
        }
        (_, Err(error)) => {
            failures.push(format!(
                "parser expectation {:?} failed: {error}",
                fixture.parser_expectation
            ));
            "unexpected_parse_failure"
        }
    };

    let fixture_dir = sfz_path.parent().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "SFZ file has no parent directory")
    })?;
    let mut asset_checks = Vec::with_capacity(fixture.sample_assets.len());
    for asset in &fixture.sample_assets {
        asset_checks.push(validate_asset(fixture_dir, asset));
    }
    for asset in &asset_checks {
        if !asset.passed {
            failures.push(format!("asset {}: {}", asset.path, asset.failure.as_deref().unwrap_or("validation failed")));
        }
    }

    let manifest = parse_result.ok();
    let unhandled_opcodes = manifest
        .as_ref()
        .map(|value| value.unhandled_opcodes.clone())
        .unwrap_or_default();
    for semantic in &fixture.unsupported_semantics {
        let opcode = if let Some((key, _)) = semantic.split_once('=') {
            Some(key)
        } else if !semantic.is_empty()
            && !semantic.chars().any(char::is_whitespace)
            && semantic.chars().all(|character| character.is_ascii_alphanumeric() || character == '_')
        {
            Some(semantic.as_str())
        } else {
            None
        };
        if let Some(opcode) = opcode {
            if !unhandled_opcodes.iter().any(|actual| actual == opcode) {
                failures.push(format!(
                    "declared unsupported opcode {opcode:?} was not reported as unhandled by the parser"
                ));
            }
        }
    }
    let mut timeline_checks = Vec::with_capacity(fixture.timeline.len());
    if let Some(manifest) = manifest.as_ref() {
        let timeline_results = validate_timeline(manifest, &fixture.timeline);
        for check in &timeline_results {
            if !check.matched {
                failures.push(format!(
                    "timeline event {} ({}) failed: {}",
                    check.ordinal,
                    check.event,
                    check.failure.as_deref().unwrap_or("expectation mismatch")
                ));
            }
        }
        timeline_checks = timeline_results;
    } else if !fixture.timeline.is_empty() {
        failures.push("fixture declares events even though parsing was rejected".to_owned());
    }

    // Missing-sample fixture is intentionally allowed to reference one absent
    // asset. Every other timeline asset reference must be represented by the
    // declared fixture asset list.
    if fixture.fixture_id == "missing_sample_asset" {
        match safe_join(fixture_dir, "intentionally_missing.wav") {
            Ok(path) => match fs::symlink_metadata(&path) {
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Ok(_) => failures.push("the intentionally missing sample unexpectedly exists".to_owned()),
                Err(error) => failures.push(format!("cannot verify intentionally missing sample: {error}")),
            },
            Err(error) => failures.push(format!("invalid intentionally-missing sample path: {error}")),
        }
    }
    for step in &fixture.timeline {
        if let Some(sample_path) = step.expected_sample_path.as_deref() {
            let declared = fixture.sample_assets.iter().any(|asset| asset.path == sample_path);
            let intentional_missing =
                fixture.fixture_id == "missing_sample_asset"
                    && sample_path == "intentionally_missing.wav"
                    && step.expected_render_outcome == "asset_missing_render_failure_expected";
            if !declared && !intentional_missing {
                failures.push(format!(
                    "timeline event {} references undeclared asset {:?}",
                    step.ordinal, sample_path
                ));
            }
        }
    }

    Ok(FixtureValidationV1 {
        fixture_id: fixture.fixture_id.clone(),
        negative_control: fixture.negative_control,
        sfz_sha256_verified,
        parser_outcome: parser_outcome.to_owned(),
        unhandled_opcodes,
        asset_checks,
        timeline_checks,
        passed: failures.is_empty(),
        failures,
    })
}

fn validate_asset(root: &Path, asset: &AssetRecordV1) -> AssetValidationV1 {
    let fail = |message: String| AssetValidationV1 {
        path: asset.path.clone(),
        sha256_verified: false,
        byte_len_verified: false,
        metadata_verified: false,
        safe_regular_file: false,
        passed: false,
        failure: Some(message),
    };

    let path = match safe_join(root, &asset.path) {
        Ok(path) => path,
        Err(error) => return fail(error.to_string()),
    };
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) => return fail(error.to_string()),
    };
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return fail("asset is not a regular non-symlink file".to_owned());
    }
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => return fail(error.to_string()),
    };
    let sha_ok = sha256_hex(&bytes) == asset.sha256;
    let len_ok = bytes.len() as u64 == asset.byte_len;
    let safe_sample = matches!(
        inspect_sample_reference(root, &asset.path, true),
        Ok(Some((len, Some(ref digest)))) if len == asset.byte_len && digest == &asset.sha256
    );
    let metadata_ok = asset.sample_rate_hz == 44_100
        && asset.channels == 1
        && asset.frames > 0
        && asset.encoding == "RIFF_WAVE_PCM_S16LE"
        && wav_header_matches(&bytes, asset);
    let failure = if !sha_ok {
        Some("asset SHA-256 mismatch".to_owned())
    } else if !len_ok {
        Some("asset byte length mismatch".to_owned())
    } else if !safe_sample {
        Some("asset is missing, unsafe, or rejected by sample-root inspection".to_owned())
    } else if !metadata_ok {
        Some("declared audio metadata does not match this fixture-pack profile".to_owned())
    } else {
        None
    };
    AssetValidationV1 {
        path: asset.path.clone(),
        sha256_verified: sha_ok,
        byte_len_verified: len_ok,
        metadata_verified: metadata_ok,
        safe_regular_file: safe_sample,
        passed: failure.is_none(),
        failure,
    }
}

fn validate_timeline(
    manifest: &SfzMappingManifestV1,
    timeline: &[TimelineExpectationV1],
) -> Vec<TimelineValidationV1> {
    let mut sequencer = SfzRoundRobinSequencerV1::new();
    let mut results = Vec::new();
    for (index, step) in timeline.iter().enumerate() {
        let mut failure = None;
        if step.ordinal != index + 1 {
            failure = Some("timeline ordinal is not contiguous".to_owned());
        }
        if step.event != "note_on" {
            if step.event != "note_off"
                || step.expected_selection_decision != "not_applicable_note_off"
                || step.expected_region_index.is_some()
                || step.expected_sample_path.is_some()
            {
                failure = Some("non-note-on timeline entry is not an explicit non-selecting event".to_owned());
            }
            results.push(TimelineValidationV1 {
                ordinal: step.ordinal,
                event: step.event.clone(),
                stream_id: step.stream_id.clone(),
                expected_decision: step.expected_selection_decision.clone(),
                actual_decision: None,
                expected_region_index: step.expected_region_index,
                actual_region_index: None,
                expected_sample_path: step.expected_sample_path.clone(),
                actual_sample_path: None,
                matched: failure.is_none(),
                failure,
            });
            continue;
        }

        let actual: SfzZoneSelectionV1 = sequencer.next_selection(
            manifest,
            &step.stream_id,
            step.midi_key,
            step.velocity,
        );
        let actual_path = actual.selected_sample_path.clone();
        let matched = failure.is_none()
            && actual.decision == step.expected_selection_decision
            && actual.selected_region_index == step.expected_region_index
            && actual_path == step.expected_sample_path;
        if !matched && failure.is_none() {
            failure = Some(format!(
                "expected decision/region/path {:?}/{:?}/{:?}, got {:?}/{:?}/{:?}",
                step.expected_selection_decision,
                step.expected_region_index,
                step.expected_sample_path,
                actual.decision,
                actual.selected_region_index,
                actual_path,
            ));
        }
        results.push(TimelineValidationV1 {
            ordinal: step.ordinal,
            event: step.event.clone(),
            stream_id: step.stream_id.clone(),
            expected_decision: step.expected_selection_decision.clone(),
            actual_decision: Some(actual.decision),
            expected_region_index: step.expected_region_index,
            actual_region_index: actual.selected_region_index,
            expected_sample_path: step.expected_sample_path.clone(),
            actual_sample_path: actual_path,
            matched: failure.is_none() && matched,
            failure,
        });
    }
    results
}

/// Verify the actual WAV container header instead of trusting manifest metadata.
fn wav_header_matches(bytes: &[u8], asset: &AssetRecordV1) -> bool {
    fn read_u16(bytes: &[u8], start: usize) -> Option<u16> {
        Some(u16::from_le_bytes(bytes.get(start..start + 2)?.try_into().ok()?))
    }
    fn read_u32(bytes: &[u8], start: usize) -> Option<u32> {
        Some(u32::from_le_bytes(bytes.get(start..start + 4)?.try_into().ok()?))
    }

    if bytes.len() < 44
        || bytes.get(0..4) != Some(&b"RIFF"[..])
        || bytes.get(8..12) != Some(&b"WAVE"[..])
        || bytes.get(12..16) != Some(&b"fmt "[..])
        || bytes.get(36..40) != Some(&b"data"[..])
    {
        return false;
    }
    let Some(riff_len) = read_u32(bytes, 4) else { return false; };
    let Some(format_len) = read_u32(bytes, 16) else { return false; };
    let Some(audio_format) = read_u16(bytes, 20) else { return false; };
    let Some(channels) = read_u16(bytes, 22) else { return false; };
    let Some(sample_rate) = read_u32(bytes, 24) else { return false; };
    let Some(byte_rate) = read_u32(bytes, 28) else { return false; };
    let Some(block_align) = read_u16(bytes, 32) else { return false; };
    let Some(bits_per_sample) = read_u16(bytes, 34) else { return false; };
    let Some(data_len) = read_u32(bytes, 40) else { return false; };

    let expected_data_len = asset.frames.checked_mul(usize::from(asset.channels)).and_then(|value| value.checked_mul(2));
    let expected_byte_rate = asset.sample_rate_hz.checked_mul(u32::from(asset.channels)).and_then(|value| value.checked_mul(2));
    let expected_block_align = asset.channels.checked_mul(2);

    usize::try_from(riff_len).ok() == Some(bytes.len().saturating_sub(8))
        && format_len == 16
        && audio_format == 1
        && channels == asset.channels
        && sample_rate == asset.sample_rate_hz
        && expected_byte_rate == Some(byte_rate)
        && expected_block_align == Some(block_align)
        && bits_per_sample == 16
        && usize::try_from(data_len).ok() == expected_data_len
        && bytes.len() == 44_usize.saturating_add(data_len as usize)
        && usize::try_from(data_len).ok().is_some_and(|len| len % 2 == 0)
        && usize::try_from(data_len).ok().and_then(|len| len.checked_div(usize::from(asset.channels) * 2)) == Some(asset.frames)
}

fn pack_tree_sha256(root: &Path) -> Result<String, Box<dyn Error>> {
    fn collect_files(directory: &Path, root: &Path, files: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("fixture pack contains symlink {}", path.display()),
                )
                .into());
            }
            if metadata.is_dir() {
                collect_files(&path, root, files)?;
            } else if metadata.is_file() {
                let relative = path.strip_prefix(root)?.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/");
                if relative != "pack.sha256" {
                    files.push(path);
                }
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("fixture pack contains non-regular file {}", path.display()),
                )
                .into());
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    collect_files(root, root, &mut files)?;
    files.sort();
    let mut hasher = Sha256::new();
    for path in files {
        let relative = path.strip_prefix(root)?.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/");
        hasher.update(relative.as_bytes());
        hasher.update([0_u8]);
        hasher.update(Sha256::digest(&fs::read(&path)?));
        hasher.update([b'\n']);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_sha256_vector_is_correct() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn pack_paths_reject_absolute_and_parent_components() {
        let root = Path::new("/tmp/fixture-pack");
        assert!(safe_join(root, "../escape.wav").is_err());
        assert!(safe_join(root, "/absolute.wav").is_err());
        assert!(safe_join(root, "fixture/sample.wav").is_ok());
    }

    #[test]
    fn wav_header_validator_rejects_untrusted_manifest_metadata() {
        let asset = AssetRecordV1 {
            path: "sample.wav".to_owned(),
            sha256: String::new(),
            byte_len: 46,
            sample_rate_hz: 44_100,
            channels: 1,
            frames: 1,
            encoding: "RIFF_WAVE_PCM_S16LE".to_owned(),
        };
        let mut bytes = vec![0_u8; 46];
        bytes[0..4].copy_from_slice(b"RIFF");
        bytes[4..8].copy_from_slice(&38_u32.to_le_bytes());
        bytes[8..12].copy_from_slice(b"WAVE");
        bytes[12..16].copy_from_slice(b"fmt ");
        bytes[16..20].copy_from_slice(&16_u32.to_le_bytes());
        bytes[20..22].copy_from_slice(&1_u16.to_le_bytes());
        bytes[22..24].copy_from_slice(&1_u16.to_le_bytes());
        bytes[24..28].copy_from_slice(&44_100_u32.to_le_bytes());
        bytes[28..32].copy_from_slice(&88_200_u32.to_le_bytes());
        bytes[32..34].copy_from_slice(&2_u16.to_le_bytes());
        bytes[34..36].copy_from_slice(&16_u16.to_le_bytes());
        bytes[36..40].copy_from_slice(b"data");
        bytes[40..44].copy_from_slice(&2_u32.to_le_bytes());
        bytes[44..46].copy_from_slice(&123_i16.to_le_bytes());
        assert!(wav_header_matches(&bytes, &asset));

        let mut wrong_rate = bytes.clone();
        wrong_rate[24..28].copy_from_slice(&48_000_u32.to_le_bytes());
        assert!(!wav_header_matches(&wrong_rate, &asset));

        let mut wrong_frames = bytes;
        wrong_frames[40..44].copy_from_slice(&4_u32.to_le_bytes());
        assert!(!wav_header_matches(&wrong_frames, &asset));
    }

    #[test]
    fn alias_of_known_fixture_schema_is_stable() {
        assert_eq!(EXPECTED_PACK_SCHEMA, "muse-sfz-compat-fixture-pack-v1");
    }
}
