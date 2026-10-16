// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generate a deterministic, hash-bound synthetic SFZ compatibility fixture pack.
//!
//! Example:
//!   cargo run --manifest-path crates/domains/symthaea-muse/Cargo.toml --bin sfz_compat_fixture_pack -- --output-dir /tmp/muse-sfz-compat-v1
//!
//! This pack separates SFZ behavior under test from Muse's current-subset
//! expectations. It uses no third-party sample assets and does not claim
//! reference-player compatibility without an independently captured result.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    env,
    error::Error,
    fs,
    io,
    path::{Path, PathBuf},
    process::ExitCode,
};

const SAMPLE_RATE_HZ: u32 = 44_100;
const ASSET_FRAMES: usize = 4_410;
const PCM_PATTERN: [i16; 16] = [
    0, 12_539, 23_170, 30_274, 32_000, 30_274, 23_170, 12_539,
    0, -12_539, -23_170, -30_274, -32_000, -30_274, -23_170, -12_539,
];

#[derive(Debug, Serialize)]
struct AssetReceiptV1 {
    path: String,
    sha256: String,
    byte_len: u64,
    sample_rate_hz: u32,
    channels: u16,
    frames: usize,
    encoding: &'static str,
}

#[derive(Debug, Serialize)]
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
struct FixtureRecordV1 {
    fixture_id: String,
    category: String,
    sfz_file: String,
    sfz_sha256: String,
    sfz_byte_len: usize,
    parser_expectation: String,
    behavior_under_test: String,
    current_muse_subset_expectation: String,
    sample_assets: Vec<AssetReceiptV1>,
    timeline: Vec<TimelineExpectationV1>,
    unsupported_semantics: Vec<String>,
    negative_control: bool,
}

#[derive(Debug, Serialize)]
struct FixturePackV1 {
    schema: &'static str,
    generator_version: &'static str,
    generated_from: &'static str,
    output_sample_rate_hz: u32,
    asset_frames: usize,
    fixtures: Vec<FixtureRecordV1>,
    evidence_boundary: &'static str,
}

struct AssetSpec {
    path: &'static str,
    phase_offset: usize,
    amplitude_percent: i32,
}

struct FixtureDefinition {
    fixture_id: &'static str,
    category: &'static str,
    sfz_source: &'static str,
    assets: Vec<AssetSpec>,
    parser_expectation: &'static str,
    behavior_under_test: &'static str,
    current_muse_subset_expectation: &'static str,
    timeline: Vec<TimelineExpectationV1>,
    unsupported_semantics: Vec<&'static str>,
    negative_control: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sfz_compat_fixture_pack: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let output_dir = parse_output_dir()?;
    if output_dir.exists() {
        if fs::read_dir(&output_dir)?.next().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "output directory {} is not empty; refusing to overwrite fixtures",
                    output_dir.display()
                ),
            )
            .into());
        }
    } else {
        fs::create_dir_all(&output_dir)?;
    }

    let definitions = fixture_definitions();
    let mut fixtures = Vec::with_capacity(definitions.len());
    for definition in definitions {
        let fixture_dir = output_dir.join(definition.fixture_id);
        fs::create_dir_all(&fixture_dir)?;
        let sfz_file_name = "instrument.sfz";
        let sfz_bytes = definition.sfz_source.as_bytes();
        fs::write(fixture_dir.join(sfz_file_name), sfz_bytes)?;

        let mut assets = Vec::with_capacity(definition.assets.len());
        for asset in &definition.assets {
            assets.push(write_asset(&fixture_dir, asset)?);
        }

        fixtures.push(FixtureRecordV1 {
            fixture_id: definition.fixture_id.to_owned(),
            category: definition.category.to_owned(),
            sfz_file: format!("{}/{}", definition.fixture_id, sfz_file_name),
            sfz_sha256: sha256_hex(sfz_bytes),
            sfz_byte_len: sfz_bytes.len(),
            parser_expectation: definition.parser_expectation.to_owned(),
            behavior_under_test: definition.behavior_under_test.to_owned(),
            current_muse_subset_expectation: definition.current_muse_subset_expectation.to_owned(),
            sample_assets: assets,
            timeline: definition.timeline,
            unsupported_semantics: definition.unsupported_semantics.into_iter().map(str::to_owned).collect(),
            negative_control: definition.negative_control,
        });
    }

    let pack = FixturePackV1 {
        schema: "muse-sfz-compat-fixture-pack-v1",
        generator_version: "sfz-compat-fixture-pack-v1",
        generated_from: "deterministic synthetic SFZ and integer PCM fixtures; no third-party audio",
        output_sample_rate_hz: SAMPLE_RATE_HZ,
        asset_frames: ASSET_FRAMES,
        fixtures,
        evidence_boundary: "fixture intent and current-subset expectations are authored expectations; this pack contains no observed reference-engine results and does not prove compatibility",
    };
    fs::write(
        output_dir.join("fixtures.json"),
        serde_json::to_vec_pretty(&pack)?,
    )?;
    fs::write(output_dir.join("README.md"), pack_readme())?;
    // Hash a sorted list of every generated path plus each file content hash.
    // The digest excludes pack.sha256 itself, avoiding a self-referential hash.
    let pack_digest = pack_tree_sha256(&output_dir)?;
    fs::write(output_dir.join("pack.sha256"), format!("{pack_digest}  fixture-pack-v1\n"))?;

    println!(
        "generated {} fixtures under {}; manifest: {}",
        pack.fixtures.len(),
        output_dir.display(),
        output_dir.join("fixtures.json").display()
    );
    println!("pack digest: {pack_digest}");
    println!("No reference-player result is inferred or marked as passing.");
    Ok(())
}

fn pack_tree_sha256(root: &Path) -> Result<String, Box<dyn Error>> {
    fn collect_files(directory: &Path, result: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path.is_dir() {
                collect_files(&path, result)?;
            } else {
                result.push(path);
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    collect_files(root, &mut files)?;
    files.sort();
    let mut hasher = Sha256::new();
    for path in files {
        let relative = path.strip_prefix(root)?.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/");
        let bytes = fs::read(&path)?;
        hasher.update(relative.as_bytes());
        hasher.update([0_u8]);
        hasher.update(Sha256::digest(&bytes));
        hasher.update([b'\n']);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn parse_output_dir() -> Result<PathBuf, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let flag = args.next();
    let path = args.next();
    if flag.as_deref() == Some("--output-dir") {
        if let Some(path) = path {
            if args.next().is_none() {
                return Ok(PathBuf::from(path));
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "usage: sfz_compat_fixture_pack --output-dir EMPTY_DIRECTORY",
    )
    .into())
}

fn write_asset(directory: &Path, spec: &AssetSpec) -> Result<AssetReceiptV1, Box<dyn Error>> {
    let path = directory.join(spec.path);
    let mut writer = hound::WavWriter::create(
        &path,
        hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE_HZ,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for frame in 0..ASSET_FRAMES {
        let pattern = PCM_PATTERN[(frame + spec.phase_offset) % PCM_PATTERN.len()];
        let scaled = i32::from(pattern) * spec.amplitude_percent / 100;
        writer.write_sample(scaled as i16)?;
    }
    writer.finalize()?;
    let bytes = fs::read(&path)?;
    Ok(AssetReceiptV1 {
        path: spec.path.to_owned(),
        sha256: sha256_hex(&bytes),
        byte_len: bytes.len() as u64,
        sample_rate_hz: SAMPLE_RATE_HZ,
        channels: 1,
        frames: ASSET_FRAMES,
        encoding: "RIFF_WAVE_PCM_S16LE",
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn asset(path: &'static str, phase_offset: usize, amplitude_percent: i32) -> AssetSpec {
    AssetSpec { path, phase_offset, amplitude_percent }
}

fn event(
    ordinal: usize,
    event: &str,
    stream_id: &str,
    midi_key: u8,
    velocity: u8,
    decision: &str,
    region_index: Option<usize>,
    sample_path: Option<&str>,
    render_outcome: &str,
) -> TimelineExpectationV1 {
    TimelineExpectationV1 {
        ordinal,
        event: event.to_owned(),
        stream_id: stream_id.to_owned(),
        midi_key,
        velocity,
        expected_selection_decision: decision.to_owned(),
        expected_region_index: region_index,
        expected_sample_path: sample_path.map(str::to_owned),
        expected_render_outcome: render_outcome.to_owned(),
    }
}

fn fixture_definitions() -> Vec<FixtureDefinition> {
    vec![
        FixtureDefinition {
            fixture_id: "key_boundary_mapping",
            category: "mapping_boundaries",
            sfz_source: "<region> lokey=48 hikey=72 pitch_keycenter=60 sample=boundary.wav\n",
            assets: vec![asset("boundary.wav", 0, 100)],
            parser_expectation: "parse_success",
            behavior_under_test: "inclusive lower and upper MIDI key bounds plus uncovered adjacent key",
            current_muse_subset_expectation: "keys 48 and 72 select region 1; key 73 is uncovered",
            timeline: vec![
                event(1, "note_on", "boundary", 48, 90, "unique_candidate", Some(1), Some("boundary.wav"), "one_shot_expected"),
                event(2, "note_on", "boundary", 72, 90, "unique_candidate", Some(1), Some("boundary.wav"), "one_shot_expected"),
                event(3, "note_on", "boundary", 73, 90, "uncovered", None, None, "no_audio_expected"),
            ],
            unsupported_semantics: vec![],
            negative_control: false,
        },
        FixtureDefinition {
            fixture_id: "complete_round_robin_streams",
            category: "round_robin",
            sfz_source: "<group> key=60 seq_length=2\n<region> seq_position=1 sample=rr_a.wav\n<region> seq_position=2 sample=rr_b.wav\n",
            assets: vec![asset("rr_a.wav", 0, 100), asset("rr_b.wav", 3, 85)],
            parser_expectation: "parse_success",
            behavior_under_test: "two-position sequence advances only on note-on selections and isolates stream state",
            current_muse_subset_expectation: "voice-a selects 1, then 2, then wraps to 1; voice-b independently begins at 1",
            timeline: vec![
                event(1, "note_on", "voice-a", 60, 100, "round_robin_candidate", Some(1), Some("rr_a.wav"), "one_shot_expected"),
                event(2, "note_off", "voice-a", 60, 0, "not_applicable_note_off", None, None, "release_semantics_not_implemented"),
                event(3, "note_on", "voice-a", 60, 100, "round_robin_candidate", Some(2), Some("rr_b.wav"), "one_shot_expected"),
                event(4, "note_on", "voice-b", 60, 100, "round_robin_candidate", Some(1), Some("rr_a.wav"), "one_shot_expected"),
                event(5, "note_on", "voice-a", 60, 100, "round_robin_candidate", Some(1), Some("rr_a.wav"), "one_shot_expected"),
            ],
            unsupported_semantics: vec!["voice allocation and sample tails are outside the mapping-only expectation"],
            negative_control: false,
        },
        FixtureDefinition {
            fixture_id: "incomplete_round_robin",
            category: "round_robin_negative_control",
            sfz_source: "<group> key=60 seq_length=2\n<region> seq_position=1 sample=rr_only.wav\n",
            assets: vec![asset("rr_only.wav", 2, 100)],
            parser_expectation: "parse_success",
            behavior_under_test: "refuse incomplete sequence rather than treating it as a complete bank",
            current_muse_subset_expectation: "selection decision incomplete_round_robin_set; no selected region",
            timeline: vec![
                event(1, "note_on", "incomplete", 60, 100, "incomplete_round_robin_set", None, None, "render_must_refuse_selection"),
            ],
            unsupported_semantics: vec![],
            negative_control: true,
        },
        FixtureDefinition {
            fixture_id: "velocity_split_sequence_mismatch",
            category: "velocity_round_robin",
            sfz_source: "<group> key=60 seq_length=2\n<region> seq_position=1 lovel=0 hivel=63 sample=soft_a.wav\n<region> seq_position=2 lovel=0 hivel=63 sample=soft_b.wav\n<region> seq_position=1 lovel=64 hivel=127 sample=hard_a.wav\n",
            assets: vec![
                asset("soft_a.wav", 0, 100),
                asset("soft_b.wav", 4, 92),
                asset("hard_a.wav", 7, 80),
            ],
            parser_expectation: "parse_success",
            behavior_under_test: "low-velocity layer has two complete positions while high-velocity layer has only one",
            current_muse_subset_expectation: "velocity 40 selects complete round robin; velocity 100 is rejected as incomplete",
            timeline: vec![
                event(1, "note_on", "soft-layer", 60, 40, "round_robin_candidate", Some(1), Some("soft_a.wav"), "one_shot_expected"),
                event(2, "note_on", "soft-layer", 60, 40, "round_robin_candidate", Some(2), Some("soft_b.wav"), "one_shot_expected"),
                event(3, "note_on", "hard-layer", 60, 100, "incomplete_round_robin_set", None, None, "render_must_refuse_selection"),
            ],
            unsupported_semantics: vec!["player-specific sequence-group behavior requires an independent reference capture"],
            negative_control: true,
        },
        FixtureDefinition {
            fixture_id: "release_trigger_semantics",
            category: "note_off_release",
            sfz_source: "<group> key=60\n<region> sample=sustain.wav\n<region> trigger=release sample=release.wav\n",
            assets: vec![asset("sustain.wav", 0, 100), asset("release.wav", 5, 90)],
            parser_expectation: "parse_success_with_unhandled_trigger_opcode",
            behavior_under_test: "note-on sustain region versus note-off release-trigger region",
            current_muse_subset_expectation: "overlapping regions remain ambiguous in the current mapping subset; release-trigger behavior is not claimed",
            timeline: vec![
                event(1, "note_on", "release-test", 60, 90, "ambiguous_zones", None, None, "must_not_claim_full_sfz_playback"),
                event(2, "note_off", "release-test", 60, 0, "not_applicable_note_off", None, None, "release_trigger_not_implemented"),
            ],
            unsupported_semantics: vec!["trigger=release", "voice release lifecycle", "release sample mixing"],
            negative_control: false,
        },
        FixtureDefinition {
            fixture_id: "percussion_retrigger",
            category: "percussion",
            sfz_source: "<region> key=36 pitch_keycenter=36 sample=kick.wav\n",
            assets: vec![asset("kick.wav", 1, 100)],
            parser_expectation: "parse_success",
            behavior_under_test: "repeated note-on attacks for a percussion one-shot",
            current_muse_subset_expectation: "valid attacks map to region 1; voice stealing, tails and choke groups are not inferred",
            timeline: vec![
                event(1, "note_on", "drum-track", 128, 100, "invalid_request", None, None, "must_reject_out_of_range_midi_key"),
                event(2, "note_on", "drum-track", 36, 100, "unique_candidate", Some(1), Some("kick.wav"), "one_shot_expected"),
                event(3, "note_off", "drum-track", 36, 0, "not_applicable_note_off", None, None, "one_shot_tail_semantics_not_implemented"),
                event(4, "note_on", "drum-track", 36, 100, "unique_candidate", Some(1), Some("kick.wav"), "one_shot_expected"),
            ],
            unsupported_semantics: vec!["voice allocation", "choke groups", "sample tails and envelopes"],
            negative_control: true,
        },
        FixtureDefinition {
            fixture_id: "unsupported_filter_and_loop",
            category: "unsupported_opcodes",
            sfz_source: "<region> key=60 sample=filtered_loop.wav cutoff=8000 loop_mode=loop_continuous ampeg_release=0.2\n",
            assets: vec![asset("filtered_loop.wav", 3, 95)],
            parser_expectation: "parse_success_with_unhandled_opcodes",
            behavior_under_test: "a mapped region is not evidence that filter, looping or envelope opcodes were executed",
            current_muse_subset_expectation: "unique candidate may be selected; unhandled opcodes remain explicit and are not applied",
            timeline: vec![
                event(1, "note_on", "unsupported-opcodes", 60, 90, "unique_candidate", Some(1), Some("filtered_loop.wav"), "one_shot_only_no_filter_loop_or_envelope"),
            ],
            unsupported_semantics: vec!["cutoff", "loop_mode", "ampeg_release"],
            negative_control: false,
        },
        FixtureDefinition {
            fixture_id: "missing_sample_asset",
            category: "asset_integrity_negative_control",
            sfz_source: "<region> key=60 sample=intentionally_missing.wav\n",
            assets: vec![],
            parser_expectation: "parse_success",
            behavior_under_test: "selection metadata must not be confused with successful sample loading or rendering",
            current_muse_subset_expectation: "selection may resolve but playback must fail with a missing/unsafe asset result",
            timeline: vec![
                event(1, "note_on", "missing-sample", 60, 90, "unique_candidate", Some(1), Some("intentionally_missing.wav"), "asset_missing_render_failure_expected"),
            ],
            unsupported_semantics: vec![],
            negative_control: true,
        },
        FixtureDefinition {
            fixture_id: "path_traversal_rejection",
            category: "path_safety_negative_control",
            sfz_source: "<region> key=60 sample=../outside.wav\n",
            assets: vec![],
            parser_expectation: "parse_rejected_path_escape",
            behavior_under_test: "parent traversal must not escape the SFZ/sample root",
            current_muse_subset_expectation: "parse fails closed before selection or rendering",
            timeline: vec![],
            unsupported_semantics: vec![],
            negative_control: true,
        },
    ]
}

fn pack_readme() -> &'static str {
    "# Muse synthetic SFZ compatibility fixtures v1

This directory is generated by sfz_compat_fixture_pack. Run:

    cargo run --manifest-path crates/domains/symthaea-muse/Cargo.toml --bin sfz_compat_fixture_pack -- --output-dir /tmp/muse-sfz-compat-v1

fixtures.json records stable fixture IDs, SFZ source hashes, exact WAV hashes and byte lengths, event timelines, authored expectations, and current-subset limitations. pack.sha256 records a deterministic tree digest over the generated relative paths and file content hashes. All WAV assets use a fixed integer PCM pattern; no external sample bank or third-party recording is bundled.

## Evidence rules

- Parser and current-subset expectations are authored expectations, not observed reference-engine results.
- A future reference-run record must add exact player name/version/build/configuration, fixture pack hash, capture method, raw output hash, comparison profile, and outcome.
- Missing reference captures are unqualified, never pass.
- Negative controls (incomplete round robin, velocity-layer mismatch, out-of-range MIDI key, missing sample, and path traversal) must remain in the corpus.
- Release triggers, filters, loops, envelopes, voice allocation and choke groups are not qualified by these mapping/one-shot fixtures.

This pack enables deterministic differential testing; it is not proof of full SFZ compatibility.
"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_fixture_pattern_is_integer_only_and_non_silent() {
        assert!(PCM_PATTERN.iter().any(|sample| *sample > 0));
        assert!(PCM_PATTERN.iter().any(|sample| *sample < 0));
        assert!(PCM_PATTERN.iter().all(|sample| i32::from(*sample).abs() <= 32_000));
    }

    #[test]
    fn fixture_ids_are_unique_and_negative_controls_are_present() {
        let fixtures = fixture_definitions();
        let mut ids = std::collections::BTreeSet::new();
        for fixture in &fixtures {
            assert!(ids.insert(fixture.fixture_id), "duplicate fixture ID");
            assert!(!fixture.sfz_source.is_empty());
        }
        assert!(fixtures.iter().any(|fixture| fixture.negative_control));
        assert!(fixtures.len() >= 8);
    }

    #[test]
    fn velocity_split_fixture_preserves_the_incomplete_high_layer() {
        let fixture = fixture_definitions()
            .into_iter()
            .find(|fixture| fixture.fixture_id == "velocity_split_sequence_mismatch")
            .unwrap();
        assert!(fixture.sfz_source.contains("lovel=0 hivel=63"));
        assert!(fixture.sfz_source.contains("lovel=64 hivel=127"));
        assert!(fixture.unsupported_semantics.iter().any(|value| value.contains("reference capture")));
    }
}
