// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bounded parser for the numeric-key SFZ mapping subset used by Muse.
//!
//! This is an inventory/inspection parser, not a complete SFZ renderer. It
//! applies global/group/region inheritance for the mapping opcodes below,
//! reports unhandled opcodes, and never infers pitch from a sample filename.
//! Audio decoding and playback semantics remain separate qualification steps.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::path::{Component, Path};

const DEFAULT_LOW_KEY: u8 = 0;
const DEFAULT_HIGH_KEY: u8 = 127;
const DEFAULT_PITCH_KEYCENTER: u8 = 60;
const DEFAULT_LOW_VELOCITY: u8 = 0;
const DEFAULT_HIGH_VELOCITY: u8 = 127;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SfzParseError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for SfzParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "SFZ line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for SfzParseError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SfzRegionMappingV1 {
    /// One-based ordinal of the region in source order.
    pub region_index: usize,
    pub source_line: usize,
    /// Effective path relative to the SFZ/sample root; never absolute.
    pub sample_path: String,
    pub key_low: u8,
    pub key_high: u8,
    pub pitch_keycenter: u8,
    /// The SFZ tune opcode, in cents. This is preserved, not applied to audio.
    pub tune_cents: f32,
    pub velocity_low: u8,
    pub velocity_high: u8,
    pub sequence_length: Option<u16>,
    pub sequence_position: Option<u16>,
    /// Set only by the file inspector, not by the pure text parser.
    pub sample_asset_state: String,
    pub sample_byte_len: Option<u64>,
    pub sample_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SfzMappingManifestV1 {
    pub schema: String,
    /// SHA-256 of the exact SFZ text bytes supplied to the parser.
    pub sfz_sha256: String,
    pub declared_source_label: Option<String>,
    /// Caller-provided annotation; not independently authenticated.
    pub declared_source_revision: Option<String>,
    pub parser_scope: String,
    pub region_count: usize,
    pub mapped_midi_keys: Vec<u8>,
    pub unhandled_opcodes: Vec<String>,
    /// This parser inspects key/velocity/sample mapping; it is not a full SFZ renderer.
    pub audio_decode_evidence: String,
    pub regions: Vec<SfzRegionMappingV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    Global,
    Control,
    Group,
    Region,
}

type Opcodes = std::collections::BTreeMap<String, String>;

/// Parse SFZ numeric MIDI key/velocity mappings and preserve enough metadata
/// to qualify externally authored sample banks.
///
/// Supported sections are control, global, group and region. Mapping opcodes include sample, default_path, key, lokey, hikey,
/// pitch_keycenter, lovel, hivel, tune, seq_length, and seq_position. Other
/// opcodes are listed in unhandled_opcodes; this function does not claim full
/// playback fidelity or execute opcodes.
///
/// Section inheritance follows the document order: each group starts with the
/// global opcodes; each region starts with the current group's opcodes. Values
/// inside a narrower section override inherited values.
pub fn parse_sfz_mapping(source: &str) -> Result<SfzMappingManifestV1, SfzParseError> {
    let mut scope = Scope::Global;
    let mut global: Opcodes = Opcodes::new();
    let mut control: Opcodes = Opcodes::new();
    let mut group: Opcodes = Opcodes::new();
    let mut region: Option<(Opcodes, usize)> = None;
    let mut regions = Vec::new();
    let mut unhandled_opcodes = BTreeSet::new();

    for (zero_based_line, raw_line) in source.lines().enumerate() {
        let line_number = zero_based_line + 1;
        let line = raw_line.split("//").next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        for token in line.split_whitespace() {
            if token.starts_with('<') && token.ends_with('>') {
                let section = token[1..token.len() - 1].to_ascii_lowercase();
                match section.as_str() {
                    "control" => {
                        finalize_region(&mut region, &mut regions, &mut unhandled_opcodes, &control)?;
                        scope = Scope::Control;
                    }
                    "global" => {
                        finalize_region(&mut region, &mut regions, &mut unhandled_opcodes, &control)?;
                        global.clear();
                        group = global.clone();
                        scope = Scope::Global;
                    }
                    "group" => {
                        finalize_region(&mut region, &mut regions, &mut unhandled_opcodes, &control)?;
                        group = global.clone();
                        scope = Scope::Group;
                    }
                    "region" => {
                        finalize_region(&mut region, &mut regions, &mut unhandled_opcodes, &control)?;
                        region = Some((group.clone(), line_number));
                        scope = Scope::Region;
                    }
                    other => {
                        return Err(parse_error(
                            line_number,
                            format!("unsupported section <{other}>; this parser accepts only control/global/group/region"),
                        ));
                    }
                }
                continue;
            }

            let Some((key, value)) = token.split_once('=') else {
                return Err(parse_error(
                    line_number,
                    format!("expected an SFZ opcode assignment, got {token:?}"),
                ));
            };
            if key.is_empty() || value.is_empty() {
                return Err(parse_error(
                    line_number,
                    format!("empty opcode name or value in {token:?}"),
                ));
            }

            if !is_handled_opcode(key) {
                unhandled_opcodes.insert(key.to_ascii_lowercase());
            }
            match scope {
                Scope::Global => {
                    global.insert(key.to_ascii_lowercase(), value.to_owned());
                }
                Scope::Control => {
                    control.insert(key.to_ascii_lowercase(), value.to_owned());
                }
                Scope::Group => {
                    group.insert(key.to_ascii_lowercase(), value.to_owned());
                }
                Scope::Region => {
                    let Some((opcodes, _)) = region.as_mut() else {
                        return Err(parse_error(
                            line_number,
                            "region opcodes encountered before a region section",
                        ));
                    };
                    opcodes.insert(key.to_ascii_lowercase(), value.to_owned());
                }
            }
        }
    }

    finalize_region(&mut region, &mut regions, &mut unhandled_opcodes, &control)?;
    if regions.is_empty() {
        return Err(parse_error(1, "no region entries with sample mappings found"));
    }

    let mut mapped_keys = BTreeSet::new();
    for entry in &regions {
        for key in entry.key_low..=entry.key_high {
            mapped_keys.insert(key);
        }
    }

    Ok(SfzMappingManifestV1 {
        schema: "muse-sfz-mapping-manifest-v1".to_owned(),
        sfz_sha256: format!("{:x}", Sha256::digest(source.as_bytes())),
        declared_source_label: None,
        declared_source_revision: None,
        parser_scope: "numeric_midi_key_velocity_sample_tuning_sequence_subset_not_full_sfz_playback".to_owned(),
        region_count: regions.len(),
        mapped_midi_keys: mapped_keys.into_iter().collect(),
        unhandled_opcodes: unhandled_opcodes.into_iter().collect(),
        audio_decode_evidence: "not_performed".to_owned(),
        regions,
    })
}

/// A deterministic zone-selection plan for one MIDI key and velocity.
///
/// This is a selection receipt, not a rendered-audio receipt. If several
/// regions match, the planner selects only a complete round-robin set whose
/// zones have identical key/velocity bounds. Other overlapping regions remain
/// ambiguous rather than being silently layered or choosing one arbitrarily.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SfzZoneSelectionV1 {
    pub requested_key: u8,
    pub requested_velocity: u8,
    pub round_robin_index: u64,
    pub matching_region_indices: Vec<usize>,
    pub selected_region_index: Option<usize>,
    /// One of: uncovered, invalid_request, unique_candidate,
    /// round_robin_candidate, incomplete_round_robin_set, ambiguous_zones.
    /// None of these names imply audio decoding or rendering.
    pub decision: String,
    pub semitone_shift: Option<f64>,
    pub frequency_ratio: Option<f64>,
}

/// Plan a note lookup against parsed SFZ zones.
///
/// round_robin_index is zero-based; for a valid sequence with N positions,
/// index 0 picks position 1 and index N picks position 1 again. We preserve
/// region/source order in receipts and require a complete sequence before
/// selecting one of multiple matching zones. This is intentionally not a full
/// SFZ voice engine: velocity crossfades, switches, sequence groups across
/// alternate group layouts, sample modes, envelopes, filters and audio
/// playback are outside the supported subset.
pub fn plan_sfz_zone_selection(
    manifest: &SfzMappingManifestV1,
    requested_key: u8,
    requested_velocity: u8,
    round_robin_index: u64,
) -> SfzZoneSelectionV1 {
    let mut receipt = SfzZoneSelectionV1 {
        requested_key,
        requested_velocity,
        round_robin_index,
        matching_region_indices: Vec::new(),
        selected_region_index: None,
        decision: "uncovered".to_owned(),
        semitone_shift: None,
        frequency_ratio: None,
    };
    if requested_key > 127 || requested_velocity > 127 {
        receipt.decision = "invalid_request".to_owned();
        return receipt;
    }

    let candidates: Vec<&SfzRegionMappingV1> = manifest
        .regions
        .iter()
        .filter(|region| {
            (region.key_low..=region.key_high).contains(&requested_key)
                && (region.velocity_low..=region.velocity_high).contains(&requested_velocity)
        })
        .collect();
    receipt.matching_region_indices = candidates.iter().map(|region| region.region_index).collect();

    let selected = match candidates.as_slice() {
        [] => return receipt,
        [only] => match (only.sequence_length, only.sequence_position) {
            (None, None) => {
                receipt.decision = "unique_candidate".to_owned();
                Some(*only)
            }
            (Some(1), Some(1)) => {
                receipt.decision = "round_robin_candidate".to_owned();
                Some(*only)
            }
            (Some(_), _) | (_, Some(_)) => {
                receipt.decision = "incomplete_round_robin_set".to_owned();
                None
            }
        },
        many => {
            let first = many[0];
            let shared_zone = many.iter().all(|region| {
                region.key_low == first.key_low
                    && region.key_high == first.key_high
                    && region.velocity_low == first.velocity_low
                    && region.velocity_high == first.velocity_high
            });
            let Some(sequence_length) = first.sequence_length else {
                receipt.decision = "ambiguous_zones".to_owned();
                return receipt;
            };
            let mut positions = Vec::with_capacity(many.len());
            let complete_metadata = many.iter().all(|region| {
                if region.sequence_length != Some(sequence_length) {
                    return false;
                }
                match region.sequence_position {
                    Some(position) => {
                        positions.push(position);
                        true
                    }
                    None => false,
                }
            });
            positions.sort_unstable();
            let expected: Vec<u16> = (1..=sequence_length).collect();
            if !shared_zone
                || sequence_length == 0
                || many.len() != sequence_length as usize
                || !complete_metadata
                || positions != expected
            {
                receipt.decision = "ambiguous_zones".to_owned();
                return receipt;
            }

            let desired_position = (round_robin_index % u64::from(sequence_length)) as u16 + 1;
            let Some(choice) = many
                .iter()
                .copied()
                .find(|region| region.sequence_position == Some(desired_position))
            else {
                receipt.decision = "ambiguous_zones".to_owned();
                return receipt;
            };
            receipt.decision = "round_robin_candidate".to_owned();
            Some(choice)
        }
    };

    if let Some(region) = selected {
        let semitone_shift = f64::from(requested_key) - f64::from(region.pitch_keycenter)
            + f64::from(region.tune_cents) / 100.0;
        receipt.selected_region_index = Some(region.region_index);
        receipt.semitone_shift = Some(semitone_shift);
        receipt.frequency_ratio = Some(2.0_f64.powf(semitone_shift / 12.0));
    }
    receipt
}

fn finalize_region(
    pending: &mut Option<(Opcodes, usize)>,
    output: &mut Vec<SfzRegionMappingV1>,
    unhandled: &mut BTreeSet<String>,
    control: &Opcodes,
) -> Result<(), SfzParseError> {
    let Some((opcodes, source_line)) = pending.take() else {
        return Ok(());
    };
    let sample_value = get_required(&opcodes, "sample", source_line)?;
    let default_path = opcodes
        .get("default_path")
        .or_else(|| control.get("default_path"))
        .map(String::as_str)
        .unwrap_or("");
    let sample_path = normalize_sample_path(default_path, sample_value, source_line)?;
    let single_key = opcodes.get("key").map(|value| parse_u8(value, "key", source_line)).transpose()?;

    let low_default = single_key.unwrap_or(DEFAULT_LOW_KEY);
    let high_default = single_key.unwrap_or(DEFAULT_HIGH_KEY);
    let center_default = single_key.unwrap_or(DEFAULT_PITCH_KEYCENTER);
    let key_low = opcodes
        .get("lokey")
        .map(|value| parse_u8(value, "lokey", source_line))
        .transpose()?
        .unwrap_or(low_default);
    let key_high = opcodes
        .get("hikey")
        .map(|value| parse_u8(value, "hikey", source_line))
        .transpose()?
        .unwrap_or(high_default);
    let pitch_keycenter = opcodes
        .get("pitch_keycenter")
        .map(|value| parse_u8(value, "pitch_keycenter", source_line))
        .transpose()?
        .unwrap_or(center_default);
    if key_low > key_high {
        return Err(parse_error(source_line, format!("lokey {key_low} exceeds hikey {key_high}")));
    }

    let velocity_low = opcodes
        .get("lovel")
        .map(|value| parse_u8(value, "lovel", source_line))
        .transpose()?
        .unwrap_or(DEFAULT_LOW_VELOCITY);
    let velocity_high = opcodes
        .get("hivel")
        .map(|value| parse_u8(value, "hivel", source_line))
        .transpose()?
        .unwrap_or(DEFAULT_HIGH_VELOCITY);
    if velocity_low > velocity_high {
        return Err(parse_error(
            source_line,
            format!("lovel {velocity_low} exceeds hivel {velocity_high}"),
        ));
    }

    let tune_cents = opcodes
        .get("tune")
        .map(|value| {
            value.parse::<f32>().ok().filter(|number| number.is_finite())
                .ok_or_else(|| parse_error(source_line, format!("invalid finite tune value {value:?}")))
        })
        .transpose()?
        .unwrap_or(0.0);
    let sequence_length = opcodes
        .get("seq_length")
        .map(|value| parse_positive_u16(value, "seq_length", source_line))
        .transpose()?;
    let sequence_position = opcodes
        .get("seq_position")
        .map(|value| parse_positive_u16(value, "seq_position", source_line))
        .transpose()?;
    if let (Some(length), Some(position)) = (sequence_length, sequence_position) {
        if position > length {
            return Err(parse_error(
                source_line,
                format!("seq_position {position} exceeds seq_length {length}"),
            ));
        }
    }

    for key in opcodes.keys() {
        if !is_handled_opcode(key) {
            unhandled.insert(key.clone());
        }
    }
    output.push(SfzRegionMappingV1 {
        region_index: output.len() + 1,
        source_line,
        sample_path,
        key_low,
        key_high,
        pitch_keycenter,
        tune_cents,
        velocity_low,
        velocity_high,
        sequence_length,
        sequence_position,
        sample_asset_state: "not_checked".to_owned(),
        sample_byte_len: None,
        sample_sha256: None,
    });
    Ok(())
}

fn is_handled_opcode(opcode: &str) -> bool {
    matches!(
        opcode.to_ascii_lowercase().as_str(),
        "sample" | "default_path" | "key" | "lokey" | "hikey" | "pitch_keycenter"
            | "lovel" | "hivel" | "tune" | "seq_length" | "seq_position"
    )
}

fn get_required<'a>(
    opcodes: &'a Opcodes,
    name: &str,
    line: usize,
) -> Result<&'a str, SfzParseError> {
    opcodes
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| parse_error(line, format!("region has no effective {name}= value")))
}

fn parse_u8(value: &str, name: &str, line: usize) -> Result<u8, SfzParseError> {
    value
        .parse::<u8>()
        .map_err(|_| parse_error(line, format!("{name} must be a numeric value from 0 to 127")))
        .and_then(|number| {
            if number <= 127 {
                Ok(number)
            } else {
                Err(parse_error(line, format!("{name} must be from 0 to 127")))
            }
        })
}

fn parse_positive_u16(value: &str, name: &str, line: usize) -> Result<u16, SfzParseError> {
    value
        .parse::<u16>()
        .ok()
        .filter(|number| *number > 0)
        .ok_or_else(|| parse_error(line, format!("{name} must be a positive integer")))
}

fn normalize_sample_path(
    default_path: &str,
    sample_value: &str,
    line: usize,
) -> Result<String, SfzParseError> {
    let mut parts = Vec::new();
    for raw in [default_path, sample_value] {
        if raw.is_empty() {
            continue;
        }
        let bytes = raw.as_bytes();
        let drive_prefixed =
            bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':';
        if raw.starts_with('/') || raw.contains('\\') || drive_prefixed {
            return Err(parse_error(line, "absolute or backslash-separated sample paths are not supported"));
        }
        for component in Path::new(raw).components() {
            match component {
                Component::Normal(segment) => {
                    let text = segment.to_str().ok_or_else(|| {
                        parse_error(line, "sample path contains a non-UTF-8 component")
                    })?;
                    if text.is_empty() {
                        return Err(parse_error(line, "sample path contains an empty component"));
                    }
                    parts.push(text.to_owned());
                }
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(parse_error(line, "sample path escapes the SFZ/sample root"));
                }
            }
        }
    }
    if parts.is_empty() {
        return Err(parse_error(line, "sample path resolves to an empty path"));
    }
    Ok(parts.join("/"))
}

/// Verify a relative sample reference without following symlinks.
///
/// Returns the regular-file byte length and, when requested, a streaming SHA-256.
/// Ok(None) means missing/non-file or a path containing a symlink. The caller
/// must keep that distinct from an audio decode result.
pub fn inspect_sample_reference(
    root: &Path,
    relative_sample_path: &str,
    hash_content: bool,
) -> std::io::Result<Option<(u64, Option<String>)>> {
    let normalized = normalize_sample_path("", relative_sample_path, 1)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
    let root_metadata = std::fs::symlink_metadata(root)?;
    if root_metadata.file_type().is_symlink() || !root_metadata.file_type().is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "sample root must be a real directory, not a symlink",
        ));
    }

    let mut current = root.to_path_buf();
    let parts: Vec<&str> = normalized.split('/').collect();
    for (index, part) in parts.iter().enumerate() {
        current.push(part);
        let metadata = match std::fs::symlink_metadata(&current) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        if metadata.file_type().is_symlink() {
            return Ok(None);
        }
        let last = index + 1 == parts.len();
        if (!last && !metadata.file_type().is_dir()) || (last && !metadata.file_type().is_file()) {
            return Ok(None);
        }
    }

    let metadata = std::fs::metadata(&current)?;
    let digest = if hash_content {
        let mut file = std::fs::File::open(&current)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        Some(format!("{:x}", hasher.finalize()))
    } else {
        None
    };
    Ok(Some((metadata.len(), digest)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SFZ_FIXTURE: &str = r#"
// Representative mapping excerpt from FreePats Timpani, version 2024-08-10.
// Full source bank and audio assets remain external.
<global>
 ampeg_release=4

<group>
 lokey=36 hikey=38
 pitch_keycenter=38
 seq_length=2
<region>
 seq_position=1
 sample=samples/1B_v5_rr1.flac
<region>
 seq_position=2
 sample=samples/1B_v5_rr2.flac

<group>
 key=43
 seq_length=2
<region>
 seq_position=1
 tune=60
 sample=samples/2A_v5_rr1.flac
<region>
 seq_position=2
 tune=50
 sample=samples/2A_v5_rr2.flac
"#;

    #[test]
    fn inherits_group_key_ranges_and_preserves_round_robin_and_tune() {
        let manifest = parse_sfz_mapping(SFZ_FIXTURE).unwrap();
        assert_eq!(manifest.region_count, 4);
        assert_eq!(manifest.mapped_midi_keys, vec![36, 37, 38, 43]);
        assert_eq!(manifest.audio_decode_evidence, "not_performed");
        assert_eq!(manifest.unhandled_opcodes, vec!["ampeg_release"]);

        assert_eq!((manifest.regions[0].key_low, manifest.regions[0].key_high), (36, 38));
        assert_eq!(manifest.regions[0].pitch_keycenter, 38);
        assert_eq!(manifest.regions[0].sequence_length, Some(2));
        assert_eq!(manifest.regions[0].sequence_position, Some(1));
        assert_eq!(manifest.regions[1].sequence_position, Some(2));
        assert_eq!((manifest.regions[2].key_low, manifest.regions[2].key_high), (43, 43));
        assert_eq!(manifest.regions[2].pitch_keycenter, 43);
        assert_eq!(manifest.regions[2].tune_cents, 60.0);
        assert_eq!(manifest.regions[3].tune_cents, 50.0);
    }

    #[test]
    fn rejects_path_traversal_invalid_keys_and_invalid_sequences() {
        let traversal = "<region> sample=../outside.wav";
        assert!(parse_sfz_mapping(traversal).unwrap_err().message.contains("escapes"));

        let invalid_key = "<region> key=128 sample=samples/test.wav";
        assert!(parse_sfz_mapping(invalid_key).unwrap_err().message.contains("numeric value"));

        let windows_absolute = "<region> sample=C:/Windows/tone.wav";
        assert!(parse_sfz_mapping(windows_absolute).unwrap_err().message.contains("absolute"));

        let invalid_sequence = "<group> seq_length=2\n<region> seq_position=3 sample=samples/test.wav";
        assert!(parse_sfz_mapping(invalid_sequence).unwrap_err().message.contains("exceeds"));
    }

    #[test]
    fn control_default_path_is_applied_to_relative_samples() {
        let source = "<control> default_path=samples\n<region> sample=test.wav";
        let manifest = parse_sfz_mapping(source).unwrap();
        assert_eq!(manifest.regions[0].sample_path, "samples/test.wav");
    }

    #[test]
    fn zone_planner_selects_a_complete_round_robin_deterministically() {
        let source = r#"
<group> lokey=36 hikey=38 pitch_keycenter=38 seq_length=2
<region> seq_position=1 sample=samples/one.wav
<region> seq_position=2 tune=50 sample=samples/two.wav
"#;
        let manifest = parse_sfz_mapping(source).unwrap();
        let first = plan_sfz_zone_selection(&manifest, 38, 90, 0);
        let second = plan_sfz_zone_selection(&manifest, 38, 90, 1);
        let repeat = plan_sfz_zone_selection(&manifest, 38, 90, 2);

        assert_eq!(first.decision, "round_robin_candidate");
        assert_eq!(first.selected_region_index, Some(1));
        assert_eq!(second.selected_region_index, Some(2));
        assert_eq!(repeat.selected_region_index, Some(1));
        assert_eq!(first.semitone_shift, Some(0.0));
        assert_eq!(first.frequency_ratio, Some(1.0));
        assert!((second.frequency_ratio.unwrap() - 2.0_f64.powf(0.5 / 12.0)).abs() < 1e-12);
    }

    #[test]
    fn zone_planner_refuses_incomplete_or_ambiguous_zone_sets() {
        let incomplete = parse_sfz_mapping(
            "<group> key=60 seq_length=2\n<region> seq_position=1 sample=one.wav",
        ).unwrap();
        let incomplete_plan = plan_sfz_zone_selection(&incomplete, 60, 100, 0);
        assert_eq!(incomplete_plan.decision, "incomplete_round_robin_set");
        assert_eq!(incomplete_plan.selected_region_index, None);

        let overlap = parse_sfz_mapping(
            "<group> lokey=60 hikey=60 lovel=0 hivel=127\n<region> sample=soft.wav\n<region> sample=bright.wav",
        ).unwrap();
        let overlap_plan = plan_sfz_zone_selection(&overlap, 60, 100, 0);
        assert_eq!(overlap_plan.decision, "ambiguous_zones");
        assert_eq!(overlap_plan.matching_region_indices, vec![1, 2]);
        assert_eq!(overlap_plan.selected_region_index, None);
    }

    #[test]
    fn zone_planner_reports_uncovered_and_invalid_requests() {
        let manifest = parse_sfz_mapping("<region> key=60 sample=only.wav").unwrap();
        assert_eq!(
            plan_sfz_zone_selection(&manifest, 59, 100, 0).decision,
            "uncovered"
        );
        assert_eq!(
            plan_sfz_zone_selection(&manifest, 128, 100, 0).decision,
            "invalid_request"
        );
        assert_eq!(
            plan_sfz_zone_selection(&manifest, 60, 128, 0).decision,
            "invalid_request"
        );
    }

    #[test]
    fn output_hash_and_coverage_are_deterministic() {
        let one = parse_sfz_mapping(SFZ_FIXTURE).unwrap();
        let two = parse_sfz_mapping(SFZ_FIXTURE).unwrap();
        assert_eq!(one.sfz_sha256, two.sfz_sha256);
        assert_eq!(one.mapped_midi_keys, two.mapped_midi_keys);
        assert_eq!(one.regions, two.regions);
    }
}
