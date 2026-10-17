// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Filename-derived metadata for the pitched-percussion sample boundary.
//!
//! Timpani samples in VCSL and VSCO-2-CE do not use the scientific-pitch
//! tokens consumed by the ordinary sample bank. Their filenames instead
//! encode a drum identity, hit/roll gesture, velocity layer and round-robin.
//! This module preserves those facts without inventing a MIDI pitch assignment.
//!
//! The indexer intentionally does not decode audio or claim that the declared
//! drum identity is correctly tuned. It is a deterministic asset-inventory
//! tranche that can be joined to a separately verified pitch/tuning manifest.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimpaniGesture {
    Hit,
    Roll,
}

/// Structured filename identity. The source_variant retains labels such as A
/// in Timpani1A; it is not interpreted as a pitch or a tuning frequency.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimpaniSampleDescriptor {
    pub drum_number: u8,
    pub source_variant: Option<char>,
    pub gesture: TimpaniGesture,
    pub velocity_layer: u8,
    pub round_robin: u8,
}

/// Caller-declared source identity. These fields are provenance annotations,
/// not independently verified by inspecting a local directory.
#[derive(Debug, Clone, Default)]
pub struct TimpaniIndexOptions {
    pub source_label: Option<String>,
    pub source_revision: Option<String>,
    pub hash_content: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimpaniSampleFile {
    /// Path relative to the supplied instrument root; never an absolute path.
    pub relative_path: String,
    pub byte_len: u64,
    pub descriptor: TimpaniSampleDescriptor,
    /// Present only when content hashing was explicitly enabled.
    pub content_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimpaniSampleIndexV1 {
    pub schema: String,
    pub declared_source_label: Option<String>,
    pub declared_source_revision: Option<String>,
    pub file_count: usize,
    pub hit_count: usize,
    pub roll_count: usize,
    pub drum_numbers: Vec<u8>,
    pub source_variants: Vec<char>,
    pub velocity_layers: Vec<u8>,
    pub round_robin_indices: Vec<u8>,
    /// SHA-256 of sorted relative-path + byte-length records (not audio bytes).
    pub path_size_manifest_sha256: String,
    /// SHA-256 binding sorted relative paths to content hashes, if requested.
    pub content_manifest_sha256: Option<String>,
    pub content_hashes_included: bool,
    /// Asset indexing does not decode or qualify the audio.
    pub audio_decode_evidence: String,
    /// Drum labels are not silently converted into MIDI pitch assignments.
    pub pitch_mapping_evidence: String,
    pub files: Vec<TimpaniSampleFile>,
}

/// Parse filenames used by the pinned VCSL and VSCO-2-CE timpani sample trees.
///
/// Accepted examples include:
/// - Timpani1_Hit_v3_rr2_Sum.wav
/// - Timpani1_Roll_v3_rr1_Sum.wav
/// - Timpani1A_hit_v2_rr1_main.wav
///
/// No MIDI note or tuning is inferred. Timpani1A remains drum 1, source variant A.
pub fn parse_timpani_sample_filename(
    filename: impl AsRef<Path>,
) -> Option<TimpaniSampleDescriptor> {
    let path = filename.as_ref();
    if !path
        .extension()?
        .to_str()?
        .eq_ignore_ascii_case("wav")
    {
        return None;
    }

    let stem = path.file_stem()?.to_str()?;
    let tokens: Vec<&str> = stem.split('_').collect();
    let (drum_number, source_variant) = parse_drum_identifier(tokens.first()?)?;

    let mut gesture = None;
    let mut velocity_layer = None;
    let mut round_robin = None;

    for token in tokens.iter().skip(1) {
        if token.eq_ignore_ascii_case("hit") {
            if gesture.replace(TimpaniGesture::Hit).is_some() {
                return None;
            }
            continue;
        }
        if token.eq_ignore_ascii_case("roll") {
            if gesture.replace(TimpaniGesture::Roll).is_some() {
                return None;
            }
            continue;
        }
        if let Some(layer) = parse_positive_number_token(token, "v") {
            if velocity_layer.replace(layer).is_some() {
                return None;
            }
            continue;
        }
        if let Some(take) = parse_positive_number_token(token, "rr") {
            if round_robin.replace(take).is_some() {
                return None;
            }
        }
    }

    Some(TimpaniSampleDescriptor {
        drum_number,
        source_variant,
        gesture: gesture?,
        velocity_layer: velocity_layer?,
        round_robin: round_robin?,
    })
}

fn parse_drum_identifier(token: &str) -> Option<(u8, Option<char>)> {
    const PREFIX: &str = "Timpani";
    if !token.get(..PREFIX.len())?.eq_ignore_ascii_case(PREFIX) {
        return None;
    }

    let suffix = token.get(PREFIX.len()..)?;
    let digits_end = suffix
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(suffix.len());
    if digits_end == 0 {
        return None;
    }

    let drum_number = suffix.get(..digits_end)?.parse::<u8>().ok()?;
    if drum_number == 0 {
        return None;
    }

    let tail = suffix.get(digits_end..)?;
    let source_variant = if tail.is_empty() {
        None
    } else if tail.len() == 1 && tail.as_bytes()[0].is_ascii_alphabetic() {
        Some(tail.as_bytes()[0].to_ascii_uppercase() as char)
    } else {
        return None;
    };

    Some((drum_number, source_variant))
}

fn parse_positive_number_token(token: &str, prefix: &str) -> Option<u8> {
    let lower = token.to_ascii_lowercase();
    let number = lower.strip_prefix(prefix)?;
    let parsed = number.parse::<u8>().ok()?;
    (parsed > 0).then_some(parsed)
}

/// Build a deterministic, path-relative index of named timpani WAV files.
///
/// Symlinks are not followed. Unrecognized files are omitted rather than
/// guessed into the inventory. If hash_content is false, only paths and file
/// sizes are read; this is not a content hash or an audio-decode result.
pub fn index_timpani_samples(
    root: &Path,
    options: &TimpaniIndexOptions,
) -> io::Result<TimpaniSampleIndexV1> {
    let root_metadata = fs::symlink_metadata(root)?;
    if !root_metadata.file_type().is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "timpani sample root must be a real directory, not a symlink",
        ));
    }
    if let Some(revision) = options.source_revision.as_deref() {
        if !(7..=64).contains(&revision.len())
            || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "declared source revision must be 7-64 hexadecimal characters",
            ));
        }
    }

    let mut files = Vec::new();
    visit_directory(root, root, options.hash_content, &mut files)?;
    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));

    let path_size_manifest_sha256 = hash_path_size_manifest(&files);
    let content_manifest_sha256 = options
        .hash_content
        .then(|| hash_content_manifest(&files));

    let drum_numbers = sorted_unique(files.iter().map(|file| file.descriptor.drum_number));
    let source_variants = sorted_unique(
        files
            .iter()
            .filter_map(|file| file.descriptor.source_variant),
    );
    let velocity_layers = sorted_unique(files.iter().map(|file| file.descriptor.velocity_layer));
    let round_robin_indices = sorted_unique(files.iter().map(|file| file.descriptor.round_robin));
    let hit_count = files
        .iter()
        .filter(|file| file.descriptor.gesture == TimpaniGesture::Hit)
        .count();
    let roll_count = files.len() - hit_count;

    Ok(TimpaniSampleIndexV1 {
        schema: "muse-timpani-sample-index-v1".to_owned(),
        declared_source_label: options.source_label.clone(),
        declared_source_revision: options.source_revision.clone(),
        file_count: files.len(),
        hit_count,
        roll_count,
        drum_numbers,
        source_variants,
        velocity_layers,
        round_robin_indices,
        path_size_manifest_sha256,
        content_manifest_sha256,
        content_hashes_included: options.hash_content,
        audio_decode_evidence: "not_performed".to_owned(),
        pitch_mapping_evidence: "not_inferred_from_filename".to_owned(),
        files,
    })
}

fn visit_directory(
    root: &Path,
    current: &Path,
    hash_content: bool,
    indexed: &mut Vec<TimpaniSampleFile>,
) -> io::Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let entry_type = entry.file_type()?;
        if entry_type.is_symlink() {
            continue;
        }

        let path = entry.path();
        if entry_type.is_dir() {
            visit_directory(root, &path, hash_content, indexed)?;
            continue;
        }
        if !entry_type.is_file() {
            continue;
        }

        let Some(descriptor) = parse_timpani_sample_filename(&path) else {
            continue;
        };
        let metadata = entry.metadata()?;
        let relative = path.strip_prefix(root).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "indexed sample escaped the supplied root",
            )
        })?;
        let relative_path = relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        let content_sha256 = if hash_content {
            Some(hash_file(&path)?)
        } else {
            None
        };

        indexed.push(TimpaniSampleFile {
            relative_path,
            byte_len: metadata.len(),
            descriptor,
            content_sha256,
        });
    }
    Ok(())
}

fn hash_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn hash_path_size_manifest(files: &[TimpaniSampleFile]) -> String {
    let mut hash = Sha256::new();
    for file in files {
        hash.update(file.relative_path.as_bytes());
        hash.update([0]);
        hash.update(file.byte_len.to_be_bytes());
        hash.update(b"\n");
    }
    format!("{:x}", hash.finalize())
}

fn hash_content_manifest(files: &[TimpaniSampleFile]) -> String {
    let mut hash = Sha256::new();
    for file in files {
        hash.update(file.relative_path.as_bytes());
        hash.update([0]);
        hash.update(file.content_sha256.as_deref().unwrap_or("").as_bytes());
        hash.update(b"\n");
    }
    format!("{:x}", hash.finalize())
}

fn sorted_unique<T: Ord>(values: impl IntoIterator<Item = T>) -> Vec<T> {
    values.into_iter().collect::<BTreeSet<_>>().into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

    fn temporary_root() -> PathBuf {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "symthaea-muse-timpani-index-{}-{id}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create temporary test root");
        root
    }

    #[test]
    fn parses_both_pinned_library_filename_families() {
        let hit = parse_timpani_sample_filename("Timpani1_Hit_v3_rr2_Sum.wav").unwrap();
        assert_eq!(hit.drum_number, 1);
        assert_eq!(hit.source_variant, None);
        assert_eq!(hit.gesture, TimpaniGesture::Hit);
        assert_eq!(hit.velocity_layer, 3);
        assert_eq!(hit.round_robin, 2);

        let roll = parse_timpani_sample_filename("Timpani1_Roll_v3_rr1_Sum.wav").unwrap();
        assert_eq!(roll.gesture, TimpaniGesture::Roll);

        let variant = parse_timpani_sample_filename("Timpani1A_hit_v2_rr1_main.wav").unwrap();
        assert_eq!(variant.drum_number, 1);
        assert_eq!(variant.source_variant, Some('A'));
        assert_eq!(variant.gesture, TimpaniGesture::Hit);
    }

    #[test]
    fn rejects_ambiguous_or_non_timpani_names_without_guessing() {
        assert!(parse_timpani_sample_filename("Violin_F3_v2_rr1.wav").is_none());
        assert!(parse_timpani_sample_filename("Timpani1_Hit_v0_rr1_Sum.wav").is_none());
        assert!(parse_timpani_sample_filename("Timpani1_Hit_v2_rr1_Sum.flac").is_none());
        assert!(parse_timpani_sample_filename("Timpani0_Hit_v2_rr1_Sum.wav").is_none());
        assert!(parse_timpani_sample_filename("Timpani1AB_hit_v2_rr1_main.wav").is_none());
        assert!(parse_timpani_sample_filename("Timpani1_Hit_Roll_v2_rr1_Sum.wav").is_none());
    }

    #[test]
    fn recursive_index_is_sorted_and_keeps_pitch_claims_absent() {
        let root = temporary_root();
        fs::create_dir_all(root.join("Rolls")).unwrap();
        fs::write(root.join("Timpani2_Hit_v1_rr2_Sum.wav"), b"hit-2").unwrap();
        fs::write(
            root.join("Rolls/Timpani1_Roll_v3_rr1_Sum.wav"),
            b"roll-1",
        )
        .unwrap();
        fs::write(root.join("not-a-timpani-sample.wav"), b"ignored").unwrap();

        let report = index_timpani_samples(&root, &TimpaniIndexOptions::default()).unwrap();
        assert_eq!(report.file_count, 2);
        assert_eq!(report.hit_count, 1);
        assert_eq!(report.roll_count, 1);
        assert_eq!(report.drum_numbers, vec![1, 2]);
        assert_eq!(
            report.files[0].relative_path,
            "Rolls/Timpani1_Roll_v3_rr1_Sum.wav"
        );
        assert_eq!(report.audio_decode_evidence, "not_performed");
        assert_eq!(report.pitch_mapping_evidence, "not_inferred_from_filename");
        assert!(report.content_manifest_sha256.is_none());
        assert!(!report.files[0].relative_path.starts_with('/'));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn content_manifest_detects_same_size_sample_replacement() {
        let root = temporary_root();
        let sample = root.join("Timpani5_Hit_v3_rr1_Sum.wav");
        fs::write(&sample, b"AAAA").unwrap();
        let options = TimpaniIndexOptions {
            source_label: Some("VSCO-2-CE".to_owned()),
            source_revision: Some("440300901dfe9275fd84e0b7763af1f8443ae62e".to_owned()),
            hash_content: true,
        };

        let first = index_timpani_samples(&root, &options).unwrap();
        fs::write(&sample, b"BBBB").unwrap();
        let second = index_timpani_samples(&root, &options).unwrap();

        assert_eq!(first.path_size_manifest_sha256, second.path_size_manifest_sha256);
        assert_ne!(first.files[0].content_sha256, second.files[0].content_sha256);
        assert_ne!(first.content_manifest_sha256, second.content_manifest_sha256);
        assert_eq!(
            first.declared_source_revision.as_deref(),
            Some("440300901dfe9275fd84e0b7763af1f8443ae62e")
        );

        fs::remove_dir_all(root).unwrap();
    }
}
