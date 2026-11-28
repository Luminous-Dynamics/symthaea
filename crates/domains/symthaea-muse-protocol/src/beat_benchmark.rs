// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Leakage-safe benchmark manifests for beat/downbeat evaluation (MEL-DJ-009).
//!
//! This module validates benchmark metadata and annotation coordinates. It
//! does not load audio, run a tracker, or compute MIREX scores. In particular,
//! it enforces MIREX 2026's GTZAN/SMC locked-test policy before a score can be
//! considered eligible for that protocol.
//!
//! A clear manifest is necessary but not sufficient for independent test
//! evidence: declared rights, overlap audits and source-master identities
//! still require trustworthy external records and adjudication.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::transition::MAX_SAFE_JSON_INTEGER;

/// Current benchmark-manifest schema version.
pub const BEAT_BENCHMARK_MANIFEST_SCHEMA_VERSION: u16 = 1;

/// MIREX 2026's audio beat-tracking protocol identifier.
pub const MIREX_2026_BEAT_PROTOCOL_ID: &str = "mirex-2026-audio-beat-tracking";

/// The MIREX 2026 locked-test datasets that may not be used for development.
pub const MIREX_2026_LOCKED_DATASETS: [&str; 2] = ["gtzan", "smc"];

/// Defensive input limits for the portable benchmark manifest.
pub const MAX_BENCHMARK_TRACKS: usize = 10_000;
pub const MAX_VALID_SEGMENTS_PER_TRACK: usize = 4_096;
pub const MAX_REFERENCE_EVENTS_PER_TRACK: usize = 200_000;
pub const MAX_TOTAL_REFERENCE_EVENTS: usize = 5_000_000;

/// Dataset partition represented by one source record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeatDatasetSplit {
    Train,
    Validation,
    Development,
    LockedTest,
    PrivateExternal,
}

/// Declared purpose for which this record is being used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeatDatasetPurpose {
    Training,
    Validation,
    ModelSelection,
    ParameterTuning,
    Development,
    BenchmarkEvaluation,
    ExternalValidation,
}

/// Rights state for the exact audio and annotation assets used by this record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeatDatasetRightsStatus {
    Approved,
    Restricted,
    Unknown,
}

/// Provenance class for the reference beat/downbeat annotations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeatAnnotationProvenance {
    HumanAdjudicated,
    AutoThenCorrected,
    AutoOnly,
    SyntheticKnownAnswer,
    Unknown,
}

/// Audit outcome against known training and evaluation corpora.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeatOverlapAuditStatus {
    Clear,
    KnownOverlap,
    Unknown,
}

/// A source-local half-open segment of valid annotation time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeatValidSegment {
    pub start_frame: u64,
    pub end_frame: u64,
}

/// One content-addressed track and its reference annotations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeatBenchmarkTrack {
    /// Stable identifier unique within this manifest.
    pub track_id: String,
    /// Canonical dataset identifier, e.g. "gtzan", "smc", "ballroom".
    pub dataset_id: String,
    /// Exact dataset revision, release, or collection snapshot.
    pub dataset_revision: String,
    /// SHA-256 of the exact audio bytes evaluated.
    pub audio_sha256: String,
    /// Stable identity for the source recording/master, when known.
    /// Different encodings, excerpts, edits and stems should share this ID.
    pub source_master_family_id: Option<String>,
    pub split: BeatDatasetSplit,
    pub purpose: BeatDatasetPurpose,
    pub rights_status: BeatDatasetRightsStatus,
    /// URI or immutable reference to the rights/permission decision.
    pub rights_reference: String,
    /// Required for private external cohorts; records the permitted publication ceiling.
    pub publication_ceiling: Option<String>,
    /// SHA-256 of the exact annotation artifact.
    pub annotation_sha256: String,
    pub annotation_provenance: BeatAnnotationProvenance,
    /// Version of the annotation procedure, toolchain or adjudication.
    pub annotation_revision: String,
    pub overlap_audit: BeatOverlapAuditStatus,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub total_frames: u64,
    /// Valid annotation segments, source-local and half-open.
    pub valid_segments: Vec<BeatValidSegment>,
    /// Beat anchors in source-local frame coordinates.
    pub beat_frames: Vec<u64>,
    /// Whether downbeat annotations are available for this track.
    pub downbeat_annotations_available: bool,
    /// Downbeat anchors; each must also be an annotated beat.
    pub downbeat_frames: Vec<u64>,
}

/// Versioned dataset manifest for a benchmark run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeatBenchmarkManifestV1 {
    pub schema_version: u16,
    pub suite_id: String,
    pub protocol_id: String,
    pub manifest_revision: String,
    pub tracks: Vec<BeatBenchmarkTrack>,
}

/// Whether a manifest problem is a hard rejection or weakens the claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeatBenchmarkViolationSeverity {
    Reject,
    Unqualified,
}

/// Stable machine-readable benchmark-manifest validation failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeatBenchmarkViolationCode {
    UnsupportedSchemaVersion,
    EmptySuiteId,
    EmptyProtocolId,
    EmptyManifestRevision,
    EmptyTrackSet,
    TooManyTracks,
    TooManyValidSegments,
    TooManyReferenceEventsPerTrack,
    TooManyReferenceEventsInManifest,
    EmptyTrackId,
    DuplicateTrackId,
    EmptyDatasetId,
    EmptyDatasetRevision,
    EmptyDatasetReference,
    EmptyCollectionProcessReference,
    InvalidAudioSha256,
    InvalidAnnotationSha256,
    MirexLockedDatasetUsedOutsideLockedTest,
    LockedTestUsedForNonBenchmarkPurpose,
    SplitPurposeMismatch,
    RightsNotApproved,
    EmptyRightsReference,
    MissingPrivatePublicationCeiling,
    UnknownAnnotationProvenance,
    AutoOnlyAnnotationsUnqualified,
    EmptyAnnotationRevision,
    KnownDatasetOverlap,
    OverlapAuditUnknown,
    InvalidSampleRate,
    InvalidChannelCount,
    EmptySourceAudio,
    SourceFrameCountExceedsJsonSafeInteger,
    MissingSourceMasterFamilyId,
    EmptySourceMasterFamilyId,
    NonCanonicalSourceMasterFamilyId,
    EmptyValidSegments,
    EmptyValidSegment,
    ValidSegmentOutOfBounds,
    ValidSegmentsNotOrderedOrOverlapping,
    EmptyBeatAnnotations,
    BeatFramesNotStrictlyIncreasing,
    BeatFrameOutsideValidSegments,
    DownbeatsDeclaredButMissing,
    DownbeatsPresentButNotDeclared,
    DownbeatFramesNotStrictlyIncreasing,
    DownbeatNotPresentInBeatAnnotations,
    AudioDigestCrossesSplits,
    SourceMasterFamilyCrossesSplits,
}

/// One deterministic validation result with a stable field path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeatBenchmarkViolation {
    pub code: BeatBenchmarkViolationCode,
    pub severity: BeatBenchmarkViolationSeverity,
    pub path: String,
}

/// Validate a benchmark manifest in deterministic field and track order.
///
/// Reject-severity findings make the manifest inadmissible. Unqualified
/// findings allow exploratory use but block a strong leakage-safe claim.
pub fn validate_beat_benchmark_manifest(
    manifest: &BeatBenchmarkManifestV1,
) -> Vec<BeatBenchmarkViolation> {
    let mut violations = Vec::new();

    if manifest.schema_version != BEAT_BENCHMARK_MANIFEST_SCHEMA_VERSION {
        push(
            &mut violations,
            BeatBenchmarkViolationCode::UnsupportedSchemaVersion,
            BeatBenchmarkViolationSeverity::Reject,
            "schema_version",
        );
    }
    if manifest.suite_id.trim().is_empty() {
        push(
            &mut violations,
            BeatBenchmarkViolationCode::EmptySuiteId,
            BeatBenchmarkViolationSeverity::Reject,
            "suite_id",
        );
    }
    if manifest.protocol_id.trim().is_empty() {
        push(
            &mut violations,
            BeatBenchmarkViolationCode::EmptyProtocolId,
            BeatBenchmarkViolationSeverity::Reject,
            "protocol_id",
        );
    }
    if manifest.manifest_revision.trim().is_empty() {
        push(
            &mut violations,
            BeatBenchmarkViolationCode::EmptyManifestRevision,
            BeatBenchmarkViolationSeverity::Reject,
            "manifest_revision",
        );
    }
    if manifest.tracks.is_empty() {
        push(
            &mut violations,
            BeatBenchmarkViolationCode::EmptyTrackSet,
            BeatBenchmarkViolationSeverity::Reject,
            "tracks",
        );
    }
    if manifest.tracks.len() > MAX_BENCHMARK_TRACKS {
        push(
            &mut violations,
            BeatBenchmarkViolationCode::TooManyTracks,
            BeatBenchmarkViolationSeverity::Reject,
            "tracks",
        );
    }

    let is_mirex_2026 = manifest
        .protocol_id
        .eq_ignore_ascii_case(MIREX_2026_BEAT_PROTOCOL_ID);
    let mut track_ids = BTreeSet::new();
    let mut audio_splits: BTreeMap<String, BTreeSet<BeatDatasetSplit>> = BTreeMap::new();
    let mut source_master_splits: BTreeMap<String, BTreeSet<BeatDatasetSplit>> =
        BTreeMap::new();
    let mut total_reference_events = 0usize;

    for (index, track) in manifest.tracks.iter().enumerate() {
        let base = format!("tracks[{index}]");

        if track.track_id.trim().is_empty() {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::EmptyTrackId,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.track_id"),
            );
        } else if !track_ids.insert(track.track_id.as_str()) {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::DuplicateTrackId,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.track_id"),
            );
        }

        if track.dataset_id.trim().is_empty() {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::EmptyDatasetId,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.dataset_id"),
            );
        }
        if track.dataset_revision.trim().is_empty() {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::EmptyDatasetRevision,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.dataset_revision"),
            );
        }
        if track.dataset_reference.trim().is_empty() {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::EmptyDatasetReference,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.dataset_reference"),
            );
        }
        if track.collection_process_reference.trim().is_empty() {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::EmptyCollectionProcessReference,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.collection_process_reference"),
            );
        }
        if !is_lowercase_sha256(&track.audio_sha256) {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::InvalidAudioSha256,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.audio_sha256"),
            );
        }
        if !is_lowercase_sha256(&track.annotation_sha256) {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::InvalidAnnotationSha256,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.annotation_sha256"),
            );
        }

        let dataset_id = track.dataset_id.trim().to_ascii_lowercase();
        let mirex_locked = is_mirex_2026
            && MIREX_2026_LOCKED_DATASETS
                .iter()
                .any(|locked| *locked == dataset_id.as_str());
        if mirex_locked
            && (track.split != BeatDatasetSplit::LockedTest
                || track.purpose != BeatDatasetPurpose::BenchmarkEvaluation)
        {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::MirexLockedDatasetUsedOutsideLockedTest,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.split"),
            );
        }
        let purpose_matches_split = match track.split {
            BeatDatasetSplit::Train => {
                track.purpose == BeatDatasetPurpose::Training
            }
            BeatDatasetSplit::Validation => matches!(
                track.purpose,
                BeatDatasetPurpose::Validation
                    | BeatDatasetPurpose::ModelSelection
                    | BeatDatasetPurpose::ParameterTuning
            ),
            BeatDatasetSplit::Development => matches!(
                track.purpose,
                BeatDatasetPurpose::Development
                    | BeatDatasetPurpose::ModelSelection
                    | BeatDatasetPurpose::ParameterTuning
            ),
            BeatDatasetSplit::LockedTest => {
                track.purpose == BeatDatasetPurpose::BenchmarkEvaluation
            }
            BeatDatasetSplit::PrivateExternal => {
                track.purpose == BeatDatasetPurpose::ExternalValidation
            }
        };
        if !purpose_matches_split {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::SplitPurposeMismatch,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.purpose"),
            );
        }
        if track.split == BeatDatasetSplit::LockedTest
            && track.purpose != BeatDatasetPurpose::BenchmarkEvaluation
        {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::LockedTestUsedForNonBenchmarkPurpose,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.purpose"),
            );
        }

        if track.rights_status != BeatDatasetRightsStatus::Approved {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::RightsNotApproved,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.rights_status"),
            );
        }
        if track.rights_reference.trim().is_empty() {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::EmptyRightsReference,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.rights_reference"),
            );
        }
        if track.split == BeatDatasetSplit::PrivateExternal
            && track
                .publication_ceiling
                .as_deref()
                .is_none_or(|ceiling| ceiling.trim().is_empty())
        {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::MissingPrivatePublicationCeiling,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.publication_ceiling"),
            );
        }
        if track.annotation_provenance == BeatAnnotationProvenance::Unknown {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::UnknownAnnotationProvenance,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.annotation_provenance"),
            );
        }
        if track.annotation_provenance == BeatAnnotationProvenance::AutoOnly {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::AutoOnlyAnnotationsUnqualified,
                BeatBenchmarkViolationSeverity::Unqualified,
                format!("{base}.annotation_provenance"),
            );
        }
        if track.annotation_revision.trim().is_empty() {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::EmptyAnnotationRevision,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.annotation_revision"),
            );
        }
        match track.overlap_audit {
            BeatOverlapAuditStatus::Clear => {}
            BeatOverlapAuditStatus::KnownOverlap => push(
                &mut violations,
                BeatBenchmarkViolationCode::KnownDatasetOverlap,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.overlap_audit"),
            ),
            BeatOverlapAuditStatus::Unknown => push(
                &mut violations,
                BeatBenchmarkViolationCode::OverlapAuditUnknown,
                BeatBenchmarkViolationSeverity::Unqualified,
                format!("{base}.overlap_audit"),
            ),
        }

        if track.sample_rate_hz == 0 {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::InvalidSampleRate,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.sample_rate_hz"),
            );
        }
        if track.channels == 0 {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::InvalidChannelCount,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.channels"),
            );
        }
        if track.total_frames == 0 {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::EmptySourceAudio,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.total_frames"),
            );
        }
        if track.total_frames > MAX_SAFE_JSON_INTEGER {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::SourceFrameCountExceedsJsonSafeInteger,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.total_frames"),
            );
        }

        match track.source_master_family_id.as_deref() {
            Some(family) if family.trim().is_empty() => push(
                &mut violations,
                BeatBenchmarkViolationCode::EmptySourceMasterFamilyId,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.source_master_family_id"),
            ),
            Some(family) if family.trim() != family => push(
                &mut violations,
                BeatBenchmarkViolationCode::NonCanonicalSourceMasterFamilyId,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.source_master_family_id"),
            ),
            Some(family) => {
                source_master_splits
                    .entry(family.to_ascii_lowercase())
                    .or_default()
                    .insert(track.split);
            }
            None => push(
                &mut violations,
                BeatBenchmarkViolationCode::MissingSourceMasterFamilyId,
                BeatBenchmarkViolationSeverity::Unqualified,
                format!("{base}.source_master_family_id"),
            ),
        }
        audio_splits
            .entry(track.audio_sha256.clone())
            .or_default()
            .insert(track.split);

        let reference_events = track
            .beat_frames
            .len()
            .saturating_add(track.downbeat_frames.len());
        if reference_events > MAX_REFERENCE_EVENTS_PER_TRACK {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::TooManyReferenceEventsPerTrack,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{base}.beat_frames"),
            );
        }
        total_reference_events = total_reference_events.saturating_add(reference_events);

        validate_track_annotations(track, &base, &mut violations);
    }

    if total_reference_events > MAX_TOTAL_REFERENCE_EVENTS {
        push(
            &mut violations,
            BeatBenchmarkViolationCode::TooManyReferenceEventsInManifest,
            BeatBenchmarkViolationSeverity::Reject,
            "tracks",
        );
    }

    for (audio_sha256, splits) in audio_splits {
        if splits.len() > 1 {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::AudioDigestCrossesSplits,
                BeatBenchmarkViolationSeverity::Reject,
                format!("audio_sha256[{audio_sha256}]"),
            );
        }
    }
    for (family_id, splits) in source_master_splits {
        if splits.len() > 1 {
            push(
                &mut violations,
                BeatBenchmarkViolationCode::SourceMasterFamilyCrossesSplits,
                BeatBenchmarkViolationSeverity::Reject,
                format!("source_master_family_id[{family_id}]"),
            );
        }
    }

    violations
}

fn validate_track_annotations(
    track: &BeatBenchmarkTrack,
    base: &str,
    violations: &mut Vec<BeatBenchmarkViolation>,
) {
    if track.valid_segments.is_empty() {
        push(
            violations,
            BeatBenchmarkViolationCode::EmptyValidSegments,
            BeatBenchmarkViolationSeverity::Reject,
            format!("{base}.valid_segments"),
        );
    }
    if track.valid_segments.len() > MAX_VALID_SEGMENTS_PER_TRACK {
        push(
            violations,
            BeatBenchmarkViolationCode::TooManyValidSegments,
            BeatBenchmarkViolationSeverity::Reject,
            format!("{base}.valid_segments"),
        );
    }

    let mut previous_segment_end = None;
    for (index, segment) in track.valid_segments.iter().enumerate() {
        let path = format!("{base}.valid_segments[{index}]");
        if segment.start_frame >= segment.end_frame {
            push(
                violations,
                BeatBenchmarkViolationCode::EmptyValidSegment,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{path}.start_frame"),
            );
        }
        if segment.end_frame > track.total_frames {
            push(
                violations,
                BeatBenchmarkViolationCode::ValidSegmentOutOfBounds,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{path}.end_frame"),
            );
        }
        if previous_segment_end
            .is_some_and(|previous| segment.start_frame < previous)
        {
            push(
                violations,
                BeatBenchmarkViolationCode::ValidSegmentsNotOrderedOrOverlapping,
                BeatBenchmarkViolationSeverity::Reject,
                path,
            );
        }
        previous_segment_end = Some(segment.end_frame);
    }

    if track.beat_frames.is_empty() {
        push(
            violations,
            BeatBenchmarkViolationCode::EmptyBeatAnnotations,
            BeatBenchmarkViolationSeverity::Reject,
            format!("{base}.beat_frames"),
        );
    }
    validate_ordered_frames(
        &track.beat_frames,
        &track.valid_segments,
        track.total_frames,
        BeatBenchmarkViolationCode::BeatFramesNotStrictlyIncreasing,
        BeatBenchmarkViolationCode::BeatFrameOutsideValidSegments,
        &format!("{base}.beat_frames"),
        violations,
    );

    if track.downbeat_annotations_available && track.downbeat_frames.is_empty() {
        push(
            violations,
            BeatBenchmarkViolationCode::DownbeatsDeclaredButMissing,
            BeatBenchmarkViolationSeverity::Reject,
            format!("{base}.downbeat_frames"),
        );
    }
    if !track.downbeat_annotations_available && !track.downbeat_frames.is_empty() {
        push(
            violations,
            BeatBenchmarkViolationCode::DownbeatsPresentButNotDeclared,
            BeatBenchmarkViolationSeverity::Reject,
            format!("{base}.downbeat_annotations_available"),
        );
    }

    validate_ordered_frames(
        &track.downbeat_frames,
        &track.valid_segments,
        track.total_frames,
        BeatBenchmarkViolationCode::DownbeatFramesNotStrictlyIncreasing,
        BeatBenchmarkViolationCode::BeatFrameOutsideValidSegments,
        &format!("{base}.downbeat_frames"),
        violations,
    );
    for (index, frame) in track.downbeat_frames.iter().enumerate() {
        if track.beat_frames.binary_search(frame).is_err() {
            push(
                violations,
                BeatBenchmarkViolationCode::DownbeatNotPresentInBeatAnnotations,
                format!("{base}.downbeat_frames[{index}]"),
            );
        }
    }
}

/// Binary-search a sorted, non-overlapping valid-segment list.
fn frame_in_valid_segments(frame: u64, segments: &[BeatValidSegment]) -> bool {
    let insertion = segments.partition_point(|segment| segment.start_frame <= frame);
    insertion > 0 && frame < segments[insertion - 1].end_frame
}

fn validate_ordered_frames(
    frames: &[u64],
    segments: &[BeatValidSegment],
    total_frames: u64,
    order_code: BeatBenchmarkViolationCode,
    segment_code: BeatBenchmarkViolationCode,
    path: &str,
    violations: &mut Vec<BeatBenchmarkViolation>,
) {
    for (index, frame) in frames.iter().enumerate() {
        if *frame >= total_frames
            || *frame > MAX_SAFE_JSON_INTEGER
            || !frame_in_valid_segments(*frame, segments)
        {
            push(
                violations,
                segment_code,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{path}[{index}]"),
            );
        }
        if index > 0 && frames[index - 1] >= *frame {
            push(
                violations,
                order_code,
                BeatBenchmarkViolationSeverity::Reject,
                format!("{path}[{index}]"),
            );
        }
    }
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn push(
    violations: &mut Vec<BeatBenchmarkViolation>,
    code: BeatBenchmarkViolationCode,
    severity: BeatBenchmarkViolationSeverity,
    path: impl Into<String>,
) {
    violations.push(BeatBenchmarkViolation {
        code,
        severity,
        path: path.into(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_track(
        track_id: &str,
        dataset_id: &str,
        split: BeatDatasetSplit,
        purpose: BeatDatasetPurpose,
        audio_sha: &str,
        master_id: &str,
    ) -> BeatBenchmarkTrack {
        BeatBenchmarkTrack {
            track_id: track_id.to_string(),
            dataset_id: dataset_id.to_string(),
            dataset_revision: "2026-01".to_string(),
            dataset_reference: "urn:fixture:synthetic-dataset".to_string(),
            collection_process_reference: "urn:fixture:synthetic-collection-process".to_string(),
            audio_sha256: audio_sha.to_string(),
            source_master_family_id: Some(master_id.to_string()),
            split,
            purpose,
            rights_status: BeatDatasetRightsStatus::Approved,
            rights_reference: "urn:fixture:synthetic-rights-reference".to_string(),
            publication_ceiling: None,
            annotation_sha256: "a".repeat(64),
            annotation_provenance: BeatAnnotationProvenance::HumanAdjudicated,
            annotation_revision: "annotation-v1".to_string(),
            overlap_audit: BeatOverlapAuditStatus::Clear,
            sample_rate_hz: 44_100,
            channels: 1,
            total_frames: 100_000,
            valid_segments: vec![BeatValidSegment {
                start_frame: 0,
                end_frame: 100_000,
            }],
            beat_frames: vec![0, 22_050, 44_100, 66_150, 88_200],
            downbeat_annotations_available: true,
            downbeat_frames: vec![0, 44_100, 88_200],
        }
    }

    fn manifest(tracks: Vec<BeatBenchmarkTrack>) -> BeatBenchmarkManifestV1 {
        BeatBenchmarkManifestV1 {
            schema_version: BEAT_BENCHMARK_MANIFEST_SCHEMA_VERSION,
            suite_id: "mel-dj-007-known-answer".to_string(),
            protocol_id: MIREX_2026_BEAT_PROTOCOL_ID.to_string(),
            manifest_revision: "manifest-v1".to_string(),
            tracks,
        }
    }

    fn codes(manifest: &BeatBenchmarkManifestV1) -> Vec<BeatBenchmarkViolationCode> {
        validate_beat_benchmark_manifest(manifest)
            .into_iter()
            .map(|violation| violation.code)
            .collect()
    }

    #[test]
    fn accepts_mirex_locked_test_track_with_complete_rights_and_annotations() {
        let track = valid_track(
            "gtzan-0001",
            "gtzan",
            BeatDatasetSplit::LockedTest,
            BeatDatasetPurpose::BenchmarkEvaluation,
            &"1".repeat(64),
            "master-gtzan-0001",
        );
        assert!(validate_beat_benchmark_manifest(&manifest(vec![track])).is_empty());
    }

    #[test]
    fn rejects_too_many_valid_segments_before_processing_them() {
        let mut track = valid_track(
            "too-many-segments",
            "private-corpus",
            BeatDatasetSplit::Development,
            BeatDatasetPurpose::Development,
            &"f".repeat(64),
            "master-segments",
        );
        track.valid_segments = vec![
            BeatValidSegment {
                start_frame: 0,
                end_frame: 1,
            };
            MAX_VALID_SEGMENTS_PER_TRACK + 1
        ];
        assert!(codes(&manifest(vec![track]))
            .contains(&BeatBenchmarkViolationCode::TooManyValidSegments));
    }

    #[test]
    fn rejects_too_many_reference_events_per_track() {
        let mut track = valid_track(
            "too-many-events",
            "private-corpus",
            BeatDatasetSplit::Development,
            BeatDatasetPurpose::Development,
            &"e".repeat(64),
            "master-events",
        );
        track.beat_frames = (0..=MAX_REFERENCE_EVENTS_PER_TRACK as u64)
            .collect();
        track.valid_segments = vec![BeatValidSegment {
            start_frame: 0,
            end_frame: MAX_REFERENCE_EVENTS_PER_TRACK as u64 + 1,
        }];
        track.total_frames = MAX_REFERENCE_EVENTS_PER_TRACK as u64 + 1;
        track.downbeat_annotations_available = false;
        track.downbeat_frames.clear();
        assert!(codes(&manifest(vec![track]))
            .contains(&BeatBenchmarkViolationCode::TooManyReferenceEventsPerTrack));
    }

    #[test]
    fn rejects_gtzan_development_and_model_selection_use() {
        let track = valid_track(
            "gtzan-0001",
            "GTZAN",
            BeatDatasetSplit::Development,
            BeatDatasetPurpose::ModelSelection,
            &"1".repeat(64),
            "master-gtzan-0001",
        );
        let found = codes(&manifest(vec![track]));
        assert!(found.contains(
            &BeatBenchmarkViolationCode::MirexLockedDatasetUsedOutsideLockedTest
        ));
    }

    #[test]
    fn rejects_smc_parameter_tuning_use() {
        let track = valid_track(
            "smc-0001",
            "smc",
            BeatDatasetSplit::Validation,
            BeatDatasetPurpose::ParameterTuning,
            &"2".repeat(64),
            "master-smc-0001",
        );
        assert!(codes(&manifest(vec![track])).contains(
            &BeatBenchmarkViolationCode::MirexLockedDatasetUsedOutsideLockedTest
        ));
    }

    #[test]
    fn rejects_locked_test_used_for_parameter_tuning() {
        let track = valid_track(
            "ballroom-0001",
            "ballroom",
            BeatDatasetSplit::LockedTest,
            BeatDatasetPurpose::ParameterTuning,
            &"3".repeat(64),
            "master-ballroom-0001",
        );
        assert!(codes(&manifest(vec![track])).contains(
            &BeatBenchmarkViolationCode::LockedTestUsedForNonBenchmarkPurpose
        ));
    }

    #[test]
    fn rejects_identical_audio_digest_across_disjoint_splits() {
        let train = valid_track(
            "train-track",
            "private-corpus",
            BeatDatasetSplit::Train,
            BeatDatasetPurpose::Training,
            &"4".repeat(64),
            "master-train",
        );
        let test = valid_track(
            "test-track",
            "private-corpus",
            BeatDatasetSplit::LockedTest,
            BeatDatasetPurpose::BenchmarkEvaluation,
            &"4".repeat(64),
            "master-test",
        );
        assert!(codes(&manifest(vec![train, test]))
            .contains(&BeatBenchmarkViolationCode::AudioDigestCrossesSplits));
    }

    #[test]
    fn rejects_same_source_master_across_different_file_hashes_and_splits() {
        let train = valid_track(
            "train-encoding",
            "private-corpus",
            BeatDatasetSplit::Train,
            BeatDatasetPurpose::Training,
            &"5".repeat(64),
            "same-source-master",
        );
        let test = valid_track(
            "test-remaster",
            "private-corpus",
            BeatDatasetSplit::LockedTest,
            BeatDatasetPurpose::BenchmarkEvaluation,
            &"6".repeat(64),
            "same-source-master",
        );
        assert!(codes(&manifest(vec![train, test]))
            .contains(&BeatBenchmarkViolationCode::SourceMasterFamilyCrossesSplits));
    }

    #[test]
    fn reports_unknown_source_master_and_overlap_as_unqualified() {
        let mut track = valid_track(
            "unknown-source",
            "private-corpus",
            BeatDatasetSplit::Development,
            BeatDatasetPurpose::Development,
            &"7".repeat(64),
            "placeholder",
        );
        track.source_master_family_id = None;
        track.overlap_audit = BeatOverlapAuditStatus::Unknown;
        let result = validate_beat_benchmark_manifest(&manifest(vec![track]));
        assert!(result.iter().any(|violation| {
            violation.code == BeatBenchmarkViolationCode::MissingSourceMasterFamilyId
                && violation.severity == BeatBenchmarkViolationSeverity::Unqualified
        }));
        assert!(result.iter().any(|violation| {
            violation.code == BeatBenchmarkViolationCode::OverlapAuditUnknown
                && violation.severity == BeatBenchmarkViolationSeverity::Unqualified
        }));
    }

    #[test]
    fn rejects_non_approved_rights_and_unknown_annotation_provenance() {
        let mut track = valid_track(
            "rights-unknown",
            "private-corpus",
            BeatDatasetSplit::Development,
            BeatDatasetPurpose::Development,
            &"8".repeat(64),
            "master-8",
        );
        track.rights_status = BeatDatasetRightsStatus::Unknown;
        track.annotation_provenance = BeatAnnotationProvenance::Unknown;
        track.rights_reference.clear();
        let found = codes(&manifest(vec![track]));
        assert!(found.contains(&BeatBenchmarkViolationCode::RightsNotApproved));
        assert!(found.contains(&BeatBenchmarkViolationCode::EmptyRightsReference));
        assert!(found.contains(&BeatBenchmarkViolationCode::UnknownAnnotationProvenance));
    }

    #[test]
    fn requires_publication_ceiling_for_private_external_cohorts() {
        let mut track = valid_track(
            "yamaha-private-0001",
            "yamaha-balanced-private",
            BeatDatasetSplit::PrivateExternal,
            BeatDatasetPurpose::ExternalValidation,
            &"f".repeat(64),
            "master-yamaha-0001",
        );
        track.publication_ceiling = None;
        assert!(codes(&manifest(vec![track])).contains(
            &BeatBenchmarkViolationCode::MissingPrivatePublicationCeiling
        ));
    }

    #[test]
    fn rejects_overlapping_or_out_of_bounds_valid_segments() {
        let mut track = valid_track(
            "bad-segments",
            "private-corpus",
            BeatDatasetSplit::Development,
            BeatDatasetPurpose::Development,
            &"9".repeat(64),
            "master-9",
        );
        track.valid_segments = vec![
            BeatValidSegment {
                start_frame: 0,
                end_frame: 60_000,
            },
            BeatValidSegment {
                start_frame: 50_000,
                end_frame: 120_000,
            },
        ];
        let found = codes(&manifest(vec![track]));
        assert!(found.contains(
            &BeatBenchmarkViolationCode::ValidSegmentsNotOrderedOrOverlapping
        ));
        assert!(found.contains(&BeatBenchmarkViolationCode::ValidSegmentOutOfBounds));
    }

    #[test]
    fn rejects_unsorted_duplicate_and_out_of_segment_beats() {
        let mut track = valid_track(
            "bad-beats",
            "private-corpus",
            BeatDatasetSplit::Development,
            BeatDatasetPurpose::Development,
            &"b".repeat(64),
            "master-b",
        );
        track.valid_segments = vec![BeatValidSegment {
            start_frame: 10_000,
            end_frame: 80_000,
        }];
        track.beat_frames = vec![10_000, 40_000, 40_000, 90_000];
        let found = codes(&manifest(vec![track]));
        assert!(found.contains(&BeatBenchmarkViolationCode::BeatFramesNotStrictlyIncreasing));
        assert!(found.contains(&BeatBenchmarkViolationCode::BeatFrameOutsideValidSegments));
    }

    #[test]
    fn rejects_downbeat_that_is_not_an_annotated_beat() {
        let mut track = valid_track(
            "bad-downbeat",
            "private-corpus",
            BeatDatasetSplit::Development,
            BeatDatasetPurpose::Development,
            &"c".repeat(64),
            "master-c",
        );
        track.downbeat_frames = vec![11_025];
        assert!(codes(&manifest(vec![track]))
            .contains(&BeatBenchmarkViolationCode::DownbeatNotPresentInBeatAnnotations));
    }

    #[test]
    fn rejects_unknown_fields_instead_of_ignoring_policy_data() {
        let mut value = serde_json::to_value(manifest(vec![valid_track(
            "gtzan-0001",
            "gtzan",
            BeatDatasetSplit::LockedTest,
            BeatDatasetPurpose::BenchmarkEvaluation,
            &"d".repeat(64),
            "master-d",
        )]))
        .unwrap();
        value["tracks"][0]
            .as_object_mut()
            .unwrap()
            .insert("allow_model_selection".to_string(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<BeatBenchmarkManifestV1>(value).is_err());
    }

    #[test]
    fn rejects_source_clock_above_exact_json_integer_range() {
        let mut track = valid_track(
            "large-clock",
            "private-corpus",
            BeatDatasetSplit::Development,
            BeatDatasetPurpose::Development,
            &"e".repeat(64),
            "master-e",
        );
        track.total_frames = MAX_SAFE_JSON_INTEGER + 1;
        assert!(codes(&manifest(vec![track])).contains(
            &BeatBenchmarkViolationCode::SourceFrameCountExceedsJsonSafeInteger
        ));
    }

    #[test]
    fn checked_in_valid_fixture_deserializes_and_validates() {
        let json = include_str!("../fixtures/beat-benchmark-v1/valid.json");
        let manifest: BeatBenchmarkManifestV1 = serde_json::from_str(json).unwrap();
        assert!(validate_beat_benchmark_manifest(&manifest).is_empty());
    }

    #[test]
    fn checked_in_locked_test_violation_fixture_is_rejected() {
        let json = include_str!("../fixtures/beat-benchmark-v1/gtzan-development-invalid.json");
        let manifest: BeatBenchmarkManifestV1 = serde_json::from_str(json).unwrap();
        assert!(codes(&manifest).contains(
            &BeatBenchmarkViolationCode::MirexLockedDatasetUsedOutsideLockedTest
        ));
    }
}
