// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metadata-only admission and leakage checks for bioacoustic corpora.
//!
//! This module does not download or authenticate recordings and cannot decide
//! whether a license interpretation is legally correct. It validates that a
//! caller supplied explicit rights-admission evidence, stable content hashes,
//! valid signal/annotation bounds, and a split plan whose declared provenance
//! does not leak across evaluation partitions.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const BIOACOUSTIC_CORPUS_SCHEMA_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BioacousticSplit {
    Development,
    Train,
    Validation,
    Test,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BioacousticIntendedUse {
    NonCommercialResearch,
    CommercialResearch,
    PublishDerivedMetrics,
    RedistributeSourceAudio,
    RedistributeDerivedArtifacts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RightsAdmissionDecision {
    Approved,
    PendingReview,
    Denied,
}

/// Human-reviewed rights evidence for the exact intended use.
///
/// `Approved` is an asserted decision record, not a legal conclusion made by
/// this code. The referenced terms must cover the actual audio, annotations,
/// and any source recordings on which the dataset depends.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BioacousticRightsEvidence {
    pub license_id: String,
    pub license_uri: String,
    pub source_asset_terms_uri: String,
    pub decision_reference: String,
    pub intended_use: BioacousticIntendedUse,
    pub decision: RightsAdmissionDecision,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BioacousticAnnotation {
    pub annotation_id: String,
    /// Event interval relative to the audio asset/crop, not the source timeline.
    pub interval: crate::TimeSpan,
    pub label: String,
    pub annotation_source: String,
    pub frequency_min_hz: Option<f64>,
    pub frequency_max_hz: Option<f64>,
}

/// Explicitly unscored time span (e.g. ambiguous call, overlapping caller,
/// occlusion, or annotation uncertainty). Predictions sufficiently contained in
/// these spans are excluded from detection metrics, not counted as negatives.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BioacousticUnscoredRegion {
    pub interval: crate::TimeSpan,
    pub reason: String,
    /// Provenance for the human/model annotation decision to exclude this span.
    pub annotation_source: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BioacousticRecording {
    /// Stable ID for this exact evaluated sample/file/crop.
    pub sample_id: String,
    /// URI or repository-relative provenance pointer; never inferred from a filename.
    pub asset_uri: String,
    /// SHA-256 of the exact bytes of this audio asset.
    pub content_sha256: String,
    pub split: BioacousticSplit,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub sample_count: u64,
    /// Original continuous recording ID; distinct crops must share this value.
    pub source_recording_id: Option<String>,
    /// Start offset (seconds) of this asset/crop within its original source recording.
    /// Required by event evaluation when multiple Test assets share a source ID.
    /// Schema v3 requires the key to be present; use null when unavailable.
    pub source_start_s: Option<f64>,
    pub session_id: Option<String>,
    pub site_id: Option<String>,
    pub individual_ids: Vec<String>,
    pub annotations: Vec<BioacousticAnnotation>,
    /// Regions explicitly excluded from scoring because annotation is uncertain.
    /// Schema v3 requires the key to be present; use an empty array when none exist.
    pub unscored_regions: Vec<BioacousticUnscoredRegion>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BioacousticCorpusManifest {
    pub schema_version: u32,
    pub dataset_id: String,
    pub dataset_revision: String,
    pub source_uri: String,
    pub intended_use: BioacousticIntendedUse,
    pub rights: BioacousticRightsEvidence,
    pub recordings: Vec<BioacousticRecording>,
}

impl BioacousticCorpusManifest {
    /// Compute a deterministic BLAKE3 identity for the declared manifest.
    ///
    /// Recording order, annotation order, and individual-ID order are
    /// canonicalized because those vector orders do not change the corpus
    /// meaning. Split assignments, rights evidence, hashes, timing, channel
    /// metadata and all other declared fields remain identity-bearing.
    pub fn computed_manifest_blake3(&self) -> Result<String, String> {
        let mut canonical = self.clone();
        canonical
            .recordings
            .sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
        for recording in &mut canonical.recordings {
            // Hex digests are case-insensitive; canonicalize their spelling so
            // equivalent manifests do not acquire different identities.
            recording.content_sha256 = recording.content_sha256.to_ascii_lowercase();
            recording.individual_ids.sort();
            recording
                .annotations
                .sort_by(|left, right| left.annotation_id.cmp(&right.annotation_id));
            recording.unscored_regions.sort_by(|left, right| {
                left.interval
                    .start_s
                    .total_cmp(&right.interval.start_s)
                    .then_with(|| left.interval.end_s.total_cmp(&right.interval.end_s))
                    .then_with(|| left.reason.cmp(&right.reason))
            });
        }
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| format!("cannot serialize bioacoustic manifest: {error}"))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

/// Claims that an evaluation split policy intends to test.
///
/// Set a holdout flag only when the corresponding IDs are sourced from
/// credible metadata. Unknown identity/site data must remain unknown rather
/// than being fabricated to satisfy a benchmark.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BioacousticSplitPolicy {
    pub require_source_recording_ids: bool,
    pub require_annotations: bool,
    pub hold_out_sessions: bool,
    pub hold_out_sites: bool,
    pub hold_out_individuals: bool,
}

/// Validate a metadata-only corpus before downloading/training or reporting
/// metrics. This checks declared metadata and cross-split relationships, not
/// the referenced bytes, authenticity of labels, or legal correctness of the
/// human-reviewed rights decision.
pub fn validate_bioacoustic_corpus(
    manifest: &BioacousticCorpusManifest,
    policy: BioacousticSplitPolicy,
) -> Result<(), String> {
    if manifest.schema_version != BIOACOUSTIC_CORPUS_SCHEMA_VERSION {
        return Err("unsupported bioacoustic corpus schema version".into());
    }
    if manifest.dataset_id.trim().is_empty()
        || manifest.dataset_revision.trim().is_empty()
        || manifest.source_uri.trim().is_empty()
    {
        return Err("dataset ID, immutable revision, and source URI are required".into());
    }
    if manifest.rights.license_id.trim().is_empty()
        || manifest.rights.license_uri.trim().is_empty()
        || manifest.rights.source_asset_terms_uri.trim().is_empty()
        || manifest.rights.decision_reference.trim().is_empty()
    {
        return Err(
            "rights decision must reference the license and exact source-asset terms".into(),
        );
    }
    if manifest.rights.intended_use != manifest.intended_use {
        return Err("rights evidence does not match the corpus intended use".into());
    }
    match manifest.rights.decision {
        RightsAdmissionDecision::Approved => {}
        RightsAdmissionDecision::PendingReview => {
            return Err("corpus rights admission is pending review".into());
        }
        RightsAdmissionDecision::Denied => {
            return Err("corpus rights admission is denied for the declared use".into());
        }
    }
    if manifest.recordings.is_empty() {
        return Err("corpus manifest must contain at least one recording".into());
    }
    let distinct_splits: BTreeSet<_> =
        manifest.recordings.iter().map(|recording| recording.split).collect();
    if (policy.hold_out_sessions || policy.hold_out_sites || policy.hold_out_individuals)
        && distinct_splits.len() < 2
    {
        return Err("a declared holdout policy requires at least two distinct data splits".into());
    }

    let mut sample_ids = BTreeSet::new();
    let mut content_hashes = BTreeSet::new();

    for recording in &manifest.recordings {
        if recording.sample_id.trim().is_empty()
            || !sample_ids.insert(recording.sample_id.as_str())
        {
            return Err("sample IDs must be non-empty and unique".into());
        }
        if recording.asset_uri.trim().is_empty() {
            return Err(format!("sample {} is missing its asset URI", recording.sample_id));
        }
        if !is_sha256_hex(&recording.content_sha256)
            || !content_hashes.insert(recording.content_sha256.to_ascii_lowercase())
        {
            return Err(format!(
                "sample {} has an invalid or duplicate content SHA-256",
                recording.sample_id
            ));
        }
        if recording.sample_rate_hz < 2
            || recording.channels == 0
            || recording.sample_count == 0
        {
            return Err(format!(
                "sample {} has invalid sample rate, channel count, or sample count",
                recording.sample_id
            ));
        }
        if policy.require_source_recording_ids && recording.source_recording_id.is_none() {
            return Err(format!(
                "sample {} lacks required source-recording lineage",
                recording.sample_id
            ));
        }
        if recording
            .source_recording_id
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
            || recording
                .session_id
                .as_ref()
                .is_some_and(|value| value.trim().is_empty())
            || recording
                .site_id
                .as_ref()
                .is_some_and(|value| value.trim().is_empty())
        {
            return Err(format!("sample {} has a blank provenance ID", recording.sample_id));
        }
        validate_unique_optional_ids(
            &recording.individual_ids,
            "individual",
            &recording.sample_id,
        )?;
        if policy.hold_out_sessions && recording.session_id.is_none() {
            return Err(format!(
                "sample {} lacks session ID required by the holdout policy",
                recording.sample_id
            ));
        }
        if policy.hold_out_sites && recording.site_id.is_none() {
            return Err(format!(
                "sample {} lacks site ID required by the holdout policy",
                recording.sample_id
            ));
        }
        if policy.hold_out_individuals && recording.individual_ids.is_empty() {
            return Err(format!(
                "sample {} lacks individual IDs required by the holdout policy",
                recording.sample_id
            ));
        }
        if policy.require_annotations && recording.annotations.is_empty() {
            return Err(format!(
                "sample {} has no annotations required by the benchmark",
                recording.sample_id
            ));
        }

        let duration_s = recording.sample_count as f64 / recording.sample_rate_hz as f64;
        if !duration_s.is_finite() || duration_s <= 0.0 {
            return Err(format!("sample {} has invalid derived duration", recording.sample_id));
        }
        if recording.source_start_s.is_some() && recording.source_recording_id.is_none() {
            return Err(format!(
                "sample {} declares a source offset without a source-recording ID",
                recording.sample_id
            ));
        }
        if recording
            .source_start_s
            .is_some_and(|offset| !offset.is_finite() || offset < 0.0 || !(offset + duration_s).is_finite())
        {
            return Err(format!(
                "sample {} has an invalid source-recording start offset",
                recording.sample_id
            ));
        }
        let mut previous_unscored_end = None;
        for region in &recording.unscored_regions {
            if !region.interval.start_s.is_finite()
                || !region.interval.end_s.is_finite()
                || region.interval.start_s < 0.0
                || region.interval.end_s <= region.interval.start_s
                || region.interval.end_s > duration_s
                || region.reason.trim().is_empty()
                || region.annotation_source.trim().is_empty()
            {
                return Err(format!(
                    "sample {} has an invalid unscored region",
                    recording.sample_id
                ));
            }
            if previous_unscored_end
                .is_some_and(|previous_end| region.interval.start_s < previous_end)
            {
                return Err(format!(
                    "sample {} has overlapping or unsorted unscored regions",
                    recording.sample_id
                ));
            }
            previous_unscored_end = Some(region.interval.end_s);
        }

        let nyquist_hz = recording.sample_rate_hz as f64 / 2.0;
        let mut annotation_ids = BTreeSet::new();
        for annotation in &recording.annotations {
            if annotation.annotation_id.trim().is_empty()
                || !annotation_ids.insert(annotation.annotation_id.as_str())
                || annotation.label.trim().is_empty()
                || annotation.annotation_source.trim().is_empty()
            {
                return Err(format!(
                    "sample {} has incomplete or duplicate annotation metadata",
                    recording.sample_id
                ));
            }
            if !annotation.interval.start_s.is_finite()
                || !annotation.interval.end_s.is_finite()
                || annotation.interval.start_s < 0.0
                || annotation.interval.end_s <= annotation.interval.start_s
                || annotation.interval.end_s > duration_s
            {
                return Err(format!(
                    "annotation {} falls outside sample {}",
                    annotation.annotation_id, recording.sample_id
                ));
            }
            let first_possible_unscored_region = recording
                .unscored_regions
                .partition_point(|region| {
                    region.interval.end_s <= annotation.interval.start_s
                });
            if recording
                .unscored_regions
                .get(first_possible_unscored_region)
                .is_some_and(|region| region.interval.start_s < annotation.interval.end_s)
            {
                return Err(format!(
                    "annotation {} overlaps an explicitly unscored region in sample {}",
                    annotation.annotation_id, recording.sample_id
                ));
            }

            match (annotation.frequency_min_hz, annotation.frequency_max_hz) {
                (None, None) => {}
                (Some(min_hz), Some(max_hz)) => {
                    if !min_hz.is_finite()
                        || !max_hz.is_finite()
                        || min_hz < 0.0
                        || max_hz <= min_hz
                        || max_hz >= nyquist_hz
                    {
                        return Err(format!(
                            "annotation {} has an invalid frequency band for sample {}",
                            annotation.annotation_id, recording.sample_id
                        ));
                    }
                }
                _ => {
                    return Err(format!(
                        "annotation {} must declare both frequency bounds or neither",
                        annotation.annotation_id
                    ));
                }
            }
        }
    }

    // Track the owning split for each provenance ID instead of comparing every
    // recording pair. Corpora with tens of thousands of windows must remain
    // practical; pairwise leakage checks would otherwise be O(recordings^2).
    let mut source_owners: BTreeMap<&str, (BioacousticSplit, &str)> = BTreeMap::new();
    let mut session_owners: BTreeMap<&str, (BioacousticSplit, &str)> = BTreeMap::new();
    let mut site_owners: BTreeMap<&str, (BioacousticSplit, &str)> = BTreeMap::new();
    let mut individual_owners: BTreeMap<&str, (BioacousticSplit, &str)> = BTreeMap::new();

    for recording in &manifest.recordings {
        if let Some(source_id) = recording.source_recording_id.as_deref() {
            reject_cross_split_owner(
                &mut source_owners,
                source_id,
                recording,
                "source-recording",
            )?;
        }
        if policy.hold_out_sessions {
            if let Some(session_id) = recording.session_id.as_deref() {
                reject_cross_split_owner(
                    &mut session_owners,
                    session_id,
                    recording,
                    "session",
                )?;
            }
        }
        if policy.hold_out_sites {
            if let Some(site_id) = recording.site_id.as_deref() {
                reject_cross_split_owner(&mut site_owners, site_id, recording, "site")?;
            }
        }
        if policy.hold_out_individuals {
            for individual_id in &recording.individual_ids {
                reject_cross_split_owner(
                    &mut individual_owners,
                    individual_id,
                    recording,
                    "individual",
                )?;
            }
        }
    }

    Ok(())
}

fn reject_cross_split_owner<'a>(
    owners: &mut BTreeMap<&'a str, (BioacousticSplit, &'a str)>,
    provenance_id: &'a str,
    recording: &'a BioacousticRecording,
    kind: &str,
) -> Result<(), String> {
    match owners.get(provenance_id) {
        Some((previous_split, previous_sample)) if *previous_split != recording.split => {
            Err(format!(
                "{kind} leakage between samples {previous_sample} and {}",
                recording.sample_id
            ))
        }
        Some(_) => Ok(()),
        None => {
            owners.insert(
                provenance_id,
                (recording.split, recording.sample_id.as_str()),
            );
            Ok(())
        }
    }
}

fn validate_unique_optional_ids(ids: &[String], kind: &str, sample_id: &str) -> Result<(), String> {
    let mut unique = BTreeSet::new();
    if ids
        .iter()
        .any(|id| id.trim().is_empty() || !unique.insert(id.as_str()))
    {
        return Err(format!(
            "sample {sample_id} has blank or duplicate {kind} IDs"
        ));
    }
    Ok(())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeSpan;

    fn annotation(
        id: &str,
        start_s: f64,
        end_s: f64,
        frequency_min_hz: Option<f64>,
        frequency_max_hz: Option<f64>,
    ) -> BioacousticAnnotation {
        BioacousticAnnotation {
            annotation_id: id.into(),
            interval: TimeSpan { start_s, end_s },
            label: "whistle".into(),
            annotation_source: "expert:annotation-v1".into(),
            frequency_min_hz,
            frequency_max_hz,
        }
    }

    fn recording(
        sample_id: &str,
        hash_byte: char,
        source_id: &str,
        split: BioacousticSplit,
        site_id: &str,
        individual_id: &str,
    ) -> BioacousticRecording {
        BioacousticRecording {
            sample_id: sample_id.into(),
            asset_uri: format!("dataset/{sample_id}.flac"),
            content_sha256: std::iter::repeat_n(hash_byte, 64).collect(),
            split,
            sample_rate_hz: 512_000,
            channels: 1,
            sample_count: 512_000,
            source_recording_id: Some(source_id.into()),
            source_start_s: None,
            session_id: Some(format!("session-{source_id}")),
            site_id: Some(site_id.into()),
            individual_ids: vec![individual_id.into()],
            annotations: vec![annotation("ann-1", 0.1, 0.2, Some(1_000.0), Some(150_000.0))],
            unscored_regions: Vec::new(),
        }
    }

    fn manifest() -> BioacousticCorpusManifest {
        BioacousticCorpusManifest {
            schema_version: BIOACOUSTIC_CORPUS_SCHEMA_VERSION,
            dataset_id: "dolphin-pilot".into(),
            dataset_revision: "v1".into(),
            source_uri: "https://example.org/dataset".into(),
            intended_use: BioacousticIntendedUse::NonCommercialResearch,
            rights: BioacousticRightsEvidence {
                license_id: "CC-BY-4.0".into(),
                license_uri: "https://creativecommons.org/licenses/by/4.0/".into(),
                source_asset_terms_uri: "https://example.org/dataset-terms".into(),
                decision_reference: "https://example.org/review/rights-001".into(),
                intended_use: BioacousticIntendedUse::NonCommercialResearch,
                decision: RightsAdmissionDecision::Approved,
            },
            recordings: vec![
                recording(
                    "sample-a",
                    'a',
                    "recording-a",
                    BioacousticSplit::Train,
                    "site-a",
                    "animal-a",
                ),
                recording(
                    "sample-b",
                    'b',
                    "recording-b",
                    BioacousticSplit::Test,
                    "site-b",
                    "animal-b",
                ),
            ],
        }
    }

    #[test]
    fn manifest_hash_is_order_invariant_but_identity_sensitive() {
        let mut left = manifest();
        let left_hash = left.computed_manifest_blake3().unwrap();

        left.recordings.reverse();
        left.recordings[0].individual_ids.reverse();
        let reordered_hash = left.computed_manifest_blake3().unwrap();
        assert_eq!(left_hash, reordered_hash);

        left.intended_use = BioacousticIntendedUse::CommercialResearch;
        left.rights.intended_use = BioacousticIntendedUse::CommercialResearch;
        let changed_rights_scope_hash = left.computed_manifest_blake3().unwrap();
        assert_ne!(left_hash, changed_rights_scope_hash);

        let mut upper_case_digest = manifest();
        upper_case_digest.recordings[0].content_sha256 =
            upper_case_digest.recordings[0].content_sha256.to_ascii_uppercase();
        assert_eq!(
            manifest().computed_manifest_blake3().unwrap(),
            upper_case_digest.computed_manifest_blake3().unwrap()
        );

        let mut offset_changed = manifest();
        offset_changed.recordings[0].source_start_s = Some(12.5);
        assert_ne!(
            manifest().computed_manifest_blake3().unwrap(),
            offset_changed.computed_manifest_blake3().unwrap()
        );

        let mut unscored_changed = manifest();
        unscored_changed.recordings[0].unscored_regions.push(BioacousticUnscoredRegion {
            interval: TimeSpan {
                start_s: 0.3,
                end_s: 0.4,
            },
            reason: "uncertain caller".into(),
                annotation_source: "expert-reviewed-uncertainty-v1".into(),
        });
        assert_ne!(
            manifest().computed_manifest_blake3().unwrap(),
            unscored_changed.computed_manifest_blake3().unwrap()
        );
    }

    #[test]
    fn accepts_complete_metadata_with_disjoint_sources() {
        let corpus = manifest();
        assert!(
            validate_bioacoustic_corpus(
                &corpus,
                BioacousticSplitPolicy {
                    require_source_recording_ids: true,
                    require_annotations: true,
                    hold_out_sessions: true,
                    hold_out_sites: true,
                    hold_out_individuals: true,
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_unresolved_rights_and_use_mismatch() {
        let mut corpus = manifest();
        corpus.rights.decision = RightsAdmissionDecision::PendingReview;
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());

        let mut corpus = manifest();
        corpus.intended_use = BioacousticIntendedUse::CommercialResearch;
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());
    }

    #[test]
    fn holdout_policy_requires_distinct_split_partitions() {
        let mut corpus = manifest();
        corpus.recordings.truncate(1);
        assert!(validate_bioacoustic_corpus(
            &corpus,
            BioacousticSplitPolicy {
                hold_out_sessions: true,
                ..BioacousticSplitPolicy::default()
            },
        ).is_err());
    }

    #[test]
    fn rejects_missing_source_recording_lineage_when_required() {
        let mut corpus = manifest();
        corpus.recordings[0].source_recording_id = None;
        assert!(validate_bioacoustic_corpus(
            &corpus,
            BioacousticSplitPolicy {
                require_source_recording_ids: true,
                ..BioacousticSplitPolicy::default()
            },
        ).is_err());
    }

    #[test]
    fn rejects_source_recording_overlap_across_splits() {
        let mut corpus = manifest();
        corpus.recordings[1].source_recording_id = Some("recording-a".into());
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());
    }

    #[test]
    fn rejects_duplicate_content_hashes_even_when_sample_ids_differ() {
        let mut corpus = manifest();
        corpus.recordings[1].content_sha256 = corpus.recordings[0].content_sha256.clone();
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());
    }

    #[test]
    fn rejects_annotations_outside_recording_duration() {
        let mut corpus = manifest();
        corpus.recordings[0].annotations[0].interval.end_s = 2.0;
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());
    }

    #[test]
    fn rejects_frequency_bounds_at_or_above_source_nyquist() {
        let mut corpus = manifest();
        corpus.recordings[0].annotations[0].frequency_max_hz = Some(256_000.0);
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());
    }

    #[test]
    fn rejects_one_sided_frequency_bounds_and_blank_labels() {
        let mut corpus = manifest();
        corpus.recordings[0].annotations[0].frequency_max_hz = None;
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());

        let mut corpus = manifest();
        corpus.recordings[0].annotations[0].label = " ".into();
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());
    }

    #[test]
    fn accepts_sorted_non_overlapping_unscored_regions() {
        let mut corpus = manifest();
        corpus.recordings[0].unscored_regions = vec![
            BioacousticUnscoredRegion {
                interval: TimeSpan {
                    start_s: 0.3,
                    end_s: 0.4,
                },
                reason: "caller overlap".into(),
                annotation_source: "expert-reviewed-uncertainty-v1".into(),
            },
            BioacousticUnscoredRegion {
                interval: TimeSpan {
                    start_s: 0.5,
                    end_s: 0.6,
                },
                reason: "low signal-to-noise".into(),
                annotation_source: "expert-reviewed-uncertainty-v1".into(),
            },
        ];
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_ok());
    }

    #[test]
    fn rejects_overlapping_or_unsorted_unscored_regions() {
        let mut corpus = manifest();
        corpus.recordings[0].unscored_regions = vec![
            BioacousticUnscoredRegion {
                interval: TimeSpan {
                    start_s: 0.3,
                    end_s: 0.6,
                },
                reason: "uncertain".into(),
                annotation_source: "expert-reviewed-uncertainty-v1".into(),
            },
            BioacousticUnscoredRegion {
                interval: TimeSpan {
                    start_s: 0.5,
                    end_s: 0.7,
                },
                reason: "uncertain".into(),
                annotation_source: "expert-reviewed-uncertainty-v1".into(),
            },
        ];
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());

        let mut corpus = manifest();
        corpus.recordings[0].unscored_regions = vec![
            BioacousticUnscoredRegion {
                interval: TimeSpan {
                    start_s: 0.5,
                    end_s: 0.6,
                },
                reason: "later".into(),
                annotation_source: "expert-reviewed-uncertainty-v1".into(),
            },
            BioacousticUnscoredRegion {
                interval: TimeSpan {
                    start_s: 0.3,
                    end_s: 0.4,
                },
                reason: "earlier".into(),
                annotation_source: "expert-reviewed-uncertainty-v1".into(),
            },
        ];
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());
    }

    #[test]
    fn rejects_unscored_regions_without_annotation_provenance() {
        let mut corpus = manifest();
        corpus.recordings[0].unscored_regions.push(BioacousticUnscoredRegion {
            interval: TimeSpan {
                start_s: 0.3,
                end_s: 0.4,
            },
            reason: "ambiguous caller".into(),
            annotation_source: " ".into(),
        });
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());
    }

    #[test]
    fn rejects_positive_annotations_overlapping_unscored_regions() {
        let mut corpus = manifest();
        corpus.recordings[0].unscored_regions.push(BioacousticUnscoredRegion {
            interval: TimeSpan {
                start_s: 0.15,
                end_s: 0.25,
            },
            reason: "annotation ambiguous".into(),
                annotation_source: "expert-reviewed-uncertainty-v1".into(),
        });
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());
    }

    #[test]
    fn schema_v3_requires_explicit_crop_offset_and_unscored_regions_keys() {
        let corpus = manifest();
        let mut json = serde_json::to_value(corpus).unwrap();
        let recording = json
            .get_mut("recordings")
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|recordings| recordings.first_mut())
            .unwrap();
        recording.as_object_mut().unwrap().remove("source_start_s");
        assert!(serde_json::from_value::<BioacousticCorpusManifest>(json.clone()).is_err());

        let recording = json
            .get_mut("recordings")
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|recordings| recordings.first_mut())
            .unwrap();
        recording
            .as_object_mut()
            .unwrap()
            .insert("source_start_s".into(), serde_json::Value::Null);
        recording.as_object_mut().unwrap().remove("unscored_regions");
        assert!(serde_json::from_value::<BioacousticCorpusManifest>(json).is_err());
    }

    #[test]
    fn rejects_source_offsets_without_source_recording_ids() {
        let mut corpus = manifest();
        corpus.recordings[0].source_recording_id = None;
        corpus.recordings[0].source_start_s = Some(3.0);
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());
    }

    #[test]
    fn rejects_invalid_source_recording_offsets() {
        for offset in [-0.1, f64::NAN, f64::INFINITY, f64::MAX] {
            let mut corpus = manifest();
            corpus.recordings[0].source_start_s = Some(offset);
            assert!(
                validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err(),
                "invalid source offset must be rejected: {offset}"
            );
        }
    }

    #[test]
    fn rejects_duplicates_or_invalid_sample_metadata() {
        let mut corpus = manifest();
        corpus.recordings.push(corpus.recordings[0].clone());
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());

        let mut corpus = manifest();
        corpus.recordings[0].sample_rate_hz = 0;
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());

        let mut corpus = manifest();
        corpus.recordings[0].content_sha256 = "not-a-hash".into();
        assert!(validate_bioacoustic_corpus(&corpus, BioacousticSplitPolicy::default()).is_err());
    }

    #[test]
    fn validates_ten_thousand_recordings_with_disjoint_holdout_metadata() {
        let mut corpus = manifest();
        corpus.recordings.clear();
        for index in 0..10_000_usize {
            let sample_id = format!("sample-{index}");
            let source_id = format!("source-{index}");
            let split = if index % 2 == 0 {
                BioacousticSplit::Train
            } else {
                BioacousticSplit::Test
            };
            let mut item = recording(
                &sample_id,
                'a',
                &source_id,
                split,
                &format!("site-{index}"),
                &format!("animal-{index}"),
            );
            item.content_sha256 = format!("{:064x}", index + 1);
            item.session_id = Some(format!("session-{index}"));
            item.annotations.clear();
            corpus.recordings.push(item);
        }

        assert!(
            validate_bioacoustic_corpus(
                &corpus,
                BioacousticSplitPolicy {
                    require_source_recording_ids: true,
                    hold_out_sessions: true,
                    hold_out_sites: true,
                    hold_out_individuals: true,
                    ..BioacousticSplitPolicy::default()
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn requires_annotations_only_for_tasks_that_need_them() {
        let mut corpus = manifest();
        corpus.recordings[0].annotations.clear();
        assert!(
            validate_bioacoustic_corpus(
                &corpus,
                BioacousticSplitPolicy {
                    require_annotations: true,
                    ..BioacousticSplitPolicy::default()
                },
            )
            .is_err()
        );
    }
}
