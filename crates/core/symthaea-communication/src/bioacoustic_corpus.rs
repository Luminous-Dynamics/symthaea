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
use std::collections::BTreeSet;

pub const BIOACOUSTIC_CORPUS_SCHEMA_VERSION: u32 = 1;

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
    pub interval: crate::TimeSpan,
    pub label: String,
    pub annotation_source: String,
    pub frequency_min_hz: Option<f64>,
    pub frequency_max_hz: Option<f64>,
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
    pub session_id: Option<String>,
    pub site_id: Option<String>,
    pub individual_ids: Vec<String>,
    pub annotations: Vec<BioacousticAnnotation>,
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

    for (index, left) in manifest.recordings.iter().enumerate() {
        for right in &manifest.recordings[index + 1..] {
            if left.split == right.split {
                continue;
            }
            if same_optional_id(&left.source_recording_id, &right.source_recording_id) {
                return Err(format!(
                    "source-recording leakage between samples {} and {}",
                    left.sample_id, right.sample_id
                ));
            }
            if policy.hold_out_sessions && same_optional_id(&left.session_id, &right.session_id) {
                return Err(format!(
                    "session leakage between samples {} and {}",
                    left.sample_id, right.sample_id
                ));
            }
            if policy.hold_out_sites && same_optional_id(&left.site_id, &right.site_id) {
                return Err(format!(
                    "site leakage between samples {} and {}",
                    left.sample_id, right.sample_id
                ));
            }
            if policy.hold_out_individuals
                && left
                    .individual_ids
                    .iter()
                    .any(|id| right.individual_ids.iter().any(|other| other == id))
            {
                return Err(format!(
                    "individual leakage between samples {} and {}",
                    left.sample_id, right.sample_id
                ));
            }
        }
    }

    Ok(())
}

fn same_optional_id(left: &Option<String>, right: &Option<String>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right,
        _ => false,
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
            session_id: Some(format!("session-{source_id}")),
            site_id: Some(site_id.into()),
            individual_ids: vec![individual_id.into()],
            annotations: vec![annotation("ann-1", 0.1, 0.2, Some(1_000.0), Some(150_000.0))],
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
