// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed admission and split-integrity checks for speech evaluation cohorts.
//!
//! This module validates provenance, rights authorization, independent review,
//! pair membership, and source-group disjointness before any model is evaluated.
//! It does not download audio, infer speaker identity, establish legal rights,
//! or prove that a source annotation is correct.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Current serialized cohort manifest schema revision.
pub const COHORT_MANIFEST_SCHEMA_VERSION: &str = "0.1.0";

/// Declared evidence stage of a cohort. This is metadata, not independent proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CohortEvidenceState {
    /// Synthetic fixtures only; never report as real-audio performance.
    SyntheticOnly,
    /// Real-audio pilot; exploratory and not confirmatory.
    Pilot,
    /// Confirmatory real-audio cohort, subject to all admission and review gates.
    Confirmatory,
}

/// Data split for one pseudonymous utterance record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CohortSplit {
    Training,
    Development,
    Pilot,
    Confirmatory,
    PreferenceTraining,
}

/// Explicit consent/authority state recorded by the data steward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ConsentStatus {
    Confirmed,
    DocumentedNotRequired,
    Unverified,
    Withdrawn,
}

/// Redistribution is separate from permission to evaluate a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RedistributionPolicy {
    Prohibited,
    ControlledAccess,
    Public,
}

/// Evaluation role for a paired counterfactual unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PairPurpose {
    PreferenceTraining,
    DevelopmentEvaluation,
    PilotEvaluation,
    ConfirmatoryEvaluation,
}

impl PairPurpose {
    fn required_split(self) -> CohortSplit {
        match self {
            Self::PreferenceTraining => CohortSplit::PreferenceTraining,
            Self::DevelopmentEvaluation => CohortSplit::Development,
            Self::PilotEvaluation => CohortSplit::Pilot,
            Self::ConfirmatoryEvaluation => CohortSplit::Confirmatory,
        }
    }
}

/// Rights and processing authority documented for one source record.
///
/// These booleans are steward assertions backed by the referenced license,
/// consent, or permission record. They are not legal conclusions inferred
/// from a public URL or a dataset's ability to be downloaded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RightsAuthorization {
    /// Stable license or agreement identifier; use an explicit identifier for bespoke permission.
    pub license_id: String,
    /// Human-readable purpose scope supported by the agreement.
    pub authorized_purpose: String,
    /// Whether the exact proposed benchmark evaluation is authorized.
    pub benchmark_evaluation_authorized: bool,
    /// Whether the proposed derivative audio processing is authorized.
    pub derivative_processing_authorized: bool,
    /// Whether audio may be sent to an external model/API processor.
    pub external_processing_authorized: bool,
    /// Documented consent status.
    pub consent_status: ConsentStatus,
    /// Whether redistribution is prohibited, controlled, or public.
    pub redistribution: RedistributionPolicy,
}

/// One pseudonymous audio item and its provenance/adjudication lineage.
///
/// Cluster IDs must be stable within the cohort but must not contain direct
/// identifiers or personal data. Use opaque, steward-generated IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortRecord {
    /// Stable unique record ID.
    pub record_id: String,
    /// Pair membership, when this record is one side of a registered pair.
    pub pair_id: Option<String>,
    /// Underlying corpus/source-family cluster (not a URL).
    pub source_family_id: String,
    /// Pseudonymous speaker cluster, when known.
    pub speaker_cluster_id: Option<String>,
    /// Pseudonymous recording-session cluster, when known.
    pub session_cluster_id: Option<String>,
    /// Pseudonymous utterance/near-duplicate family cluster.
    pub utterance_family_id: String,
    /// Split membership fixed before adaptation or evaluation.
    pub split: CohortSplit,
    /// Language/variety tags; mixed speech may contain multiple entries.
    pub language_varieties: Vec<String>,
    /// SHA-256 of the original admitted source audio.
    pub source_audio_sha256: String,
    /// SHA-256 of the exact waveform supplied to the evaluated system.
    pub evaluation_audio_sha256: String,
    /// SHA-256 of the source/rights manifest for this record.
    pub source_manifest_sha256: String,
    /// SHA-256 of the transcript, annotation, and adjudication package manifest.
    pub annotation_manifest_sha256: String,
    /// SHA-256 of the pinned derivation manifest when evaluation audio differs.
    pub derivation_manifest_sha256: Option<String>,
    /// Pseudonymous independent reviewer IDs; never personal names or emails.
    pub reviewer_ids: Vec<String>,
    /// SHA-256 of adjudication evidence when confirmatory evaluation is claimed.
    pub adjudication_manifest_sha256: Option<String>,
    /// Optional versioned phonological profile identifier.
    pub phonological_profile_id: Option<String>,
    /// Optional immutable phonological profile revision.
    pub phonological_profile_revision: Option<String>,
    /// True only after the profile has passed its independent native-review gate.
    pub phonological_profile_qualified: bool,
    /// SHA-256 of independent native-speaker/linguist review when a profile is qualified.
    pub native_review_manifest_sha256: Option<String>,
    /// Rights and processing authority for this exact record.
    pub rights: RightsAuthorization,
}

/// A pair binds two records to one exclusive use and split.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortPair {
    /// Stable unique pair ID.
    pub pair_id: String,
    /// Baseline record.
    pub baseline_record_id: String,
    /// Variant record.
    pub variant_record_id: String,
    /// SHA-256 of the exact #11157 counterfactual pair manifest.
    pub counterfactual_pair_manifest_sha256: String,
    /// Split containing both records.
    pub split: CohortSplit,
    /// Whether this pair is for preference training or evaluation.
    pub purpose: PairPurpose,
}

/// Frozen cohort admission manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortManifest {
    /// Schema revision.
    pub schema_version: String,
    /// Stable cohort identifier.
    pub cohort_id: String,
    /// Evidence stage declared by the data steward; not proof of real-audio validity.
    pub evidence_state: CohortEvidenceState,
    /// Records in the frozen cohort.
    pub records: Vec<CohortRecord>,
    /// Pair definitions. Pair IDs may occur only once across all purposes.
    pub pairs: Vec<CohortPair>,
}

/// Summary returned only after all admission checks pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CohortValidationSummary {
    pub record_count: usize,
    pub pair_count: usize,
    pub preference_pair_count: usize,
    pub development_pair_count: usize,
    pub pilot_pair_count: usize,
    pub confirmatory_pair_count: usize,
    pub unique_source_family_count: usize,
    pub all_source_groups_split_disjoint: bool,
    pub preference_evaluation_pair_overlap: usize,
}

/// Typed fail-closed cohort admission errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CohortAdmissionError {
    UnsupportedSchemaVersion(String),
    EmptyCohortId,
    EmptyRecords,
    EmptyPairs,
    EmptyIdentifier(&'static str),
    DuplicateRecordId(String),
    DuplicatePairId(String),
    EvidenceStageRequiresPairType(CohortEvidenceState),
    DuplicateLanguageTag(String),
    InvalidDigest(&'static str, String),
    MissingDerivedAudioManifest(String),
    DerivativeProcessingNotAuthorized(String),
    BenchmarkEvaluationNotAuthorized(String),
    ConsentNotEstablished(String),
    InsufficientIndependentReview(String),
    MissingAdjudicationManifest(String),
    IncompletePhonologicalProfile(String),
    MissingNativeReview(String),
    MissingSpeakerCluster(String),
    MissingSessionCluster(String),
    SplitLeakage {
        dimension: &'static str,
        cluster_id: String,
        first_split: CohortSplit,
        second_split: CohortSplit,
    },
    AudioDigestAcrossSplits {
        digest: String,
        first_split: CohortSplit,
        second_split: CohortSplit,
    },
    MissingPairRecord(String),
    SamePairRecordReference(String),
    PairRecordSplitMismatch(String),
    PairPurposeSplitMismatch(String),
    RecordReusedAcrossPairs(String),
    OrphanPairRecord(String),
    ExternalProcessingNotAuthorized(String),
    SerializationFailure(String),
}

impl fmt::Display for CohortAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CohortAdmissionError {}

/// Validate a cohort before any training, preference generation, or evaluation.
///
/// The validator fails closed on unknown/withdrawn consent, missing benchmark
/// authorization, split leakage, duplicate pair use, and incomplete review.
/// Public redistribution is deliberately not required for controlled-access
/// evaluation; it is separately represented in each record's rights contract.
pub fn validate_cohort_manifest(
    manifest: &CohortManifest,
) -> Result<CohortValidationSummary, CohortAdmissionError> {
    if manifest.schema_version != COHORT_MANIFEST_SCHEMA_VERSION {
        return Err(CohortAdmissionError::UnsupportedSchemaVersion(
            manifest.schema_version.clone(),
        ));
    }
    require_identifier(&manifest.cohort_id, "cohort_id")?;
    if manifest.records.is_empty() {
        return Err(CohortAdmissionError::EmptyRecords);
    }
    if manifest.pairs.is_empty() {
        return Err(CohortAdmissionError::EmptyPairs);
    }

    let mut record_index: HashMap<&str, &CohortRecord> =
        HashMap::with_capacity(manifest.records.len());
    let mut group_splits: HashMap<(&'static str, String), CohortSplit> = HashMap::new();
    let mut audio_splits: HashMap<String, CohortSplit> = HashMap::new();
    let mut source_families = HashSet::new();

    for record in &manifest.records {
        require_identifier(&record.record_id, "record_id")?;
        require_identifier(&record.source_family_id, "source_family_id")?;
        require_identifier(&record.utterance_family_id, "utterance_family_id")?;
        if let Some(pair_id) = record.pair_id.as_deref() {
            require_identifier(pair_id, "pair_id")?;
        }
        if record_index.insert(&record.record_id, record).is_some() {
            return Err(CohortAdmissionError::DuplicateRecordId(
                record.record_id.clone(),
            ));
        }
        source_families.insert(record.source_family_id.as_str());

        if record.language_varieties.is_empty() {
            return Err(CohortAdmissionError::EmptyIdentifier("language_varieties"));
        }
        let mut language_tags = HashSet::with_capacity(record.language_varieties.len());
        for tag in &record.language_varieties {
            require_identifier(tag, "language_variety")?;
            if !language_tags.insert(tag.as_str()) {
                return Err(CohortAdmissionError::DuplicateLanguageTag(tag.clone()));
            }
        }

        check_digest("source_audio_sha256", &record.source_audio_sha256)?;
        check_digest(
            "evaluation_audio_sha256",
            &record.evaluation_audio_sha256,
        )?;
        check_digest(
            "source_manifest_sha256",
            &record.source_manifest_sha256,
        )?;
        check_digest(
            "annotation_manifest_sha256",
            &record.annotation_manifest_sha256,
        )?;

        if !record.source_audio_sha256.eq_ignore_ascii_case(&record.evaluation_audio_sha256) {
            let derivation = record
                .derivation_manifest_sha256
                .as_deref()
                .ok_or_else(|| {
                    CohortAdmissionError::MissingDerivedAudioManifest(record.record_id.clone())
                })?;
            check_digest("derivation_manifest_sha256", derivation)?;
            if !record.rights.derivative_processing_authorized {
                return Err(CohortAdmissionError::DerivativeProcessingNotAuthorized(
                    record.record_id.clone(),
                ));
            }
        } else if let Some(derivation) = record.derivation_manifest_sha256.as_deref() {
            check_digest("derivation_manifest_sha256", derivation)?;
        }

        require_identifier(&record.rights.license_id, "license_id")?;
        require_identifier(
            &record.rights.authorized_purpose,
            "authorized_purpose",
        )?;
        if !record.rights.benchmark_evaluation_authorized {
            return Err(CohortAdmissionError::BenchmarkEvaluationNotAuthorized(
                record.record_id.clone(),
            ));
        }
        if !matches!(
            record.rights.consent_status,
            ConsentStatus::Confirmed | ConsentStatus::DocumentedNotRequired
        ) {
            return Err(CohortAdmissionError::ConsentNotEstablished(
                record.record_id.clone(),
            ));
        }

        let mut reviewers = HashSet::with_capacity(record.reviewer_ids.len());
        for reviewer in &record.reviewer_ids {
            require_identifier(reviewer, "reviewer_id")?;
            reviewers.insert(reviewer.as_str());
        }
        if record.split == CohortSplit::Confirmatory {
            if reviewers.len() < 2 {
                return Err(CohortAdmissionError::InsufficientIndependentReview(
                    record.record_id.clone(),
                ));
            }
            let adjudication = record
                .adjudication_manifest_sha256
                .as_deref()
                .ok_or_else(|| {
                    CohortAdmissionError::MissingAdjudicationManifest(record.record_id.clone())
                })?;
            check_digest("adjudication_manifest_sha256", adjudication)?;
            require_identifier(
                record
                    .speaker_cluster_id
                    .as_deref()
                    .ok_or_else(|| {
                        CohortAdmissionError::MissingSpeakerCluster(record.record_id.clone())
                    })?,
                "speaker_cluster_id",
            )?;
            require_identifier(
                record
                    .session_cluster_id
                    .as_deref()
                    .ok_or_else(|| {
                        CohortAdmissionError::MissingSessionCluster(record.record_id.clone())
                    })?,
                "session_cluster_id",
            )?;
        } else if let Some(adjudication) = record.adjudication_manifest_sha256.as_deref() {
            check_digest("adjudication_manifest_sha256", adjudication)?;
        }

        let any_profile_field = record.phonological_profile_id.is_some()
            || record.phonological_profile_revision.is_some()
            || record.native_review_manifest_sha256.is_some()
            || record.phonological_profile_qualified;
        if any_profile_field {
            let profile_id = record
                .phonological_profile_id
                .as_deref()
                .ok_or_else(|| {
                    CohortAdmissionError::IncompletePhonologicalProfile(record.record_id.clone())
                })?;
            let profile_revision = record
                .phonological_profile_revision
                .as_deref()
                .ok_or_else(|| {
                    CohortAdmissionError::IncompletePhonologicalProfile(record.record_id.clone())
                })?;
            require_identifier(profile_id, "phonological_profile_id")?;
            require_identifier(profile_revision, "phonological_profile_revision")?;
            if record.phonological_profile_qualified {
                let native_review = record
                    .native_review_manifest_sha256
                    .as_deref()
                    .ok_or_else(|| {
                        CohortAdmissionError::MissingNativeReview(record.record_id.clone())
                    })?;
                check_digest("native_review_manifest_sha256", native_review)?;
                if reviewers.len() < 2 {
                    return Err(CohortAdmissionError::InsufficientIndependentReview(
                        record.record_id.clone(),
                    ));
                }
            } else if let Some(native_review) =
                record.native_review_manifest_sha256.as_deref()
            {
                check_digest("native_review_manifest_sha256", native_review)?;
            }
        }

        insert_split_group(
            &mut group_splits,
            "source_family",
            &record.source_family_id,
            record.split,
        )?;
        insert_split_group(
            &mut group_splits,
            "utterance_family",
            &record.utterance_family_id,
            record.split,
        )?;
        if let Some(speaker) = record.speaker_cluster_id.as_deref() {
            require_identifier(speaker, "speaker_cluster_id")?;
            insert_split_group(&mut group_splits, "speaker", speaker, record.split)?;
        }
        if let Some(session) = record.session_cluster_id.as_deref() {
            require_identifier(session, "session_cluster_id")?;
            insert_split_group(&mut group_splits, "session", session, record.split)?;
        }

        for digest in [
            record.source_audio_sha256.as_str(),
            record.evaluation_audio_sha256.as_str(),
        ] {
            if let Some(first_split) = audio_splits.insert(digest.to_ascii_lowercase(), record.split)
            {
                if first_split != record.split {
                    return Err(CohortAdmissionError::AudioDigestAcrossSplits {
                        digest: digest.to_ascii_lowercase(),
                        first_split,
                        second_split: record.split,
                    });
                }
            }
        }
    }

    let mut pair_ids = HashSet::with_capacity(manifest.pairs.len());
    let mut used_records = HashSet::with_capacity(manifest.pairs.len() * 2);
    let mut preference_pair_count = 0;
    let mut development_pair_count = 0;
    let mut pilot_pair_count = 0;
    let mut confirmatory_pair_count = 0;

    for pair in &manifest.pairs {
        require_identifier(&pair.pair_id, "pair_id")?;
        if !pair_ids.insert(pair.pair_id.as_str()) {
            return Err(CohortAdmissionError::DuplicatePairId(
                pair.pair_id.clone(),
            ));
        }
        if pair.baseline_record_id == pair.variant_record_id {
            return Err(CohortAdmissionError::SamePairRecordReference(
                pair.pair_id.clone(),
            ));
        }
        check_digest(
            "counterfactual_pair_manifest_sha256",
            &pair.counterfactual_pair_manifest_sha256,
        )?;
        let expected_split = pair.purpose.required_split();
        if pair.split != expected_split {
            return Err(CohortAdmissionError::PairPurposeSplitMismatch(
                pair.pair_id.clone(),
            ));
        }

        let baseline = record_index
            .get(pair.baseline_record_id.as_str())
            .ok_or_else(|| {
                CohortAdmissionError::MissingPairRecord(pair.baseline_record_id.clone())
            })?;
        let variant = record_index
            .get(pair.variant_record_id.as_str())
            .ok_or_else(|| {
                CohortAdmissionError::MissingPairRecord(pair.variant_record_id.clone())
            })?;

        if baseline.split != pair.split || variant.split != pair.split {
            return Err(CohortAdmissionError::PairRecordSplitMismatch(
                pair.pair_id.clone(),
            ));
        }
        if !used_records.insert(pair.baseline_record_id.as_str()) {
            return Err(CohortAdmissionError::RecordReusedAcrossPairs(
                pair.baseline_record_id.clone(),
            ));
        }
        if !used_records.insert(pair.variant_record_id.as_str()) {
            return Err(CohortAdmissionError::RecordReusedAcrossPairs(
                pair.variant_record_id.clone(),
            ));
        }
        if baseline.pair_id.as_deref() != Some(pair.pair_id.as_str())
            || variant.pair_id.as_deref() != Some(pair.pair_id.as_str())
        {
            return Err(CohortAdmissionError::OrphanPairRecord(
                pair.pair_id.clone(),
            ));
        }

        match pair.purpose {
            PairPurpose::PreferenceTraining => preference_pair_count += 1,
            PairPurpose::DevelopmentEvaluation => development_pair_count += 1,
            PairPurpose::PilotEvaluation => pilot_pair_count += 1,
            PairPurpose::ConfirmatoryEvaluation => confirmatory_pair_count += 1,
        }
    }

    for record in &manifest.records {
        if let Some(pair_id) = record.pair_id.as_deref() {
            if !pair_ids.contains(pair_id) || !used_records.contains(record.record_id.as_str()) {
                return Err(CohortAdmissionError::OrphanPairRecord(
                    record.record_id.clone(),
                ));
            }
        }
    }

    match manifest.evidence_state {
        CohortEvidenceState::SyntheticOnly => {}
        CohortEvidenceState::Pilot if pilot_pair_count == 0 => {
            return Err(CohortAdmissionError::EvidenceStageRequiresPairType(
                CohortEvidenceState::Pilot,
            ));
        }
        CohortEvidenceState::Confirmatory if confirmatory_pair_count == 0 => {
            return Err(CohortAdmissionError::EvidenceStageRequiresPairType(
                CohortEvidenceState::Confirmatory,
            ));
        }
        _ => {}
    }

    Ok(CohortValidationSummary {
        record_count: manifest.records.len(),
        pair_count: manifest.pairs.len(),
        preference_pair_count,
        development_pair_count,
        pilot_pair_count,
        confirmatory_pair_count,
        unique_source_family_count: source_families.len(),
        all_source_groups_split_disjoint: true,
        preference_evaluation_pair_overlap: 0,
    })
}

/// Reject use with a hosted/external model unless that record permits it.
///
/// This gate is deliberately independent of benchmark permission: consent to
/// evaluate locally does not imply permission to upload audio to a provider.
pub fn require_external_processing_authorization(
    record: &CohortRecord,
) -> Result<(), CohortAdmissionError> {
    if !record.rights.external_processing_authorized {
        return Err(CohortAdmissionError::ExternalProcessingNotAuthorized(
            record.record_id.clone(),
        ));
    }
    Ok(())
}

/// Compute a deterministic BLAKE3 digest of the serialized cohort manifest.
///
/// Record and pair input order does not affect the digest. Struct field order
/// and schema version do; changing the manifest contract therefore changes
/// the digest intentionally.
pub fn canonical_cohort_manifest_digest(
    manifest: &CohortManifest,
) -> Result<String, CohortAdmissionError> {
    let mut canonical = manifest.clone();
    canonical.records.sort_by(|a, b| a.record_id.cmp(&b.record_id));
    canonical.pairs.sort_by(|a, b| a.pair_id.cmp(&b.pair_id));
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| CohortAdmissionError::SerializationFailure(error.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn require_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), CohortAdmissionError> {
    if value.trim().is_empty() {
        Err(CohortAdmissionError::EmptyIdentifier(field))
    } else {
        Ok(())
    }
}

fn check_digest(
    field: &'static str,
    value: &str,
) -> Result<(), CohortAdmissionError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CohortAdmissionError::InvalidDigest(
            field,
            value.to_string(),
        ));
    }
    Ok(())
}

fn insert_split_group(
    groups: &mut HashMap<(&'static str, String), CohortSplit>,
    dimension: &'static str,
    cluster_id: &str,
    split: CohortSplit,
) -> Result<(), CohortAdmissionError> {
    let key = (dimension, cluster_id.to_string());
    if let Some(first_split) = groups.insert(key, split) {
        if first_split != split {
            return Err(CohortAdmissionError::SplitLeakage {
                dimension,
                cluster_id: cluster_id.to_string(),
                first_split,
                second_split: split,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn rights() -> RightsAuthorization {
        RightsAuthorization {
            license_id: "SYNTHETIC-ORIGINAL".into(),
            authorized_purpose: "synthetic conformance only".into(),
            benchmark_evaluation_authorized: true,
            derivative_processing_authorized: true,
            external_processing_authorized: false,
            consent_status: ConsentStatus::DocumentedNotRequired,
            redistribution: RedistributionPolicy::Public,
        }
    }

    fn record(
        id: &str,
        pair_id: &str,
        family: &str,
        speaker: &str,
        session: &str,
        utterance: &str,
        split: CohortSplit,
        source_audio: &str,
        eval_audio: &str,
    ) -> CohortRecord {
        CohortRecord {
            record_id: id.into(),
            pair_id: Some(pair_id.into()),
            source_family_id: family.into(),
            speaker_cluster_id: Some(speaker.into()),
            session_cluster_id: Some(session.into()),
            utterance_family_id: utterance.into(),
            split,
            language_varieties: vec!["zu-ZA".into(), "en-ZA".into()],
            source_audio_sha256: source_audio.into(),
            evaluation_audio_sha256: eval_audio.into(),
            source_manifest_sha256: C.into(),
            annotation_manifest_sha256: D.into(),
            derivation_manifest_sha256: None,
            reviewer_ids: vec!["reviewer-a".into(), "reviewer-b".into()],
            adjudication_manifest_sha256: None,
            phonological_profile_id: None,
            phonological_profile_revision: None,
            phonological_profile_qualified: false,
            native_review_manifest_sha256: None,
            rights: rights(),
        }
    }

    fn fixture() -> CohortManifest {
        CohortManifest {
            schema_version: COHORT_MANIFEST_SCHEMA_VERSION.into(),
            cohort_id: "synthetic-cohort-v0".into(),
            evidence_state: CohortEvidenceState::SyntheticOnly,
            records: vec![
                record("pilot-a", "pilot-pair", "source-pilot", "speaker-pilot", "session-pilot",
                    "utterance-pilot", CohortSplit::Pilot, A, A),
                record("pilot-b", "pilot-pair", "source-pilot", "speaker-pilot", "session-pilot",
                    "utterance-pilot", CohortSplit::Pilot, A, A),
                record("pref-a", "pref-pair", "source-pref", "speaker-pref", "session-pref",
                    "utterance-pref", CohortSplit::PreferenceTraining, B, B),
                record("pref-b", "pref-pair", "source-pref", "speaker-pref", "session-pref",
                    "utterance-pref", CohortSplit::PreferenceTraining, B, B),
            ],
            pairs: vec![
                CohortPair {
                    pair_id: "pilot-pair".into(),
                    baseline_record_id: "pilot-a".into(),
                    variant_record_id: "pilot-b".into(),
                    counterfactual_pair_manifest_sha256: A.into(),
                    split: CohortSplit::Pilot,
                    purpose: PairPurpose::PilotEvaluation,
                },
                CohortPair {
                    pair_id: "pref-pair".into(),
                    baseline_record_id: "pref-a".into(),
                    variant_record_id: "pref-b".into(),
                    counterfactual_pair_manifest_sha256: A.into(),
                    split: CohortSplit::PreferenceTraining,
                    purpose: PairPurpose::PreferenceTraining,
                },
            ],
        }
    }

    #[test]
    fn evidence_stage_must_have_matching_pair_type() {
        let mut manifest = fixture();
        manifest.evidence_state = CohortEvidenceState::Confirmatory;
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::EvidenceStageRequiresPairType(
                CohortEvidenceState::Confirmatory
            ))
        ));
    }

    #[test]
    fn accepts_disjoint_synthetic_pilot_and_preference_pairs() {
        let result = validate_cohort_manifest(&fixture()).unwrap();
        assert_eq!(result.record_count, 4);
        assert_eq!(result.pair_count, 2);
        assert_eq!(result.preference_pair_count, 1);
        assert_eq!(result.pilot_pair_count, 1);
        assert_eq!(result.preference_evaluation_pair_overlap, 0);
        assert!(result.all_source_groups_split_disjoint);
    }

    #[test]
    fn rejects_source_family_leakage_across_splits() {
        let mut manifest = fixture();
        manifest.records[2].source_family_id = "source-pilot".into();
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::SplitLeakage { dimension: "source_family", .. })
        ));
    }

    #[test]
    fn rejects_speaker_leakage_across_splits() {
        let mut manifest = fixture();
        manifest.records[2].speaker_cluster_id = Some("speaker-pilot".into());
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::SplitLeakage { dimension: "speaker", .. })
        ));
    }

    #[test]
    fn rejects_session_leakage_across_splits() {
        let mut manifest = fixture();
        manifest.records[2].session_cluster_id = Some("session-pilot".into());
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::SplitLeakage { dimension: "session", .. })
        ));
    }

    #[test]
    fn rejects_utterance_family_leakage_across_splits() {
        let mut manifest = fixture();
        manifest.records[2].utterance_family_id = "utterance-pilot".into();
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::SplitLeakage { dimension: "utterance_family", .. })
        ));
    }

    #[test]
    fn rejects_same_audio_digest_across_splits() {
        let mut manifest = fixture();
        manifest.records[2].source_audio_sha256 = A.into();
        manifest.records[2].evaluation_audio_sha256 = A.into();
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::AudioDigestAcrossSplits { .. })
        ));
    }

    #[test]
    fn rejects_unauthorized_benchmark_evaluation() {
        let mut manifest = fixture();
        manifest.records[0].rights.benchmark_evaluation_authorized = false;
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::BenchmarkEvaluationNotAuthorized(_))
        ));
    }

    #[test]
    fn rejects_unverified_or_withdrawn_consent() {
        let mut manifest = fixture();
        manifest.records[0].rights.consent_status = ConsentStatus::Unverified;
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::ConsentNotEstablished(_))
        ));
    }

    #[test]
    fn rejects_derived_audio_without_transform_manifest() {
        let mut manifest = fixture();
        manifest.records[0].evaluation_audio_sha256 = C.into();
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::MissingDerivedAudioManifest(_))
        ));
    }

    #[test]
    fn rejects_derivation_when_not_authorized() {
        let mut manifest = fixture();
        manifest.records[0].evaluation_audio_sha256 = C.into();
        manifest.records[0].derivation_manifest_sha256 = Some(D.into());
        manifest.records[0].rights.derivative_processing_authorized = false;
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::DerivativeProcessingNotAuthorized(_))
        ));
    }

    #[test]
    fn rejects_confirmatory_record_without_two_reviewers() {
        let mut manifest = fixture();
        let mut a = record("confirm-a", "confirm-pair", "source-confirm", "speaker-confirm",
            "session-confirm", "utterance-confirm", CohortSplit::Confirmatory, C, C);
        let mut b = record("confirm-b", "confirm-pair", "source-confirm", "speaker-confirm",
            "session-confirm", "utterance-confirm", CohortSplit::Confirmatory, C, C);
        a.reviewer_ids = vec!["reviewer-a".into(), "reviewer-a".into()];
        b.adjudication_manifest_sha256 = Some(D.into());
        a.adjudication_manifest_sha256 = Some(D.into());
        manifest.records.extend([a, b]);
        manifest.pairs.push(CohortPair {
            pair_id: "confirm-pair".into(),
            baseline_record_id: "confirm-a".into(),
            variant_record_id: "confirm-b".into(),
            counterfactual_pair_manifest_sha256: A.into(),
            split: CohortSplit::Confirmatory,
            purpose: PairPurpose::ConfirmatoryEvaluation,
        });
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::InsufficientIndependentReview(_))
        ));
    }

    #[test]
    fn rejects_confirmatory_record_without_adjudication_manifest() {
        let mut manifest = fixture();
        for (id, speaker, session) in [
            ("confirm-a", "speaker-confirm-a", "session-confirm-a"),
            ("confirm-b", "speaker-confirm-b", "session-confirm-b"),
        ] {
            manifest.records.push(record(id, "confirm-pair", "source-confirm", speaker,
                session, "utterance-confirm", CohortSplit::Confirmatory, C, C));
        }
        manifest.pairs.push(CohortPair {
            pair_id: "confirm-pair".into(),
            baseline_record_id: "confirm-a".into(),
            variant_record_id: "confirm-b".into(),
            counterfactual_pair_manifest_sha256: A.into(),
            split: CohortSplit::Confirmatory,
            purpose: PairPurpose::ConfirmatoryEvaluation,
        });
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::MissingAdjudicationManifest(_))
        ));
    }

    #[test]
    fn rejects_pair_whose_purpose_disagrees_with_split() {
        let mut manifest = fixture();
        manifest.pairs[0].purpose = PairPurpose::ConfirmatoryEvaluation;
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::PairPurposeSplitMismatch(_))
        ));
    }

    #[test]
    fn rejects_cross_split_pair_members() {
        let mut manifest = fixture();
        manifest.records[1].split = CohortSplit::Development;
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::SplitLeakage { .. })
                | Err(CohortAdmissionError::PairRecordSplitMismatch(_))
        ));
    }

    #[test]
    fn rejects_reused_record_roles() {
        let mut manifest = fixture();
        manifest.pairs.push(CohortPair {
            pair_id: "pilot-pair-2".into(),
            baseline_record_id: "pilot-a".into(),
            variant_record_id: "pilot-b".into(),
            counterfactual_pair_manifest_sha256: A.into(),
            split: CohortSplit::Pilot,
            purpose: PairPurpose::PilotEvaluation,
        });
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::DuplicatePairId(_))
                | Err(CohortAdmissionError::RecordReusedAcrossPairs(_))
        ));
    }

    #[test]
    fn rejects_external_processing_without_separate_authorization() {
        let manifest = fixture();
        assert!(matches!(
            require_external_processing_authorization(&manifest.records[0]),
            Err(CohortAdmissionError::ExternalProcessingNotAuthorized(_))
        ));
    }

    #[test]
    fn permits_qualified_profile_only_with_native_review_digest() {
        let mut manifest = fixture();
        manifest.records[0].phonological_profile_id = Some("isiZulu-core".into());
        manifest.records[0].phonological_profile_revision = Some("1.0.0".into());
        manifest.records[0].phonological_profile_qualified = true;
        manifest.records[0].native_review_manifest_sha256 = Some(C.into());
        assert!(validate_cohort_manifest(&manifest).is_ok());
        manifest.records[0].native_review_manifest_sha256 = None;
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::MissingNativeReview(_))
        ));
    }

    #[test]
    fn canonical_digest_is_independent_of_input_record_order() {
        let manifest = fixture();
        let digest = canonical_cohort_manifest_digest(&manifest).unwrap();
        let mut reversed = manifest.clone();
        reversed.records.reverse();
        reversed.pairs.reverse();
        assert_eq!(
            digest,
            canonical_cohort_manifest_digest(&reversed).unwrap()
        );
    }

    #[test]
    fn rejects_extra_record_claiming_existing_pair_without_membership() {
        let mut manifest = fixture();
        manifest.records.push(record(
            "pilot-extra",
            "pilot-pair",
            "source-pilot",
            "speaker-pilot",
            "session-pilot",
            "utterance-pilot",
            CohortSplit::Pilot,
            A,
            A,
        ));
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::OrphanPairRecord(_))
        ));
    }

    #[test]
    fn rejects_invalid_counterfactual_pair_manifest_digest() {
        let mut manifest = fixture();
        manifest.pairs[0].counterfactual_pair_manifest_sha256 = "not-a-digest".into();
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::InvalidDigest(
                "counterfactual_pair_manifest_sha256",
                _
            ))
        ));
    }

    #[test]
    fn accepts_hex_digest_case_insensitively_for_same_audio() {
        let mut manifest = fixture();
        manifest.records[1].evaluation_audio_sha256 = A.to_ascii_uppercase();
        assert!(validate_cohort_manifest(&manifest).is_ok());
    }

    #[test]
    fn rejects_duplicate_pair_ids_across_uses() {
        let mut manifest = fixture();
        manifest.pairs[1].pair_id = manifest.pairs[0].pair_id.clone();
        assert!(matches!(
            validate_cohort_manifest(&manifest),
            Err(CohortAdmissionError::DuplicatePairId(_))
        ));
    }
}
