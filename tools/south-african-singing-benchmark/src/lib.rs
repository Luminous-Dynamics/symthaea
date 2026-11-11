//! SA-SING-001: validation of performer-consented singing benchmark manifests.
//!
//! This crate consumes metadata and receipts. It does not download or ingest
//! audio, decide whether a licence is legally sufficient, infer language,
//! annotate phones, synthesize voices, or train models.
//!
//! Rights eligibility remains owned by the corpus registry (#9509). Each case
//! must reference a registry evaluation-only receipt. Actual singing audio is
//! required for a singing case; speech-only resources cannot qualify F0,
//! melody, voice or singing-preservation tasks.
//!
//! A valid manifest is only a valid evaluation manifest. It is not proof that
//! the upstream rights review, annotations or acoustic measurements are true.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const SA_SING_SCHEMA_V1: &str = "sa-sing-benchmark/v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingingManifestError {
    pub message: String,
}

impl SingingManifestError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for SingingManifestError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for SingingManifestError {}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum IntendedUseV1 {
    BenchmarkEvaluationOnly,
    ModelTraining,
    VoiceIdentityReference,
    SyntheticGeneration,
    CommercialUse,
    PublicRedistribution,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RightsDispositionV1 {
    ApprovedForEvaluation,
    Restricted,
    Unknown,
    NotApproved,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RecordingModalityV1 {
    SungVocalPerformance,
    SpeechOnly,
    InstrumentalOnly,
    SignedPerformance,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSplitV1 {
    FeasibilityPilot,
    Calibration,
    Development,
    Confirmatory,
    LockedTest,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkTrackV1 {
    LyricPhoneticRealization,
    ScorePitchFidelity,
    LocalEditPreservation,
    CodeSwitchBoundaries,
    HumanPerceptualQuality,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKindV1 {
    LyricsText,
    PhoneSequence,
    PhoneTimeAlignment,
    ScoreEventMap,
    MeasuredF0Track,
    TargetRegionMap,
    ProtectedRegionMap,
    CodeSwitchBoundaryLabels,
    HumanPerceptualPanel,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationMethodV1 {
    HumanExpertAdjudication,
    IndependentReferenceMeasurement,
    ScoreAnnotation,
    UnreviewedAutomatic,
    CandidateSystemOutput,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageProfileRefV1 {
    pub language_id: String,
    pub profile_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SingingBenchmarkSpecV1 {
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub intended_use: IntendedUseV1,
    /// Immutable snapshot identity of the authoritative corpus registry.
    pub rights_registry_snapshot_sha256: String,
    pub language_profiles: Vec<LanguageProfileRefV1>,
    pub require_singer_disjoint_splits: bool,
    pub require_work_family_disjoint_splits: bool,
    pub require_session_disjoint_splits: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvaluationRightsReceiptV1 {
    pub registry_item_id: String,
    pub registry_snapshot_sha256: String,
    pub eligibility_receipt_sha256: String,
    pub intended_use: IntendedUseV1,
    pub disposition: RightsDispositionV1,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageSpanV1 {
    /// Inclusive start in source-audio samples.
    pub start_sample: u64,
    /// Exclusive end in source-audio samples.
    pub end_sample: u64,
    pub language_id: String,
    pub language_profile_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReferenceArtifactV1 {
    pub kind: ReferenceKindV1,
    pub artifact_sha256: String,
    pub annotation_receipt_sha256: String,
    pub method: AnnotationMethodV1,
    pub human_adjudicated: bool,
    pub native_speaker_reviewed: bool,
    /// For F0 truth this must be independent of the candidate pitch tracker
    /// and must not be synthesized from the score MIDI.
    pub independent_of_candidate_system: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SingingBenchmarkCaseV1 {
    pub case_id: String,
    pub modality: RecordingModalityV1,
    pub source_audio_sha256: String,
    pub sample_rate_hz: u32,
    pub duration_samples: u64,
    /// Public manifests must use pseudonymous identifiers. The mapping back to
    /// participant identity is held outside this repository.
    pub performer_pseudonym_id: String,
    pub song_work_family_pseudonym_id: String,
    pub recording_session_pseudonym_id: String,
    pub split: DatasetSplitV1,
    pub rights_receipt: EvaluationRightsReceiptV1,
    pub language_spans: Vec<LanguageSpanV1>,
    pub supported_tracks: Vec<BenchmarkTrackV1>,
    pub references: Vec<ReferenceArtifactV1>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SingingBenchmarkReportV1 {
    pub schema: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub intended_use: IntendedUseV1,
    pub case_count: usize,
    pub case_counts_by_split: BTreeMap<DatasetSplitV1, usize>,
    pub case_counts_by_language: BTreeMap<String, usize>,
    pub case_counts_by_track: BTreeMap<BenchmarkTrackV1, usize>,
    pub rights_approved_case_count: usize,
    pub speech_only_case_count: usize,
    /// Always zero for a valid v1 manifest: speech-only samples are rejected.
    pub non_singing_case_count: usize,
}

/// Validate a benchmark manifest against the frozen language profiles and
/// registry snapshot. The only admitted use in v1 is evaluation-only.
///
/// The returned report does not authorize training or synthesis. Speech-only
/// and signed-performance samples are rejected from this acoustic singing
/// benchmark and require separate modality-specific evaluation programmes.
pub fn validate_singing_benchmark_manifest(
    spec: &SingingBenchmarkSpecV1,
    cases: &[SingingBenchmarkCaseV1],
) -> Result<SingingBenchmarkReportV1, SingingManifestError> {
    validate_spec(spec)?;
    if cases.is_empty() {
        return Err(SingingManifestError::new(
            "at least one singing benchmark case is required",
        ));
    }

    let profiles: BTreeMap<&str, &str> = spec
        .language_profiles
        .iter()
        .map(|profile| (profile.language_id.as_str(), profile.profile_sha256.as_str()))
        .collect();

    let mut seen_case_ids = BTreeSet::new();
    let mut split_counts = BTreeMap::new();
    let mut language_counts = BTreeMap::new();
    let mut track_counts = BTreeMap::new();
    let mut singer_split_owners = BTreeMap::new();
    let mut work_split_owners = BTreeMap::new();
    let mut session_split_owners = BTreeMap::new();
    let mut audio_split_owners = BTreeMap::new();
    let mut audio_digests = BTreeMap::new();

    for case in cases {
        validate_case(spec, case, &profiles)?;
        if !seen_case_ids.insert(case.case_id.as_str()) {
            return Err(SingingManifestError::new(format!(
                "duplicate case_id: {}",
                case.case_id
            )));
        }

        claim_split_owner(
            &mut singer_split_owners,
            &case.performer_pseudonym_id,
            case.split,
            "performer",
            spec.require_singer_disjoint_splits,
        )?;
        claim_split_owner(
            &mut work_split_owners,
            &case.song_work_family_pseudonym_id,
            case.split,
            "song/work family",
            spec.require_work_family_disjoint_splits,
        )?;
        claim_split_owner(
            &mut session_split_owners,
            &case.recording_session_pseudonym_id,
            case.split,
            "recording session",
            spec.require_session_disjoint_splits,
        )?;
        claim_split_owner(
            &mut audio_split_owners,
            &case.source_audio_sha256,
            case.split,
            "source audio digest",
            true,
        )?;

        if let Some(previous_case) =
            audio_digests.insert(case.source_audio_sha256.as_str(), case.case_id.as_str())
        {
            if previous_case != case.case_id {
                return Err(SingingManifestError::new(format!(
                    "source audio digest is reused by cases {} and {}",
                    previous_case, case.case_id
                )));
            }
        }

        *split_counts.entry(case.split).or_insert(0) += 1;
        for language_id in unique_case_languages(case) {
            *language_counts.entry(language_id.to_owned()).or_insert(0) += 1;
        }
        for track in &case.supported_tracks {
            *track_counts.entry(*track).or_insert(0) += 1;
        }
    }

    Ok(SingingBenchmarkReportV1 {
        schema: SA_SING_SCHEMA_V1.to_owned(),
        benchmark_id: spec.benchmark_id.clone(),
        benchmark_version: spec.benchmark_version.clone(),
        intended_use: IntendedUseV1::BenchmarkEvaluationOnly,
        case_count: cases.len(),
        case_counts_by_split: split_counts,
        case_counts_by_language: language_counts,
        case_counts_by_track: track_counts,
        rights_approved_case_count: cases.len(),
        speech_only_case_count: 0,
        non_singing_case_count: 0,
    })
}

fn validate_spec(spec: &SingingBenchmarkSpecV1) -> Result<(), SingingManifestError> {
    require_non_empty(&spec.benchmark_id, "benchmark_id")?;
    require_non_empty(&spec.benchmark_version, "benchmark_version")?;
    validate_digest(
        &spec.rights_registry_snapshot_sha256,
        "rights_registry_snapshot_sha256",
    )?;
    if spec.intended_use != IntendedUseV1::BenchmarkEvaluationOnly {
        return Err(SingingManifestError::new(
            "SA-SING v1 accepts benchmark evaluation only; training and synthesis uses are out of scope",
        ));
    }
    if spec.language_profiles.is_empty() {
        return Err(SingingManifestError::new(
            "at least one approved language profile must be declared",
        ));
    }
    let mut seen = BTreeSet::new();
    for profile in &spec.language_profiles {
        require_non_empty(&profile.language_id, "language_id")?;
        validate_digest(&profile.profile_sha256, "language_profile_sha256")?;
        if !seen.insert(profile.language_id.as_str()) {
            return Err(SingingManifestError::new(format!(
                "duplicate language profile: {}",
                profile.language_id
            )));
        }
    }
    Ok(())
}

fn validate_case(
    spec: &SingingBenchmarkSpecV1,
    case: &SingingBenchmarkCaseV1,
    profiles: &BTreeMap<&str, &str>,
) -> Result<(), SingingManifestError> {
    require_non_empty(&case.case_id, "case_id")?;
    if case.modality != RecordingModalityV1::SungVocalPerformance {
        return Err(SingingManifestError::new(format!(
            "case {} is not a sung-vocal recording; speech-only, instrumental and signed modalities require separate tracks",
            case.case_id
        )));
    }
    validate_digest(&case.source_audio_sha256, "source_audio_sha256")?;
    if !(8_000..=192_000).contains(&case.sample_rate_hz) || case.duration_samples == 0 {
        return Err(SingingManifestError::new(format!(
            "case {} has invalid sample rate or zero duration",
            case.case_id
        )));
    }
    require_non_empty(
        &case.performer_pseudonym_id,
        "performer_pseudonym_id",
    )?;
    require_non_empty(
        &case.song_work_family_pseudonym_id,
        "song_work_family_pseudonym_id",
    )?;
    require_non_empty(
        &case.recording_session_pseudonym_id,
        "recording_session_pseudonym_id",
    )?;

    if case.rights_receipt.intended_use != IntendedUseV1::BenchmarkEvaluationOnly {
        return Err(SingingManifestError::new(format!(
            "case {} rights receipt is not scoped to benchmark evaluation only",
            case.case_id
        )));
    }
    if case.rights_receipt.disposition != RightsDispositionV1::ApprovedForEvaluation {
        return Err(SingingManifestError::new(format!(
            "case {} lacks an approved evaluation-only rights disposition",
            case.case_id
        )));
    }
    require_non_empty(
        &case.rights_receipt.registry_item_id,
        "rights_registry_item_id",
    )?;
    validate_digest(
        &case.rights_receipt.registry_snapshot_sha256,
        "case_rights_registry_snapshot_sha256",
    )?;
    if case.rights_receipt.registry_snapshot_sha256
        != spec.rights_registry_snapshot_sha256
    {
        return Err(SingingManifestError::new(format!(
            "case {} references a different rights-registry snapshot",
            case.case_id
        )));
    }
    validate_digest(
        &case.rights_receipt.eligibility_receipt_sha256,
        "rights_eligibility_receipt_sha256",
    )?;

    if case.language_spans.is_empty() {
        return Err(SingingManifestError::new(format!(
            "case {} has no language spans",
            case.case_id
        )));
    }
    let mut previous_end = 0u64;
    let mut language_ids = BTreeSet::new();
    for (index, span) in case.language_spans.iter().enumerate() {
        if span.start_sample >= span.end_sample || span.end_sample > case.duration_samples {
            return Err(SingingManifestError::new(format!(
                "case {} has an invalid language-span interval",
                case.case_id
            )));
        }
        if index > 0 && span.start_sample < previous_end {
            return Err(SingingManifestError::new(format!(
                "case {} has overlapping or out-of-order language spans",
                case.case_id
            )));
        }
        previous_end = span.end_sample;
        let expected_profile_sha = profiles
            .get(span.language_id.as_str())
            .ok_or_else(|| {
                SingingManifestError::new(format!(
                    "case {} uses undeclared language profile {}",
                    case.case_id, span.language_id
                ))
            })?;
        validate_digest(
            &span.language_profile_sha256,
            "language_span_profile_sha256",
        )?;
        if span.language_profile_sha256.as_str() != *expected_profile_sha {
            return Err(SingingManifestError::new(format!(
                "case {} has a stale language-profile digest for {}",
                case.case_id, span.language_id
            )));
        }
        language_ids.insert(span.language_id.as_str());
    }

    if case.supported_tracks.is_empty() {
        return Err(SingingManifestError::new(format!(
            "case {} does not declare any supported evaluation tracks",
            case.case_id
        )));
    }
    let mut tracks = BTreeSet::new();
    for track in &case.supported_tracks {
        if !tracks.insert(*track) {
            return Err(SingingManifestError::new(format!(
                "case {} has a duplicate supported track",
                case.case_id
            )));
        }
    }

    let mut annotations = BTreeMap::new();
    for annotation in &case.references {
        validate_annotation(case, annotation)?;
        if annotations.insert(annotation.kind, annotation).is_some() {
            return Err(SingingManifestError::new(format!(
                "case {} has duplicate reference artifact kind {:?}",
                case.case_id, annotation.kind
            )));
        }
    }

    for track in &case.supported_tracks {
        validate_track_evidence(
            case,
            *track,
            &annotations,
            language_ids.len(),
        )?;
    }

    Ok(())
}

fn validate_annotation(
    case: &SingingBenchmarkCaseV1,
    annotation: &ReferenceArtifactV1,
) -> Result<(), SingingManifestError> {
    validate_digest(&annotation.artifact_sha256, "reference_artifact_sha256")?;
    validate_digest(
        &annotation.annotation_receipt_sha256,
        "annotation_receipt_sha256",
    )?;

    match annotation.kind {
        ReferenceKindV1::LyricsText | ReferenceKindV1::PhoneSequence => {
            if !annotation.native_speaker_reviewed || !annotation.human_adjudicated {
                return Err(SingingManifestError::new(format!(
                    "case {} {:?} reference requires native-speaker human adjudication",
                    case.case_id, annotation.kind
                )));
            }
        }
        ReferenceKindV1::PhoneTimeAlignment
        | ReferenceKindV1::CodeSwitchBoundaryLabels => {
            if !annotation.human_adjudicated || !annotation.native_speaker_reviewed {
                return Err(SingingManifestError::new(format!(
                    "case {} {:?} reference requires native-speaker adjudication",
                    case.case_id, annotation.kind
                )));
            }
        }
        ReferenceKindV1::ScoreEventMap => {
            if !annotation.human_adjudicated
                || !matches!(
                    annotation.method,
                    AnnotationMethodV1::ScoreAnnotation
                        | AnnotationMethodV1::HumanExpertAdjudication
                )
            {
                return Err(SingingManifestError::new(format!(
                    "case {} score event map requires a reviewed score annotation",
                    case.case_id
                )));
            }
        }
        ReferenceKindV1::TargetRegionMap
        | ReferenceKindV1::ProtectedRegionMap
        | ReferenceKindV1::HumanPerceptualPanel => {
            if !annotation.human_adjudicated {
                return Err(SingingManifestError::new(format!(
                    "case {} {:?} reference is not human-adjudicated",
                    case.case_id, annotation.kind
                )));
            }
        }
        ReferenceKindV1::MeasuredF0Track => {
            if !annotation.independent_of_candidate_system
                || !annotation.human_adjudicated
                || matches!(
                    annotation.method,
                    AnnotationMethodV1::ScoreAnnotation
                        | AnnotationMethodV1::UnreviewedAutomatic
                        | AnnotationMethodV1::CandidateSystemOutput
                )
            {
                return Err(SingingManifestError::new(format!(
                    "case {} measured F0 reference must be human-reviewed and independent of the candidate tracker and score prior",
                    case.case_id
                )));
            }
        }
    }
    Ok(())
}

fn validate_track_evidence(
    case: &SingingBenchmarkCaseV1,
    track: BenchmarkTrackV1,
    annotations: &BTreeMap<ReferenceKindV1, &ReferenceArtifactV1>,
    distinct_language_count: usize,
) -> Result<(), SingingManifestError> {
    let required: &[ReferenceKindV1] = match track {
        BenchmarkTrackV1::LyricPhoneticRealization => &[
            ReferenceKindV1::LyricsText,
            ReferenceKindV1::PhoneSequence,
            ReferenceKindV1::PhoneTimeAlignment,
        ],
        BenchmarkTrackV1::ScorePitchFidelity => &[
            ReferenceKindV1::ScoreEventMap,
            ReferenceKindV1::MeasuredF0Track,
        ],
        BenchmarkTrackV1::LocalEditPreservation => &[
            ReferenceKindV1::TargetRegionMap,
            ReferenceKindV1::ProtectedRegionMap,
            ReferenceKindV1::PhoneTimeAlignment,
        ],
        BenchmarkTrackV1::CodeSwitchBoundaries => &[
            ReferenceKindV1::LyricsText,
            ReferenceKindV1::PhoneSequence,
            ReferenceKindV1::PhoneTimeAlignment,
            ReferenceKindV1::CodeSwitchBoundaryLabels,
        ],
        BenchmarkTrackV1::HumanPerceptualQuality => &[
            ReferenceKindV1::HumanPerceptualPanel,
        ],
    };

    for kind in required {
        if !annotations.contains_key(kind) {
            return Err(SingingManifestError::new(format!(
                "case {} claims track {:?} but is missing {:?}",
                case.case_id, track, kind
            )));
        }
    }
    if track == BenchmarkTrackV1::CodeSwitchBoundaries && distinct_language_count < 2 {
        return Err(SingingManifestError::new(format!(
            "case {} claims code-switch evaluation without at least two declared language profiles",
            case.case_id
        )));
    }
    Ok(())
}

fn claim_split_owner(
    owners: &mut BTreeMap<String, DatasetSplitV1>,
    identity: &str,
    split: DatasetSplitV1,
    field: &str,
    enforce: bool,
) -> Result<(), SingingManifestError> {
    if !enforce {
        return Ok(());
    }
    match owners.get(identity) {
        Some(previous) if *previous != split => Err(SingingManifestError::new(format!(
            "{field} leakage: {identity} appears in both {previous:?} and {split:?}"
        ))),
        Some(_) => Ok(()),
        None => {
            owners.insert(identity.to_owned(), split);
            Ok(())
        }
    }
}

fn unique_case_languages(case: &SingingBenchmarkCaseV1) -> BTreeSet<&str> {
    case.language_spans
        .iter()
        .map(|span| span.language_id.as_str())
        .collect()
}

fn validate_digest(value: &str, field: &str) -> Result<(), SingingManifestError> {
    let valid = value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex.bytes().all(|byte| {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            })
    });
    if !valid {
        return Err(SingingManifestError::new(format!(
            "{field} must be sha256: followed by 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), SingingManifestError> {
    if value.trim().is_empty() {
        Err(SingingManifestError::new(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        format!("sha256:{}", ch.to_string().repeat(64))
    }

    fn spec() -> SingingBenchmarkSpecV1 {
        SingingBenchmarkSpecV1 {
            benchmark_id: "sa-sing-pilot".to_owned(),
            benchmark_version: "v1".to_owned(),
            intended_use: IntendedUseV1::BenchmarkEvaluationOnly,
            rights_registry_snapshot_sha256: digest('1'),
            language_profiles: vec![
                LanguageProfileRefV1 {
                    language_id: "zu-ZA".to_owned(),
                    profile_sha256: digest('2'),
                },
                LanguageProfileRefV1 {
                    language_id: "en-ZA".to_owned(),
                    profile_sha256: digest('3'),
                },
            ],
            require_singer_disjoint_splits: true,
            require_work_family_disjoint_splits: true,
            require_session_disjoint_splits: true,
        }
    }

    fn annotation(
        kind: ReferenceKindV1,
        id: char,
        method: AnnotationMethodV1,
        human: bool,
        native: bool,
        independent: bool,
    ) -> ReferenceArtifactV1 {
        ReferenceArtifactV1 {
            kind,
            artifact_sha256: digest(id),
            annotation_receipt_sha256: digest('f'),
            method,
            human_adjudicated: human,
            native_speaker_reviewed: native,
            independent_of_candidate_system: independent,
        }
    }

    fn valid_case() -> SingingBenchmarkCaseV1 {
        SingingBenchmarkCaseV1 {
            case_id: "case-001".to_owned(),
            modality: RecordingModalityV1::SungVocalPerformance,
            source_audio_sha256: digest('4'),
            sample_rate_hz: 48_000,
            duration_samples: 480_000,
            performer_pseudonym_id: "performer-pseudo-01".to_owned(),
            song_work_family_pseudonym_id: "work-pseudo-01".to_owned(),
            recording_session_pseudonym_id: "session-pseudo-01".to_owned(),
            split: DatasetSplitV1::FeasibilityPilot,
            rights_receipt: EvaluationRightsReceiptV1 {
                registry_item_id: "muse-corpus-item-01".to_owned(),
                registry_snapshot_sha256: digest('1'),
                eligibility_receipt_sha256: digest('5'),
                intended_use: IntendedUseV1::BenchmarkEvaluationOnly,
                disposition: RightsDispositionV1::ApprovedForEvaluation,
            },
            language_spans: vec![LanguageSpanV1 {
                start_sample: 0,
                end_sample: 480_000,
                language_id: "zu-ZA".to_owned(),
                language_profile_sha256: digest('2'),
            }],
            supported_tracks: vec![
                BenchmarkTrackV1::LyricPhoneticRealization,
                BenchmarkTrackV1::ScorePitchFidelity,
                BenchmarkTrackV1::LocalEditPreservation,
            ],
            references: vec![
                annotation(
                    ReferenceKindV1::LyricsText,
                    '6',
                    AnnotationMethodV1::HumanExpertAdjudication,
                    true,
                    true,
                    true,
                ),
                annotation(
                    ReferenceKindV1::PhoneSequence,
                    '7',
                    AnnotationMethodV1::HumanExpertAdjudication,
                    true,
                    true,
                    true,
                ),
                annotation(
                    ReferenceKindV1::PhoneTimeAlignment,
                    '8',
                    AnnotationMethodV1::IndependentReferenceMeasurement,
                    true,
                    true,
                    true,
                ),
                annotation(
                    ReferenceKindV1::ScoreEventMap,
                    '9',
                    AnnotationMethodV1::ScoreAnnotation,
                    true,
                    false,
                    true,
                ),
                annotation(
                    ReferenceKindV1::MeasuredF0Track,
                    'a',
                    AnnotationMethodV1::IndependentReferenceMeasurement,
                    true,
                    false,
                    true,
                ),
                annotation(
                    ReferenceKindV1::TargetRegionMap,
                    'b',
                    AnnotationMethodV1::HumanExpertAdjudication,
                    true,
                    false,
                    true,
                ),
                annotation(
                    ReferenceKindV1::ProtectedRegionMap,
                    'c',
                    AnnotationMethodV1::HumanExpertAdjudication,
                    true,
                    false,
                    true,
                ),
            ],
        }
    }

    #[test]
    fn valid_evaluation_only_singing_case_is_accepted() {
        let report =
            validate_singing_benchmark_manifest(&spec(), &[valid_case()]).unwrap();
        assert_eq!(report.schema, SA_SING_SCHEMA_V1);
        assert_eq!(report.case_count, 1);
        assert_eq!(report.rights_approved_case_count, 1);
        assert_eq!(
            report.case_counts_by_track[&BenchmarkTrackV1::ScorePitchFidelity],
            1
        );
    }

    #[test]
    fn speech_only_instrumental_and_signed_modalities_are_rejected() {
        for modality in [
            RecordingModalityV1::SpeechOnly,
            RecordingModalityV1::InstrumentalOnly,
            RecordingModalityV1::SignedPerformance,
        ] {
            let mut case = valid_case();
            case.modality = modality;
            assert!(
                validate_singing_benchmark_manifest(&spec(), &[case]).is_err()
            );
        }
    }

    #[test]
    fn unknown_or_restricted_rights_are_rejected() {
        for disposition in [
            RightsDispositionV1::Unknown,
            RightsDispositionV1::Restricted,
            RightsDispositionV1::NotApproved,
        ] {
            let mut case = valid_case();
            case.rights_receipt.disposition = disposition;
            assert!(
                validate_singing_benchmark_manifest(&spec(), &[case]).is_err()
            );
        }
    }

    #[test]
    fn rights_registry_snapshot_must_match_frozen_benchmark_spec() {
        let mut case = valid_case();
        case.rights_receipt.registry_snapshot_sha256 = digest('9');
        assert!(validate_singing_benchmark_manifest(&spec(), &[case]).is_err());
    }

    #[test]
    fn non_evaluation_purposes_are_rejected_at_benchmark_boundary() {
        for purpose in [
            IntendedUseV1::ModelTraining,
            IntendedUseV1::VoiceIdentityReference,
            IntendedUseV1::SyntheticGeneration,
            IntendedUseV1::CommercialUse,
            IntendedUseV1::PublicRedistribution,
        ] {
            let mut benchmark_spec = spec();
            benchmark_spec.intended_use = purpose;
            assert!(
                validate_singing_benchmark_manifest(&benchmark_spec, &[valid_case()]).is_err()
            );

            let mut case = valid_case();
            case.rights_receipt.intended_use = purpose;
            assert!(
                validate_singing_benchmark_manifest(&spec(), &[case]).is_err()
            );
        }
    }

    #[test]
    fn invalid_or_undeclared_language_profile_is_rejected() {
        let mut case = valid_case();
        case.language_spans[0].language_profile_sha256 = digest('8');
        assert!(validate_singing_benchmark_manifest(&spec(), &[case]).is_err());

        let mut undeclared = valid_case();
        undeclared.language_spans[0].language_id = "af-ZA".to_owned();
        assert!(validate_singing_benchmark_manifest(&spec(), &[undeclared]).is_err());
    }

    #[test]
    fn invalid_overlapping_or_out_of_range_language_spans_are_rejected() {
        let mut overlap = valid_case();
        overlap.language_spans = vec![
            LanguageSpanV1 {
                start_sample: 0,
                end_sample: 200_000,
                language_id: "zu-ZA".to_owned(),
                language_profile_sha256: digest('2'),
            },
            LanguageSpanV1 {
                start_sample: 199_000,
                end_sample: 480_000,
                language_id: "en-ZA".to_owned(),
                language_profile_sha256: digest('3'),
            },
        ];
        assert!(
            validate_singing_benchmark_manifest(&spec(), &[overlap]).is_err()
        );

        let mut out_of_range = valid_case();
        out_of_range.language_spans[0].end_sample = 500_000;
        assert!(
            validate_singing_benchmark_manifest(&spec(), &[out_of_range]).is_err()
        );
    }

    #[test]
    fn score_pitch_track_requires_independent_measured_f0() {
        let mut case = valid_case();
        let f0 = case
            .references
            .iter_mut()
            .find(|item| item.kind == ReferenceKindV1::MeasuredF0Track)
            .unwrap();
        f0.method = AnnotationMethodV1::ScoreAnnotation;
        f0.independent_of_candidate_system = false;
        assert!(validate_singing_benchmark_manifest(&spec(), &[case]).is_err());
    }

    #[test]
    fn lyric_phone_track_requires_native_speaker_human_adjudication() {
        let mut case = valid_case();
        let phones = case
            .references
            .iter_mut()
            .find(|item| item.kind == ReferenceKindV1::PhoneSequence)
            .unwrap();
        phones.native_speaker_reviewed = false;
        assert!(validate_singing_benchmark_manifest(&spec(), &[case]).is_err());
    }

    #[test]
    fn every_supported_track_requires_its_reference_artifact_set() {
        let mut case = valid_case();
        case.references.retain(|item| item.kind != ReferenceKindV1::MeasuredF0Track);
        assert!(validate_singing_benchmark_manifest(&spec(), &[case]).is_err());

        let mut local_edit = valid_case();
        local_edit.references.retain(|item| item.kind != ReferenceKindV1::ProtectedRegionMap);
        assert!(validate_singing_benchmark_manifest(&spec(), &[local_edit]).is_err());
    }

    #[test]
    fn code_switch_track_requires_two_language_profiles_and_boundary_reference() {
        let mut case = valid_case();
        case.supported_tracks = vec![BenchmarkTrackV1::CodeSwitchBoundaries];
        case.references.retain(|item| {
            matches!(
                item.kind,
                ReferenceKindV1::PhoneSequence | ReferenceKindV1::PhoneTimeAlignment
            )
        });
        case.references.push(annotation(
            ReferenceKindV1::CodeSwitchBoundaryLabels,
            'd',
            AnnotationMethodV1::HumanExpertAdjudication,
            true,
            true,
            true,
        ));
        assert!(validate_singing_benchmark_manifest(&spec(), &[case.clone()]).is_err());

        case.language_spans = vec![
            LanguageSpanV1 {
                start_sample: 0,
                end_sample: 200_000,
                language_id: "zu-ZA".to_owned(),
                language_profile_sha256: digest('2'),
            },
            LanguageSpanV1 {
                start_sample: 200_000,
                end_sample: 480_000,
                language_id: "en-ZA".to_owned(),
                language_profile_sha256: digest('3'),
            },
        ];
        assert!(validate_singing_benchmark_manifest(&spec(), &[case]).is_ok());
    }

    #[test]
    fn duplicate_case_id_and_reference_kind_are_rejected() {
        let case = valid_case();
        assert!(
            validate_singing_benchmark_manifest(&spec(), &[case.clone(), case]).is_err()
        );

        let mut duplicate_reference = valid_case();
        duplicate_reference.references.push(
            duplicate_reference.references[0].clone()
        );
        assert!(
            validate_singing_benchmark_manifest(&spec(), &[duplicate_reference]).is_err()
        );
    }

    #[test]
    fn singer_work_and_session_leakage_across_splits_is_rejected() {
        let mut first = valid_case();
        first.split = DatasetSplitV1::Calibration;

        let mut second = valid_case();
        second.case_id = "case-002".to_owned();
        second.split = DatasetSplitV1::Confirmatory;
        second.performer_pseudonym_id = "performer-pseudo-02".to_owned();
        second.song_work_family_pseudonym_id = "work-pseudo-02".to_owned();
        second.recording_session_pseudonym_id = "session-pseudo-02".to_owned();
        second.source_audio_sha256 = digest('d');

        for mutate in 0..3 {
            let mut leaked = second.clone();
            match mutate {
                0 => leaked.performer_pseudonym_id = first.performer_pseudonym_id.clone(),
                1 => leaked.song_work_family_pseudonym_id = first.song_work_family_pseudonym_id.clone(),
                _ => leaked.recording_session_pseudonym_id = first.recording_session_pseudonym_id.clone(),
            }
            assert!(
                validate_singing_benchmark_manifest(&spec(), &[first.clone(), leaked]).is_err()
            );
        }
    }

    #[test]
    fn same_split_can_contain_multiple_cases_from_same_performer_and_session() {
        let mut first = valid_case();
        first.split = DatasetSplitV1::Development;
        let mut second = first.clone();
        second.case_id = "case-002".to_owned();
        second.song_work_family_pseudonym_id = "work-pseudo-02".to_owned();
        second.source_audio_sha256 = digest('d');
        let report =
            validate_singing_benchmark_manifest(&spec(), &[first, second]).unwrap();
        assert_eq!(report.case_counts_by_split[&DatasetSplitV1::Development], 2);
    }

    #[test]
    fn duplicate_audio_digest_in_distinct_cases_is_rejected() {
        let first = valid_case();
        let mut second = first.clone();
        second.case_id = "case-002".to_owned();
        assert!(
            validate_singing_benchmark_manifest(&spec(), &[first, second]).is_err()
        );
    }

    #[test]
    fn malformed_digest_or_sample_clock_is_rejected() {
        let mut bad_digest = valid_case();
        bad_digest.source_audio_sha256 = "not-a-digest".to_owned();
        assert!(validate_singing_benchmark_manifest(&spec(), &[bad_digest]).is_err());

        let mut bad_clock = valid_case();
        bad_clock.sample_rate_hz = 0;
        assert!(validate_singing_benchmark_manifest(&spec(), &[bad_clock]).is_err());
    }

    #[test]
    fn benchmark_must_remain_evaluation_only() {
        let mut benchmark_spec = spec();
        benchmark_spec.intended_use = IntendedUseV1::BenchmarkEvaluationOnly;
        assert!(validate_singing_benchmark_manifest(&benchmark_spec, &[valid_case()]).is_ok());
    }

    #[test]
    fn no_reference_artifact_may_use_candidate_output_as_f0_truth() {
        let mut case = valid_case();
        let f0 = case
            .references
            .iter_mut()
            .find(|item| item.kind == ReferenceKindV1::MeasuredF0Track)
            .unwrap();
        f0.method = AnnotationMethodV1::CandidateSystemOutput;
        f0.independent_of_candidate_system = false;
        assert!(validate_singing_benchmark_manifest(&spec(), &[case]).is_err());
    }
}
