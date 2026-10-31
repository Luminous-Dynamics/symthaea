// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Candidate corpus registry for source-grounded musical learning.
//!
//! Registering a dataset never approves it for training. Composition, score
//! edition, performance, sound recording, transcription, alignment metadata
//! and scholarly annotation can have different owners and rights. This
//! registry records published facts and blockers; item-level ingest approval
//! is a separate future workflow.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MUSIC_CORPUS_REGISTRY_SCHEMA: &str = "muse-music-corpus-registry-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusModality {
    NotatedScore,
    SymbolicMidi,
    PerformanceMidi,
    AudioRecording,
    PerformanceAlignment,
    Metadata,
    ScholarlyAnnotation,
    InstrumentSampleBank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsDisposition {
    /// Work, edition, recording or individual row rights remain to be checked.
    PerItemRightsReviewRequired,
    /// Published dataset structure/annotations have non-commercial restrictions.
    NonCommercialResearchOnly,
    /// The published description reports a permissive license, but local
    /// release/file identity and upstream terms still require pinning.
    ReportedPermissiveNeedsPinnedRelease,
    /// License/reuse information is not sufficiently verified.
    LicenseUnverified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingDisposition {
    /// Registry discovery is not permission to ingest or train.
    BlockedPendingRightsReview,
    /// May be considered for non-commercial research only after checking
    /// exact release, source rows and the project's terms.
    NonCommercialResearchAfterVerification,
    /// Asset license is permissive, but exact files and provenance still
    /// need a pinned manifest before local use.
    AssetUseAfterPinnedSourceVerification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusEvidenceStage {
    CandidateDiscovered,
    ReleasePinned,
    RightsReviewed,
    ArtifactDigestsVerified,
    IngestApproved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusScopeCounts {
    pub items: Option<u64>,
    pub performances: Option<u64>,
    pub works_or_scores: Option<u64>,
    pub composers_or_creators: Option<u64>,
    pub audio_hours: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicCorpusRecord {
    pub id: &'static str,
    pub title: &'static str,
    pub release: &'static str,
    pub upstream_uri: &'static str,
    pub modalities: &'static [CorpusModality],
    pub scope: &'static str,
    pub counts: CorpusScopeCounts,
    pub reported_rights: &'static str,
    pub rights_disposition: RightsDisposition,
    pub training_disposition: TrainingDisposition,
    pub evidence_stage: CorpusEvidenceStage,
    pub known_limitations: &'static [&'static str],
    pub ingest_gate: &'static str,
}

const MAESTRO_MODALITIES: &[CorpusModality] = &[
    CorpusModality::SymbolicMidi,
    CorpusModality::PerformanceMidi,
    CorpusModality::AudioRecording,
    CorpusModality::PerformanceAlignment,
    CorpusModality::Metadata,
];
const GIANT_MIDI_MODALITIES: &[CorpusModality] = &[
    CorpusModality::PerformanceMidi,
    CorpusModality::SymbolicMidi,
    CorpusModality::Metadata,
];
const PERISCOPE_MODALITIES: &[CorpusModality] = &[
    CorpusModality::NotatedScore,
    CorpusModality::SymbolicMidi,
    CorpusModality::PerformanceMidi,
    CorpusModality::PerformanceAlignment,
    CorpusModality::Metadata,
];
const GANGQINPU_MODALITIES: &[CorpusModality] = &[
    CorpusModality::NotatedScore,
    CorpusModality::SymbolicMidi,
    CorpusModality::PerformanceMidi,
    CorpusModality::AudioRecording,
    CorpusModality::PerformanceAlignment,
    CorpusModality::Metadata,
];
const IMSLP_MODALITIES: &[CorpusModality] = &[
    CorpusModality::NotatedScore,
    CorpusModality::Metadata,
];
const UNESCO_MODALITIES: &[CorpusModality] = &[
    CorpusModality::AudioRecording,
    CorpusModality::Metadata,
    CorpusModality::ScholarlyAnnotation,
];
const VCSL_MODALITIES: &[CorpusModality] = &[
    CorpusModality::InstrumentSampleBank,
    CorpusModality::AudioRecording,
    CorpusModality::Metadata,
];

/// Candidate sources only. None is approved for automatic ingestion by
/// inclusion in this constant.
pub const MUSIC_CORPORA: &[MusicCorpusRecord] = &[
    MusicCorpusRecord {
        id: "maestro",
        title: "MAESTRO",
        release: "2.0.0",
        upstream_uri: "https://magenta.withgoogle.com/datasets/maestro",
        modalities: MAESTRO_MODALITIES,
        scope: "Virtuoso classical piano performances; MIDI and audio aligned to approximately 3 ms; repertoire predominantly from the 17th to early 20th centuries.",
        counts: CorpusScopeCounts {
            items: Some(1_282),
            performances: Some(1_282),
            works_or_scores: None,
            composers_or_creators: Some(62),
            audio_hours: Some(201.2),
        },
        reported_rights: "CC BY-NC-SA 4.0 (dataset)",
        rights_disposition: RightsDisposition::NonCommercialResearchOnly,
        training_disposition: TrainingDisposition::NonCommercialResearchAfterVerification,
        evidence_stage: CorpusEvidenceStage::CandidateDiscovered,
        known_limitations: &[
            "Piano-only; not a multi-instrument or world-music corpus",
            "Dataset license is non-commercial and share-alike",
            "Use the exact 2.0.0 release and its incompatible split definition; do not mix release versions silently",
            "Performance capture is not equivalent to a complete engraved score archive",
        ],
        ingest_gate: "Pin MAESTRO 2.0.0 archive SHA-256, retain its split files, verify licence and provenance, and keep its non-commercial restriction outside commercial training",
    },
    MusicCorpusRecord {
        id: "giant-midi-piano",
        title: "GiantMIDI-Piano",
        release: "paper-described corpus; exact artifact release must be pinned",
        upstream_uri: "https://doi.org/10.5334/tismir.80",
        modalities: GIANT_MIDI_MODALITIES,
        scope: "Large classical piano MIDI collection with both performed and sequenced MIDI; useful for historical composer/work breadth and score-pattern analysis.",
        counts: CorpusScopeCounts {
            items: Some(10_855),
            performances: None,
            works_or_scores: Some(10_855),
            composers_or_creators: Some(2_786),
            audio_hours: Some(1_237.0),
        },
        reported_rights: "Not verified from the cited paper record; individual MIDI provenance may differ",
        rights_disposition: RightsDisposition::LicenseUnverified,
        training_disposition: TrainingDisposition::BlockedPendingRightsReview,
        evidence_stage: CorpusEvidenceStage::CandidateDiscovered,
        known_limitations: &[
            "Piano-only; cannot establish ensemble/instrument technique coverage",
            "Mixes performance and sequenced MIDI; preserve that distinction",
            "Work/edition duplicates and composition identity need deduplication",
            "Dataset-wide reuse/training license has not been established in this registry",
        ],
        ingest_gate: "Do not ingest until exact files, source lineage, license and per-item reuse rights are reviewed",
    },
    MusicCorpusRecord {
        id: "periscope",
        title: "PERiScoPe: Piano Expression Refined Score and Performance MIDI Dataset",
        release: "v1.1 (January 2026)",
        upstream_uri: "https://huggingface.co/datasets/SyMuPe/PERiScoPe",
        modalities: PERISCOPE_MODALITIES,
        scope: "Aligned piano scores and performances assembled from (n)ASAP, ATEPP and web-collected/transcribed performances.",
        counts: CorpusScopeCounts {
            items: Some(35_815),
            performances: Some(35_815),
            works_or_scores: Some(1_163),
            composers_or_creators: Some(66),
            audio_hours: Some(2_848.0),
        },
        reported_rights: "CC BY-NC-SA 4.0 for dataset structure, metadata and alignment annotations; underlying works and recordings retain their own rights",
        rights_disposition: RightsDisposition::NonCommercialResearchOnly,
        training_disposition: TrainingDisposition::NonCommercialResearchAfterVerification,
        evidence_stage: CorpusEvidenceStage::CandidateDiscovered,
        known_limitations: &[
            "Dataset card states some MIDI originates in web collection and automatic transcription",
            "It explicitly disclaims ownership of underlying compositions and original audio performances",
            "Some alignment, transcription and composition identity errors remain",
            "Non-commercial research only; not a commercial-training source",
        ],
        ingest_gate: "Pin v1.1 and its file list; preserve per-source/row lineage; verify disclaimer and non-commercial use before research-only ingest",
    },
    MusicCorpusRecord {
        id: "gangqinpu",
        title: "Gangqinpu: Multi-Genre Piano Scores, Performance MIDI and Audio",
        release: "2026-10-26 paper; release and access terms require upstream verification",
        upstream_uri: "https://arxiv.org/abs/2610.34694",
        modalities: GANGQINPU_MODALITIES,
        scope: "Piano across pop, classical, jazz, folk/traditional, rock and blues, with score-performance pairs, source audio, automatic note alignment and beat/downbeat annotations.",
        counts: CorpusScopeCounts {
            items: Some(18_786),
            performances: Some(18_786),
            works_or_scores: None,
            composers_or_creators: None,
            audio_hours: Some(1_007.1),
        },
        reported_rights: "Author-created metadata/transcriptions/alignments reported as CC BY-NC-SA 4.0 in a dataset mirror; underlying works, recordings, scores and source metadata retain separate rights",
        rights_disposition: RightsDisposition::ReportedPermissiveNeedsPinnedRelease,
        training_disposition: TrainingDisposition::BlockedPendingRightsReview,
        evidence_stage: CorpusEvidenceStage::CandidateDiscovered,
        known_limitations: &[
            "Piano-only despite multi-genre coverage",
            "Note alignments and genre labels are automatically derived and can be noisy",
            "Authors report mainly in-domain benefits for transcription/rendering and complementary cross-dataset gains for beat tracking",
            "A mirrored licence statement is not a substitute for checking the official release and per-item rights",
        ],
        ingest_gate: "Verify official upstream release, access process, dataset card, licence and underlying score/recording rights; quarantine rows without adequate clearance",
    },
    MusicCorpusRecord {
        id: "imslp",
        title: "IMSLP / Petrucci Music Library",
        release: "Per-item and per-edition",
        upstream_uri: "https://imslp.org/wiki/Public_domain",
        modalities: IMSLP_MODALITIES,
        scope: "Historical printed scores and metadata across many composers, periods, regions and instrumentations.",
        counts: CorpusScopeCounts {
            items: None,
            performances: None,
            works_or_scores: None,
            composers_or_creators: None,
            audio_hours: None,
        },
        reported_rights: "Underlying works may be public domain in some jurisdictions; editions, engravings, editorial additions and recordings can have separate rights",
        rights_disposition: RightsDisposition::PerItemRightsReviewRequired,
        training_disposition: TrainingDisposition::BlockedPendingRightsReview,
        evidence_stage: CorpusEvidenceStage::CandidateDiscovered,
        known_limitations: &[
            "IMSLP explicitly warns that public-domain status varies by country and does not guarantee legal accuracy",
            "A new critical/urtext edition can have separate rights from the underlying composition",
            "Sound recordings are separately protected",
            "PDF or image scans require OCR/OMR and editorial-error handling; do not treat the scan as canonical symbolic truth",
        ],
        ingest_gate: "Record jurisdiction, composer death/publication, edition/engraver/editor status, source file rights and recording rights for each item",
    },
    MusicCorpusRecord {
        id: "unesco-traditional-music-smithsonian",
        title: "UNESCO Collection of Traditional Music of the World (Smithsonian archive)",
        release: "Archive/recording-specific; collections published 1961–2003",
        upstream_uri: "https://folklife.si.edu/archives-and-resources/unesco-collection-of-traditional-music-of-the-world-records",
        modalities: UNESCO_MODALITIES,
        scope: "In-situ field recordings and scholarly annotations from diverse traditional and sacred music-making contexts across Africa, Asia, Europe, the Americas and Oceania.",
        counts: CorpusScopeCounts {
            items: None,
            performances: None,
            works_or_scores: None,
            composers_or_creators: None,
            audio_hours: None,
        },
        reported_rights: "No blanket ML training or redistribution licence assumed; verify each recording, performer/community, annotation and archive access condition",
        rights_disposition: RightsDisposition::PerItemRightsReviewRequired,
        training_disposition: TrainingDisposition::BlockedPendingRightsReview,
        evidence_stage: CorpusEvidenceStage::CandidateDiscovered,
        known_limitations: &[
            "Audio recordings are not necessarily accompanied by symbolic scores",
            "Scholarly annotations do not imply permission to train or redistribute",
            "Performer, community, sacred/ceremonial and Indigenous cultural protocols may govern access and use",
            "Tradition identity must retain local/period context; avoid flattening it to a generic genre label",
        ],
        ingest_gate: "Obtain archive and rights-holder permissions; document community protocols and permitted analytical/training uses for each recording",
    },
    MusicCorpusRecord {
        id: "vcsl",
        title: "Versilian Community Sample Library",
        release: "Pin exact upstream commit/release and file manifest before local use",
        upstream_uri: "https://versilian-studios.com/vcsl/",
        modalities: VCSL_MODALITIES,
        scope: "CC0 instrument samples spanning orchestral, world and experimental instruments; useful for instrument and renderer qualification, not complete compositions.",
        counts: CorpusScopeCounts {
            items: None,
            performances: None,
            works_or_scores: None,
            composers_or_creators: None,
            audio_hours: None,
        },
        reported_rights: "VCSL declares CC0; exact upstream revision and local asset digests must still be pinned",
        rights_disposition: RightsDisposition::ReportedPermissiveNeedsPinnedRelease,
        training_disposition: TrainingDisposition::AssetUseAfterPinnedSourceVerification,
        evidence_stage: CorpusEvidenceStage::CandidateDiscovered,
        known_limitations: &[
            "Instrument sample bank, not a full composition/performance corpus",
            "A library listing does not prove every advertised sample is present in a local checkout",
            "Coverage varies by articulation, pitch, velocity layer and round-robin",
            "Decoded sample is not proof of idiomatic performance or acoustic fidelity",
        ],
        ingest_gate: "Pin source revision, verify CC0 at that revision, hash every used audio file, and qualify pitch/velocity/articulation render cells separately",
    },
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicCorpusRegistry {
    pub schema: String,
    pub candidate_count: usize,
    pub current_ingest_approved_count: usize,
    pub records: Vec<MusicCorpusRecord>,
    pub registry_sha256: String,
}

pub fn build_music_corpus_registry() -> MusicCorpusRegistry {
    let records = MUSIC_CORPORA.to_vec();
    let mut registry = MusicCorpusRegistry {
        schema: MUSIC_CORPUS_REGISTRY_SCHEMA.into(),
        candidate_count: records.len(),
        current_ingest_approved_count: records
            .iter()
            .filter(|record| record.evidence_stage == CorpusEvidenceStage::IngestApproved)
            .count(),
        records,
        registry_sha256: String::new(),
    };
    registry.registry_sha256 = registry_digest(&registry);
    registry
}

fn registry_digest(registry: &MusicCorpusRegistry) -> String {
    let mut copy = registry.clone();
    copy.registry_sha256.clear();
    let bytes = serde_json::to_vec(&copy).expect("corpus registry DTO must serialize");
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea:muse-music-corpus-registry:v1\n");
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

impl MusicCorpusRegistry {
    pub fn verify_digest(&self) -> bool {
        registry_digest(self) == self.registry_sha256
    }

    /// The registry never implicitly grants ingestion or training permission.
    /// This returns only records that have reached an explicit IngestApproved
    /// stage; candidate records remain blocked regardless of their published
    /// licence summary.
    pub fn approved_for_ingest(&self) -> impl Iterator<Item = &MusicCorpusRecord> {
        self.records
            .iter()
            .filter(|record| record.evidence_stage == CorpusEvidenceStage::IngestApproved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_contains_distinct_score_performance_audio_and_instrument_sources() {
        let registry = build_music_corpus_registry();
        assert_eq!(registry.candidate_count, 7);
        assert!(registry.records.iter().any(|record| record.id == "imslp"));
        assert!(registry.records.iter().any(|record| record.id == "unesco-traditional-music-smithsonian"));
        assert!(registry.records.iter().any(|record| record.id == "vcsl"));
        assert!(registry.records.iter().any(|record| record.id == "maestro"));
        assert!(registry.records.iter().any(|record| record.id == "gangqinpu"));
    }

    #[test]
    fn no_candidate_is_implicitly_approved_for_training() {
        let registry = build_music_corpus_registry();
        assert_eq!(registry.current_ingest_approved_count, 0);
        assert_eq!(registry.approved_for_ingest().count(), 0);
        assert!(registry.records.iter().all(|record| {
            record.evidence_stage == CorpusEvidenceStage::CandidateDiscovered
                && record.training_disposition != TrainingDisposition::AssetUseAfterPinnedSourceVerification
                    || record.id == "vcsl"
        }));
    }

    #[test]
    fn noncommercial_corpora_are_not_marked_as_commercially_usable() {
        let registry = build_music_corpus_registry();
        for id in ["maestro", "periscope"] {
            let record = registry.records.iter().find(|record| record.id == id).unwrap();
            assert_eq!(record.rights_disposition, RightsDisposition::NonCommercialResearchOnly);
            assert_eq!(
                record.training_disposition,
                TrainingDisposition::NonCommercialResearchAfterVerification
            );
        }
    }

    #[test]
    fn registry_digest_is_stable_and_verifiable() {
        let first = build_music_corpus_registry();
        let second = build_music_corpus_registry();
        assert_eq!(first, second);
        assert!(first.verify_digest());
        assert_eq!(first.registry_sha256.len(), 64);
    }
}
