// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Replay-verifiable, topology-specific calibration for EVAL-034 tail guards.
//!
//! A non-zero tail guard is conservative padding, not measured influence. This
//! module only qualifies one when EVAL-017 v0.2 evidence replays, renderer and
//! complete downstream topology are signed and exact, source-composition
//! calibration/qualification splits are disjoint, a distribution-free upper
//! confidence bound supports the requested quantile, the held-out miss bound
//! passes, and a pinned registry authorizes this exact profile/topology.
//!
//! Current EVAL-025 reports explicitly mark downstream routing incomplete.
//! Those reports must not produce a VerifiedTailProfile.

use crate::counterfactual_influence::{
    CounterfactualInfluenceInput, CounterfactualInfluenceSegmentedReport,
    EventIntervention, InfluenceDisposition, InfluenceSegmentationConfig, VerifiedRepeatabilityCalibration,
    verify_counterfactual_influence_segmented,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const TAIL_PROFILE_REGISTRY_VERSION: &str =
    "mel-mr-eval-035-topology-tail-profile-v0.1";
const TOPOLOGY_DOMAIN: &[u8] = b"luminous-dynamics-mel-mr-eval-035-topology-v1\0";
const PROFILE_DOMAIN: &[u8] = b"luminous-dynamics-mel-mr-eval-035-profile-v1\0";
const REGISTRY_DOMAIN: &[u8] = b"luminous-dynamics-mel-mr-eval-035-registry-v1\0";
const PPM: u64 = 1_000_000;
const MAX_LINEAGES: usize = 10_000;
const MAX_SIGNATURE_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TailPcmFormat {
    Pcm16,
    Pcm24,
    Pcm32,
    Float32,
}

/// Signed evidence for the renderer and complete downstream acoustic route.
/// A partial effects-group map is not complete routing evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailTopologyEvidence {
    pub version: String,
    pub publisher_key_id: String,
    pub renderer_identity_sha256: String,
    pub renderer_config_sha256: String,
    pub soundfont_sha256: String,
    pub effects_group_assignment_sha256: String,
    pub effects_settings_sha256: String,
    pub downstream_route_graph_sha256: String,
    pub output_chain_sha256: String,
    pub segmentation_policy_sha256: String,
    pub dataset_revision_sha256: String,
    pub profile_builder_artifact_sha256: String,
    pub operation_family: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub pcm_format: TailPcmFormat,
    pub routing_complete: bool,
    pub evidence_sha256: String,
    pub signature: Vec<u8>,
}

/// The implementation must resolve key_id through a trusted, revocation-aware
/// public-key registry. A caller-supplied key is not a trust root.
pub trait TailEvidenceSignatureVerifier {
    fn verify(&self, key_id: &str, message: &[u8], signature: &[u8]) -> bool;
}

const SOURCE_LINEAGE_DOMAIN: &[u8] =
    b"luminous-dynamics-mel-mr-eval-035-source-lineage-v1\\0";

/// Signed proof that one exact source Score belongs to a declared root
/// composition lineage. Different edits/exports/versions from the same root
/// reuse lineage_root_sha256 and therefore cannot inflate the independent N.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailSourceLineageEvidence {
    pub version: String,
    pub publisher_key_id: String,
    pub lineage_root_sha256: String,
    pub observed_source_score_sha256: String,
    pub provenance_chain_sha256: String,
    pub evidence_sha256: String,
    pub signature: Vec<u8>,
}

fn source_lineage_evidence_digest(
    evidence: &TailSourceLineageEvidence,
) -> Result<String, TailProfileError> {
    let mut unsigned = evidence.clone();
    unsigned.evidence_sha256.clear();
    unsigned.signature.clear();
    let bytes = serde_json::to_vec(&unsigned)
        .map_err(|_| TailProfileError::SerializationFailure)?;
    Ok(hash_domain(SOURCE_LINEAGE_DOMAIN, &bytes))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedTailSourceLineageEvidence {
    lineage_root_sha256: String,
    observed_source_score_sha256: String,
}

impl VerifiedTailSourceLineageEvidence {
    pub fn lineage_root_sha256(&self) -> &str { &self.lineage_root_sha256 }
    pub fn observed_source_score_sha256(&self) -> &str { &self.observed_source_score_sha256 }
}

pub fn verify_tail_source_lineage_evidence(
    evidence: &TailSourceLineageEvidence,
    verifier: &dyn TailEvidenceSignatureVerifier,
) -> Result<VerifiedTailSourceLineageEvidence, TailProfileError> {
    if evidence.version != TAIL_PROFILE_REGISTRY_VERSION
        || evidence.publisher_key_id.trim().is_empty()
        || evidence.publisher_key_id.len() > 256
    {
        return Err(TailProfileError::InvalidEvidence("source_lineage_identity"));
    }
    validate_sha256("lineage_root_sha256", &evidence.lineage_root_sha256)?;
    validate_sha256("observed_source_score_sha256", &evidence.observed_source_score_sha256)?;
    validate_sha256("provenance_chain_sha256", &evidence.provenance_chain_sha256)?;
    if evidence.signature.is_empty() || evidence.signature.len() > MAX_SIGNATURE_BYTES {
        return Err(TailProfileError::InvalidSignature);
    }
    let expected = source_lineage_evidence_digest(evidence)?;
    if expected != evidence.evidence_sha256 {
        return Err(TailProfileError::InvalidEvidence("source_lineage_digest"));
    }
    let mut message = Vec::with_capacity(SOURCE_LINEAGE_DOMAIN.len() + expected.len());
    message.extend_from_slice(SOURCE_LINEAGE_DOMAIN);
    message.extend_from_slice(expected.as_bytes());
    if !verifier.verify(&evidence.publisher_key_id, &message, &evidence.signature) {
        return Err(TailProfileError::InvalidSignature);
    }
    Ok(VerifiedTailSourceLineageEvidence {
        lineage_root_sha256: evidence.lineage_root_sha256.clone(),
        observed_source_score_sha256: evidence.observed_source_score_sha256.clone(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TailProfileError {
    InvalidDigest(&'static str),
    InvalidEvidence(&'static str),
    InvalidSignature,
    IncompleteRouting,
    UnsupportedOperation,
    RendererContextMismatch,
    MissingSourceEvent,
    DuplicateSourceEvent,
    ClippedTailEvidence,
    GuardAlreadyApplied,
    EmptyObservationSet(&'static str),
    TooManyLineages,
    DuplicateEventReceipt,
    DuplicateSourceEventObservation,
    IncompleteSourceEventCoverage,
    SourceScoreEventCountMismatch,
    CalibrationQualificationOverlap,
    TopologyMismatch,
    InsufficientCalibrationLineages,
    InsufficientQualificationLineages,
    InsufficientQuantileConfidence,
    GuardExceedsMaximum,
    InvalidConfig(&'static str),
    SerializationFailure,
    InvalidProfileReceipt,
    ProfileNotQualified,
    ProfileExpired,
    ProfileRevoked,
    PublisherRevoked,
    RegistryDigestMismatch,
    RegistryEntryMissing,
    RegistryEntryAmbiguous,
    ProfileTopologyMismatch,
    ProfileSignatureInvalid,
    InfluenceReplayFailed,
}

impl std::fmt::Display for TailProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EVAL-035 tail-profile validation failed: {self:?}")
    }
}
impl std::error::Error for TailProfileError {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn validate_sha256(field: &'static str, value: &str) -> Result<(), TailProfileError> {
    if is_sha256(value) { Ok(()) } else { Err(TailProfileError::InvalidDigest(field)) }
}

fn hash_domain(domain: &[u8], payload: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(domain);
    h.update((payload.len() as u64).to_le_bytes());
    h.update(payload);
    format!("{:x}", h.finalize())
}

fn topology_evidence_digest(
    evidence: &TailTopologyEvidence,
) -> Result<String, TailProfileError> {
    let mut unsigned = evidence.clone();
    unsigned.evidence_sha256.clear();
    unsigned.signature.clear();
    let bytes = serde_json::to_vec(&unsigned)
        .map_err(|_| TailProfileError::SerializationFailure)?;
    Ok(hash_domain(TOPOLOGY_DOMAIN, &bytes))
}

/// Opaque topology proof. Its constructor verifies content and the publisher's
/// signature. It is never created from EVAL-025's incomplete routing flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedTailTopologyEvidence {
    evidence: TailTopologyEvidence,
    topology_sha256: String,
}
impl VerifiedTailTopologyEvidence {
    pub fn evidence(&self) -> &TailTopologyEvidence { &self.evidence }
    pub fn topology_sha256(&self) -> &str { &self.topology_sha256 }
}

pub fn verify_tail_topology_evidence(
    evidence: &TailTopologyEvidence,
    verifier: &dyn TailEvidenceSignatureVerifier,
) -> Result<VerifiedTailTopologyEvidence, TailProfileError> {
    if evidence.version != TAIL_PROFILE_REGISTRY_VERSION
        || evidence.publisher_key_id.trim().is_empty()
        || evidence.publisher_key_id.len() > 256
        || evidence.sample_rate_hz == 0
        || evidence.channels == 0
        || evidence.channels > 64
        || evidence.operation_family != "remove_event"
    {
        return Err(TailProfileError::InvalidEvidence("topology_identity"));
    }
    if !evidence.routing_complete {
        return Err(TailProfileError::IncompleteRouting);
    }
    for (field, digest) in [
        ("renderer_identity_sha256", &evidence.renderer_identity_sha256),
        ("renderer_config_sha256", &evidence.renderer_config_sha256),
        ("soundfont_sha256", &evidence.soundfont_sha256),
        ("effects_group_assignment_sha256", &evidence.effects_group_assignment_sha256),
        ("effects_settings_sha256", &evidence.effects_settings_sha256),
        ("downstream_route_graph_sha256", &evidence.downstream_route_graph_sha256),
        ("output_chain_sha256", &evidence.output_chain_sha256),
        ("segmentation_policy_sha256", &evidence.segmentation_policy_sha256),
        ("dataset_revision_sha256", &evidence.dataset_revision_sha256),
        ("profile_builder_artifact_sha256", &evidence.profile_builder_artifact_sha256),
    ] {
        validate_sha256(field, digest)?;
    }
    if evidence.signature.is_empty() || evidence.signature.len() > MAX_SIGNATURE_BYTES {
        return Err(TailProfileError::InvalidSignature);
    }
    let expected = topology_evidence_digest(evidence)?;
    if expected != evidence.evidence_sha256 {
        return Err(TailProfileError::InvalidEvidence("topology_digest"));
    }
    let mut message = Vec::with_capacity(TOPOLOGY_DOMAIN.len() + expected.len());
    message.extend_from_slice(TOPOLOGY_DOMAIN);
    message.extend_from_slice(expected.as_bytes());
    if !verifier.verify(&evidence.publisher_key_id, &message, &evidence.signature) {
        return Err(TailProfileError::InvalidSignature);
    }
    Ok(VerifiedTailTopologyEvidence {
        evidence: evidence.clone(),
        topology_sha256: expected,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfileTopology {
    pub topology_publisher_key_id: String,
    pub renderer_identity_sha256: String,
    pub renderer_config_sha256: String,
    pub soundfont_sha256: String,
    pub effects_group_assignment_sha256: String,
    pub effects_settings_sha256: String,
    pub downstream_route_graph_sha256: String,
    pub output_chain_sha256: String,
    pub segmentation_policy_sha256: String,
    pub dataset_revision_sha256: String,
    pub profile_builder_artifact_sha256: String,
    pub operation_family: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub pcm_format: TailPcmFormat,
    pub routing_complete: bool,
}

fn topology_from_evidence(e: &TailTopologyEvidence) -> TailProfileTopology {
    TailProfileTopology {
        topology_publisher_key_id: e.publisher_key_id.clone(),
        renderer_identity_sha256: e.renderer_identity_sha256.clone(),
        renderer_config_sha256: e.renderer_config_sha256.clone(),
        soundfont_sha256: e.soundfont_sha256.clone(),
        effects_group_assignment_sha256: e.effects_group_assignment_sha256.clone(),
        effects_settings_sha256: e.effects_settings_sha256.clone(),
        downstream_route_graph_sha256: e.downstream_route_graph_sha256.clone(),
        output_chain_sha256: e.output_chain_sha256.clone(),
        segmentation_policy_sha256: e.segmentation_policy_sha256.clone(),
        dataset_revision_sha256: e.dataset_revision_sha256.clone(),
        profile_builder_artifact_sha256: e.profile_builder_artifact_sha256.clone(),
        operation_family: e.operation_family.clone(),
        sample_rate_hz: e.sample_rate_hz,
        channels: e.channels,
        pcm_format: e.pcm_format,
        routing_complete: e.routing_complete,
    }
}

fn topology_digest(topology: &TailProfileTopology) -> Result<String, TailProfileError> {
    let bytes = serde_json::to_vec(topology).map_err(|_| TailProfileError::SerializationFailure)?;
    Ok(hash_domain(PROFILE_DOMAIN, &bytes))
}

/// One replay-verified event tail. The note-off frame is read from the baseline
/// renderer receipt's performed-event lineage, never supplied as a free scalar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailEventObservation {
    pub source_lineage_sha256: String,
    pub source_score_sha256: String,
    pub source_score_event_count: usize,
    pub source_event_index: usize,
    pub source_event_id: String,
    pub event_final_note_off_frame: u64,
    pub raw_measured_end_frame: Option<u64>,
    pub hysteresis_support_end_frame: Option<u64>,
    pub measured_tail_frames: u64,
    pub output_frame_count: u64,
    pub segmented_influence_report_sha256: String,
    pub baseline_renderer_receipt_sha256: String,
    pub counterfactual_renderer_receipt_sha256: String,
    pub repeatability_calibration_sha256: String,
    pub topology_sha256: String,
    pub observation_sha256: String,
}

fn observation_digest(o: &TailEventObservation) -> Result<String, TailProfileError> {
    let mut unsigned = o.clone();
    unsigned.observation_sha256.clear();
    let bytes = serde_json::to_vec(&unsigned)
        .map_err(|_| TailProfileError::SerializationFailure)?;
    Ok(hash_domain(PROFILE_DOMAIN, &bytes))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedTailEventObservation {
    observation: TailEventObservation,
    topology: TailProfileTopology,
}
impl VerifiedTailEventObservation {
    pub fn observation(&self) -> &TailEventObservation { &self.observation }
}

/// Replay EVAL-017 v0.2, bind the performed note-off frame, and require an
/// independently verified complete topology proof.
fn parse_score_event_index(value: &str) -> Option<usize> {
    let suffix = value.strip_prefix("score_event_index:")?;
    if suffix.is_empty()
        || (suffix.len() > 1 && suffix.starts_with('0'))
        || !suffix.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    suffix.parse().ok()
}

pub fn verify_tail_event_observation(
    report: &CounterfactualInfluenceSegmentedReport,
    input: &CounterfactualInfluenceInput<'_, '_>,
    calibration: &VerifiedRepeatabilityCalibration,
    topology: &VerifiedTailTopologyEvidence,
    source_lineage: &VerifiedTailSourceLineageEvidence,
) -> Result<VerifiedTailEventObservation, TailProfileError> {
    verify_counterfactual_influence_segmented(report, input, calibration)
        .map_err(|_| TailProfileError::InfluenceReplayFailed)?;
    let evidence = topology.evidence();
    if !evidence.routing_complete { return Err(TailProfileError::IncompleteRouting); }
    if report.segmentation.tail_guard_frames != 0
        || report.segmentation.tail_guard_profile_sha256.is_some()
        || !report.tail_guard_intervals.is_empty()
        || report.tail_guard_was_clipped
    {
        return Err(TailProfileError::ClippedTailEvidence);
    }
    if report.base_report.intervention != EventIntervention::Remove {
        return Err(TailProfileError::UnsupportedOperation);
    }
    if report.base_report.disposition != InfluenceDisposition::EffectDetected {
        return Err(TailProfileError::InvalidEvidence("no_effect_above_repeatability_floor"));
    }
    if report.hysteresis_support_intervals.is_empty() {
        return Err(TailProfileError::InvalidEvidence("missing_hysteresis_support"));
    }
    if report.base_report.source_event_id != input.source_event_id
        || report.base_report.source_score_sha256 != source_lineage.observed_source_score_sha256
        || input.baseline_renderer_evidence.source_score_sha256()
            != source_lineage.observed_source_score_sha256
        || input.baseline_renderer_evidence.renderer_identity_sha256()
            != evidence.renderer_identity_sha256
        || input.baseline_renderer_evidence.renderer_config_sha256()
            != evidence.renderer_config_sha256
        || input.counterfactual_renderer_evidence.renderer_identity_sha256()
            != evidence.renderer_identity_sha256
        || input.counterfactual_renderer_evidence.renderer_config_sha256()
            != evidence.renderer_config_sha256
        || input.counterfactual_renderer_evidence.source_score_sha256()
            != source_lineage.observed_source_score_sha256
        || input.baseline_audio.sample_rate_hz != evidence.sample_rate_hz
        || input.baseline_audio.channels != evidence.channels
        || report.base_report.sample_rate_hz != evidence.sample_rate_hz
        || report.base_report.channels != evidence.channels
    {
        return Err(TailProfileError::RendererContextMismatch);
    }
    if input.expected_score_event_count == 0 {
        return Err(TailProfileError::InvalidEvidence("zero_score_event_count"));
    }
    let source_event_index = parse_score_event_index(input.source_event_id)
        .filter(|index| *index < input.expected_score_event_count)
        .ok_or(TailProfileError::InvalidEvidence("noncanonical_source_event_id"))?;
    let matches: Vec<_> = input.baseline_renderer_evidence.performed_event_lineage()
        .iter()
        .filter(|event| event.source_event_id == input.source_event_id)
        .collect();
    if matches.is_empty() { return Err(TailProfileError::MissingSourceEvent); }
    if matches.len() != 1 { return Err(TailProfileError::DuplicateSourceEvent); }
    let note_off = matches[0].end_frame;
    let raw_end = report.raw_measured_intervals.iter()
        .map(|interval| interval.end_frame as u64).max();
    let hysteresis_end = report.hysteresis_support_intervals.iter()
        .map(|interval| interval.end_frame as u64).max();
    let output_frames = report.base_report.total_frames as u64;
    if note_off > output_frames
        || raw_end.is_some_and(|end| end > output_frames)
        || hysteresis_end.is_some_and(|end| end > output_frames)
    {
        return Err(TailProfileError::InvalidEvidence("frame_horizon"));
    }
    let mut observation = TailEventObservation {
        source_lineage_sha256: source_lineage.lineage_root_sha256.clone(),
        source_score_sha256: source_lineage.observed_source_score_sha256.clone(),
        source_score_event_count: input.expected_score_event_count,
        source_event_index,
        source_event_id: input.source_event_id.to_owned(),
        event_final_note_off_frame: note_off,
        raw_measured_end_frame: raw_end,
        hysteresis_support_end_frame: hysteresis_end,
        measured_tail_frames: hysteresis_end.unwrap_or(note_off).saturating_sub(note_off),
        output_frame_count: output_frames,
        segmented_influence_report_sha256: report.receipt_sha256.clone(),
        baseline_renderer_receipt_sha256: input.baseline_renderer_evidence.receipt_sha256().to_owned(),
        counterfactual_renderer_receipt_sha256: input.counterfactual_renderer_evidence.receipt_sha256().to_owned(),
        repeatability_calibration_sha256: report.base_report.repeatability_calibration_sha256.clone(),
        topology_sha256: topology.topology_sha256().to_owned(),
        observation_sha256: String::new(),
    };
    observation.observation_sha256 = observation_digest(&observation)?;
    Ok(VerifiedTailEventObservation {
        observation,
        topology: topology_from_evidence(evidence),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfileBuildConfig {
    pub quantile_ppm: u32,
    pub confidence_ppm: u32,
    pub safety_margin_frames: u64,
    pub maximum_guard_frames: u64,
    pub minimum_calibration_lineages: usize,
    pub minimum_qualification_lineages: usize,
    pub maximum_qualification_miss_rate_ppm: u32,
}
impl Default for TailProfileBuildConfig {
    fn default() -> Self {
        Self {
            quantile_ppm: 950_000,
            confidence_ppm: 950_000,
            safety_margin_frames: 0,
            maximum_guard_frames: 48_000,
            minimum_calibration_lineages: 59,
            minimum_qualification_lineages: 59,
            maximum_qualification_miss_rate_ppm: 50_000,
        }
    }
}
impl TailProfileBuildConfig {
    fn validate(&self) -> Result<(), TailProfileError> {
        if !(1..1_000_000).contains(&self.quantile_ppm)
            || !(1..1_000_000).contains(&self.confidence_ppm)
            || self.maximum_guard_frames == 0
            || self.minimum_calibration_lineages < 2
            || self.minimum_qualification_lineages < 2
            || self.minimum_calibration_lineages > MAX_LINEAGES
            || self.minimum_qualification_lineages > MAX_LINEAGES
            || self.maximum_qualification_miss_rate_ppm >= 1_000_000
        {
            return Err(TailProfileError::InvalidConfig("profile_config"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TailProfileDisposition { Qualified, Unqualified }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfile {
    pub version: String,
    pub topology: TailProfileTopology,
    pub topology_sha256: String,
    pub config: TailProfileBuildConfig,
    pub calibration_lineage_sha256s: Vec<String>,
    pub qualification_lineage_sha256s: Vec<String>,
    pub calibration_observation_sha256s: Vec<String>,
    pub qualification_observation_sha256s: Vec<String>,
    /// One maximum tail per independent composition lineage, ordered by lineage digest.
    pub calibration_cluster_tail_frames: Vec<u64>,
    pub qualification_cluster_tail_frames: Vec<u64>,
    pub calibration_quantile_rank_one_based: usize,
    pub calibration_quantile_observed_frames: u64,
    pub calibration_quantile_upper_confidence_frames: u64,
    pub safety_margin_frames: u64,
    pub guard_frames: u64,
    pub qualification_miss_count: usize,
    pub qualification_miss_rate_upper_bound_ppm: u32,
    pub disposition: TailProfileDisposition,
    pub reasons: Vec<String>,
    pub profile_sha256: String,
}

fn profile_digest(profile: &TailProfile) -> Result<String, TailProfileError> {
    let mut unsigned = profile.clone();
    unsigned.profile_sha256.clear();
    let bytes = serde_json::to_vec(&unsigned)
        .map_err(|_| TailProfileError::SerializationFailure)?;
    Ok(hash_domain(PROFILE_DOMAIN, &bytes))
}

fn cluster_maxima(
    observations: &[VerifiedTailEventObservation],
    expected_topology: &TailProfileTopology,
) -> Result<(Vec<String>, Vec<String>, Vec<u64>), TailProfileError> {
    if observations.is_empty() { return Err(TailProfileError::EmptyObservationSet("tail_observations")); }
    if observations.len() > MAX_LINEAGES * 64 { return Err(TailProfileError::TooManyLineages); }
    let mut event_receipts = BTreeSet::new();
    let mut lineage_maxima = BTreeMap::<String, u64>::new();
    let mut score_coverage = BTreeMap::<(String, String), (usize, BTreeSet<usize>)>::new();
    let mut receipts = Vec::with_capacity(observations.len());
    for verified in observations {
        let o = &verified.observation;
        if verified.topology != *expected_topology { return Err(TailProfileError::TopologyMismatch); }
        validate_sha256("source_lineage_sha256", &o.source_lineage_sha256)?;
        validate_sha256("source_score_sha256", &o.source_score_sha256)?;
        validate_sha256("observation_sha256", &o.observation_sha256)?;
        validate_sha256("segmented_influence_report_sha256", &o.segmented_influence_report_sha256)?;
        validate_sha256("baseline_renderer_receipt_sha256", &o.baseline_renderer_receipt_sha256)?;
        validate_sha256("counterfactual_renderer_receipt_sha256", &o.counterfactual_renderer_receipt_sha256)?;
        validate_sha256("repeatability_calibration_sha256", &o.repeatability_calibration_sha256)?;
        if o.source_score_event_count == 0
            || o.source_event_index >= o.source_score_event_count
            || parse_score_event_index(&o.source_event_id) != Some(o.source_event_index)
        {
            return Err(TailProfileError::InvalidEvidence("source_event_index"));
        }
        if o.observation_sha256 != observation_digest(o)? {
            return Err(TailProfileError::InvalidEvidence("event_observation_digest"));
        }
        if o.measured_tail_frames > o.output_frame_count {
            return Err(TailProfileError::InvalidEvidence("tail_exceeds_horizon"));
        }
        if !event_receipts.insert(o.observation_sha256.clone()) {
            return Err(TailProfileError::DuplicateEventReceipt);
        }
        let score_key = (o.source_lineage_sha256.clone(), o.source_score_sha256.clone());
        let coverage = score_coverage.entry(score_key).or_insert_with(|| {
            (o.source_score_event_count, BTreeSet::new())
        });
        if coverage.0 != o.source_score_event_count {
            return Err(TailProfileError::SourceScoreEventCountMismatch);
        }
        if !coverage.1.insert(o.source_event_index) {
            return Err(TailProfileError::DuplicateSourceEventObservation);
        }
        let current = lineage_maxima.entry(o.source_lineage_sha256.clone()).or_insert(0);
        *current = (*current).max(o.measured_tail_frames);
        receipts.push(o.observation_sha256.clone());
    }
    for (expected_count, observed_indices) in score_coverage.values() {
        if observed_indices.len() != *expected_count
            || (0..*expected_count).any(|index| !observed_indices.contains(&index))
        {
            return Err(TailProfileError::IncompleteSourceEventCoverage);
        }
    }
    if lineage_maxima.len() > MAX_LINEAGES { return Err(TailProfileError::TooManyLineages); }
    let lineages: Vec<String> = lineage_maxima.keys().cloned().collect();
    let maxima: Vec<u64> = lineage_maxima.values().copied().collect();
    receipts.sort();
    Ok((lineages, receipts, maxima))
}

pub fn build_tail_profile(
    calibration: &[VerifiedTailEventObservation],
    qualification: &[VerifiedTailEventObservation],
    config: &TailProfileBuildConfig,
) -> Result<TailProfile, TailProfileError> {
    config.validate()?;
    let topology = calibration.first().or_else(|| qualification.first())
        .ok_or(TailProfileError::EmptyObservationSet("both_splits"))?
        .topology.clone();
    if !topology.routing_complete { return Err(TailProfileError::IncompleteRouting); }
    let topology_sha256 = topology_digest(&topology)?;
    let (cal_lineages, cal_receipts, cal_tail_by_lineage) =
        cluster_maxima(calibration, &topology)?;
    let (qual_lineages, qual_receipts, qual_tail_by_lineage) =
        cluster_maxima(qualification, &topology)?;

    let calibration_set: BTreeSet<&str> = cal_lineages.iter().map(String::as_str).collect();
    if qual_lineages.iter().any(|lineage| calibration_set.contains(lineage.as_str())) {
        return Err(TailProfileError::CalibrationQualificationOverlap);
    }
    if cal_lineages.len() < config.minimum_calibration_lineages {
        return Err(TailProfileError::InsufficientCalibrationLineages);
    }
    if qual_lineages.len() < config.minimum_qualification_lineages {
        return Err(TailProfileError::InsufficientQualificationLineages);
    }

    let mut cal_sorted = cal_tail_by_lineage.clone();
    cal_sorted.sort_unstable();
    let rank = upper_order_statistic_rank(cal_sorted.len(), config.quantile_ppm, config.confidence_ppm)
        .ok_or(TailProfileError::InsufficientQuantileConfidence)?;
    let quantile_observed = cal_sorted[rank - 1];
    // The selected order statistic is the one-sided, distribution-free UCB
    // for the requested population quantile at the configured confidence.
    let quantile_upper = quantile_observed;
    let guard_frames = quantile_upper.checked_add(config.safety_margin_frames)
        .ok_or(TailProfileError::GuardExceedsMaximum)?;
    if guard_frames > config.maximum_guard_frames {
        return Err(TailProfileError::GuardExceedsMaximum);
    }

    let miss_count = qual_tail_by_lineage.iter().filter(|tail| **tail > guard_frames).count();
    let alpha_ppm = 1_000_000u32 - config.confidence_ppm;
    let miss_upper = clopper_pearson_upper_ppm(miss_count, qual_tail_by_lineage.len(), alpha_ppm)?;
    let mut reasons = Vec::new();
    let disposition = if miss_upper <= config.maximum_qualification_miss_rate_ppm {
        TailProfileDisposition::Qualified
    } else {
        reasons.push(format!(
            "held-out miss-rate upper bound {miss_upper} ppm exceeds limit {} ppm",
            config.maximum_qualification_miss_rate_ppm
        ));
        TailProfileDisposition::Unqualified
    };

    let mut profile = TailProfile {
        version: TAIL_PROFILE_REGISTRY_VERSION.to_owned(),
        topology,
        topology_sha256,
        config: config.clone(),
        calibration_lineage_sha256s: cal_lineages,
        qualification_lineage_sha256s: qual_lineages,
        calibration_observation_sha256s: cal_receipts,
        qualification_observation_sha256s: qual_receipts,
        calibration_cluster_tail_frames: cal_tail_by_lineage,
        qualification_cluster_tail_frames: qual_tail_by_lineage,
        calibration_quantile_rank_one_based: rank,
        calibration_quantile_observed_frames: quantile_observed,
        calibration_quantile_upper_confidence_frames: quantile_upper,
        safety_margin_frames: config.safety_margin_frames,
        guard_frames,
        qualification_miss_count: miss_count,
        qualification_miss_rate_upper_bound_ppm: miss_upper,
        disposition,
        reasons,
        profile_sha256: String::new(),
    };
    profile.profile_sha256 = profile_digest(&profile)?;
    Ok(profile)
}

pub fn verify_tail_profile(
    supplied: &TailProfile,
    calibration: &[VerifiedTailEventObservation],
    qualification: &[VerifiedTailEventObservation],
) -> Result<bool, TailProfileError> {
    if supplied.version != TAIL_PROFILE_REGISTRY_VERSION
        || !is_sha256(&supplied.profile_sha256)
        || profile_digest(supplied)? != supplied.profile_sha256
    {
        return Err(TailProfileError::InvalidProfileReceipt);
    }
    let expected = build_tail_profile(calibration, qualification, &supplied.config)?;
    Ok(expected == *supplied)
}

/// Smallest one-based order-statistic rank k for which
/// P(Binomial(n, q) <= k-1) >= confidence. None means the sample cannot support
/// the requested distribution-free quantile/confidence pair.
pub fn upper_order_statistic_rank(
    n: usize,
    quantile_ppm: u32,
    confidence_ppm: u32,
) -> Option<usize> {
    if n == 0 || n > MAX_LINEAGES
        || !(1..1_000_000).contains(&quantile_ppm)
        || !(1..1_000_000).contains(&confidence_ppm)
    { return None; }
    let q = quantile_ppm as f64 / PPM as f64;
    let confidence = confidence_ppm as f64 / PPM as f64;
    let log_fact = log_factorials(n);
    let (mut low, mut high, mut found) = (1usize, n, None);
    while low <= high {
        let mid = low + (high - low) / 2;
        if binomial_cdf_at_most(n, mid - 1, q, &log_fact) >= confidence {
            found = Some(mid);
            if mid == 1 { break; }
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    found
}

/// Exact-binomial one-sided Clopper–Pearson upper bound, conservatively rounded
/// upward to integer ppm. Trials are independent composition lineages.
pub fn clopper_pearson_upper_ppm(
    failures: usize,
    trials: usize,
    alpha_ppm: u32,
) -> Result<u32, TailProfileError> {
    if trials == 0 || trials > MAX_LINEAGES || failures > trials
        || !(1..1_000_000).contains(&alpha_ppm)
    { return Err(TailProfileError::InvalidConfig("binomial_bound_inputs")); }
    if failures == trials { return Ok(1_000_000); }
    let alpha = alpha_ppm as f64 / PPM as f64;
    if failures == 0 {
        let upper = 1.0 - alpha.powf(1.0 / trials as f64);
        return Ok((upper * PPM as f64).ceil().clamp(0.0, PPM as f64) as u32);
    }
    let log_fact = log_factorials(trials);
    let (mut low, mut high) = (0.0f64, 1.0f64);
    for _ in 0..80 {
        let mid = (low + high) * 0.5;
        if binomial_cdf_at_most(trials, failures, mid, &log_fact) > alpha {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok((high * PPM as f64).ceil().clamp(0.0, PPM as f64) as u32)
}

fn log_factorials(n: usize) -> Vec<f64> {
    let mut values = vec![0.0; n + 1];
    for i in 1..=n { values[i] = values[i - 1] + (i as f64).ln(); }
    values
}

fn binomial_cdf_at_most(n: usize, k: usize, p: f64, log_fact: &[f64]) -> f64 {
    if k >= n { return 1.0; }
    if p <= 0.0 { return 1.0; }
    if p >= 1.0 { return 0.0; }
    let (ln_p, ln_q) = (p.ln(), (-p).ln_1p());
    let mut logs = Vec::with_capacity(k + 1);
    let mut max_log = f64::NEG_INFINITY;
    for i in 0..=k {
        let choose = log_fact[n] - log_fact[i] - log_fact[n - i];
        let log_prob = choose + i as f64 * ln_p + (n - i) as f64 * ln_q;
        max_log = max_log.max(log_prob);
        logs.push(log_prob);
    }
    let sum: f64 = logs.iter().map(|v| (*v - max_log).exp()).sum();
    (max_log + sum.ln()).exp().clamp(0.0, 1.0)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedTailProfile {
    pub profile: TailProfile,
    pub publisher_key_id: String,
    pub issued_at_unix_seconds: u64,
    pub expires_at_unix_seconds: u64,
    pub signature: Vec<u8>,
}

fn signed_profile_message(signed: &SignedTailProfile) -> Result<Vec<u8>, TailProfileError> {
    let metadata = serde_json::to_vec(&(
        signed.profile.profile_sha256.as_str(),
        signed.publisher_key_id.as_str(),
        signed.issued_at_unix_seconds,
        signed.expires_at_unix_seconds,
    ))
    .map_err(|_| TailProfileError::SerializationFailure)?;
    let mut message = Vec::with_capacity(PROFILE_DOMAIN.len() + metadata.len());
    message.extend_from_slice(PROFILE_DOMAIN);
    message.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    message.extend_from_slice(&metadata);
    Ok(message)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfileRegistryEntry {
    pub profile_sha256: String,
    pub topology_sha256: String,
    pub publisher_key_id: String,
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
    pub revoked_at_unix_seconds: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfileRegistrySnapshot {
    pub version: String,
    pub created_at_unix_seconds: u64,
    pub expires_at_unix_seconds: u64,
    pub entries: Vec<TailProfileRegistryEntry>,
    pub revoked_publisher_key_ids: Vec<String>,
    pub registry_sha256: String,
}

fn registry_digest(registry: &TailProfileRegistrySnapshot) -> Result<String, TailProfileError> {
    let mut unsigned = registry.clone();
    unsigned.registry_sha256.clear();
    let bytes = serde_json::to_vec(&unsigned)
        .map_err(|_| TailProfileError::SerializationFailure)?;
    Ok(hash_domain(REGISTRY_DOMAIN, &bytes))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedTailProfile {
    profile_sha256: String,
    topology_sha256: String,
    topology: TailProfileTopology,
    guard_frames: u64,
}
impl VerifiedTailProfile {
    pub fn profile_sha256(&self) -> &str { &self.profile_sha256 }
    pub fn guard_frames(&self) -> u64 { self.guard_frames }
    pub fn topology_sha256(&self) -> &str { &self.topology_sha256 }

    pub fn apply_to_segmentation_config(
        &self,
        base: &InfluenceSegmentationConfig,
        renderer_identity_sha256: &str,
        renderer_config_sha256: &str,
        sample_rate_hz: u32,
        channels: usize,
    ) -> Result<InfluenceSegmentationConfig, TailProfileError> {
        if base.tail_guard_frames != 0 || base.tail_guard_profile_sha256.is_some() {
            return Err(TailProfileError::GuardAlreadyApplied);
        }
        if renderer_identity_sha256 != self.topology.renderer_identity_sha256
            || renderer_config_sha256 != self.topology.renderer_config_sha256
            || sample_rate_hz != self.topology.sample_rate_hz
            || channels != self.topology.channels
        {
            return Err(TailProfileError::ProfileTopologyMismatch);
        }
        let mut config = base.clone();
        config.tail_guard_frames = usize::try_from(self.guard_frames)
            .map_err(|_| TailProfileError::GuardExceedsMaximum)?;
        config.tail_guard_profile_sha256 = Some(self.profile_sha256.clone());
        Ok(config)
    }
}

/// Resolve a signed profile through a digest-pinned registry snapshot. The
/// snapshot digest is the trust anchor; the profile signature is checked by
/// the caller's trusted cryptographic verifier. Registry updates/revocations
/// create a new snapshot and never rewrite historical reports.
pub fn resolve_verified_tail_profile(
    signed: &SignedTailProfile,
    registry: &TailProfileRegistrySnapshot,
    expected_registry_sha256: &str,
    expected_topology: &TailProfileTopology,
    now_unix_seconds: u64,
    verifier: &dyn TailEvidenceSignatureVerifier,
) -> Result<VerifiedTailProfile, TailProfileError> {
    validate_sha256("expected_registry_sha256", expected_registry_sha256)?;
    if registry.version != TAIL_PROFILE_REGISTRY_VERSION
        || !is_sha256(&registry.registry_sha256)
        || registry_digest(registry)? != registry.registry_sha256
        || registry.registry_sha256 != expected_registry_sha256
    {
        return Err(TailProfileError::RegistryDigestMismatch);
    }
    if now_unix_seconds < registry.created_at_unix_seconds
        || now_unix_seconds > registry.expires_at_unix_seconds
        || now_unix_seconds < signed.issued_at_unix_seconds
        || now_unix_seconds > signed.expires_at_unix_seconds
    {
        return Err(TailProfileError::ProfileExpired);
    }
    let profile = &signed.profile;
    if profile.version != TAIL_PROFILE_REGISTRY_VERSION
        || profile.disposition != TailProfileDisposition::Qualified
        || !is_sha256(&profile.profile_sha256)
        || profile_digest(profile)? != profile.profile_sha256
    {
        return Err(TailProfileError::ProfileNotQualified);
    }
    if profile.topology != *expected_topology
        || !profile.topology.routing_complete
        || topology_digest(&profile.topology)? != profile.topology_sha256
    {
        return Err(TailProfileError::ProfileTopologyMismatch);
    }
    if signed.publisher_key_id.trim().is_empty()
        || signed.publisher_key_id.len() > 256
        || signed.signature.is_empty()
        || signed.signature.len() > MAX_SIGNATURE_BYTES
    {
        return Err(TailProfileError::ProfileSignatureInvalid);
    }
    if registry.revoked_publisher_key_ids.iter().any(|key| key == &signed.publisher_key_id) {
        return Err(TailProfileError::PublisherRevoked);
    }
    let matches: Vec<&TailProfileRegistryEntry> = registry.entries.iter()
        .filter(|entry| entry.profile_sha256 == profile.profile_sha256)
        .collect();
    if matches.is_empty() { return Err(TailProfileError::RegistryEntryMissing); }
    if matches.len() != 1 { return Err(TailProfileError::RegistryEntryAmbiguous); }
    let entry = matches[0];
    if entry.topology_sha256 != profile.topology_sha256
        || entry.publisher_key_id != signed.publisher_key_id
        || now_unix_seconds < entry.valid_from_unix_seconds
        || now_unix_seconds > entry.valid_until_unix_seconds
    {
        return Err(TailProfileError::ProfileTopologyMismatch);
    }
    if entry.revoked_at_unix_seconds.is_some_and(|time| time <= now_unix_seconds) {
        return Err(TailProfileError::ProfileRevoked);
    }
    if signed.issued_at_unix_seconds > signed.expires_at_unix_seconds {
        return Err(TailProfileError::ProfileExpired);
    }
    let message = signed_profile_message(signed)?;
    if !verifier.verify(&signed.publisher_key_id, &message, &signed.signature) {
        return Err(TailProfileError::ProfileSignatureInvalid);
    }
    Ok(VerifiedTailProfile {
        profile_sha256: profile.profile_sha256.clone(),
        topology_sha256: profile.topology_sha256.clone(),
        topology: profile.topology.clone(),
        guard_frames: profile.guard_frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestSignatureVerifier;
    impl TailEvidenceSignatureVerifier for TestSignatureVerifier {
        fn verify(&self, key_id: &str, message: &[u8], signature: &[u8]) -> bool {
            key_id == "test-key" && signature == Sha256::digest(message).as_slice()
        }
    }

    fn digest(label: &str) -> String { format!("{:x}", Sha256::digest(label.as_bytes())) }

    fn topology(routing_complete: bool) -> TailTopologyEvidence {
        let mut e = TailTopologyEvidence {
            version: TAIL_PROFILE_REGISTRY_VERSION.to_owned(),
            publisher_key_id: "test-key".to_owned(),
            renderer_identity_sha256: digest("renderer-identity"),
            renderer_config_sha256: digest("renderer-config"),
            soundfont_sha256: digest("soundfont"),
            effects_group_assignment_sha256: digest("effects-assignment"),
            effects_settings_sha256: digest("effects-settings"),
            downstream_route_graph_sha256: digest("downstream-route"),
            output_chain_sha256: digest("output-chain"),
            segmentation_policy_sha256: digest("segmentation-policy"),
            dataset_revision_sha256: digest("dataset-revision"),
            profile_builder_artifact_sha256: digest("profile-builder"),
            operation_family: "remove_event".to_owned(),
            sample_rate_hz: 48_000,
            channels: 2,
            pcm_format: TailPcmFormat::Float32,
            routing_complete,
            evidence_sha256: String::new(),
            signature: Vec::new(),
        };
        e.evidence_sha256 = topology_evidence_digest(&e).unwrap();
        let mut message = Vec::new();
        message.extend_from_slice(TOPOLOGY_DOMAIN);
        message.extend_from_slice(e.evidence_sha256.as_bytes());
        e.signature = Sha256::digest(message).to_vec();
        e
    }


    fn signed_source_lineage(
        lineage_root: &str,
        source_score: &str,
    ) -> TailSourceLineageEvidence {
        let mut evidence = TailSourceLineageEvidence {
            version: TAIL_PROFILE_REGISTRY_VERSION.to_owned(),
            publisher_key_id: "test-key".to_owned(),
            lineage_root_sha256: digest(lineage_root),
            observed_source_score_sha256: digest(source_score),
            provenance_chain_sha256: digest("provenance-chain"),
            evidence_sha256: String::new(),
            signature: Vec::new(),
        };
        evidence.evidence_sha256 = source_lineage_evidence_digest(&evidence).unwrap();
        let mut message = Vec::new();
        message.extend_from_slice(SOURCE_LINEAGE_DOMAIN);
        message.extend_from_slice(evidence.evidence_sha256.as_bytes());
        evidence.signature = Sha256::digest(message).to_vec();
        evidence
    }

    fn test_observation(
        lineage: &str,
        event: &str,
        tail: u64,
        topology: &VerifiedTailTopologyEvidence,
    ) -> VerifiedTailEventObservation {
        let mut o = TailEventObservation {
            source_lineage_sha256: digest(lineage),
            source_score_sha256: digest("source-score"),
            source_score_event_count: 2,
            source_event_index: parse_score_event_index(event).unwrap(),
            source_event_id: event.to_owned(),
            event_final_note_off_frame: 1_000,
            raw_measured_end_frame: Some(1_000 + tail),
            hysteresis_support_end_frame: Some(1_000 + tail),
            measured_tail_frames: tail,
            output_frame_count: 10_000,
            segmented_influence_report_sha256: digest(&format!("segmented-{lineage}-{event}")),
            baseline_renderer_receipt_sha256: digest(&format!("base-{lineage}-{event}")),
            counterfactual_renderer_receipt_sha256: digest(&format!("counter-{lineage}-{event}")),
            repeatability_calibration_sha256: digest("repeatability"),
            topology_sha256: topology.topology_sha256().to_owned(),
            observation_sha256: String::new(),
        };
        o.observation_sha256 = observation_digest(&o).unwrap();
        VerifiedTailEventObservation {
            observation: o,
            topology: topology_from_evidence(topology.evidence()),
        }
    }

    fn splits(
        topology: &VerifiedTailTopologyEvidence,
        qualification_max: u64,
    ) -> (Vec<VerifiedTailEventObservation>, Vec<VerifiedTailEventObservation>) {
        let mut cal = Vec::new();
        let mut qual = Vec::new();
        for i in 0..59 {
            cal.push(test_observation(&format!("cal-{i}"), "score_event_index:0", (i % 7) as u64, topology));
            cal.push(test_observation(&format!("cal-{i}"), "score_event_index:1", (i % 7 + 2) as u64, topology));
            qual.push(test_observation(&format!("qual-{i}"), "score_event_index:0", qualification_max, topology));
            qual.push(test_observation(&format!("qual-{i}"), "score_event_index:1", qualification_max, topology));
        }
        (cal, qual)
    }

    fn config() -> TailProfileBuildConfig {
        TailProfileBuildConfig {
            quantile_ppm: 950_000,
            confidence_ppm: 950_000,
            safety_margin_frames: 32,
            maximum_guard_frames: 10_000,
            minimum_calibration_lineages: 59,
            minimum_qualification_lineages: 59,
            maximum_qualification_miss_rate_ppm: 50_000,
        }
    }


    #[test]
    fn source_lineage_requires_a_signature_and_binds_the_observed_score() {
        let verifier = TestSignatureVerifier;
        let evidence = signed_source_lineage("composition-root", "source-score");
        let verified = verify_tail_source_lineage_evidence(&evidence, &verifier).unwrap();
        assert_eq!(verified.lineage_root_sha256(), evidence.lineage_root_sha256);
        assert_eq!(verified.observed_source_score_sha256(), evidence.observed_source_score_sha256);

        let mut tampered = evidence.clone();
        tampered.lineage_root_sha256 = digest("another-composition");
        assert!(verify_tail_source_lineage_evidence(&tampered, &verifier).is_err());

        let mut bad_signature = evidence;
        bad_signature.signature[0] ^= 0x01;
        assert_eq!(
            verify_tail_source_lineage_evidence(&bad_signature, &verifier).unwrap_err(),
            TailProfileError::InvalidSignature
        );
    }

    #[test]
    fn topology_evidence_requires_complete_route_and_valid_signature() {
        let verifier = TestSignatureVerifier;
        assert_eq!(
            verify_tail_topology_evidence(&topology(false), &verifier).unwrap_err(),
            TailProfileError::IncompleteRouting
        );
        let mut tampered = topology(true);
        tampered.downstream_route_graph_sha256 = digest("different-route");
        assert!(verify_tail_topology_evidence(&tampered, &verifier).is_err());
    }

    #[test]
    fn distribution_free_q95_ucb_requires_59_lineages_at_95_percent() {
        assert_eq!(upper_order_statistic_rank(58, 950_000, 950_000), None);
        assert_eq!(upper_order_statistic_rank(59, 950_000, 950_000), Some(59));
    }

    #[test]
    fn zero_misses_in_59_qualification_lineages_meets_five_percent_ucb() {
        assert!(clopper_pearson_upper_ppm(0, 59, 50_000).unwrap() <= 50_000);
        assert!(clopper_pearson_upper_ppm(1, 59, 50_000).unwrap() > 50_000);
    }

    #[test]
    fn event_rows_are_clustered_to_one_worst_tail_per_composition() {
        let topology = verify_tail_topology_evidence(&topology(true), &TestSignatureVerifier).unwrap();
        let (cal, qual) = splits(&topology, 20);
        let profile = build_tail_profile(&cal, &qual, &config()).unwrap();
        assert_eq!(profile.calibration_lineage_sha256s.len(), 59);
        assert_eq!(profile.calibration_cluster_tail_frames.len(), 59);
        assert!(profile.guard_frames >= 8 + 32);
    }


    #[test]
    fn each_composition_score_requires_complete_unique_event_coverage() {
        let topology = verify_tail_topology_evidence(&topology(true), &TestSignatureVerifier).unwrap();
        let (cal, qual) = splits(&topology, 20);
        let mut missing = cal.clone();
        missing.remove(0);
        assert_eq!(
            build_tail_profile(&missing, &qual, &config()).unwrap_err(),
            TailProfileError::IncompleteSourceEventCoverage
        );
        let mut duplicate = cal.clone();
        duplicate.push(cal[0].clone());
        assert_eq!(
            build_tail_profile(&duplicate, &qual, &config()).unwrap_err(),
            TailProfileError::DuplicateSourceEventObservation
        );
    }

    #[test]
    fn calibration_and_qualification_lineage_overlap_rejects() {
        let topology = verify_tail_topology_evidence(&topology(true), &TestSignatureVerifier).unwrap();
        let (cal, mut qual) = splits(&topology, 20);
        qual[0] = cal[0].clone();
        assert_eq!(
            build_tail_profile(&cal, &qual, &config()).unwrap_err(),
            TailProfileError::CalibrationQualificationOverlap
        );
    }

    #[test]
    fn held_out_tail_misses_prevent_profile_promotion() {
        let topology = verify_tail_topology_evidence(&topology(true), &TestSignatureVerifier).unwrap();
        let (cal, qual) = splits(&topology, 100_000);
        let profile = build_tail_profile(&cal, &qual, &config()).unwrap();
        assert_eq!(profile.disposition, TailProfileDisposition::Unqualified);
        assert!(profile.qualification_miss_count > 0);
    }

    #[test]
    fn profile_receipt_replays_and_rejects_mutation() {
        let topology = verify_tail_topology_evidence(&topology(true), &TestSignatureVerifier).unwrap();
        let (cal, qual) = splits(&topology, 20);
        let profile = build_tail_profile(&cal, &qual, &config()).unwrap();
        assert!(verify_tail_profile(&profile, &cal, &qual).unwrap());
        let mut tampered = profile.clone();
        tampered.guard_frames += 1;
        assert_eq!(verify_tail_profile(&tampered, &cal, &qual).unwrap_err(), TailProfileError::InvalidProfileReceipt);
    }

    #[test]
    fn profile_registry_checks_signature_revocation_and_pinned_snapshot() {
        let verifier = TestSignatureVerifier;
        let topology = verify_tail_topology_evidence(&topology(true), &verifier).unwrap();
        let (cal, qual) = splits(&topology, 20);
        let profile = build_tail_profile(&cal, &qual, &config()).unwrap();
        let mut signed = SignedTailProfile {
            profile: profile.clone(),
            publisher_key_id: "test-key".to_owned(),
            issued_at_unix_seconds: 100,
            expires_at_unix_seconds: 500,
            signature: Vec::new(),
        };
        signed.signature = Sha256::digest(signed_profile_message(&signed).unwrap()).to_vec();
        let mut registry = TailProfileRegistrySnapshot {
            version: TAIL_PROFILE_REGISTRY_VERSION.to_owned(),
            created_at_unix_seconds: 90,
            expires_at_unix_seconds: 600,
            entries: vec![TailProfileRegistryEntry {
                profile_sha256: profile.profile_sha256.clone(),
                topology_sha256: profile.topology_sha256.clone(),
                publisher_key_id: "test-key".to_owned(),
                valid_from_unix_seconds: 100,
                valid_until_unix_seconds: 500,
                revoked_at_unix_seconds: None,
            }],
            revoked_publisher_key_ids: Vec::new(),
            registry_sha256: String::new(),
        };
        registry.registry_sha256 = registry_digest(&registry).unwrap();
        let verified = resolve_verified_tail_profile(
            &signed, &registry, &registry.registry_sha256, &profile.topology, 200, &verifier
        ).unwrap();
        assert_eq!(verified.guard_frames(), profile.guard_frames);
        let mut validity_tamper = signed.clone();
        validity_tamper.expires_at_unix_seconds += 10_000;
        assert_eq!(
            resolve_verified_tail_profile(
                &validity_tamper,
                &registry,
                &registry.registry_sha256,
                &profile.topology,
                200,
                &verifier,
            ).unwrap_err(),
            TailProfileError::ProfileSignatureInvalid
        );
        let mut revoked = registry.clone();
        revoked.revoked_publisher_key_ids.push("test-key".to_owned());
        revoked.registry_sha256 = registry_digest(&revoked).unwrap();
        assert_eq!(
            resolve_verified_tail_profile(
                &signed, &revoked, &revoked.registry_sha256, &profile.topology, 200, &verifier
            ).unwrap_err(),
            TailProfileError::PublisherRevoked
        );
    }

    #[test]
    fn resolved_profile_applies_only_to_exact_unguarded_renderer_context() {
        let verifier = TestSignatureVerifier;
        let topology = verify_tail_topology_evidence(&topology(true), &verifier).unwrap();
        let (cal, qual) = splits(&topology, 20);
        let profile = build_tail_profile(&cal, &qual, &config()).unwrap();
        let mut signed = SignedTailProfile {
            profile: profile.clone(),
            publisher_key_id: "test-key".to_owned(),
            issued_at_unix_seconds: 100,
            expires_at_unix_seconds: 500,
            signature: Vec::new(),
        };
        signed.signature = Sha256::digest(signed_profile_message(&signed).unwrap()).to_vec();
        let mut registry = TailProfileRegistrySnapshot {
            version: TAIL_PROFILE_REGISTRY_VERSION.to_owned(),
            created_at_unix_seconds: 90,
            expires_at_unix_seconds: 600,
            entries: vec![TailProfileRegistryEntry {
                profile_sha256: profile.profile_sha256.clone(),
                topology_sha256: profile.topology_sha256.clone(),
                publisher_key_id: "test-key".to_owned(),
                valid_from_unix_seconds: 100,
                valid_until_unix_seconds: 500,
                revoked_at_unix_seconds: None,
            }],
            revoked_publisher_key_ids: Vec::new(),
            registry_sha256: String::new(),
        };
        registry.registry_sha256 = registry_digest(&registry).unwrap();
        let verified = resolve_verified_tail_profile(
            &signed, &registry, &registry.registry_sha256, &profile.topology, 200, &verifier
        ).unwrap();
        let resolved = verified.apply_to_segmentation_config(
            &InfluenceSegmentationConfig::default(),
            &profile.topology.renderer_identity_sha256,
            &profile.topology.renderer_config_sha256,
            profile.topology.sample_rate_hz,
            profile.topology.channels,
        ).unwrap();
        assert_eq!(resolved.tail_guard_frames as u64, profile.guard_frames);
        assert_eq!(resolved.tail_guard_profile_sha256.as_deref(), Some(profile.profile_sha256.as_str()));
        assert!(verified.apply_to_segmentation_config(
            &InfluenceSegmentationConfig::default(),
            &digest("different-renderer"),
            &profile.topology.renderer_config_sha256,
            profile.topology.sample_rate_hz,
            profile.topology.channels,
        ).is_err());
    }
}
