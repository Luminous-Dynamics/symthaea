// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Signed, topology-specific tail-profile construction and resolution.
//!
//! A profile is usable only when it was built from a replay-verified EVAL-025
//! adapter report, a signed complete renderer-topology attestation, disjoint
//! calibration/qualification composition lineages, and a distribution-free
//! upper tolerance bound. The current EVAL-025 adapter deliberately reports
//! incomplete routing, so it cannot yet mint a production tail profile.
//!
//! The registry keeps content integrity, signer identity, qualification and
//! revocation distinct. A profile digest alone is not an authority or a proof
//! that its renderer domain matches the current render.

use crate::benchmark_lineage::{BenchmarkLineageAudit, BenchmarkLineageManifest, DatasetSplit, verify_benchmark_lineage_audit};
use crate::counterfactual_influence::InfluenceSegmentationConfig;
use crate::counterfactual_support_adapter::{
    CounterfactualSupportAdapterReport, VerifiedCounterfactualSupportAdapterReport,
};
use crate::fluid_render::{FluidSynthEffectsGroupAssignment, verify_effects_group_assignment};
use crate::renderer_execution_receipt::DecodedSampleFormat;
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const TOPOLOGY_DOMAIN: &[u8] = b"luminous-dynamics-muse-renderer-topology-attestation-v1\0";
const PROFILE_DOMAIN: &[u8] = b"luminous-dynamics-muse-tail-profile-v1\0";
const LINEAGE_SET_DOMAIN: &[u8] = b"luminous-dynamics-muse-tail-lineage-set-v1\0";
const BUILDER_DOMAIN: &[u8] = b"luminous-dynamics-muse-tail-profile-builder-v1\0";
const PROFILE_VERSION: &str = "muse-tail-profile-v1";
const TOPOLOGY_VERSION: &str = "muse-renderer-topology-attestation-v1";
const MAX_PROFILE_EVENTS: usize = 1_000_000;
const MAX_PROFILE_LINEAGES: usize = 100_000;
const MAX_GUARD_FRAMES: u64 = 10_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TailProfileSplit {
    Calibration,
    Qualification,
}

/// One rendered source composition in the calibration or qualification
/// partition. Its lineage and split are derived from the EVAL-031 audited
/// benchmark manifest, never hand-entered by the profile caller.
#[derive(Debug, Clone)]
pub struct TailProfileSourceBundle<'a> {
    pub benchmark_sample_id: String,
    pub adapter_report: &'a VerifiedCounterfactualSupportAdapterReport,
}

/// A publisher-attested description of the actual renderer topology used by
/// the event-support receipts. A signature binds the declaration to a trusted
/// authority; it does not replace independent inspection of the renderer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RendererTopologyAttestation {
    pub version: String,
    pub authority_key_id: String,
    pub renderer_identity_sha256: String,
    pub renderer_executable_sha256: String,
    pub soundfont_sha256: String,
    pub renderer_config_sha256: String,
    pub render_config_sha256: String,
    pub effects_group_assignment: FluidSynthEffectsGroupAssignment,
    pub effect_settings_sha256: String,
    pub reverb_settings_sha256: String,
    pub chorus_settings_sha256: String,
    pub route_graph_sha256: String,
    pub route_graph_complete: bool,
    pub output_chain_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub bits_per_sample: u16,
    pub sample_format: DecodedSampleFormat,
    /// Ed25519 signature over the domain-separated attestation digest.
    pub signature_ed25519_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfileAuthority {
    pub key_id: String,
    pub public_key_ed25519_hex: String,
}

#[derive(Debug, Clone)]
pub struct VerifiedRendererTopology {
    attestation: RendererTopologyAttestation,
    attestation_sha256: String,
}

impl VerifiedRendererTopology {
    pub fn attestation(&self) -> &RendererTopologyAttestation {
        &self.attestation
    }

    pub fn attestation_sha256(&self) -> &str {
        &self.attestation_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfileContext {
    pub renderer_identity_sha256: String,
    pub renderer_executable_sha256: String,
    pub soundfont_sha256: String,
    pub renderer_config_sha256: String,
    pub render_config_sha256: String,
    pub effects_group_assignment: FluidSynthEffectsGroupAssignment,
    pub effect_settings_sha256: String,
    pub reverb_settings_sha256: String,
    pub chorus_settings_sha256: String,
    pub route_graph_sha256: String,
    pub output_chain_sha256: String,
    pub topology_attestation_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub bits_per_sample: u16,
    pub sample_format: DecodedSampleFormat,
    pub segmentation_config_sha256: String,
    pub operation_family: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfileBuildSpec {
    pub profile_id: String,
    pub dataset_revision: String,
    /// Desired tail-duration quantile in millionths (990_000 = 99th percentile).
    pub quantile_micros: u32,
    /// One-sided confidence requirement in millionths (950_000 = 95%).
    pub confidence_micros: u32,
    /// Additional conservative padding, kept separate from the statistical bound.
    pub safety_margin_frames: u64,
    /// Minimum independent qualification lineages; calibration minimum is
    /// derived from the selected quantile/confidence requirement.
    pub minimum_qualification_lineages: usize,
    /// Number of consecutive all-channel quiet windows required after the final influence window.
    pub minimum_trailing_quiet_windows: usize,
    pub builder_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfileEventObservation {
    pub source_event_index: usize,
    pub source_event_id: String,
    pub source_lineage_sha256: String,
    pub split: TailProfileSplit,
    pub event_evidence_sha256: String,
    pub final_note_off_frame: u64,
    pub raw_measured_end_frame: Option<u64>,
    pub hysteresis_support_end_frame: Option<u64>,
    pub observed_tail_frames: u64,
    pub trailing_quiet_windows: usize,
    pub total_windows: usize,
    pub output_frame_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfileStatistics {
    pub selected_quantile_micros: u32,
    pub required_confidence_micros: u32,
    pub independent_lineage_count: usize,
    pub empirical_quantile_frames: u64,
    pub empirical_raw_tail_quantile_frames: u64,
    pub empirical_hysteresis_tail_quantile_frames: u64,
    /// Distribution-free upper confidence bound: the maximum independent
    /// lineage-cluster tail duration, provided its order-statistic coverage
    /// meets the requested confidence.
    pub upper_confidence_bound_frames: u64,
    pub confidence_coverage_lower_bound_micros: u32,
    pub maximum_observed_tail_frames: u64,
    pub safety_margin_frames: u64,
    pub guard_frames: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfileQualification {
    pub independent_lineage_count: usize,
    pub missed_lineage_count: usize,
    pub maximum_held_out_tail_frames: u64,
    /// One-sided 95% upper bound on miss probability, defined only for zero
    /// misses. When misses occur, qualification fails and this is None.
    pub zero_miss_upper_rate_95_micros: Option<u32>,
    pub qualified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfileLineageSummary {
    pub source_lineage_sha256: String,
    pub split: TailProfileSplit,
    pub event_count: usize,
    pub maximum_raw_tail_frames: u64,
    pub maximum_hysteresis_tail_frames: u64,
    pub maximum_observed_tail_frames: u64,
    pub minimum_trailing_quiet_windows: usize,
    pub minimum_total_windows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfile {
    pub version: String,
    pub profile_id: String,
    pub dataset_revision: String,
    pub benchmark_manifest_sha256: String,
    pub benchmark_audit_sha256: String,
    pub context: TailProfileContext,
    pub builder_spec: TailProfileBuildSpec,
    pub event_observations: Vec<TailProfileEventObservation>,
    pub lineage_summaries: Vec<TailProfileLineageSummary>,
    pub statistics: TailProfileStatistics,
    pub qualification: TailProfileQualification,
    pub calibration_lineage_set_sha256: String,
    pub qualification_lineage_set_sha256: String,
    pub event_evidence_set_sha256: String,
    pub builder_spec_sha256: String,
    pub signer_key_id: String,
    pub profile_sha256: String,
    pub signature_ed25519_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TailProfileError {
    InvalidField(&'static str),
    InvalidDigest(&'static str),
    InvalidAuthority,
    InvalidSignature,
    IncompleteRouteGraph,
    InvalidEffectsGroupAssignment,
    RendererContextMismatch,
    AdapterReportInvalid,
    TailGuardWasClipped,
    EventCountMismatch,
    InvalidLineageAssignment,
    SourceLineageMismatch,
    UnsupportedOperationFamily,
    CrossSplitLineageLeakage,
    InsufficientCalibrationLineages,
    InsufficientQualificationLineages,
    QualificationTailMiss,
    GuardOverflow,
    DuplicateAuthority,
    DuplicateProfile,
    UnknownProfile,
    RevokedProfile,
    ProfileNotQualified,
    ProfileContextMismatch,
    ProfileReceiptInvalid,
    SerializationFailure,
}

impl std::fmt::Display for TailProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tail profile operation failed: {self:?}")
    }
}

impl std::error::Error for TailProfileError {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_label(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= 256
        && !value.chars().any(char::is_control)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn hex_decode<const N: usize>(value: &str) -> Option<[u8; N]> {
    if value.len() != N * 2 {
        return None;
    }
    let bytes = value.as_bytes();
    let mut output = [0_u8; N];
    for (index, target) in output.iter_mut().enumerate() {
        let high = decode_hex_nibble(bytes[index * 2])?;
        let low = decode_hex_nibble(bytes[index * 2 + 1])?;
        *target = (high << 4) | low;
    }
    Some(output)
}

fn decode_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn hash_component(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn topology_digest(
    attestation: &RendererTopologyAttestation,
) -> Result<String, TailProfileError> {
    let mut unsigned = attestation.clone();
    unsigned.signature_ed25519_hex.clear();
    let bytes = serde_json::to_vec(&unsigned)
        .map_err(|_| TailProfileError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(TOPOLOGY_DOMAIN);
    hash_component(&mut hasher, &bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn profile_digest(profile: &TailProfile) -> Result<String, TailProfileError> {
    let mut unsigned = profile.clone();
    unsigned.profile_sha256.clear();
    unsigned.signature_ed25519_hex.clear();
    let bytes = serde_json::to_vec(&unsigned)
        .map_err(|_| TailProfileError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(PROFILE_DOMAIN);
    hash_component(&mut hasher, &bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn lineage_set_digest(domain: &[u8], lineages: &BTreeSet<String>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((lineages.len() as u64).to_le_bytes());
    for lineage in lineages {
        hash_component(&mut hasher, lineage.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn event_set_digest(observations: &[TailProfileEventObservation]) -> String {
    let mut ordered = observations.to_vec();
    ordered.sort_by(|left, right| {
        (left.source_event_index, &left.source_lineage_sha256, &left.event_evidence_sha256)
            .cmp(&(right.source_event_index, &right.source_lineage_sha256, &right.event_evidence_sha256))
    });
    let bytes = serde_json::to_vec(&ordered).expect("serializing fixed event observations cannot fail");
    sha256_hex(&bytes)
}

fn builder_spec_digest(spec: &TailProfileBuildSpec) -> Result<String, TailProfileError> {
    let bytes = serde_json::to_vec(spec).map_err(|_| TailProfileError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(BUILDER_DOMAIN);
    hash_component(&mut hasher, &bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn verify_signature(
    public_key_hex: &str,
    message: &[u8],
    signature_hex: &str,
) -> Result<(), TailProfileError> {
    let public_key_bytes = hex_decode::<32>(public_key_hex).ok_or(TailProfileError::InvalidAuthority)?;
    let signature_bytes = hex_decode::<64>(signature_hex).ok_or(TailProfileError::InvalidSignature)?;
    let public_key = VerifyingKey::from_bytes(&public_key_bytes)
        .map_err(|_| TailProfileError::InvalidAuthority)?;
    let signature = Signature::from_bytes(&signature_bytes);
    public_key
        .verify_strict(message, &signature)
        .map_err(|_| TailProfileError::InvalidSignature)
}

/// Sign a complete renderer-topology declaration. Production callers should
/// only sign after independently inspecting the exact FluidSynth build,
/// SoundFont, effects settings, master/output path, plugins and buses.
pub fn sign_renderer_topology_attestation(
    attestation: &mut RendererTopologyAttestation,
    signing_key: &SigningKey,
) -> Result<(), TailProfileError> {
    if attestation.version != TOPOLOGY_VERSION || !attestation.route_graph_complete {
        return Err(TailProfileError::IncompleteRouteGraph);
    }
    attestation.signature_ed25519_hex.clear();
    let digest = topology_digest(attestation)?;
    attestation.signature_ed25519_hex = hex_encode(signing_key.sign(digest.as_bytes()).to_bytes().as_ref());
    Ok(())
}

/// Verify the signed topology against a trusted authority and mint an opaque
/// token used by the tail-profile builder.
pub fn verify_renderer_topology_attestation(
    attestation: &RendererTopologyAttestation,
    authority: &TailProfileAuthority,
) -> Result<VerifiedRendererTopology, TailProfileError> {
    if attestation.version != TOPOLOGY_VERSION
        || attestation.authority_key_id != authority.key_id
        || !attestation.route_graph_complete
    {
        return Err(TailProfileError::IncompleteRouteGraph);
    }
    if !valid_label(&authority.key_id)
        || !is_sha256(&attestation.renderer_identity_sha256)
        || !is_sha256(&attestation.renderer_executable_sha256)
        || !is_sha256(&attestation.soundfont_sha256)
        || !is_sha256(&attestation.renderer_config_sha256)
        || !is_sha256(&attestation.render_config_sha256)
        || !is_sha256(&attestation.effect_settings_sha256)
        || !is_sha256(&attestation.reverb_settings_sha256)
        || !is_sha256(&attestation.chorus_settings_sha256)
        || !is_sha256(&attestation.route_graph_sha256)
        || !is_sha256(&attestation.output_chain_sha256)
        || attestation.sample_rate_hz == 0
        || attestation.channels == 0
        || attestation.bits_per_sample == 0
        || !verify_effects_group_assignment(&attestation.effects_group_assignment)
    {
        return Err(TailProfileError::InvalidDigest("renderer topology"));
    }
    let digest = topology_digest(attestation)?;
    verify_signature(
        &authority.public_key_ed25519_hex,
        digest.as_bytes(),
        &attestation.signature_ed25519_hex,
    )?;
    Ok(VerifiedRendererTopology {
        attestation: attestation.clone(),
        attestation_sha256: digest,
    })
}

#[derive(Debug, Clone, Copy)]
struct LineageTail {
    raw_tail_frames: u64,
    hysteresis_tail_frames: u64,
    observed_tail_frames: u64,
}

fn distribution_free_upper_bound_coverage(
    quantile_micros: u32,
    independent_lineages: usize,
) -> Result<u32, TailProfileError> {
    if quantile_micros == 0 || quantile_micros >= 1_000_000 || independent_lineages == 0 {
        return Err(TailProfileError::InvalidField("quantile or lineage count"));
    }
    let q = quantile_micros as f64 / 1_000_000.0;
    let coverage = 1.0 - q.powf(independent_lineages as f64);
    Ok((coverage * 1_000_000.0).floor().clamp(0.0, 1_000_000.0) as u32)
}

/// Minimum number of independent lineage clusters for the sample maximum to
/// be a one-sided distribution-free upper confidence bound on the requested
/// quantile. For q=.99 and confidence=.95, this returns 299.
pub fn minimum_lineages_for_upper_quantile_bound(
    quantile_micros: u32,
    confidence_micros: u32,
) -> Result<usize, TailProfileError> {
    if quantile_micros == 0
        || quantile_micros >= 1_000_000
        || confidence_micros == 0
        || confidence_micros >= 1_000_000
    {
        return Err(TailProfileError::InvalidField("quantile/confidence"));
    }
    let q = quantile_micros as f64 / 1_000_000.0;
    let confidence = confidence_micros as f64 / 1_000_000.0;
    let raw = ((1.0 - confidence).ln() / q.ln()).ceil();
    if !raw.is_finite() || raw < 1.0 || raw > MAX_PROFILE_LINEAGES as f64 {
        return Err(TailProfileError::InsufficientCalibrationLineages);
    }
    let n = raw as usize;
    if distribution_free_upper_bound_coverage(quantile_micros, n)? < confidence_micros {
        return n.checked_add(1).ok_or(TailProfileError::InsufficientCalibrationLineages);
    }
    Ok(n)
}

fn nearest_rank_quantile(values: &[u64], quantile_micros: u32) -> Result<u64, TailProfileError> {
    if values.is_empty() || quantile_micros == 0 || quantile_micros >= 1_000_000 {
        return Err(TailProfileError::InsufficientCalibrationLineages);
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = ((quantile_micros as u128 * sorted.len() as u128) + 999_999) / 1_000_000;
    let index = usize::try_from(rank.saturating_sub(1))
        .map_err(|_| TailProfileError::GuardOverflow)?;
    sorted.get(index).copied().ok_or(TailProfileError::GuardOverflow)
}

fn operation_family_from_report(
    report: &CounterfactualSupportAdapterReport,
) -> Result<String, TailProfileError> {
    let mut families = BTreeSet::new();
    for event in &report.events {
        let family = match &event.intervention {
            crate::counterfactual_influence::EventIntervention::Remove => "remove",
            _ => return Err(TailProfileError::InvalidField("unsupported intervention family")),
        };
        families.insert(family.to_owned());
    }
    if families.len() != 1 {
        return Err(TailProfileError::InvalidField("mixed operation families"));
    }
    families.into_iter().next().ok_or(TailProfileError::InvalidField("operation family"))
}

fn observations_from_verified_report(
    verified_report: &VerifiedCounterfactualSupportAdapterReport,
    topology: &VerifiedRendererTopology,
    source_lineage_sha256: &str,
    split: TailProfileSplit,
) -> Result<(TailProfileContext, Vec<TailProfileEventObservation>), TailProfileError> {
    let report = verified_report.report();
    if !report.receipt_is_intact() {
        return Err(TailProfileError::AdapterReportInvalid);
    }
    if report.tail_guard_was_clipped {
        return Err(TailProfileError::TailGuardWasClipped);
    }
    let attestation = &topology.attestation;
    if report.renderer_identity_sha256 != attestation.renderer_identity_sha256
        || report.renderer_executable_sha256 != attestation.renderer_executable_sha256
        || report.soundfont_sha256 != attestation.soundfont_sha256
        || report.renderer_config_sha256 != attestation.renderer_config_sha256
        || report.render_config_sha256 != attestation.render_config_sha256
        || report.effects_group_assignment != attestation.effects_group_assignment
        || report.sample_rate_hz != attestation.sample_rate_hz
        || report.channels != attestation.channels
        || report.output_bits_per_sample != attestation.bits_per_sample
        || report.output_sample_format != attestation.sample_format
        || report.route_graph_sha256.as_ref().is_some_and(|digest| digest != &attestation.route_graph_sha256)
        || !attestation.route_graph_complete
    {
        return Err(TailProfileError::RendererContextMismatch);
    }
    if !is_sha256(source_lineage_sha256) || report.event_count == 0 {
        return Err(TailProfileError::InvalidLineageAssignment);
    }
    if report.event_count > MAX_PROFILE_EVENTS {
        return Err(TailProfileError::InvalidField("too many profile events"));
    }

    let operation_family = operation_family_from_report(report)?;
    let segmentation_digest = report.events[0].segmentation_config_sha256.clone();
    if !is_sha256(&segmentation_digest)
        || report.events.iter().any(|event| event.segmentation_config_sha256 != segmentation_digest)
    {
        return Err(TailProfileError::RendererContextMismatch);
    }

    let mut observations = Vec::with_capacity(report.event_count);
    for (index, event) in report.events.iter().enumerate() {
        if event.source_event_index != index
            || !is_sha256(&event.evidence_sha256)
            || event.tail_guard_was_clipped
            || event.final_note_off_frame > event.output_frame_count
        {
            return Err(TailProfileError::AdapterReportInvalid);
        }
        let raw_end = event.raw_measured_intervals.iter().map(|interval| interval.end_frame).max();
        let hysteresis_end = event.hysteresis_support_intervals.iter().map(|interval| interval.end_frame).max();
        let channels = report.channels;
        if channels == 0 || report.window_segmentation.is_empty()
            || report.window_segmentation.len() % channels != 0
        {
            return Err(TailProfileError::TailWindowCensored);
        }
        let mut quiet_by_window = BTreeMap::<usize, (usize, bool)>::new();
        for window in &report.window_segmentation {
            if window.channel >= channels {
                return Err(TailProfileError::TailWindowCensored);
            }
            let entry = quiet_by_window.entry(window.window_index).or_insert((0, true));
            entry.0 += 1;
            entry.1 &= !window.onset_triggered
                && !window.release_triggered
                && !window.hysteresis_active;
        }
        let total_windows = quiet_by_window.len();
        if total_windows == 0
            || quiet_by_window.keys().copied().ne(0..total_windows)
            || quiet_by_window.values().any(|(channel_count, _)| *channel_count != channels)
        {
            return Err(TailProfileError::TailWindowCensored);
        }
        let mut trailing_quiet_windows = 0usize;
        for window_index in (0..total_windows).rev() {
            if quiet_by_window.get(&window_index).is_some_and(|(_, quiet)| *quiet) {
                trailing_quiet_windows += 1;
            } else {
                break;
            }
        }
        let raw_tail = raw_end
            .map(|end| end.saturating_sub(event.final_note_off_frame))
            .unwrap_or(0);
        let hysteresis_tail = hysteresis_end
            .map(|end| end.saturating_sub(event.final_note_off_frame))
            .unwrap_or(0);
        observations.push(TailProfileEventObservation {
            source_event_index: index,
            source_event_id: event.source_event_id.clone(),
            source_lineage_sha256: source_lineage_sha256.to_owned(),
            split,
            event_evidence_sha256: event.evidence_sha256.clone(),
            final_note_off_frame: event.final_note_off_frame,
            raw_measured_end_frame: raw_end,
            hysteresis_support_end_frame: hysteresis_end,
            observed_tail_frames: hysteresis_tail.max(raw_tail),
            trailing_quiet_windows,
            total_windows,
            output_frame_count: event.output_frame_count,
        });
    }

    let context = TailProfileContext {
        renderer_identity_sha256: report.renderer_identity_sha256.clone(),
        renderer_executable_sha256: report.renderer_executable_sha256.clone(),
        soundfont_sha256: report.soundfont_sha256.clone(),
        renderer_config_sha256: report.renderer_config_sha256.clone(),
        render_config_sha256: report.render_config_sha256.clone(),
        effects_group_assignment: report.effects_group_assignment.clone(),
        effect_settings_sha256: attestation.effect_settings_sha256.clone(),
        reverb_settings_sha256: attestation.reverb_settings_sha256.clone(),
        chorus_settings_sha256: attestation.chorus_settings_sha256.clone(),
        route_graph_sha256: attestation.route_graph_sha256.clone(),
        output_chain_sha256: attestation.output_chain_sha256.clone(),
        topology_attestation_sha256: topology.attestation_sha256.clone(),
        sample_rate_hz: report.sample_rate_hz,
        channels: report.channels,
        bits_per_sample: report.output_bits_per_sample,
        sample_format: report.output_sample_format,
        segmentation_config_sha256: segmentation_digest,
        operation_family,
    };
    Ok((context, observations))
}

fn aggregate_lineage_tails(
    observations: &[TailProfileEventObservation],
    split: TailProfileSplit,
) -> Result<BTreeMap<String, LineageTail>, TailProfileError> {
    let mut grouped = BTreeMap::<String, LineageTail>::new();
    for observation in observations.iter().filter(|observation| observation.split == split) {
        let entry = grouped
            .entry(observation.source_lineage_sha256.clone())
            .or_insert(LineageTail {
                raw_tail_frames: 0,
                hysteresis_tail_frames: 0,
                observed_tail_frames: 0,
            });
        let raw = observation.raw_measured_end_frame
            .map(|end| end.saturating_sub(observation.final_note_off_frame))
            .unwrap_or(0);
        let hysteresis = observation.hysteresis_support_end_frame
            .map(|end| end.saturating_sub(observation.final_note_off_frame))
            .unwrap_or(0);
        entry.raw_tail_frames = entry.raw_tail_frames.max(raw);
        entry.hysteresis_tail_frames = entry.hysteresis_tail_frames.max(hysteresis);
        entry.observed_tail_frames = entry.observed_tail_frames.max(raw.max(hysteresis));
    }
    if grouped.len() > MAX_PROFILE_LINEAGES {
        return Err(TailProfileError::InvalidField("too many lineage clusters"));
    }
    Ok(grouped)
}

fn lineage_summaries(
    observations: &[TailProfileEventObservation],
) -> Result<Vec<TailProfileLineageSummary>, TailProfileError> {
    let mut grouped = BTreeMap::<(TailProfileSplit, String), (usize, u64, u64, u64, usize, usize)>::new();
    for observation in observations {
        if !is_sha256(&observation.source_lineage_sha256)
            || !is_sha256(&observation.event_evidence_sha256)
            || observation.source_event_id.trim().is_empty()
            || observation.output_frame_count == 0
            || observation.final_note_off_frame > observation.output_frame_count
            || observation.total_windows == 0
            || observation.trailing_quiet_windows > observation.total_windows
            || observation.raw_measured_end_frame.is_some_and(|end| end > observation.output_frame_count)
            || observation.hysteresis_support_end_frame.is_some_and(|end| end > observation.output_frame_count)
        {
            return Err(TailProfileError::AdapterReportInvalid);
        }
        if observation.split == TailProfileSplit::Test {
            return Err(TailProfileError::InvalidLineageAssignment);
        }
        let entry = grouped
            .entry((observation.split, observation.source_lineage_sha256.clone()))
            .or_insert((0, 0, 0, 0, usize::MAX, usize::MAX));
        entry.0 += 1;
        let raw = observation.raw_measured_end_frame
            .map(|end| end.saturating_sub(observation.final_note_off_frame))
            .unwrap_or(0);
        let hysteresis = observation.hysteresis_support_end_frame
            .map(|end| end.saturating_sub(observation.final_note_off_frame))
            .unwrap_or(0);
        if observation.observed_tail_frames != raw.max(hysteresis) {
            return Err(TailProfileError::AdapterReportInvalid);
        }
        entry.1 = entry.1.max(raw);
        entry.2 = entry.2.max(hysteresis);
        entry.3 = entry.3.max(raw.max(hysteresis));
        entry.4 = entry.4.min(observation.trailing_quiet_windows);
        entry.5 = entry.5.min(observation.total_windows);
    }
    let mut summaries = grouped
        .into_iter()
        .map(|((split, lineage), (event_count, raw, hysteresis, observed, trailing_quiet, total_windows))| {
            TailProfileLineageSummary {
                source_lineage_sha256: lineage,
                split,
                event_count,
                maximum_raw_tail_frames: raw,
                maximum_hysteresis_tail_frames: hysteresis,
                maximum_observed_tail_frames: observed,
                minimum_trailing_quiet_windows: trailing_quiet,
                minimum_total_windows: total_windows,
            }
        })
        .collect::<Vec<_>>();
    summaries.sort_by(|left, right| {
        (left.split, &left.source_lineage_sha256)
            .cmp(&(right.split, &right.source_lineage_sha256))
    });
    if summaries.len() > MAX_PROFILE_LINEAGES * 2 {
        return Err(TailProfileError::InvalidField("too many lineage summaries"));
    }
    Ok(summaries)
}

fn zero_miss_upper_rate_95_micros(n: usize) -> Option<u32> {
    if n == 0 {
        return None;
    }
    let upper = 1.0 - 0.05_f64.powf(1.0 / n as f64);
    Some((upper * 1_000_000.0).ceil().clamp(0.0, 1_000_000.0) as u32)
}

fn create_profile_unsigned(
    spec: &TailProfileBuildSpec,
    context: TailProfileContext,
    observations: &[TailProfileEventObservation],
    benchmark_manifest_sha256: &str,
    benchmark_audit_sha256: &str,
    signer_key_id: &str,
) -> Result<TailProfile, TailProfileError> {
    if !valid_label(&spec.profile_id)
        || !valid_label(&spec.dataset_revision)
        || !valid_label(&spec.builder_version)
        || !valid_label(signer_key_id)
        || spec.minimum_qualification_lineages == 0
        || spec.minimum_trailing_quiet_windows == 0
        || spec.safety_margin_frames > MAX_GUARD_FRAMES
    {
        return Err(TailProfileError::InvalidField("profile builder specification"));
    }
    if observations.is_empty() {
        return Err(TailProfileError::EventCountMismatch);
    }
    if observations.iter().any(|observation| observation.trailing_quiet_windows < spec.minimum_trailing_quiet_windows) {
        return Err(TailProfileError::TailWindowCensored);
    }
    if !is_sha256(benchmark_manifest_sha256) || !is_sha256(benchmark_audit_sha256) {
        return Err(TailProfileError::InvalidDigest("benchmark lineage audit"));
    }
    let required_lineages = minimum_lineages_for_upper_quantile_bound(
        spec.quantile_micros,
        spec.confidence_micros,
    )?;
    let calibration = aggregate_lineage_tails(observations, TailProfileSplit::Calibration)?;
    let qualification = aggregate_lineage_tails(observations, TailProfileSplit::Qualification)?;
    if calibration.len() < required_lineages {
        return Err(TailProfileError::InsufficientCalibrationLineages);
    }
    if qualification.len() < spec.minimum_qualification_lineages {
        return Err(TailProfileError::InsufficientQualificationLineages);
    }

    let calibration_values: Vec<u64> = calibration
        .values()
        .map(|tail| tail.observed_tail_frames)
        .collect();
    let raw_calibration_values: Vec<u64> = calibration
        .values()
        .map(|tail| tail.raw_tail_frames)
        .collect();
    let hysteresis_calibration_values: Vec<u64> = calibration
        .values()
        .map(|tail| tail.hysteresis_tail_frames)
        .collect();
    let empirical_quantile_frames = nearest_rank_quantile(&calibration_values, spec.quantile_micros)?;
    let maximum_observed_tail_frames = calibration_values.iter().copied().max().unwrap_or(0);
    let coverage = distribution_free_upper_bound_coverage(spec.quantile_micros, calibration.len())?;
    if coverage < spec.confidence_micros {
        return Err(TailProfileError::InsufficientCalibrationLineages);
    }
    let guard_frames = maximum_observed_tail_frames
        .checked_add(spec.safety_margin_frames)
        .filter(|value| *value <= MAX_GUARD_FRAMES)
        .ok_or(TailProfileError::GuardOverflow)?;

    let qualification_values: Vec<u64> = qualification
        .values()
        .map(|tail| tail.observed_tail_frames)
        .collect();
    let maximum_held_out_tail_frames = qualification_values.iter().copied().max().unwrap_or(0);
    let missed_lineage_count = qualification_values
        .iter()
        .filter(|tail| **tail > guard_frames)
        .count();
    let qualified = missed_lineage_count == 0;
    let calibration_lineages: BTreeSet<String> = calibration.keys().cloned().collect();
    let qualification_lineages: BTreeSet<String> = qualification.keys().cloned().collect();
    if !calibration_lineages.is_disjoint(&qualification_lineages) {
        return Err(TailProfileError::CrossSplitLineageLeakage);
    }
    let calibration_lineage_set_sha256 =
        lineage_set_digest(LINEAGE_SET_DOMAIN, &calibration_lineages);
    let mut qual_domain = LINEAGE_SET_DOMAIN.to_vec();
    qual_domain.extend_from_slice(b":qualification");
    let qualification_lineage_set_sha256 =
        lineage_set_digest(&qual_domain, &qualification_lineages);

    let builder_spec_sha256 = builder_spec_digest(spec)?;
    let event_evidence_set_sha256 = event_set_digest(observations);
    let statistics = TailProfileStatistics {
        selected_quantile_micros: spec.quantile_micros,
        required_confidence_micros: spec.confidence_micros,
        independent_lineage_count: calibration.len(),
        empirical_quantile_frames,
        empirical_raw_tail_quantile_frames: nearest_rank_quantile(&raw_calibration_values, spec.quantile_micros)?,
        empirical_hysteresis_tail_quantile_frames: nearest_rank_quantile(&hysteresis_calibration_values, spec.quantile_micros)?,
        upper_confidence_bound_frames: maximum_observed_tail_frames,
        confidence_coverage_lower_bound_micros: coverage,
        maximum_observed_tail_frames,
        safety_margin_frames: spec.safety_margin_frames,
        guard_frames,
    };
    let qualification = TailProfileQualification {
        independent_lineage_count: qualification.len(),
        missed_lineage_count,
        maximum_held_out_tail_frames,
        zero_miss_upper_rate_95_micros: if missed_lineage_count == 0 {
            zero_miss_upper_rate_95_micros(qualification_values.len())
        } else {
            None
        },
        qualified,
    };

    Ok(TailProfile {
        version: PROFILE_VERSION.to_owned(),
        profile_id: spec.profile_id.clone(),
        dataset_revision: spec.dataset_revision.clone(),
        benchmark_manifest_sha256: benchmark_manifest_sha256.to_owned(),
        benchmark_audit_sha256: benchmark_audit_sha256.to_owned(),
        context,
        builder_spec: spec.clone(),
        event_observations: observations.to_vec(),
        lineage_summaries: lineage_summaries(observations)?,
        statistics,
        qualification,
        calibration_lineage_set_sha256,
        qualification_lineage_set_sha256,
        event_evidence_set_sha256,
        builder_spec_sha256,
        signer_key_id: signer_key_id.to_owned(),
        profile_sha256: String::new(),
        signature_ed25519_hex: String::new(),
    })
}

/// Build a tail profile exclusively from an opaque, replay-verified adapter
/// token and a separately signed topology declaration. The signature is made
/// only after held-out qualification reports zero misses.
pub fn build_and_sign_tail_profile(
    source_bundles: &[TailProfileSourceBundle<'_>],
    benchmark_manifest: &BenchmarkLineageManifest,
    benchmark_audit: &BenchmarkLineageAudit,
    topology: &VerifiedRendererTopology,
    spec: &TailProfileBuildSpec,
    signer_key_id: &str,
    signing_key: &SigningKey,
) -> Result<TailProfile, TailProfileError> {
    if benchmark_manifest.dataset_revision != spec.dataset_revision
        || !benchmark_audit.is_clean()
        || !verify_benchmark_lineage_audit(benchmark_manifest, benchmark_audit)
        || source_bundles.is_empty()
    {
        return Err(TailProfileError::InvalidLineageAssignment);
    }

    // The benchmark manifest is authoritative for source composition lineage
    // and partition. Test samples are deliberately unavailable to profile fitting.
    let mut records_by_sample = BTreeMap::new();
    for record in &benchmark_manifest.records {
        records_by_sample.insert(record.sample_id.as_str(), record);
    }

    let mut seen_samples = BTreeSet::new();
    let mut context: Option<TailProfileContext> = None;
    let mut observations = Vec::new();

    for bundle in source_bundles {
        if !seen_samples.insert(bundle.benchmark_sample_id.as_str()) {
            return Err(TailProfileError::InvalidLineageAssignment);
        }
        let record = records_by_sample
            .get(bundle.benchmark_sample_id.as_str())
            .ok_or(TailProfileError::InvalidLineageAssignment)?;
        let split = match record.split {
            DatasetSplit::Train => TailProfileSplit::Calibration,
            DatasetSplit::Validation => TailProfileSplit::Qualification,
            DatasetSplit::Test => return Err(TailProfileError::InvalidLineageAssignment),
        };
        let report = bundle.adapter_report.report();
        if record.source_score_sha256.as_deref() != Some(report.source_score_sha256.as_str()) {
            return Err(TailProfileError::SourceLineageMismatch);
        }
        let manifest_operation = record.operation_family.to_ascii_lowercase().replace('-', "_");
        let observed_operation = operation_family_from_report(report)?;
        let operation_matches = match observed_operation.as_str() {
            "remove" => matches!(
                manifest_operation.as_str(),
                "remove" | "remove_event" | "remove_instrument"
            ),
            _ => false,
        };
        if !operation_matches {
            return Err(TailProfileError::UnsupportedOperationFamily);
        }
        let (bundle_context, mut bundle_observations) = observations_from_verified_report(
            bundle.adapter_report,
            topology,
            &record.composition_lineage_sha256,
            split,
        )?;
        if let Some(expected) = &context {
            if expected != &bundle_context {
                return Err(TailProfileError::RendererContextMismatch);
            }
        } else {
            context = Some(bundle_context);
        }
        observations.append(&mut bundle_observations);
    }

    let context = context.ok_or(TailProfileError::EventCountMismatch)?;
    let mut profile = create_profile_unsigned(
        spec,
        context,
        &observations,
        &benchmark_audit.manifest_sha256,
        &benchmark_audit.audit_sha256,
        signer_key_id,
    )?;
    if !profile.qualification.qualified {
        return Err(TailProfileError::QualificationTailMiss);
    }
    let digest = profile_digest(&profile)?;
    profile.profile_sha256 = digest.clone();
    profile.signature_ed25519_hex =
        hex_encode(signing_key.sign(digest.as_bytes()).to_bytes().as_ref());
    Ok(profile)
}

pub fn verify_tail_profile(
    profile: &TailProfile,
    authorities: &BTreeMap<String, TailProfileAuthority>,
) -> Result<(), TailProfileError> {
    if profile.version != PROFILE_VERSION
        || !profile.qualification.qualified
        || !valid_label(&profile.profile_id)
        || !valid_label(&profile.dataset_revision)
        || profile.profile_sha256 != profile_digest(profile)?
        || !is_sha256(&profile.profile_sha256)
        || !is_sha256(&profile.calibration_lineage_set_sha256)
        || !is_sha256(&profile.qualification_lineage_set_sha256)
        || !is_sha256(&profile.event_evidence_set_sha256)
        || !is_sha256(&profile.builder_spec_sha256)
        || !is_sha256(&profile.benchmark_manifest_sha256)
        || !is_sha256(&profile.benchmark_audit_sha256)
        || !is_sha256(&profile.context.renderer_identity_sha256)
        || !is_sha256(&profile.context.renderer_executable_sha256)
        || !is_sha256(&profile.context.soundfont_sha256)
        || !is_sha256(&profile.context.renderer_config_sha256)
        || !is_sha256(&profile.context.render_config_sha256)
        || !is_sha256(&profile.context.effect_settings_sha256)
        || !is_sha256(&profile.context.reverb_settings_sha256)
        || !is_sha256(&profile.context.chorus_settings_sha256)
        || !is_sha256(&profile.context.route_graph_sha256)
        || !is_sha256(&profile.context.output_chain_sha256)
        || !is_sha256(&profile.context.topology_attestation_sha256)
        || !is_sha256(&profile.context.segmentation_config_sha256)
        || !verify_effects_group_assignment(&profile.context.effects_group_assignment)
        || profile.context.sample_rate_hz == 0
        || profile.context.channels == 0
        || !valid_label(&profile.context.operation_family)
        || !valid_label(&profile.signer_key_id)
        || profile.builder_spec.dataset_revision != profile.dataset_revision
        || profile.builder_spec.profile_id != profile.profile_id
        || builder_spec_digest(&profile.builder_spec)? != profile.builder_spec_sha256
        || profile.event_observations.is_empty()
        || profile.event_observations.len() > MAX_PROFILE_EVENTS
        || event_set_digest(&profile.event_observations) != profile.event_evidence_set_sha256
        || lineage_summaries(&profile.event_observations)? != profile.lineage_summaries
    {
        return Err(TailProfileError::ProfileReceiptInvalid);
    }

    // Recompute every published statistic from the retained, source-lineage
    // clustered observations. A signed aggregate number is not accepted alone.
    let expected = create_profile_unsigned(
        &profile.builder_spec,
        profile.context.clone(),
        &profile.event_observations,
        &profile.benchmark_manifest_sha256,
        &profile.benchmark_audit_sha256,
        &profile.signer_key_id,
    )?;
    if expected.statistics != profile.statistics
        || expected.qualification != profile.qualification
        || expected.calibration_lineage_set_sha256 != profile.calibration_lineage_set_sha256
        || expected.qualification_lineage_set_sha256 != profile.qualification_lineage_set_sha256
        || expected.event_evidence_set_sha256 != profile.event_evidence_set_sha256
        || expected.lineage_summaries != profile.lineage_summaries
    {
        return Err(TailProfileError::ProfileReceiptInvalid);
    }

    let authority = authorities
        .get(&profile.signer_key_id)
        .ok_or(TailProfileError::InvalidAuthority)?;
    verify_signature(
        &authority.public_key_ed25519_hex,
        profile.profile_sha256.as_bytes(),
        &profile.signature_ed25519_hex,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTailProfile {
    profile_sha256: String,
    guard_frames: u64,
}

impl ResolvedTailProfile {
    pub fn profile_sha256(&self) -> &str {
        &self.profile_sha256
    }

    pub fn guard_frames(&self) -> u64 {
        self.guard_frames
    }

    /// Apply the resolved guard to EVAL-017 v0.2. The profile digest remains
    /// attached to the segmentation report and its replay receipt.
    pub fn apply_to_segmentation_config(
        &self,
        config: &mut InfluenceSegmentationConfig,
    ) -> Result<(), TailProfileError> {
        let guard = usize::try_from(self.guard_frames).map_err(|_| TailProfileError::GuardOverflow)?;
        config.tail_guard_frames = guard;
        config.tail_guard_profile_sha256 = Some(self.profile_sha256.clone());
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct TailProfileRegistry {
    authorities: BTreeMap<String, TailProfileAuthority>,
    profiles: BTreeMap<String, TailProfile>,
    revoked_profile_sha256: BTreeSet<String>,
}

impl TailProfileRegistry {
    pub fn new(authorities: Vec<TailProfileAuthority>) -> Result<Self, TailProfileError> {
        let mut registry = Self::default();
        for authority in authorities {
            if !valid_label(&authority.key_id)
                || hex_decode::<32>(&authority.public_key_ed25519_hex).is_none()
                || registry.authorities.insert(authority.key_id.clone(), authority).is_some()
            {
                return Err(TailProfileError::DuplicateAuthority);
            }
        }
        Ok(registry)
    }

    pub fn register_profile(&mut self, profile: TailProfile) -> Result<(), TailProfileError> {
        verify_tail_profile(&profile, &self.authorities)?;
        if self.profiles.contains_key(&profile.profile_sha256) {
            return Err(TailProfileError::DuplicateProfile);
        }
        self.profiles.insert(profile.profile_sha256.clone(), profile);
        Ok(())
    }

    /// Revocation is append-only from the registry's perspective. It does not
    /// rewrite the signed profile or historical support reports.
    pub fn revoke_profile(&mut self, profile_sha256: &str) -> Result<(), TailProfileError> {
        if !self.profiles.contains_key(profile_sha256) {
            return Err(TailProfileError::UnknownProfile);
        }
        self.revoked_profile_sha256.insert(profile_sha256.to_owned());
        Ok(())
    }

    pub fn resolve(
        &self,
        profile_sha256: &str,
        expected_context: &TailProfileContext,
    ) -> Result<ResolvedTailProfile, TailProfileError> {
        if self.revoked_profile_sha256.contains(profile_sha256) {
            return Err(TailProfileError::RevokedProfile);
        }
        let profile = self.profiles.get(profile_sha256).ok_or(TailProfileError::UnknownProfile)?;
        verify_tail_profile(profile, &self.authorities)?;
        if !profile.qualification.qualified {
            return Err(TailProfileError::ProfileNotQualified);
        }
        if &profile.context != expected_context {
            return Err(TailProfileError::ProfileContextMismatch);
        }
        Ok(ResolvedTailProfile {
            profile_sha256: profile.profile_sha256.clone(),
            guard_frames: profile.statistics.guard_frames,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&[19_u8; 32])
    }

    fn authority(key_id: &str, signing_key: &SigningKey) -> TailProfileAuthority {
        TailProfileAuthority {
            key_id: key_id.to_owned(),
            public_key_ed25519_hex: hex_encode(&signing_key.verifying_key().to_bytes()),
        }
    }

    fn topology_attestation(key_id: &str, signing_key: &SigningKey) -> RendererTopologyAttestation {
        let mut attestation = RendererTopologyAttestation {
            version: TOPOLOGY_VERSION.to_owned(),
            authority_key_id: key_id.to_owned(),
            renderer_identity_sha256: digest("renderer-identity"),
            renderer_executable_sha256: digest("fluidsynth-exe"),
            soundfont_sha256: digest("soundfont"),
            renderer_config_sha256: digest("renderer-config"),
            render_config_sha256: digest("render-config"),
            effects_group_assignment: crate::fluid_render::build_effects_group_assignment(4).unwrap(),
            effect_settings_sha256: digest("effect-settings"),
            reverb_settings_sha256: digest("reverb-settings"),
            chorus_settings_sha256: digest("chorus-settings"),
            route_graph_sha256: digest("full-route-graph"),
            route_graph_complete: true,
            output_chain_sha256: digest("output-chain"),
            sample_rate_hz: 48_000,
            channels: 2,
            bits_per_sample: 32,
            sample_format: DecodedSampleFormat::PcmFloat,
            signature_ed25519_hex: String::new(),
        };
        attestation.signature_ed25519_hex = hex_encode(
            signing_key.sign(topology_digest(&attestation).unwrap().as_bytes()).to_bytes().as_ref()
        );
        attestation
    }

    fn context(attestation: &RendererTopologyAttestation) -> TailProfileContext {
        TailProfileContext {
            renderer_identity_sha256: attestation.renderer_identity_sha256.clone(),
            renderer_executable_sha256: attestation.renderer_executable_sha256.clone(),
            soundfont_sha256: attestation.soundfont_sha256.clone(),
            renderer_config_sha256: attestation.renderer_config_sha256.clone(),
            render_config_sha256: attestation.render_config_sha256.clone(),
            effects_group_assignment: attestation.effects_group_assignment.clone(),
            effect_settings_sha256: attestation.effect_settings_sha256.clone(),
            reverb_settings_sha256: attestation.reverb_settings_sha256.clone(),
            chorus_settings_sha256: attestation.chorus_settings_sha256.clone(),
            route_graph_sha256: attestation.route_graph_sha256.clone(),
            output_chain_sha256: attestation.output_chain_sha256.clone(),
            topology_attestation_sha256: topology_digest(attestation).unwrap(),
            sample_rate_hz: 48_000,
            channels: 2,
            bits_per_sample: 32,
            sample_format: DecodedSampleFormat::PcmFloat,
            segmentation_config_sha256: digest("segmentation"),
            operation_family: "remove".to_owned(),
        }
    }

    fn spec() -> TailProfileBuildSpec {
        TailProfileBuildSpec {
            profile_id: "fluidsynth-fluidr3-48k-g4".to_owned(),
            dataset_revision: "test-release-v1".to_owned(),
            quantile_micros: 990_000,
            confidence_micros: 950_000,
            safety_margin_frames: 64,
            minimum_qualification_lineages: 30,
            minimum_trailing_quiet_windows: 3,
            builder_version: "tail-profile-builder-v1".to_owned(),
        }
    }

    fn signed_test_profile(
        context: TailProfileContext,
        key_id: &str,
        signing_key: &SigningKey,
    ) -> TailProfile {
        let mut build_spec = spec();
        build_spec.dataset_revision = "test".to_owned();
        let mut observations = Vec::new();
        for (split, count, prefix) in [
            (TailProfileSplit::Calibration, 299usize, "cal"),
            (TailProfileSplit::Qualification, 30usize, "qual"),
        ] {
            for index in 0..count {
                let lineage = format!("{prefix}-lineage-{index}");
                observations.push(TailProfileEventObservation {
                    source_event_index: index,
                    source_event_id: "score_event_index:0".to_owned(),
                    source_lineage_sha256: digest(&lineage),
                    split,
                    event_evidence_sha256: digest(&format!("{prefix}-event-{index}")),
                    final_note_off_frame: 100,
                    raw_measured_end_frame: Some(220),
                    hysteresis_support_end_frame: Some(220),
                    observed_tail_frames: 120,
                    trailing_quiet_windows: 4,
                    total_windows: 10,
                    output_frame_count: 1_000,
                });
            }
        }
        let mut profile = create_profile_unsigned(
            &build_spec,
            context,
            &observations,
            &digest("manifest"),
            &digest("audit"),
            key_id,
        )
        .unwrap();
        let profile_sha = profile_digest(&profile).unwrap();
        profile.profile_sha256 = profile_sha.clone();
        profile.signature_ed25519_hex =
            hex_encode(signing_key.sign(profile_sha.as_bytes()).to_bytes().as_ref());
        profile
    }

    fn observation(index: usize, lineage: u8, split: TailProfileSplit, tail: u64) -> TailProfileEventObservation {
        TailProfileEventObservation {
            source_event_index: index,
            source_event_id: format!("score_event_index:{index}"),
            source_lineage_sha256: digest(&format!("lineage-{lineage}")),
            split,
            event_evidence_sha256: digest(&format!("event-{index}")),
            final_note_off_frame: 100,
            raw_measured_end_frame: Some(100 + tail),
            hysteresis_support_end_frame: Some(100 + tail),
            observed_tail_frames: tail,
            trailing_quiet_windows: 4,
            total_windows: 10,
            output_frame_count: 10_000,
        }
    }

    #[test]
    fn distribution_free_99th_quantile_95pct_bound_needs_299_lineages() {
        assert_eq!(minimum_lineages_for_upper_quantile_bound(990_000, 950_000).unwrap(), 299);
        assert!(distribution_free_upper_bound_coverage(990_000, 299).unwrap() >= 950_000);
        assert!(distribution_free_upper_bound_coverage(990_000, 298).unwrap() < 950_000);
    }

    #[test]
    fn empirical_quantile_is_nearest_rank_and_bound_is_sample_maximum() {
        let values = vec![1, 4, 9, 12, 100];
        assert_eq!(nearest_rank_quantile(&values, 800_000).unwrap(), 12);
        assert_eq!(values.iter().copied().max().unwrap(), 100);
    }

    #[test]
    fn authority_signature_binds_complete_route_topology() {
        let key = signing_key();
        let auth = authority("renderer-authority-v1", &key);
        let mut attestation = topology_attestation("renderer-authority-v1", &key);
        let verified = verify_renderer_topology_attestation(&attestation, &auth).unwrap();
        assert!(is_sha256(verified.attestation_sha256()));
        attestation.route_graph_sha256 = digest("different-graph");
        assert_eq!(
            verify_renderer_topology_attestation(&attestation, &auth).unwrap_err(),
            TailProfileError::InvalidSignature
        );
    }

    #[test]
    fn incomplete_route_graph_cannot_mint_verified_topology() {
        let key = signing_key();
        let auth = authority("renderer-authority-v1", &key);
        let mut attestation = topology_attestation("renderer-authority-v1", &key);
        attestation.route_graph_complete = false;
        assert_eq!(
            verify_renderer_topology_attestation(&attestation, &auth).unwrap_err(),
            TailProfileError::IncompleteRouteGraph
        );
    }

    #[test]
    fn aggregate_tails_use_maximum_per_composition_cluster() {
        let observations = vec![
            observation(0, 1, TailProfileSplit::Calibration, 10),
            observation(1, 1, TailProfileSplit::Calibration, 90),
            observation(2, 2, TailProfileSplit::Calibration, 20),
        ];
        let tails = aggregate_lineage_tails(&observations, TailProfileSplit::Calibration).unwrap();
        assert_eq!(tails.len(), 2);
        assert_eq!(tails[&digest("lineage-1")].hysteresis_tail_frames, 90);
    }

    #[test]
    fn empty_qualification_or_cross_split_lineages_are_rejected() {
        let mut observations = vec![observation(0, 1, TailProfileSplit::Calibration, 20)];
        assert!(matches!(
            create_profile_unsigned(&spec(), context(&topology_attestation("k", &signing_key())), &observations, &digest("manifest"), &digest("audit"), "k"),
            Err(TailProfileError::InsufficientCalibrationLineages)
        ));
        observations.push(observation(1, 1, TailProfileSplit::Qualification, 10));
        assert_eq!(
            aggregate_lineage_tails(&observations, TailProfileSplit::Calibration).unwrap().len(),
            1
        );
        assert_eq!(
            aggregate_lineage_tails(&observations, TailProfileSplit::Qualification).unwrap().len(),
            1
        );
    }

    #[test]
    fn signed_profile_registry_requires_exact_context_and_revocation() {
        let key = signing_key();
        let authority = authority("profile-publisher-v1", &key);
        let attestation = topology_attestation("profile-publisher-v1", &key);
        let profile = signed_test_profile(context(&attestation), "profile-publisher-v1", &key);
        let profile_sha = profile.profile_sha256.clone();
        let mut registry = TailProfileRegistry::new(vec![authority]).unwrap();
        registry.register_profile(profile.clone()).unwrap();
        let resolved = registry.resolve(&profile_sha, &profile.context).unwrap();
        assert_eq!(resolved.guard_frames(), 184);
        assert_eq!(
            resolved.apply_to_segmentation_config(&mut InfluenceSegmentationConfig {
                threshold_combination: crate::counterfactual_influence::InfluenceThresholdCombination::AnyMetricExceeds,
                release_threshold_ratio: 0.5,
                release_persistence_windows: 2,
                tail_guard_frames: 0,
                tail_guard_profile_sha256: None,
            }),
            Ok(())
        );
        registry.revoke_profile(&profile_sha).unwrap();
        assert_eq!(
            registry.resolve(&profile_sha, &profile.context).unwrap_err(),
            TailProfileError::RevokedProfile
        );
    }

    #[test]
    fn recomputed_statistics_reject_a_resigned_but_forged_guard() {
        let key = signing_key();
        let attestation = topology_attestation("profile-publisher-v1", &key);
        let mut profile = signed_test_profile(context(&attestation), "profile-publisher-v1", &key);
        profile.statistics.guard_frames += 1;
        let changed_digest = profile_digest(&profile).unwrap();
        profile.profile_sha256 = changed_digest.clone();
        profile.signature_ed25519_hex =
            hex_encode(key.sign(changed_digest.as_bytes()).to_bytes().as_ref());
        let mut authorities = BTreeMap::new();
        authorities.insert(
            "profile-publisher-v1".to_owned(),
            authority("profile-publisher-v1", &key),
        );
        assert_eq!(
            verify_tail_profile(&profile, &authorities).unwrap_err(),
            TailProfileError::ProfileReceiptInvalid
        );
    }

    #[test]
    fn event_observation_tampering_is_detected_even_after_resigning_profile() {
        let key = signing_key();
        let attestation = topology_attestation("profile-publisher-v1", &key);
        let mut profile = signed_test_profile(context(&attestation), "profile-publisher-v1", &key);
        profile.event_observations[0].raw_measured_end_frame = Some(221);
        profile.event_evidence_set_sha256 = event_set_digest(&profile.event_observations);
        profile.lineage_summaries = lineage_summaries(&profile.event_observations).unwrap();
        let changed_digest = profile_digest(&profile).unwrap();
        profile.profile_sha256 = changed_digest.clone();
        profile.signature_ed25519_hex =
            hex_encode(key.sign(changed_digest.as_bytes()).to_bytes().as_ref());
        let mut authorities = BTreeMap::new();
        authorities.insert(
            "profile-publisher-v1".to_owned(),
            authority("profile-publisher-v1", &key),
        );
        assert_eq!(
            verify_tail_profile(&profile, &authorities).unwrap_err(),
            TailProfileError::ProfileReceiptInvalid
        );
    }

    #[test]
    fn modified_profile_signature_or_context_cannot_resolve() {
        let key = signing_key();
        let authority = authority("profile-publisher-v1", &key);
        let attestation = topology_attestation("profile-publisher-v1", &key);
        let mut profile = signed_test_profile(context(&attestation), "profile-publisher-v1", &key);
        let profile_sha = profile.profile_sha256.clone();
        let mut registry = TailProfileRegistry::new(vec![authority]).unwrap();
        registry.register_profile(profile.clone()).unwrap();
        let mut changed = profile.context.clone();
        changed.sample_rate_hz = 44_100;
        assert_eq!(
            registry.resolve(&profile_sha, &changed).unwrap_err(),
            TailProfileError::ProfileContextMismatch
        );
        profile.statistics.guard_frames += 1;
        assert_eq!(
            verify_tail_profile(&profile, &registry.authorities).unwrap_err(),
            TailProfileError::ProfileReceiptInvalid
        );
    }
}
