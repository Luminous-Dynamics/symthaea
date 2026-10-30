// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Replay-verifiable, renderer-topology-specific tail calibration.
//!
//! Tail duration is derived only from replay-verified EVAL-017 v0.2 event
//! support. Independent statistical units are source-composition lineages,
//! not events, channels, windows or repeated renders. The selected calibration
//! quantile is accompanied by a distribution-free one-sided upper confidence
//! bound based on the maximum cluster observation. Held-out qualification
//! lineages do not influence the guard estimate.

use crate::counterfactual_influence::{
    CounterfactualInfluenceInput, CounterfactualInfluenceSegmentedReport,
    InfluenceSegmentationConfig, VerifiedRepeatabilityCalibration,
};
use crate::counterfactual_support_adapter::{
    CounterfactualSupportAdapterError, VerifiedCounterfactualEventSupport,
    verify_counterfactual_event_support,
};
use crate::fluid_render::FluidSynthEffectsGroupAssignment;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const TAIL_PROFILE_VERSION: &str = "muse-topology-tail-profile-v1";
const TAIL_OBSERVATION_DOMAIN: &[u8] =
    b"luminous-dynamics-muse-verified-tail-observation-v1\0";
const TAIL_PROFILE_DOMAIN: &[u8] =
    b"luminous-dynamics-muse-qualified-tail-profile-v1\0";
const MAX_TAIL_PROFILE_EVENTS: usize = 1_000_000;
const MAX_TAIL_PROFILE_LINEAGES: usize = 100_000;
const MAX_TAIL_GUARD_FRAMES: u64 = 10_000_000;

/// Exact renderer/effects/output domain for which a tail guard may be reused.
///
/// renderer_identity_sha256 and renderer_config_sha256 must commit to the
/// executable/build, SoundFont and complete declared settings. If the
/// downstream route graph is unknown, this key is not a proof of topology
/// completeness; it only scopes the profile to the same observed artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailTopologyKey {
    pub renderer_identity_sha256: String,
    pub renderer_config_sha256: String,
    pub effects_group_assignment: FluidSynthEffectsGroupAssignment,
    pub sample_rate_hz: u32,
    pub channels: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfileBuildConfig {
    /// Target population quantile in permille: 950 = 95th percentile.
    pub target_quantile_permille: u16,
    /// One-sided confidence in permille for the calibration quantile bound.
    pub confidence_permille: u16,
    /// One-sided confidence for the held-out zero-miss coverage lower bound.
    pub qualification_confidence_permille: u16,
    /// Minimum independent held-out source lineages required for qualification.
    pub minimum_qualification_lineages: usize,
    /// Added conservative frame margin, separate from the calibration bound.
    pub safety_margin_frames: u64,
}

impl Default for TailProfileBuildConfig {
    fn default() -> Self {
        Self {
            target_quantile_permille: 950,
            confidence_permille: 950,
            qualification_confidence_permille: 950,
            minimum_qualification_lineages: 59,
            safety_margin_frames: 0,
        }
    }
}

/// One opaque, replay-verified observation for a source event. The measured
/// tail is hysteresis support extending beyond the MIDI-projected final
/// note-off. It can include renderer latency as well as effect decay and is
/// therefore bound to a specific renderer topology.
#[derive(Debug, Clone)]
pub struct VerifiedTailObservation {
    lineage_sha256: String,
    source_event_id: String,
    support_evidence_sha256: String,
    topology: TailTopologyKey,
    source_event_last_note_off_frame: u64,
    output_frame_count: u64,
    hysteresis_support_end_frame: u64,
    observed_tail_frames: u64,
    /// A support interval reaching the final PCM frame after note-off means
    /// the tail may be right-censored by an incomplete render horizon.
    horizon_censored: bool,
    observation_sha256: String,
}

impl VerifiedTailObservation {
    pub fn lineage_sha256(&self) -> &str {
        &self.lineage_sha256
    }

    pub fn observed_tail_frames(&self) -> u64 {
        self.observed_tail_frames
    }

    pub fn horizon_censored(&self) -> bool {
        self.horizon_censored
    }
}

/// Original inputs required to replay one event observation.
pub struct TailProfileReplayInput<'a, 'view, 'audio> {
    pub report: &'a CounterfactualInfluenceSegmentedReport,
    pub input: &'a CounterfactualInfluenceInput<'view, 'audio>,
    pub calibration: &'a VerifiedRepeatabilityCalibration,
    pub source_lineage_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailProfileReport {
    pub profile_version: String,
    pub build_config: TailProfileBuildConfig,
    pub topology: TailTopologyKey,
    pub calibration_event_count: usize,
    pub calibration_lineage_count: usize,
    pub calibration_lineage_set_sha256: String,
    pub calibration_observation_set_sha256: String,
    pub empirical_quantile_tail_frames: u64,
    /// Maximum per-lineage calibration tail. For q-quantile, this is an
    /// upper confidence bound with coverage 1 - q^n under IID lineages.
    pub distribution_free_upper_bound_frames: u64,
    pub minimum_required_calibration_lineages: usize,
    pub safety_margin_frames: u64,
    pub guard_frames: u64,
    pub qualification_event_count: usize,
    pub qualification_lineage_count: usize,
    pub qualification_lineage_set_sha256: String,
    pub qualification_observation_set_sha256: String,
    pub qualification_max_tail_frames: u64,
    pub qualification_miss_lineage_count: usize,
    pub qualification_coverage_lower_bound_permille: u16,
    pub qualified: bool,
    pub profile_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TailProfileError {
    InvalidBuildConfig(&'static str),
    InvalidLineageDigest,
    InvalidTopology,
    EmptyCalibrationPartition,
    EmptyQualificationPartition,
    TooManyObservations,
    TooManyLineages,
    DuplicateObservation,
    MixedTopology,
    SplitLineageOverlap,
    TailHorizonCensored,
    TailProfileReceiptInvalid,
    TailProfileNotQualified,
    TopologyMismatch,
    RegistryDigestMismatch,
    GuardOverflow,
    ReplayMismatch,
    SerializationFailure,
}

impl std::fmt::Display for TailProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tail profile qualification failed: {self:?}")
    }
}

impl std::error::Error for TailProfileError {}

impl From<CounterfactualSupportAdapterError> for TailProfileError {
    fn from(_: CounterfactualSupportAdapterError) -> Self {
        Self::ReplayMismatch
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_component(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn observation_digest(
    observation: &VerifiedTailObservation,
) -> Result<String, TailProfileError> {
    let bytes = serde_json::to_vec(&(
        &observation.lineage_sha256,
        &observation.source_event_id,
        &observation.support_evidence_sha256,
        &observation.topology,
        observation.source_event_last_note_off_frame,
        observation.output_frame_count,
        observation.hysteresis_support_end_frame,
        observation.observed_tail_frames,
        observation.horizon_censored,
    ))
    .map_err(|_| TailProfileError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(TAIL_OBSERVATION_DOMAIN);
    hash_component(&mut hasher, &bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_topology(topology: &TailTopologyKey) -> Result<(), TailProfileError> {
    if !is_sha256(&topology.renderer_identity_sha256)
        || !is_sha256(&topology.renderer_config_sha256)
        || topology.sample_rate_hz == 0
        || topology.channels == 0
        || !crate::fluid_render::verify_effects_group_assignment(
            &topology.effects_group_assignment,
        )
    {
        return Err(TailProfileError::InvalidTopology);
    }
    Ok(())
}

fn validate_config(config: &TailProfileBuildConfig) -> Result<(), TailProfileError> {
    if !(500..=999).contains(&config.target_quantile_permille)
        || !(500..=999).contains(&config.confidence_permille)
        || !(500..=999).contains(&config.qualification_confidence_permille)
        || config.minimum_qualification_lineages == 0
        || config.minimum_qualification_lineages > MAX_TAIL_PROFILE_LINEAGES
        || config.safety_margin_frames > MAX_TAIL_GUARD_FRAMES
    {
        return Err(TailProfileError::InvalidBuildConfig(
            "quantile/confidence/minimum lineages/margin are outside bounded ranges",
        ));
    }
    Ok(())
}

/// Minimum n for which max(X_1..X_n) is an upper confidence bound for the
/// population q-quantile with coverage 1 - q^n >= requested confidence.
pub fn minimum_calibration_lineages(
    target_quantile_permille: u16,
    confidence_permille: u16,
) -> Result<usize, TailProfileError> {
    if !(500..=999).contains(&target_quantile_permille)
        || !(500..=999).contains(&confidence_permille)
    {
        return Err(TailProfileError::InvalidBuildConfig(
            "quantile and confidence must be in 500..=999 permille",
        ));
    }
    let q = f64::from(target_quantile_permille) / 1000.0;
    let alpha = 1.0 - f64::from(confidence_permille) / 1000.0;
    let required = (alpha.ln() / q.ln()).ceil();
    if !required.is_finite() || required < 1.0 || required > MAX_TAIL_PROFILE_LINEAGES as f64 {
        return Err(TailProfileError::InvalidBuildConfig(
            "derived independent calibration lineage requirement is out of range",
        ));
    }
    Ok(required as usize)
}

fn make_observation(
    verified: &VerifiedCounterfactualEventSupport,
    lineage_sha256: &str,
) -> Result<VerifiedTailObservation, TailProfileError> {
    if !is_sha256(lineage_sha256) {
        return Err(TailProfileError::InvalidLineageDigest);
    }
    let evidence = verified.evidence();
    if !is_sha256(&evidence.evidence_sha256)
        || evidence.tail_guard_was_clipped
        || evidence.source_event_last_note_off_frame > evidence.output_frame_count
    {
        return Err(TailProfileError::TailHorizonCensored);
    }

    let topology = TailTopologyKey {
        renderer_identity_sha256: evidence.renderer_identity_sha256.clone(),
        renderer_config_sha256: evidence.renderer_config_sha256.clone(),
        effects_group_assignment: evidence.effects_group_assignment.clone(),
        sample_rate_hz: evidence.sample_rate_hz,
        channels: evidence.channels,
    };
    validate_topology(&topology)?;

    let support_end = evidence
        .hysteresis_support_intervals
        .iter()
        .map(|interval| interval.end_frame)
        .max()
        .unwrap_or(evidence.source_event_last_note_off_frame);
    if support_end > evidence.output_frame_count {
        return Err(TailProfileError::TailHorizonCensored);
    }
    let horizon_censored = !evidence.hysteresis_support_intervals.is_empty()
        && support_end == evidence.output_frame_count
        && support_end > evidence.source_event_last_note_off_frame;
    if horizon_censored {
        return Err(TailProfileError::TailHorizonCensored);
    }

    let observed_tail_frames = support_end.saturating_sub(
        evidence.source_event_last_note_off_frame,
    );
    let mut observation = VerifiedTailObservation {
        lineage_sha256: lineage_sha256.to_owned(),
        source_event_id: evidence.source_event_id.clone(),
        support_evidence_sha256: evidence.evidence_sha256.clone(),
        topology,
        source_event_last_note_off_frame: evidence.source_event_last_note_off_frame,
        output_frame_count: evidence.output_frame_count,
        hysteresis_support_end_frame: support_end,
        observed_tail_frames,
        horizon_censored,
        observation_sha256: String::new(),
    };
    observation.observation_sha256 = observation_digest(&observation)?;
    Ok(observation)
}

fn replay_partition(
    inputs: &[TailProfileReplayInput<'_, '_, '_>],
) -> Result<Vec<VerifiedTailObservation>, TailProfileError> {
    if inputs.is_empty() {
        return Err(TailProfileError::EmptyCalibrationPartition);
    }
    if inputs.len() > MAX_TAIL_PROFILE_EVENTS {
        return Err(TailProfileError::TooManyObservations);
    }
    let mut observations = Vec::with_capacity(inputs.len());
    for replay in inputs {
        // A tail profile may only be built from a scan with no previously
        // applied guard, avoiding recursive calibration of a guard against itself.
        if replay.report.segmentation.tail_guard_frames != 0
            || replay.report.segmentation.tail_guard_profile_sha256.is_some()
        {
            return Err(TailProfileError::InvalidBuildConfig(
                "profile observations must come from zero-tail-guard segmented reports",
            ));
        }
        let verified = verify_counterfactual_event_support(
            replay.report,
            replay.input,
            replay.calibration,
        )?;
        observations.push(make_observation(&verified, replay.source_lineage_sha256)?);
    }
    Ok(observations)
}

fn manifest_sha256(
    observations: &[VerifiedTailObservation],
    domain: &[u8],
) -> Result<String, TailProfileError> {
    let mut digests = observations
        .iter()
        .map(|observation| observation.observation_sha256.clone())
        .collect::<Vec<_>>();
    digests.sort();
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((digests.len() as u64).to_le_bytes());
    for digest in digests {
        hash_component(&mut hasher, digest.as_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn clustered_tail_maxima(
    observations: &[VerifiedTailObservation],
) -> Result<BTreeMap<String, u64>, TailProfileError> {
    if observations.is_empty() {
        return Err(TailProfileError::EmptyCalibrationPartition);
    }
    if observations.iter().any(|observation| {
        observation.observation_sha256 != observation_digest(observation).unwrap_or_default()
    }) {
        return Err(TailProfileError::TailProfileReceiptInvalid);
    }
    let mut maxima = BTreeMap::<String, u64>::new();
    let mut seen_event_rows = BTreeSet::new();
    for observation in observations {
        let key = (observation.lineage_sha256.clone(), observation.source_event_id.clone());
        if !seen_event_rows.insert(key) {
            return Err(TailProfileError::DuplicateObservation);
        }
        maxima
            .entry(observation.lineage_sha256.clone())
            .and_modify(|tail| *tail = (*tail).max(observation.observed_tail_frames))
            .or_insert(observation.observed_tail_frames);
    }
    if maxima.len() > MAX_TAIL_PROFILE_LINEAGES {
        return Err(TailProfileError::TooManyLineages);
    }
    Ok(maxima)
}

fn profile_digest(profile: &TailProfileReport) -> Result<String, TailProfileError> {
    let mut unsigned = profile.clone();
    unsigned.profile_sha256.clear();
    let bytes = serde_json::to_vec(&unsigned)
        .map_err(|_| TailProfileError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(TAIL_PROFILE_DOMAIN);
    hash_component(&mut hasher, &bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Build a topology-specific profile from replayed calibration and held-out
/// qualification reports. The held-out partition never changes guard_frames.
pub fn build_tail_profile(
    config: &TailProfileBuildConfig,
    calibration_inputs: &[TailProfileReplayInput<'_, '_, '_>],
    qualification_inputs: &[TailProfileReplayInput<'_, '_, '_>],
) -> Result<TailProfileReport, TailProfileError> {
    validate_config(config)?;
    let calibration = replay_partition(calibration_inputs)?;
    let qualification = replay_partition(qualification_inputs)?;
    let topology = calibration
        .first()
        .ok_or(TailProfileError::EmptyCalibrationPartition)?
        .topology
        .clone();
    validate_topology(&topology)?;
    if calibration.iter().chain(qualification.iter()).any(|row| row.topology != topology) {
        return Err(TailProfileError::MixedTopology);
    }

    let calibration_maxima = clustered_tail_maxima(&calibration)?;
    let qualification_maxima = clustered_tail_maxima(&qualification)?;
    let calibration_lineages: BTreeSet<String> = calibration_maxima.keys().cloned().collect();
    let qualification_lineages: BTreeSet<String> = qualification_maxima.keys().cloned().collect();
    if !calibration_lineages.is_disjoint(&qualification_lineages) {
        return Err(TailProfileError::SplitLineageOverlap);
    }

    let required_lineages = minimum_calibration_lineages(
        config.target_quantile_permille,
        config.confidence_permille,
    )?;
    if calibration_lineages.len() < required_lineages {
        return Err(TailProfileError::InvalidBuildConfig(
            "insufficient independent calibration lineages for the requested quantile confidence",
        ));
    }

    let mut calibration_tails = calibration_maxima.values().copied().collect::<Vec<_>>();
    calibration_tails.sort_unstable();
    let rank = ((u128::from(config.target_quantile_permille)
        * calibration_tails.len() as u128
        + 999)
        / 1000) as usize;
    let empirical_quantile_tail_frames = calibration_tails[rank.saturating_sub(1)];
    let distribution_free_upper_bound_frames =
        *calibration_tails.last().ok_or(TailProfileError::EmptyCalibrationPartition)?;
    let guard_frames = distribution_free_upper_bound_frames
        .checked_add(config.safety_margin_frames)
        .filter(|guard| *guard <= MAX_TAIL_GUARD_FRAMES)
        .ok_or(TailProfileError::GuardOverflow)?;

    let qualification_max_tail_frames = qualification_maxima.values().copied().max().unwrap_or(0);
    let qualification_miss_lineage_count = qualification_maxima
        .values()
        .filter(|tail| **tail > guard_frames)
        .count();
    if qualification_maxima.len() < config.minimum_qualification_lineages {
        return Err(TailProfileError::InvalidBuildConfig(
            "insufficient independent qualification lineages",
        ));
    }
    let qualification_coverage_lower_bound_permille =
        if qualification_miss_lineage_count == 0 {
            let alpha =
                1.0 - f64::from(config.qualification_confidence_permille) / 1000.0;
            let lower = alpha.powf(1.0 / qualification_maxima.len() as f64);
            (lower * 1000.0).floor().clamp(0.0, 1000.0) as u16
        } else {
            0
        };
    let qualified = qualification_miss_lineage_count == 0
        && calibration_lineages.len() >= required_lineages
        && qualification_maxima.len() >= config.minimum_qualification_lineages;

    let mut report = TailProfileReport {
        profile_version: TAIL_PROFILE_VERSION.to_owned(),
        build_config: config.clone(),
        topology,
        calibration_event_count: calibration.len(),
        calibration_lineage_count: calibration_lineages.len(),
        calibration_lineage_set_sha256: lineage_set_digest(&calibration_lineages),
        calibration_observation_set_sha256: manifest_sha256(
            &calibration,
            b"muse-tail-profile-calibration-observations-v1\0",
        )?,
        empirical_quantile_tail_frames,
        distribution_free_upper_bound_frames,
        minimum_required_calibration_lineages: required_lineages,
        safety_margin_frames: config.safety_margin_frames,
        guard_frames,
        qualification_event_count: qualification.len(),
        qualification_lineage_count: qualification_maxima.len(),
        qualification_lineage_set_sha256: lineage_set_digest(&qualification_lineages),
        qualification_observation_set_sha256: manifest_sha256(
            &qualification,
            b"muse-tail-profile-qualification-observations-v1\0",
        )?,
        qualification_max_tail_frames,
        qualification_miss_lineage_count,
        qualification_coverage_lower_bound_permille,
        qualified,
        profile_sha256: String::new(),
    };
    report.profile_sha256 = profile_digest(&report)?;
    Ok(report)
}

fn lineage_set_digest(lineages: &BTreeSet<String>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"muse-tail-profile-lineage-set-v1\0");
    hasher.update((lineages.len() as u64).to_le_bytes());
    for lineage in lineages {
        hash_component(&mut hasher, lineage.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Rebuild the complete profile from original event reports and compare every
/// statistic, lineage split, topology binding and receipt.
pub fn verify_tail_profile(
    supplied: &TailProfileReport,
    calibration_inputs: &[TailProfileReplayInput<'_, '_, '_>],
    qualification_inputs: &[TailProfileReplayInput<'_, '_, '_>],
) -> Result<bool, TailProfileError> {
    if !is_sha256(&supplied.profile_sha256)
        || supplied.profile_sha256 != profile_digest(supplied)?
    {
        return Err(TailProfileError::TailProfileReceiptInvalid);
    }
    let expected = build_tail_profile(
        &supplied.build_config,
        calibration_inputs,
        qualification_inputs,
    )?;
    if expected != *supplied {
        return Err(TailProfileError::ReplayMismatch);
    }
    Ok(true)
}

/// Opaque profile token accepted by the runtime guard resolver.
#[derive(Debug, Clone)]
pub struct VerifiedTailProfile {
    report: TailProfileReport,
}

impl VerifiedTailProfile {
    pub fn report(&self) -> &TailProfileReport {
        &self.report
    }

    pub fn guard_frames(&self) -> u64 {
        self.report.guard_frames
    }

    pub fn profile_sha256(&self) -> &str {
        &self.report.profile_sha256
    }

    pub fn topology(&self) -> &TailTopologyKey {
        &self.report.topology
    }

    /// Populate the v0.2 segmenter with the exact qualified profile.
    pub fn apply_to_segmentation_config(
        &self,
        config: &mut InfluenceSegmentationConfig,
    ) {
        config.tail_guard_frames = self.report.guard_frames as usize;
        config.tail_guard_profile_sha256 = Some(self.report.profile_sha256.clone());
    }
}

/// Resolve a replay-verified profile only when the caller's trusted registry
/// allowlists that exact profile digest and the renderer topology matches.
pub fn resolve_verified_tail_profile(
    profile: TailProfileReport,
    expected_topology: &TailTopologyKey,
    trusted_profile_sha256: &str,
) -> Result<VerifiedTailProfile, TailProfileError> {
    if !is_sha256(&profile.profile_sha256)
        || profile.profile_sha256 != profile_digest(&profile)?
    {
        return Err(TailProfileError::TailProfileReceiptInvalid);
    }
    if !profile.qualified {
        return Err(TailProfileError::TailProfileNotQualified);
    }
    if profile.topology != *expected_topology {
        return Err(TailProfileError::TopologyMismatch);
    }
    if trusted_profile_sha256 != profile.profile_sha256
        || !is_sha256(trusted_profile_sha256)
    {
        return Err(TailProfileError::RegistryDigestMismatch);
    }
    Ok(VerifiedTailProfile { report: profile })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> String {
        std::iter::repeat_n(format!("{byte:02x}"), 32).collect()
    }

    fn topology() -> TailTopologyKey {
        TailTopologyKey {
            renderer_identity_sha256: digest(1),
            renderer_config_sha256: digest(2),
            effects_group_assignment:
                crate::fluid_render::build_effects_group_assignment(4).unwrap(),
            sample_rate_hz: 48_000,
            channels: 2,
        }
    }

    fn observation(lineage: u8, event_index: usize, tail: u64) -> VerifiedTailObservation {
        let mut value = VerifiedTailObservation {
            lineage_sha256: digest(lineage),
            source_event_id: format!("score_event_index:{event_index}"),
            support_evidence_sha256: digest(event_index as u8),
            topology: topology(),
            source_event_last_note_off_frame: 100,
            output_frame_count: 1000,
            hysteresis_support_end_frame: 100 + tail,
            observed_tail_frames: tail,
            horizon_censored: false,
            observation_sha256: String::new(),
        };
        value.observation_sha256 = observation_digest(&value).unwrap();
        value
    }

    #[test]
    fn minimum_calibration_n_for_95th_percentile_95_confidence_is_59() {
        assert_eq!(minimum_calibration_lineages(950, 950).unwrap(), 59);
    }

    #[test]
    fn maximum_cluster_statistic_is_distribution_free_quantile_bound() {
        let calibration = (0..59_u8)
            .map(|lineage| observation(lineage, 0, u64::from(lineage)))
            .collect::<Vec<_>>();
        let qualification = (100..159_u16)
            .map(|lineage| observation((lineage - 100) as u8, 0, 58))
            .collect::<Vec<_>>();
        let report = profile_from_test_observations(
            &TailProfileBuildConfig::default(),
            calibration,
            qualification,
        )
        .unwrap();
        assert_eq!(report.distribution_free_upper_bound_frames, 58);
        assert_eq!(report.guard_frames, 58);
        assert!(report.qualified);
    }

    #[test]
    fn qualification_partition_does_not_change_guard_estimate() {
        let calibration = (0..59_u8)
            .map(|lineage| observation(lineage, 0, u64::from(lineage)))
            .collect::<Vec<_>>();
        let first_qualification = (100..159_u16)
            .map(|lineage| observation((lineage - 100) as u8, 0, 58))
            .collect::<Vec<_>>();
        let second_qualification = (200..259_u16)
            .map(|lineage| observation((lineage - 200) as u8, 0, 40))
            .collect::<Vec<_>>();
        let first = profile_from_test_observations(
            &TailProfileBuildConfig::default(),
            calibration.clone(),
            first_qualification,
        ).unwrap();
        let second = profile_from_test_observations(
            &TailProfileBuildConfig::default(),
            calibration,
            second_qualification,
        ).unwrap();
        assert_eq!(first.guard_frames, second.guard_frames);
        assert_ne!(
            first.qualification_lineage_set_sha256,
            second.qualification_lineage_set_sha256
        );
    }

    #[test]
    fn heldout_miss_marks_profile_unqualified() {
        let calibration = (0..59_u8)
            .map(|lineage| observation(lineage, 0, u64::from(lineage)))
            .collect::<Vec<_>>();
        let qualification = (100..159_u16)
            .map(|lineage| {
                let tail = if lineage == 100 { 100 } else { 58 };
                observation((lineage - 100) as u8, 0, tail)
            })
            .collect::<Vec<_>>();
        let report = profile_from_test_observations(
            &TailProfileBuildConfig::default(),
            calibration,
            qualification,
        )
        .unwrap();
        assert!(!report.qualified);
        assert_eq!(report.qualification_miss_lineage_count, 1);
        assert!(matches!(
            resolve_verified_tail_profile(
                report.clone(),
                &report.topology,
                &report.profile_sha256
            ),
            Err(TailProfileError::TailProfileNotQualified)
        ));
    }

    #[test]
    fn topology_mismatch_and_registry_mismatch_reject() {
        let calibration = (0..59_u8)
            .map(|lineage| observation(lineage, 0, u64::from(lineage)))
            .collect::<Vec<_>>();
        let qualification = (100..159_u16)
            .map(|lineage| observation((lineage - 100) as u8, 0, 58))
            .collect::<Vec<_>>();
        let report = profile_from_test_observations(
            &TailProfileBuildConfig::default(),
            calibration,
            qualification,
        )
        .unwrap();
        let mut wrong_topology = report.topology.clone();
        wrong_topology.sample_rate_hz = 44_100;
        assert!(matches!(
            resolve_verified_tail_profile(
                report.clone(),
                &wrong_topology,
                &report.profile_sha256
            ),
            Err(TailProfileError::TopologyMismatch)
        ));
        assert!(matches!(
            resolve_verified_tail_profile(
                report.clone(),
                &report.topology,
                &digest(99)
            ),
            Err(TailProfileError::RegistryDigestMismatch)
        ));
    }

    #[test]
    fn overlapping_calibration_and_qualification_lineages_reject() {
        let calibration = (0..59_u8)
            .map(|lineage| observation(lineage, 0, u64::from(lineage)))
            .collect::<Vec<_>>();
        let mut qualification = (100..159_u16)
            .map(|lineage| observation((lineage - 100) as u8, 0, 58))
            .collect::<Vec<_>>();
        qualification[0].lineage_sha256 = calibration[0].lineage_sha256.clone();
        qualification[0].observation_sha256 = observation_digest(&qualification[0]).unwrap();
        assert!(matches!(
            profile_from_test_observations(
                &TailProfileBuildConfig::default(),
                calibration,
                qualification
            ),
            Err(TailProfileError::SplitLineageOverlap)
        ));
    }

    // Unit-only constructor. Production callers must use replay_partition().
    fn profile_from_test_observations(
        config: &TailProfileBuildConfig,
        calibration: Vec<VerifiedTailObservation>,
        qualification: Vec<VerifiedTailObservation>,
    ) -> Result<TailProfileReport, TailProfileError> {
        build_tail_profile_from_verified_observations(config, calibration, qualification)
    }
}
