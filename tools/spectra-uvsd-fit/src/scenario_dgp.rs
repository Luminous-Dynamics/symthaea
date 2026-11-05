// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Known-truth observer and measurement-channel simulation for SPECTRA.
//!
//! Experimenter-controlled trial requests are separated from observer state:
//! latent confidence criteria, signal variance, criterion drift, lapses,
//! participant effects and measurement noise live only in ObserverScenario.
//! This module generates synthetic data; it does not model human behavior as
//! known, estimate model adequacy, or authorize a live experiment.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::sequential::{OperationalActionSet, StimulusClass};

/// Evidence distribution family used by the synthetic observer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceFamily {
    Gaussian,
    /// Student-t evidence rescaled to unit variance (requires df >= 3).
    VarianceStandardizedStudentT { degrees_of_freedom: u16 },
}

/// Latent observer criteria associated with one experimenter-presented
/// response-scale ID. These are simulated observer parameters, not actions.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseScaleSpec {
    pub response_scale_id: u16,
    /// Strictly increasing latent evidence criteria. K criteria imply K+1
    /// ordered categories. The experimenter chooses the scale ID, not values.
    pub latent_criteria: Vec<f64>,
}

/// Between-participant random effects. Effects are sampled once per
/// (scenario, participant, root seed) and reused on every trial for that
/// participant. Correlations are not yet modeled in v0.1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticipantHeterogeneity {
    pub signal_mean_sd: f64,
    pub log_signal_sigma_sd: f64,
    pub criterion_offset_sd: f64,
}

/// Optional synthetic response-time channel.
///
/// RT follows a lognormal evidence-speed model with additive non-decision
/// time, then is quantized. This is a declared stress-test channel, not a
/// universal psychophysical law.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtChannelSpec {
    pub base_latency_ms: f64,
    pub evidence_speed_gain: f64,
    pub log_noise_sd: f64,
    pub non_decision_time_ms: f64,
    pub quantization_ms: u32,
    pub default_deadline_ms: Option<u32>,
}

/// Full synthetic observer/task scenario. All criteria and observer
/// parameters are latent DGP inputs; none is accepted by the action selector.
#[derive(Debug, Clone, PartialEq)]
pub struct ObserverScenario {
    pub scenario_id: u16,
    pub evidence_family: EvidenceFamily,
    pub base_signal_mean: f64,
    pub signal_mean_per_stimulus_unit: f64,
    /// Signal evidence SD relative to unit-SD noise evidence.
    pub signal_sigma: f64,
    /// Common offset applied to all latent response criteria.
    pub criterion_offset: f64,
    /// Linear criterion drift per 100 within-participant trials.
    pub criterion_drift_per_100_trials: f64,
    pub lapse_probability: f64,
    /// Baseline probability of a missing confidence/category response.
    pub confidence_missing_probability: f64,
    /// Additional missingness probability scaled by |evidence|/(1+|evidence|).
    pub confidence_missing_evidence_slope: f64,
    pub response_scales: Vec<ResponseScaleSpec>,
    pub heterogeneity: Option<ParticipantHeterogeneity>,
    pub rt_channel: Option<RtChannelSpec>,
}

impl ObserverScenario {
    pub fn validate(&self) -> Result<(), DgpError> {
        for value in [
            self.base_signal_mean,
            self.signal_mean_per_stimulus_unit,
            self.signal_sigma,
            self.criterion_offset,
            self.criterion_drift_per_100_trials,
            self.lapse_probability,
            self.confidence_missing_probability,
            self.confidence_missing_evidence_slope,
        ] {
            if !value.is_finite() {
                return Err(DgpError::NonFiniteScenarioParameter);
            }
        }
        if self.signal_sigma <= 0.0 {
            return Err(DgpError::InvalidSignalSigma);
        }
        if !(0.0..1.0).contains(&self.lapse_probability) {
            return Err(DgpError::InvalidLapseProbability);
        }
        if !(0.0..=1.0).contains(&self.confidence_missing_probability)
            || self.confidence_missing_evidence_slope < 0.0
            || self.confidence_missing_probability + self.confidence_missing_evidence_slope > 1.0
        {
            return Err(DgpError::InvalidMissingnessProbability);
        }
        match self.evidence_family {
            EvidenceFamily::Gaussian => {}
            EvidenceFamily::VarianceStandardizedStudentT { degrees_of_freedom } => {
                if degrees_of_freedom < 3 {
                    return Err(DgpError::InvalidStudentTDegreesOfFreedom);
                }
            }
        }
        if self.response_scales.is_empty() {
            return Err(DgpError::NoResponseScales);
        }
        if self.response_scales.len() > u16::MAX as usize {
            return Err(DgpError::TooManyResponseScales);
        }
        let mut scale_ids = BTreeSet::new();
        for scale in &self.response_scales {
            if !scale_ids.insert(scale.response_scale_id) {
                return Err(DgpError::DuplicateResponseScaleId);
            }
            if scale.latent_criteria.len() < 2
                || scale.latent_criteria.len() >= u16::MAX as usize
            {
                return Err(DgpError::InvalidCriterionCount);
            }
            if scale.latent_criteria.iter().any(|value| !value.is_finite())
                || scale
                    .latent_criteria
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
            {
                return Err(DgpError::InvalidLatentCriteria);
            }
        }
        if let Some(heterogeneity) = self.heterogeneity {
            for sd in [
                heterogeneity.signal_mean_sd,
                heterogeneity.log_signal_sigma_sd,
                heterogeneity.criterion_offset_sd,
            ] {
                if !sd.is_finite() || sd < 0.0 {
                    return Err(DgpError::InvalidParticipantHeterogeneity);
                }
            }
        }
        if let Some(rt) = self.rt_channel {
            if !rt.base_latency_ms.is_finite()
                || !rt.evidence_speed_gain.is_finite()
                || !rt.log_noise_sd.is_finite()
                || !rt.non_decision_time_ms.is_finite()
                || rt.base_latency_ms <= 0.0
                || rt.evidence_speed_gain < 0.0
                || rt.log_noise_sd < 0.0
                || rt.non_decision_time_ms < 0.0
                || rt.quantization_ms == 0
                || rt.default_deadline_ms == Some(0)
            {
                return Err(DgpError::InvalidRtChannel);
            }
        }
        Ok(())
    }

    fn response_scale(&self, response_scale_id: u16) -> Option<&ResponseScaleSpec> {
        self.response_scales
            .iter()
            .find(|scale| scale.response_scale_id == response_scale_id)
    }
}

/// One experimenter-controlled assignment in a frozen schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrialRequest {
    pub action_id: u32,
    pub participant_id: u32,
    pub within_participant_trial: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtStatus {
    NotCollected,
    Observed,
    TimedOut,
    OutOfRange,
}

/// One synthetic trial. Fields marked oracle_* are generated truth and must
/// never be fed to an estimator as observed measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticTrial {
    pub action_id: u32,
    pub participant_id: u32,
    pub within_participant_trial: u32,
    pub stimulus_class: StimulusClass,
    pub stimulus_level: f64,
    pub response_scale_id: u16,
    pub oracle_evidence: f64,
    pub oracle_category: u16,
    pub oracle_lapse: bool,
    pub oracle_signal_mean: f64,
    pub oracle_signal_sigma: f64,
    pub oracle_criterion_offset: f64,
    pub observed_category: Option<u16>,
    pub rt_ms: Option<u32>,
    pub rt_status: RtStatus,
}

/// Per-action observed category counts. Missing categories and RT timeouts
/// are explicit denominators, not silently discarded rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionCategorySummary {
    pub action_id: u32,
    pub stimulus_class: StimulusClass,
    pub requested_trials: u64,
    pub category_counts: Vec<u64>,
    pub missing_category: u64,
    pub lapse_count_oracle: u64,
    pub rt_observed: u64,
    pub rt_timeouts: u64,
    pub rt_out_of_range: u64,
    pub rt_not_collected: u64,
}

/// Reproducible synthetic dataset receipt. Caller-supplied digests are opaque
/// commitments; this module does not hash, authenticate or sign the payload.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticDataset {
    pub scenario: ObserverScenario,
    pub action_set_digest: String,
    pub root_seed: u64,
    pub prng_version: &'static str,
    pub requests: Vec<TrialRequest>,
    pub trials: Vec<SyntheticTrial>,
    pub summaries: Vec<ActionCategorySummary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DgpError {
    NonFiniteScenarioParameter,
    InvalidSignalSigma,
    InvalidLapseProbability,
    InvalidMissingnessProbability,
    InvalidStudentTDegreesOfFreedom,
    NoResponseScales,
    TooManyResponseScales,
    DuplicateResponseScaleId,
    InvalidCriterionCount,
    InvalidLatentCriteria,
    InvalidParticipantHeterogeneity,
    InvalidRtChannel,
    MissingScenarioDigest,
    EmptyTrialSchedule,
    DuplicateParticipantTrial,
    UnknownActionId,
    MissingResponseScale,
    TrialCountOverflow,
    RtOutOfRange,
}

impl fmt::Display for DgpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NonFiniteScenarioParameter => "scenario parameters must be finite",
            Self::InvalidSignalSigma => "signal sigma must be positive",
            Self::InvalidLapseProbability => "lapse probability must be in [0, 1)",
            Self::InvalidMissingnessProbability => "missingness probabilities must define a valid [0, 1] bound",
            Self::InvalidStudentTDegreesOfFreedom => "variance-standardized Student-t requires df >= 3",
            Self::NoResponseScales => "scenario requires at least one response scale",
            Self::TooManyResponseScales => "too many response scales",
            Self::DuplicateResponseScaleId => "response-scale IDs must be unique",
            Self::InvalidCriterionCount => "each response scale needs at least two criteria",
            Self::InvalidLatentCriteria => "latent criteria must be finite and strictly increasing",
            Self::InvalidParticipantHeterogeneity => "heterogeneity standard deviations must be finite and nonnegative",
            Self::InvalidRtChannel => "RT channel parameters are invalid",
            Self::MissingScenarioDigest => "scenario digest must not be empty",
            Self::EmptyTrialSchedule => "trial schedule must not be empty",
            Self::DuplicateParticipantTrial => "participant/trial identities must be unique",
            Self::UnknownActionId => "trial request references an action outside the frozen action set",
            Self::MissingResponseScale => "action response scale is absent from the scenario",
            Self::TrialCountOverflow => "trial count overflowed u64",
            Self::RtOutOfRange => "generated RT cannot be represented in the receipt",
        };
        f.write_str(message)
    }
}

impl std::error::Error for DgpError {}

#[derive(Debug, Clone, Copy)]
struct ParticipantEffects {
    signal_mean: f64,
    log_signal_sigma: f64,
    criterion_offset: f64,
}

/// Generate a canonical, known-truth synthetic dataset.
///
/// Random streams are independently keyed by (root seed, scenario, participant,
/// trial index). Reordering the input request vector therefore cannot change
/// any trial. The input schedule itself determines signal/noise imbalance.
pub fn generate_synthetic_dataset(
    action_set: &OperationalActionSet,
    scenario: &ObserverScenario,
    requests: &[TrialRequest],
    scenario_digest: impl Into<String>,
    root_seed: u64,
) -> Result<SyntheticDataset, DgpError> {
    scenario.validate()?;
    let scenario_digest = scenario_digest.into();
    if scenario_digest.trim().is_empty() {
        return Err(DgpError::MissingScenarioDigest);
    }
    if requests.is_empty() {
        return Err(DgpError::EmptyTrialSchedule);
    }

    let mut canonical_requests = requests.to_vec();
    canonical_requests.sort_by_key(|request| {
        (request.participant_id, request.within_participant_trial)
    });
    if canonical_requests.windows(2).any(|pair| {
        pair[0].participant_id == pair[1].participant_id
            && pair[0].within_participant_trial == pair[1].within_participant_trial
    }) {
        return Err(DgpError::DuplicateParticipantTrial);
    }

    let mut action_by_id = BTreeMap::new();
    for action in action_set.actions() {
        action_by_id.insert(action.action_id, action);
    }
    for request in &canonical_requests {
        if !action_by_id.contains_key(&request.action_id) {
            return Err(DgpError::UnknownActionId);
        }
        let action = action_by_id[&request.action_id];
        if scenario.response_scale(action.response_scale_id).is_none() {
            return Err(DgpError::MissingResponseScale);
        }
    }

    let mut effects_by_participant = BTreeMap::new();
    let mut trials = Vec::with_capacity(canonical_requests.len());
    for request in &canonical_requests {
        let action = action_by_id[&request.action_id];
        let scale = scenario
            .response_scale(action.response_scale_id)
            .ok_or(DgpError::MissingResponseScale)?;
        let effects = *effects_by_participant
            .entry(request.participant_id)
            .or_insert_with(|| participant_effects(scenario, root_seed, request.participant_id));
        let stream_seed = trial_seed(
            root_seed,
            scenario.scenario_id,
            request.participant_id,
            request.within_participant_trial,
        );
        let mut rng = SplitMix64::new(stream_seed);

        let signal_mean = if action.stimulus_class == StimulusClass::Signal {
            scenario.base_signal_mean
                + scenario.signal_mean_per_stimulus_unit * action.stimulus_level
                + effects.signal_mean
        } else {
            0.0
        };
        let signal_sigma = (scenario.signal_sigma * effects.log_signal_sigma.exp()).max(f64::MIN_POSITIVE);
        let evidence_noise = sample_unit_evidence(&mut rng, scenario.evidence_family)?;
        let oracle_evidence = match action.stimulus_class {
            StimulusClass::Noise => evidence_noise,
            StimulusClass::Signal => signal_mean + signal_sigma * evidence_noise,
        };

        let trial_fraction = request.within_participant_trial as f64 / 100.0;
        let criterion_drift = scenario.criterion_drift_per_100_trials * trial_fraction;
        let criterion_offset = scenario.criterion_offset
            + effects.criterion_offset
            + criterion_drift;
        let shifted_criteria: Vec<f64> = scale
            .latent_criteria
            .iter()
            .map(|criterion| criterion + criterion_offset)
            .collect();
        let category_count = shifted_criteria.partition_point(|criterion| oracle_evidence >= *criterion);
        let oracle_category = u16::try_from(category_count).map_err(|_| DgpError::TrialCountOverflow)?;
        let oracle_lapse = rng.uniform() < scenario.lapse_probability;
        let oracle_category = if oracle_lapse {
            rng.uniform_index(scale.latent_criteria.len() + 1) as u16
        } else {
            oracle_category
        };

        let missing_probability = scenario.confidence_missing_probability
            + scenario.confidence_missing_evidence_slope
                * (oracle_evidence.abs() / (1.0 + oracle_evidence.abs()));
        let observed_category = if rng.uniform() < missing_probability {
            None
        } else {
            Some(oracle_category)
        };

        let (rt_ms, rt_status) = generate_rt(action.rt_deadline_ms, scenario.rt_channel, oracle_evidence, &mut rng)?;
        trials.push(SyntheticTrial {
            action_id: action.action_id,
            participant_id: request.participant_id,
            within_participant_trial: request.within_participant_trial,
            stimulus_class: action.stimulus_class,
            stimulus_level: action.stimulus_level,
            response_scale_id: action.response_scale_id,
            oracle_evidence,
            oracle_category,
            oracle_lapse,
            oracle_signal_mean: signal_mean,
            oracle_signal_sigma: signal_sigma,
            oracle_criterion_offset: criterion_offset,
            observed_category,
            rt_ms,
            rt_status,
        });
    }

    let summaries = summarize_trials(action_set, scenario, &trials)?;
    Ok(SyntheticDataset {
        scenario: scenario.clone(),
        action_set_digest: action_set.digest().to_owned(),
        root_seed,
        prng_version: "splitmix64-v1",
        requests: canonical_requests,
        trials,
        summaries,
    })
}

fn participant_effects(
    scenario: &ObserverScenario,
    root_seed: u64,
    participant_id: u32,
) -> ParticipantEffects {
    let mut rng = SplitMix64::new(mix64(
        root_seed
            ^ ((scenario.scenario_id as u64) << 48)
            ^ ((participant_id as u64) << 16)
            ^ 0xA076_1D64_78BD_642F,
    ));
    let heterogeneity = scenario.heterogeneity.unwrap_or(ParticipantHeterogeneity {
        signal_mean_sd: 0.0,
        log_signal_sigma_sd: 0.0,
        criterion_offset_sd: 0.0,
    });
    ParticipantEffects {
        signal_mean: heterogeneity.signal_mean_sd * rng.normal(),
        log_signal_sigma: heterogeneity.log_signal_sigma_sd * rng.normal(),
        criterion_offset: heterogeneity.criterion_offset_sd * rng.normal(),
    }
}

fn sample_unit_evidence(
    rng: &mut SplitMix64,
    family: EvidenceFamily,
) -> Result<f64, DgpError> {
    match family {
        EvidenceFamily::Gaussian => Ok(rng.normal()),
        EvidenceFamily::VarianceStandardizedStudentT { degrees_of_freedom } => {
            if degrees_of_freedom < 3 {
                return Err(DgpError::InvalidStudentTDegreesOfFreedom);
            }
            let numerator = rng.normal();
            let mut chi_square = 0.0;
            for _ in 0..degrees_of_freedom {
                let draw = rng.normal();
                chi_square += draw * draw;
            }
            let t = numerator / (chi_square / degrees_of_freedom as f64).sqrt();
            Ok(t * (((degrees_of_freedom - 2) as f64) / degrees_of_freedom as f64).sqrt())
        }
    }
}

fn generate_rt(
    action_deadline_ms: Option<u32>,
    channel: Option<RtChannelSpec>,
    evidence: f64,
    rng: &mut SplitMix64,
) -> Result<(Option<u32>, RtStatus), DgpError> {
    let Some(channel) = channel else {
        return Ok((None, RtStatus::NotCollected));
    };
    let log_noise = channel.log_noise_sd * rng.normal();
    let latency = channel.non_decision_time_ms
        + channel.base_latency_ms
            * (-channel.evidence_speed_gain * evidence.abs() + log_noise).exp();
    if !latency.is_finite() || latency < 0.0 {
        return Ok((None, RtStatus::OutOfRange));
    }
    let quantum = channel.quantization_ms as f64;
    let quantized = (latency / quantum).round() * quantum;
    let effective_deadline = action_deadline_ms.or(channel.default_deadline_ms);
    if effective_deadline.is_some_and(|deadline| quantized > deadline as f64) {
        return Ok((None, RtStatus::TimedOut));
    }
    if quantized > u32::MAX as f64 {
        return Ok((None, RtStatus::OutOfRange));
    }
    let rounded = (quantized as u32).max(1);
    Ok((Some(rounded), RtStatus::Observed))
}

fn summarize_trials(
    action_set: &OperationalActionSet,
    scenario: &ObserverScenario,
    trials: &[SyntheticTrial],
) -> Result<Vec<ActionCategorySummary>, DgpError> {
    let mut summaries = Vec::with_capacity(action_set.actions().len());
    let mut index_by_action = BTreeMap::new();
    for action in action_set.actions() {
        let scale = scenario
            .response_scale(action.response_scale_id)
            .ok_or(DgpError::MissingResponseScale)?;
        let index = summaries.len();
        index_by_action.insert(action.action_id, index);
        summaries.push(ActionCategorySummary {
            action_id: action.action_id,
            stimulus_class: action.stimulus_class,
            requested_trials: 0,
            category_counts: vec![0; scale.latent_criteria.len() + 1],
            missing_category: 0,
            lapse_count_oracle: 0,
            rt_observed: 0,
            rt_timeouts: 0,
            rt_out_of_range: 0,
            rt_not_collected: 0,
        });
    }
    for trial in trials {
        let index = *index_by_action
            .get(&trial.action_id)
            .ok_or(DgpError::UnknownActionId)?;
        let summary = &mut summaries[index];
        summary.requested_trials = summary
            .requested_trials
            .checked_add(1)
            .ok_or(DgpError::TrialCountOverflow)?;
        if trial.oracle_lapse {
            summary.lapse_count_oracle += 1;
        }
        if let Some(category) = trial.observed_category {
            let category_index = category as usize;
            let count = summary
                .category_counts
                .get_mut(category_index)
                .ok_or(DgpError::InvalidCriterionCount)?;
            *count += 1;
        } else {
            summary.missing_category += 1;
        }
        match trial.rt_status {
            RtStatus::NotCollected => summary.rt_not_collected += 1,
            RtStatus::Observed => summary.rt_observed += 1,
            RtStatus::TimedOut => summary.rt_timeouts += 1,
            RtStatus::OutOfRange => summary.rt_out_of_range += 1,
        }
    }
    Ok(summaries)
}

fn trial_seed(
    root_seed: u64,
    scenario_id: u16,
    participant_id: u32,
    trial_index: u32,
) -> u64 {
    mix64(
        root_seed
            ^ ((scenario_id as u64) << 48)
            ^ ((participant_id as u64) << 16)
            ^ (trial_index as u64).rotate_left(29)
            ^ 0xE703_7ED1_A0B4_28DB,
    )
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Uniform variate strictly inside (0, 1), avoiding log(0) in Box-Muller.
    fn uniform(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }

    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        radius * angle.cos()
    }

    fn uniform_index(&mut self, upper_exclusive: usize) -> usize {
        debug_assert!(upper_exclusive > 0);
        let upper = upper_exclusive as u64;
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let draw = self.next_u64();
            if draw >= threshold {
                return (draw % upper) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequential::{OperationalAction, OperationalActionSet};

    fn action_set() -> OperationalActionSet {
        OperationalActionSet::new(
            vec![
                OperationalAction {
                    action_id: 1,
                    audit_stratum: 1,
                    condition_id: 1,
                    stimulus_class: StimulusClass::Noise,
                    stimulus_level: 0.0,
                    response_scale_id: 5,
                    rt_deadline_ms: Some(1_800),
                },
                OperationalAction {
                    action_id: 2,
                    audit_stratum: 1,
                    condition_id: 1,
                    stimulus_class: StimulusClass::Signal,
                    stimulus_level: 1.0,
                    response_scale_id: 5,
                    rt_deadline_ms: Some(1_800),
                },
                OperationalAction {
                    action_id: 3,
                    audit_stratum: 2,
                    condition_id: 2,
                    stimulus_class: StimulusClass::Signal,
                    stimulus_level: 2.0,
                    response_scale_id: 5,
                    rt_deadline_ms: None,
                },
            ],
            "sha256:action-set",
        )
        .unwrap()
    }

    fn scenario() -> ObserverScenario {
        ObserverScenario {
            scenario_id: 7,
            evidence_family: EvidenceFamily::Gaussian,
            base_signal_mean: 1.2,
            signal_mean_per_stimulus_unit: 0.4,
            signal_sigma: 1.3,
            criterion_offset: 0.0,
            criterion_drift_per_100_trials: 0.15,
            lapse_probability: 0.02,
            confidence_missing_probability: 0.01,
            confidence_missing_evidence_slope: 0.15,
            response_scales: vec![ResponseScaleSpec {
                response_scale_id: 5,
                latent_criteria: vec![-1.0, -0.3, 0.3, 1.0],
            }],
            heterogeneity: Some(ParticipantHeterogeneity {
                signal_mean_sd: 0.2,
                log_signal_sigma_sd: 0.1,
                criterion_offset_sd: 0.15,
            }),
            rt_channel: Some(RtChannelSpec {
                base_latency_ms: 700.0,
                evidence_speed_gain: 0.18,
                log_noise_sd: 0.2,
                non_decision_time_ms: 180.0,
                quantization_ms: 10,
                default_deadline_ms: Some(1_900),
            }),
        }
    }

    fn requests() -> Vec<TrialRequest> {
        (0..100)
            .map(|trial| TrialRequest {
                action_id: if trial % 4 == 0 { 1 } else if trial % 4 == 1 { 2 } else { 3 },
                participant_id: (trial % 5) as u32,
                within_participant_trial: (trial / 5) as u32,
            })
            .collect()
    }

    #[test]
    fn generation_is_exactly_replayable() {
        let actions = action_set();
        let scenario = scenario();
        let requests = requests();
        let first = generate_synthetic_dataset(
            &actions,
            &scenario,
            &requests,
            "sha256:scenario",
            42,
        )
        .unwrap();
        let second = generate_synthetic_dataset(
            &actions,
            &scenario,
            &requests,
            "sha256:scenario",
            42,
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.prng_version, "splitmix64-v1");
    }

    #[test]
    fn request_permutation_does_not_change_generated_trials() {
        let actions = action_set();
        let scenario = scenario();
        let mut requests = requests();
        let first = generate_synthetic_dataset(
            &actions,
            &scenario,
            &requests,
            "sha256:scenario",
            77,
        )
        .unwrap();
        requests.reverse();
        let second = generate_synthetic_dataset(
            &actions,
            &scenario,
            &requests,
            "sha256:scenario",
            77,
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn signal_noise_imbalance_follows_schedule_and_counts_keep_missingness() {
        let actions = action_set();
        let mut schedule = requests();
        schedule.truncate(40);
        let data = generate_synthetic_dataset(
            &actions,
            &scenario(),
            &schedule,
            "sha256:scenario",
            123,
        )
        .unwrap();
        let total: u64 = data.summaries.iter().map(|summary| summary.requested_trials).sum();
        assert_eq!(total, 40);
        assert!(data
            .summaries
            .iter()
            .find(|summary| summary.action_id == 1)
            .unwrap()
            .requested_trials
            > data
                .summaries
                .iter()
                .find(|summary| summary.action_id == 2)
                .unwrap()
                .requested_trials);
        for summary in data.summaries {
            assert_eq!(
                summary.category_counts.iter().sum::<u64>() + summary.missing_category,
                summary.requested_trials
            );
        }
    }

    #[test]
    fn stimulus_level_affects_signal_mean_but_not_noise_mean() {
        let actions = action_set();
        let schedule = (0..4_000)
            .map(|trial| TrialRequest {
                action_id: if trial % 2 == 0 { 2 } else { 3 },
                participant_id: 0,
                within_participant_trial: trial,
            })
            .collect::<Vec<_>>();
        let data = generate_synthetic_dataset(
            &actions,
            &scenario(),
            &schedule,
            "sha256:scenario",
            2026,
        )
        .unwrap();
        let means = data
            .trials
            .iter()
            .filter(|trial| trial.stimulus_class == StimulusClass::Signal)
            .fold(BTreeMap::<u32, (f64, u64)>::new(), |mut acc, trial| {
                let entry = acc.entry(trial.action_id).or_insert((0.0, 0));
                entry.0 += trial.oracle_evidence;
                entry.1 += 1;
                acc
            });
        let low = means[&2].0 / means[&2].1 as f64;
        let high = means[&3].0 / means[&3].1 as f64;
        assert!(high > low + 0.2);
    }

    #[test]
    fn participant_effects_are_stable_within_identity() {
        let actions = action_set();
        let schedule = (0..200)
            .map(|trial| TrialRequest {
                action_id: 2,
                participant_id: (trial % 2) as u32,
                within_participant_trial: (trial / 2) as u32,
            })
            .collect::<Vec<_>>();
        let data = generate_synthetic_dataset(
            &actions,
            &scenario(),
            &schedule,
            "sha256:scenario",
            7,
        )
        .unwrap();
        let mut means = BTreeMap::<u32, f64>::new();
        for trial in &data.trials {
            means.entry(trial.participant_id)
                .or_insert(trial.oracle_signal_mean);
            assert_eq!(
                *means.get(&trial.participant_id).unwrap(),
                trial.oracle_signal_mean
            );
        }
        assert_ne!(means[&0], means[&1]);
    }

    #[test]
    fn lapse_missingness_and_rt_are_explicitly_preserved() {
        let actions = action_set();
        let mut spec = scenario();
        spec.lapse_probability = 0.5;
        spec.confidence_missing_probability = 0.2;
        spec.confidence_missing_evidence_slope = 0.5;
        let data = generate_synthetic_dataset(
            &actions,
            &spec,
            &requests(),
            "sha256:scenario",
            101,
        )
        .unwrap();
        assert!(data.trials.iter().any(|trial| trial.oracle_lapse));
        assert!(data.trials.iter().any(|trial| trial.observed_category.is_none()));
        assert!(data.trials.iter().any(|trial| trial.rt_status == RtStatus::Observed));
    }

    #[test]
    fn student_t_generator_has_unit_variance_and_heavy_tails() {
        let mut rng = SplitMix64::new(999);
        let draws = (0..80_000)
            .map(|_| {
                sample_unit_evidence(
                    &mut rng,
                    EvidenceFamily::VarianceStandardizedStudentT {
                        degrees_of_freedom: 5,
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let variance = draws.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
            / (draws.len() - 1) as f64;
        let tail = draws.iter().filter(|value| value.abs() > 3.0).count() as f64
            / draws.len() as f64;
        assert!(mean.abs() < 0.03);
        assert!((variance - 1.0).abs() < 0.10);
        assert!(tail > 0.01);
    }

    #[test]
    fn rt_quantization_creates_ties_and_deadline_produces_timeouts() {
        let actions = action_set();
        let mut spec = scenario();
        spec.rt_channel = Some(RtChannelSpec {
            base_latency_ms: 1_500.0,
            evidence_speed_gain: 0.0,
            log_noise_sd: 0.0,
            non_decision_time_ms: 0.0,
            quantization_ms: 100,
            default_deadline_ms: Some(800),
        });
        let schedule = (0..100)
            .map(|trial| TrialRequest {
                action_id: if trial % 2 == 0 { 1 } else { 2 },
                participant_id: 0,
                within_participant_trial: trial,
            })
            .collect::<Vec<_>>();
        let data = generate_synthetic_dataset(
            &actions,
            &spec,
            &schedule,
            "sha256:scenario",
            5,
        )
        .unwrap();
        assert!(data.trials.iter().all(|trial| trial.rt_status == RtStatus::TimedOut));
        assert_eq!(data.trials.len(), 100);
    }

    #[test]
    fn malformed_scenarios_duplicate_trials_and_unknown_actions_reject() {
        let actions = action_set();
        let mut invalid = scenario();
        invalid.signal_sigma = 0.0;
        assert_eq!(invalid.validate(), Err(DgpError::InvalidSignalSigma));
        invalid = scenario();
        invalid.response_scales[0].latent_criteria[1] = -2.0;
        assert_eq!(invalid.validate(), Err(DgpError::InvalidLatentCriteria));
        invalid = scenario();
        invalid.evidence_family = EvidenceFamily::VarianceStandardizedStudentT {
            degrees_of_freedom: 2,
        };
        assert_eq!(invalid.validate(), Err(DgpError::InvalidStudentTDegreesOfFreedom));

        let duplicate = vec![
            TrialRequest { action_id: 1, participant_id: 0, within_participant_trial: 0 },
            TrialRequest { action_id: 2, participant_id: 0, within_participant_trial: 0 },
        ];
        assert_eq!(
            generate_synthetic_dataset(&actions, &scenario(), &duplicate, "sha256:scenario", 1),
            Err(DgpError::DuplicateParticipantTrial)
        );
        let unknown = vec![
            TrialRequest { action_id: 99, participant_id: 0, within_participant_trial: 0 },
        ];
        assert_eq!(
            generate_synthetic_dataset(&actions, &scenario(), &unknown, "sha256:scenario", 1),
            Err(DgpError::UnknownActionId)
        );
    }
}
