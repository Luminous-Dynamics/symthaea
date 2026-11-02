#![forbid(unsafe_code)]

//! SPECTRA-001B deterministic synthetic campaign generator and leakage baseline.
//!
//! Candidate-facing evaluation observations are separated from evaluator-only
//! target truth. The worlds are synthetic fixtures, not physical side-channel
//! models. SplitMix64 is used for reproducibility, not cryptographic security.

use serde::{Deserialize, Serialize};
use spectra_001_qualifier::{
    evaluate_trial_v1, BoundedTimeV1, EvaluatorTruthV1 as TrialTruthV1,
    JudgeRankingV1, LeakageModalityV1, TrialV1,
    ValidationErrorV1,
};
use std::collections::BTreeMap;

pub const CORPUS_SCHEMA_V0_1: &str = "symthaea.spectra.synthetic_corpus.v0.1";
pub const GENERATOR_VERSION_V0_1: &str = "splitmix64-fy-reject-v0.1";
pub const MAX_CAMPAIGN_TRIALS: u32 = 10_000;
pub const MIN_CAMPAIGN_TRIALS: u32 = 16;
pub const MAX_CANDIDATES: u8 = 64;
pub const MIN_CANDIDATES: u8 = 2;
pub const LEAKAGE_CHANNEL_COUNT: usize = 9;
pub const RED_TEAM_THRESHOLD_PERMILLE: u16 = 850;
pub const RED_TEAM_MINIMUM_PREDICTIONS: u32 = 64;

pub const LEAKAGE_MODALITIES_V1: [LeakageModalityV1; LEAKAGE_CHANNEL_COUNT] = [
    LeakageModalityV1::Rf,
    LeakageModalityV1::LowFrequencyMagnetic,
    LeakageModalityV1::Acoustic,
    LeakageModalityV1::Optical,
    LeakageModalityV1::Vibration,
    LeakageModalityV1::ThermalAirflow,
    LeakageModalityV1::PowerConducted,
    LeakageModalityV1::NetworkTiming,
    LeakageModalityV1::Procedural,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignConfigV1 {
    /// Caller-supplied opaque run identity. It must not encode target or world labels.
    pub campaign_instance_id: u64,
    pub calibration_trials: u32,
    pub evaluation_trials: u32,
    pub candidate_count: u8,
    /// Probability that a synthetic injected signal ranks the target first.
    pub signal_hit_rate_permille: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignWorldV1 {
    Null,
    SignalInjected,
    KnownLeakageInjected,
    Confounded,
    AdversarialMutation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryV1 {
    /// Categorical synthetic features, not calibrated physical measurements.
    pub channel_codes: [Option<u8>; LEAKAGE_CHANNEL_COUNT],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicObservationV1 {
    pub trial: TrialV1,
    pub telemetry: TelemetryV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationRowV1 {
    /// The calibration target is revealed to the red-team training baseline.
    pub observation: PublicObservationV1,
    pub target_index: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationRowV1 {
    /// No hidden target label is included in evaluation observations.
    pub observation: PublicObservationV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCampaignV1 {
    pub schema: String,
    pub generator_version: String,
    /// Opaque identity; does not encode the hidden world, seed or target.
    pub campaign_id: String,
    pub candidate_count: u8,
    pub calibration_rows: Vec<CalibrationRowV1>,
    pub evaluation_rows: Vec<EvaluationRowV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignTruthV1 {
    pub schema: String,
    pub world: CampaignWorldV1,
    pub seed: u64,
    pub injected_leakage_modality: Option<LeakageModalityV1>,
    pub evaluation_target_indices: Vec<u8>,
    pub evaluation_target_ids: Vec<String>,
    pub expected_chronology_valid: bool,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedCampaignV1 {
    /// Candidate-facing data. No evaluation target labels, seed or world tag.
    pub public: PublicCampaignV1,
    /// Evaluator-only truth. Never pass this sidecar to the candidate or attacker.
    pub evaluator_truth: CampaignTruthV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorpusErrorV1 {
    InvalidCampaignId,
    InvalidCampaignSize,
    InvalidCandidateCount,
    InvalidSignalRate,
    WorldLeakageMismatch,
    EmptyCalibration,
    TruthLengthMismatch,
    InvalidEvaluationIndex,
    DuplicatePrediction,
    InvalidPredictionTarget,
    InvalidTrialRecord(Vec<ValidationErrorV1>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupModelV1 {
    candidate_count: u8,
    maps: [BTreeMap<u8, u8>; LEAKAGE_CHANNEL_COUNT],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelPredictionV1 {
    pub evaluation_index: u32,
    pub modality: LeakageModalityV1,
    pub predicted_target_index: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelScoreV1 {
    pub modality: LeakageModalityV1,
    pub prediction_count: u32,
    pub correct_count: u32,
    pub accuracy_permille: Option<u16>,
    pub detected_at_declared_threshold: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RedTeamReportV1 {
    pub schema: String,
    pub campaign_id: String,
    pub threshold_permille: u16,
    pub minimum_predictions: u32,
    pub channel_scores: Vec<ChannelScoreV1>,
    pub best_channel_accuracy_permille: Option<u16>,
    pub detected_modalities: Vec<LeakageModalityV1>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Rejection sampling prevents modulo bias in bounded categorical draws.
    fn bounded(&mut self, upper_exclusive: u64) -> u64 {
        assert!(upper_exclusive > 0);
        let threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return value % upper_exclusive;
            }
        }
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let other = self.bounded((index + 1) as u64) as usize;
            values.swap(index, other);
        }
    }
}

fn modality_index(modality: &LeakageModalityV1) -> usize {
    match modality {
        LeakageModalityV1::Rf => 0,
        LeakageModalityV1::LowFrequencyMagnetic => 1,
        LeakageModalityV1::Acoustic => 2,
        LeakageModalityV1::Optical => 3,
        LeakageModalityV1::Vibration => 4,
        LeakageModalityV1::ThermalAirflow => 5,
        LeakageModalityV1::PowerConducted => 6,
        LeakageModalityV1::NetworkTiming => 7,
        LeakageModalityV1::Procedural => 8,
    }
}

fn validate_config(
    config: CampaignConfigV1,
    world: &CampaignWorldV1,
    leakage: &Option<LeakageModalityV1>,
) -> Result<(), CorpusErrorV1> {
    if config.campaign_instance_id == 0 {
        return Err(CorpusErrorV1::InvalidCampaignId);
    }
    if !(MIN_CAMPAIGN_TRIALS..=MAX_CAMPAIGN_TRIALS).contains(&config.calibration_trials)
        || !(MIN_CAMPAIGN_TRIALS..=MAX_CAMPAIGN_TRIALS).contains(&config.evaluation_trials)
    {
        return Err(CorpusErrorV1::InvalidCampaignSize);
    }
    if !(MIN_CANDIDATES..=MAX_CANDIDATES).contains(&config.candidate_count) {
        return Err(CorpusErrorV1::InvalidCandidateCount);
    }
    if config.signal_hit_rate_permille > 1_000 {
        return Err(CorpusErrorV1::InvalidSignalRate);
    }
    let annotation_matches = match world {
        CampaignWorldV1::KnownLeakageInjected => leakage.is_some(),
        _ => leakage.is_none(),
    };
    if !annotation_matches {
        return Err(CorpusErrorV1::WorldLeakageMismatch);
    }
    Ok(())
}

fn candidate_ids(candidate_count: u8) -> Vec<String> {
    (0..candidate_count)
        .map(|index| format!("candidate-{index:02}"))
        .collect()
}

fn generate_rank_order(
    rng: &mut SplitMix64,
    candidate_count: u8,
    target_index: u8,
    inject_signal: bool,
    signal_hit_rate_permille: u16,
) -> Vec<String> {
    let ids = candidate_ids(candidate_count);
    let mut indices: Vec<u8> = (0..candidate_count).collect();
    rng.shuffle(&mut indices);

    let force_target_first = inject_signal
        && rng.bounded(1_000) < signal_hit_rate_permille as u64;

    if force_target_first {
        indices.retain(|index| *index != target_index);
        indices.insert(0, target_index);
    }

    indices
        .into_iter()
        .map(|index| ids[index as usize].clone())
        .collect()
}

fn make_observation(
    rng: &mut SplitMix64,
    config: CampaignConfigV1,
    trial_index: u32,
    target_index: u8,
    inject_signal: bool,
    leakage: &Option<LeakageModalityV1>,
    confounded: bool,
    invalid_chronology: bool,
) -> (PublicObservationV1, TrialTruthV1) {
    let ids = candidate_ids(config.candidate_count);
    let target_id = ids[target_index as usize].clone();
    let base = (trial_index as u64) * 1_000;
    let judge_order = generate_rank_order(
        rng,
        config.candidate_count,
        target_index,
        inject_signal,
        config.signal_hit_rate_permille,
    );

    let trial_id = match leakage {
        Some(LeakageModalityV1::Procedural) => {
            format!("trial-{trial_index:06}-target-{target_index:02}")
        }
        _ => format!("trial-{trial_index:06}"),
    };

    let mut channel_codes = [None; LEAKAGE_CHANNEL_COUNT];
    for channel in 0..LEAKAGE_CHANNEL_COUNT {
        let random_code = rng.bounded(config.candidate_count as u64) as u8;
        channel_codes[channel] = Some(match leakage {
            Some(modality)
                if modality_index(modality) == channel
                    && !matches!(modality, LeakageModalityV1::Procedural) =>
            {
                target_index
            }
            _ => random_code,
        });
    }

    let response_sealed = if invalid_chronology {
        (base + 500, base + 501)
    } else {
        (base + 300, base + 301)
    };

    let trial = TrialV1 {
        trial_id,
        participant_id: if confounded {
            format!("participant-{target_index:02}")
        } else {
            format!("participant-{:02}", trial_index % 4)
        },
        operator_id: "operator-01".to_owned(),
        target_custodian_id: "custodian-01".to_owned(),
        evaluator_id: "evaluator-01".to_owned(),
        candidate_ids: ids,
        protocol_sha256: "a".repeat(64),
        target_pool_sha256: "b".repeat(64),
        target_selection_receipt_sha256: "c".repeat(64),
        response_sha256: "d".repeat(64),
        apparatus_locked_at: BoundedTimeV1 {
            clock_domain_id: "synthetic-clock-01".to_owned(),
            earliest_ns: base + 10,
            latest_ns: base + 11,
        },
        target_selected_at: BoundedTimeV1 {
            clock_domain_id: "synthetic-clock-01".to_owned(),
            earliest_ns: base + 100,
            latest_ns: base + 101,
        },
        response_sealed_at: BoundedTimeV1 {
            clock_domain_id: "synthetic-clock-01".to_owned(),
            earliest_ns: response_sealed.0,
            latest_ns: response_sealed.1,
        },
        target_revealed_at: BoundedTimeV1 {
            clock_domain_id: "synthetic-clock-01".to_owned(),
            earliest_ns: base + 400,
            latest_ns: base + 401,
        },
        judge_rankings: vec![JudgeRankingV1 {
            judge_id: format!("judge-{trial_index:06}"),
            ranked_candidate_ids: judge_order,
        }],
    };

    (
        PublicObservationV1 {
            trial,
            telemetry: TelemetryV1 { channel_codes },
        },
        TrialTruthV1 { target_id },
    )
}

/// Generate separate candidate-facing and evaluator-only campaign artifacts.
///
/// Null: target-independent response and telemetry.
/// SignalInjected: target is ranked first with the configured probability.
/// KnownLeakageInjected: one declared synthetic telemetry channel encodes target.
/// Confounded: participant identity is associated with target assignment while
///             the response remains target-independent.
/// AdversarialMutation: response-seal time follows target-reveal time.
pub fn generate_campaign_v1(
    config: CampaignConfigV1,
    seed: u64,
    world: CampaignWorldV1,
    leakage: Option<LeakageModalityV1>,
) -> Result<GeneratedCampaignV1, CorpusErrorV1> {
    validate_config(config, &world, &leakage)?;
    let mut rng = SplitMix64::new(seed);

    let mut calibration_rows = Vec::with_capacity(config.calibration_trials as usize);
    for index in 0..config.calibration_trials {
        let target_index = rng.bounded(config.candidate_count as u64) as u8;
        let (observation, _) = make_observation(
            &mut rng,
            config,
            index,
            target_index,
            world == CampaignWorldV1::SignalInjected,
            &leakage,
            world == CampaignWorldV1::Confounded,
            false,
        );
        calibration_rows.push(CalibrationRowV1 {
            observation,
            target_index,
        });
    }

    let mut evaluation_rows = Vec::with_capacity(config.evaluation_trials as usize);
    let mut evaluation_target_ids = Vec::with_capacity(config.evaluation_trials as usize);
    let mut evaluation_target_indices = Vec::with_capacity(config.evaluation_trials as usize);
    let evaluation_offset = config.calibration_trials;

    for offset in 0..config.evaluation_trials {
        let index = evaluation_offset + offset;
        let target_index = rng.bounded(config.candidate_count as u64) as u8;
        let (observation, truth) = make_observation(
            &mut rng,
            config,
            index,
            target_index,
            world == CampaignWorldV1::SignalInjected,
            &leakage,
            world == CampaignWorldV1::Confounded,
            world == CampaignWorldV1::AdversarialMutation,
        );
        evaluation_rows.push(EvaluationRowV1 { observation });
        evaluation_target_ids.push(truth.target_id);
        evaluation_target_indices.push(target_index);
    }

    let expected_chronology_valid = world != CampaignWorldV1::AdversarialMutation;

    Ok(GeneratedCampaignV1 {
        public: PublicCampaignV1 {
            schema: CORPUS_SCHEMA_V0_1.to_owned(),
            generator_version: GENERATOR_VERSION_V0_1.to_owned(),
            campaign_id: format!("spectra-{0:016x}", config.campaign_instance_id),
            candidate_count: config.candidate_count,
            calibration_rows,
            evaluation_rows,
        },
        evaluator_truth: CampaignTruthV1 {
            schema: CORPUS_SCHEMA_V0_1.to_owned(),
            world,
            seed,
            injected_leakage_modality: leakage,
            evaluation_target_indices,
            evaluation_target_ids,
            expected_chronology_valid,
            claim_ceiling: "synthetic_only".to_owned(),
        },
    })
}

fn observable_code(
    observation: &PublicObservationV1,
    modality: &LeakageModalityV1,
) -> Option<u8> {
    match modality {
        LeakageModalityV1::Procedural => observation
            .trial
            .trial_id
            .rsplit_once("-target-")
            .and_then(|(_, suffix)| suffix.parse::<u8>().ok())
            .or_else(|| {
                observation
                    .trial
                    .participant_id
                    .strip_prefix("participant-")
                    .and_then(|suffix| suffix.parse::<u8>().ok())
            }),
        _ => observation.telemetry.channel_codes[modality_index(modality)],
    }
}

/// Fit a transparent per-channel lookup attacker from revealed calibration rows.
pub fn fit_red_team_v1(
    public: &PublicCampaignV1,
) -> Result<LookupModelV1, CorpusErrorV1> {
    if public.calibration_rows.is_empty() {
        return Err(CorpusErrorV1::EmptyCalibration);
    }

    let mut counts: [BTreeMap<u8, BTreeMap<u8, u32>>; LEAKAGE_CHANNEL_COUNT] =
        std::array::from_fn(|_| BTreeMap::new());

    for row in &public.calibration_rows {
        if row.target_index >= public.candidate_count {
            return Err(CorpusErrorV1::InvalidPredictionTarget);
        }
        for modality in &LEAKAGE_MODALITIES_V1 {
            if let Some(code) = observable_code(&row.observation, modality) {
                *counts[modality_index(modality)]
                    .entry(code)
                    .or_default()
                    .entry(row.target_index)
                    .or_default() += 1;
            }
        }
    }

    let mut maps: [BTreeMap<u8, u8>; LEAKAGE_CHANNEL_COUNT] =
        std::array::from_fn(|_| BTreeMap::new());

    for channel in 0..LEAKAGE_CHANNEL_COUNT {
        for (code, target_counts) in &counts[channel] {
            let mut best = None::<(u8, u32)>;
            for (&target, &count) in target_counts {
                let replace = match best {
                    None => true,
                    Some((best_target, best_count)) => {
                        count > best_count || (count == best_count && target < best_target)
                    }
                };
                if replace {
                    best = Some((target, count));
                }
            }
            let (best_target, _) =
                best.ok_or(CorpusErrorV1::InvalidPredictionTarget)?;
            maps[channel].insert(*code, best_target);
        }
    }

    Ok(LookupModelV1 {
        candidate_count: public.candidate_count,
        maps,
    })
}

/// Predict hidden evaluation targets using public telemetry only.
pub fn predict_red_team_v1(
    model: &LookupModelV1,
    public: &PublicCampaignV1,
) -> Result<Vec<ChannelPredictionV1>, CorpusErrorV1> {
    if model.candidate_count != public.candidate_count {
        return Err(CorpusErrorV1::InvalidCandidateCount);
    }

    let mut predictions = Vec::new();
    for (evaluation_index, row) in public.evaluation_rows.iter().enumerate() {
        for modality in &LEAKAGE_MODALITIES_V1 {
            let Some(code) = observable_code(&row.observation, modality) else {
                continue;
            };
            let Some(target) = model.maps[modality_index(modality)].get(&code).copied() else {
                continue;
            };
            predictions.push(ChannelPredictionV1 {
                evaluation_index: evaluation_index as u32,
                modality: modality.clone(),
                predicted_target_index: target,
            });
        }
    }
    Ok(predictions)
}

/// Score attacker predictions against evaluator-only truth.
pub fn score_red_team_v1(
    public: &PublicCampaignV1,
    truth: &CampaignTruthV1,
    predictions: &[ChannelPredictionV1],
    threshold_permille: u16,
    minimum_predictions: u32,
) -> Result<RedTeamReportV1, CorpusErrorV1> {
    if truth.evaluation_target_indices.len() != public.evaluation_rows.len()
        || truth.evaluation_target_ids.len() != public.evaluation_rows.len()
    {
        return Err(CorpusErrorV1::TruthLengthMismatch);
    }
    if threshold_permille > 1_000 {
        return Err(CorpusErrorV1::InvalidSignalRate);
    }

    let mut seen = BTreeMap::<(u32, usize), ()>::new();
    let mut counts = [0_u32; LEAKAGE_CHANNEL_COUNT];
    let mut correct = [0_u32; LEAKAGE_CHANNEL_COUNT];

    for prediction in predictions {
        let index = prediction.evaluation_index as usize;
        if index >= truth.evaluation_target_indices.len() {
            return Err(CorpusErrorV1::InvalidEvaluationIndex);
        }
        if prediction.predicted_target_index >= public.candidate_count {
            return Err(CorpusErrorV1::InvalidPredictionTarget);
        }
        let channel = modality_index(&prediction.modality);
        if seen.insert((prediction.evaluation_index, channel), ()).is_some() {
            return Err(CorpusErrorV1::DuplicatePrediction);
        }
        counts[channel] += 1;
        if prediction.predicted_target_index == truth.evaluation_target_indices[index] {
            correct[channel] += 1;
        }
    }

    let mut channel_scores = Vec::with_capacity(LEAKAGE_CHANNEL_COUNT);
    let mut detected_modalities = Vec::new();
    let mut best_accuracy = None::<u16>;

    for channel in 0..LEAKAGE_CHANNEL_COUNT {
        let accuracy = if counts[channel] == 0 {
            None
        } else {
            Some(((correct[channel] as u64 * 1_000) / counts[channel] as u64) as u16)
        };
        let detected = counts[channel] >= minimum_predictions
            && accuracy.is_some_and(|value| value >= threshold_permille);
        if detected {
            detected_modalities.push(LEAKAGE_MODALITIES_V1[channel].clone());
        }
        if let Some(value) = accuracy {
            best_accuracy = Some(best_accuracy.map_or(value, |best| best.max(value)));
        }
        channel_scores.push(ChannelScoreV1 {
            modality: LEAKAGE_MODALITIES_V1[channel].clone(),
            prediction_count: counts[channel],
            correct_count: correct[channel],
            accuracy_permille: accuracy,
            detected_at_declared_threshold: detected,
        });
    }

    Ok(RedTeamReportV1 {
        schema: "symthaea.spectra.red_team_report.v0.1".to_owned(),
        campaign_id: public.campaign_id.clone(),
        threshold_permille,
        minimum_predictions,
        channel_scores,
        best_channel_accuracy_permille: best_accuracy,
        detected_modalities,
        claim_ceiling: "synthetic_only".to_owned(),
    })
}

pub fn top_rank_accuracy_permille_v1(
    public: &PublicCampaignV1,
    truth: &CampaignTruthV1,
) -> Result<u16, CorpusErrorV1> {
    if truth.evaluation_target_ids.len() != public.evaluation_rows.len()
        || public.evaluation_rows.is_empty()
    {
        return Err(CorpusErrorV1::TruthLengthMismatch);
    }

    let mut hits = 0_u64;
    for (index, row) in public.evaluation_rows.iter().enumerate() {
        let target_id = &truth.evaluation_target_ids[index];
        if row
            .observation
            .trial
            .judge_rankings
            .first()
            .is_some_and(|judge| judge.ranked_candidate_ids.first() == Some(target_id))
        {
            hits += 1;
        }
    }
    Ok(((hits * 1_000) / public.evaluation_rows.len() as u64) as u16)
}

/// Validate every evaluation trial using the independent trial-level scorer.
pub fn validate_evaluation_trials_v1(
    public: &PublicCampaignV1,
    truth: &CampaignTruthV1,
) -> Result<(usize, usize), CorpusErrorV1> {
    if truth.evaluation_target_ids.len() != public.evaluation_rows.len() {
        return Err(CorpusErrorV1::TruthLengthMismatch);
    }

    let mut valid = 0;
    let mut invalid = 0;
    for (index, row) in public.evaluation_rows.iter().enumerate() {
        let trial_truth = TrialTruthV1 {
            target_id: truth.evaluation_target_ids[index].clone(),
        };
        match evaluate_trial_v1(&row.observation.trial, &trial_truth) {
            Ok(_) => valid += 1,
            Err(_) => invalid += 1,
        }
    }
    Ok((valid, invalid))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> CampaignConfigV1 {
        CampaignConfigV1 {
            campaign_instance_id: 600_000_001,
            calibration_trials: 512,
            evaluation_trials: 512,
            candidate_count: 4,
            signal_hit_rate_permille: 400,
        }
    }

    #[test]
    fn null_world_is_deterministic_and_has_no_leakage_alert() {
        let first = generate_campaign_v1(
            config(),
            0x5eed_1234_9876_abcd,
            CampaignWorldV1::Null,
            None,
        )
        .expect("valid null campaign");
        let second = generate_campaign_v1(
            config(),
            0x5eed_1234_9876_abcd,
            CampaignWorldV1::Null,
            None,
        )
        .expect("same null campaign");
        assert_eq!(first, second);

        let model = fit_red_team_v1(&first.public).expect("fit red team");
        let predictions = predict_red_team_v1(&model, &first.public).expect("predict");
        let report = score_red_team_v1(
            &first.public,
            &first.evaluator_truth,
            &predictions,
            RED_TEAM_THRESHOLD_PERMILLE,
            RED_TEAM_MINIMUM_PREDICTIONS,
        )
        .expect("score");
        assert!(report.detected_modalities.is_empty());
    }

    #[test]
    fn all_nine_injected_channels_are_recoverable_by_the_baseline() {
        for modality in &LEAKAGE_MODALITIES_V1 {
            let campaign = generate_campaign_v1(
                config(),
                0xface_1000_0000_0001 + modality_index(modality) as u64,
                CampaignWorldV1::KnownLeakageInjected,
                Some(modality.clone()),
            )
            .expect("valid leakage campaign");
            let model = fit_red_team_v1(&campaign.public).expect("fit red team");
            let predictions = predict_red_team_v1(&model, &campaign.public).expect("predict");
            let report = score_red_team_v1(
                &campaign.public,
                &campaign.evaluator_truth,
                &predictions,
                RED_TEAM_THRESHOLD_PERMILLE,
                RED_TEAM_MINIMUM_PREDICTIONS,
            )
            .expect("score");
            assert!(
                report.detected_modalities.contains(modality),
                "expected modality {modality:?} to be detected"
            );
            let score = report
                .channel_scores
                .iter()
                .find(|row| &row.modality == modality)
                .expect("channel score");
            assert_eq!(score.accuracy_permille, Some(1_000));
        }
    }

    #[test]
    fn injected_response_signal_changes_rank_but_not_telemetry() {
        let campaign = generate_campaign_v1(
            config(),
            0x1111_2222_3333_4444,
            CampaignWorldV1::SignalInjected,
            None,
        )
        .expect("valid signal campaign");
        let accuracy =
            top_rank_accuracy_permille_v1(&campaign.public, &campaign.evaluator_truth)
                .expect("top-rank score");
        assert!(accuracy >= 430, "expected a clear injected response effect, got {accuracy}");

        let model = fit_red_team_v1(&campaign.public).expect("fit red team");
        let predictions = predict_red_team_v1(&model, &campaign.public).expect("predict");
        let report = score_red_team_v1(
            &campaign.public,
            &campaign.evaluator_truth,
            &predictions,
            RED_TEAM_THRESHOLD_PERMILLE,
            RED_TEAM_MINIMUM_PREDICTIONS,
        )
        .expect("score");
        assert!(report.detected_modalities.is_empty());
    }

    #[test]
    fn adversarial_world_rejects_response_sealed_after_reveal() {
        let campaign = generate_campaign_v1(
            config(),
            0x7777_8888_9999_aaaa,
            CampaignWorldV1::AdversarialMutation,
            None,
        )
        .expect("valid adversarial campaign");
        let (valid, invalid) =
            validate_evaluation_trials_v1(&campaign.public, &campaign.evaluator_truth)
                .expect("matching evaluator truth");
        assert_eq!(valid, 0);
        assert_eq!(invalid, config().evaluation_trials as usize);
    }

    #[test]
    fn public_json_contains_no_hidden_evaluation_truth() {
        let campaign = generate_campaign_v1(
            config(),
            123,
            CampaignWorldV1::Null,
            None,
        )
        .expect("valid campaign");
        let json = serde_json::to_string(&campaign.public).expect("serialize public campaign");
        assert!(!json.contains("\"target_id\""));
        assert!(!json.contains("\"evaluation_target_indices\""));
        assert!(!json.contains("\"seed\""));
        assert!(!json.contains("\"world\""));
        assert!(
            campaign
                .public
                .evaluation_rows
                .iter()
                .all(|row| row.observation.trial.candidate_ids.iter()
                    .all(|id| id.starts_with("candidate-")))
        );
    }


    #[test]
    fn null_and_signal_campaigns_match_independent_known_answers() {
        let fixture: serde_json::Value = serde_json::from_str(
            include_str!("../../../research/spectra-001/fixtures/corpus-known-answer-v0.1.json")
        ).expect("known-answer fixture JSON");

        let config = config();
        let null_seed = u64::from_str_radix(
            fixture["scenarios"][0]["seed_hex"].as_str().expect("null seed").trim_start_matches("0x"),
            16
        ).expect("valid null seed");
        let null_campaign = generate_campaign_v1(
            config, null_seed, CampaignWorldV1::Null, None
        ).expect("null campaign");
        assert_eq!(
            top_rank_accuracy_permille_v1(&null_campaign.public, &null_campaign.evaluator_truth).unwrap(),
            fixture["scenarios"][0]["expected"]["top_rank_accuracy_permille"].as_u64().unwrap() as u16
        );
        let null_model = fit_red_team_v1(&null_campaign.public).unwrap();
        let null_predictions = predict_red_team_v1(&null_model, &null_campaign.public).unwrap();
        let null_report = score_red_team_v1(
            &null_campaign.public, &null_campaign.evaluator_truth, &null_predictions,
            RED_TEAM_THRESHOLD_PERMILLE, RED_TEAM_MINIMUM_PREDICTIONS
        ).unwrap();
        let null_scores: Vec<Option<u16>> = null_report.channel_scores.iter()
            .map(|row| row.accuracy_permille).collect();
        let expected_null_scores: Vec<Option<u16>> = fixture["scenarios"][0]["expected"]["channel_accuracies_permille"]
            .as_array().unwrap().iter()
            .map(|value| value.as_u64().map(|n| n as u16)).collect();
        assert_eq!(null_scores, expected_null_scores);
        assert!(null_report.detected_modalities.is_empty());

        let signal_seed = u64::from_str_radix(
            fixture["scenarios"][1]["seed_hex"].as_str().expect("signal seed").trim_start_matches("0x"),
            16
        ).expect("valid signal seed");
        let signal_campaign = generate_campaign_v1(
            config, signal_seed, CampaignWorldV1::SignalInjected, None
        ).expect("signal campaign");
        assert_eq!(
            top_rank_accuracy_permille_v1(&signal_campaign.public, &signal_campaign.evaluator_truth).unwrap(),
            fixture["scenarios"][1]["expected"]["top_rank_accuracy_permille"].as_u64().unwrap() as u16
        );
        let signal_model = fit_red_team_v1(&signal_campaign.public).unwrap();
        let signal_predictions = predict_red_team_v1(&signal_model, &signal_campaign.public).unwrap();
        let signal_report = score_red_team_v1(
            &signal_campaign.public, &signal_campaign.evaluator_truth, &signal_predictions,
            RED_TEAM_THRESHOLD_PERMILLE, RED_TEAM_MINIMUM_PREDICTIONS
        ).unwrap();
        let signal_scores: Vec<Option<u16>> = signal_report.channel_scores.iter()
            .map(|row| row.accuracy_permille).collect();
        let expected_signal_scores: Vec<Option<u16>> = fixture["scenarios"][1]["expected"]["channel_accuracies_permille"]
            .as_array().unwrap().iter()
            .map(|value| value.as_u64().map(|n| n as u16)).collect();
        assert_eq!(signal_scores, expected_signal_scores);
        assert!(signal_report.detected_modalities.is_empty());
    }


    #[test]
    fn confounded_participant_assignment_is_detected_as_procedural() {
        let campaign = generate_campaign_v1(
            config(),
            0x3333_4444_5555_6666,
            CampaignWorldV1::Confounded,
            None,
        ).expect("valid confounded campaign");
        let response_accuracy =
            top_rank_accuracy_permille_v1(&campaign.public, &campaign.evaluator_truth)
                .expect("response score");
        assert!(response_accuracy < 430, "the response itself should remain null-like");

        let model = fit_red_team_v1(&campaign.public).expect("fit red team");
        let predictions = predict_red_team_v1(&model, &campaign.public).expect("predict");
        let report = score_red_team_v1(
            &campaign.public,
            &campaign.evaluator_truth,
            &predictions,
            RED_TEAM_THRESHOLD_PERMILLE,
            RED_TEAM_MINIMUM_PREDICTIONS,
        ).expect("score");
        assert_eq!(
            report.detected_modalities,
            vec![LeakageModalityV1::Procedural]
        );
        let procedural = report.channel_scores.iter()
            .find(|row| row.modality == LeakageModalityV1::Procedural)
            .expect("procedural score");
        assert_eq!(procedural.accuracy_permille, Some(1_000));
    }


    #[test]
    fn confounded_campaign_matches_frozen_known_answer() {
        let fixture: serde_json::Value = serde_json::from_str(
            include_str!("../../../research/spectra-001/fixtures/corpus-known-answer-v0.1.json")
        ).expect("known-answer fixture JSON");
        let seed = u64::from_str_radix(
            fixture["scenarios"][2]["seed_hex"].as_str().expect("confounded seed").trim_start_matches("0x"),
            16
        ).expect("valid confounded seed");
        let campaign = generate_campaign_v1(
            config(), seed, CampaignWorldV1::Confounded, None
        ).expect("confounded campaign");
        assert_eq!(
            top_rank_accuracy_permille_v1(&campaign.public, &campaign.evaluator_truth).unwrap(),
            fixture["scenarios"][2]["expected"]["top_rank_accuracy_permille"].as_u64().unwrap() as u16
        );
        let model = fit_red_team_v1(&campaign.public).unwrap();
        let predictions = predict_red_team_v1(&model, &campaign.public).unwrap();
        let report = score_red_team_v1(
            &campaign.public, &campaign.evaluator_truth, &predictions,
            RED_TEAM_THRESHOLD_PERMILLE, RED_TEAM_MINIMUM_PREDICTIONS
        ).unwrap();
        let scores: Vec<Option<u16>> = report.channel_scores.iter()
            .map(|row| row.accuracy_permille).collect();
        let expected: Vec<Option<u16>> = fixture["scenarios"][2]["expected"]["channel_accuracies_permille"]
            .as_array().unwrap().iter()
            .map(|value| value.as_u64().map(|n| n as u16)).collect();
        assert_eq!(scores, expected);
        assert_eq!(report.detected_modalities, vec![LeakageModalityV1::Procedural]);
    }


    #[test]
    fn zero_campaign_instance_id_is_rejected() {
        let mut invalid = config();
        invalid.campaign_instance_id = 0;
        assert_eq!(
            generate_campaign_v1(invalid, 1, CampaignWorldV1::Null, None),
            Err(CorpusErrorV1::InvalidCampaignId)
        );
    }

    #[test]
    fn invalid_world_leakage_pairs_are_rejected() {
        assert_eq!(
            generate_campaign_v1(
                config(),
                1,
                CampaignWorldV1::Null,
                Some(LeakageModalityV1::Rf),
            ),
            Err(CorpusErrorV1::WorldLeakageMismatch)
        );
    }

    #[test]
    fn invalid_configurations_fail_closed() {
        let mut invalid = config();
        invalid.candidate_count = 1;
        assert_eq!(
            generate_campaign_v1(invalid, 1, CampaignWorldV1::Null, None),
            Err(CorpusErrorV1::InvalidCandidateCount)
        );
        invalid = config();
        invalid.signal_hit_rate_permille = 1_001;
        assert_eq!(
            generate_campaign_v1(invalid, 1, CampaignWorldV1::Null, None),
            Err(CorpusErrorV1::InvalidSignalRate)
        );
    }
}
