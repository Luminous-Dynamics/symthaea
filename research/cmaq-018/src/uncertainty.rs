//! Selection-aware uncertainty metrics for CMAQ-018.
//!
//! These are metric and interval-construction primitives, not a production
//! predictor and not a claim that any model is calibrated. In particular,
//! simultaneous coverage must be measured over the complete action set;
//! pointwise intervals do not automatically remain calibrated after selecting
//! the most attractive action.

use std::fmt::Write as _;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictiveInterval {
    pub lower: f64,
    pub upper: f64,
    pub nominal_coverage: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalObservation {
    pub interval: PredictiveInterval,
    pub realized: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalMetrics {
    pub sample_count: usize,
    pub nominal_coverage: f64,
    pub empirical_coverage: f64,
    pub mean_width: f64,
    pub mean_interval_score: f64,
    pub absolute_coverage_error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimultaneousMetrics {
    pub episode_count: usize,
    pub intervals_per_episode: usize,
    pub familywise_nominal_coverage: f64,
    pub empirical_simultaneous_coverage: f64,
    pub mean_max_interval_width: f64,
    pub absolute_coverage_error: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectionEpisode {
    pub intervals: Vec<PredictiveInterval>,
    pub realized_outcomes: Vec<f64>,
    pub selected_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectionMetrics {
    pub episode_count: usize,
    pub actions_per_episode: usize,
    pub mean_realized_regret: f64,
    pub mean_regret_upper_bound: f64,
    pub regret_bound_coverage: f64,
    pub simultaneous_interval_coverage: f64,
    pub dominant_certificate_count: usize,
    pub false_dominant_certificate_count: usize,
    pub false_dominant_certificate_rate: f64,
    pub mean_selected_interval_width: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostedDecisionOutcome {
    pub realized_utility: f64,
    pub inference_cost: f64,
    pub acquisition_cost: f64,
    pub failure_recovery_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionValueEpisode {
    pub world_model: CostedDecisionOutcome,
    pub direct_d0: CostedDecisionOutcome,
    pub direct_d1: CostedDecisionOutcome,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetDecisionValueMetrics {
    pub episode_count: usize,
    pub mean_gross_advantage_vs_d0: f64,
    pub mean_net_advantage_vs_d0: f64,
    pub net_win_rate_vs_d0: f64,
    pub mean_gross_advantage_vs_d1: f64,
    pub mean_net_advantage_vs_d1: f64,
    pub net_win_rate_vs_d1: f64,
    pub mean_world_model_net_utility: f64,
    pub mean_d0_net_utility: f64,
    pub mean_d1_net_utility: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbstainReason {
    NoIntervalDominantAction,
    AmbiguousDominantActions,
    NoConstraintFeasibleActions,
    NoPositiveInformationValue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RobustActionDecision {
    Certified { action_index: usize },
    Abstain { reason: AbstainReason },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointDrawCoupling {
    SharedPosteriorAndWorldNoise,
    PairedExogenousDisturbance,
    DeterministicSharedLatent,
}

/// All action outcomes predicted from one exact observed history and one
/// shared joint-draw bank. The sample matrix is [action][joint_draw]; column
/// j must denote the same posterior/latent/disturbance draw for every action.
#[derive(Clone, Debug, PartialEq)]
pub struct JointActionSetSamples {
    pub history_sha256: String,
    pub horizon_steps: usize,
    pub action_manifest_sha256: String,
    pub outcome_channel: String,
    pub action_ids: Vec<u64>,
    pub joint_draw_ids: Vec<u64>,
    pub joint_draw_bank_sha256: String,
    pub coupling: JointDrawCoupling,
    pub outcome_samples: Vec<Vec<f64>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisualBeliefChannelKind {
    Geometry,
    Motion,
    Contact,
    Visibility,
    TaskUtility,
    TerminalRisk,
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisualBeliefChannelSamples {
    /// Stable semantic key, e.g. object_42.position.x or gripper.contact_probability.
    pub channel_id: String,
    pub kind: VisualBeliefChannelKind,
    pub unit: String,
    /// Matrix shape is [action][joint_draw], sharing the parent's draw axis.
    pub outcome_samples: Vec<Vec<f64>>,
}

/// Structured counterpart to JointActionSetSamples. Channels remain separate:
/// position, velocity, contact, visibility, risk and utility are never collapsed
/// into an arbitrary scalar. All channels share one history, action manifest
/// and joint-draw axis.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuredJointActionSetSamples {
    pub history_sha256: String,
    pub horizon_steps: usize,
    pub action_manifest_sha256: String,
    pub action_ids: Vec<u64>,
    pub joint_draw_ids: Vec<u64>,
    pub joint_draw_bank_sha256: String,
    pub coupling: JointDrawCoupling,
    pub channels: Vec<VisualBeliefChannelSamples>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisualChannelActionPredictionInterval {
    pub channel_id: String,
    pub kind: VisualBeliefChannelKind,
    pub unit: String,
    pub action_id: u64,
    pub draw_count: usize,
    pub interval: PredictiveInterval,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisualChannelActionEffectInterval {
    pub channel_id: String,
    pub kind: VisualBeliefChannelKind,
    pub unit: String,
    pub left_action_id: u64,
    pub right_action_id: u64,
    pub paired_draw_count: usize,
    pub interval: PredictiveInterval,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionPredictionInterval {
    pub action_id: u64,
    pub interval: PredictiveInterval,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionEffectInterval {
    pub left_action_id: u64,
    pub right_action_id: u64,
    pub paired_draw_count: usize,
    pub interval: PredictiveInterval,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntervalError {
    InvalidNominalCoverage,
    NonFiniteLowerBound,
    NonFiniteUpperBound,
    ReversedBounds,
    NonFiniteIntervalWidth,
    NonFiniteRealization,
    NonFiniteDerivedMetric,
    MixedNominalCoverage,
    EmptySamples,
    InsufficientSamples,
    UnequalPairedSampleCount,
    EmptyEpisodeSet,
    EmptyActionSet,
    UnequalActionSetCardinality,
    InsufficientActions,
    InvalidSelectedIndex,
    InvalidHorizon,
    InvalidHistoryDigest,
    InvalidActionManifestDigest,
    EmptyOutcomeChannel,
    InvalidJointDrawDigest,
    JointDrawBankDigestMismatch,
    InsufficientJointDraws,
    DuplicateActionId,
    DuplicateJointDrawId,
    ActionSampleCardinalityMismatch,
    SameActionRequested,
    NegativeDecisionCost,
    EmptyVisualChannels,
    DuplicateVisualChannelId,
    InvalidVisualChannelDescriptor,
    StructuredChannelSampleCardinalityMismatch,
    NonUtilitySelectionChannel,
    InvalidInformationOption,
    InvalidOutcomeProbabilities,
    NegativeExpectedInformationValue,
}

/// Construct a central empirical interval from predictive samples.
///
/// Quantiles use linear interpolation (the common type-7 sample quantile).
/// This construction alone is not a finite-sample calibration guarantee;
/// calibration must be assessed on held-out data or added by a separately
/// specified procedure such as split conformal prediction.
pub fn central_interval(
    samples: &[f64],
    nominal_coverage: f64,
) -> Result<PredictiveInterval, IntervalError> {
    validate_nominal_coverage(nominal_coverage)?;
    if samples.is_empty() {
        return Err(IntervalError::EmptySamples);
    }
    if samples.len() < 2 {
        return Err(IntervalError::InsufficientSamples);
    }

    let mut sorted = samples.to_vec();
    if sorted.iter().any(|sample| !sample.is_finite()) {
        return Err(IntervalError::NonFiniteDerivedMetric);
    }
    sorted.sort_by(f64::total_cmp);

    let tail_probability = (1.0 - nominal_coverage) / 2.0;
    let lower = linear_quantile(&sorted, tail_probability);
    let upper = linear_quantile(&sorted, 1.0 - tail_probability);
    let interval = PredictiveInterval {
        lower,
        upper,
        nominal_coverage,
    };
    validate_interval(interval)?;
    Ok(interval)
}

/// Construct an action-effect interval from paired joint predictive samples.
///
/// left_samples[i] and right_samples[i] must belong to the same observed
/// history and the same joint posterior/latent draw. Pairing preserves
/// common-mode error cancellation. Subtracting independently constructed
/// marginal interval endpoints is not an equivalent operation.
pub fn paired_effect_interval(
    left_samples: &[f64],
    right_samples: &[f64],
    nominal_coverage: f64,
) -> Result<PredictiveInterval, IntervalError> {
    if left_samples.len() != right_samples.len() {
        return Err(IntervalError::UnequalPairedSampleCount);
    }
    if left_samples.is_empty() {
        return Err(IntervalError::EmptySamples);
    }
    if left_samples.len() < 2 {
        return Err(IntervalError::InsufficientSamples);
    }

    let mut paired_differences = Vec::with_capacity(left_samples.len());
    for (left, right) in left_samples.iter().zip(right_samples) {
        if !left.is_finite() || !right.is_finite() {
            return Err(IntervalError::NonFiniteDerivedMetric);
        }
        let difference = left - right;
        if !difference.is_finite() {
            return Err(IntervalError::NonFiniteDerivedMetric);
        }
        paired_differences.push(difference);
    }
    central_interval(&paired_differences, nominal_coverage)
}

/// Validate the shape and provenance bindings of one same-history action bank.
///
/// This checks that an adapter supplies a shared draw axis. It cannot prove
/// that the producer honestly generated the values using the declared
/// coupling; the source/recorder and draw-bank bytes must be independently
/// retained and verified by the adapter's evidence layer.
pub fn validate_joint_action_set(bank: &JointActionSetSamples) -> Result<(), IntervalError> {
    validate_joint_action_set_shape(bank)?;
    let expected_digest = compute_joint_draw_bank_sha256(bank)?;
    if expected_digest != bank.joint_draw_bank_sha256 {
        return Err(IntervalError::JointDrawBankDigestMismatch);
    }
    Ok(())
}

fn validate_joint_action_set_shape(bank: &JointActionSetSamples) -> Result<(), IntervalError> {
    if bank.horizon_steps == 0 {
        return Err(IntervalError::InvalidHorizon);
    }
    if !is_lower_hex_sha256(&bank.history_sha256) {
        return Err(IntervalError::InvalidHistoryDigest);
    }
    if !is_lower_hex_sha256(&bank.action_manifest_sha256) {
        return Err(IntervalError::InvalidActionManifestDigest);
    }
    if bank.outcome_channel.trim().is_empty() {
        return Err(IntervalError::EmptyOutcomeChannel);
    }
    if !is_lower_hex_sha256(&bank.joint_draw_bank_sha256) {
        return Err(IntervalError::InvalidJointDrawDigest);
    }
    if bank.action_ids.len() < 2 {
        return Err(IntervalError::InsufficientActions);
    }
    if bank.outcome_samples.len() != bank.action_ids.len() {
        return Err(IntervalError::UnequalActionSetCardinality);
    }
    if bank.joint_draw_ids.len() < 2 {
        return Err(IntervalError::InsufficientJointDraws);
    }

    let mut action_ids = bank.action_ids.clone();
    action_ids.sort_unstable();
    if action_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(IntervalError::DuplicateActionId);
    }

    let mut draw_ids = bank.joint_draw_ids.clone();
    draw_ids.sort_unstable();
    if draw_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(IntervalError::DuplicateJointDrawId);
    }

    for samples in &bank.outcome_samples {
        if samples.len() != bank.joint_draw_ids.len() {
            return Err(IntervalError::ActionSampleCardinalityMismatch);
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(IntervalError::NonFiniteDerivedMetric);
        }
    }
    Ok(())
}

fn append_len_prefixed(buffer: &mut Vec<u8>, bytes: &[u8]) {
    buffer.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    buffer.extend_from_slice(bytes);
}

fn append_u64_values(buffer: &mut Vec<u8>, values: &[u64]) {
    buffer.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
}

/// Canonical SHA-256 of the action/draw/sample content. The digest excludes
/// the digest field itself and includes action order, so a row permutation
/// must also permute the sample rows and recompute the bank digest.
pub fn compute_joint_draw_bank_sha256(
    bank: &JointActionSetSamples,
) -> Result<String, IntervalError> {
    validate_joint_action_set_shape(bank)?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"cmaq-018-joint-action-bank-v0.1\0");
    append_len_prefixed(&mut bytes, bank.history_sha256.as_bytes());
    bytes.extend_from_slice(&(bank.horizon_steps as u64).to_le_bytes());
    append_len_prefixed(&mut bytes, bank.action_manifest_sha256.as_bytes());
    append_len_prefixed(&mut bytes, bank.outcome_channel.as_bytes());
    append_u64_values(&mut bytes, &bank.action_ids);
    append_u64_values(&mut bytes, &bank.joint_draw_ids);
    bytes.push(match bank.coupling {
        JointDrawCoupling::SharedPosteriorAndWorldNoise => 0,
        JointDrawCoupling::PairedExogenousDisturbance => 1,
        JointDrawCoupling::DeterministicSharedLatent => 2,
    });
    bytes.extend_from_slice(&(bank.outcome_samples.len() as u64).to_le_bytes());
    for samples in &bank.outcome_samples {
        bytes.extend_from_slice(&(samples.len() as u64).to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_bits().to_le_bytes());
        }
    }
    Ok(sha256_hex(&bytes))
}

fn coupling_tag(coupling: JointDrawCoupling) -> u8 {
    match coupling {
        JointDrawCoupling::SharedPosteriorAndWorldNoise => 0,
        JointDrawCoupling::PairedExogenousDisturbance => 1,
        JointDrawCoupling::DeterministicSharedLatent => 2,
    }
}

fn visual_channel_kind_tag(kind: VisualBeliefChannelKind) -> u8 {
    match kind {
        VisualBeliefChannelKind::Geometry => 0,
        VisualBeliefChannelKind::Motion => 1,
        VisualBeliefChannelKind::Contact => 2,
        VisualBeliefChannelKind::Visibility => 3,
        VisualBeliefChannelKind::TaskUtility => 4,
        VisualBeliefChannelKind::TerminalRisk => 5,
        VisualBeliefChannelKind::Other => 255,
    }
}

fn validate_structured_joint_action_set_shape(
    bank: &StructuredJointActionSetSamples,
) -> Result<(), IntervalError> {
    if bank.horizon_steps == 0 {
        return Err(IntervalError::InvalidHorizon);
    }
    if !is_lower_hex_sha256(&bank.history_sha256) {
        return Err(IntervalError::InvalidHistoryDigest);
    }
    if !is_lower_hex_sha256(&bank.action_manifest_sha256) {
        return Err(IntervalError::InvalidActionManifestDigest);
    }
    if !is_lower_hex_sha256(&bank.joint_draw_bank_sha256) {
        return Err(IntervalError::InvalidJointDrawDigest);
    }
    if bank.action_ids.len() < 2 {
        return Err(IntervalError::InsufficientActions);
    }
    if bank.joint_draw_ids.len() < 2 {
        return Err(IntervalError::InsufficientJointDraws);
    }
    if bank.channels.is_empty() {
        return Err(IntervalError::EmptyVisualChannels);
    }

    let mut action_ids = bank.action_ids.clone();
    action_ids.sort_unstable();
    if action_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(IntervalError::DuplicateActionId);
    }
    let mut draw_ids = bank.joint_draw_ids.clone();
    draw_ids.sort_unstable();
    if draw_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(IntervalError::DuplicateJointDrawId);
    }

    let mut channel_ids: Vec<&str> = Vec::with_capacity(bank.channels.len());
    for channel in &bank.channels {
        if channel.channel_id.trim().is_empty()
            || channel.channel_id.trim() != channel.channel_id
            || channel.unit.trim().is_empty()
            || channel.unit.trim() != channel.unit
        {
            return Err(IntervalError::InvalidVisualChannelDescriptor);
        }
        channel_ids.push(channel.channel_id.as_str());
        if channel.outcome_samples.len() != bank.action_ids.len() {
            return Err(IntervalError::StructuredChannelSampleCardinalityMismatch);
        }
        for samples in &channel.outcome_samples {
            if samples.len() != bank.joint_draw_ids.len() {
                return Err(IntervalError::StructuredChannelSampleCardinalityMismatch);
            }
            if samples.iter().any(|sample| !sample.is_finite()) {
                return Err(IntervalError::NonFiniteDerivedMetric);
            }
        }
    }
    channel_ids.sort_unstable();
    if channel_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(IntervalError::DuplicateVisualChannelId);
    }
    Ok(())
}

/// Compute canonical content SHA-256 over all channels and shared action/draw
/// identities. The digest binds descriptor semantics, units and exact f64 bits.
pub fn compute_structured_joint_draw_bank_sha256(
    bank: &StructuredJointActionSetSamples,
) -> Result<String, IntervalError> {
    validate_structured_joint_action_set_shape(bank)?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"cmaq-018-structured-visual-action-bank-v0.1\0");
    append_len_prefixed(&mut bytes, bank.history_sha256.as_bytes());
    bytes.extend_from_slice(&(bank.horizon_steps as u64).to_le_bytes());
    append_len_prefixed(&mut bytes, bank.action_manifest_sha256.as_bytes());
    append_u64_values(&mut bytes, &bank.action_ids);
    append_u64_values(&mut bytes, &bank.joint_draw_ids);
    bytes.push(coupling_tag(bank.coupling));
    bytes.extend_from_slice(&(bank.channels.len() as u64).to_le_bytes());
    for channel in &bank.channels {
        append_len_prefixed(&mut bytes, channel.channel_id.as_bytes());
        bytes.push(visual_channel_kind_tag(channel.kind));
        append_len_prefixed(&mut bytes, channel.unit.as_bytes());
        bytes.extend_from_slice(&(channel.outcome_samples.len() as u64).to_le_bytes());
        for row in &channel.outcome_samples {
            bytes.extend_from_slice(&(row.len() as u64).to_le_bytes());
            for sample in row {
                bytes.extend_from_slice(&sample.to_bits().to_le_bytes());
            }
        }
    }
    Ok(sha256_hex(&bytes))
}

/// Validate a structured visual belief action bank and recompute its digest.
/// This verifies content integrity, not whether the producer actually used the
/// declared shared posterior/disturbance coupling.
pub fn validate_structured_joint_action_set(
    bank: &StructuredJointActionSetSamples,
) -> Result<(), IntervalError> {
    validate_structured_joint_action_set_shape(bank)?;
    if compute_structured_joint_draw_bank_sha256(bank)? != bank.joint_draw_bank_sha256 {
        return Err(IntervalError::JointDrawBankDigestMismatch);
    }
    Ok(())
}

/// Return per-action intervals for every structured channel. Results are sorted
/// by channel ID, then action ID, so row order does not affect lookup semantics.
pub fn structured_action_prediction_intervals(
    bank: &StructuredJointActionSetSamples,
    nominal_coverage: f64,
) -> Result<Vec<VisualChannelActionPredictionInterval>, IntervalError> {
    validate_structured_joint_action_set(bank)?;
    validate_nominal_coverage(nominal_coverage)?;
    let mut results = Vec::with_capacity(bank.channels.len() * bank.action_ids.len());
    let mut channels: Vec<&VisualBeliefChannelSamples> = bank.channels.iter().collect();
    channels.sort_by(|left, right| left.channel_id.cmp(&right.channel_id));
    let mut action_indices: Vec<(u64, usize)> = bank
        .action_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, action_id)| (action_id, index))
        .collect();
    action_indices.sort_unstable_by_key(|(action_id, _)| *action_id);
    for channel in channels {
        for (action_id, row_index) in &action_indices {
            results.push(VisualChannelActionPredictionInterval {
                channel_id: channel.channel_id.clone(),
                kind: channel.kind,
                unit: channel.unit.clone(),
                action_id: *action_id,
                draw_count: bank.joint_draw_ids.len(),
                interval: central_interval(&channel.outcome_samples[*row_index], nominal_coverage)?,
            });
        }
    }
    Ok(results)
}

fn structured_action_effect_interval_validated(
    bank: &StructuredJointActionSetSamples,
    channel_id: &str,
    left_action_id: u64,
    right_action_id: u64,
    nominal_coverage: f64,
) -> Result<VisualChannelActionEffectInterval, IntervalError> {
    if left_action_id == right_action_id {
        return Err(IntervalError::SameActionRequested);
    }
    let channel = bank
        .channels
        .iter()
        .find(|channel| channel.channel_id == channel_id)
        .ok_or(IntervalError::InvalidVisualChannelDescriptor)?;
    let left_index = bank
        .action_ids
        .iter()
        .position(|id| *id == left_action_id)
        .ok_or(IntervalError::InvalidSelectedIndex)?;
    let right_index = bank
        .action_ids
        .iter()
        .position(|id| *id == right_action_id)
        .ok_or(IntervalError::InvalidSelectedIndex)?;
    let interval = paired_effect_interval(
        &channel.outcome_samples[left_index],
        &channel.outcome_samples[right_index],
        nominal_coverage,
    )?;
    Ok(VisualChannelActionEffectInterval {
        channel_id: channel.channel_id.clone(),
        kind: channel.kind,
        unit: channel.unit.clone(),
        left_action_id,
        right_action_id,
        paired_draw_count: bank.joint_draw_ids.len(),
        interval,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuredActionDecision {
    Certified { action_id: u64 },
    Abstain { reason: AbstainReason },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintRelation {
    UpperAtMost,
    LowerAtLeast,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructuredActionConstraint {
    pub channel_id: String,
    pub kind: VisualBeliefChannelKind,
    pub relation: ConstraintRelation,
    pub threshold: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstrainedSelectionBasis {
    FeasibleOnly,
    UtilityDominance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstrainedActionDecision {
    Selected {
        action_id: u64,
        basis: ConstrainedSelectionBasis,
    },
    Abstain {
        reason: AbstainReason,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct InformationAcquisitionOption {
    pub history_sha256: String,
    pub action_manifest_sha256: String,
    pub query_manifest_sha256: String,
    pub query_id: String,
    pub action_ids: Vec<u64>,
    pub outcome_ids: Vec<String>,
    /// Probability of each information outcome; must sum to one.
    pub outcome_probabilities: Vec<f64>,
    /// Conditional action utilities indexed [information outcome][action row].
    pub conditional_action_utilities: Vec<Vec<f64>>,
    /// All-in query cost expressed in the same preregistered utility units.
    pub acquisition_cost: f64,
    /// Recomputed digest binding the exact scenario, utility and cost bytes.
    pub evidence_sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InformationOptionEvaluation {
    pub history_sha256: String,
    pub action_manifest_sha256: String,
    pub query_manifest_sha256: String,
    pub evidence_sha256: String,
    pub query_id: String,
    pub action_ids: Vec<u64>,
    pub outcome_ids: Vec<String>,
    pub prior_best_action_id: u64,
    pub best_action_by_outcome: Vec<u64>,
    pub expected_utility_without_information: f64,
    pub expected_utility_after_information: f64,
    pub expected_value_of_sample_information: f64,
    pub acquisition_cost: f64,
    pub net_information_value: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InformationAcquisitionChoice {
    AcquireBestQuery,
    Abstain {
        reason: AbstainReason,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct InformationAcquisitionEvaluation {
    /// Evaluations sorted by query_id, independent of caller order.
    pub options: Vec<InformationOptionEvaluation>,
    pub choice: InformationAcquisitionChoice,
    pub selected_query_id: Option<String>,
}


/// Select using only one explicitly named TaskUtility channel. Geometry,
/// contact and risk channels remain constraints/diagnostics; this function
/// never silently scalarizes or combines them with utility.
pub fn select_structured_task_action(
    bank: &StructuredJointActionSetSamples,
    utility_channel_id: &str,
    nominal_coverage: f64,
) -> Result<StructuredActionDecision, IntervalError> {
    validate_structured_joint_action_set(bank)?;
    validate_nominal_coverage(nominal_coverage)?;
    let channel = bank
        .channels
        .iter()
        .find(|channel| channel.channel_id == utility_channel_id)
        .ok_or(IntervalError::InvalidVisualChannelDescriptor)?;
    if channel.kind != VisualBeliefChannelKind::TaskUtility {
        return Err(IntervalError::NonUtilitySelectionChannel);
    }
    let mut actions: Vec<(u64, usize)> = bank
        .action_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, action_id)| (action_id, index))
        .collect();
    actions.sort_unstable_by_key(|(action_id, _)| *action_id);
    let intervals: Vec<PredictiveInterval> = actions
        .iter()
        .map(|(_, index)| central_interval(&channel.outcome_samples[*index], nominal_coverage))
        .collect::<Result<_, _>>()?;
    Ok(match select_interval_dominant_action(&intervals)? {
        RobustActionDecision::Certified { action_index } => StructuredActionDecision::Certified {
            action_id: actions[action_index].0,
        },
        RobustActionDecision::Abstain { reason } => StructuredActionDecision::Abstain { reason },
    })
}

/// Apply explicit non-utility constraints before selecting on a named utility channel.
///
/// Constraint semantics are interval-based: UpperAtMost requires the complete
/// predictive upper bound to be at or below the threshold; LowerAtLeast
/// requires the complete lower bound to be at or above it. A sole feasible
/// action is reported as FeasibleOnly, not as utility-dominant. With multiple
/// feasible actions, a unique utility-dominant action is required; otherwise
/// the selector abstains. This is conditional on simultaneous channel coverage
/// and is not physical actuator authority.
pub fn select_constrained_structured_action(
    bank: &StructuredJointActionSetSamples,
    utility_channel_id: &str,
    constraints: &[StructuredActionConstraint],
    nominal_coverage: f64,
) -> Result<ConstrainedActionDecision, IntervalError> {
    validate_structured_joint_action_set(bank)?;
    validate_nominal_coverage(nominal_coverage)?;
    let utility = bank
        .channels
        .iter()
        .find(|channel| channel.channel_id == utility_channel_id)
        .ok_or(IntervalError::InvalidVisualChannelDescriptor)?;
    if utility.kind != VisualBeliefChannelKind::TaskUtility {
        return Err(IntervalError::NonUtilitySelectionChannel);
    }

    let mut seen_constraints = Vec::with_capacity(constraints.len());
    for constraint in constraints {
        if constraint.channel_id.trim().is_empty()
            || constraint.channel_id.trim() != constraint.channel_id
            || !constraint.threshold.is_finite()
            || constraint.channel_id == utility_channel_id
            || seen_constraints.contains(&constraint.channel_id.as_str())
        {
            return Err(IntervalError::InvalidVisualChannelDescriptor);
        }
        let channel = bank
            .channels
            .iter()
            .find(|channel| channel.channel_id == constraint.channel_id)
            .ok_or(IntervalError::InvalidVisualChannelDescriptor)?;
        if channel.kind != constraint.kind || channel.kind == VisualBeliefChannelKind::TaskUtility {
            return Err(IntervalError::InvalidVisualChannelDescriptor);
        }
        seen_constraints.push(constraint.channel_id.as_str());
    }

    let mut action_rows: Vec<(u64, usize)> = bank
        .action_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect();
    action_rows.sort_unstable_by_key(|(id, _)| *id);

    let mut feasible = Vec::new();
    for (action_id, row) in action_rows {
        let mut action_is_feasible = true;
        for constraint in constraints {
            let channel = bank.channels.iter()
                .find(|channel| channel.channel_id == constraint.channel_id)
                .expect("constraints were resolved above");
            let interval = central_interval(&channel.outcome_samples[row], nominal_coverage)?;
            let passes = match constraint.relation {
                ConstraintRelation::UpperAtMost => interval.upper <= constraint.threshold,
                ConstraintRelation::LowerAtLeast => interval.lower >= constraint.threshold,
            };
            if !passes {
                action_is_feasible = false;
                break;
            }
        }
        if action_is_feasible {
            feasible.push((action_id, row));
        }
    }

    if feasible.is_empty() {
        return Ok(ConstrainedActionDecision::Abstain {
            reason: AbstainReason::NoConstraintFeasibleActions,
        });
    }
    if feasible.len() == 1 {
        return Ok(ConstrainedActionDecision::Selected {
            action_id: feasible[0].0,
            basis: ConstrainedSelectionBasis::FeasibleOnly,
        });
    }

    let intervals = feasible.iter()
        .map(|(_, row)| central_interval(&utility.outcome_samples[*row], nominal_coverage))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(match select_interval_dominant_action(&intervals)? {
        RobustActionDecision::Certified { action_index } => ConstrainedActionDecision::Selected {
            action_id: feasible[action_index].0,
            basis: ConstrainedSelectionBasis::UtilityDominance,
        },
        RobustActionDecision::Abstain { reason } => ConstrainedActionDecision::Abstain { reason },
    })
}

fn best_action_index(action_ids: &[u64], utilities: &[f64]) -> usize {
    let mut best = 0usize;
    for index in 1..utilities.len() {
        if utilities[index] > utilities[best]
            || (utilities[index].to_bits() == utilities[best].to_bits()
                && action_ids[index] < action_ids[best])
        {
            best = index;
        }
    }
    best
}

fn validate_information_option_shape(
    option: &InformationAcquisitionOption,
) -> Result<(), IntervalError> {
    if !is_lower_hex_sha256(&option.history_sha256)
        || !is_lower_hex_sha256(&option.action_manifest_sha256)
        || !is_lower_hex_sha256(&option.query_manifest_sha256)
        || !is_lower_hex_sha256(&option.evidence_sha256)
        || option.query_id.trim().is_empty()
        || option.query_id.trim() != option.query_id
        || !option.acquisition_cost.is_finite()
        || option.acquisition_cost < 0.0
        || option.action_ids.len() < 2
        || option.outcome_ids.is_empty()
        || option.outcome_ids.len() != option.outcome_probabilities.len()
        || option.conditional_action_utilities.len() != option.outcome_probabilities.len()
    {
        return Err(IntervalError::InvalidInformationOption);
    }
    let mut action_ids = option.action_ids.clone();
    action_ids.sort_unstable();
    if action_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(IntervalError::DuplicateActionId);
    }
    let mut outcome_ids: Vec<&str> = Vec::with_capacity(option.outcome_ids.len());
    for outcome_id in &option.outcome_ids {
        if outcome_id.trim().is_empty() || outcome_id.trim() != outcome_id {
            return Err(IntervalError::InvalidInformationOption);
        }
        outcome_ids.push(outcome_id);
    }
    outcome_ids.sort_unstable();
    if outcome_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(IntervalError::InvalidInformationOption);
    }
    let mut probability_sum = 0.0;
    for probability in &option.outcome_probabilities {
        if !probability.is_finite() || *probability < 0.0 || *probability > 1.0 {
            return Err(IntervalError::InvalidOutcomeProbabilities);
        }
        probability_sum += probability;
    }
    if !probability_sum.is_finite() || (probability_sum - 1.0).abs() > 1e-9 {
        return Err(IntervalError::InvalidOutcomeProbabilities);
    }
    for utilities in &option.conditional_action_utilities {
        if utilities.len() != option.action_ids.len()
            || utilities.iter().any(|utility| !utility.is_finite())
        {
            return Err(IntervalError::InvalidInformationOption);
        }
    }
    Ok(())
}

/// Canonical SHA-256 over history/action/query identities, outcome partition,
/// probabilities, conditional utilities and acquisition cost.
pub fn compute_information_option_sha256(
    option: &InformationAcquisitionOption,
) -> Result<String, IntervalError> {
    validate_information_option_shape(option)?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"cmaq-018-evsi-option-v0.1\0");
    append_len_prefixed(&mut bytes, option.history_sha256.as_bytes());
    append_len_prefixed(&mut bytes, option.action_manifest_sha256.as_bytes());
    append_len_prefixed(&mut bytes, option.query_manifest_sha256.as_bytes());
    append_len_prefixed(&mut bytes, option.query_id.as_bytes());
    append_u64_values(&mut bytes, &option.action_ids);
    bytes.extend_from_slice(&(option.outcome_ids.len() as u64).to_le_bytes());
    for outcome_id in &option.outcome_ids {
        append_len_prefixed(&mut bytes, outcome_id.as_bytes());
    }
    bytes.extend_from_slice(&option.acquisition_cost.to_bits().to_le_bytes());
    bytes.extend_from_slice(&(option.outcome_probabilities.len() as u64).to_le_bytes());
    for probability in &option.outcome_probabilities {
        bytes.extend_from_slice(&probability.to_bits().to_le_bytes());
    }
    bytes.extend_from_slice(&(option.conditional_action_utilities.len() as u64).to_le_bytes());
    for utilities in &option.conditional_action_utilities {
        bytes.extend_from_slice(&(utilities.len() as u64).to_le_bytes());
        for utility in utilities {
            bytes.extend_from_slice(&utility.to_bits().to_le_bytes());
        }
    }
    Ok(sha256_hex(&bytes))
}

fn evaluate_information_option(
    option: &InformationAcquisitionOption,
) -> Result<InformationOptionEvaluation, IntervalError> {
    validate_information_option_shape(option)?;
    if compute_information_option_sha256(option)? != option.evidence_sha256 {
        return Err(IntervalError::JointDrawBankDigestMismatch);
    }

    let mut action_ids = option.action_ids.clone();
    action_ids.sort_unstable();
    if action_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(IntervalError::DuplicateActionId);
    }

    let mut probability_sum = 0.0;
    for probability in &option.outcome_probabilities {
        if !probability.is_finite() || *probability < 0.0 || *probability > 1.0 {
            return Err(IntervalError::InvalidOutcomeProbabilities);
        }
        probability_sum += probability;
    }
    if !probability_sum.is_finite() || (probability_sum - 1.0).abs() > 1e-9 {
        return Err(IntervalError::InvalidOutcomeProbabilities);
    }

    let mut prior_action_values = vec![0.0; option.action_ids.len()];
    for (outcome_index, utilities) in option.conditional_action_utilities.iter().enumerate() {
        if utilities.len() != option.action_ids.len()
            || utilities.iter().any(|utility| !utility.is_finite())
        {
            return Err(IntervalError::InvalidInformationOption);
        }
        for (action_index, utility) in utilities.iter().enumerate() {
            prior_action_values[action_index] +=
                option.outcome_probabilities[outcome_index] * utility;
        }
    }
    if prior_action_values.iter().any(|value| !value.is_finite()) {
        return Err(IntervalError::NonFiniteDerivedMetric);
    }

    let prior_best_index = best_action_index(&option.action_ids, &prior_action_values);
    let expected_utility_without_information = prior_action_values[prior_best_index];
    let mut expected_utility_after_information = 0.0;
    let mut best_action_by_outcome = Vec::with_capacity(option.outcome_probabilities.len());
    for (outcome_index, utilities) in option.conditional_action_utilities.iter().enumerate() {
        let best_index = best_action_index(&option.action_ids, utilities);
        best_action_by_outcome.push(option.action_ids[best_index]);
        expected_utility_after_information +=
            option.outcome_probabilities[outcome_index] * utilities[best_index];
    }
    if !expected_utility_after_information.is_finite() {
        return Err(IntervalError::NonFiniteDerivedMetric);
    }
    let raw_value = expected_utility_after_information - expected_utility_without_information;
    if raw_value < -1e-9 {
        return Err(IntervalError::NegativeExpectedInformationValue);
    }
    let expected_value_of_sample_information = raw_value.max(0.0);
    let net_information_value = expected_value_of_sample_information - option.acquisition_cost;
    if !net_information_value.is_finite() {
        return Err(IntervalError::NonFiniteDerivedMetric);
    }
    Ok(InformationOptionEvaluation {
        history_sha256: option.history_sha256.clone(),
        action_manifest_sha256: option.action_manifest_sha256.clone(),
        query_manifest_sha256: option.query_manifest_sha256.clone(),
        evidence_sha256: option.evidence_sha256.clone(),
        query_id: option.query_id.clone(),
        action_ids: option.action_ids.clone(),
        outcome_ids: option.outcome_ids.clone(),
        prior_best_action_id: option.action_ids[prior_best_index],
        best_action_by_outcome,
        expected_utility_without_information,
        expected_utility_after_information,
        expected_value_of_sample_information,
        acquisition_cost: option.acquisition_cost,
        net_information_value,
    })
}

/// Evaluate candidate evidence-acquisition queries using expected value of
/// sample information (EVSI), not entropy reduction alone.
///
/// Conditional utilities/probabilities are supplied by the decision model and
/// must be calibrated for the target population. Costs must be all-in and
/// converted to the same preregistered utility scale. Acquires only the
/// highest-net-value query when its expected value strictly exceeds cost.
pub fn evaluate_information_acquisition(
    options: &[InformationAcquisitionOption],
) -> Result<InformationAcquisitionEvaluation, IntervalError> {
    let mut evaluations = Vec::with_capacity(options.len());
    let mut seen_ids = Vec::with_capacity(options.len());
    for option in options {
        if seen_ids.contains(&option.query_id.as_str()) {
            return Err(IntervalError::InvalidInformationOption);
        }
        seen_ids.push(option.query_id.as_str());
        evaluations.push(evaluate_information_option(option)?);
    }
    evaluations.sort_by(|left, right| left.query_id.cmp(&right.query_id));
    let best = evaluations.iter()
        .filter(|evaluation| evaluation.net_information_value > 0.0)
        .max_by(|left, right| {
            left.net_information_value.total_cmp(&right.net_information_value)
                .then_with(|| right.query_id.cmp(&left.query_id))
        });
    let selected_query_id = best.map(|evaluation| evaluation.query_id.clone());
    let choice = if selected_query_id.is_some() {
        InformationAcquisitionChoice::AcquireBestQuery
    } else {
        InformationAcquisitionChoice::Abstain {
            reason: AbstainReason::NoPositiveInformationValue,
        }
    };
    Ok(InformationAcquisitionEvaluation {
        options: evaluations,
        choice,
        selected_query_id,
    })
}

/// Construct a channel-specific paired effect interval from same-column draws.
pub fn structured_action_effect_interval(
    bank: &StructuredJointActionSetSamples,
    channel_id: &str,
    left_action_id: u64,
    right_action_id: u64,
    nominal_coverage: f64,
) -> Result<VisualChannelActionEffectInterval, IntervalError> {
    validate_structured_joint_action_set(bank)?;
    validate_nominal_coverage(nominal_coverage)?;
    structured_action_effect_interval_validated(
        bank,
        channel_id,
        left_action_id,
        right_action_id,
        nominal_coverage,
    )
}

/// Return all channel/action-pair effects in stable channel-ID and action-ID order.
pub fn all_structured_action_effect_intervals(
    bank: &StructuredJointActionSetSamples,
    nominal_coverage: f64,
) -> Result<Vec<VisualChannelActionEffectInterval>, IntervalError> {
    validate_structured_joint_action_set(bank)?;
    validate_nominal_coverage(nominal_coverage)?;
    let mut channels: Vec<&VisualBeliefChannelSamples> = bank.channels.iter().collect();
    channels.sort_by(|left, right| left.channel_id.cmp(&right.channel_id));
    let mut ids = bank.action_ids.clone();
    ids.sort_unstable();
    let pair_count = ids.len().saturating_mul(ids.len().saturating_sub(1)) / 2;
    let mut effects = Vec::with_capacity(channels.len().saturating_mul(pair_count));
    for channel in channels {
        for left in 0..ids.len() {
            for right in (left + 1)..ids.len() {
                effects.push(structured_action_effect_interval_validated(
                    bank,
                    &channel.channel_id,
                    ids[left],
                    ids[right],
                    nominal_coverage,
                )?);
            }
        }
    }
    Ok(effects)
}

fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut message = input.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let start = i * 4;
            w[i] = u32::from_be_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut output = String::with_capacity(64);
    for word in h {
        write!(&mut output, "{word:08x}").unwrap();
    }
    output
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Produce a predictive interval for each action in the original action-id
/// order. Every interval is derived from the same declared joint draw bank.
pub fn joint_action_prediction_intervals(
    bank: &JointActionSetSamples,
    nominal_coverage: f64,
) -> Result<Vec<ActionPredictionInterval>, IntervalError> {
    validate_joint_action_set(bank)?;
    validate_nominal_coverage(nominal_coverage)?;
    bank.action_ids
        .iter()
        .copied()
        .zip(&bank.outcome_samples)
        .map(|(action_id, samples)| {
            Ok(ActionPredictionInterval {
                action_id,
                interval: central_interval(samples, nominal_coverage)?,
            })
        })
        .collect()
}

/// Construct the paired effect distribution for two action identities from
/// the shared draw axis. Action order is semantic by ID, not row position.
pub fn joint_action_effect_interval(
    bank: &JointActionSetSamples,
    left_action_id: u64,
    right_action_id: u64,
    nominal_coverage: f64,
) -> Result<ActionEffectInterval, IntervalError> {
    validate_joint_action_set(bank)?;
    joint_action_effect_interval_validated(bank, left_action_id, right_action_id, nominal_coverage)
}

fn joint_action_effect_interval_validated(
    bank: &JointActionSetSamples,
    left_action_id: u64,
    right_action_id: u64,
    nominal_coverage: f64,
) -> Result<ActionEffectInterval, IntervalError> {
    if left_action_id == right_action_id {
        return Err(IntervalError::SameActionRequested);
    }
    let left_index = bank
        .action_ids
        .iter()
        .position(|id| *id == left_action_id)
        .ok_or(IntervalError::InvalidSelectedIndex)?;
    let right_index = bank
        .action_ids
        .iter()
        .position(|id| *id == right_action_id)
        .ok_or(IntervalError::InvalidSelectedIndex)?;
    let interval = paired_effect_interval(
        &bank.outcome_samples[left_index],
        &bank.outcome_samples[right_index],
        nominal_coverage,
    )?;
    Ok(ActionEffectInterval {
        left_action_id,
        right_action_id,
        paired_draw_count: bank.joint_draw_ids.len(),
        interval,
    })
}

/// Calculate all unordered action-pair effect intervals, in lexicographic
/// action-id order, independent of the input action-row order.
pub fn all_joint_action_effect_intervals(
    bank: &JointActionSetSamples,
    nominal_coverage: f64,
) -> Result<Vec<ActionEffectInterval>, IntervalError> {
    validate_joint_action_set(bank)?;
    let mut ids = bank.action_ids.clone();
    ids.sort_unstable();
    let mut effects = Vec::with_capacity(ids.len() * (ids.len() - 1) / 2);
    for left in 0..ids.len() {
        for right in (left + 1)..ids.len() {
            effects.push(joint_action_effect_interval_validated(
                bank,
                ids[left],
                ids[right],
                nominal_coverage,
            )?);
        }
    }
    Ok(effects)
}

/// Measure marginal interval coverage, sharpness and the proper interval score.
///
/// All observations must use the same nominal coverage. Empty input returns
/// None so a missing evaluation is not silently reported as zero error.
pub fn evaluate_intervals(
    observations: &[IntervalObservation],
) -> Result<Option<IntervalMetrics>, IntervalError> {
    let Some(first) = observations.first() else {
        return Ok(None);
    };
    let nominal_coverage = first.interval.nominal_coverage;
    let mut covered_count = 0usize;
    let mut total_width = 0.0;
    let mut total_interval_score = 0.0;

    for observation in observations {
        validate_observation(*observation)?;
        if observation.interval.nominal_coverage.to_bits() != nominal_coverage.to_bits() {
            return Err(IntervalError::MixedNominalCoverage);
        }

        let interval = observation.interval;
        let width = interval.upper - interval.lower;
        let alpha = 1.0 - nominal_coverage;
        let miss_penalty = if observation.realized < interval.lower {
            (2.0 / alpha) * (interval.lower - observation.realized)
        } else if observation.realized > interval.upper {
            (2.0 / alpha) * (observation.realized - interval.upper)
        } else {
            0.0
        };
        let interval_score = width + miss_penalty;
        if !interval_score.is_finite() {
            return Err(IntervalError::NonFiniteDerivedMetric);
        }

        if interval.lower <= observation.realized && observation.realized <= interval.upper {
            covered_count += 1;
        }
        total_width += width;
        total_interval_score += interval_score;
    }

    let count = observations.len() as f64;
    let empirical_coverage = covered_count as f64 / count;
    let mean_width = total_width / count;
    let mean_interval_score = total_interval_score / count;
    if !mean_width.is_finite() || !mean_interval_score.is_finite() {
        return Err(IntervalError::NonFiniteDerivedMetric);
    }
    Ok(Some(IntervalMetrics {
        sample_count: observations.len(),
        nominal_coverage,
        empirical_coverage,
        mean_width,
        mean_interval_score,
        absolute_coverage_error: (empirical_coverage - nominal_coverage).abs(),
    }))
}

/// Measure whether every action interval in an episode covers its realized
/// outcome. The episode—not the individual action interval—is the sampling
/// unit for simultaneous action-set coverage.
pub fn evaluate_simultaneous_sets(
    episodes: &[Vec<IntervalObservation>],
    familywise_nominal_coverage: f64,
) -> Result<Option<SimultaneousMetrics>, IntervalError> {
    validate_nominal_coverage(familywise_nominal_coverage)?;
    let Some(first_episode) = episodes.first() else {
        return Ok(None);
    };
    if first_episode.is_empty() {
        return Err(IntervalError::EmptyActionSet);
    }

    let intervals_per_episode = first_episode.len();
    let mut covered_episode_count = 0usize;
    let mut total_max_width = 0.0;

    for episode in episodes {
        if episode.is_empty() {
            return Err(IntervalError::EmptyActionSet);
        }
        if episode.len() != intervals_per_episode {
            return Err(IntervalError::UnequalActionSetCardinality);
        }

        let mut all_actions_covered = true;
        let mut max_width = 0.0_f64;
        for observation in episode {
            validate_observation(*observation)?;
            let width = observation.interval.upper - observation.interval.lower;
            max_width = max_width.max(width);
            all_actions_covered &= observation.interval.lower <= observation.realized
                && observation.realized <= observation.interval.upper;
        }
        if all_actions_covered {
            covered_episode_count += 1;
        }
        total_max_width += max_width;
    }

    let episode_count = episodes.len();
    let empirical_coverage = covered_episode_count as f64 / episode_count as f64;
    let mean_max_interval_width = total_max_width / episode_count as f64;
    if !mean_max_interval_width.is_finite() {
        return Err(IntervalError::NonFiniteDerivedMetric);
    }

    Ok(Some(SimultaneousMetrics {
        episode_count,
        intervals_per_episode,
        familywise_nominal_coverage,
        empirical_simultaneous_coverage: empirical_coverage,
        mean_max_interval_width,
        absolute_coverage_error: (empirical_coverage - familywise_nominal_coverage).abs(),
    }))
}

fn net_utility(outcome: CostedDecisionOutcome) -> Result<f64, IntervalError> {
    for cost in [
        outcome.inference_cost,
        outcome.acquisition_cost,
        outcome.failure_recovery_cost,
    ] {
        if !cost.is_finite() {
            return Err(IntervalError::NonFiniteDerivedMetric);
        }
        if cost < 0.0 {
            return Err(IntervalError::NegativeDecisionCost);
        }
    }
    if !outcome.realized_utility.is_finite() {
        return Err(IntervalError::NonFiniteRealization);
    }
    let net = outcome.realized_utility
        - outcome.inference_cost
        - outcome.acquisition_cost
        - outcome.failure_recovery_cost;
    if net.is_finite() {
        Ok(net)
    } else {
        Err(IntervalError::NonFiniteDerivedMetric)
    }
}

/// Compare world-model decisions with D0 and D1 on matched episodes.
///
/// All costs must already be converted to the preregistered common utility
/// scale. Returns paired mean advantages and episode win rates; uncertainty
/// intervals must be computed over episodes, not over individual cost terms.
pub fn evaluate_net_decision_value(
    episodes: &[DecisionValueEpisode],
) -> Result<Option<NetDecisionValueMetrics>, IntervalError> {
    if episodes.is_empty() {
        return Ok(None);
    }
    let mut gross_vs_d0 = 0.0;
    let mut net_vs_d0 = 0.0;
    let mut wins_vs_d0 = 0usize;
    let mut gross_vs_d1 = 0.0;
    let mut net_vs_d1 = 0.0;
    let mut wins_vs_d1 = 0usize;
    let mut world_net_total = 0.0;
    let mut d0_net_total = 0.0;
    let mut d1_net_total = 0.0;

    for episode in episodes {
        let world_net = net_utility(episode.world_model)?;
        let d0_net = net_utility(episode.direct_d0)?;
        let d1_net = net_utility(episode.direct_d1)?;
        let world_gross = episode.world_model.realized_utility;
        let d0_gross = episode.direct_d0.realized_utility;
        let d1_gross = episode.direct_d1.realized_utility;
        gross_vs_d0 += world_gross - d0_gross;
        net_vs_d0 += world_net - d0_net;
        gross_vs_d1 += world_gross - d1_gross;
        net_vs_d1 += world_net - d1_net;
        wins_vs_d0 += usize::from(world_net > d0_net);
        wins_vs_d1 += usize::from(world_net > d1_net);
        world_net_total += world_net;
        d0_net_total += d0_net;
        d1_net_total += d1_net;
    }

    let count = episodes.len() as f64;
    let metrics = NetDecisionValueMetrics {
        episode_count: episodes.len(),
        mean_gross_advantage_vs_d0: gross_vs_d0 / count,
        mean_net_advantage_vs_d0: net_vs_d0 / count,
        net_win_rate_vs_d0: wins_vs_d0 as f64 / count,
        mean_gross_advantage_vs_d1: gross_vs_d1 / count,
        mean_net_advantage_vs_d1: net_vs_d1 / count,
        net_win_rate_vs_d1: wins_vs_d1 as f64 / count,
        mean_world_model_net_utility: world_net_total / count,
        mean_d0_net_utility: d0_net_total / count,
        mean_d1_net_utility: d1_net_total / count,
    };
    if [
        metrics.mean_gross_advantage_vs_d0,
        metrics.mean_net_advantage_vs_d0,
        metrics.net_win_rate_vs_d0,
        metrics.mean_gross_advantage_vs_d1,
        metrics.mean_net_advantage_vs_d1,
        metrics.net_win_rate_vs_d1,
        metrics.mean_world_model_net_utility,
        metrics.mean_d0_net_utility,
        metrics.mean_d1_net_utility,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(IntervalError::NonFiniteDerivedMetric);
    }
    Ok(Some(metrics))
}

/// Select only a uniquely interval-dominant action; otherwise abstain.
///
/// The returned certificate is conditional on valid simultaneous coverage
/// for the complete candidate set. It is not a physical safety guarantee.
pub fn select_interval_dominant_action(
    intervals: &[PredictiveInterval],
) -> Result<RobustActionDecision, IntervalError> {
    if intervals.len() < 2 {
        return Err(IntervalError::InsufficientActions);
    }
    let nominal = intervals[0].nominal_coverage;
    for interval in intervals {
        validate_interval(*interval)?;
        if interval.nominal_coverage.to_bits() != nominal.to_bits() {
            return Err(IntervalError::MixedNominalCoverage);
        }
    }
    let mut dominant = None;
    for index in 0..intervals.len() {
        if is_interval_dominant(intervals, index)? {
            if dominant.is_some() {
                return Ok(RobustActionDecision::Abstain {
                    reason: AbstainReason::AmbiguousDominantActions,
                });
            }
            dominant = Some(index);
        }
    }
    Ok(match dominant {
        Some(action_index) => RobustActionDecision::Certified { action_index },
        None => RobustActionDecision::Abstain {
            reason: AbstainReason::NoIntervalDominantAction,
        },
    })
}

/// Evaluate whether uncertainty intervals support useful selected actions.
///
/// Outcomes are maximized. The regret bound is conditional on complete
/// action-set interval coverage. This reports both the empirical bound
/// coverage and false interval-dominance certificates; neither metric turns
/// the interval calculation into a safety proof.
pub fn evaluate_selected_action_sets(
    episodes: &[SelectionEpisode],
) -> Result<Option<SelectionMetrics>, IntervalError> {
    let Some(first) = episodes.first() else {
        return Ok(None);
    };
    if first.intervals.len() < 2 || first.realized_outcomes.len() != first.intervals.len() {
        return Err(IntervalError::UnequalActionSetCardinality);
    }
    let actions_per_episode = first.intervals.len();
    let nominal = first.intervals[0].nominal_coverage;
    let mut total_regret = 0.0;
    let mut total_bound = 0.0;
    let mut total_selected_width = 0.0;
    let mut covered_bounds = 0usize;
    let mut simultaneous_covered = 0usize;
    let mut dominant_certificates = 0usize;
    let mut false_dominant = 0usize;

    for episode in episodes {
        if episode.intervals.len() != actions_per_episode
            || episode.realized_outcomes.len() != actions_per_episode
        {
            return Err(IntervalError::UnequalActionSetCardinality);
        }
        if episode.selected_index >= actions_per_episode {
            return Err(IntervalError::InvalidSelectedIndex);
        }
        let mut best_realized = f64::NEG_INFINITY;
        let mut all_covered = true;
        for (interval, realized) in episode.intervals.iter().zip(&episode.realized_outcomes) {
            validate_interval(*interval)?;
            if !realized.is_finite() {
                return Err(IntervalError::NonFiniteRealization);
            }
            if interval.nominal_coverage.to_bits() != nominal.to_bits() {
                return Err(IntervalError::MixedNominalCoverage);
            }
            best_realized = best_realized.max(*realized);
            all_covered &= interval.lower <= *realized && *realized <= interval.upper;
        }

        let selected = episode.selected_index;
        let selected_realized = episode.realized_outcomes[selected];
        let regret = (best_realized - selected_realized).max(0.0);
        let bound = selected_action_regret_upper_bound(&episode.intervals, selected)?;
        let dominant = is_interval_dominant(&episode.intervals, selected)?;
        total_regret += regret;
        total_bound += bound;
        total_selected_width +=
            episode.intervals[selected].upper - episode.intervals[selected].lower;
        if bound + 1e-12 >= regret {
            covered_bounds += 1;
        }
        if all_covered {
            simultaneous_covered += 1;
        }
        if dominant {
            dominant_certificates += 1;
        }
        if dominant && regret > 1e-12 {
            false_dominant += 1;
        }
    }

    let count = episodes.len() as f64;
    let mean_realized_regret = total_regret / count;
    let mean_regret_upper_bound = total_bound / count;
    let mean_selected_interval_width = total_selected_width / count;
    if !mean_realized_regret.is_finite()
        || !mean_regret_upper_bound.is_finite()
        || !mean_selected_interval_width.is_finite()
    {
        return Err(IntervalError::NonFiniteDerivedMetric);
    }
    let dominant_certificate_count = dominant_certificates;
    Ok(Some(SelectionMetrics {
        episode_count: episodes.len(),
        actions_per_episode,
        mean_realized_regret,
        mean_regret_upper_bound,
        regret_bound_coverage: covered_bounds as f64 / count,
        simultaneous_interval_coverage: simultaneous_covered as f64 / count,
        dominant_certificate_count,
        false_dominant_certificate_count: false_dominant,
        false_dominant_certificate_rate: if dominant_certificate_count == 0 {
            0.0
        } else {
            false_dominant as f64 / dominant_certificate_count as f64
        },
        mean_selected_interval_width,
    }))
}

/// Return a conservative selected-action regret upper bound, conditional on
/// every action interval in the set containing its true outcome.
///
/// The bound is max(0, max competitor upper bound - selected lower bound).
/// Its empirical reliability depends on simultaneous—not merely marginal—
/// coverage of the complete action set.
pub fn selected_action_regret_upper_bound(
    intervals: &[PredictiveInterval],
    selected_index: usize,
) -> Result<f64, IntervalError> {
    if intervals.len() < 2 {
        return Err(IntervalError::InsufficientActions);
    }
    if selected_index >= intervals.len() {
        return Err(IntervalError::InvalidSelectedIndex);
    }
    for interval in intervals {
        validate_interval(*interval)?;
    }

    let selected_lower = intervals[selected_index].lower;
    let competitor_upper = intervals
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != selected_index)
        .map(|(_, interval)| interval.upper)
        .fold(f64::NEG_INFINITY, f64::max);
    let bound = (competitor_upper - selected_lower).max(0.0);
    if !bound.is_finite() {
        return Err(IntervalError::NonFiniteDerivedMetric);
    }
    Ok(bound)
}

/// True only when the selected interval's lower bound is at least every
/// competitor's upper bound. This is a certificate conditional on the
/// action-set intervals being jointly valid; it is not a safety proof.
pub fn is_interval_dominant(
    intervals: &[PredictiveInterval],
    selected_index: usize,
) -> Result<bool, IntervalError> {
    Ok(selected_action_regret_upper_bound(intervals, selected_index)? == 0.0)
}

fn validate_nominal_coverage(nominal_coverage: f64) -> Result<(), IntervalError> {
    if nominal_coverage.is_finite() && nominal_coverage > 0.0 && nominal_coverage < 1.0 {
        Ok(())
    } else {
        Err(IntervalError::InvalidNominalCoverage)
    }
}

fn validate_interval(interval: PredictiveInterval) -> Result<(), IntervalError> {
    validate_nominal_coverage(interval.nominal_coverage)?;
    if !interval.lower.is_finite() {
        return Err(IntervalError::NonFiniteLowerBound);
    }
    if !interval.upper.is_finite() {
        return Err(IntervalError::NonFiniteUpperBound);
    }
    if interval.lower > interval.upper {
        return Err(IntervalError::ReversedBounds);
    }
    if !(interval.upper - interval.lower).is_finite() {
        return Err(IntervalError::NonFiniteIntervalWidth);
    }
    Ok(())
}

fn validate_observation(observation: IntervalObservation) -> Result<(), IntervalError> {
    validate_interval(observation.interval)?;
    if observation.realized.is_finite() {
        Ok(())
    } else {
        Err(IntervalError::NonFiniteRealization)
    }
}

fn linear_quantile(sorted: &[f64], probability: f64) -> f64 {
    let rank = (sorted.len() - 1) as f64 * probability;
    let lower_index = rank.floor() as usize;
    let upper_index = rank.ceil() as usize;
    let fraction = rank - lower_index as f64;
    sorted[lower_index] + (sorted[upper_index] - sorted[lower_index]) * fraction
}

#[cfg(test)]
mod tests {
    use super::*;

    fn structured_visual_bank_fixture() -> StructuredJointActionSetSamples {
        let mut bank = StructuredJointActionSetSamples {
            history_sha256: "a".repeat(64),
            horizon_steps: 5,
            action_manifest_sha256: "c".repeat(64),
            action_ids: vec![17, 3, 9],
            joint_draw_ids: vec![100, 101, 102, 103, 104],
            joint_draw_bank_sha256: "d".repeat(64),
            coupling: JointDrawCoupling::SharedPosteriorAndWorldNoise,
            channels: vec![
                VisualBeliefChannelSamples {
                    channel_id: "gripper.contact_probability".into(),
                    kind: VisualBeliefChannelKind::Contact,
                    unit: "probability".into(),
                    outcome_samples: vec![
                        vec![0.9, 0.8, 0.95, 0.85, 0.9],
                        vec![0.5, 0.4, 0.55, 0.45, 0.5],
                        vec![0.2, 0.1, 0.25, 0.15, 0.2],
                    ],
                },
                VisualBeliefChannelSamples {
                    channel_id: "object_42.position.x".into(),
                    kind: VisualBeliefChannelKind::Geometry,
                    unit: "m".into(),
                    outcome_samples: vec![
                        vec![10.0, 20.0, 30.0, 40.0, 50.0],
                        vec![9.0, 19.0, 29.0, 39.0, 49.0],
                        vec![8.0, 18.0, 28.0, 38.0, 48.0],
                    ],
                },
                VisualBeliefChannelSamples {
                    channel_id: "task.utility".into(),
                    kind: VisualBeliefChannelKind::TaskUtility,
                    unit: "utility".into(),
                    outcome_samples: vec![
                        vec![5.0, 5.0, 5.0, 5.0, 5.0],
                        vec![3.0, 3.0, 3.0, 3.0, 3.0],
                        vec![4.0, 4.0, 4.0, 4.0, 4.0],
                    ],
                },
            ],
        };
        bank.joint_draw_bank_sha256 = compute_structured_joint_draw_bank_sha256(&bank).unwrap();
        bank
    }

    #[test]
    fn structured_visual_action_bank_preserves_semantic_channels_and_paired_effects() {
        let bank = structured_visual_bank_fixture();
        assert_eq!(
            bank.joint_draw_bank_sha256,
            "9202be4068e45989a72461f350c6bcea5893d44fde1575df3d671c0dc107f468"
        );
        validate_structured_joint_action_set(&bank).unwrap();

        let geometry =
            structured_action_effect_interval(&bank, "object_42.position.x", 17, 3, 0.8).unwrap();
        assert_eq!(geometry.kind, VisualBeliefChannelKind::Geometry);
        assert_eq!(geometry.unit, "m");
        assert_eq!(geometry.interval.lower, 1.0);
        assert_eq!(geometry.interval.upper, 1.0);

        let contact =
            structured_action_effect_interval(&bank, "gripper.contact_probability", 17, 3, 0.8)
                .unwrap();
        assert_eq!(contact.kind, VisualBeliefChannelKind::Contact);
        assert_eq!(contact.unit, "probability");
        assert!((contact.interval.lower - 0.4).abs() < 1e-12);
        assert!((contact.interval.upper - 0.4).abs() < 1e-12);

        let predictions = structured_action_prediction_intervals(&bank, 0.8).unwrap();
        assert_eq!(predictions.len(), 9);
        assert_eq!(predictions[0].channel_id, "gripper.contact_probability");
        assert_eq!(predictions[0].action_id, 3);

        let all_effects = all_structured_action_effect_intervals(&bank, 0.8).unwrap();
        assert_eq!(all_effects.len(), 9);
        assert_eq!(all_effects[0].channel_id, "gripper.contact_probability");
        assert_eq!(
            (
                all_effects[0].left_action_id,
                all_effects[0].right_action_id
            ),
            (3, 9)
        );
    }

    #[test]
    fn structured_visual_action_bank_rejects_channel_and_draw_tampering() {
        let mut changed = structured_visual_bank_fixture();
        changed.channels[0].outcome_samples[0][0] += 0.01;
        assert_eq!(
            validate_structured_joint_action_set(&changed),
            Err(IntervalError::JointDrawBankDigestMismatch)
        );

        let mut duplicate_channel = structured_visual_bank_fixture();
        duplicate_channel.channels[1].channel_id = duplicate_channel.channels[0].channel_id.clone();
        assert_eq!(
            validate_structured_joint_action_set_shape(&duplicate_channel),
            Err(IntervalError::DuplicateVisualChannelId)
        );

        let mut invalid_descriptor = structured_visual_bank_fixture();
        invalid_descriptor.channels[0].unit = " ".into();
        assert_eq!(
            validate_structured_joint_action_set_shape(&invalid_descriptor),
            Err(IntervalError::InvalidVisualChannelDescriptor)
        );

        let mut noncanonical_id = structured_visual_bank_fixture();
        noncanonical_id.channels[0].channel_id = " gripper.contact_probability".into();
        assert_eq!(
            validate_structured_joint_action_set_shape(&noncanonical_id),
            Err(IntervalError::InvalidVisualChannelDescriptor)
        );

        let mut bad_shape = structured_visual_bank_fixture();
        bad_shape.channels[0].outcome_samples[0].pop();
        assert_eq!(
            validate_structured_joint_action_set_shape(&bad_shape),
            Err(IntervalError::StructuredChannelSampleCardinalityMismatch)
        );
    }

    #[test]
    fn structured_action_selection_uses_only_named_task_utility_channel() {
        let bank = structured_visual_bank_fixture();
        assert_eq!(
            select_structured_task_action(&bank, "task.utility", 0.8).unwrap(),
            StructuredActionDecision::Certified { action_id: 17 }
        );
        assert_eq!(
            select_structured_task_action(&bank, "object_42.position.x", 0.8),
            Err(IntervalError::NonUtilitySelectionChannel)
        );
        assert_eq!(
            select_structured_task_action(&bank, "missing.utility", 0.8),
            Err(IntervalError::InvalidVisualChannelDescriptor)
        );
    }


    #[test]
    fn constrained_structured_selection_keeps_risk_separate_from_utility() {
        let mut bank = structured_visual_bank_fixture();
        bank.channels.push(VisualBeliefChannelSamples {
            channel_id: "terminal.risk".into(),
            kind: VisualBeliefChannelKind::TerminalRisk,
            unit: "probability".into(),
            outcome_samples: vec![
                vec![0.10, 0.12, 0.11, 0.09, 0.10],
                vec![0.50, 0.52, 0.51, 0.49, 0.50],
                vec![0.40, 0.42, 0.41, 0.39, 0.40],
            ],
        });
        bank.joint_draw_bank_sha256 = compute_structured_joint_draw_bank_sha256(&bank).unwrap();
        let risk_cap = [StructuredActionConstraint {
            channel_id: "terminal.risk".into(),
            kind: VisualBeliefChannelKind::TerminalRisk,
            relation: ConstraintRelation::UpperAtMost,
            threshold: 0.2,
        }];
        assert_eq!(
            select_constrained_structured_action(&bank, "task.utility", &risk_cap, 0.8).unwrap(),
            ConstrainedActionDecision::Selected {
                action_id: 17,
                basis: ConstrainedSelectionBasis::FeasibleOnly,
            }
        );

        let no_feasible = [StructuredActionConstraint {
            threshold: 0.05,
            ..risk_cap[0].clone()
        }];
        assert_eq!(
            select_constrained_structured_action(&bank, "task.utility", &no_feasible, 0.8).unwrap(),
            ConstrainedActionDecision::Abstain {
                reason: AbstainReason::NoConstraintFeasibleActions,
            }
        );

        let permissive = [StructuredActionConstraint {
            threshold: 0.8,
            ..risk_cap[0].clone()
        }];
        assert_eq!(
            select_constrained_structured_action(&bank, "task.utility", &permissive, 0.8).unwrap(),
            ConstrainedActionDecision::Selected {
                action_id: 17,
                basis: ConstrainedSelectionBasis::UtilityDominance,
            }
        );
    }

    #[test]
    fn constrained_structured_selection_rejects_duplicate_or_mistyped_constraints() {
        let bank = structured_visual_bank_fixture();
        let duplicate = [
            StructuredActionConstraint {
                channel_id: "gripper.contact_probability".into(),
                kind: VisualBeliefChannelKind::Contact,
                relation: ConstraintRelation::LowerAtLeast,
                threshold: 0.7,
            },
            StructuredActionConstraint {
                channel_id: "gripper.contact_probability".into(),
                kind: VisualBeliefChannelKind::Contact,
                relation: ConstraintRelation::LowerAtLeast,
                threshold: 0.6,
            },
        ];
        assert_eq!(
            select_constrained_structured_action(&bank, "task.utility", &duplicate, 0.8),
            Err(IntervalError::InvalidVisualChannelDescriptor)
        );
        let mistyped = [StructuredActionConstraint {
            channel_id: "gripper.contact_probability".into(),
            kind: VisualBeliefChannelKind::TerminalRisk,
            relation: ConstraintRelation::UpperAtMost,
            threshold: 0.3,
        }];
        assert_eq!(
            select_constrained_structured_action(&bank, "task.utility", &mistyped, 0.8),
            Err(IntervalError::InvalidVisualChannelDescriptor)
        );
    }

    fn info_option(
        query_id: &str,
        cost: f64,
        probabilities: Vec<f64>,
        utilities: Vec<Vec<f64>>,
    ) -> InformationAcquisitionOption {
        let mut option = InformationAcquisitionOption {
            history_sha256: "a".repeat(64),
            action_manifest_sha256: "b".repeat(64),
            query_manifest_sha256: "c".repeat(64),
            query_id: query_id.into(),
            action_ids: vec![17, 3],
            outcome_ids: (0..probabilities.len()).map(|i| format!("outcome-{i}")).collect(),
            outcome_probabilities: probabilities,
            conditional_action_utilities: utilities,
            acquisition_cost: cost,
            evidence_sha256: "d".repeat(64),
        };
        option.evidence_sha256 = compute_information_option_sha256(&option).unwrap();
        option
    }

    #[test]
    fn evsi_selects_only_information_with_positive_net_decision_value() {
        let informative = info_option(
            "camera_peek",
            1.0,
            vec![0.5, 0.5],
            vec![vec![6.0, 3.0], vec![2.0, 7.0]],
        );
        let expensive = info_option(
            "expensive_probe",
            2.0,
            vec![0.5, 0.5],
            vec![vec![6.0, 3.0], vec![2.0, 7.0]],
        );
        let evaluation = evaluate_information_acquisition(&[expensive, informative]).unwrap();
        assert_eq!(evaluation.choice, InformationAcquisitionChoice::AcquireBestQuery);
        assert_eq!(evaluation.selected_query_id.as_deref(), Some("camera_peek"));
        assert_eq!(evaluation.options[0].query_id, "camera_peek");
        assert_eq!(evaluation.options[0].prior_best_action_id, 3);
        assert_eq!(evaluation.options[0].best_action_by_outcome, vec![17, 3]);
        assert_eq!(evaluation.options[0].expected_utility_without_information, 5.0);
        assert_eq!(evaluation.options[0].expected_utility_after_information, 6.5);
        assert_eq!(evaluation.options[0].expected_value_of_sample_information, 1.5);
        assert_eq!(evaluation.options[0].net_information_value, 0.5);
        assert_eq!(evaluation.options[1].net_information_value, -0.5);
    }

    #[test]
    fn evsi_abstains_when_information_cost_exceeds_its_decision_value() {
        let option = info_option(
            "camera_peek",
            2.0,
            vec![0.5, 0.5],
            vec![vec![6.0, 3.0], vec![2.0, 7.0]],
        );
        let evaluation = evaluate_information_acquisition(&[option]).unwrap();
        assert_eq!(
            evaluation.choice,
            InformationAcquisitionChoice::Abstain {
                reason: AbstainReason::NoPositiveInformationValue,
            }
        );
        assert_eq!(evaluation.selected_query_id, None);
    }

    #[test]
    fn evsi_digest_binds_history_manifests_outcomes_probabilities_utilities_and_cost() {
        let mut option = info_option(
            "camera_peek",
            1.0,
            vec![0.5, 0.5],
            vec![vec![6.0, 3.0], vec![2.0, 7.0]],
        );
        option.outcome_ids = vec!["grasp_success".into(), "slip_detected".into()];
        option.evidence_sha256 = compute_information_option_sha256(&option).unwrap();
        assert_eq!(
            option.evidence_sha256,
            "80f500f511bff1bccfd0ca03aa164de21ea84a64e8c924a4b2385132e62257e8"
        );
        let mut tampered = option.clone();
        tampered.conditional_action_utilities[0][0] += 0.01;
        assert_eq!(
            evaluate_information_acquisition(&[tampered]),
            Err(IntervalError::JointDrawBankDigestMismatch)
        );
        let mut changed_history = option.clone();
        changed_history.history_sha256 = "e".repeat(64);
        assert_ne!(
            compute_information_option_sha256(&changed_history).unwrap(),
            option.evidence_sha256
        );
    }

    #[test]
    fn evsi_rejects_invalid_probabilities_costs_and_action_shapes() {
        let bad_probability = info_option(
            "bad",
            0.0,
            vec![0.4, 0.4],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        assert_eq!(
            evaluate_information_acquisition(&[bad_probability]),
            Err(IntervalError::InvalidOutcomeProbabilities)
        );
        let bad_cost = info_option(
            "bad_cost",
            -1.0,
            vec![0.5, 0.5],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        assert_eq!(
            evaluate_information_acquisition(&[bad_cost]),
            Err(IntervalError::InvalidInformationOption)
        );
        let bad_shape = info_option(
            "bad_shape",
            0.0,
            vec![0.5, 0.5],
            vec![vec![1.0], vec![0.0, 1.0]],
        );
        assert_eq!(
            evaluate_information_acquisition(&[bad_shape]),
            Err(IntervalError::InvalidInformationOption)
        );
    }

    #[test]
    fn structured_visual_action_effect_is_action_row_permutation_invariant() {
        let original = structured_visual_bank_fixture();
        let expected =
            structured_action_effect_interval(&original, "object_42.position.x", 17, 9, 0.8)
                .unwrap();

        let mut permuted = original.clone();
        permuted.action_ids = vec![9, 17, 3];
        for channel in &mut permuted.channels {
            channel.outcome_samples = vec![
                channel.outcome_samples[2].clone(),
                channel.outcome_samples[0].clone(),
                channel.outcome_samples[1].clone(),
            ];
        }
        permuted.joint_draw_bank_sha256 =
            compute_structured_joint_draw_bank_sha256(&permuted).unwrap();

        let actual =
            structured_action_effect_interval(&permuted, "object_42.position.x", 17, 9, 0.8)
                .unwrap();
        assert_eq!(actual.interval, expected.interval);
        assert_eq!(actual.paired_draw_count, expected.paired_draw_count);
    }

    fn joint_bank_fixture() -> JointActionSetSamples {
        let mut bank = JointActionSetSamples {
            history_sha256: "a".repeat(64),
            horizon_steps: 5,
            action_manifest_sha256: "c".repeat(64),
            outcome_channel: "predicted_task_utility".into(),
            action_ids: vec![17, 3, 9],
            joint_draw_ids: vec![100, 101, 102, 103, 104],
            joint_draw_bank_sha256: "b".repeat(64),
            coupling: JointDrawCoupling::SharedPosteriorAndWorldNoise,
            outcome_samples: vec![
                vec![100.0, 110.0, 120.0, 130.0, 140.0],
                vec![99.0, 109.0, 119.0, 129.0, 139.0],
                vec![97.0, 107.0, 117.0, 127.0, 137.0],
            ],
        };
        bank.joint_draw_bank_sha256 = compute_joint_draw_bank_sha256(&bank).unwrap();
        bank
    }

    #[test]
    fn joint_action_bank_preserves_common_mode_effect_cancellation() {
        let bank = joint_bank_fixture();
        validate_joint_action_set(&bank).unwrap();
        let effect = joint_action_effect_interval(&bank, 17, 3, 0.8).unwrap();
        assert_eq!(effect.left_action_id, 17);
        assert_eq!(effect.right_action_id, 3);
        assert_eq!(effect.paired_draw_count, 5);
        assert!((effect.interval.lower - 1.0).abs() < 1e-12);
        assert!((effect.interval.upper - 1.0).abs() < 1e-12);
    }

    #[test]
    fn joint_action_effects_are_invariant_to_action_row_permutation() {
        let original = joint_bank_fixture();
        let original_effect = joint_action_effect_interval(&original, 17, 9, 0.8).unwrap();
        let mut permuted = JointActionSetSamples {
            action_ids: vec![9, 17, 3],
            outcome_samples: vec![
                original.outcome_samples[2].clone(),
                original.outcome_samples[0].clone(),
                original.outcome_samples[1].clone(),
            ],
            ..original.clone()
        };
        permuted.joint_draw_bank_sha256 = compute_joint_draw_bank_sha256(&permuted).unwrap();
        let permuted_effect = joint_action_effect_interval(&permuted, 17, 9, 0.8).unwrap();
        assert_eq!(original_effect, permuted_effect);
        let all = all_joint_action_effect_intervals(&permuted, 0.8).unwrap();
        assert_eq!(
            all.iter()
                .map(|effect| (effect.left_action_id, effect.right_action_id))
                .collect::<Vec<_>>(),
            vec![(3, 9), (3, 17), (9, 17)]
        );
    }

    #[test]
    fn joint_draw_bank_digest_has_sha256_known_answer_and_binds_samples() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut bank = joint_bank_fixture();
        let digest = bank.joint_draw_bank_sha256.clone();
        assert_eq!(
            digest,
            "47a1f2b937faef8235704f6907523967c946cad8184bf7c5f643916bd3441979"
        );
        assert_eq!(compute_joint_draw_bank_sha256(&bank).unwrap(), digest);
        bank.outcome_samples[0][0] += 0.25;
        assert_eq!(
            validate_joint_action_set(&bank),
            Err(IntervalError::JointDrawBankDigestMismatch)
        );
    }

    #[test]
    fn joint_draw_bank_hash_binds_action_identity_and_coupling() {
        let bank = joint_bank_fixture();
        let digest = bank.joint_draw_bank_sha256.clone();
        let mut changed = bank.clone();
        changed.action_ids.swap(0, 1);
        changed.outcome_samples.swap(0, 1);
        assert_ne!(compute_joint_draw_bank_sha256(&changed).unwrap(), digest);
        changed = bank.clone();
        changed.coupling = JointDrawCoupling::PairedExogenousDisturbance;
        assert_ne!(compute_joint_draw_bank_sha256(&changed).unwrap(), digest);
    }

    #[test]
    fn joint_action_bank_rejects_duplicate_ids_and_bad_cardinality() {
        let mut bank = joint_bank_fixture();
        bank.action_ids[1] = bank.action_ids[0];
        assert_eq!(
            validate_joint_action_set(&bank),
            Err(IntervalError::DuplicateActionId)
        );

        let mut bank = joint_bank_fixture();
        bank.joint_draw_ids[1] = bank.joint_draw_ids[0];
        assert_eq!(
            validate_joint_action_set(&bank),
            Err(IntervalError::DuplicateJointDrawId)
        );

        let mut bank = joint_bank_fixture();
        bank.outcome_samples[1].pop();
        assert_eq!(
            validate_joint_action_set(&bank),
            Err(IntervalError::ActionSampleCardinalityMismatch)
        );
    }

    #[test]
    fn joint_action_bank_requires_action_semantics_and_outcome_channel_identity() {
        let mut bank = joint_bank_fixture();
        bank.action_manifest_sha256 = "bad".into();
        assert_eq!(
            validate_joint_action_set(&bank),
            Err(IntervalError::InvalidActionManifestDigest)
        );

        let mut bank = joint_bank_fixture();
        bank.outcome_channel = "  ".into();
        assert_eq!(
            validate_joint_action_set(&bank),
            Err(IntervalError::EmptyOutcomeChannel)
        );
    }

    #[test]
    fn joint_action_bank_rejects_missing_or_invalid_provenance_and_horizon() {
        let mut bank = joint_bank_fixture();
        bank.history_sha256 = "not-a-digest".into();
        assert_eq!(
            validate_joint_action_set(&bank),
            Err(IntervalError::InvalidHistoryDigest)
        );

        let mut bank = joint_bank_fixture();
        bank.joint_draw_bank_sha256 = "B".repeat(64);
        assert_eq!(
            validate_joint_action_set(&bank),
            Err(IntervalError::InvalidJointDrawDigest)
        );

        let mut bank = joint_bank_fixture();
        bank.horizon_steps = 0;
        assert_eq!(
            validate_joint_action_set(&bank),
            Err(IntervalError::InvalidHorizon)
        );
    }

    #[test]
    fn joint_action_bank_rejects_nonfinite_draws_and_same_action_effect() {
        let mut bank = joint_bank_fixture();
        bank.outcome_samples[0][0] = f64::NAN;
        assert_eq!(
            validate_joint_action_set(&bank),
            Err(IntervalError::NonFiniteDerivedMetric)
        );

        let bank = joint_bank_fixture();
        assert_eq!(
            joint_action_effect_interval(&bank, 17, 17, 0.8),
            Err(IntervalError::SameActionRequested)
        );
        assert_eq!(
            joint_action_effect_interval(&bank, 17, 999, 0.8),
            Err(IntervalError::InvalidSelectedIndex)
        );
    }

    fn interval(lower: f64, upper: f64, nominal_coverage: f64) -> PredictiveInterval {
        PredictiveInterval {
            lower,
            upper,
            nominal_coverage,
        }
    }

    fn observation(
        lower: f64,
        upper: f64,
        nominal_coverage: f64,
        realized: f64,
    ) -> IntervalObservation {
        IntervalObservation {
            interval: interval(lower, upper, nominal_coverage),
            realized,
        }
    }

    #[test]
    fn interval_metrics_report_coverage_width_and_proper_score() {
        let observations = [
            observation(0.0, 2.0, 0.9, 0.0),
            observation(0.0, 2.0, 0.9, 1.0),
            observation(0.0, 2.0, 0.9, 2.0),
            observation(0.0, 2.0, 0.9, 3.0),
        ];
        let metrics = evaluate_intervals(&observations).unwrap().unwrap();
        assert_eq!(metrics.sample_count, 4);
        assert!((metrics.empirical_coverage - 0.75).abs() < 1e-12);
        assert!((metrics.mean_width - 2.0).abs() < 1e-12);
        assert!((metrics.mean_interval_score - 7.0).abs() < 1e-12);
        assert!((metrics.absolute_coverage_error - 0.15).abs() < 1e-12);
    }

    #[test]
    fn interval_score_penalizes_misses_on_both_sides() {
        let observations = [
            observation(0.0, 2.0, 0.8, -1.0),
            observation(0.0, 2.0, 0.8, 3.0),
        ];
        let metrics = evaluate_intervals(&observations).unwrap().unwrap();
        assert!((metrics.mean_interval_score - 12.0).abs() < 1e-12);
        assert_eq!(metrics.empirical_coverage, 0.0);
    }

    #[test]
    fn interval_boundaries_count_as_covered_and_empty_input_is_missing() {
        let observations = [
            observation(0.0, 1.0, 0.9, 0.0),
            observation(0.0, 1.0, 0.9, 1.0),
        ];
        let metrics = evaluate_intervals(&observations).unwrap().unwrap();
        assert_eq!(metrics.empirical_coverage, 1.0);
        assert_eq!(evaluate_intervals(&[]).unwrap(), None);
    }

    #[test]
    fn mixed_nominal_levels_are_rejected() {
        let observations = [
            observation(0.0, 1.0, 0.9, 0.5),
            observation(0.0, 1.0, 0.8, 0.5),
        ];
        assert_eq!(
            evaluate_intervals(&observations),
            Err(IntervalError::MixedNominalCoverage)
        );
    }

    #[test]
    fn invalid_intervals_and_nonfinite_realizations_fail_closed() {
        assert_eq!(
            evaluate_intervals(&[observation(2.0, 1.0, 0.9, 1.5)]),
            Err(IntervalError::ReversedBounds)
        );
        assert_eq!(
            evaluate_intervals(&[observation(0.0, 1.0, 1.0, 0.5)]),
            Err(IntervalError::InvalidNominalCoverage)
        );
        assert_eq!(
            evaluate_intervals(&[observation(0.0, 1.0, 0.9, f64::NAN)]),
            Err(IntervalError::NonFiniteRealization)
        );
    }

    #[test]
    fn paired_effect_interval_preserves_common_mode_cancellation() {
        let left = [100.0, 110.0, 120.0, 130.0, 140.0];
        let right = [99.0, 109.0, 119.0, 129.0, 139.0];
        let effect = paired_effect_interval(&left, &right, 0.8).unwrap();
        assert!((effect.lower - 1.0).abs() < 1e-12);
        assert!((effect.upper - 1.0).abs() < 1e-12);
    }

    #[test]
    fn paired_effect_interval_rejects_unpaired_or_insufficient_samples() {
        assert_eq!(
            paired_effect_interval(&[1.0, 2.0], &[0.0], 0.9),
            Err(IntervalError::UnequalPairedSampleCount)
        );
        assert_eq!(
            paired_effect_interval(&[1.0], &[0.0], 0.9),
            Err(IntervalError::InsufficientSamples)
        );
    }

    #[test]
    fn simultaneous_coverage_counts_whole_action_sets_not_individual_pairs() {
        let episodes = vec![
            vec![
                observation(0.0, 2.0, 0.9, 1.0),
                observation(0.0, 2.0, 0.9, 1.0),
            ],
            vec![
                observation(0.0, 2.0, 0.9, 1.0),
                observation(0.0, 2.0, 0.9, 3.0),
            ],
        ];
        let metrics = evaluate_simultaneous_sets(&episodes, 0.9).unwrap().unwrap();
        assert_eq!(metrics.episode_count, 2);
        assert_eq!(metrics.intervals_per_episode, 2);
        assert_eq!(metrics.empirical_simultaneous_coverage, 0.5);
        assert_eq!(metrics.mean_max_interval_width, 2.0);
        assert_eq!(metrics.absolute_coverage_error, 0.4);
    }

    #[test]
    fn simultaneous_coverage_rejects_empty_and_mismatched_action_banks() {
        assert_eq!(evaluate_simultaneous_sets(&[], 0.9).unwrap(), None);
        assert_eq!(
            evaluate_simultaneous_sets(&[vec![]], 0.9),
            Err(IntervalError::EmptyActionSet)
        );
        let mismatched = vec![
            vec![observation(0.0, 1.0, 0.9, 0.5)],
            vec![
                observation(0.0, 1.0, 0.9, 0.5),
                observation(0.0, 1.0, 0.9, 0.5),
            ],
        ];
        assert_eq!(
            evaluate_simultaneous_sets(&mismatched, 0.9),
            Err(IntervalError::UnequalActionSetCardinality)
        );
    }

    #[test]
    fn selected_action_regret_bound_and_dominance_use_competitor_intervals() {
        let intervals = [
            interval(3.0, 4.0, 0.9),
            interval(0.0, 2.0, 0.9),
            interval(1.0, 2.5, 0.9),
        ];
        assert_eq!(
            selected_action_regret_upper_bound(&intervals, 0).unwrap(),
            0.0
        );
        assert!(is_interval_dominant(&intervals, 0).unwrap());

        let uncertain = [interval(1.5, 2.0, 0.9), interval(0.0, 2.5, 0.9)];
        assert_eq!(
            selected_action_regret_upper_bound(&uncertain, 0).unwrap(),
            1.0
        );
        assert!(!is_interval_dominant(&uncertain, 0).unwrap());
    }

    fn costed(
        utility: f64,
        inference: f64,
        acquisition: f64,
        recovery: f64,
    ) -> CostedDecisionOutcome {
        CostedDecisionOutcome {
            realized_utility: utility,
            inference_cost: inference,
            acquisition_cost: acquisition,
            failure_recovery_cost: recovery,
        }
    }

    #[test]
    fn net_decision_value_charges_inference_acquisition_and_failure_costs() {
        let episodes = vec![
            DecisionValueEpisode {
                world_model: costed(10.0, 1.0, 0.5, 0.0),
                direct_d0: costed(8.0, 0.0, 0.0, 0.0),
                direct_d1: costed(9.0, 0.5, 0.0, 0.0),
            },
            DecisionValueEpisode {
                world_model: costed(12.0, 2.0, 0.0, 1.0),
                direct_d0: costed(8.0, 0.0, 0.0, 0.0),
                direct_d1: costed(8.5, 0.5, 0.0, 0.0),
            },
            DecisionValueEpisode {
                world_model: costed(9.0, 1.0, 0.5, 0.0),
                direct_d0: costed(8.0, 0.0, 0.0, 0.0),
                direct_d1: costed(8.5, 0.5, 0.0, 0.0),
            },
        ];
        let metrics = evaluate_net_decision_value(&episodes).unwrap().unwrap();
        assert_eq!(metrics.episode_count, 3);
        assert!((metrics.mean_gross_advantage_vs_d0 - (7.0 / 3.0)).abs() < 1e-12);
        assert!((metrics.mean_net_advantage_vs_d0 - (1.0 / 3.0)).abs() < 1e-12);
        assert!((metrics.net_win_rate_vs_d0 - (2.0 / 3.0)).abs() < 1e-12);
        assert!((metrics.mean_gross_advantage_vs_d1 - (5.0 / 3.0)).abs() < 1e-12);
        assert!((metrics.mean_net_advantage_vs_d1 - (0.5 / 3.0)).abs() < 1e-12);
        assert!((metrics.net_win_rate_vs_d1 - (1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn net_decision_value_rejects_negative_costs_and_nonfinite_utilities() {
        let invalid_cost = [DecisionValueEpisode {
            world_model: costed(1.0, -1.0, 0.0, 0.0),
            direct_d0: costed(1.0, 0.0, 0.0, 0.0),
            direct_d1: costed(1.0, 0.0, 0.0, 0.0),
        }];
        assert_eq!(
            evaluate_net_decision_value(&invalid_cost),
            Err(IntervalError::NegativeDecisionCost)
        );
        let invalid_utility = [DecisionValueEpisode {
            world_model: costed(f64::NAN, 0.0, 0.0, 0.0),
            direct_d0: costed(1.0, 0.0, 0.0, 0.0),
            direct_d1: costed(1.0, 0.0, 0.0, 0.0),
        }];
        assert_eq!(
            evaluate_net_decision_value(&invalid_utility),
            Err(IntervalError::NonFiniteRealization)
        );
        assert_eq!(evaluate_net_decision_value(&[]).unwrap(), None);
    }

    #[test]
    fn selected_action_evaluator_scores_realized_regret_and_certificate_quality() {
        let episodes = vec![
            SelectionEpisode {
                intervals: vec![interval(2.0, 4.0, 0.9), interval(0.0, 3.0, 0.9)],
                realized_outcomes: vec![3.5, 3.0],
                selected_index: 0,
            },
            SelectionEpisode {
                intervals: vec![interval(0.0, 2.0, 0.9), interval(3.0, 4.0, 0.9)],
                realized_outcomes: vec![1.5, 3.5],
                selected_index: 0,
            },
            SelectionEpisode {
                intervals: vec![interval(3.0, 4.0, 0.9), interval(0.0, 2.0, 0.9)],
                realized_outcomes: vec![3.5, 1.0],
                selected_index: 0,
            },
            SelectionEpisode {
                intervals: vec![interval(3.0, 4.0, 0.9), interval(0.0, 2.0, 0.9)],
                realized_outcomes: vec![1.0, 4.0],
                selected_index: 0,
            },
        ];
        let metrics = evaluate_selected_action_sets(&episodes).unwrap().unwrap();
        assert_eq!(metrics.episode_count, 4);
        assert_eq!(metrics.actions_per_episode, 2);
        assert!((metrics.mean_realized_regret - 1.25).abs() < 1e-12);
        assert!((metrics.mean_regret_upper_bound - 1.25).abs() < 1e-12);
        assert_eq!(metrics.regret_bound_coverage, 0.75);
        assert_eq!(metrics.simultaneous_interval_coverage, 0.75);
        assert_eq!(metrics.dominant_certificate_count, 2);
        assert_eq!(metrics.false_dominant_certificate_count, 1);
        assert_eq!(metrics.false_dominant_certificate_rate, 0.5);
    }

    #[test]
    fn selected_action_evaluator_rejects_bad_episode_shapes_and_nonfinite_truth() {
        assert_eq!(evaluate_selected_action_sets(&[]).unwrap(), None);
        let bad_shape = [SelectionEpisode {
            intervals: vec![interval(0.0, 1.0, 0.9), interval(0.0, 1.0, 0.9)],
            realized_outcomes: vec![0.5],
            selected_index: 0,
        }];
        assert_eq!(
            evaluate_selected_action_sets(&bad_shape),
            Err(IntervalError::UnequalActionSetCardinality)
        );
        let bad_index = [SelectionEpisode {
            intervals: vec![interval(0.0, 1.0, 0.9), interval(0.0, 1.0, 0.9)],
            realized_outcomes: vec![0.5, 0.5],
            selected_index: 2,
        }];
        assert_eq!(
            evaluate_selected_action_sets(&bad_index),
            Err(IntervalError::InvalidSelectedIndex)
        );
        let bad_truth = [SelectionEpisode {
            intervals: vec![interval(0.0, 1.0, 0.9), interval(0.0, 1.0, 0.9)],
            realized_outcomes: vec![0.5, f64::NAN],
            selected_index: 0,
        }];
        assert_eq!(
            evaluate_selected_action_sets(&bad_truth),
            Err(IntervalError::NonFiniteRealization)
        );
    }

    #[test]
    fn robust_action_selection_certifies_only_a_unique_interval_dominant_action() {
        let intervals = [
            interval(3.0, 4.0, 0.9),
            interval(0.0, 2.0, 0.9),
            interval(1.0, 2.5, 0.9),
        ];
        assert_eq!(
            select_interval_dominant_action(&intervals).unwrap(),
            RobustActionDecision::Certified { action_index: 0 }
        );

        let overlapping = [interval(1.0, 3.0, 0.9), interval(2.0, 4.0, 0.9)];
        assert_eq!(
            select_interval_dominant_action(&overlapping).unwrap(),
            RobustActionDecision::Abstain {
                reason: AbstainReason::NoIntervalDominantAction
            }
        );

        let boundary_tie = [interval(2.0, 2.0, 0.9), interval(2.0, 2.0, 0.9)];
        assert_eq!(
            select_interval_dominant_action(&boundary_tie).unwrap(),
            RobustActionDecision::Abstain {
                reason: AbstainReason::AmbiguousDominantActions
            }
        );
    }

    #[test]
    fn robust_action_selection_rejects_mixed_levels_and_too_small_sets() {
        assert_eq!(
            select_interval_dominant_action(&[interval(0.0, 1.0, 0.9)]),
            Err(IntervalError::InsufficientActions)
        );
        assert_eq!(
            select_interval_dominant_action(&[interval(2.0, 3.0, 0.9), interval(0.0, 1.0, 0.8),]),
            Err(IntervalError::MixedNominalCoverage)
        );
    }

    #[test]
    fn selection_certificate_rejects_invalid_action_selection() {
        assert_eq!(
            selected_action_regret_upper_bound(&[interval(0.0, 1.0, 0.9)], 0),
            Err(IntervalError::InsufficientActions)
        );
        assert_eq!(
            selected_action_regret_upper_bound(
                &[interval(0.0, 1.0, 0.9), interval(0.0, 1.0, 0.9)],
                2
            ),
            Err(IntervalError::InvalidSelectedIndex)
        );
    }
}
