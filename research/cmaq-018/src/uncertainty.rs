//! Selection-aware uncertainty metrics for CMAQ-018.
//!
//! These are metric and interval-construction primitives, not a production
//! predictor and not a claim that any model is calibrated. In particular,
//! simultaneous coverage must be measured over the complete action set;
//! pointwise intervals do not automatically remain calibrated after selecting
//! the most attractive action.

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbstainReason {
    NoIntervalDominantAction,
    AmbiguousDominantActions,
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
pub fn validate_joint_action_set(
    bank: &JointActionSetSamples,
) -> Result<(), IntervalError> {
    validate_joint_action_set_shape(bank)?;
    let expected_digest = compute_joint_draw_bank_sha256(bank)?;
    if expected_digest != bank.joint_draw_bank_sha256 {
        return Err(IntervalError::JointDrawBankDigestMismatch);
    }
    Ok(())
}

fn validate_joint_action_set_shape(
    bank: &JointActionSetSamples,
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

fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ];
    let mut h = [0x6a09e667u32,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut message = input.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 { message.push(0); }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let start = i * 4;
            w[i] = u32::from_be_bytes([chunk[start], chunk[start+1], chunk[start+2], chunk[start+3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temp1 = hh.wrapping_add(s1).wrapping_add(choice).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            hh = g; g = f; f = e; e = d.wrapping_add(temp1);
            d = c; c = b; b = a; a = temp1.wrapping_add(temp2);
        }
        h[0]=h[0].wrapping_add(a); h[1]=h[1].wrapping_add(b); h[2]=h[2].wrapping_add(c); h[3]=h[3].wrapping_add(d);
        h[4]=h[4].wrapping_add(e); h[5]=h[5].wrapping_add(f); h[6]=h[6].wrapping_add(g); h[7]=h[7].wrapping_add(hh);
    }
    let mut output = String::with_capacity(64);
    for word in h { write!(&mut output, "{word:08x}").unwrap(); }
    output
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
    let left_index = bank.action_ids.iter().position(|id| *id == left_action_id)
        .ok_or(IntervalError::InvalidSelectedIndex)?;
    let right_index = bank.action_ids.iter().position(|id| *id == right_action_id)
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
                bank, ids[left], ids[right], nominal_coverage
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
        total_selected_width += episode.intervals[selected].upper - episode.intervals[selected].lower;
        if bound + 1e-12 >= regret { covered_bounds += 1; }
        if all_covered { simultaneous_covered += 1; }
        if dominant { dominant_certificates += 1; }
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
            all.iter().map(|effect| (effect.left_action_id, effect.right_action_id)).collect::<Vec<_>>(),
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
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::JointDrawBankDigestMismatch));
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
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::DuplicateActionId));

        let mut bank = joint_bank_fixture();
        bank.joint_draw_ids[1] = bank.joint_draw_ids[0];
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::DuplicateJointDrawId));

        let mut bank = joint_bank_fixture();
        bank.outcome_samples[1].pop();
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::ActionSampleCardinalityMismatch));
    }

    #[test]
    fn joint_action_bank_requires_action_semantics_and_outcome_channel_identity() {
        let mut bank = joint_bank_fixture();
        bank.action_manifest_sha256 = "bad".into();
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::InvalidActionManifestDigest));

        let mut bank = joint_bank_fixture();
        bank.outcome_channel = "  ".into();
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::EmptyOutcomeChannel));
    }

    #[test]
    fn joint_action_bank_rejects_missing_or_invalid_provenance_and_horizon() {
        let mut bank = joint_bank_fixture();
        bank.history_sha256 = "not-a-digest".into();
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::InvalidHistoryDigest));

        let mut bank = joint_bank_fixture();
        bank.joint_draw_bank_sha256 = "B".repeat(64);
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::InvalidJointDrawDigest));

        let mut bank = joint_bank_fixture();
        bank.horizon_steps = 0;
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::InvalidHorizon));
    }

    #[test]
    fn joint_action_bank_rejects_nonfinite_draws_and_same_action_effect() {
        let mut bank = joint_bank_fixture();
        bank.outcome_samples[0][0] = f64::NAN;
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::NonFiniteDerivedMetric));

        let bank = joint_bank_fixture();
        assert_eq!(joint_action_effect_interval(&bank, 17, 17, 0.8), Err(IntervalError::SameActionRequested));
        assert_eq!(joint_action_effect_interval(&bank, 17, 999, 0.8), Err(IntervalError::InvalidSelectedIndex));
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
        let metrics = evaluate_simultaneous_sets(&episodes, 0.9)
            .unwrap()
            .unwrap();
        assert_eq!(metrics.episode_count, 2);
        assert_eq!(metrics.intervals_per_episode, 2);
        assert_eq!(metrics.empirical_simultaneous_coverage, 0.5);
        assert_eq!(metrics.mean_max_interval_width, 2.0);
        assert_eq!(metrics.absolute_coverage_error, 0.4);
    }

    #[test]
    fn simultaneous_coverage_rejects_empty_and_mismatched_action_banks() {
        assert_eq!(
            evaluate_simultaneous_sets(&[], 0.9).unwrap(),
            None
        );
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
        assert_eq!(selected_action_regret_upper_bound(&intervals, 0).unwrap(), 0.0);
        assert!(is_interval_dominant(&intervals, 0).unwrap());

        let uncertain = [
            interval(1.5, 2.0, 0.9),
            interval(0.0, 2.5, 0.9),
        ];
        assert_eq!(selected_action_regret_upper_bound(&uncertain, 0).unwrap(), 1.0);
        assert!(!is_interval_dominant(&uncertain, 0).unwrap());
    }


    #[test]
    fn selected_action_evaluator_scores_realized_regret_and_certificate_quality() {
        let episodes = vec![
            SelectionEpisode {
                intervals: vec![interval(2.0,4.0,0.9), interval(0.0,3.0,0.9)],
                realized_outcomes: vec![3.5,3.0],
                selected_index: 0,
            },
            SelectionEpisode {
                intervals: vec![interval(0.0,2.0,0.9), interval(3.0,4.0,0.9)],
                realized_outcomes: vec![1.5,3.5],
                selected_index: 0,
            },
            SelectionEpisode {
                intervals: vec![interval(3.0,4.0,0.9), interval(0.0,2.0,0.9)],
                realized_outcomes: vec![3.5,1.0],
                selected_index: 0,
            },
            SelectionEpisode {
                intervals: vec![interval(3.0,4.0,0.9), interval(0.0,2.0,0.9)],
                realized_outcomes: vec![1.0,4.0],
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
            intervals: vec![interval(0.0,1.0,0.9), interval(0.0,1.0,0.9)],
            realized_outcomes: vec![0.5],
            selected_index: 0,
        }];
        assert_eq!(evaluate_selected_action_sets(&bad_shape), Err(IntervalError::UnequalActionSetCardinality));
        let bad_index = [SelectionEpisode {
            intervals: vec![interval(0.0,1.0,0.9), interval(0.0,1.0,0.9)],
            realized_outcomes: vec![0.5,0.5],
            selected_index: 2,
        }];
        assert_eq!(evaluate_selected_action_sets(&bad_index), Err(IntervalError::InvalidSelectedIndex));
        let bad_truth = [SelectionEpisode {
            intervals: vec![interval(0.0,1.0,0.9), interval(0.0,1.0,0.9)],
            realized_outcomes: vec![0.5,f64::NAN],
            selected_index: 0,
        }];
        assert_eq!(evaluate_selected_action_sets(&bad_truth), Err(IntervalError::NonFiniteRealization));
    }


    #[test]
    fn robust_action_selection_certifies_only_a_unique_interval_dominant_action() {
        let intervals = [
            interval(3.0,4.0,0.9),
            interval(0.0,2.0,0.9),
            interval(1.0,2.5,0.9),
        ];
        assert_eq!(
            select_interval_dominant_action(&intervals).unwrap(),
            RobustActionDecision::Certified { action_index: 0 }
        );

        let overlapping = [
            interval(1.0,3.0,0.9),
            interval(2.0,4.0,0.9),
        ];
        assert_eq!(
            select_interval_dominant_action(&overlapping).unwrap(),
            RobustActionDecision::Abstain { reason: AbstainReason::NoIntervalDominantAction }
        );

        let boundary_tie = [
            interval(2.0,2.0,0.9),
            interval(2.0,2.0,0.9),
        ];
        assert_eq!(
            select_interval_dominant_action(&boundary_tie).unwrap(),
            RobustActionDecision::Abstain { reason: AbstainReason::AmbiguousDominantActions }
        );
    }

    #[test]
    fn robust_action_selection_rejects_mixed_levels_and_too_small_sets() {
        assert_eq!(
            select_interval_dominant_action(&[interval(0.0,1.0,0.9)]),
            Err(IntervalError::InsufficientActions)
        );
        assert_eq!(
            select_interval_dominant_action(&[
                interval(2.0,3.0,0.9),
                interval(0.0,1.0,0.8),
            ]),
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
