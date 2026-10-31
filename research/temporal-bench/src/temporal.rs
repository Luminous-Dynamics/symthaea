//! Deterministic multi-timescale forecasting and delayed-feedback controls.
//!
//! The task stream exposes noisy observations and delayed online feedback.
//! The oracle contains the hidden periods, phases, regime boundary, and all
//! target labels. This module intentionally returns both artifacts for local
//! prototype testing; it is NOT a sealed evaluator boundary.

use super::{digest::CanonicalHasher, mix64, SplitMix64};
use std::fmt;

const MAX_STEPS: usize = 100_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultiTimescaleConfig {
    pub seed: u64,
    pub steps: usize,
    pub forecast_horizon: u64,
    pub feedback_delay: u64,
    pub regime_shift: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TemporalEventKind {
    Observation {
        fast: i16,
        medium: i16,
        slow: i16,
    },
    Feedback {
        source_step: u64,
        label: i8,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalEvent {
    pub step: u64,
    /// Observations are ordinal 0; same-time delayed feedback is ordinal 1.
    pub ordinal: u8,
    pub kind: TemporalEventKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiTimescaleTask {
    pub stream_id: u64,
    pub forecast_horizon: u64,
    pub feedback_delay: u64,
    pub events: Vec<TemporalEvent>,
    /// SHA-256 over the versioned canonical public multiscale stream.
    pub content_sha256: [u8; 32],
}

/// Domain-separated canonical SHA-256 of a public multiscale stream.
pub fn multiscale_content_digest(task: &MultiTimescaleTask) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.multiscale.public.v1");
    hash.u64(task.stream_id);
    hash.u64(task.forecast_horizon);
    hash.u64(task.feedback_delay);
    hash.usize(task.events.len());
    for event in &task.events {
        hash.u64(event.step);
        hash.u8(event.ordinal);
        match &event.kind {
            TemporalEventKind::Observation { fast, medium, slow } => {
                hash.u8(0);
                hash.u16(*fast as u16);
                hash.u16(*medium as u16);
                hash.u16(*slow as u16);
            }
            TemporalEventKind::Feedback { source_step, label } => {
                hash.u8(1);
                hash.u64(*source_step);
                hash.u8(*label as u8);
            }
        }
    }
    hash.finish()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalTarget {
    pub source_step: u64,
    pub target_step: u64,
    pub label: i8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiTimescaleOracle {
    pub stream_id: u64,
    pub seed: u64,
    pub targets: Vec<TemporalTarget>,
    pub hidden_change_step: Option<u64>,
    pub hidden_periods: [u64; 3],
    pub hidden_phases: [u64; 3],
    pub weights_before: [i8; 3],
    pub weights_after: [i8; 3],
    /// SHA-256 over the versioned canonical oracle record.
    pub content_sha256: [u8; 32],
}

/// Domain-separated canonical SHA-256 of the multiscale evaluator oracle.
pub fn multiscale_oracle_digest(oracle: &MultiTimescaleOracle) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.multiscale.oracle.v1");
    hash.u64(oracle.stream_id);
    hash.u64(oracle.seed);
    hash.usize(oracle.targets.len());
    for target in &oracle.targets {
        hash.u64(target.source_step);
        hash.u64(target.target_step);
        hash.u8(target.label as u8);
    }
    match oracle.hidden_change_step {
        Some(step) => {
            hash.u8(1);
            hash.u64(step);
        }
        None => hash.u8(0),
    }
    for period in oracle.hidden_periods {
        hash.u64(period);
    }
    for phase in oracle.hidden_phases {
        hash.u64(phase);
    }
    for weight in oracle.weights_before {
        hash.u8(weight as u8);
    }
    for weight in oracle.weights_after {
        hash.u8(weight as u8);
    }
    hash.finish()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TemporalVerificationError {
    StreamIdMismatch,
    ContentDigestMismatch,
    OracleDigestMismatch,
    InvalidOracleModel,
    OracleLengthMismatch,
    InvalidOracleTarget { source_step: u64 },
    InvalidObservationValue { step: u64 },
    MissingObservation { step: u64 },
    InvalidObservationOrder { step: u64 },
    MissingFeedback { source_step: u64 },
    InvalidFeedback { source_step: u64 },
    UnexpectedTrailingEvent,
}

impl fmt::Display for TemporalVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamIdMismatch => write!(f, "public task and oracle stream IDs differ"),
            Self::ContentDigestMismatch => write!(
                f,
                "multiscale public task SHA-256 digest does not match its contents"
            ),
            Self::OracleDigestMismatch => write!(
                f,
                "multiscale oracle SHA-256 digest does not match its contents"
            ),
            Self::InvalidOracleModel => write!(
                f,
                "multiscale oracle parameters are outside the generator contract"
            ),
            Self::OracleLengthMismatch => {
                write!(f, "oracle target count differs from observation count")
            }
            Self::InvalidOracleTarget { source_step } => {
                write!(f, "oracle target record is invalid at source step {source_step}")
            }
            Self::InvalidObservationValue { step } => {
                write!(f, "multiscale observation exceeds its declared noise bound at step {step}")
            }
            Self::MissingObservation { step } => {
                write!(f, "missing observation event at step {step}")
            }
            Self::InvalidObservationOrder { step } => {
                write!(f, "observation event has invalid time or ordinal at step {step}")
            }
            Self::MissingFeedback { source_step } => {
                write!(f, "missing delayed feedback for source step {source_step}")
            }
            Self::InvalidFeedback { source_step } => {
                write!(
                    f,
                    "delayed feedback/order/label error at source step {source_step}"
                )
            }
            Self::UnexpectedTrailingEvent => {
                write!(f, "public task has unexpected trailing events")
            }
        }
    }
}

impl std::error::Error for TemporalVerificationError {}

/// Verify the public event trace against the evaluator-owned oracle.
///
/// Run this only inside the evaluator process, never in the candidate model's
/// execution process. The function checks stream identity, complete ordered
/// observations, target-horizon binding, feedback timing, feedback labels,
/// and absence of extra events. It does not itself create process isolation.
pub fn verify_multiscale_trace(
    task: &MultiTimescaleTask,
    oracle: &MultiTimescaleOracle,
) -> Result<(), TemporalVerificationError> {
    if task.stream_id != oracle.stream_id {
        return Err(TemporalVerificationError::StreamIdMismatch);
    }
    if task.content_sha256 != multiscale_content_digest(task) {
        return Err(TemporalVerificationError::ContentDigestMismatch);
    }
    if oracle.content_sha256 != multiscale_oracle_digest(oracle) {
        return Err(TemporalVerificationError::OracleDigestMismatch);
    }

    let target_count = u64::try_from(oracle.targets.len()).unwrap_or(u64::MAX);
    if task.forecast_horizon == 0
        || task.feedback_delay == 0
        || oracle.hidden_periods[0] < 2
        || oracle.hidden_periods[0] > 8
        || oracle.hidden_periods[1] < 9
        || oracle.hidden_periods[1] > 24
        || oracle.hidden_periods[2] < 25
        || oracle.hidden_periods[2] > 64
        || oracle
            .hidden_phases
            .iter()
            .zip(oracle.hidden_periods)
            .any(|(phase, period)| *phase >= period)
        || oracle
            .weights_before
            .iter()
            .chain(oracle.weights_after.iter())
            .any(|weight| *weight == 0 || !(-4..=4).contains(weight))
    {
        return Err(TemporalVerificationError::InvalidOracleModel);
    }
    let expected_after = [
        oracle.weights_before[2],
        -oracle.weights_before[1],
        oracle.weights_before[0],
    ];
    match oracle.hidden_change_step {
        Some(change)
            if change < target_count / 3
                || change > target_count.saturating_mul(2) / 3
                || oracle.weights_after != expected_after =>
        {
            return Err(TemporalVerificationError::InvalidOracleModel);
        }
        None if oracle.weights_after != oracle.weights_before => {
            return Err(TemporalVerificationError::InvalidOracleModel);
        }
        _ => {}
    }
    if target_count != oracle.targets.len() as u64 {
        return Err(TemporalVerificationError::OracleLengthMismatch);
    }

    for (index, target) in oracle.targets.iter().enumerate() {
        let source_step = u64::try_from(index).unwrap_or(u64::MAX);
        if target.source_step != source_step
            || target.target_step != source_step.saturating_add(task.forecast_horizon)
            || !(-1..=1).contains(&target.label)
            || target.label == 0
            || target.label
                != target_label(
                    target.target_step,
                    oracle.hidden_periods,
                    oracle.hidden_phases,
                    weights_for_step(
                        target.target_step,
                        oracle.hidden_change_step,
                        oracle.weights_before,
                        oracle.weights_after,
                    ),
                )
        {
            return Err(TemporalVerificationError::InvalidOracleTarget { source_step });
        }
    }

    let mut cursor = 0_usize;
    for step in 0..target_count {
        let observation = task.events.get(cursor).ok_or(
            TemporalVerificationError::MissingObservation { step },
        )?;
        if observation.step != step
            || observation.ordinal != 0
            || !matches!(&observation.kind, TemporalEventKind::Observation { .. })
        {
            return Err(TemporalVerificationError::InvalidObservationOrder { step });
        }
        if let TemporalEventKind::Observation { fast, medium, slow } = &observation.kind {
            let expected = [
                wave(step, oracle.hidden_periods[0], oracle.hidden_phases[0]),
                wave(step, oracle.hidden_periods[1], oracle.hidden_phases[1]),
                wave(step, oracle.hidden_periods[2], oracle.hidden_phases[2]),
            ];
            let actual = [*fast, *medium, *slow];
            if actual.into_iter().zip(expected).any(|(observed, latent)| {
                (i32::from(observed) - i32::from(latent)).abs() > 20
            }) {
                return Err(TemporalVerificationError::InvalidObservationValue { step });
            }
        }
        cursor = cursor.saturating_add(1);

        if step >= task.feedback_delay {
            let source_step = step - task.feedback_delay;
            let feedback = task.events.get(cursor).ok_or(
                TemporalVerificationError::MissingFeedback { source_step },
            )?;
            let expected_label = oracle.targets[
                usize::try_from(source_step).unwrap_or(usize::MAX)
            ].label;
            let TemporalEventKind::Feedback {
                source_step: actual_source,
                label,
            } = &feedback.kind
            else {
                return Err(TemporalVerificationError::MissingFeedback { source_step });
            };
            if feedback.step != step
                || feedback.ordinal != 1
                || *actual_source != source_step
                || *label != expected_label
            {
                return Err(TemporalVerificationError::InvalidFeedback { source_step });
            }
            cursor = cursor.saturating_add(1);
        }
    }

    if cursor != task.events.len() {
        return Err(TemporalVerificationError::UnexpectedTrailingEvent);
    }

    // Rebuild from the evaluator seed and public timing metadata. This catches
    // semantically plausible but recommitted mutations to noisy observations,
    // latent parameters, or delayed labels. It is generator consistency, not
    // a substitute for process isolation or a protected oracle commitment.
    let steps = usize::try_from(target_count).unwrap_or(usize::MAX);
    let (expected_task, expected_oracle) = generate_multiscale_task(MultiTimescaleConfig {
        seed: oracle.seed,
        steps,
        forecast_horizon: task.forecast_horizon,
        feedback_delay: task.feedback_delay,
        regime_shift: oracle.hidden_change_step.is_some(),
    })
    .map_err(|_| TemporalVerificationError::InvalidOracleModel)?;
    if expected_task != *task || expected_oracle != *oracle {
        return Err(TemporalVerificationError::InvalidOracleModel);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TemporalGenerationError {
    ZeroSteps,
    TooManySteps { requested: usize, maximum: usize },
    TooFewStepsForRegimeShift { requested: usize, minimum: usize },
    ZeroForecastHorizon,
    ZeroFeedbackDelay,
    FeedbackDelayOutsideStream { delay: u64, steps: usize },
}

impl fmt::Display for TemporalGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSteps => write!(f, "steps must be greater than zero"),
            Self::TooManySteps { requested, maximum } => {
                write!(f, "steps {requested} exceeds the maximum {maximum}")
            }
            Self::TooFewStepsForRegimeShift { requested, minimum } => write!(
                f,
                "regime-shift tasks require at least {minimum} steps, received {requested}"
            ),
            Self::ZeroForecastHorizon => {
                write!(f, "forecast_horizon must be greater than zero")
            }
            Self::ZeroFeedbackDelay => {
                write!(f, "feedback_delay must be greater than zero")
            }
            Self::FeedbackDelayOutsideStream { delay, steps } => {
                write!(f, "feedback_delay {delay} must be less than stream steps {steps}")
            }
        }
    }
}

impl std::error::Error for TemporalGenerationError {}

/// Generate a causal multiscale stream with delayed labels.
///
/// The public task exposes observations and feedback only. It does not expose
/// latent periods, phases, target labels before feedback, or the regime-change
/// time. The returned oracle is for evaluator-side validation only. Because
/// this function returns both records in one process, deployment must move
/// oracle handling into a separate process or permission boundary.
pub fn generate_multiscale_task(
    config: MultiTimescaleConfig,
) -> Result<(MultiTimescaleTask, MultiTimescaleOracle), TemporalGenerationError> {
    validate_config(config)?;

    let stream_id = multiscale_stream_id(config);
    let mut rng = SplitMix64::new(config.seed ^ 0x4D55_4C54_4953_4341);

    let periods = [
        2 + (rng.next_u64() % 7),
        9 + (rng.next_u64() % 16),
        25 + (rng.next_u64() % 40),
    ];
    let phases = [
        rng.next_u64() % periods[0],
        rng.next_u64() % periods[1],
        rng.next_u64() % periods[2],
    ];
    let weights_before = [
        random_weight(&mut rng),
        random_weight(&mut rng),
        random_weight(&mut rng),
    ];
    let weights_after = if config.regime_shift {
        [
            weights_before[2],
            -weights_before[1],
            weights_before[0],
        ]
    } else {
        weights_before
    };
    let hidden_change_step = if config.regime_shift {
        let low = u64::try_from(config.steps / 3).unwrap_or(u64::MAX);
        let high = u64::try_from((config.steps * 2) / 3).unwrap_or(u64::MAX);
        let width = high.saturating_sub(low).saturating_add(1);
        Some(low.saturating_add(rng.next_u64() % width))
    } else {
        None
    };

    let step_count = u64::try_from(config.steps).unwrap_or(u64::MAX);
    let mut targets = Vec::with_capacity(config.steps);
    for source_step in 0..step_count {
        let target_step = source_step.saturating_add(config.forecast_horizon);
        let weights = weights_for_step(
            target_step,
            hidden_change_step,
            weights_before,
            weights_after,
        );
        let label = target_label(target_step, periods, phases, weights);
        targets.push(TemporalTarget {
            source_step,
            target_step,
            label,
        });
    }

    let mut events = Vec::with_capacity(config.steps.saturating_mul(2));
    for step in 0..step_count {
        let fast = noisy_wave(step, periods[0], phases[0], &mut rng);
        let medium = noisy_wave(step, periods[1], phases[1], &mut rng);
        let slow = noisy_wave(step, periods[2], phases[2], &mut rng);

        events.push(TemporalEvent {
            step,
            ordinal: 0,
            kind: TemporalEventKind::Observation {
                fast,
                medium,
                slow,
            },
        });

        if step >= config.feedback_delay {
            let source_step = step - config.feedback_delay;
            let index = usize::try_from(source_step).unwrap_or(usize::MAX);
            let label = targets[index].label;
            events.push(TemporalEvent {
                step,
                ordinal: 1,
                kind: TemporalEventKind::Feedback { source_step, label },
            });
        }
    }

    let mut public_task = MultiTimescaleTask {
        stream_id,
        forecast_horizon: config.forecast_horizon,
        feedback_delay: config.feedback_delay,
        events,
        content_sha256: [0; 32],
    };
    public_task.content_sha256 = multiscale_content_digest(&public_task);

    Ok((
        public_task,
        {
            let mut oracle = MultiTimescaleOracle {
                stream_id,
                seed: config.seed,
                targets,
                hidden_change_step,
                hidden_periods: periods,
                hidden_phases: phases,
                weights_before,
                weights_after,
                content_sha256: [0; 32],
            };
            oracle.content_sha256 = multiscale_oracle_digest(&oracle);
            oracle
        },
    ))
}

fn validate_config(config: MultiTimescaleConfig) -> Result<(), TemporalGenerationError> {
    if config.steps == 0 {
        return Err(TemporalGenerationError::ZeroSteps);
    }
    if config.steps > MAX_STEPS {
        return Err(TemporalGenerationError::TooManySteps {
            requested: config.steps,
            maximum: MAX_STEPS,
        });
    }
    if config.regime_shift && config.steps < 3 {
        return Err(TemporalGenerationError::TooFewStepsForRegimeShift {
            requested: config.steps,
            minimum: 3,
        });
    }
    if config.forecast_horizon == 0 {
        return Err(TemporalGenerationError::ZeroForecastHorizon);
    }
    if config.feedback_delay == 0 {
        return Err(TemporalGenerationError::ZeroFeedbackDelay);
    }
    if config.feedback_delay >= u64::try_from(config.steps).unwrap_or(u64::MAX) {
        return Err(TemporalGenerationError::FeedbackDelayOutsideStream {
            delay: config.feedback_delay,
            steps: config.steps,
        });
    }
    Ok(())
}

fn multiscale_stream_id(config: MultiTimescaleConfig) -> u64 {
    let mut value = config.seed ^ (config.steps as u64).rotate_left(11);
    value ^= config.forecast_horizon.rotate_left(23);
    value ^= config.feedback_delay.rotate_left(37);
    value ^= if config.regime_shift {
        0x5245_4749_4D45_5348
    } else {
        0x5354_4154_4943_0000
    };
    mix64(value)
}

fn random_weight(rng: &mut SplitMix64) -> i8 {
    let magnitude = i8::try_from(1 + (rng.next_u64() % 4)).unwrap_or(1);
    if rng.next_u64() & 1 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn weights_for_step(
    target_step: u64,
    change_step: Option<u64>,
    before: [i8; 3],
    after: [i8; 3],
) -> [i8; 3] {
    match change_step {
        Some(change) if target_step >= change => after,
        _ => before,
    }
}

fn target_label(
    step: u64,
    periods: [u64; 3],
    phases: [u64; 3],
    weights: [i8; 3],
) -> i8 {
    let fast = i64::from(wave(step, periods[0], phases[0]));
    let medium = i64::from(wave(step, periods[1], phases[1]));
    let slow = i64::from(wave(step, periods[2], phases[2]));
    let weighted = fast * i64::from(weights[0])
        + medium * i64::from(weights[1])
        + slow * i64::from(weights[2]);
    if weighted >= 0 { 1 } else { -1 }
}

fn wave(step: u64, period: u64, phase: u64) -> i16 {
    let position = step.wrapping_add(phase) % period;
    let scaled = (u128::from(position) * 2_000) / u128::from(period);
    i16::try_from(i32::try_from(scaled).unwrap_or(i32::MAX) - 1_000).unwrap_or(0)
}

fn noisy_wave(
    step: u64,
    period: u64,
    phase: u64,
    rng: &mut SplitMix64,
) -> i16 {
    let noise = i16::try_from(rng.next_u64() % 41).unwrap_or(0) - 20;
    wave(step, period, phase).saturating_add(noise).clamp(-1_000, 1_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(regime_shift: bool) -> MultiTimescaleConfig {
        MultiTimescaleConfig {
            seed: 0x5EED_2026_1031,
            steps: 512,
            forecast_horizon: 7,
            feedback_delay: 3,
            regime_shift,
        }
    }

    #[test]
    fn generation_is_deterministic_for_fixed_seed() {
        let first = generate_multiscale_task(config(true)).unwrap();
        let second = generate_multiscale_task(config(true)).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn periods_are_in_preregistered_fast_medium_slow_bands() {
        let (_, oracle) = generate_multiscale_task(config(false)).unwrap();
        assert!((2..=8).contains(&oracle.hidden_periods[0]));
        assert!((9..=24).contains(&oracle.hidden_periods[1]));
        assert!((25..=64).contains(&oracle.hidden_periods[2]));
    }

    #[test]
    fn regime_change_is_hidden_and_bounded_to_middle_third_band() {
        let (task, oracle) = generate_multiscale_task(config(true)).unwrap();
        let change = oracle.hidden_change_step.unwrap();
        assert!(change < 512);
        assert!((512 / 3..=(512 * 2) / 3).contains(&(change as usize)));
        assert_ne!(oracle.weights_before, oracle.weights_after);
    }

    #[test]
    fn static_task_has_no_change_point() {
        let (_, oracle) = generate_multiscale_task(config(false)).unwrap();
        assert_eq!(oracle.hidden_change_step, None);
        assert_eq!(oracle.weights_before, oracle.weights_after);
    }

    #[test]
    fn feedback_arrives_after_declared_delay_and_matches_oracle() {
        let (task, oracle) = generate_multiscale_task(config(true)).unwrap();
        for event in &task.events {
            if let TemporalEventKind::Feedback { source_step, label } = &event.kind {
                assert_eq!(event.step - *source_step, task.feedback_delay);
                assert_eq!(
                    oracle.targets[*source_step as usize].label,
                    *label
                );
            }
        }
    }

    #[test]
    fn observations_precede_same_time_feedback() {
        let (task, _) = generate_multiscale_task(config(true)).unwrap();
        for pair in task.events.windows(2) {
            if pair[0].step == pair[1].step {
                assert!(pair[0].ordinal < pair[1].ordinal);
            }
        }
    }

    #[test]
    fn oracle_forecast_horizon_matches_configuration() {
        let (_, oracle) = generate_multiscale_task(config(false)).unwrap();
        for target in oracle.targets {
            assert_eq!(target.target_step - target.source_step, 7);
        }
    }

    #[test]
    fn generated_trace_passes_independent_oracle_verifier() {
        let (task, oracle) = generate_multiscale_task(config(true)).unwrap();
        assert_eq!(verify_multiscale_trace(&task, &oracle), Ok(()));
    }

    #[test]
    fn generated_oracle_has_valid_sha256() {
        let (_, oracle) = generate_multiscale_task(config(true)).unwrap();
        assert_eq!(oracle.content_sha256, multiscale_oracle_digest(&oracle));
    }

    #[test]
    fn verifier_recomputes_labels_from_hidden_periods_and_weights() {
        let (task, mut oracle) = generate_multiscale_task(config(true)).unwrap();
        oracle.hidden_periods[0] = 1_000;
        // Recommit the modified oracle digest; structural validation must
        // still reject an out-of-contract latent model.
        oracle.content_sha256 = multiscale_oracle_digest(&oracle);
        assert_eq!(
            verify_multiscale_trace(&task, &oracle),
            Err(TemporalVerificationError::InvalidOracleModel)
        );
    }

    #[test]
    fn verifier_rejects_target_label_inconsistent_with_recommitted_oracle() {
        let (task, mut oracle) = generate_multiscale_task(config(true)).unwrap();
        oracle.targets[0].label = -oracle.targets[0].label;
        oracle.content_sha256 = multiscale_oracle_digest(&oracle);
        assert_eq!(
            verify_multiscale_trace(&task, &oracle),
            Err(TemporalVerificationError::InvalidOracleTarget { source_step: 0 })
        );
    }

    #[test]
    fn verifier_rejects_oracle_target_mutation() {
        let (task, mut oracle) = generate_multiscale_task(config(true)).unwrap();
        oracle.targets[0].label = -oracle.targets[0].label;
        assert_eq!(
            verify_multiscale_trace(&task, &oracle),
            Err(TemporalVerificationError::OracleDigestMismatch)
        );
    }

    #[test]
    fn generated_public_stream_has_valid_sha256() {
        let (task, _) = generate_multiscale_task(config(true)).unwrap();
        assert_eq!(task.content_sha256, multiscale_content_digest(&task));
    }

    #[test]
    fn verifier_rejects_recommitted_observation_outside_noise_bound() {
        let (mut task, oracle) = generate_multiscale_task(config(true)).unwrap();
        if let TemporalEventKind::Observation { fast, .. } = &mut task.events[0].kind {
            *fast = 1_000;
        }
        task.content_sha256 = multiscale_content_digest(&task);
        assert_eq!(
            verify_multiscale_trace(&task, &oracle),
            Err(TemporalVerificationError::InvalidObservationValue { step: 0 })
        );
    }

    #[test]
    fn verifier_rejects_recommitted_seed_inconsistent_observation() {
        let (mut task, oracle) = generate_multiscale_task(config(true)).unwrap();
        if let Some(event) = task.events.iter_mut().find(|event| {
            matches!(&event.kind, TemporalEventKind::Observation { .. })
        }) {
            if let TemporalEventKind::Observation { fast, .. } = &mut event.kind {
                *fast = fast.saturating_add(1);
            }
        }
        task.content_sha256 = multiscale_content_digest(&task);
        assert_eq!(
            verify_multiscale_trace(&task, &oracle),
            Err(TemporalVerificationError::InvalidOracleModel)
        );
    }

    #[test]
    fn verifier_rejects_post_generation_observation_mutation() {
        let (mut task, oracle) = generate_multiscale_task(config(true)).unwrap();
        task.events[0].step = task.events[0].step.saturating_add(1);
        assert_eq!(
            verify_multiscale_trace(&task, &oracle),
            Err(TemporalVerificationError::ContentDigestMismatch)
        );
    }

    #[test]
    fn verifier_rejects_mutated_feedback_label() {
        let (mut task, oracle) = generate_multiscale_task(config(true)).unwrap();
        let event = task
            .events
            .iter_mut()
            .find(|event| matches!(&event.kind, TemporalEventKind::Feedback { .. }))
            .unwrap();
        if let TemporalEventKind::Feedback { label, .. } = &mut event.kind {
            *label = -*label;
        }
        task.content_sha256 = multiscale_content_digest(&task);
        assert!(matches!(
            verify_multiscale_trace(&task, &oracle),
            Err(TemporalVerificationError::InvalidFeedback { .. })
        ));
    }

    #[test]
    fn verifier_rejects_early_feedback() {
        let (mut task, oracle) = generate_multiscale_task(config(true)).unwrap();
        let event = task
            .events
            .iter_mut()
            .find(|event| matches!(&event.kind, TemporalEventKind::Feedback { .. }))
            .unwrap();
        event.step = event.step.saturating_sub(1);
        task.content_sha256 = multiscale_content_digest(&task);
        assert!(matches!(
            verify_multiscale_trace(&task, &oracle),
            Err(TemporalVerificationError::InvalidFeedback { .. })
        ));
    }

    #[test]
    fn verifier_rejects_missing_feedback() {
        let (mut task, oracle) = generate_multiscale_task(config(true)).unwrap();
        let index = task
            .events
            .iter()
            .position(|event| matches!(&event.kind, TemporalEventKind::Feedback { .. }))
            .unwrap();
        task.events.remove(index);
        task.content_sha256 = multiscale_content_digest(&task);
        assert!(matches!(
            verify_multiscale_trace(&task, &oracle),
            Err(TemporalVerificationError::MissingFeedback { .. })
        ));
    }

    #[test]
    fn verifier_rejects_oracle_mismatch() {
        let (task, mut oracle) = generate_multiscale_task(config(true)).unwrap();
        oracle.stream_id ^= 1;
        assert_eq!(
            verify_multiscale_trace(&task, &oracle),
            Err(TemporalVerificationError::StreamIdMismatch)
        );
    }

    #[test]
    fn zero_feedback_delay_is_rejected() {
        let mut cfg = config(false);
        cfg.feedback_delay = 0;
        assert_eq!(
            generate_multiscale_task(cfg),
            Err(TemporalGenerationError::ZeroFeedbackDelay)
        );
    }

    #[test]
    fn feedback_delay_must_arrive_within_stream() {
        let mut cfg = config(false);
        cfg.feedback_delay = cfg.steps as u64;
        assert_eq!(
            generate_multiscale_task(cfg),
            Err(TemporalGenerationError::FeedbackDelayOutsideStream {
                delay: cfg.steps as u64,
                steps: cfg.steps,
            })
        );
    }

    #[test]
    fn regime_shift_rejects_too_short_stream() {
        let mut cfg = config(true);
        cfg.steps = 2;
        assert_eq!(
            generate_multiscale_task(cfg),
            Err(TemporalGenerationError::TooFewStepsForRegimeShift {
                requested: 2,
                minimum: 3
            })
        );
    }
}
