// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Prequential probability-integral-transform (PIT) martingale for adaptive experiments.
//!
//! Each PIT value must be computed from a predictive CDF issued before the current outcome,
//! conditional on the chosen experimental action and all permitted past information. Under the
//! null that this conditional predictive CDF is correct, the PIT is Uniform(0,1) conditional on
//! the past and on the action. This enables an anytime-valid test even when actions are selected
//! adaptively, provided they are predictable and the conditional-PIT premise is valid.
//!
//! The implementation forms a fixed-weight portfolio of order and dispersion betting processes.
//! Optional context channels multiply each feature by a predeclared, pre-outcome value in [-1,1].
//! Context-conditioned features can detect errors that cancel in the pooled PIT stream. The
//! channel set and weighting are fixed at construction; the values supplied at each observation
//! must be measurable before that observation arrives.
//!
//! This is a model-based null, unlike the marginal distribution-free rank martingale in
//! predictive_rank_martingale. A plug-in CDF, a CDF fit or tuned on the current outcome, an
//! unaccounted-for parameter estimate, or a misspecified predictive distribution need not yield
//! conditionally uniform PIT values. The code validates numeric inputs but cannot prove that the
//! caller's predictive CDF is correct or that context values were fixed before the outcome.

/// Errors in the prequential PIT detector configuration, input protocol, or arithmetic.
#[derive(Clone, Debug, PartialEq)]
pub enum PrequentialPitError {
    /// The significance level must be finite and strictly between zero and one.
    InvalidSignificanceLevel,
    /// The PIT lies outside [0,1] or is not finite.
    InvalidPitValue,
    /// This observation's context vector has the wrong number of predeclared features.
    ContextFeatureCountMismatch {
        /// Number of context features fixed when the monitor was constructed.
        expected: usize,
        /// Number of context features supplied for this observation.
        actual: usize,
    },
    /// A context feature is not finite or lies outside [-1,1].
    InvalidContextFeature {
        /// Zero-based index in the supplied context-feature vector.
        index: usize,
    },
    /// The detector has alarmed and cannot be continued in the same epoch.
    AlreadyAlarmed,
    /// An intermediate computation produced a non-finite or invalid value.
    NumericalFailure,
    /// The observation counter would overflow.
    ObservationCounterOverflow,
}

/// Terminal state of one prequential PIT monitoring epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrequentialPitState {
    /// The monitor can accept another PIT and its predeclared context features.
    Monitoring,
    /// Portfolio wealth reached 1/alpha; the epoch is terminal.
    Alarmed,
}

/// Decision for the latest PIT value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrequentialPitDecision {
    /// The portfolio has not crossed the alarm threshold.
    Continue,
    /// The portfolio crossed 1/alpha.
    Alarm,
}

/// Rank-scale feature tracked by a prequential PIT betting process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PitFeatureKind {
    /// Directional/location feature h(u) = u - 1/2.
    Order,
    /// Center-versus-tail feature h(u) = |2u - 1| - 1/2.
    Dispersion,
}

/// Observable wealth for one fixed portfolio component.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PitFeatureWealth {
    /// None means the unconditional intercept channel; Some(i) identifies caller channel i.
    pub context_feature_index: Option<usize>,
    /// Feature tracked by this component.
    pub kind: PitFeatureKind,
    /// Natural logarithm of this component's e-process wealth.
    pub log_wealth: f64,
}

/// Result for a single observation's PIT.
#[derive(Clone, Debug, PartialEq)]
pub struct PrequentialPitResult {
    /// One-based index within the current monitoring epoch.
    pub observation_index: usize,
    /// The supplied PIT, including randomized PITs for discrete outcomes when required.
    pub pit_value: f64,
    /// Natural log wealth of the fixed-weight portfolio of all components.
    pub portfolio_log_wealth: f64,
    /// Natural log of the alarm threshold, -ln(alpha).
    pub log_alarm_threshold: f64,
    /// Policy decision after this PIT.
    pub decision: PrequentialPitDecision,
    /// Per-component log wealth for diagnostics and auditable decision receipts.
    pub feature_wealths: Vec<PitFeatureWealth>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BettingState {
    lambda: f64,
    log_wealth: f64,
    squared_gradient_sum: f64,
}

impl BettingState {
    fn initial() -> Self {
        Self {
            lambda: 0.0,
            log_wealth: 0.0,
            squared_gradient_sum: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ChannelState {
    order: BettingState,
    dispersion: BettingState,
}

impl ChannelState {
    fn initial() -> Self {
        Self {
            order: BettingState::initial(),
            dispersion: BettingState::initial(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FeatureObservation {
    log_wealth: f64,
    gradient: f64,
    squared_gradient_sum: f64,
}

/// Anytime-valid PIT monitor for fixed conditional-predictive-distribution hypotheses.
///
/// Channel zero is an implicit intercept (weight 1). The caller declares the number of context
/// channels at construction; each later observation must provide exactly that many values. Every
/// context value must be in [-1,1] and chosen using information available before the current
/// outcome. The fixed-weight portfolio has 2 * (1 + context_feature_count) components.
#[derive(Clone, Debug, PartialEq)]
pub struct PrequentialPitMartingale {
    channels: Vec<ChannelState>,
    context_feature_count: usize,
    alpha: f64,
    log_alarm_threshold: f64,
    observations_seen: usize,
    state: PrequentialPitState,
}

impl PrequentialPitMartingale {
    /// Construct a monitor with alpha and a fixed number of predictable context channels.
    ///
    /// The total false-alarm threshold is 1/alpha for the equal-weight portfolio. Pass zero
    /// context features for a marginal PIT test; add predeclared bounded context channels to test
    /// calibration conditional on known experimental choices or other pre-outcome covariates.
    pub fn new(
        alpha: f64,
        context_feature_count: usize,
    ) -> Result<Self, PrequentialPitError> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(PrequentialPitError::InvalidSignificanceLevel);
        }
        let channel_count = context_feature_count
            .checked_add(1)
            .ok_or(PrequentialPitError::ObservationCounterOverflow)?;
        Ok(Self {
            channels: vec![ChannelState::initial(); channel_count],
            context_feature_count,
            alpha,
            log_alarm_threshold: -alpha.ln(),
            observations_seen: 0,
            state: PrequentialPitState::Monitoring,
        })
    }

    /// Consume one prequential PIT and its context features.
    ///
    /// The forecast must have been issued before observing the outcome, conditional on the
    /// experiment chosen for that observation and the past. If the observation is discrete, the
    /// caller must compute a randomized PIT using a fresh independent Uniform(0,1) tie breaker.
    /// Numeric checks cannot establish these protocol conditions.
    pub fn observe(
        &mut self,
        pit_value: f64,
        context_features: &[f64],
    ) -> Result<PrequentialPitResult, PrequentialPitError> {
        if self.state == PrequentialPitState::Alarmed {
            return Err(PrequentialPitError::AlreadyAlarmed);
        }
        if !pit_value.is_finite() || !(0.0..=1.0).contains(&pit_value) {
            return Err(PrequentialPitError::InvalidPitValue);
        }
        if context_features.len() != self.context_feature_count {
            return Err(PrequentialPitError::ContextFeatureCountMismatch {
                expected: self.context_feature_count,
                actual: context_features.len(),
            });
        }
        for (index, value) in context_features.iter().copied().enumerate() {
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err(PrequentialPitError::InvalidContextFeature { index });
            }
        }
        let observation_index = self
            .observations_seen
            .checked_add(1)
            .ok_or(PrequentialPitError::ObservationCounterOverflow)?;

        // The intercept channel is fixed at one. Caller-supplied channels must be predictable:
        // context may depend on the chosen action and past, never on the current PIT/outcome.
        let weights = std::iter::once(1.0)
            .chain(context_features.iter().copied())
            .collect::<Vec<_>>();
        let mut order_observations = Vec::with_capacity(self.channels.len());
        let mut dispersion_observations = Vec::with_capacity(self.channels.len());
        let mut feature_wealths = Vec::with_capacity(2 * self.channels.len());

        let order_pit = pit_value - 0.5;
        let dispersion_pit = (2.0 * pit_value - 1.0).abs() - 0.5;
        for (channel_index, (channel, weight)) in
            self.channels.iter().zip(weights.iter().copied()).enumerate()
        {
            let order = observe_feature(channel.order, weight * order_pit)?;
            let dispersion = observe_feature(channel.dispersion, weight * dispersion_pit)?;
            let context_feature_index = if channel_index == 0 {
                None
            } else {
                Some(channel_index - 1)
            };
            feature_wealths.push(PitFeatureWealth {
                context_feature_index,
                kind: PitFeatureKind::Order,
                log_wealth: order.log_wealth,
            });
            feature_wealths.push(PitFeatureWealth {
                context_feature_index,
                kind: PitFeatureKind::Dispersion,
                log_wealth: dispersion.log_wealth,
            });
            order_observations.push(order);
            dispersion_observations.push(dispersion);
        }

        let component_log_wealths = feature_wealths
            .iter()
            .map(|snapshot| snapshot.log_wealth)
            .collect::<Vec<_>>();
        let portfolio_log_wealth = log_mean_exp(&component_log_wealths)?;
        let decision = if portfolio_log_wealth >= self.log_alarm_threshold {
            PrequentialPitDecision::Alarm
        } else {
            PrequentialPitDecision::Continue
        };
        let result = PrequentialPitResult {
            observation_index,
            pit_value,
            portfolio_log_wealth,
            log_alarm_threshold: self.log_alarm_threshold,
            decision,
            feature_wealths,
        };

        if decision == PrequentialPitDecision::Alarm {
            // Match test-by-betting order: evaluate threshold before the next ONS update.
            for (index, channel) in self.channels.iter_mut().enumerate() {
                channel.order.log_wealth = order_observations[index].log_wealth;
                channel.order.squared_gradient_sum =
                    order_observations[index].squared_gradient_sum;
                channel.dispersion.log_wealth = dispersion_observations[index].log_wealth;
                channel.dispersion.squared_gradient_sum =
                    dispersion_observations[index].squared_gradient_sum;
            }
            self.observations_seen = observation_index;
            self.state = PrequentialPitState::Alarmed;
            return Ok(result);
        }

        let mut next_channels = Vec::with_capacity(self.channels.len());
        for (index, channel) in self.channels.iter().copied().enumerate() {
            let next_order_lambda = next_lambda(channel.order.lambda, order_observations[index])?;
            let next_dispersion_lambda =
                next_lambda(channel.dispersion.lambda, dispersion_observations[index])?;
            next_channels.push(ChannelState {
                order: BettingState {
                    lambda: next_order_lambda,
                    log_wealth: order_observations[index].log_wealth,
                    squared_gradient_sum: order_observations[index].squared_gradient_sum,
                },
                dispersion: BettingState {
                    lambda: next_dispersion_lambda,
                    log_wealth: dispersion_observations[index].log_wealth,
                    squared_gradient_sum: dispersion_observations[index].squared_gradient_sum,
                },
            });
        }

        self.channels = next_channels;
        self.observations_seen = observation_index;
        Ok(result)
    }

    /// Number of context channels fixed at construction.
    pub fn context_feature_count(&self) -> usize {
        self.context_feature_count
    }

    /// Number of valid PIT observations consumed during this epoch.
    pub fn observations_seen(&self) -> usize {
        self.observations_seen
    }

    /// Significance level for the portfolio alarm threshold.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Natural-log alarm threshold, -ln(alpha).
    pub fn log_alarm_threshold(&self) -> f64 {
        self.log_alarm_threshold
    }

    /// Current terminal state.
    pub fn state(&self) -> PrequentialPitState {
        self.state
    }
}

fn observe_feature(
    state: BettingState,
    payoff: f64,
) -> Result<FeatureObservation, PrequentialPitError> {
    if !payoff.is_finite() || payoff.abs() > 0.5 + 1e-12 {
        return Err(PrequentialPitError::NumericalFailure);
    }
    // The contextual multiplier is bounded in [-1,1] and each PIT feature in [-1/2,1/2].
    // Therefore lambda in [-1,1] ensures every factor is at least 1/2.
    let factor = 1.0 + state.lambda * payoff;
    if !factor.is_finite() || factor <= 0.0 {
        return Err(PrequentialPitError::NumericalFailure);
    }
    let log_wealth = state.log_wealth + factor.ln();
    let gradient = payoff / factor;
    let squared_gradient_sum = state.squared_gradient_sum + gradient * gradient;
    if !log_wealth.is_finite()
        || !gradient.is_finite()
        || !squared_gradient_sum.is_finite()
    {
        return Err(PrequentialPitError::NumericalFailure);
    }
    Ok(FeatureObservation {
        log_wealth,
        gradient,
        squared_gradient_sum,
    })
}

fn next_lambda(
    current_lambda: f64,
    observation: FeatureObservation,
) -> Result<f64, PrequentialPitError> {
    let candidate = current_lambda
        + 4.5 * observation.gradient / (1.0 + observation.squared_gradient_sum);
    if !candidate.is_finite() {
        return Err(PrequentialPitError::NumericalFailure);
    }
    Ok(candidate.clamp(-1.0, 1.0))
}

fn log_mean_exp(log_wealths: &[f64]) -> Result<f64, PrequentialPitError> {
    if log_wealths.is_empty() || log_wealths.iter().any(|wealth| !wealth.is_finite()) {
        return Err(PrequentialPitError::NumericalFailure);
    }
    let maximum = log_wealths.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total = log_wealths
        .iter()
        .map(|wealth| (*wealth - maximum).exp())
        .sum::<f64>();
    let result = maximum + total.ln() - (log_wealths.len() as f64).ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(PrequentialPitError::NumericalFailure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_unit_interval(state: &mut u64) -> f64 {
        // Deterministic test generator only; not suitable for production randomization.
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        let value = (*state).wrapping_mul(0x2545_F491_4F6C_DD1D);
        (value >> 11) as f64 / ((1_u64 << 53) as f64)
    }

    #[test]
    fn rejects_invalid_configuration_pit_and_context() {
        assert!(matches!(
            PrequentialPitMartingale::new(0.0, 0),
            Err(PrequentialPitError::InvalidSignificanceLevel)
        ));
        assert!(matches!(
            PrequentialPitMartingale::new(f64::INFINITY, 0),
            Err(PrequentialPitError::InvalidSignificanceLevel)
        ));

        let mut monitor = PrequentialPitMartingale::new(0.05, 1).unwrap();
        assert!(matches!(
            monitor.observe(f64::NAN, &[0.0]),
            Err(PrequentialPitError::InvalidPitValue)
        ));
        assert!(matches!(
            monitor.observe(0.5, &[]),
            Err(PrequentialPitError::ContextFeatureCountMismatch { .. })
        ));
        assert!(matches!(
            monitor.observe(0.5, &[1.01]),
            Err(PrequentialPitError::InvalidContextFeature { index: 0 })
        ));
        assert_eq!(monitor.observations_seen(), 0);
        assert_eq!(monitor.state(), PrequentialPitState::Monitoring);
    }

    fn standard_normal_sample(state: &mut u64) -> f64 {
        let first = next_unit_interval(state).max(1e-12);
        let second = next_unit_interval(state);
        (-2.0 * first.ln()).sqrt() * (std::f64::consts::TAU * second).cos()
    }

    // Abramowitz-Stegun-style approximation for a unit normal CDF. Test fixture only; scientific
    // deployments should supply a vetted predictive CDF and include its numerical error in the
    // model-validation protocol.
    fn approximate_standard_normal_cdf(value: f64) -> f64 {
        let sign = if value < 0.0 { -1.0 } else { 1.0 };
        let x = value.abs() / std::f64::consts::SQRT_2;
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let polynomial = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
            - 0.284496736)
            * t)
            + 0.254829592)
            * t;
        0.5 * (1.0 + sign * (1.0 - polynomial * (-x * x).exp()))
    }

    #[test]
    fn adaptive_experiment_pit_detects_context_bias_faster_than_pooled_monitor() {
        const HORIZON: usize = 200;
        const MEAN_SHIFT: f64 = 0.8;
        let mut rng = 0x8CB9_2BA7_4F3D_1A65_u64;
        let mut intercept =
            PrequentialPitMartingale::new(0.05, 0).expect("valid pooled monitor");
        let mut contextual =
            PrequentialPitMartingale::new(0.05, 1).expect("valid context-aware monitor");
        let mut accumulated_observations = 0.0;
        let mut intercept_alarm = None;
        let mut contextual_alarm = None;

        for observation_index in 1..=HORIZON {
            // This action is chosen using only prior outcomes, so it is predictable. Under the
            // misspecified world, the physical mean shifts with the chosen action while the
            // issued candidate forecast remains N(0,1). Under the true conditional model (shift
            // zero), the current innovation is independent of this adaptive action.
            let action = if accumulated_observations >= 0.0 {
                -1.0
            } else {
                1.0
            };
            let observed =
                MEAN_SHIFT * action + standard_normal_sample(&mut rng);
            let pit = approximate_standard_normal_cdf(observed);
            accumulated_observations += observed;

            let pooled = intercept.observe(pit, &[]).expect("finite PIT");
            let conditional = contextual
                .observe(pit, &[action])
                .expect("finite PIT and predictable action context");

            if pooled.decision == PrequentialPitDecision::Alarm {
                intercept_alarm = Some(observation_index);
            }
            if conditional.decision == PrequentialPitDecision::Alarm {
                contextual_alarm = Some(observation_index);
            }
            if intercept_alarm.is_some() && contextual_alarm.is_some() {
                break;
            }
        }

        assert!(
            contextual_alarm.is_some(),
            "context-aware detector should catch the predeclared misspecification"
        );
        assert!(
            intercept_alarm.is_some(),
            "pooled detector should eventually catch this strong pooled shift"
        );
        assert!(
            contextual_alarm.unwrap() < intercept_alarm.unwrap(),
            "expected a shorter detection delay from the predeclared action context: contextual={contextual_alarm:?}, pooled={intercept_alarm:?}"
        );
    }

    #[test]
    fn context_channel_detects_conditional_bias_hidden_by_pooled_order_and_dispersion() {
        let mut monitor = PrequentialPitMartingale::new(0.05, 1).unwrap();
        let mut alarm_at = None;
        let mut context_order_dominates = false;

        for index in 0..100 {
            // Pooled PIT values alternate symmetrically between 0.25 and 0.75: their
            // order effects cancel, while the dispersion feature is exactly zero.
            // A predictable context sign reveals that each context's conditional PIT is biased.
            let (pit, context) = if index % 2 == 0 {
                (0.75, [1.0])
            } else {
                (0.25, [-1.0])
            };
            let result = monitor.observe(pit, &context).unwrap();
            if result.feature_wealths.iter().any(|snapshot| {
                snapshot.context_feature_index == Some(0)
                    && snapshot.kind == PitFeatureKind::Order
                    && snapshot.log_wealth > 0.5
            }) {
                context_order_dominates = true;
            }
            if result.decision == PrequentialPitDecision::Alarm {
                alarm_at = Some(result.observation_index);
                break;
            }
        }

        assert!(context_order_dominates);
        assert!(
            alarm_at.is_some_and(|index| index <= 100),
            "context-conditional PIT bias was not detected"
        );
        assert_eq!(monitor.state(), PrequentialPitState::Alarmed);
        assert!(matches!(
            monitor.observe(0.5, &[0.0]),
            Err(PrequentialPitError::AlreadyAlarmed)
        ));
    }

    #[test]
    fn predictable_contexts_preserve_null_smoke_alarm_control() {
        const STREAMS: usize = 1_000;
        const HORIZON: usize = 300;
        let mut rng = 0x8CB9_2BA7_4F3D_1A65_u64;
        let mut alarms = 0_usize;

        for _ in 0..STREAMS {
            let mut monitor = PrequentialPitMartingale::new(0.05, 2).unwrap();
            for time in 0..HORIZON {
                let pit = next_unit_interval(&mut rng);
                // These channels are deterministic/predictable functions of time and so are
                // available before the next observation. Uniform PIT values remain conditionally
                // centered against every channel under the specified null.
                let context = [
                    if time % 2 == 0 { 1.0 } else { -1.0 },
                    ((time % 7) as f64 - 3.0) / 3.0,
                ];
                let result = monitor.observe(pit, &context).unwrap();
                if result.decision == PrequentialPitDecision::Alarm {
                    alarms += 1;
                    break;
                }
            }
        }

        let empirical_rate = alarms as f64 / STREAMS as f64;
        assert!(
            empirical_rate <= 0.10,
            "prequential PIT null-smoke alarm rate was {empirical_rate:.4}"
        );
    }

    #[test]
    fn invalid_input_does_not_consume_observation_index() {
        let mut monitor = PrequentialPitMartingale::new(0.05, 0).unwrap();
        assert!(matches!(
            monitor.observe(1.1, &[]),
            Err(PrequentialPitError::InvalidPitValue)
        ));
        let result = monitor.observe(0.5, &[]).unwrap();
        assert_eq!(result.observation_index, 1);
        assert_eq!(monitor.observations_seen(), 1);
    }
}
