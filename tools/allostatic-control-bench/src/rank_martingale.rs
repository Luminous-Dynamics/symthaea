//! Fixed-reference predictive-rank martingales for anytime shift monitoring.
//!
//! This implements the predictive-rank conditioning and a bounded ONS-style
//! bettor inspired by Kuang and Xia (2026): each score is ranked against a fixed
//! reference sample; the next-rank null probabilities account for dependence
//! induced by reuse of that reference; and a predictably centered feature payoff
//! updates an online wealth process. The local ONS step-size coefficient is set
//! explicitly below and has not yet been independently checked against the
//! authors' exact algorithm. Do not claim exact method reproduction or the
//! paper's detection-delay bounds until that comparison is complete. The
//! nonnegative-wealth martingale argument remains conditional on valid rank
//! exchangeability, predictable bets, and positive betting factors; this module
//! does not make those assumptions true for a real plant or certify an action.

use std::cmp::Ordering;

/// Prototype ONS gain, fixed before benchmark evaluation. This is not asserted
/// to match the source paper's exact optimizer parameterization.
const ONS_STEP_SIZE: f64 = 4.5;

/// A score paired with an independent tie-breaking key.
/// The tie key is needed to retain an exchangeable total ordering when scores tie.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreToken {
    score: f64,
    tie_breaker: f64,
}

impl ScoreToken {
    pub fn new(score: f64, tie_breaker: f64) -> Option<Self> {
        if !score.is_finite()
            || !tie_breaker.is_finite()
            || !(0.0..=1.0).contains(&score)
            || !(0.0..=1.0).contains(&tie_breaker)
        {
            return None;
        }
        Some(Self {
            score,
            tie_breaker,
        })
    }

    pub const fn score(self) -> f64 {
        self.score
    }

    fn cmp_key(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.tie_breaker.total_cmp(&other.tie_breaker))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankFeature {
    /// Linear rank feature; responds to upward and downward stochastic shifts.
    Order,
    /// Center-versus-tail feature; responds to dispersion changes whose two tails
    /// reinforce instead of canceling.
    Dispersion,
}

impl RankFeature {
    fn value(self, rank_index: usize, reference_size: usize) -> f64 {
        let u = rank_index as f64 / reference_size as f64;
        match self {
            Self::Order => u - 0.5,
            Self::Dispersion => (2.0 * u - 1.0).abs() - 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankObservation {
    rank: usize,
    centered_payoff: f64,
    betting_factor: f64,
    wealth: f64,
    alarm: bool,
}

impl RankObservation {
    /// One-based rank among the fixed reference sample plus the test point.
    pub const fn rank(self) -> usize {
        self.rank
    }

    /// Feature value minus its exact predictive-null conditional mean.
    pub const fn centered_payoff(self) -> f64 {
        self.centered_payoff
    }

    pub const fn betting_factor(self) -> f64 {
        self.betting_factor
    }

    pub const fn wealth(self) -> f64 {
        self.wealth
    }

    pub const fn alarm(self) -> bool {
        self.alarm
    }
}

/// One feature-specific predictive-rank martingale.
#[derive(Clone, Debug)]
pub struct RankMartingale {
    reference: Vec<ScoreToken>,
    feature: RankFeature,
    feature_values: Vec<f64>,
    feature_min: f64,
    feature_max: f64,
    base_feature_sum: f64,
    observed_feature_sum: f64,
    observations_seen: u64,
    lambda: f64,
    accumulated_squared_gradients: f64,
    wealth: f64,
    wealth_threshold: f64,
}

impl RankMartingale {
    /// Build a martingale from a frozen reference set and a target false-alarm
    /// probability. The reference set is never updated using test-time samples.
    pub fn new(
        reference: &[ScoreToken],
        feature: RankFeature,
        target_false_alarm_rate: f64,
    ) -> Option<Self> {
        if reference.len() < 2
            || !target_false_alarm_rate.is_finite()
            || !(0.0..1.0).contains(&target_false_alarm_rate)
        {
            return None;
        }
        let mut ordered = reference.to_vec();
        ordered.sort_by(ScoreToken::cmp_key);
        if ordered.windows(2).any(|pair| pair[0].cmp_key(&pair[1]).is_eq()) {
            return None;
        }

        let n = ordered.len();
        let feature_values: Vec<f64> = (0..=n).map(|index| feature.value(index, n)).collect();
        let feature_min = feature_values.iter().copied().fold(f64::INFINITY, f64::min);
        let feature_max = feature_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let base_feature_sum = feature_values.iter().sum();

        Some(Self {
            reference: ordered,
            feature,
            feature_values,
            feature_min,
            feature_max,
            base_feature_sum,
            observed_feature_sum: 0.0,
            observations_seen: 0,
            lambda: 0.0,
            // ONS denominator is 1 + sum of past squared gradients.
            accumulated_squared_gradients: 1.0,
            wealth: 1.0,
            wealth_threshold: 1.0 / target_false_alarm_rate,
        })
    }

    pub const fn feature(&self) -> RankFeature {
        self.feature
    }

    pub const fn wealth(&self) -> f64 {
        self.wealth
    }

    pub const fn observations_seen(&self) -> u64 {
        self.observations_seen
    }

    pub const fn target_wealth_threshold(&self) -> f64 {
        self.wealth_threshold
    }

    pub fn observe(&mut self, token: ScoreToken) -> Option<RankObservation> {
        if !token.score.is_finite()
            || !token.tie_breaker.is_finite()
            || !(0.0..=1.0).contains(&token.score)
            || !(0.0..=1.0).contains(&token.tie_breaker)
        {
            return None;
        }

        let rank_index = self
            .reference
            .partition_point(|reference| reference.cmp_key(&token).is_lt());
        let rank = rank_index + 1;
        let reference_size = self.reference.len();
        let step_denominator = reference_size as f64 + self.observations_seen as f64 + 1.0;

        // Under the null, P(R_t = j | previous ranks) = (1 + N_{t-1,j})/(n+t).
        // Maintain the numerator's feature sum incrementally to avoid O(n) per step.
        let predictive_mean =
            (self.base_feature_sum + self.observed_feature_sum) / step_denominator;
        let feature_value = self.feature_values[rank_index];
        let centered_payoff = feature_value - predictive_mean;

        // B_t is the largest possible absolute centered payoff over all ranks.
        // b_t = 1 - B_t/4 ensures |lambda_t * Z_t| <= 3/4 and the betting
        // factor stays at least 1/4, avoiding singular ONS gradients.
        let bound = (self.feature_min - predictive_mean)
            .abs()
            .max((self.feature_max - predictive_mean).abs())
            .min(1.0);
        let projection_radius = 1.0 - bound / 4.0;
        self.lambda = self.lambda.clamp(-projection_radius, projection_radius);

        let betting_factor = 1.0 + self.lambda * centered_payoff;
        if !betting_factor.is_finite() || betting_factor < 0.25 - 1e-12 {
            return None;
        }
        self.wealth *= betting_factor;
        if !self.wealth.is_finite() {
            return None;
        }

        let gradient = centered_payoff / betting_factor;
        self.accumulated_squared_gradients += gradient * gradient;
        self.observed_feature_sum += feature_value;
        self.observations_seen += 1;

        let next_denominator = reference_size as f64 + self.observations_seen as f64 + 1.0;
        let next_predictive_mean =
            (self.base_feature_sum + self.observed_feature_sum) / next_denominator;
        let next_bound = (self.feature_min - next_predictive_mean)
            .abs()
            .max((self.feature_max - next_predictive_mean).abs())
            .min(1.0);
        let next_radius = 1.0 - next_bound / 4.0;
        let next_lambda =
            self.lambda + ONS_STEP_SIZE * gradient / self.accumulated_squared_gradients;
        self.lambda = next_lambda.clamp(-next_radius, next_radius);

        Some(RankObservation {
            rank,
            centered_payoff,
            betting_factor,
            wealth: self.wealth,
            alarm: self.wealth >= self.wealth_threshold,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankPortfolioObservation {
    order: RankObservation,
    dispersion: RankObservation,
    portfolio_wealth: f64,
    portfolio_alarm: bool,
}

impl RankPortfolioObservation {
    pub const fn order(self) -> RankObservation {
        self.order
    }

    pub const fn dispersion(self) -> RankObservation {
        self.dispersion
    }

    pub const fn portfolio_wealth(self) -> f64 {
        self.portfolio_wealth
    }

    pub const fn portfolio_alarm(self) -> bool {
        self.portfolio_alarm
    }
}

/// Equal-weight portfolio of order and dispersion rank martingales.
/// Because both components start at unit wealth, the portfolio also starts at one.
#[derive(Clone, Debug)]
pub struct RankMartingalePortfolio {
    order: RankMartingale,
    dispersion: RankMartingale,
    wealth_threshold: f64,
}

impl RankMartingalePortfolio {
    pub fn new(reference: &[ScoreToken], target_false_alarm_rate: f64) -> Option<Self> {
        Some(Self {
            order: RankMartingale::new(reference, RankFeature::Order, target_false_alarm_rate)?,
            dispersion: RankMartingale::new(
                reference,
                RankFeature::Dispersion,
                target_false_alarm_rate,
            )?,
            wealth_threshold: 1.0 / target_false_alarm_rate,
        })
    }

    pub const fn order_wealth(&self) -> f64 {
        self.order.wealth()
    }

    pub const fn dispersion_wealth(&self) -> f64 {
        self.dispersion.wealth()
    }

    pub fn observe(&mut self, token: ScoreToken) -> Option<RankPortfolioObservation> {
        let order = self.order.observe(token)?;
        let dispersion = self.dispersion.observe(token)?;
        let portfolio_wealth = (order.wealth() + dispersion.wealth()) / 2.0;
        if !portfolio_wealth.is_finite() {
            return None;
        }
        Some(RankPortfolioObservation {
            order,
            dispersion,
            portfolio_wealth,
            portfolio_alarm: portfolio_wealth >= self.wealth_threshold,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{RankFeature, RankMartingale, RankMartingalePortfolio, ScoreToken};

    fn reference() -> Vec<ScoreToken> {
        [0.1, 0.3, 0.5, 0.7, 0.9]
            .into_iter()
            .enumerate()
            .map(|(i, score)| {
                ScoreToken::new(score, (i as f64 + 1.0) / 10.0).expect("valid token")
            })
            .collect()
    }

    #[test]
    fn rejects_invalid_calibration_inputs_and_exact_duplicate_keys() {
        assert!(RankMartingale::new(&[], RankFeature::Order, 0.05).is_none());
        assert!(RankMartingale::new(&reference(), RankFeature::Order, 0.0).is_none());
        assert!(RankMartingale::new(&reference(), RankFeature::Order, f64::NAN).is_none());
        let duplicates = vec![
            ScoreToken::new(0.2, 0.4).expect("valid token"),
            ScoreToken::new(0.2, 0.4).expect("valid token"),
        ];
        assert!(RankMartingale::new(&duplicates, RankFeature::Order, 0.05).is_none());
    }

    #[test]
    fn order_feature_distinguishes_upper_and_lower_ranks() {
        let calibration = reference();
        let mut upper =
            RankMartingale::new(&calibration, RankFeature::Order, 0.05).expect("upper");
        let mut lower =
            RankMartingale::new(&calibration, RankFeature::Order, 0.05).expect("lower");
        let high = upper
            .observe(ScoreToken::new(1.0, 0.95).expect("high token"))
            .expect("high observation");
        let low = lower
            .observe(ScoreToken::new(0.0, 0.05).expect("low token"))
            .expect("low observation");
        assert!(high.centered_payoff() > 0.0);
        assert!(low.centered_payoff() < 0.0);
        assert_eq!(high.rank(), 6);
        assert_eq!(low.rank(), 1);
    }

    #[test]
    fn dispersion_feature_reinforces_both_tails() {
        let calibration = reference();
        let mut upper =
            RankMartingale::new(&calibration, RankFeature::Dispersion, 0.05).expect("upper");
        let mut lower =
            RankMartingale::new(&calibration, RankFeature::Dispersion, 0.05).expect("lower");
        let high = upper
            .observe(ScoreToken::new(1.0, 0.95).expect("high token"))
            .expect("high observation");
        let low = lower
            .observe(ScoreToken::new(0.0, 0.05).expect("low token"))
            .expect("low observation");
        assert!(high.centered_payoff() > 0.0);
        assert!(low.centered_payoff() > 0.0);
    }

    #[test]
    fn betting_factor_stays_positive_and_portfolio_starts_at_unit_wealth() {
        let calibration = reference();
        let mut monitor =
            RankMartingale::new(&calibration, RankFeature::Order, 0.05).expect("monitor");
        for i in 0..64 {
            let token = ScoreToken::new(
                if i % 2 == 0 { 0.0 } else { 1.0 },
                (i as f64 + 0.5) / 65.0,
            )
            .expect("valid token");
            let observation = monitor.observe(token).expect("valid observation");
            assert!(observation.betting_factor() >= 0.25 - 1e-12);
            assert!(observation.wealth().is_finite());
        }

        let mut portfolio =
            RankMartingalePortfolio::new(&calibration, 0.05).expect("portfolio");
        assert!((portfolio.order_wealth() - 1.0).abs() < 1e-12);
        assert!((portfolio.dispersion_wealth() - 1.0).abs() < 1e-12);
        let first = portfolio
            .observe(ScoreToken::new(0.4, 0.6).expect("token"))
            .expect("first observation");
        assert!((first.portfolio_wealth() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejected_observation_does_not_advance_rank_history() {
        let mut monitor =
            RankMartingale::new(&reference(), RankFeature::Order, 0.05).expect("monitor");
        assert!(
            monitor
                .observe(ScoreToken {
                    score: f64::NAN,
                    tie_breaker: 0.2,
                })
                .is_none()
        );
        assert_eq!(monitor.observations_seen(), 0);
    }
}
