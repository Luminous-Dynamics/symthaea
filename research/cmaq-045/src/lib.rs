#![forbid(unsafe_code)]

//! Predictable conditional rank-spectrum monitoring for temporal dependence shifts.
//!
//! Each component bets on one fixed Haar rank contrast only when its frozen
//! predictable context (derived solely from previous ranks) is active. The
//! current feature is centered against the exact Pólya-urn predictive rank law.
//! A normalized fixed mixture over contexts, features, directions and rates
//! therefore remains a nonnegative e-process under the PRM exchangeability null.
//!
//! The crate detects only departures visible to its finite context/feature bank.
//! It does not authenticate input ranks, prove physical independence or infer
//! hazard semantics.

const BETTING_RATES: [f64; 3] = [0.25, 0.5, 1.0];
const PREVIOUS_RANK_BUCKETS: usize = 8;
const CONTEXT_COUNT: usize = PREVIOUS_RANK_BUCKETS + 6;
const MAX_RANK_BUCKETS: usize = 1 << 18;
const NEG_INFINITY: f64 = f64::NEG_INFINITY;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConditionalError {
    InvalidCalibrationCount,
    InvalidMiscoverage,
    TooManyRankBuckets,
    InvalidRank,
    CounterOverflow,
    ArithmeticOverflow,
    InvalidPredictivePayoff,
    InvalidBettingFactor,
    EmptyFeatureBank,
    AlreadyQuarantined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    Positive,
    Negative,
}

impl Direction {
    fn sign(self) -> f64 {
        match self {
            Self::Positive => 1.0,
            Self::Negative => -1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HaarContrast {
    start: usize,
    midpoint: usize,
    end: usize,
    contrasts_at_level: usize,
    active_resolution_levels: usize,
}

impl HaarContrast {
    fn code_for_rank(&self, rank_one_based: usize) -> i128 {
        let bucket = rank_one_based - 1;
        if (self.start..self.midpoint).contains(&bucket) {
            1
        } else if (self.midpoint..self.end).contains(&bucket) {
            -1
        } else {
            0
        }
    }

    fn log_weight(&self) -> f64 {
        -(self.active_resolution_levels as f64).ln()
            - (self.contrasts_at_level as f64).ln()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Context {
    PreviousRankBucket(usize),
    PreviousTwoStepUp,
    PreviousTwoStepDown,
    PreviousExtremeLow,
    PreviousExtremeHigh,
    PreviousHighRun,
    PreviousLowRun,
}

impl Context {
    fn name(self) -> &'static str {
        match self {
            Self::PreviousRankBucket(_) => "previous_rank_bucket",
            Self::PreviousTwoStepUp => "previous_two_step_up",
            Self::PreviousTwoStepDown => "previous_two_step_down",
            Self::PreviousExtremeLow => "previous_extreme_low",
            Self::PreviousExtremeHigh => "previous_extreme_high",
            Self::PreviousHighRun => "previous_high_run",
            Self::PreviousLowRun => "previous_low_run",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Component {
    context: Context,
    contrast: HaarContrast,
    direction: Direction,
    betting_rate: f64,
    log_weight: f64,
    log_wealth: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RankHistory {
    last_rank: Option<usize>,
    second_last_rank: Option<usize>,
    high_run: u64,
    low_run: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConditionalUpdate {
    Accepted {
        rank_one_based: usize,
        log_mixture_wealth: f64,
        active_contexts: usize,
        component_count: usize,
    },
    Quarantined {
        rank_one_based: usize,
        log_mixture_wealth: f64,
        threshold_log_wealth: f64,
        active_contexts: usize,
        component_count: usize,
    },
}

/// Complete finite rank-histogram basis conditioned on a frozen predictable
/// context bank. All contexts are selected from prior observations only.
#[derive(Clone, Debug, PartialEq)]
pub struct ConditionalRankSpectrumMonitor {
    calibration_count: usize,
    alpha_numerator: u32,
    alpha_denominator: u32,
    rank_counts: Vec<u64>,
    observations_seen: u64,
    history: RankHistory,
    contexts: Vec<Context>,
    components: Vec<Component>,
    resolution_levels: usize,
    quarantined: bool,
}

impl ConditionalRankSpectrumMonitor {
    pub fn new(
        calibration_count: usize,
        alpha_numerator: u32,
        alpha_denominator: u32,
    ) -> Result<Self, ConditionalError> {
        if calibration_count == 0 {
            return Err(ConditionalError::InvalidCalibrationCount);
        }
        if alpha_numerator == 0
            || alpha_denominator == 0
            || alpha_numerator >= alpha_denominator
        {
            return Err(ConditionalError::InvalidMiscoverage);
        }

        let rank_buckets = calibration_count
            .checked_add(1)
            .ok_or(ConditionalError::ArithmeticOverflow)?;
        if rank_buckets > MAX_RANK_BUCKETS {
            return Err(ConditionalError::TooManyRankBuckets);
        }
        let padded = rank_buckets
            .checked_next_power_of_two()
            .ok_or(ConditionalError::ArithmeticOverflow)?;
        let maximum_levels = padded.trailing_zeros() as usize;
        if maximum_levels == 0 {
            return Err(ConditionalError::EmptyFeatureBank);
        }

        let mut levels = Vec::<Vec<HaarContrast>>::with_capacity(maximum_levels);
        for _level in 0..maximum_levels {
            let width = padded >> _level;
            let nodes = padded / width;
            let mut ranges = Vec::new();
            for node in 0..nodes {
                let start = node
                    .checked_mul(width)
                    .ok_or(ConditionalError::ArithmeticOverflow)?;
                let midpoint = start
                    .checked_add(width / 2)
                    .ok_or(ConditionalError::ArithmeticOverflow)?;
                let end = start
                    .checked_add(width)
                    .ok_or(ConditionalError::ArithmeticOverflow)?
                    .min(rank_buckets);
                if start < rank_buckets && midpoint < rank_buckets && end > midpoint {
                    ranges.push((start, midpoint, end));
                }
            }
            if !ranges.is_empty() {
                let count = ranges.len();
                levels.push(
                    ranges
                        .into_iter()
                        .map(|(start, midpoint, end)| HaarContrast {
                            start,
                            midpoint,
                            end,
                            contrasts_at_level: count,
                            active_resolution_levels: 0,
                        })
                        .collect(),
                );
            }
        }

        let resolution_levels = levels.len();
        if resolution_levels == 0 {
            return Err(ConditionalError::EmptyFeatureBank);
        }
        let feature_count = levels.iter().map(Vec::len).sum::<usize>();
        if feature_count != calibration_count {
            return Err(ConditionalError::EmptyFeatureBank);
        }

        let contexts = build_context_bank();
        let component_count = feature_count
            .checked_mul(contexts.len())
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_mul(BETTING_RATES.len()))
            .ok_or(ConditionalError::ArithmeticOverflow)?;
        let mut components = Vec::with_capacity(component_count);

        for level in levels {
            for mut contrast in level {
                contrast.active_resolution_levels = resolution_levels;
                let feature_log_weight = contrast.log_weight();
                for context in &contexts {
                    for direction in [Direction::Positive, Direction::Negative] {
                        for betting_rate in BETTING_RATES {
                            let log_weight = -(contexts.len() as f64).ln()
                                + feature_log_weight
                                - 2.0f64.ln()
                                - (BETTING_RATES.len() as f64).ln();
                            components.push(Component {
                                context: *context,
                                contrast,
                                direction,
                                betting_rate,
                                log_weight,
                                log_wealth: 0.0,
                            });
                        }
                    }
                }
            }
        }

        if components.len() != component_count {
            return Err(ConditionalError::EmptyFeatureBank);
        }

        let monitor = Self {
            calibration_count,
            alpha_numerator,
            alpha_denominator,
            rank_counts: vec![0; rank_buckets],
            observations_seen: 0,
            history: RankHistory::default(),
            contexts,
            components,
            resolution_levels,
            quarantined: false,
        };
        if monitor.log_mixture_wealth()?.abs() > 1e-10 {
            return Err(ConditionalError::ArithmeticOverflow);
        }
        Ok(monitor)
    }

    /// Returns active context IDs using history as it stood before the next rank.
    pub fn active_context_names(&self) -> Vec<&'static str> {
        self.active_context_indices()
            .into_iter()
            .map(|index| self.contexts[index].name())
            .collect()
    }

    pub fn active_context_indices(&self) -> Vec<usize> {
        let mut active = Vec::with_capacity(4);
        let Some(last_rank) = self.history.last_rank else {
            return active;
        };

        let rank_buckets = self.calibration_count + 1;
        let previous_bucket = ((last_rank - 1) * PREVIOUS_RANK_BUCKETS / rank_buckets)
            .min(PREVIOUS_RANK_BUCKETS - 1);
        active.push(previous_bucket);

        if let Some(second_last_rank) = self.history.second_last_rank {
            if last_rank > second_last_rank {
                active.push(PREVIOUS_RANK_BUCKETS);
            } else if last_rank < second_last_rank {
                active.push(PREVIOUS_RANK_BUCKETS + 1);
            }
        }

        let low_cutoff = (rank_buckets + 3) / 4;
        let high_cutoff = (3 * rank_buckets + 3) / 4;
        if last_rank <= low_cutoff {
            active.push(PREVIOUS_RANK_BUCKETS + 2);
        }
        if last_rank >= high_cutoff {
            active.push(PREVIOUS_RANK_BUCKETS + 3);
        }
        if self.history.high_run >= 2 {
            active.push(PREVIOUS_RANK_BUCKETS + 4);
        }
        if self.history.low_run >= 2 {
            active.push(PREVIOUS_RANK_BUCKETS + 5);
        }
        active.sort_unstable();
        active.dedup();
        active
    }

    /// Adds one already-qualified rank. Contexts are computed before this rank
    /// is recorded, so the current observation cannot select its own betting
    /// condition. State mutation is atomic with respect to validation/errors.
    pub fn observe_rank(
        &mut self,
        rank_one_based: usize,
    ) -> Result<ConditionalUpdate, ConditionalError> {
        if self.quarantined {
            return Err(ConditionalError::AlreadyQuarantined);
        }
        if rank_one_based == 0 || rank_one_based > self.calibration_count + 1 {
            return Err(ConditionalError::InvalidRank);
        }

        let next_seen = self
            .observations_seen
            .checked_add(1)
            .ok_or(ConditionalError::CounterOverflow)?;
        let rank_index = rank_one_based - 1;
        let next_rank_count = self.rank_counts[rank_index]
            .checked_add(1)
            .ok_or(ConditionalError::CounterOverflow)?;
        let active_contexts = self.active_context_indices();
        let active_context_mask = self
            .contexts
            .iter()
            .enumerate()
            .map(|(index, _)| active_contexts.binary_search(&index).is_ok())
            .collect::<Vec<_>>();

        let mut weighted_prefix = Vec::with_capacity(self.rank_counts.len() + 1);
        weighted_prefix.push(0i128);
        for count in &self.rank_counts {
            let weight = i128::from(*count)
                .checked_add(1)
                .ok_or(ConditionalError::ArithmeticOverflow)?;
            let next = weighted_prefix
                .last()
                .copied()
                .ok_or(ConditionalError::ArithmeticOverflow)?
                .checked_add(weight)
                .ok_or(ConditionalError::ArithmeticOverflow)?;
            weighted_prefix.push(next);
        }

        let mut next_log_wealths = Vec::with_capacity(self.components.len());
        for component in &self.components {
            if !active_context_mask[context_index(component.context)] {
                next_log_wealths.push(component.log_wealth);
                continue;
            }

            let base_payoff = centered_payoff(
                self.calibration_count,
                &component.contrast,
                rank_one_based,
                next_seen,
                &weighted_prefix,
            )?;
            let payoff = base_payoff * component.direction.sign();
            if !payoff.is_finite() || !(-1.0 - 1e-12..=1.0 + 1e-12).contains(&payoff) {
                return Err(ConditionalError::InvalidPredictivePayoff);
            }

            let factor = 1.0 + component.betting_rate * payoff;
            if !factor.is_finite() || factor < -1e-12 {
                return Err(ConditionalError::InvalidBettingFactor);
            }
            let next_log_wealth = if factor <= 0.0 {
                NEG_INFINITY
            } else {
                component.log_wealth + factor.ln()
            };
            if next_log_wealth.is_nan() || next_log_wealth == f64::INFINITY {
                return Err(ConditionalError::ArithmeticOverflow);
            }
            next_log_wealths.push(next_log_wealth);
        }

        let log_mixture = log_mixture(&self.components, &next_log_wealths)?;
        let threshold = (f64::from(self.alpha_denominator) / f64::from(self.alpha_numerator)).ln();
        if !log_mixture.is_finite() || !threshold.is_finite() {
            return Err(ConditionalError::ArithmeticOverflow);
        }

        let next_history = update_history(self.history, rank_one_based, self.calibration_count + 1);
        self.rank_counts[rank_index] = next_rank_count;
        self.observations_seen = next_seen;
        self.history = next_history;
        for (component, next_log_wealth) in self.components.iter_mut().zip(next_log_wealths) {
            component.log_wealth = next_log_wealth;
        }

        if log_mixture >= threshold {
            self.quarantined = true;
            Ok(ConditionalUpdate::Quarantined {
                rank_one_based,
                log_mixture_wealth: log_mixture,
                threshold_log_wealth: threshold,
                active_contexts: active_contexts.len(),
                component_count: self.components.len(),
            })
        } else {
            Ok(ConditionalUpdate::Accepted {
                rank_one_based,
                log_mixture_wealth: log_mixture,
                active_contexts: active_contexts.len(),
                component_count: self.components.len(),
            })
        }
    }

    pub fn log_mixture_wealth(&self) -> Result<f64, ConditionalError> {
        let values = self
            .components
            .iter()
            .map(|component| component.log_wealth)
            .collect::<Vec<_>>();
        log_mixture(&self.components, &values)
    }

    pub fn calibration_count(&self) -> usize {
        self.calibration_count
    }

    pub fn observations_seen(&self) -> u64 {
        self.observations_seen
    }

    pub fn context_count(&self) -> usize {
        self.contexts.len()
    }

    pub fn context_names(&self) -> Vec<&'static str> {
        self.contexts.iter().map(|context| context.name()).collect()
    }

    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    pub fn resolution_levels(&self) -> usize {
        self.resolution_levels
    }

    pub fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    pub fn mixture_weight_sum(&self) -> f64 {
        self.components
            .iter()
            .map(|component| component.log_weight.exp())
            .sum()
    }

    pub fn rank_counts(&self) -> &[u64] {
        &self.rank_counts
    }
}

fn build_context_bank() -> Vec<Context> {
    let mut contexts = (0..PREVIOUS_RANK_BUCKETS)
        .map(Context::PreviousRankBucket)
        .collect::<Vec<_>>();
    contexts.extend([
        Context::PreviousTwoStepUp,
        Context::PreviousTwoStepDown,
        Context::PreviousExtremeLow,
        Context::PreviousExtremeHigh,
        Context::PreviousHighRun,
        Context::PreviousLowRun,
    ]);
    contexts
}

fn context_index(context: Context) -> usize {
    match context {
        Context::PreviousRankBucket(index) => index,
        Context::PreviousTwoStepUp => PREVIOUS_RANK_BUCKETS,
        Context::PreviousTwoStepDown => PREVIOUS_RANK_BUCKETS + 1,
        Context::PreviousExtremeLow => PREVIOUS_RANK_BUCKETS + 2,
        Context::PreviousExtremeHigh => PREVIOUS_RANK_BUCKETS + 3,
        Context::PreviousHighRun => PREVIOUS_RANK_BUCKETS + 4,
        Context::PreviousLowRun => PREVIOUS_RANK_BUCKETS + 5,
    }
}

fn update_history(previous: RankHistory, current_rank: usize, rank_buckets: usize) -> RankHistory {
    let low_cutoff = (rank_buckets + 3) / 4;
    let high_cutoff = (3 * rank_buckets + 3) / 4;
    let is_low = current_rank <= low_cutoff;
    let is_high = current_rank >= high_cutoff;
    RankHistory {
        last_rank: Some(current_rank),
        second_last_rank: previous.last_rank,
        high_run: if is_high {
            previous.high_run.saturating_add(1)
        } else {
            0
        },
        low_run: if is_low {
            previous.low_run.saturating_add(1)
        } else {
            0
        },
    }
}

fn centered_payoff(
    calibration_count: usize,
    contrast: &HaarContrast,
    rank_one_based: usize,
    observation_index_one_based: u64,
    weighted_prefix: &[i128],
) -> Result<f64, ConditionalError> {
    if rank_one_based == 0 || rank_one_based > calibration_count + 1
        || weighted_prefix.len() != calibration_count + 2
    {
        return Err(ConditionalError::InvalidRank);
    }
    let total = (calibration_count as u128)
        .checked_add(u128::from(observation_index_one_based))
        .ok_or(ConditionalError::ArithmeticOverflow)?;
    let total_i128 = i128::try_from(total).map_err(|_| ConditionalError::ArithmeticOverflow)?;

    let positive = weighted_prefix[contrast.midpoint]
        .checked_sub(weighted_prefix[contrast.start])
        .ok_or(ConditionalError::ArithmeticOverflow)?;
    let negative = weighted_prefix[contrast.end]
        .checked_sub(weighted_prefix[contrast.midpoint])
        .ok_or(ConditionalError::ArithmeticOverflow)?;
    let expected_code = positive
        .checked_sub(negative)
        .ok_or(ConditionalError::ArithmeticOverflow)?;
    let observed = contrast.code_for_rank(rank_one_based)
        .checked_mul(total_i128)
        .ok_or(ConditionalError::ArithmeticOverflow)?;
    let numerator = observed
        .checked_sub(expected_code)
        .ok_or(ConditionalError::ArithmeticOverflow)?;
    let denominator = total
        .checked_mul(2)
        .ok_or(ConditionalError::ArithmeticOverflow)?;
    let payoff = numerator as f64 / denominator as f64;
    if !payoff.is_finite() || !(-1.0 - 1e-12..=1.0 + 1e-12).contains(&payoff) {
        return Err(ConditionalError::InvalidPredictivePayoff);
    }
    Ok(payoff.clamp(-1.0, 1.0))
}

fn log_mixture(components: &[Component], log_wealths: &[f64]) -> Result<f64, ConditionalError> {
    if components.is_empty() || components.len() != log_wealths.len() {
        return Err(ConditionalError::ArithmeticOverflow);
    }
    let mut total = NEG_INFINITY;
    for (component, log_wealth) in components.iter().zip(log_wealths) {
        let weighted = component.log_weight + log_wealth;
        if weighted.is_nan() || weighted == f64::INFINITY {
            return Err(ConditionalError::ArithmeticOverflow);
        }
        total = log_add_exp(total, weighted);
    }
    if !total.is_finite() {
        return Err(ConditionalError::ArithmeticOverflow);
    }
    Ok(total)
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    if left == NEG_INFINITY {
        return right;
    }
    if right == NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monitor() -> ConditionalRankSpectrumMonitor {
        ConditionalRankSpectrumMonitor::new(15, 1, 20).unwrap()
    }

    fn prefix(counts: &[u64]) -> Vec<i128> {
        let mut values = vec![0i128];
        for count in counts {
            let weight = i128::from(*count) + 1;
            values.push(values.last().copied().unwrap() + weight);
        }
        values
    }

    #[test]
    fn initial_mixture_is_one_and_full_component_inventory_is_frozen() {
        let monitor = monitor();
        assert_eq!(monitor.context_count(), CONTEXT_COUNT);
        assert_eq!(monitor.component_count(), 15 * CONTEXT_COUNT * 2 * 3);
        assert!((monitor.mixture_weight_sum() - 1.0).abs() < 1e-10);
        assert!(monitor.log_mixture_wealth().unwrap().abs() < 1e-10);
    }

    #[test]
    fn no_context_is_active_before_the_first_rank() {
        let monitor = monitor();
        assert!(monitor.active_context_names().is_empty());
    }

    #[test]
    fn context_selection_uses_only_previously_committed_ranks() {
        let mut monitor = monitor();
        let first = monitor.observe_rank(1).unwrap();
        assert!(matches!(first, ConditionalUpdate::Accepted { active_contexts: 0, .. }));
        assert_eq!(monitor.active_context_names().len(), 2); // previous rank bin + low extreme

        let second_active_before = monitor.active_context_names();
        assert!(second_active_before.len() >= 2);
        let _ = monitor.observe_rank(16).unwrap();
        // The context used for rank 16 came from rank 1, not rank 16 itself.
        assert!(matches!(
            monitor.observe_rank(16).unwrap(),
            ConditionalUpdate::Accepted { .. } | ConditionalUpdate::Quarantined { .. }
        ));
    }

    #[test]
    fn active_context_payoffs_remain_conditionally_centered() {
        let mut monitor = monitor();
        monitor.history = RankHistory {
            last_rank: Some(16),
            second_last_rank: Some(1),
            high_run: 2,
            low_run: 0,
        };
        let active = monitor.active_context_indices();
        let counts = [2u64, 0, 4, 1, 3, 0, 0, 2, 1, 0, 1, 0, 0, 0, 1, 2];
        let prefix = prefix(&counts);
        let t = 18u64;
        let denominator = 15.0 + t as f64;

        for component in &monitor.components {
            let indicator = if active.binary_search(&context_index(component.context)).is_ok() {
                1.0
            } else {
                0.0
            };
            let expectation: f64 = (1..=16)
                .map(|rank| {
                    let mass = (1.0 + counts[rank - 1] as f64) / denominator;
                    let payoff = centered_payoff(
                        15,
                        &component.contrast,
                        rank,
                        t,
                        &prefix,
                    )
                    .unwrap();
                    mass * indicator * payoff * component.direction.sign()
                })
                .sum();
            assert!(
                expectation.abs() < 1e-12,
                "conditional context {} lost centering",
                context_index(component.context)
            );
        }
    }

    #[test]
    fn betting_factors_are_nonnegative_for_all_ranks_and_rates() {
        let monitor = monitor();
        let prefix = prefix(&[0; 16]);
        for component in &monitor.components {
            for rank in 1..=16 {
                let payoff = centered_payoff(15, &component.contrast, rank, 1, &prefix).unwrap()
                    * component.direction.sign();
                let factor = 1.0 + component.betting_rate * payoff;
                assert!(factor >= -1e-12);
            }
        }
    }

    #[test]
    fn invalid_rank_leaves_rank_history_and_wealth_unchanged() {
        let mut monitor = monitor();
        monitor.observe_rank(2).unwrap();
        let before = monitor.clone();
        assert_eq!(monitor.observe_rank(0), Err(ConditionalError::InvalidRank));
        assert_eq!(monitor, before);
        assert_eq!(monitor.observe_rank(17), Err(ConditionalError::InvalidRank));
        assert_eq!(monitor, before);
    }

    #[test]
    fn high_and_low_runs_activate_only_after_two_prior_observations() {
        let mut monitor = monitor();
        monitor.observe_rank(16).unwrap();
        assert!(!monitor.active_context_names().contains(&"previous_high_run"));
        monitor.observe_rank(16).unwrap();
        assert!(monitor.active_context_names().contains(&"previous_high_run"));

        let mut monitor = monitor();
        monitor.observe_rank(1).unwrap();
        assert!(!monitor.active_context_names().contains(&"previous_low_run"));
        monitor.observe_rank(1).unwrap();
        assert!(monitor.active_context_names().contains(&"previous_low_run"));
    }

    #[test]
    fn movement_context_requires_two_prior_ranks() {
        let mut monitor = monitor();
        monitor.observe_rank(2).unwrap();
        assert!(!monitor.active_context_names().contains(&"previous_two_step_up"));
        monitor.observe_rank(10).unwrap();
        assert!(monitor.active_context_names().contains(&"previous_two_step_up"));
        assert!(!monitor.active_context_names().contains(&"previous_two_step_down"));
    }

    #[test]
    fn counter_overflow_fails_before_mutation() {
        let mut monitor = monitor();
        monitor.observations_seen = u64::MAX;
        let before_counts = monitor.rank_counts.clone();
        let before_history = monitor.history;
        let before_components = monitor.components.clone();
        assert_eq!(monitor.observe_rank(3), Err(ConditionalError::CounterOverflow));
        assert_eq!(monitor.rank_counts, before_counts);
        assert_eq!(monitor.history, before_history);
        assert_eq!(monitor.components, before_components);
    }

    #[test]
    fn exact_predictive_centering_holds_after_a_nonuniform_history() {
        let mut monitor = monitor();
        for rank in [1, 2, 2, 8, 14, 16, 15, 3] {
            monitor.observe_rank(rank).unwrap();
        }
        let t = monitor.observations_seen() + 1;
        let counts = &monitor.rank_counts;
        let prefix = prefix(counts);
        let denominator = 15.0 + t as f64;
        for contrast in &monitor.components.iter().map(|component| component.contrast).collect::<Vec<_>>() {
            let expectation: f64 = (1..=16)
                .map(|rank| {
                    let probability = (1.0 + counts[rank - 1] as f64) / denominator;
                    probability * centered_payoff(15, contrast, rank, t, &prefix).unwrap()
                })
                .sum();
            assert!(expectation.abs() < 1e-12);
        }
    }

    #[test]
    fn quarantine_is_permanent() {
        let mut monitor = monitor();
        for component in &mut monitor.components {
            component.log_wealth = 100.0;
        }
        let result = monitor.observe_rank(1).unwrap();
        assert!(matches!(result, ConditionalUpdate::Quarantined { .. }));
        assert!(monitor.is_quarantined());
        assert_eq!(
            monitor.observe_rank(1),
            Err(ConditionalError::AlreadyQuarantined)
        );
    }
}
