//! Deterministic synthetic simulator for validating adaptive MIS sampling policies.
//!
//! This is a policy-validation harness, not a quantum-device or graph-optimization
//! simulator. Each candidate's observations are independent Bernoulli streams with
//! known synthetic means. It measures selection coverage, selective error, and sample
//! use; it does not model cross-vertex correlations from shared QAOA bitstrings, device
//! drift, circuit compilation, graph dynamics, or physical runtime.

use core::fmt;

use crate::mis_shot_budget::{
    MisCandidateEstimate, MisShotAbstentionReason, MisShotBudgetConfig,
    MisShotBudgetDecision, MisShotBudgetError, plan_mis_shots,
};
use crate::mis_informer::MisSelectionMode;

/// Known candidate inclusion probabilities for one synthetic policy-validation case.
#[derive(Debug, Clone, PartialEq)]
pub struct MisSamplingScenario {
    /// Stable scenario label for raw benchmark output.
    pub name: String,
    /// Synthetic Bernoulli mean for each zero-based vertex/candidate.
    pub inclusion_probabilities: Vec<f64>,
}

impl MisSamplingScenario {
    /// Validate and create a synthetic scenario.
    pub fn new(
        name: impl Into<String>,
        inclusion_probabilities: Vec<f64>,
    ) -> Result<Self, MisSamplingSimulationError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(MisSamplingSimulationError::EmptyScenarioName);
        }
        if inclusion_probabilities.is_empty() {
            return Err(MisSamplingSimulationError::EmptyCandidates);
        }
        for (vertex, probability) in inclusion_probabilities.iter().copied().enumerate() {
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return Err(MisSamplingSimulationError::InvalidProbability { vertex });
            }
        }
        Ok(Self { name, inclusion_probabilities })
    }

    /// Index of the candidate with the highest known mean; ties resolve to lowest index.
    pub fn true_best_vertex(&self) -> usize {
        let mut best = 0;
        for vertex in 1..self.inclusion_probabilities.len() {
            if self.inclusion_probabilities[vertex] > self.inclusion_probabilities[best] {
                best = vertex;
            }
        }
        best
    }
}

/// One paired simulation trial, using common deterministic observation streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisSamplingTrial {
    /// Scenario label.
    pub scenario_name: String,
    /// Zero-based trial number within the scenario.
    pub trial_index: usize,
    /// Exact derived seed for this trial.
    pub trial_seed: u64,
    /// Highest-mean candidate under the synthetic oracle.
    pub true_best_vertex: usize,
    /// Selected vertex, or `None` if the adaptive policy abstained.
    pub adaptive_selected_vertex: Option<usize>,
    /// Whether the adaptive selection equals the synthetic oracle's best candidate.
    pub adaptive_correct: Option<bool>,
    /// Abstention reason, or `None` when a candidate was selected.
    pub adaptive_abstention_reason: Option<MisShotAbstentionReason>,
    /// Total observations used by the adaptive policy.
    pub adaptive_total_shots: u64,
    /// Adaptive observation count for each candidate.
    pub adaptive_shots_by_vertex: Vec<u64>,
    /// Adaptive inclusion count for each candidate.
    pub adaptive_successes_by_vertex: Vec<u64>,
    /// Selected candidate for uniform fixed-budget empirical-rate baseline.
    pub uniform_selected_vertex: usize,
    /// Whether the uniform selection equals the synthetic oracle's best candidate.
    pub uniform_correct: bool,
    /// Total observations used by the uniform baseline (equals configured budget).
    pub uniform_total_shots: u64,
    /// Uniform observation count for each candidate.
    pub uniform_shots_by_vertex: Vec<u64>,
    /// Uniform inclusion count for each candidate.
    pub uniform_successes_by_vertex: Vec<u64>,
}

/// Counts accumulated from paired adaptive-vs-uniform synthetic trials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisSamplingAggregate {
    /// Number of trials.
    pub trial_count: u64,
    /// Adaptive trials that returned a candidate.
    pub adaptive_selected: u64,
    /// Correct adaptive selections.
    pub adaptive_correct: u64,
    /// Incorrect adaptive selections among trials that returned a candidate.
    pub adaptive_incorrect: u64,
    /// Adaptive abstentions.
    pub adaptive_abstained: u64,
    /// Sum of candidate observations used by the adaptive policy.
    pub adaptive_total_shots: u128,
    /// Correct uniform-baseline selections.
    pub uniform_correct: u64,
    /// Incorrect uniform-baseline selections.
    pub uniform_incorrect: u64,
    /// Sum of candidate observations used by the uniform baseline.
    pub uniform_total_shots: u128,
}

impl MisSamplingAggregate {
    /// Fraction of adaptive trials that returned a recommendation.
    pub fn adaptive_coverage(&self) -> f64 {
        ratio(self.adaptive_selected, self.trial_count)
    }

    /// Incorrect selections divided by adaptive selections; zero when none selected.
    pub fn adaptive_selective_error(&self) -> f64 {
        ratio(self.adaptive_incorrect, self.adaptive_selected)
    }

    /// Incorrect adaptive selections divided by all trials, with abstentions counted
    /// as unresolved rather than incorrectly selected.
    pub fn adaptive_unconditional_error(&self) -> f64 {
        ratio(self.adaptive_incorrect, self.trial_count)
    }

    /// Mean adaptive candidate observations per trial.
    pub fn adaptive_mean_shots(&self) -> f64 {
        self.adaptive_total_shots as f64 / self.trial_count as f64
    }

    /// Mean uniform-baseline candidate observations per trial.
    pub fn uniform_mean_shots(&self) -> f64 {
        self.uniform_total_shots as f64 / self.trial_count as f64
    }

    /// Fraction of uniform-baseline trials selecting the synthetic best candidate.
    pub fn uniform_accuracy(&self) -> f64 {
        ratio(self.uniform_correct, self.trial_count)
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 { 0.0 } else { numerator as f64 / denominator as f64 }
}

/// Errors in the synthetic sampling harness or its policy invariants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisSamplingSimulationError {
    /// Scenario names must be non-empty.
    EmptyScenarioName,
    /// At least one candidate is required.
    EmptyCandidates,
    /// Candidate Bernoulli probability is invalid.
    InvalidProbability { vertex: usize },
    /// At least one trial is required.
    ZeroTrials,
    /// Uniform comparison needs at least one observation per candidate.
    BudgetSmallerThanCandidateCount { budget: u64, candidates: usize },
    /// Error returned by the adaptive shot-budget planner.
    Policy(MisShotBudgetError),
    /// The policy returned an allocation for an unknown candidate.
    UnknownAllocatedVertex { vertex: usize },
    /// Total number of shots overflowed the configured integer range.
    ShotCountOverflow,
    /// The planner returned a non-progressing or over-budget allocation.
    BudgetInvariantViolation,
    /// A selected vertex is outside the scenario.
    InvalidSelectedVertex { vertex: usize },
}

impl fmt::Display for MisSamplingSimulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyScenarioName => write!(f, "scenario name must not be empty"),
            Self::EmptyCandidates => write!(f, "at least one candidate is required"),
            Self::InvalidProbability { vertex } => {
                write!(f, "candidate {vertex} has a non-finite or out-of-range probability")
            }
            Self::ZeroTrials => write!(f, "simulation trial count must be positive"),
            Self::BudgetSmallerThanCandidateCount { budget, candidates } => write!(
                f,
                "fixed-budget baseline requires budget {budget} >= candidate count {candidates}"
            ),
            Self::Policy(error) => write!(f, "adaptive policy failed: {error}"),
            Self::UnknownAllocatedVertex { vertex } => {
                write!(f, "policy allocated shots to unknown candidate {vertex}")
            }
            Self::ShotCountOverflow => write!(f, "shot count overflowed"),
            Self::BudgetInvariantViolation => {
                write!(f, "policy shot allocation failed progress or budget invariant")
            }
            Self::InvalidSelectedVertex { vertex } => {
                write!(f, "policy selected unknown candidate {vertex}")
            }
        }
    }
}

impl std::error::Error for MisSamplingSimulationError {}

impl From<MisShotBudgetError> for MisSamplingSimulationError {
    fn from(error: MisShotBudgetError) -> Self {
        Self::Policy(error)
    }
}

/// Simulate paired adaptive and uniform fixed-budget policies on synthetic Bernoulli arms.
///
/// The policies receive common random-number streams, derived deterministically from
/// `master_seed`, trial index, and candidate identity. The adaptive arm receives only
/// the observations it requests; the baseline takes a uniform fixed allocation that
/// spends exactly `config.max_total_shots`. Since this is a synthetic independent-arm
/// model, results do not establish quantum advantage or predict final MIS quality.
pub fn simulate_mis_sampling_scenario(
    scenario: &MisSamplingScenario,
    config: MisShotBudgetConfig,
    trials: usize,
    master_seed: u64,
) -> Result<Vec<MisSamplingTrial>, MisSamplingSimulationError> {
    if trials == 0 {
        return Err(MisSamplingSimulationError::ZeroTrials);
    }
    if scenario.inclusion_probabilities.is_empty() {
        return Err(MisSamplingSimulationError::EmptyCandidates);
    }
    if config.max_total_shots < scenario.inclusion_probabilities.len() as u64 {
        return Err(MisSamplingSimulationError::BudgetSmallerThanCandidateCount {
            budget: config.max_total_shots,
            candidates: scenario.inclusion_probabilities.len(),
        });
    }
    // Re-run the constructor's probability validation in case the public fields were
    // mutated after construction.
    for (vertex, probability) in scenario.inclusion_probabilities.iter().copied().enumerate() {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(MisSamplingSimulationError::InvalidProbability { vertex });
        }
    }

    let candidate_count = scenario.inclusion_probabilities.len();
    let true_best_vertex = scenario.true_best_vertex();
    let mut output = Vec::with_capacity(trials);

    for trial_index in 0..trials {
        let trial_seed = derive_trial_seed(master_seed, trial_index as u64);
        let mut adaptive_rngs = make_candidate_rngs(trial_seed, candidate_count);
        let mut uniform_rngs = make_candidate_rngs(trial_seed, candidate_count);
        let mut adaptive_shots = vec![0_u64; candidate_count];
        let mut adaptive_successes = vec![0_u64; candidate_count];
        let mut estimates = (0..candidate_count)
            .map(|vertex| MisCandidateEstimate::new(vertex, 0, 0))
            .collect::<Result<Vec<_>, _>>()?;

        let (adaptive_selected, adaptive_correct, adaptive_abstention, adaptive_total) = loop {
            match plan_mis_shots(&estimates, MisSelectionMode::PreferInSet, config)? {
                MisShotBudgetDecision::RequestMoreShots {
                    allocations,
                    total_shots,
                    ..
                } => {
                    let actual_total = sum_shots(&adaptive_shots)?;
                    if total_shots != actual_total || allocations.is_empty() {
                        return Err(MisSamplingSimulationError::BudgetInvariantViolation);
                    }
                    for allocation in allocations {
                        let vertex = allocation.vertex;
                        let Some(probability) = scenario.inclusion_probabilities.get(vertex).copied() else {
                            return Err(MisSamplingSimulationError::UnknownAllocatedVertex { vertex });
                        };
                        if allocation.additional_shots == 0 {
                            return Err(MisSamplingSimulationError::BudgetInvariantViolation);
                        }
                        let successes = sample_bernoulli(
                            probability,
                            allocation.additional_shots,
                            &mut adaptive_rngs[vertex],
                        );
                        adaptive_shots[vertex] = adaptive_shots[vertex]
                            .checked_add(allocation.additional_shots)
                            .ok_or(MisSamplingSimulationError::ShotCountOverflow)?;
                        adaptive_successes[vertex] = adaptive_successes[vertex]
                            .checked_add(successes)
                            .ok_or(MisSamplingSimulationError::ShotCountOverflow)?;
                        estimates[vertex] = MisCandidateEstimate::new(
                            vertex,
                            adaptive_shots[vertex],
                            adaptive_successes[vertex],
                        )?;
                    }
                    let updated_total = sum_shots(&adaptive_shots)?;
                    if updated_total <= actual_total || updated_total > config.max_total_shots {
                        return Err(MisSamplingSimulationError::BudgetInvariantViolation);
                    }
                }
                MisShotBudgetDecision::Selected { vertex, total_shots, .. } => {
                    if vertex >= candidate_count {
                        return Err(MisSamplingSimulationError::InvalidSelectedVertex { vertex });
                    }
                    if total_shots != sum_shots(&adaptive_shots)? {
                        return Err(MisSamplingSimulationError::BudgetInvariantViolation);
                    }
                    break (
                        Some(vertex),
                        Some(scenario.inclusion_probabilities[vertex] == scenario.inclusion_probabilities[true_best_vertex]),
                        None,
                        total_shots,
                    );
                }
                MisShotBudgetDecision::Abstain { reason, total_shots } => {
                    if total_shots != sum_shots(&adaptive_shots)? {
                        return Err(MisSamplingSimulationError::BudgetInvariantViolation);
                    }
                    break (None, None, Some(reason), total_shots);
                }
            }
        };

        let uniform_shots = uniform_allocations(config.max_total_shots, candidate_count);
        let mut uniform_successes = vec![0_u64; candidate_count];
        for vertex in 0..candidate_count {
            uniform_successes[vertex] = sample_bernoulli(
                scenario.inclusion_probabilities[vertex],
                uniform_shots[vertex],
                &mut uniform_rngs[vertex],
            );
        }
        let uniform_selected_vertex = best_empirical_rate(&uniform_shots, &uniform_successes)?;
        let uniform_total_shots = sum_shots(&uniform_shots)?;
        if uniform_total_shots != config.max_total_shots {
            return Err(MisSamplingSimulationError::BudgetInvariantViolation);
        }

        output.push(MisSamplingTrial {
            scenario_name: scenario.name.clone(),
            trial_index,
            trial_seed,
            true_best_vertex,
            adaptive_selected_vertex: adaptive_selected,
            adaptive_correct,
            adaptive_abstention_reason: adaptive_abstention,
            adaptive_total_shots: adaptive_total,
            adaptive_shots_by_vertex: adaptive_shots,
            adaptive_successes_by_vertex: adaptive_successes,
            uniform_selected_vertex,
            uniform_correct: scenario.inclusion_probabilities[uniform_selected_vertex]
                == scenario.inclusion_probabilities[true_best_vertex],
            uniform_total_shots,
            uniform_shots_by_vertex: uniform_shots,
            uniform_successes_by_vertex: uniform_successes,
        });
    }

    Ok(output)
}

/// Aggregate trial rows without dropping abstentions or failed selections.
pub fn aggregate_mis_sampling_trials(
    trials: &[MisSamplingTrial],
) -> Result<MisSamplingAggregate, MisSamplingSimulationError> {
    if trials.is_empty() {
        return Err(MisSamplingSimulationError::ZeroTrials);
    }
    let mut aggregate = MisSamplingAggregate {
        trial_count: u64::try_from(trials.len()).map_err(|_| MisSamplingSimulationError::ShotCountOverflow)?,
        adaptive_selected: 0,
        adaptive_correct: 0,
        adaptive_incorrect: 0,
        adaptive_abstained: 0,
        adaptive_total_shots: 0,
        uniform_correct: 0,
        uniform_incorrect: 0,
        uniform_total_shots: 0,
    };

    for trial in trials {
        aggregate.adaptive_total_shots += u128::from(trial.adaptive_total_shots);
        aggregate.uniform_total_shots += u128::from(trial.uniform_total_shots);
        if trial.adaptive_selected_vertex.is_some() {
            aggregate.adaptive_selected += 1;
            match trial.adaptive_correct {
                Some(true) => aggregate.adaptive_correct += 1,
                Some(false) => aggregate.adaptive_incorrect += 1,
                None => return Err(MisSamplingSimulationError::BudgetInvariantViolation),
            }
            if trial.adaptive_abstention_reason.is_some() {
                return Err(MisSamplingSimulationError::BudgetInvariantViolation);
            }
        } else {
            aggregate.adaptive_abstained += 1;
            if trial.adaptive_correct.is_some() || trial.adaptive_abstention_reason.is_none() {
                return Err(MisSamplingSimulationError::BudgetInvariantViolation);
            }
        }
        if trial.uniform_correct {
            aggregate.uniform_correct += 1;
        } else {
            aggregate.uniform_incorrect += 1;
        }
    }
    Ok(aggregate)
}

fn uniform_allocations(total_budget: u64, candidate_count: usize) -> Vec<u64> {
    let count = candidate_count as u64;
    let base = total_budget / count;
    let remainder = total_budget % count;
    (0..candidate_count)
        .map(|vertex| base + if (vertex as u64) < remainder { 1 } else { 0 })
        .collect()
}

fn best_empirical_rate(
    shots: &[u64],
    successes: &[u64],
) -> Result<usize, MisSamplingSimulationError> {
    if shots.is_empty() || shots.len() != successes.len() || shots.iter().any(|&count| count == 0) {
        return Err(MisSamplingSimulationError::BudgetInvariantViolation);
    }
    let mut best = 0;
    for vertex in 1..shots.len() {
        let candidate_rate = successes[vertex] as f64 / shots[vertex] as f64;
        let best_rate = successes[best] as f64 / shots[best] as f64;
        if candidate_rate > best_rate {
            best = vertex;
        }
    }
    Ok(best)
}

fn sum_shots(shots: &[u64]) -> Result<u64, MisSamplingSimulationError> {
    shots.iter().try_fold(0_u64, |total, &count| {
        total.checked_add(count).ok_or(MisSamplingSimulationError::ShotCountOverflow)
    })
}

fn sample_bernoulli(probability: f64, shots: u64, rng: &mut SplitMix64) -> u64 {
    let mut successes = 0;
    for _ in 0..shots {
        if rng.next_unit_f64() < probability {
            successes += 1;
        }
    }
    successes
}

fn make_candidate_rngs(trial_seed: u64, candidate_count: usize) -> Vec<SplitMix64> {
    (0..candidate_count)
        .map(|vertex| {
            let candidate_salt = (vertex as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
            SplitMix64::new(mix64(trial_seed ^ candidate_salt))
        })
        .collect()
}

fn derive_trial_seed(master_seed: u64, trial_index: u64) -> u64 {
    mix64(master_seed ^ trial_index.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Small deterministic PRNG used only for reproducible simulation fixtures; not cryptographic.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.state)
    }

    fn next_unit_f64(&mut self) -> f64 {
        let top_53_bits = self.next_u64() >> 11;
        top_53_bits as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> MisShotBudgetConfig {
        MisShotBudgetConfig::new(8, 8, 256, 0.05).expect("valid policy config")
    }

    #[test]
    fn identical_seeds_replay_bit_identically() {
        let scenario = MisSamplingScenario::new("wide-gap", vec![0.8, 0.2])
            .expect("valid synthetic scenario");
        let first = simulate_mis_sampling_scenario(&scenario, default_config(), 80, 20261022)
            .expect("simulation succeeds");
        let second = simulate_mis_sampling_scenario(&scenario, default_config(), 80, 20261022)
            .expect("simulation repeats");
        assert_eq!(first, second);
    }

    #[test]
    fn adaptive_policy_respects_budget_and_uniform_baseline_spends_exact_budget() {
        let scenario = MisSamplingScenario::new("three-arms", vec![0.7, 0.5, 0.2])
            .expect("valid synthetic scenario");
        let config = MisShotBudgetConfig::new(8, 8, 257, 0.05).expect("valid policy config");
        let trials = simulate_mis_sampling_scenario(&scenario, config, 40, 17)
            .expect("simulation succeeds");
        for trial in &trials {
            assert!(trial.adaptive_total_shots <= config.max_total_shots);
            assert_eq!(trial.adaptive_shots_by_vertex.iter().sum::<u64>(), trial.adaptive_total_shots);
            assert_eq!(trial.uniform_shots_by_vertex.iter().sum::<u64>(), config.max_total_shots);
            assert_eq!(trial.uniform_total_shots, config.max_total_shots);
            assert_eq!(
                trial.adaptive_selected_vertex.is_some(),
                trial.adaptive_correct.is_some()
            );
        }
    }

    #[test]
    fn aggregate_keeps_abstentions_separate_from_wrong_choices() {
        let scenario = MisSamplingScenario::new("near-tie", vec![0.51, 0.49])
            .expect("valid synthetic scenario");
        let trials = simulate_mis_sampling_scenario(&scenario, default_config(), 80, 8)
            .expect("simulation succeeds");
        let aggregate = aggregate_mis_sampling_trials(&trials).expect("aggregate succeeds");
        assert_eq!(aggregate.adaptive_selected + aggregate.adaptive_abstained, aggregate.trial_count);
        assert_eq!(aggregate.adaptive_correct + aggregate.adaptive_incorrect, aggregate.adaptive_selected);
        assert_eq!(aggregate.uniform_correct + aggregate.uniform_incorrect, aggregate.trial_count);
        assert!(aggregate.adaptive_mean_shots() <= default_config().max_total_shots as f64);
        assert!(aggregate.uniform_mean_shots() == default_config().max_total_shots as f64);
    }

    #[test]
    fn every_equal_mean_candidate_counts_as_a_correct_best_arm() {
        let scenario = MisSamplingScenario::new("exact-tie", vec![0.5, 0.5, 0.5])
            .expect("valid tie scenario");
        let trials = simulate_mis_sampling_scenario(&scenario, default_config(), 40, 2026)
            .expect("simulation succeeds");
        assert!(trials.iter().all(|trial| trial.uniform_correct));
        assert!(trials.iter().all(|trial| {
            trial.adaptive_selected_vertex.is_none() || trial.adaptive_correct == Some(true)
        }));
    }

    #[test]
    fn rejects_invalid_probabilities_and_too_small_baseline_budget() {
        assert_eq!(
            MisSamplingScenario::new("invalid", vec![0.4, f64::NAN])
                .expect_err("non-finite probability"),
            MisSamplingSimulationError::InvalidProbability { vertex: 1 }
        );
        let scenario = MisSamplingScenario::new("two-arms", vec![0.8, 0.2])
            .expect("valid scenario");
        let config = MisShotBudgetConfig::new(1, 1, 1, 0.05).expect("valid planner config");
        assert_eq!(
            simulate_mis_sampling_scenario(&scenario, config, 1, 1)
                .expect_err("baseline cannot cover both candidates"),
            MisSamplingSimulationError::BudgetSmallerThanCandidateCount { budget: 1, candidates: 2 }
        );
    }
}
