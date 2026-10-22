//! Fixed-budget Continuous Rejects (CR-C/CR-A) baselines for Bernoulli best-arm experiments.
//!
//! Adapted from Wang, Tzeng, and Proutiere, NeurIPS 2023. This is a synthetic
//! policy baseline, not a QPU sampler, MIS solver, or proof that paper guarantees
//! transfer unchanged to this implementation. Counts are candidate observations,
//! not physical QPU shots when one result updates multiple observables.

use core::fmt;

/// Continuous Rejects elimination rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MisContinuousRejectsVariant {
    /// Conservative criterion: the worst arm must be separated from every rival.
    Conservative,
    /// Aggressive criterion: the worst arm must be separated from the average rival.
    Aggressive,
}

/// One fixed-budget Continuous Rejects outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisContinuousRejectsOutcome {
    /// Selected candidate index, chosen by final empirical mean among survivors.
    pub selected_vertex: usize,
    /// Candidate observations consumed; equals the configured fixed budget.
    pub total_observations: u64,
    /// Observations allocated to each candidate.
    pub observations_by_vertex: Vec<u64>,
    /// Bernoulli successes observed for each candidate.
    pub successes_by_vertex: Vec<u64>,
    /// Candidates eliminated in chronological order.
    pub eliminated_vertices_order: Vec<usize>,
    /// Number of observations consumed before continuous elimination checks start.
    pub warmup_observations: u64,
}

/// Input or invariant error from the Continuous Rejects reference implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisContinuousRejectsError {
    /// At least one candidate must be provided.
    EmptyCandidates,
    /// Candidate probability is non-finite or outside [0, 1].
    InvalidProbability { /// Zero-based candidate index.
        vertex: usize,
    },
    /// The budget must provide one initial observation per candidate.
    BudgetSmallerThanCandidateCount {
        /// Configured candidate-observation budget.
        budget: u64,
        /// Number of candidates.
        candidates: usize,
    },
    /// A checked integer count overflowed.
    CountOverflow,
    /// An internal active-set or accounting invariant was violated.
    InvariantViolation,
}

impl fmt::Display for MisContinuousRejectsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidates => write!(f, "at least one candidate is required"),
            Self::InvalidProbability { vertex } => write!(
                f,
                "candidate {vertex} has a non-finite or out-of-range probability"
            ),
            Self::BudgetSmallerThanCandidateCount { budget, candidates } => write!(
                f,
                "fixed budget {budget} is smaller than candidate count {candidates}"
            ),
            Self::CountOverflow => write!(f, "candidate-observation count overflowed"),
            Self::InvariantViolation => write!(f, "Continuous Rejects invariant was violated"),
        }
    }
}

impl std::error::Error for MisContinuousRejectsError {}

/// Run one Continuous Rejects variant on independent synthetic Bernoulli candidates.
///
/// The seed derives independent, deterministic candidate streams compatible with
/// the MIS sampling simulator's seed scheme. Budget means candidate observations,
/// not physical-QPU cost.
pub fn run_mis_continuous_rejects(
    probabilities: &[f64],
    budget: u64,
    trial_seed: u64,
    variant: MisContinuousRejectsVariant,
) -> Result<MisContinuousRejectsOutcome, MisContinuousRejectsError> {
    let k = probabilities.len();
    if k == 0 {
        return Err(MisContinuousRejectsError::EmptyCandidates);
    }
    if budget < k as u64 {
        return Err(MisContinuousRejectsError::BudgetSmallerThanCandidateCount {
            budget,
            candidates: k,
        });
    }
    for (vertex, probability) in probabilities.iter().copied().enumerate() {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(MisContinuousRejectsError::InvalidProbability { vertex });
        }
    }

    let mut rngs = make_candidate_rngs(trial_seed, k);
    let mut observations = vec![0_u64; k];
    let mut successes = vec![0_u64; k];
    let mut active = (0..k).collect::<Vec<_>>();
    let mut is_active = vec![true; k];
    let mut eliminated = Vec::new();
    let mut total = 0_u64;

    // One initial sample from each arm.
    for vertex in 0..k {
        sample_one(vertex, probabilities, &mut rngs, &mut observations, &mut successes, &mut total)?;
    }

    // Rational theta_0 = 1/(floor(ln(K)) + 2) is less than 1/ln(K)
    // and does not depend on the fixed budget.
    let warmup_denominator = (k.max(2) as f64).ln().floor() as u64 + 2;
    let warmup_target = (budget / warmup_denominator).max(k as u64).min(budget);
    while total < warmup_target {
        let vertex = least_observed(&active, &observations)?;
        sample_one(vertex, probabilities, &mut rngs, &mut observations, &mut successes, &mut total)?;
    }
    let warmup_observations = total;

    while total < budget {
        if active.len() > 2 {
            let worst = empirical_worst(&active, &observations, &successes)?;
            let equal_counts = active
                .iter()
                .all(|&vertex| observations[vertex] == observations[active[0]]);
            let mut excluded_total = 0_u64;
            let mut excluded_max = 0_u64;
            for vertex in 0..k {
                if !is_active[vertex] {
                    excluded_total = excluded_total.checked_add(observations[vertex])
                        .ok_or(MisContinuousRejectsError::CountOverflow)?;
                    excluded_max = excluded_max.max(observations[vertex]);
                }
            }

            // The paper's eligibility condition requires equal active-arm counts
            // and more observations on the rejected arm than on any earlier reject.
            if equal_counts && observations[worst] > excluded_max {
                let active_total = active.iter().try_fold(0_u64, |sum, &vertex| {
                    sum.checked_add(observations[vertex])
                        .ok_or(MisContinuousRejectsError::CountOverflow)
                })?;
                let remaining_budget = budget.checked_sub(excluded_total)
                    .ok_or(MisContinuousRejectsError::InvariantViolation)?;
                if active_total == 0 || remaining_budget == 0 {
                    return Err(MisContinuousRejectsError::InvariantViolation);
                }
                let beta = active_total as f64 * (active.len() as f64).ln()
                    / remaining_budget as f64;
                if !beta.is_finite() || beta <= 0.0 {
                    return Err(MisContinuousRejectsError::InvariantViolation);
                }
                let threshold = 1.0 / beta.sqrt() - 1.0;
                let gap = rejection_gap(&active, worst, &observations, &successes, variant);
                if gap >= threshold {
                    active.retain(|&vertex| vertex != worst);
                    is_active[worst] = false;
                    eliminated.push(worst);
                    if active.len() < 2 {
                        return Err(MisContinuousRejectsError::InvariantViolation);
                    }
                }
            }
        }

        let vertex = least_observed(&active, &observations)?;
        sample_one(vertex, probabilities, &mut rngs, &mut observations, &mut successes, &mut total)?;
    }

    let selected = empirical_best(&active, &observations, &successes)?;
    let accounted = observations.iter().try_fold(0_u64, |sum, &count| {
        sum.checked_add(count).ok_or(MisContinuousRejectsError::CountOverflow)
    })?;
    if accounted != budget || total != budget {
        return Err(MisContinuousRejectsError::InvariantViolation);
    }
    Ok(MisContinuousRejectsOutcome {
        selected_vertex: selected,
        total_observations: total,
        observations_by_vertex: observations,
        successes_by_vertex: successes,
        eliminated_vertices_order: eliminated,
        warmup_observations,
    })
}

fn mean(vertex: usize, observations: &[u64], successes: &[u64]) -> f64 {
    successes[vertex] as f64 / observations[vertex] as f64
}

fn empirical_worst(
    active: &[usize], observations: &[u64], successes: &[u64],
) -> Result<usize, MisContinuousRejectsError> {
    let mut worst = *active.first().ok_or(MisContinuousRejectsError::InvariantViolation)?;
    for &candidate in active.iter().skip(1) {
        let cm = mean(candidate, observations, successes);
        let wm = mean(worst, observations, successes);
        if cm < wm || (cm == wm && candidate > worst) { worst = candidate; }
    }
    Ok(worst)
}

fn empirical_best(
    active: &[usize], observations: &[u64], successes: &[u64],
) -> Result<usize, MisContinuousRejectsError> {
    let mut best = *active.first().ok_or(MisContinuousRejectsError::InvariantViolation)?;
    for &candidate in active.iter().skip(1) {
        if mean(candidate, observations, successes) > mean(best, observations, successes) {
            best = candidate;
        }
    }
    Ok(best)
}

fn least_observed(active: &[usize], observations: &[u64]) -> Result<usize, MisContinuousRejectsError> {
    let mut least = *active.first().ok_or(MisContinuousRejectsError::InvariantViolation)?;
    for &candidate in active.iter().skip(1) {
        if observations[candidate] < observations[least] { least = candidate; }
    }
    Ok(least)
}

fn rejection_gap(
    active: &[usize], worst: usize, observations: &[u64], successes: &[u64],
    variant: MisContinuousRejectsVariant,
) -> f64 {
    let worst_mean = mean(worst, observations, successes);
    let others = active.iter().copied().filter(|&v| v != worst);
    match variant {
        MisContinuousRejectsVariant::Conservative => others
            .map(|v| mean(v, observations, successes))
            .fold(f64::INFINITY, f64::min) - worst_mean,
        MisContinuousRejectsVariant::Aggressive => {
            let mut sum = 0.0;
            let mut count = 0_usize;
            for vertex in others {
                sum += mean(vertex, observations, successes);
                count += 1;
            }
            if count == 0 { f64::NEG_INFINITY } else { sum / count as f64 - worst_mean }
        }
    }
}

fn sample_one(
    vertex: usize, probabilities: &[f64], rngs: &mut [SplitMix64],
    observations: &mut [u64], successes: &mut [u64], total: &mut u64,
) -> Result<(), MisContinuousRejectsError> {
    let rng = rngs.get_mut(vertex).ok_or(MisContinuousRejectsError::InvariantViolation)?;
    let count = observations.get_mut(vertex).ok_or(MisContinuousRejectsError::InvariantViolation)?;
    let wins = successes.get_mut(vertex).ok_or(MisContinuousRejectsError::InvariantViolation)?;
    if rng.next_unit_f64() < probabilities[vertex] {
        *wins = wins.checked_add(1).ok_or(MisContinuousRejectsError::CountOverflow)?;
    }
    *count = count.checked_add(1).ok_or(MisContinuousRejectsError::CountOverflow)?;
    *total = total.checked_add(1).ok_or(MisContinuousRejectsError::CountOverflow)?;
    Ok(())
}

fn make_candidate_rngs(seed: u64, k: usize) -> Vec<SplitMix64> {
    (0..k).map(|vertex| {
        let salt = (vertex as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        SplitMix64::new(mix64(seed ^ salt))
    }).collect()
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Small deterministic pseudorandom generator for reproducible fixtures; not cryptographic.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SplitMix64 { state: u64 }

impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.state)
    }
    fn next_unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_variants_are_reproducible_and_spend_exact_budget() {
        for variant in [MisContinuousRejectsVariant::Conservative, MisContinuousRejectsVariant::Aggressive] {
            let a = run_mis_continuous_rejects(&[0.8, 0.6, 0.3, 0.2], 257, 20261022, variant).unwrap();
            let b = run_mis_continuous_rejects(&[0.8, 0.6, 0.3, 0.2], 257, 20261022, variant).unwrap();
            assert_eq!(a, b);
            assert_eq!(a.total_observations, 257);
            assert_eq!(a.observations_by_vertex.iter().sum::<u64>(), 257);
            assert!(a.selected_vertex < 4);
            assert!(a.eliminated_vertices_order.len() <= 2);
        }
    }

    #[test]
    fn supports_single_candidate_and_minimum_budget() {
        let one = run_mis_continuous_rejects(&[0.7], 9, 1, MisContinuousRejectsVariant::Conservative).unwrap();
        assert_eq!(one.selected_vertex, 0);
        assert_eq!(one.observations_by_vertex, vec![9]);
        let minimum = run_mis_continuous_rejects(&[0.9, 0.1, 0.2], 3, 2, MisContinuousRejectsVariant::Aggressive).unwrap();
        assert_eq!(minimum.total_observations, 3);
        assert!(minimum.eliminated_vertices_order.is_empty());
    }

    #[test]
    fn exact_budget_invariant_holds_across_candidate_counts() {
        for k in 1..=12 {
            let probabilities = (0..k).map(|i| 0.1 + 0.8 * i as f64 / k as f64).collect::<Vec<_>>();
            for budget in [k as u64, k as u64 + 1, 32, 128, 512] {
                if budget < k as u64 { continue; }
                for variant in [MisContinuousRejectsVariant::Conservative, MisContinuousRejectsVariant::Aggressive] {
                    let out = run_mis_continuous_rejects(&probabilities, budget, 17, variant).unwrap();
                    assert_eq!(out.total_observations, budget);
                    assert_eq!(out.observations_by_vertex.iter().sum::<u64>(), budget);
                    assert!(out.selected_vertex < k);
                }
            }
        }
    }

    #[test]
    fn handles_ties_and_bernoulli_boundaries() {
        for variant in [MisContinuousRejectsVariant::Conservative, MisContinuousRejectsVariant::Aggressive] {
            for probabilities in [&[0.5, 0.5, 0.5, 0.5][..], &[1.0, 0.0, 1.0][..]] {
                let out = run_mis_continuous_rejects(probabilities, 128, 7, variant).unwrap();
                assert!(out.selected_vertex < probabilities.len());
                assert_eq!(out.total_observations, 128);
            }
        }
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(
            run_mis_continuous_rejects(&[], 0, 1, MisContinuousRejectsVariant::Conservative).unwrap_err(),
            MisContinuousRejectsError::EmptyCandidates
        );
        assert_eq!(
            run_mis_continuous_rejects(&[0.2, f64::NAN], 2, 1, MisContinuousRejectsVariant::Conservative).unwrap_err(),
            MisContinuousRejectsError::InvalidProbability { vertex: 1 }
        );
        assert_eq!(
            run_mis_continuous_rejects(&[0.8, 0.2], 1, 1, MisContinuousRejectsVariant::Conservative).unwrap_err(),
            MisContinuousRejectsError::BudgetSmallerThanCandidateCount { budget: 1, candidates: 2 }
        );
    }
}
