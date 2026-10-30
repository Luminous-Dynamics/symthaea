//! Preregistered paired inference for suite-scoped quantum/classical comparisons.
//!
//! The inferential unit is one independent workload block, not a shot, retry,
//! timing sample, or repeated seed from the same workload instance. Repeated
//! within-block runs must be reduced by a preregistered rule before this API.

use std::collections::HashSet;

/// One paired quantum/classical observation for one independent workload block.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedBenchmarkObservation {
    /// Unique identifier for the independent block used for inference.
    pub independent_block_id: String,
    /// Unique fingerprint for the workload instance represented by this block.
    pub workload_fingerprint: String,
    /// Digest binding workload, objective, quality contract, verifier and resource metric.
    pub comparison_contract_fingerprint: String,
    /// Quantum end-to-end resource per independently verified acceptable answer.
    pub quantum_resource_per_accepted_answer: f64,
    /// Classical end-to-end resource per independently verified acceptable answer.
    pub classical_resource_per_accepted_answer: f64,
    /// Quantum output quality; higher values must be better.
    pub quantum_quality: f64,
    /// Classical output quality; higher values must be better.
    pub classical_quality: f64,
}

/// Preregistered practical margins for a paired comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedAdvantageConfig {
    /// Required minimum fractional reduction in resource use, non-negative.
    pub min_relative_resource_reduction: f64,
    /// Largest tolerated quantum quality decrease relative to the classical run.
    pub quality_non_inferiority_margin: f64,
    /// Tie tolerance in log(classical resource / quantum resource) units.
    pub resource_log_ratio_tie_tolerance: f64,
    /// Tie tolerance in output-quality difference units.
    pub quality_difference_tie_tolerance: f64,
}

/// Exact one-sided paired sign-test summary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactSignTest {
    /// Number of positive signs beyond the preregistered margin.
    pub wins: usize,
    /// Number of negative signs beyond the preregistered margin.
    pub losses: usize,
    /// Number of observations within the tie tolerance.
    pub ties: usize,
    /// Number of independent blocks in the exact test; ties are non-wins.
    pub effective_n: usize,
    /// P[X >= wins] for X ~ Binomial(effective_n, 0.5).
    pub p_value_one_sided: f64,
}

/// Paired result; p-values remain raw until corrected across the claim family.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedAdvantageReport {
    /// Number of independent paired workload blocks.
    pub independent_blocks: usize,
    /// Median of quantum quality minus classical quality.
    pub median_quality_difference: f64,
    /// Median log(classical resource / quantum resource).
    pub median_log_classical_over_quantum_resource: f64,
    /// Exponentiated median resource ratio, absent if it overflows.
    pub median_classical_over_quantum_resource_ratio: Option<f64>,
    /// Exact test that quantum quality exceeds classical quality minus margin.
    pub quality_non_inferiority: ExactSignTest,
    /// Exact test that quantum resource reduction exceeds the minimum margin.
    pub resource_superiority: ExactSignTest,
    /// Intersection-union p-value: max(quality p, resource p) for the joint claim.
    pub composite_claim_p_value: f64,
    /// Shared contract digest carried by every paired observation.
    pub comparison_contract_fingerprint: String,
}

/// One preregistered hypothesis in a family-wise correction set.
#[derive(Debug, Clone, PartialEq)]
pub struct RawHypothesis {
    /// Stable, unique claim identifier.
    pub hypothesis_id: String,
    /// Raw p-value from the preregistered test.
    pub p_value: f64,
}

/// Holm step-down result for one hypothesis.
#[derive(Debug, Clone, PartialEq)]
pub struct HolmAdjustedPValue {
    /// Stable claim identifier.
    pub hypothesis_id: String,
    /// Original p-value.
    pub raw_p_value: f64,
    /// Holm family-wise adjusted p-value.
    pub adjusted_p_value: f64,
    /// One-based rank in ascending raw-p order (ID breaks exact ties).
    pub rank: usize,
    /// Whether the adjusted p-value is no greater than the declared alpha.
    pub reject_at_alpha: bool,
}

/// Input/configuration validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairedAdvantageError {
    /// No independent paired workload blocks were supplied.
    EmptyObservations,
    /// At least two independent blocks are required for an inferential comparison.
    InsufficientIndependentBlocks,
    /// A required identifier is blank.
    EmptyIdentifier,
    /// An independent block identifier is duplicated.
    DuplicateBlockId,
    /// A workload fingerprint is duplicated, risking pseudo-replication.
    DuplicateWorkloadFingerprint,
    /// Comparison contract fingerprints differ across observations.
    ContractMismatch,
    /// A resource value is not finite and strictly positive.
    InvalidResource,
    /// An output quality value is not finite.
    InvalidQuality,
    /// A preregistered margin or tie tolerance is invalid.
    InvalidConfig,
    /// A hypothesis identifier is blank or duplicated.
    InvalidHypothesisId,
    /// A p-value is outside [0, 1] or is non-finite.
    InvalidPValue,
    /// Holm alpha must lie strictly inside (0, 1).
    InvalidAlpha,
}

impl std::fmt::Display for PairedAdvantageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::EmptyObservations => "no independent paired workload blocks",
            Self::InsufficientIndependentBlocks => "at least two independent blocks are required",
            Self::EmptyIdentifier => "required identifier is blank",
            Self::DuplicateBlockId => "duplicate independent block identifier",
            Self::DuplicateWorkloadFingerprint => "duplicate workload fingerprint",
            Self::ContractMismatch => "paired comparison contract fingerprints differ",
            Self::InvalidResource => "resource values must be finite and strictly positive",
            Self::InvalidQuality => "quality values must be finite",
            Self::InvalidConfig => "invalid preregistered margin or tie tolerance",
            Self::InvalidHypothesisId => "hypothesis identifiers must be nonempty and unique",
            Self::InvalidPValue => "p-values must be finite and lie in [0, 1]",
            Self::InvalidAlpha => "alpha must lie strictly inside (0, 1)",
        };
        f.write_str(message)
    }
}

impl std::error::Error for PairedAdvantageError {}

/// Analyze paired blocks using margin-aware exact one-sided sign tests.
///
/// Positive quality differences mean higher quantum quality. Positive resource
/// log-ratios mean lower quantum resource use. Both tests are necessary for a
/// quality-preserving resource-improvement claim. Returned p-values are raw;
/// use holm_adjust across every preregistered claim in the family before
/// accepting a confirmatory suite-level result.
pub fn analyze_paired_advantage(
    observations: &[PairedBenchmarkObservation],
    config: PairedAdvantageConfig,
) -> Result<PairedAdvantageReport, PairedAdvantageError> {
    if observations.is_empty() {
        return Err(PairedAdvantageError::EmptyObservations);
    }
    if observations.len() < 2 {
        return Err(PairedAdvantageError::InsufficientIndependentBlocks);
    }
    if !config.min_relative_resource_reduction.is_finite()
        || config.min_relative_resource_reduction < 0.0
        || config.min_relative_resource_reduction >= 1.0
        || !config.quality_non_inferiority_margin.is_finite()
        || config.quality_non_inferiority_margin < 0.0
        || !config.resource_log_ratio_tie_tolerance.is_finite()
        || config.resource_log_ratio_tie_tolerance < 0.0
        || !config.quality_difference_tie_tolerance.is_finite()
        || config.quality_difference_tie_tolerance < 0.0
    {
        return Err(PairedAdvantageError::InvalidConfig);
    }

    let mut block_ids = HashSet::with_capacity(observations.len());
    let mut workload_ids = HashSet::with_capacity(observations.len());
    let contract = observations[0].comparison_contract_fingerprint.trim();
    if contract.is_empty() {
        return Err(PairedAdvantageError::EmptyIdentifier);
    }

    let mut quality_differences = Vec::with_capacity(observations.len());
    let mut resource_log_ratios = Vec::with_capacity(observations.len());

    for observation in observations {
        if observation.independent_block_id.trim().is_empty()
            || observation.workload_fingerprint.trim().is_empty()
            || observation
                .comparison_contract_fingerprint
                .trim()
                .is_empty()
        {
            return Err(PairedAdvantageError::EmptyIdentifier);
        }
        if !block_ids.insert(observation.independent_block_id.as_str()) {
            return Err(PairedAdvantageError::DuplicateBlockId);
        }
        if !workload_ids.insert(observation.workload_fingerprint.as_str()) {
            return Err(PairedAdvantageError::DuplicateWorkloadFingerprint);
        }
        if observation.comparison_contract_fingerprint != contract {
            return Err(PairedAdvantageError::ContractMismatch);
        }

        let quantum_resource = observation.quantum_resource_per_accepted_answer;
        let classical_resource = observation.classical_resource_per_accepted_answer;
        if !quantum_resource.is_finite()
            || !classical_resource.is_finite()
            || quantum_resource <= 0.0
            || classical_resource <= 0.0
        {
            return Err(PairedAdvantageError::InvalidResource);
        }
        if !observation.quantum_quality.is_finite() || !observation.classical_quality.is_finite() {
            return Err(PairedAdvantageError::InvalidQuality);
        }

        let quality_difference = observation.quantum_quality - observation.classical_quality;
        if !quality_difference.is_finite() {
            return Err(PairedAdvantageError::InvalidQuality);
        }
        quality_differences.push(quality_difference);
        resource_log_ratios.push(classical_resource.ln() - quantum_resource.ln());
    }

    let quality_threshold = -config.quality_non_inferiority_margin;
    let resource_threshold = -(-config.min_relative_resource_reduction).ln_1p();
    if !(quality_threshold - config.quality_difference_tie_tolerance).is_finite()
        || !(quality_threshold + config.quality_difference_tie_tolerance).is_finite()
        || !(resource_threshold - config.resource_log_ratio_tie_tolerance).is_finite()
        || !(resource_threshold + config.resource_log_ratio_tie_tolerance).is_finite()
    {
        return Err(PairedAdvantageError::InvalidConfig);
    }
    let quality_non_inferiority = exact_margin_sign_test(
        &quality_differences,
        quality_threshold,
        config.quality_difference_tie_tolerance,
    );
    let resource_superiority = exact_margin_sign_test(
        &resource_log_ratios,
        resource_threshold,
        config.resource_log_ratio_tie_tolerance,
    );

    let composite_claim_p_value = quality_non_inferiority
        .p_value_one_sided
        .max(resource_superiority.p_value_one_sided);
    let median_quality_difference = median(&mut quality_differences);
    let median_log_classical_over_quantum_resource = median(&mut resource_log_ratios);
    let exponentiated = median_log_classical_over_quantum_resource.exp();

    Ok(PairedAdvantageReport {
        independent_blocks: observations.len(),
        median_quality_difference,
        median_log_classical_over_quantum_resource,
        median_classical_over_quantum_resource_ratio: exponentiated
            .is_finite()
            .then_some(exponentiated),
        quality_non_inferiority,
        resource_superiority,
        composite_claim_p_value,
        comparison_contract_fingerprint: contract.to_owned(),
    })
}

/// Apply Holm's step-down family-wise error correction.
///
/// The returned vector is in deterministic ascending raw-p order, with
/// hypothesis_id as the tie-breaker. Adjusted p-values are monotone and
/// bounded by one. The input alpha is a decision threshold, not an estimator.
pub fn holm_adjust(
    hypotheses: &[RawHypothesis],
    alpha: f64,
) -> Result<Vec<HolmAdjustedPValue>, PairedAdvantageError> {
    if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return Err(PairedAdvantageError::InvalidAlpha);
    }
    if hypotheses.is_empty() {
        return Err(PairedAdvantageError::EmptyObservations);
    }

    let mut ids = HashSet::with_capacity(hypotheses.len());
    for hypothesis in hypotheses {
        if hypothesis.hypothesis_id.trim().is_empty()
            || !ids.insert(hypothesis.hypothesis_id.as_str())
        {
            return Err(PairedAdvantageError::InvalidHypothesisId);
        }
        if !hypothesis.p_value.is_finite() || !(0.0..=1.0).contains(&hypothesis.p_value) {
            return Err(PairedAdvantageError::InvalidPValue);
        }
    }

    let mut order: Vec<usize> = (0..hypotheses.len()).collect();
    order.sort_by(|&left, &right| {
        hypotheses[left]
            .p_value
            .total_cmp(&hypotheses[right].p_value)
            .then_with(|| {
                hypotheses[left]
                    .hypothesis_id
                    .cmp(&hypotheses[right].hypothesis_id)
            })
    });

    let mut previous_adjusted: f64 = 0.0;
    let mut adjusted = Vec::with_capacity(hypotheses.len());
    for (offset, &index) in order.iter().enumerate() {
        let rank = offset + 1;
        let multiplier = hypotheses.len() - offset;
        let candidate = (hypotheses[index].p_value * multiplier as f64).min(1.0);
        let adjusted_p_value = previous_adjusted.max(candidate);
        previous_adjusted = adjusted_p_value;
        adjusted.push(HolmAdjustedPValue {
            hypothesis_id: hypotheses[index].hypothesis_id.clone(),
            raw_p_value: hypotheses[index].p_value,
            adjusted_p_value,
            rank,
            reject_at_alpha: adjusted_p_value <= alpha,
        });
    }
    Ok(adjusted)
}

/// Exact one-sided sign-test p-value for wins positive signs among n
/// non-tied pairs. Computes the fair-binomial tail without factorials.
pub fn exact_one_sided_sign_test_p_value(wins: usize, n: usize) -> f64 {
    if n == 0 || wins == 0 {
        return 1.0;
    }
    if wins > n {
        return 0.0;
    }
    let tail = if wins <= n / 2 {
        1.0 - lower_binomial_tail(n, wins - 1)
    } else {
        lower_binomial_tail(n, n - wins)
    };
    // A positive floor is conservative when the true tail underflows f64.
    tail.clamp(f64::MIN_POSITIVE, 1.0)
}

fn exact_margin_sign_test(values: &[f64], margin: f64, tolerance: f64) -> ExactSignTest {
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut ties = 0usize;
    for &value in values {
        if value > margin + tolerance {
            wins += 1;
        } else if value < margin - tolerance {
            losses += 1;
        } else {
            ties += 1;
        }
    }
    // Ties remain in the denominator as non-wins. This is conservative when
    // the null permits probability mass exactly at the practical margin.
    let effective_n = values.len();
    ExactSignTest {
        wins,
        losses,
        ties,
        effective_n,
        p_value_one_sided: exact_one_sided_sign_test_p_value(wins, effective_n),
    }
}

fn lower_binomial_tail(n: usize, maximum: usize) -> f64 {
    let mode = n / 2;
    if maximum >= mode {
        return 0.5;
    }
    let mut weight = 1.0f64;
    let mut lower_sum = 0.0f64;
    let mut tail_sum = 0.0f64;
    for k in (0..=mode).rev() {
        lower_sum += weight;
        if k <= maximum {
            tail_sum += weight;
        }
        if k > 0 {
            weight *= k as f64 / (n - k + 1) as f64;
        }
    }
    let total = if n % 2 == 0 {
        2.0 * lower_sum - 1.0
    } else {
        2.0 * lower_sum
    };
    tail_sum / total
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        let lower = values[middle - 1];
        let upper = values[middle];
        lower + (upper - lower) * 0.5
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_observations() -> Vec<PairedBenchmarkObservation> {
        (0..16)
            .map(|index| {
                let quantum_wins = index < 13;
                let quality_passes = index < 14;
                PairedBenchmarkObservation {
                    independent_block_id: format!("block-{index:02}"),
                    workload_fingerprint: format!("workload-{index:02}"),
                    comparison_contract_fingerprint: "sha256:shared-contract".to_owned(),
                    quantum_resource_per_accepted_answer: if quantum_wins { 8.0 } else { 10.0 },
                    classical_resource_per_accepted_answer: if quantum_wins { 10.0 } else { 8.0 },
                    quantum_quality: if quality_passes { 0.995 } else { 0.90 },
                    classical_quality: 0.99,
                }
            })
            .collect()
    }

    fn fixture_config() -> PairedAdvantageConfig {
        PairedAdvantageConfig {
            min_relative_resource_reduction: 0.10,
            quality_non_inferiority_margin: 0.02,
            resource_log_ratio_tie_tolerance: 1e-12,
            quality_difference_tie_tolerance: 1e-12,
        }
    }

    #[test]
    fn ties_remain_in_the_exact_test_denominator() {
        let report = exact_margin_sign_test(&[1.0, 0.0, -1.0], 0.0, 0.0);
        assert_eq!(report.wins, 1);
        assert_eq!(report.losses, 1);
        assert_eq!(report.ties, 1);
        assert_eq!(report.effective_n, 3);
        assert!((report.p_value_one_sided - 0.875).abs() < 1e-12);
    }

    #[test]
    fn a_single_independent_block_is_not_enough() {
        let observation = fixture_observations().remove(0);
        assert_eq!(
            analyze_paired_advantage(&[observation], fixture_config()).unwrap_err(),
            PairedAdvantageError::InsufficientIndependentBlocks
        );
    }

    #[test]
    fn exact_sign_test_matches_hand_computed_tails() {
        assert!((exact_one_sided_sign_test_p_value(4, 4) - 0.0625).abs() < 1e-12);
        assert!((exact_one_sided_sign_test_p_value(3, 4) - 0.3125).abs() < 1e-12);
        assert!((exact_one_sided_sign_test_p_value(0, 4) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn comparison_uses_independent_blocks_and_practical_thresholds() {
        let report = analyze_paired_advantage(&fixture_observations(), fixture_config()).unwrap();
        assert_eq!(report.independent_blocks, 16);
        assert_eq!(report.resource_superiority.wins, 13);
        assert_eq!(report.resource_superiority.losses, 3);
        assert_eq!(report.quality_non_inferiority.wins, 14);
        assert_eq!(report.quality_non_inferiority.losses, 2);
        assert!((report.composite_claim_p_value - 697.0 / 65_536.0).abs() < 1e-12);
        assert!(
            (report.median_classical_over_quantum_resource_ratio.unwrap() - 1.25).abs() < 1e-12
        );
        assert!((report.resource_superiority.p_value_one_sided - 697.0 / 65_536.0).abs() < 1e-12);
        assert!(
            (report.quality_non_inferiority.p_value_one_sided - 137.0 / 65_536.0).abs() < 1e-12
        );
    }

    #[test]
    fn exact_fractional_reduction_boundary_is_a_tie_not_a_win() {
        let mut observations = fixture_observations();
        for observation in &mut observations {
            observation.quantum_resource_per_accepted_answer = 90.0;
            observation.classical_resource_per_accepted_answer = 100.0;
        }
        let report = analyze_paired_advantage(&observations, fixture_config()).unwrap();
        assert_eq!(report.resource_superiority.wins, 0);
        assert_eq!(report.resource_superiority.ties, 16);
    }

    #[test]
    fn resource_wins_below_the_practical_margin_are_not_wins() {
        let mut observations = fixture_observations();
        for observation in observations.iter_mut().take(13) {
            observation.classical_resource_per_accepted_answer = 8.5;
        }
        let report = analyze_paired_advantage(&observations, fixture_config()).unwrap();
        assert_eq!(report.resource_superiority.wins, 0);
        assert_eq!(report.resource_superiority.losses, 16);
    }

    #[test]
    fn permutation_of_independent_blocks_does_not_change_result() {
        let mut forward = fixture_observations();
        let mut reverse = forward.clone();
        reverse.reverse();
        let a = analyze_paired_advantage(&forward, fixture_config()).unwrap();
        let b = analyze_paired_advantage(&reverse, fixture_config()).unwrap();
        assert_eq!(a, b);
        forward.rotate_left(5);
        let c = analyze_paired_advantage(&forward, fixture_config()).unwrap();
        assert_eq!(a, c);
    }

    #[test]
    fn duplicate_block_ids_are_rejected() {
        let mut observations = fixture_observations();
        observations[1].independent_block_id = observations[0].independent_block_id.clone();
        assert_eq!(
            analyze_paired_advantage(&observations, fixture_config()).unwrap_err(),
            PairedAdvantageError::DuplicateBlockId
        );
    }

    #[test]
    fn repeated_workload_instances_cannot_be_pseudoreplicated() {
        let mut observations = fixture_observations();
        observations[1].workload_fingerprint = observations[0].workload_fingerprint.clone();
        assert_eq!(
            analyze_paired_advantage(&observations, fixture_config()).unwrap_err(),
            PairedAdvantageError::DuplicateWorkloadFingerprint
        );
    }

    #[test]
    fn mismatched_contracts_are_rejected() {
        let mut observations = fixture_observations();
        observations[1].comparison_contract_fingerprint = "sha256:other".to_owned();
        assert_eq!(
            analyze_paired_advantage(&observations, fixture_config()).unwrap_err(),
            PairedAdvantageError::ContractMismatch
        );
    }

    #[test]
    fn zero_negative_and_non_finite_resources_are_rejected() {
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut observations = fixture_observations();
            observations[0].quantum_resource_per_accepted_answer = value;
            assert_eq!(
                analyze_paired_advantage(&observations, fixture_config()).unwrap_err(),
                PairedAdvantageError::InvalidResource
            );
        }
    }

    #[test]
    fn non_finite_quality_is_rejected() {
        let mut observations = fixture_observations();
        observations[0].quantum_quality = f64::NAN;
        assert_eq!(
            analyze_paired_advantage(&observations, fixture_config()).unwrap_err(),
            PairedAdvantageError::InvalidQuality
        );
    }

    #[test]
    fn invalid_margins_are_rejected() {
        let mut config = fixture_config();
        config.min_relative_resource_reduction = -0.1;
        assert_eq!(
            analyze_paired_advantage(&fixture_observations(), config).unwrap_err(),
            PairedAdvantageError::InvalidConfig
        );
    }

    #[test]
    fn holm_adjustment_matches_hand_computed_family() {
        let hypotheses = [
            RawHypothesis {
                hypothesis_id: "h3".to_owned(),
                p_value: 0.04,
            },
            RawHypothesis {
                hypothesis_id: "h1".to_owned(),
                p_value: 0.003,
            },
            RawHypothesis {
                hypothesis_id: "h4".to_owned(),
                p_value: 0.20,
            },
            RawHypothesis {
                hypothesis_id: "h2".to_owned(),
                p_value: 0.01,
            },
        ];
        let adjusted = holm_adjust(&hypotheses, 0.05).unwrap();
        assert_eq!(
            adjusted
                .iter()
                .map(|item| item.hypothesis_id.as_str())
                .collect::<Vec<_>>(),
            ["h1", "h2", "h3", "h4"]
        );
        let expected = [0.012, 0.03, 0.08, 0.20];
        for (item, expected) in adjusted.iter().zip(expected) {
            assert!((item.adjusted_p_value - expected).abs() < 1e-12);
            assert_eq!(item.reject_at_alpha, expected <= 0.05);
        }
    }

    #[test]
    fn holm_rejects_duplicate_ids_and_invalid_p_values() {
        let duplicate = [
            RawHypothesis {
                hypothesis_id: "same".to_owned(),
                p_value: 0.01,
            },
            RawHypothesis {
                hypothesis_id: "same".to_owned(),
                p_value: 0.02,
            },
        ];
        assert_eq!(holm_adjust(&duplicate, 0.05).unwrap_err(), PairedAdvantageError::InvalidHypothesisId);
        let invalid = [RawHypothesis { hypothesis_id: "h".to_owned(), p_value: f64::NAN }];
        assert_eq!(holm_adjust(&invalid, 0.05).unwrap_err(), PairedAdvantageError::InvalidPValue);
    }
}
