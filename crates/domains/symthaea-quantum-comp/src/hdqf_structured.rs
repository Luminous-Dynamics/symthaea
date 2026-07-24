//! Generalized-birthday and resonator-network HDQF baselines.
//!
//! The four-list search is a bounded adaptation of Wagner's generalized
//! birthday construction. Its report explicitly distinguishes ordinary
//! correctness from the stronger random/extensible-list assumptions required
//! for the published asymptotic speedup.
//!
//! The resonator implementation follows the Multiply-Add-Permute update shape:
//! unbind the target by all other estimates, apply codebook outer-product
//! cleanup, then take the bipolar sign. It uses synchronous factor updates.
//!
//! References:
//! - D. Wagner, "A Generalized Birthday Problem," CRYPTO 2002.
//! - E. P. Frady et al., "Resonator Networks, 1," Neural Computation 2020.

use std::collections::{BTreeMap, BTreeSet};

use crate::classical_hdc::BinaryHypervector;
use crate::errors::{QuantumCompError, Result};
use crate::hdqf_baselines::HdqfBaselineDisposition;
use crate::hdqf_problem::{HdqfInstanceFamily, HdqfProblemInstance};

/// Configuration for bounded four-list Wagner search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wagner4Config {
    /// Low bits constrained in each first-level join.
    pub low_bits: usize,
    /// Number of deterministic alpha buckets examined.
    pub alpha_trials: usize,
    /// Seed controlling a full-cycle permutation of alpha buckets.
    pub alpha_seed: u64,
    /// Maximum combined intermediate pair entries held for one alpha.
    pub max_intermediate_entries: usize,
}

impl Wagner4Config {
    /// Creates a conservative configuration from `D` and `N`.
    pub fn for_instance(instance: &HdqfProblemInstance, alpha_seed: u64) -> Self {
        let log_n = usize::BITS as usize - 1 - instance.codebook_size().leading_zeros() as usize;
        let low_bits = log_n.min(instance.dimension().div_ceil(3)).max(1);
        let alpha_space = 1usize.checked_shl(low_bits as u32).unwrap_or(usize::MAX);
        Self {
            low_bits,
            alpha_trials: alpha_space,
            alpha_seed,
            max_intermediate_entries: usize::MAX,
        }
    }
}

/// Logical work counters for the bounded Wagner adaptation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Wagner4Metrics {
    /// Alpha buckets actually examined.
    pub alpha_buckets_examined: usize,
    /// Candidate pairs emitted after low-bit joins.
    pub partial_pairs_generated: usize,
    /// Full-vector equality lookups.
    pub final_join_lookups: usize,
    /// Individual codeword reads, including target adjustment.
    pub codebook_accesses: usize,
    /// Packed-word XOR operations.
    pub binding_word_xors: usize,
    /// Largest combined pair-list size for one alpha.
    pub peak_intermediate_entries: usize,
    /// Returned full candidates conventionally verified.
    pub solution_candidates_verified: usize,
}

/// Report from bounded four-list generalized-birthday search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wagner4Report {
    /// Completion, non-convergence, applicability, or censoring status.
    pub disposition: HdqfBaselineDisposition,
    /// Explanation for non-completed outcomes.
    pub reason: Option<&'static str>,
    /// Whether all alpha buckets were examined, making the search exact.
    pub complete_alpha_scan: bool,
    /// Whether Wagner's random/extensible-list asymptotic premises hold.
    pub asymptotic_assumptions_satisfied: bool,
    /// Alpha bucket count `2^low_bits`.
    pub alpha_space: usize,
    /// Number of exact solutions when a complete scan was performed.
    pub exact_solution_count: Option<usize>,
    /// First discovered solution, lexicographically minimized across examined buckets.
    pub returned_indices: Option<Vec<usize>>,
    /// Whether the returned solution is the planted tuple.
    pub returned_planted: Option<bool>,
    /// Logical work counters.
    pub metrics: Wagner4Metrics,
}

#[derive(Debug, Clone)]
struct PairEntry {
    product: Vec<u64>,
    indices: [usize; 2],
}

#[derive(Debug, Clone)]
struct PairBucket {
    count: usize,
    first_indices: [usize; 2],
}

/// Runs a bounded four-list Wagner-style search for an exact target.
pub fn wagner4_factorization(
    instance: &HdqfProblemInstance,
    config: Wagner4Config,
) -> Result<Wagner4Report> {
    if instance.factor_count() != 4 {
        return Ok(wagner_incomplete(
            HdqfBaselineDisposition::NotApplicable,
            "Wagner4 requires exactly four factor codebooks",
        ));
    }
    if instance.is_noisy() {
        return Ok(wagner_incomplete(
            HdqfBaselineDisposition::NotApplicable,
            "Wagner4 baseline is exact-only",
        ));
    }
    if config.low_bits == 0
        || config.low_bits > instance.dimension()
        || config.low_bits >= usize::BITS as usize
    {
        return Err(QuantumCompError::InvalidConfig(
            "Wagner4 low_bits must fit the target and usize",
        ));
    }
    let alpha_space = 1usize << config.low_bits;
    if config.alpha_trials == 0 {
        return Err(QuantumCompError::InvalidConfig(
            "Wagner4 requires at least one alpha trial",
        ));
    }
    let trials = config.alpha_trials.min(alpha_space);
    let complete_alpha_scan = trials == alpha_space;
    let required_low_bits = instance.dimension().div_ceil(3);
    let asymptotic_assumptions_satisfied = instance.family == HdqfInstanceFamily::Random
        && config.low_bits >= required_low_bits
        && instance.codebook_size() >= alpha_space;

    let words = instance.target.word_len();
    let adjusted_fourth = instance.codebooks[3]
        .iter()
        .map(|vector| vector.bind_xor(&instance.target))
        .collect::<Result<Vec<_>>>()?;
    let grouped = [
        group_by_low_bits(&instance.codebooks[0], config.low_bits),
        group_by_low_bits(&instance.codebooks[1], config.low_bits),
        group_by_low_bits(&instance.codebooks[2], config.low_bits),
        group_by_low_bits(&adjusted_fourth, config.low_bits),
    ];

    let mask = alpha_space - 1;
    let start = config.alpha_seed as usize & mask;
    let mut step = (config.alpha_seed.rotate_left(29) as usize | 1) & mask;
    if step == 0 {
        step = 1;
    }

    let mut metrics = Wagner4Metrics {
        codebook_accesses: instance.codebook_size(),
        binding_word_xors: instance.codebook_size() * words,
        ..Wagner4Metrics::default()
    };
    let mut solution_count = 0usize;
    let mut returned_indices: Option<Vec<usize>> = None;

    for trial in 0..trials {
        let alpha = start.wrapping_add(trial.wrapping_mul(step)) & mask;
        metrics.alpha_buckets_examined += 1;
        let left_count = joined_pair_count(&grouped[0], &grouped[1], alpha)?;
        let right_count = joined_pair_count(&grouped[2], &grouped[3], alpha)?;
        let combined =
            left_count
                .checked_add(right_count)
                .ok_or(QuantumCompError::InvalidConfig(
                    "Wagner4 intermediate count overflows usize",
                ))?;
        metrics.peak_intermediate_entries = metrics.peak_intermediate_entries.max(combined);
        if combined > config.max_intermediate_entries {
            return Ok(Wagner4Report {
                disposition: HdqfBaselineDisposition::ResourceCensored,
                reason: Some("Wagner4 intermediate-entry ceiling exceeded"),
                complete_alpha_scan,
                asymptotic_assumptions_satisfied,
                alpha_space,
                exact_solution_count: None,
                returned_indices: None,
                returned_planted: None,
                metrics,
            });
        }
        let left_pairs = joined_pairs(
            &instance.codebooks[0],
            &instance.codebooks[1],
            &grouped[0],
            &grouped[1],
            alpha,
            &mut metrics,
        )?;
        let right_pairs = joined_pairs(
            &instance.codebooks[2],
            &adjusted_fourth,
            &grouped[2],
            &grouped[3],
            alpha,
            &mut metrics,
        )?;

        let mut right_map: BTreeMap<Vec<u64>, PairBucket> = BTreeMap::new();
        for pair in right_pairs {
            right_map
                .entry(pair.product)
                .and_modify(|bucket| {
                    bucket.count += 1;
                    if pair.indices < bucket.first_indices {
                        bucket.first_indices = pair.indices;
                    }
                })
                .or_insert(PairBucket {
                    count: 1,
                    first_indices: pair.indices,
                });
        }
        for pair in left_pairs {
            metrics.final_join_lookups += 1;
            if let Some(bucket) = right_map.get(&pair.product) {
                solution_count = solution_count.checked_add(bucket.count).ok_or(
                    QuantumCompError::InvalidConfig("Wagner4 solution count overflows usize"),
                )?;
                let candidate = vec![
                    pair.indices[0],
                    pair.indices[1],
                    bucket.first_indices[0],
                    bucket.first_indices[1],
                ];
                if returned_indices
                    .as_ref()
                    .is_none_or(|current| candidate < *current)
                {
                    returned_indices = Some(candidate);
                }
            }
        }
    }

    if let Some(indices) = &returned_indices {
        let product = instance.product_for_indices(indices)?;
        metrics.codebook_accesses += 4;
        metrics.binding_word_xors += 4 * words;
        metrics.solution_candidates_verified += 1;
        if product != instance.target {
            return Err(QuantumCompError::InvalidConfig(
                "Wagner4 returned an invalid exact factorization",
            ));
        }
    }

    let disposition = if returned_indices.is_some() {
        HdqfBaselineDisposition::Completed
    } else {
        HdqfBaselineDisposition::NonConverged
    };
    let reason = if returned_indices.is_none() {
        Some("no exact factorization appeared in the examined alpha buckets")
    } else {
        None
    };
    let returned_planted = returned_indices
        .as_ref()
        .map(|indices| indices == &instance.planted_indices);
    Ok(Wagner4Report {
        disposition,
        reason,
        complete_alpha_scan,
        asymptotic_assumptions_satisfied,
        alpha_space,
        exact_solution_count: complete_alpha_scan.then_some(solution_count),
        returned_indices,
        returned_planted,
        metrics,
    })
}

fn wagner_incomplete(disposition: HdqfBaselineDisposition, reason: &'static str) -> Wagner4Report {
    Wagner4Report {
        disposition,
        reason: Some(reason),
        complete_alpha_scan: false,
        asymptotic_assumptions_satisfied: false,
        alpha_space: 0,
        exact_solution_count: None,
        returned_indices: None,
        returned_planted: None,
        metrics: Wagner4Metrics::default(),
    }
}

fn group_by_low_bits(
    codebook: &[BinaryHypervector],
    low_bits: usize,
) -> BTreeMap<usize, Vec<usize>> {
    let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
    for (index, vector) in codebook.iter().enumerate() {
        grouped
            .entry(low_signature(vector, low_bits))
            .or_default()
            .push(index);
    }
    grouped
}

fn joined_pairs(
    left: &[BinaryHypervector],
    right: &[BinaryHypervector],
    left_groups: &BTreeMap<usize, Vec<usize>>,
    right_groups: &BTreeMap<usize, Vec<usize>>,
    alpha: usize,
    metrics: &mut Wagner4Metrics,
) -> Result<Vec<PairEntry>> {
    let words = left.first().map_or(0, BinaryHypervector::word_len);
    let mut pairs = Vec::new();
    for (&signature, left_indices) in left_groups {
        let Some(right_indices) = right_groups.get(&(signature ^ alpha)) else {
            continue;
        };
        for &left_index in left_indices {
            for &right_index in right_indices {
                let product = left[left_index].bind_xor(&right[right_index])?;
                pairs.push(PairEntry {
                    product: product.words().to_vec(),
                    indices: [left_index, right_index],
                });
                metrics.partial_pairs_generated += 1;
                metrics.codebook_accesses += 2;
                metrics.binding_word_xors += words;
            }
        }
    }
    Ok(pairs)
}

fn joined_pair_count(
    left_groups: &BTreeMap<usize, Vec<usize>>,
    right_groups: &BTreeMap<usize, Vec<usize>>,
    alpha: usize,
) -> Result<usize> {
    let mut count = 0usize;
    for (&signature, left_indices) in left_groups {
        let Some(right_indices) = right_groups.get(&(signature ^ alpha)) else {
            continue;
        };
        let pairs = left_indices.len().checked_mul(right_indices.len()).ok_or(
            QuantumCompError::InvalidConfig("Wagner4 pair count overflows usize"),
        )?;
        count = count
            .checked_add(pairs)
            .ok_or(QuantumCompError::InvalidConfig(
                "Wagner4 pair count overflows usize",
            ))?;
    }
    Ok(count)
}

fn low_signature(vector: &BinaryHypervector, low_bits: usize) -> usize {
    let mask = (1u64 << low_bits) - 1;
    (vector.words()[0] & mask) as usize
}

/// Initialization used by a resonator restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResonatorInitialization {
    /// Sign of the sum of every codeword in each factor codebook.
    CodebookSuperposition,
    /// Deterministic random bipolar estimates.
    Random,
}

/// Resonator-network execution configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResonatorConfig {
    /// Maximum synchronous updates per restart.
    pub max_iterations: usize,
    /// Number of restarts. Restart zero uses `initialization`; later restarts are random.
    pub restarts: usize,
    /// Initialization for restart zero.
    pub initialization: ResonatorInitialization,
    /// Seed for tie-independent random restarts.
    pub seed: u64,
    /// Stop all restarts after an exact target product is read out.
    pub stop_on_exact: bool,
}

/// Aggregate resonator work counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResonatorMetrics {
    /// Restarts entered.
    pub restarts_executed: usize,
    /// Synchronous network updates.
    pub iterations: usize,
    /// Factor cleanup-memory applications.
    pub cleanup_calls: usize,
    /// Codeword/input bipolar dot products.
    pub codebook_dot_products: usize,
    /// Coordinate-level multiply/accumulate operations in cleanup.
    pub bipolar_coordinate_ops: usize,
    /// Packed-word XORs for unbinding and candidate verification.
    pub binding_word_xors: usize,
    /// Codeword comparisons during nearest-codeword readout.
    pub readout_comparisons: usize,
    /// Fixed points encountered.
    pub fixed_points: usize,
    /// Nontrivial cycles encountered.
    pub cycles: usize,
}

/// Termination state associated with the best resonator candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResonatorTermination {
    /// An exact target product was recovered.
    ExactSolution,
    /// Dynamics reached a fixed point without an exact readout.
    FixedPoint,
    /// Dynamics revisited a prior state.
    Cycle,
    /// The iteration budget ended.
    IterationLimit,
}

/// Result of deterministic resonator-network factor recovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResonatorReport {
    /// Best tuple read out across all iterations and restarts.
    pub returned_indices: Vec<usize>,
    /// Hamming distance between its product and the observed target.
    pub achieved_hamming_distance: usize,
    /// Whether the tuple equals the planted tuple.
    pub returned_planted: bool,
    /// Termination associated with the restart producing the best tuple.
    pub termination: ResonatorTermination,
    /// Aggregate work counters.
    pub metrics: ResonatorMetrics,
}

/// Runs the bipolar outer-product resonator network.
pub fn resonator_factorization(
    instance: &HdqfProblemInstance,
    config: ResonatorConfig,
) -> Result<ResonatorReport> {
    if config.max_iterations == 0 || config.restarts == 0 {
        return Err(QuantumCompError::InvalidConfig(
            "resonator requires positive iterations and restarts",
        ));
    }
    let factors = instance.factor_count();
    let words = instance.target.word_len();
    let mut metrics = ResonatorMetrics::default();
    let mut best: Option<(usize, Vec<usize>, ResonatorTermination)> = None;

    'restarts: for restart in 0..config.restarts {
        metrics.restarts_executed += 1;
        let mut estimates = initialize_estimates(instance, config, restart)?;
        let mut seen = BTreeSet::new();
        seen.insert(estimate_signature(&estimates));

        for _ in 0..config.max_iterations {
            metrics.iterations += 1;
            let mut next = Vec::with_capacity(factors);
            for factor in 0..factors {
                let mut input = instance.target.clone();
                for (other, estimate) in estimates.iter().enumerate() {
                    if other != factor {
                        input = input.bind_xor(estimate)?;
                        metrics.binding_word_xors += words;
                    }
                }
                next.push(outer_product_cleanup(
                    &instance.codebooks[factor],
                    &input,
                    &mut metrics,
                )?);
            }

            let indices = readout_indices(instance, &next, &mut metrics)?;
            let product = instance.product_for_indices(&indices)?;
            metrics.binding_word_xors += factors * words;
            let distance = product.hamming_distance(&instance.target)?;

            let fixed = next == estimates;
            let signature = estimate_signature(&next);
            let cycle = !fixed && !seen.insert(signature);
            let termination = if distance == 0 {
                ResonatorTermination::ExactSolution
            } else if fixed {
                metrics.fixed_points += 1;
                ResonatorTermination::FixedPoint
            } else if cycle {
                metrics.cycles += 1;
                ResonatorTermination::Cycle
            } else {
                ResonatorTermination::IterationLimit
            };

            let replace = best
                .as_ref()
                .is_none_or(|(best_distance, best_indices, _)| {
                    distance < *best_distance
                        || (distance == *best_distance && indices < *best_indices)
                });
            if replace {
                best = Some((distance, indices, termination));
            }

            estimates = next;
            if distance == 0 && config.stop_on_exact {
                break 'restarts;
            }
            if fixed || cycle {
                break;
            }
        }
    }

    let (achieved_hamming_distance, returned_indices, termination) = best.ok_or(
        QuantumCompError::InvalidConfig("resonator produced no candidate readout"),
    )?;
    let returned_planted = returned_indices == instance.planted_indices;
    Ok(ResonatorReport {
        returned_indices,
        achieved_hamming_distance,
        returned_planted,
        termination,
        metrics,
    })
}

fn initialize_estimates(
    instance: &HdqfProblemInstance,
    config: ResonatorConfig,
    restart: usize,
) -> Result<Vec<BinaryHypervector>> {
    let random = restart > 0 || config.initialization == ResonatorInitialization::Random;
    if random {
        (0..instance.factor_count())
            .map(|factor| {
                let seed = config
                    .seed
                    .wrapping_add((restart as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    .wrapping_add(factor as u64);
                BinaryHypervector::random(instance.dimension(), seed)
            })
            .collect()
    } else {
        instance
            .codebooks
            .iter()
            .map(|codebook| superposition_sign(codebook))
            .collect()
    }
}

fn superposition_sign(codebook: &[BinaryHypervector]) -> Result<BinaryHypervector> {
    let dimension = codebook
        .first()
        .ok_or(QuantumCompError::InvalidConfig(
            "resonator codebook is empty",
        ))?
        .dimension();
    let mut output = BinaryHypervector::zeros(dimension)?;
    for bit in 0..dimension {
        let negative = codebook
            .iter()
            .filter(|vector| vector.bit(bit).unwrap_or(false))
            .count();
        if negative * 2 > codebook.len() {
            output.set_bit(bit, true)?;
        }
    }
    Ok(output)
}

fn outer_product_cleanup(
    codebook: &[BinaryHypervector],
    input: &BinaryHypervector,
    metrics: &mut ResonatorMetrics,
) -> Result<BinaryHypervector> {
    let dimension = input.dimension();
    let mut scores = Vec::with_capacity(codebook.len());
    for codeword in codebook {
        let distance = codeword.hamming_distance(input)?;
        scores.push(dimension as i128 - 2 * distance as i128);
        metrics.codebook_dot_products += 1;
        metrics.bipolar_coordinate_ops += dimension;
    }

    let mut output = BinaryHypervector::zeros(dimension)?;
    for bit in 0..dimension {
        let mut accumulator = 0i128;
        for (codeword, score) in codebook.iter().zip(&scores) {
            let sign = if codeword.bit(bit).unwrap_or(false) {
                -1i128
            } else {
                1i128
            };
            accumulator += sign * score;
            metrics.bipolar_coordinate_ops += 1;
        }
        if accumulator < 0 {
            output.set_bit(bit, true)?;
        }
    }
    metrics.cleanup_calls += 1;
    Ok(output)
}

fn readout_indices(
    instance: &HdqfProblemInstance,
    estimates: &[BinaryHypervector],
    metrics: &mut ResonatorMetrics,
) -> Result<Vec<usize>> {
    let mut indices = Vec::with_capacity(instance.factor_count());
    for (codebook, estimate) in instance.codebooks.iter().zip(estimates) {
        let mut best = (usize::MAX, 0usize);
        for (index, codeword) in codebook.iter().enumerate() {
            let distance = codeword.hamming_distance(estimate)?;
            metrics.readout_comparisons += 1;
            if distance < best.0 {
                best = (distance, index);
            }
        }
        indices.push(best.1);
    }
    Ok(indices)
}

fn estimate_signature(estimates: &[BinaryHypervector]) -> Vec<u64> {
    estimates
        .iter()
        .flat_map(|estimate| estimate.words().iter().copied())
        .collect()
}
