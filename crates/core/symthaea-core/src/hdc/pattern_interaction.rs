// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Operational tests for interactions in an abstract pattern space.
//!
//! "Platonic space" is used as a research shorthand for a space of abstract
//! patterns and relations, independent of a particular physical embodiment.
//! This module makes no metaphysical claim. It tests whether structural
//! operations have measurable, falsifiable properties.
//!
//! The bipolar representation keeps this benchmark auditable: binding is
//! element-wise multiplication, unbinding is the same operation, permutation
//! encodes order, bundling is majority vote with seeded tie-breaking, and
//! similarity is normalized bipolar overlap.
//!
//! These are controlled HDC experiments, not evidence of biological cognition,
//! agency, or phenomenal consciousness.

use std::error::Error;
use std::fmt;

const CLEANUP_BETA: f64 = 12.0;
const CLEANUP_STEPS: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternSpaceError {
    EmptyVector,
    DimensionMismatch { expected: usize, actual: usize },
    InvalidBipolarValue { index: usize, value: i8 },
    EmptyBundle,
    InvalidNoiseFraction,
    InvalidPermutation,
    DuplicateLabel(String),
    InvalidConfiguration(&'static str),
}

impl fmt::Display for PatternSpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVector => write!(f, "pattern vectors must not be empty"),
            Self::DimensionMismatch { expected, actual } =>
                write!(f, "pattern dimension mismatch: expected {expected}, got {actual}"),
            Self::InvalidBipolarValue { index, value } =>
                write!(f, "pattern value at index {index} must be -1 or 1, got {value}"),
            Self::EmptyBundle => write!(f, "cannot bundle an empty set of patterns"),
            Self::InvalidNoiseFraction =>
                write!(f, "noise fraction must be finite and in [0, 1]"),
            Self::InvalidPermutation =>
                write!(f, "basis mapping must be a permutation of all dimensions"),
            Self::DuplicateLabel(label) => write!(f, "pattern label already registered: {label}"),
            Self::InvalidConfiguration(message) =>
                write!(f, "invalid experiment configuration: {message}"),
        }
    }
}

impl Error for PatternSpaceError {}

/// Immutable bipolar vector used for abstract HDC pattern experiments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternVector {
    values: Vec<i8>,
}

impl PatternVector {
    /// Construct a vector, rejecting empty or non-bipolar input.
    pub fn new(values: Vec<i8>) -> Result<Self, PatternSpaceError> {
        if values.is_empty() {
            return Err(PatternSpaceError::EmptyVector);
        }
        for (index, &value) in values.iter().enumerate() {
            if value != -1 && value != 1 {
                return Err(PatternSpaceError::InvalidBipolarValue { index, value });
            }
        }
        Ok(Self { values })
    }

    /// Make a reproducible pseudorandom bipolar pattern.
    pub fn seeded_random(dimension: usize, seed: u64) -> Result<Self, PatternSpaceError> {
        if dimension == 0 {
            return Err(PatternSpaceError::EmptyVector);
        }
        let mut state = seed;
        let values = (0..dimension)
            .map(|_| if next_u64(&mut state) & 1 == 0 { -1 } else { 1 })
            .collect();
        Ok(Self { values })
    }

    pub fn dimension(&self) -> usize {
        self.values.len()
    }

    pub fn values(&self) -> &[i8] {
        &self.values
    }

    /// Normalized bipolar similarity in [-1, 1].
    /// 1 means identical, -1 means exact opposites, and values near 0 are
    /// typical of independently seeded high-dimensional patterns.
    pub fn similarity(&self, other: &Self) -> Result<f64, PatternSpaceError> {
        self.check_dimension(other)?;
        let matches = self.values.iter().zip(&other.values).filter(|(a, b)| a == b).count();
        Ok(2.0 * matches as f64 / self.dimension() as f64 - 1.0)
    }

    /// Bind two bipolar patterns. Binding is self-inverse for bipolar keys.
    pub fn bind(&self, other: &Self) -> Result<Self, PatternSpaceError> {
        self.check_dimension(other)?;
        Ok(Self {
            values: self.values.iter().zip(&other.values).map(|(a, b)| a * b).collect(),
        })
    }

    /// Unbind a bipolar key; equivalent to binding with that key again.
    pub fn unbind(&self, key: &Self) -> Result<Self, PatternSpaceError> {
        self.bind(key)
    }

    pub fn opposite(&self) -> Self {
        Self { values: self.values.iter().map(|value| -*value).collect() }
    }

    /// Circular right shift used to encode order in a sequence.
    pub fn permute(&self, shift: usize) -> Self {
        let dimension = self.dimension();
        let shift = shift % dimension;
        let mut values = vec![1; dimension];
        for (index, &value) in self.values.iter().enumerate() {
            values[(index + shift) % dimension] = value;
        }
        Self { values }
    }

    /// Re-express a pattern under an invertible change of basis.
    /// mapping[i] gives the source coordinate copied into output coordinate i.
    /// A valid mapping is a permutation; this preserves similarity and
    /// commutes with element-wise binding when applied consistently.
    pub fn reencode(&self, mapping: &[usize]) -> Result<Self, PatternSpaceError> {
        validate_permutation(mapping, self.dimension())?;
        Ok(Self { values: mapping.iter().map(|&index| self.values[index]).collect() })
    }

    /// Flip a seeded, exact number of dimensions to create controlled noise.
    pub fn with_noise(&self, fraction: f64, seed: u64) -> Result<Self, PatternSpaceError> {
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(PatternSpaceError::InvalidNoiseFraction);
        }
        let mut result = self.clone();
        let flips = (self.dimension() as f64 * fraction).round() as usize;
        let mut indices: Vec<usize> = (0..self.dimension()).collect();
        let mut state = seed;
        for index in 0..flips {
            let remaining = self.dimension() - index;
            let selected = index + (next_u64(&mut state) % remaining as u64) as usize;
            indices.swap(index, selected);
            result.values[indices[index]] *= -1;
        }
        Ok(result)
    }

    /// Bundle patterns by coordinate-wise majority vote.
    /// Exact ties use seed-derived bits instead of favoring a member or
    /// always choosing one polarity.
    pub fn bundle(patterns: &[Self], tie_seed: u64) -> Result<Self, PatternSpaceError> {
        let first = patterns.first().ok_or(PatternSpaceError::EmptyBundle)?;
        for pattern in &patterns[1..] {
            first.check_dimension(pattern)?;
        }
        let mut state = tie_seed;
        let mut values = Vec::with_capacity(first.dimension());
        for index in 0..first.dimension() {
            let vote: i32 = patterns.iter().map(|pattern| pattern.values[index] as i32).sum();
            let value = match vote.cmp(&0) {
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal =>
                    if next_u64(&mut state) & 1 == 0 { -1 } else { 1 },
            };
            values.push(value);
        }
        Ok(Self { values })
    }

    fn check_dimension(&self, other: &Self) -> Result<(), PatternSpaceError> {
        if self.dimension() != other.dimension() {
            return Err(PatternSpaceError::DimensionMismatch {
                expected: self.dimension(),
                actual: other.dimension(),
            });
        }
        Ok(())
    }
}

fn validate_permutation(mapping: &[usize], dimension: usize) -> Result<(), PatternSpaceError> {
    if mapping.len() != dimension {
        return Err(PatternSpaceError::InvalidPermutation);
    }
    let mut seen = vec![false; dimension];
    for &index in mapping {
        if index >= dimension || seen[index] {
            return Err(PatternSpaceError::InvalidPermutation);
        }
        seen[index] = true;
    }
    Ok(())
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    mix64(*state)
}

fn derived_seed(seed: u64, first: u64, second: u64) -> u64 {
    mix64(seed ^ mix64(first) ^ mix64(second.wrapping_add(0x9e37_79b9_7f4a_7c15)))
}

fn seeded_permutation(dimension: usize, seed: u64) -> Vec<usize> {
    let mut mapping: Vec<usize> = (0..dimension).collect();
    let mut state = seed;
    for index in (1..dimension).rev() {
        let selected = (next_u64(&mut state) % (index as u64 + 1)) as usize;
        mapping.swap(index, selected);
    }
    mapping
}

/// A labeled retrieval result from an abstract pattern space.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternMatch {
    pub label: String,
    pub similarity: f64,
}

/// Deliberately simple nearest-prototype recall baseline. It is not a learned
/// attractor and is kept so stronger mechanisms can be compared against it.
#[derive(Debug, Clone)]
pub struct PatternSpace {
    dimension: usize,
    prototypes: Vec<(String, PatternVector)>,
}

impl PatternSpace {
    pub fn new(dimension: usize) -> Result<Self, PatternSpaceError> {
        if dimension == 0 {
            return Err(PatternSpaceError::EmptyVector);
        }
        Ok(Self { dimension, prototypes: Vec::new() })
    }

    pub fn insert(
        &mut self,
        label: impl Into<String>,
        pattern: PatternVector,
    ) -> Result<(), PatternSpaceError> {
        if pattern.dimension() != self.dimension {
            return Err(PatternSpaceError::DimensionMismatch {
                expected: self.dimension,
                actual: pattern.dimension(),
            });
        }
        let label = label.into();
        if self.prototypes.iter().any(|(existing, _)| existing == &label) {
            return Err(PatternSpaceError::DuplicateLabel(label));
        }
        self.prototypes.push((label, pattern));
        Ok(())
    }

    pub fn rank(&self, query: &PatternVector) -> Result<Vec<PatternMatch>, PatternSpaceError> {
        if query.dimension() != self.dimension {
            return Err(PatternSpaceError::DimensionMismatch {
                expected: self.dimension,
                actual: query.dimension(),
            });
        }
        let mut ranked = self.prototypes.iter().map(|(label, pattern)| {
            Ok(PatternMatch { label: label.clone(), similarity: query.similarity(pattern)? })
        }).collect::<Result<Vec<_>, PatternSpaceError>>()?;
        ranked.sort_by(|left, right| {
            right.similarity.total_cmp(&left.similarity).then_with(|| left.label.cmp(&right.label))
        });
        Ok(ranked)
    }

    pub fn len(&self) -> usize {
        self.prototypes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prototypes.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct PatternInteractionConfig {
    pub dimension: usize,
    pub prototype_count: usize,
    pub trials: usize,
    pub noise_fractions: Vec<f64>,
    pub bundle_sizes: Vec<usize>,
}

impl Default for PatternInteractionConfig {
    fn default() -> Self {
        Self {
            dimension: 16_384,
            prototype_count: 64,
            trials: 16,
            noise_fractions: vec![0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50],
            bundle_sizes: vec![1, 3, 5, 9, 17, 33, 49],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoiseRetrievalResult {
    pub noise_fraction: f64,
    /// One-step nearest-prototype classifier accuracy.
    pub top1_accuracy: f64,
    /// Wilson 95% confidence interval for nearest-prototype accuracy.
    pub top1_accuracy_ci95: (f64, f64),
    /// Mean target similarity minus the best competing-prototype similarity.
    pub mean_similarity_margin: f64,
    /// Accuracy after iterative softmax-weighted pattern cleanup with bipolar projection.
    /// This is a Hopfield-inspired baseline, not an exact reproduction of a specific paper.
    pub cleanup_top1_accuracy: f64,
    /// Wilson 95% confidence interval for cleanup accuracy.
    pub cleanup_top1_accuracy_ci95: (f64, f64),
    /// Target-vs-best-distractor margin after projected associative cleanup.
    pub cleanup_mean_similarity_margin: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BundleCapacityResult {
    pub bundle_size: usize,
    pub mean_member_similarity: f64,
    pub mean_max_distractor_similarity: f64,
    /// Fraction of trials where the top-ranked prototype is a bundle member.
    pub member_recall_at_1: f64,
    /// Wilson 95% confidence interval for member recall.
    pub member_recall_at_1_ci95: (f64, f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssociativeCapacityResult {
    pub association_count: usize,
    pub mean_target_similarity: f64,
    /// Maximum over all non-target fillers in the candidate library.
    pub mean_max_distractor_similarity: f64,
    pub recall_at_1: f64,
    /// Wilson 95% confidence interval for raw retrieval recall.
    pub recall_at_1_ci95: (f64, f64),
    /// Recall after projected-softmax cleanup of the unbound query.
    pub cleanup_recall_at_1: f64,
    /// Wilson 95% confidence interval for cleanup retrieval recall.
    pub cleanup_recall_at_1_ci95: (f64, f64),
    pub cleanup_mean_target_similarity: f64,
    /// Maximum competing-filler similarity after projected-softmax cleanup.
    pub cleanup_mean_max_distractor_similarity: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatternInteractionReport {
    pub seed: u64,
    pub dimension: usize,
    pub prototype_count: usize,
    pub random_chance_baseline: f64,
    pub role_unbinding_similarity: f64,
    pub bundling_ablation_similarity: f64,
    pub binding_ablation_advantage: f64,
    pub antipattern_cancellation_abs_similarity: f64,
    pub sequence_order_similarity: f64,
    pub basis_equivariance_exact: bool,
    pub noise_results: Vec<NoiseRetrievalResult>,
    pub bundle_results: Vec<BundleCapacityResult>,
    pub associative_capacity_results: Vec<AssociativeCapacityResult>,
}

/// Run compositional, causal-ablation, basis-invariance, noise, and bundle
/// capacity experiments using fully seeded patterns.
pub fn run_pattern_interaction_suite(
    seed: u64,
    config: &PatternInteractionConfig,
) -> Result<PatternInteractionReport, PatternSpaceError> {
    validate_config(config)?;
    let dimension = config.dimension;
    let role = PatternVector::seeded_random(dimension, derived_seed(seed, 1, 1))?;
    let filler = PatternVector::seeded_random(dimension, derived_seed(seed, 1, 2))?;

    let relation = role.bind(&filler)?;
    let retrieved_filler = relation.unbind(&role)?;
    let role_unbinding_similarity = filler.similarity(&retrieved_filler)?;

    let naive_bundle = PatternVector::bundle(
        &[role.clone(), filler.clone()],
        derived_seed(seed, 2, 1),
    )?;
    let ablated_filler = naive_bundle.unbind(&role)?;
    let bundling_ablation_similarity = filler.similarity(&ablated_filler)?;
    let cancellation_bundle = PatternVector::bundle(
        &[role.clone(), role.opposite()],
        derived_seed(seed, 2, 2),
    )?;
    let antipattern_cancellation_abs_similarity = role.similarity(&cancellation_bundle)?.abs();

    let seq_ab = role.bind(&filler.permute(1));
    let seq_ba = filler.bind(&role.permute(1));
    let sequence_order_similarity = seq_ab?.similarity(&seq_ba?)?;

    let mapping = seeded_permutation(dimension, derived_seed(seed, 3, 1));
    let bound_reencoded = role.bind(&filler)?.reencode(&mapping)?;
    let independently_reencoded =
        role.reencode(&mapping)?.bind(&filler.reencode(&mapping)?)?;
    let basis_equivariance_exact = bound_reencoded == independently_reencoded;

    let mut noise_results = Vec::with_capacity(config.noise_fractions.len());
    for (noise_index, &noise_fraction) in config.noise_fractions.iter().enumerate() {
        let mut correct = 0usize;
        let mut margin_sum = 0.0;
        let mut cleanup_correct = 0usize;
        let mut cleanup_margin_sum = 0.0;
        for trial in 0..config.trials {
            let patterns = make_prototypes(seed, trial, config)?;
            let target_index = trial % config.prototype_count;
            let query = patterns[target_index].with_noise(
                noise_fraction,
                derived_seed(seed, (noise_index * config.trials + trial) as u64, 4),
            )?;
            let space = build_space(&patterns)?;
            let ranked = space.rank(&query)?;
            if ranked.first().is_some_and(|result| result.label == format!("p{target_index}")) {
                correct += 1;
            }
            let target_similarity = query.similarity(&patterns[target_index])?;
            let best_competitor = patterns.iter().enumerate()
                .filter(|(index, _)| *index != target_index)
                .map(|(_, pattern)| query.similarity(pattern))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max);
            margin_sum += target_similarity - best_competitor;

            let cleaned = projected_softmax_cleanup(
                &query,
                &patterns,
                CLEANUP_BETA,
                CLEANUP_STEPS,
            )?;
            let cleaned_ranked = space.rank(&cleaned)?;
            if cleaned_ranked.first().is_some_and(|result| {
                result.label == format!("p{target_index}")
            }) {
                cleanup_correct += 1;
            }
            let cleaned_target_similarity = cleaned.similarity(&patterns[target_index])?;
            let cleaned_best_competitor = patterns.iter().enumerate()
                .filter(|(index, _)| *index != target_index)
                .map(|(_, pattern)| cleaned.similarity(pattern))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max);
            cleanup_margin_sum += cleaned_target_similarity - cleaned_best_competitor;
        }
        noise_results.push(NoiseRetrievalResult {
            noise_fraction,
            top1_accuracy: correct as f64 / config.trials as f64,
            top1_accuracy_ci95: wilson_interval_95(correct, config.trials),
            mean_similarity_margin: margin_sum / config.trials as f64,
            cleanup_top1_accuracy: cleanup_correct as f64 / config.trials as f64,
            cleanup_top1_accuracy_ci95: wilson_interval_95(cleanup_correct, config.trials),
            cleanup_mean_similarity_margin: cleanup_margin_sum / config.trials as f64,
        });
    }

    let mut bundle_results = Vec::with_capacity(config.bundle_sizes.len());
    for (size_index, &bundle_size) in config.bundle_sizes.iter().enumerate() {
        let mut member_similarity_sum = 0.0;
        let mut distractor_similarity_sum = 0.0;
        let mut member_recall_count = 0usize;
        for trial in 0..config.trials {
            let patterns = make_prototypes(seed, trial, config)?;
            let bundled = PatternVector::bundle(
                &patterns[..bundle_size],
                derived_seed(seed, (size_index * config.trials + trial) as u64, 5),
            )?;
            let member_sum = patterns[..bundle_size].iter()
                .map(|pattern| bundled.similarity(pattern))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter().sum::<f64>();
            member_similarity_sum += member_sum / bundle_size as f64;

            let max_distractor = patterns[bundle_size..].iter()
                .map(|pattern| bundled.similarity(pattern))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter().fold(f64::NEG_INFINITY, f64::max);
            distractor_similarity_sum += max_distractor;

            let space = build_space(&patterns)?;
            let ranked = space.rank(&bundled)?;
            if ranked.first().is_some_and(|result| {
                result.label.strip_prefix('p')
                    .and_then(|label| label.parse::<usize>().ok())
                    .is_some_and(|index| index < bundle_size)
            }) {
                member_recall_count += 1;
            }
        }
        bundle_results.push(BundleCapacityResult {
            bundle_size,
            mean_member_similarity: member_similarity_sum / config.trials as f64,
            mean_max_distractor_similarity: distractor_similarity_sum / config.trials as f64,
            member_recall_at_1: member_recall_count as f64 / config.trials as f64,
            member_recall_at_1_ci95: wilson_interval_95(member_recall_count, config.trials),
        });
    }

    let mut associative_capacity_results = Vec::with_capacity(config.bundle_sizes.len());
    for (size_index, &association_count) in config.bundle_sizes.iter().enumerate() {
        let mut target_similarity_sum = 0.0;
        let mut max_distractor_similarity_sum = 0.0;
        let mut correct_count = 0usize;
        let mut cleanup_target_similarity_sum = 0.0;
        let mut cleanup_distractor_similarity_sum = 0.0;
        let mut cleanup_correct_count = 0usize;

        for trial in 0..config.trials {
            let fillers = make_prototypes(seed ^ 0x4153_534f_4349_4154, trial, config)?;
            let roles = (0..association_count)
                .map(|index| PatternVector::seeded_random(
                    dimension,
                    derived_seed(seed ^ 0x524f_4c45_535f_4844, trial as u64, index as u64),
                ))
                .collect::<Result<Vec<_>, _>>()?;
            let bound_pairs = roles.iter()
                .zip(fillers.iter())
                .map(|(role, filler)| role.bind(filler))
                .collect::<Result<Vec<_>, _>>()?;
            let memory = PatternVector::bundle(
                &bound_pairs,
                derived_seed(seed, (size_index * config.trials + trial) as u64, 6),
            )?;
            let target_index = (derived_seed(
                seed ^ 0x5155_4552_595f_4944,
                trial as u64,
                association_count as u64,
            ) % association_count as u64) as usize;
            let query = memory.unbind(&roles[target_index])?;
            let target_similarity = query.similarity(&fillers[target_index])?;
            let max_distractor_similarity = fillers.iter()
                .enumerate()
                .filter(|(index, _)| *index != target_index)
                .map(|(_, filler)| query.similarity(filler))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max);

            target_similarity_sum += target_similarity;
            max_distractor_similarity_sum += max_distractor_similarity;
            let space = build_space(&fillers)?;
            let ranked = space.rank(&query)?;
            if ranked.first().is_some_and(|result| result.label == format!("p{target_index}")) {
                correct_count += 1;
            }

            let cleaned = projected_softmax_cleanup(
                &query,
                &fillers,
                CLEANUP_BETA,
                CLEANUP_STEPS,
            )?;
            let cleaned_target_similarity = cleaned.similarity(&fillers[target_index])?;
            cleanup_target_similarity_sum += cleaned_target_similarity;
            let cleaned_max_distractor = fillers.iter()
                .enumerate()
                .filter(|(index, _)| *index != target_index)
                .map(|(_, filler)| cleaned.similarity(filler))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max);
            cleanup_distractor_similarity_sum += cleaned_max_distractor;
            let cleaned_ranked = space.rank(&cleaned)?;
            if cleaned_ranked.first().is_some_and(|result| {
                result.label == format!("p{target_index}")
            }) {
                cleanup_correct_count += 1;
            }
        }

        associative_capacity_results.push(AssociativeCapacityResult {
            association_count,
            mean_target_similarity: target_similarity_sum / config.trials as f64,
            mean_max_distractor_similarity: max_distractor_similarity_sum
                / config.trials as f64,
            recall_at_1: correct_count as f64 / config.trials as f64,
            recall_at_1_ci95: wilson_interval_95(correct_count, config.trials),
            cleanup_recall_at_1: cleanup_correct_count as f64 / config.trials as f64,
            cleanup_recall_at_1_ci95: wilson_interval_95(cleanup_correct_count, config.trials),
            cleanup_mean_target_similarity: cleanup_target_similarity_sum
                / config.trials as f64,
            cleanup_mean_max_distractor_similarity: cleanup_distractor_similarity_sum
                / config.trials as f64,
        });
    }

    Ok(PatternInteractionReport {
        seed,
        dimension,
        prototype_count: config.prototype_count,
        random_chance_baseline: 1.0 / config.prototype_count as f64,
        role_unbinding_similarity,
        bundling_ablation_similarity,
        binding_ablation_advantage: role_unbinding_similarity - bundling_ablation_similarity,
        antipattern_cancellation_abs_similarity,
        sequence_order_similarity,
        basis_equivariance_exact,
        noise_results,
        bundle_results,
        associative_capacity_results,
    })
}

/// Two-sided 95% Wilson score interval for a binomial proportion.
fn wilson_interval_95(successes: usize, trials: usize) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = z
        * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt())
        / denominator;
    ((center - half_width).max(0.0), (center + half_width).min(1.0))
}

/// Projected softmax associative cleanup: score all stored patterns against
/// the current query, take a beta-scaled softmax-weighted superposition, and
/// project the result back onto the bipolar hypercube. This is a deliberately
/// simple Hopfield-inspired iterative baseline, not claimed to be an exact
/// implementation of any one modern Hopfield formulation.
fn projected_softmax_cleanup(
    query: &PatternVector,
    patterns: &[PatternVector],
    beta: f64,
    steps: usize,
) -> Result<PatternVector, PatternSpaceError> {
    if patterns.is_empty() {
        return Err(PatternSpaceError::EmptyBundle);
    }
    if !beta.is_finite() || beta <= 0.0 || steps == 0 {
        return Err(PatternSpaceError::InvalidConfiguration(
            "cleanup beta and iteration count must be positive",
        ));
    }

    let mut current = query.clone();
    for _ in 0..steps {
        let scores = patterns.iter()
            .map(|pattern| current.similarity(pattern).map(|similarity| beta * similarity))
            .collect::<Result<Vec<_>, _>>()?;
        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|score| (score - max_score).exp()).collect();
        let weight_sum = weights.iter().sum::<f64>();
        if !weight_sum.is_finite() || weight_sum <= 0.0 {
            return Err(PatternSpaceError::InvalidConfiguration(
                "cleanup produced invalid softmax normalization",
            ));
        }

        let mut values = Vec::with_capacity(query.dimension());
        for coordinate in 0..query.dimension() {
            let weighted_vote = patterns.iter().zip(&weights)
                .map(|(pattern, weight)| pattern.values[coordinate] as f64 * weight)
                .sum::<f64>() / weight_sum;
            values.push(if weighted_vote > 0.0 {
                1
            } else if weighted_vote < 0.0 {
                -1
            } else {
                current.values[coordinate]
            });
        }
        current = PatternVector { values };
    }
    Ok(current)
}

fn validate_config(config: &PatternInteractionConfig) -> Result<(), PatternSpaceError> {
    if config.dimension == 0 {
        return Err(PatternSpaceError::InvalidConfiguration("dimension must be positive"));
    }
    if config.prototype_count < 2 {
        return Err(PatternSpaceError::InvalidConfiguration("prototype_count must be at least 2"));
    }
    if config.trials == 0 {
        return Err(PatternSpaceError::InvalidConfiguration("trials must be positive"));
    }
    if config.noise_fractions.iter().any(|fraction| {
        !fraction.is_finite() || !(0.0..=1.0).contains(fraction)
    }) {
        return Err(PatternSpaceError::InvalidConfiguration("noise fractions must be finite and in [0, 1]"));
    }
    if config.bundle_sizes.iter().any(|&size| size == 0 || size >= config.prototype_count) {
        return Err(PatternSpaceError::InvalidConfiguration(
            "bundle sizes must be positive and smaller than prototype_count so distractors remain",
        ));
    }
    Ok(())
}

fn make_prototypes(
    seed: u64,
    trial: usize,
    config: &PatternInteractionConfig,
) -> Result<Vec<PatternVector>, PatternSpaceError> {
    (0..config.prototype_count)
        .map(|index| PatternVector::seeded_random(
            config.dimension,
            derived_seed(seed, trial as u64, index as u64),
        ))
        .collect()
}

fn build_space(patterns: &[PatternVector]) -> Result<PatternSpace, PatternSpaceError> {
    let dimension = patterns.first().ok_or(PatternSpaceError::EmptyBundle)?.dimension();
    let mut space = PatternSpace::new(dimension)?;
    for (index, pattern) in patterns.iter().enumerate() {
        space.insert(format!("p{index}"), pattern.clone())?;
    }
    Ok(space)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bipolar_pattern_rejects_invalid_values() {
        assert_eq!(
            PatternVector::new(vec![1, 0, -1]),
            Err(PatternSpaceError::InvalidBipolarValue { index: 1, value: 0 }),
        );
    }

    #[test]
    fn binding_is_self_inverse_for_bipolar_keys() {
        let a = PatternVector::seeded_random(2_048, 11).expect("pattern");
        let b = PatternVector::seeded_random(2_048, 12).expect("pattern");
        let recovered = a.bind(&b).expect("bind").unbind(&b).expect("unbind");
        assert_eq!(recovered, a);
        assert_eq!(a.similarity(&recovered).expect("similarity"), 1.0);
    }

    #[test]
    fn basis_reencoding_preserves_binding_and_similarity() {
        let a = PatternVector::seeded_random(2_048, 21).expect("pattern");
        let b = PatternVector::seeded_random(2_048, 22).expect("pattern");
        let mapping = seeded_permutation(2_048, 23);
        let before = a.bind(&b).expect("bind");
        let after = a.reencode(&mapping).expect("reencode")
            .bind(&b.reencode(&mapping).expect("reencode")).expect("bind");
        assert_eq!(before.reencode(&mapping).expect("reencode"), after);
        assert_eq!(
            a.similarity(&b).expect("similarity"),
            a.reencode(&mapping).expect("reencode")
                .similarity(&b.reencode(&mapping).expect("reencode")).expect("similarity"),
        );
    }

    #[test]
    fn role_binding_beats_the_bundling_ablation() {
        let config = PatternInteractionConfig {
            dimension: 2_048,
            prototype_count: 32,
            trials: 8,
            noise_fractions: vec![0.1, 0.2],
            bundle_sizes: vec![1, 3, 5, 9],
        };
        let report = run_pattern_interaction_suite(42, &config).expect("suite");
        assert_eq!(report.role_unbinding_similarity, 1.0);
        assert!(report.binding_ablation_advantage > 0.25);
        assert!(report.antipattern_cancellation_abs_similarity < 0.15);
        assert!(report.basis_equivariance_exact);
        assert!(report.sequence_order_similarity.abs() < 0.15);
    }

    #[test]
    fn noisy_prototype_retrieval_beats_random_chance() {
        let config = PatternInteractionConfig {
            dimension: 1_024,
            prototype_count: 24,
            trials: 16,
            noise_fractions: vec![0.05, 0.10, 0.20],
            bundle_sizes: vec![1, 3, 5, 9],
        };
        let report = run_pattern_interaction_suite(7, &config).expect("suite");
        assert!(report.noise_results.iter().all(|result| {
            result.top1_accuracy >= report.random_chance_baseline
        }));
        assert!(report.noise_results[0].top1_accuracy >= 0.9);
        assert!(report.noise_results[0].cleanup_top1_accuracy >= 0.9);
        assert!(report.noise_results[0].cleanup_mean_similarity_margin > 0.0);
        let (low, high) = report.noise_results[0].cleanup_top1_accuracy_ci95;
        assert!(low <= report.noise_results[0].cleanup_top1_accuracy);
        assert!(high >= report.noise_results[0].cleanup_top1_accuracy);
    }

    #[test]
    fn bundle_capacity_reports_members_and_competing_distractors() {
        let config = PatternInteractionConfig {
            dimension: 2_048,
            prototype_count: 32,
            trials: 8,
            noise_fractions: vec![0.1],
            bundle_sizes: vec![1, 3, 5, 9, 17],
        };
        let report = run_pattern_interaction_suite(99, &config).expect("suite");
        assert_eq!(report.bundle_results.len(), 5);
        assert_eq!(report.bundle_results[0].member_recall_at_1, 1.0);
        assert_eq!(report.associative_capacity_results.len(), 5);
        assert_eq!(report.associative_capacity_results[0].recall_at_1, 1.0);
        assert_eq!(report.associative_capacity_results[0].cleanup_recall_at_1, 1.0);
        assert_eq!(report.associative_capacity_results[0].mean_target_similarity, 1.0);
        assert_eq!(
            report.associative_capacity_results[0].cleanup_mean_target_similarity,
            1.0
        );
        let (low, high) = report.associative_capacity_results[0].cleanup_recall_at_1_ci95;
        assert!(low <= report.associative_capacity_results[0].cleanup_recall_at_1);
        assert!(high >= report.associative_capacity_results[0].cleanup_recall_at_1);
        assert!(report.bundle_results.iter().all(|result| {
            (0.0..=1.0).contains(&result.member_recall_at_1)
                && result.mean_member_similarity > result.mean_max_distractor_similarity
        }));
    }

    #[test]
    fn fixed_seed_produces_identical_report() {
        let config = PatternInteractionConfig {
            dimension: 1_024,
            prototype_count: 24,
            trials: 8,
            noise_fractions: vec![0.05, 0.10, 0.20],
            bundle_sizes: vec![1, 3, 5, 9],
        };
        let first = run_pattern_interaction_suite(0x504c_4154, &config).expect("first run");
        let second = run_pattern_interaction_suite(0x504c_4154, &config).expect("second run");
        assert_eq!(first, second);
    }

    #[test]
    fn pattern_space_rejects_dimension_mismatch_and_duplicate_labels() {
        let mut space = PatternSpace::new(8).expect("space");
        let pattern = PatternVector::seeded_random(8, 100).expect("pattern");
        space.insert("alpha", pattern.clone()).expect("first insert");
        assert_eq!(
            space.insert("alpha", pattern),
            Err(PatternSpaceError::DuplicateLabel("alpha".to_string())),
        );
        assert!(matches!(
            space.rank(&PatternVector::seeded_random(4, 101).expect("wrong-sized query")),
            Err(PatternSpaceError::DimensionMismatch { expected: 8, actual: 4 }),
        ));
    }

    #[test]
    fn wilson_interval_bounds_include_observed_accuracy() {
        let all_correct = wilson_interval_95(16, 16);
        assert!(all_correct.0 > 0.0);
        assert_eq!(all_correct.1, 1.0);
        let half_correct = wilson_interval_95(50, 100);
        assert!(half_correct.0 < 0.5);
        assert!(half_correct.1 > 0.5);
        assert_eq!(wilson_interval_95(0, 0), (0.0, 1.0));
    }

    #[test]
    fn invalid_basis_and_noise_are_rejected() {
        let pattern = PatternVector::seeded_random(8, 1).expect("pattern");
        assert_eq!(
            pattern.reencode(&[0, 1, 1, 3, 4, 5, 6, 7]),
            Err(PatternSpaceError::InvalidPermutation),
        );
        assert_eq!(pattern.with_noise(1.1, 2), Err(PatternSpaceError::InvalidNoiseFraction));
    }
}
