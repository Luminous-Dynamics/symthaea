// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Holographic Reduced Representation (HRR) associative-memory baseline.
//!
//! This module exists to compare a different vector-symbolic algebra against
//! the bipolar MAP/BSC-style operations in pattern_interaction. HRR binds by
//! circular convolution, unbinds by circular correlation, and superposes
//! relations by vector addition. It uses rustfft for O(D log D) convolutions.
//!
//! This is a baseline implementation for controlled experiments, not a claim
//! of state-of-the-art performance or a substitute for independent validation
//! against a published HRR reference implementation.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use super::pattern_interaction::PatternSpaceError;

#[derive(Debug, Clone, PartialEq)]
pub struct HrrVector {
    values: Vec<f64>,
}

impl HrrVector {
    pub fn seeded_random(dimension: usize, seed: u64) -> Result<Self, PatternSpaceError> {
        if dimension == 0 {
            return Err(PatternSpaceError::EmptyVector);
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let values = (0..dimension)
            .map(|_| {
                let sample: f64 = StandardNormal.sample(&mut rng);
                sample
            })
            .collect();
        Ok(Self { values })
    }

    pub fn dimension(&self) -> usize {
        self.values.len()
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn similarity(&self, other: &Self) -> Result<f64, PatternSpaceError> {
        if self.dimension() != other.dimension() {
            return Err(PatternSpaceError::DimensionMismatch {
                expected: self.dimension(),
                actual: other.dimension(),
            });
        }
        let dot = self.values.iter().zip(&other.values)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        let left_norm = self.values.iter().map(|value| value * value).sum::<f64>().sqrt();
        let right_norm = other.values.iter().map(|value| value * value).sum::<f64>().sqrt();
        let denominator = left_norm * right_norm;
        if denominator <= f64::MIN_POSITIVE || !denominator.is_finite() {
            return Err(PatternSpaceError::InvalidConfiguration(
                "HRR cosine similarity requires finite, nonzero vectors",
            ));
        }
        Ok(dot / denominator)
    }
}

/// Cached FFT plans and HRR operations for a fixed dimensionality.
pub struct HrrAlgebra {
    dimension: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl HrrAlgebra {
    pub fn new(dimension: usize) -> Result<Self, PatternSpaceError> {
        if dimension == 0 {
            return Err(PatternSpaceError::EmptyVector);
        }
        let mut planner = FftPlanner::<f64>::new();
        Ok(Self {
            dimension,
            forward: planner.plan_fft_forward(dimension),
            inverse: planner.plan_fft_inverse(dimension),
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Circular convolution, the HRR binding operation.
    pub fn bind(
        &self,
        left: &HrrVector,
        right: &HrrVector,
    ) -> Result<HrrVector, PatternSpaceError> {
        self.check_dimension(left)?;
        self.check_dimension(right)?;
        let mut left_spectrum = self.spectrum(left);
        let right_spectrum = self.spectrum(right);
        for (left_bin, right_bin) in left_spectrum.iter_mut().zip(right_spectrum) {
            *left_bin *= right_bin;
        }
        self.inverse.process(&mut left_spectrum);
        let scale = self.dimension as f64;
        Ok(HrrVector {
            values: left_spectrum.iter().map(|bin| bin.re / scale).collect(),
        })
    }

    /// Circular correlation, the approximate HRR unbinding operation.
    ///
    /// Convolution/correlation round-off is retained; the output is not forced
    /// back to a stored prototype. Retrieval quality is measured by similarity.
    pub fn unbind(
        &self,
        memory: &HrrVector,
        role: &HrrVector,
    ) -> Result<HrrVector, PatternSpaceError> {
        self.check_dimension(memory)?;
        self.check_dimension(role)?;
        let mut memory_spectrum = self.spectrum(memory);
        let role_spectrum = self.spectrum(role);
        for (memory_bin, role_bin) in memory_spectrum.iter_mut().zip(role_spectrum) {
            *memory_bin *= role_bin.conj();
        }
        self.inverse.process(&mut memory_spectrum);
        let scale = self.dimension as f64;
        Ok(HrrVector {
            values: memory_spectrum.iter().map(|bin| bin.re / scale).collect(),
        })
    }

    /// Additive superposition of HRR traces.
    pub fn bundle(&self, vectors: &[HrrVector]) -> Result<HrrVector, PatternSpaceError> {
        let first = vectors.first().ok_or(PatternSpaceError::EmptyBundle)?;
        self.check_dimension(first)?;
        let mut values = vec![0.0; self.dimension];
        for vector in vectors {
            self.check_dimension(vector)?;
            for (accumulator, value) in values.iter_mut().zip(&vector.values) {
                *accumulator += value;
            }
        }
        Ok(HrrVector { values })
    }

    fn spectrum(&self, vector: &HrrVector) -> Vec<Complex<f64>> {
        let mut spectrum: Vec<Complex<f64>> = vector.values
            .iter()
            .map(|&value| Complex::new(value, 0.0))
            .collect();
        self.forward.process(&mut spectrum);
        spectrum
    }

    fn check_dimension(&self, vector: &HrrVector) -> Result<(), PatternSpaceError> {
        if vector.dimension() != self.dimension {
            return Err(PatternSpaceError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.dimension(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HrrCapacityResult {
    pub association_count: usize,
    pub raw_recall_at_1: f64,
    pub raw_recall_ci95: (f64, f64),
    pub mean_target_similarity: f64,
    pub mean_max_distractor_similarity: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HrrCapacityReport {
    pub seed: u64,
    pub dimension: usize,
    pub candidate_count: usize,
    pub trials: usize,
    pub results: Vec<HrrCapacityResult>,
}

/// Measure role/filler retrieval from a bundled HRR trace.
///
/// Each trial generates a fixed filler library and a role library up to the
/// largest requested capacity. For a capacity K, the first K role/filler
/// pairs are bound and added; one seeded role is unbound and ranked against
/// the complete filler library. Patterns and trial IDs are deterministic.
pub fn run_hrr_associative_capacity(
    seed: u64,
    dimension: usize,
    candidate_count: usize,
    trials: usize,
    association_counts: &[usize],
) -> Result<HrrCapacityReport, PatternSpaceError> {
    if dimension == 0 || candidate_count < 2 || trials == 0 {
        return Err(PatternSpaceError::InvalidConfiguration(
            "HRR dimension, candidate count, and trial count must be positive",
        ));
    }
    if association_counts.is_empty()
        || association_counts.iter().any(|&count| count == 0 || count >= candidate_count)
    {
        return Err(PatternSpaceError::InvalidConfiguration(
            "HRR association counts must be positive and smaller than candidate_count",
        ));
    }
    let algebra = HrrAlgebra::new(dimension)?;
    let max_associations = *association_counts.iter().max().expect("nonempty counts");
    let mut fillers_by_trial = Vec::with_capacity(trials);
    let mut roles_by_trial = Vec::with_capacity(trials);

    for trial in 0..trials {
        let fillers = (0..candidate_count)
            .map(|index| {
                HrrVector::seeded_random(
                    dimension,
                    derived_seed(seed ^ 0x4852_525f_4649_4c4c, trial as u64, index as u64),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let roles = (0..max_associations)
            .map(|index| {
                HrrVector::seeded_random(
                    dimension,
                    derived_seed(seed ^ 0x4852_525f_524f_4c45, trial as u64, index as u64),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        fillers_by_trial.push(fillers);
        roles_by_trial.push(roles);
    }

    let mut results = Vec::with_capacity(association_counts.len());
    for &association_count in association_counts {
        let mut correct = 0usize;
        let mut target_similarity_sum = 0.0;
        let mut max_distractor_similarity_sum = 0.0;

        for trial in 0..trials {
            let fillers = &fillers_by_trial[trial];
            let roles = &roles_by_trial[trial];
            let bound_pairs = roles[..association_count]
                .iter()
                .zip(&fillers[..association_count])
                .map(|(role, filler)| algebra.bind(role, filler))
                .collect::<Result<Vec<_>, _>>()?;
            let memory = algebra.bundle(&bound_pairs)?;

            let target_index = (derived_seed(
                seed ^ 0x5155_4552_595f_4944,
                trial as u64,
                association_count as u64,
            ) % association_count as u64) as usize;
            let query = algebra.unbind(&memory, &roles[target_index])?;

            let similarities = fillers
                .iter()
                .map(|filler| query.similarity(filler))
                .collect::<Result<Vec<_>, _>>()?;
            let target_similarity = similarities[target_index];
            let best_index = similarities
                .iter()
                .enumerate()
                .max_by(|(left_index, left), (right_index, right)| {
                    left.total_cmp(right).then_with(|| right_index.cmp(left_index))
                })
                .map(|(index, _)| index)
                .expect("candidate_count is validated as at least two");
            let max_distractor = similarities
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != target_index)
                .map(|(_, similarity)| *similarity)
                .fold(f64::NEG_INFINITY, f64::max);

            if best_index == target_index {
                correct += 1;
            }
            target_similarity_sum += target_similarity;
            max_distractor_similarity_sum += max_distractor;
        }

        results.push(HrrCapacityResult {
            association_count,
            raw_recall_at_1: correct as f64 / trials as f64,
            raw_recall_ci95: wilson_interval_95(correct, trials),
            mean_target_similarity: target_similarity_sum / trials as f64,
            mean_max_distractor_similarity: max_distractor_similarity_sum / trials as f64,
        });
    }

    Ok(HrrCapacityReport {
        seed,
        dimension,
        candidate_count,
        trials,
        results,
    })
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn derived_seed(seed: u64, first: u64, second: u64) -> u64 {
    mix64(seed ^ mix64(first) ^ mix64(second.wrapping_add(0x9e37_79b9_7f4a_7c15)))
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hrr_binding_and_unbinding_preserve_target_similarity() {
        let algebra = HrrAlgebra::new(1_024).expect("HRR algebra");
        let role = HrrVector::seeded_random(1_024, 101).expect("role");
        let filler = HrrVector::seeded_random(1_024, 102).expect("filler");
        let bound = algebra.bind(&role, &filler).expect("binding");
        let recovered = algebra.unbind(&bound, &role).expect("unbinding");
        assert!(
            recovered.similarity(&filler).expect("similarity") > 0.45,
            "HRR circular correlation should preserve target similarity"
        );
    }

    #[test]
    fn hrr_capacity_is_deterministic_and_bounded() {
        let first = run_hrr_associative_capacity(2026, 512, 24, 8, &[1, 3, 5, 9, 17])
            .expect("first capacity run");
        let second = run_hrr_associative_capacity(2026, 512, 24, 8, &[1, 3, 5, 9, 17])
            .expect("second capacity run");
        assert_eq!(first, second);
        assert!(first.results.iter().all(|result| {
            (0.0..=1.0).contains(&result.raw_recall_at_1)
                && result.raw_recall_ci95.0 <= result.raw_recall_at_1
                && result.raw_recall_ci95.1 >= result.raw_recall_at_1
        }));
        assert_eq!(first.results[0].raw_recall_at_1, 1.0);
    }

    #[test]
    fn hrr_rejects_invalid_capacity_and_dimension_mismatch() {
        assert!(run_hrr_associative_capacity(1, 256, 2, 4, &[0]).is_err());
        let algebra = HrrAlgebra::new(16).expect("algebra");
        let left = HrrVector::seeded_random(16, 1).expect("left");
        let wrong = HrrVector::seeded_random(8, 2).expect("wrong");
        assert!(algebra.bind(&left, &wrong).is_err());
    }
}
