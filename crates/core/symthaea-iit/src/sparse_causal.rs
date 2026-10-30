// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sparse local-CPT causal models with exact log-domain IIT 4.0 Eq. 4 contraction.
//!
//! Storage is proportional to the sum of local parent-table sizes, not n * 2^n.
//! Exact inference is governed by factor-graph induced width and a caller-supplied
//! intermediate-table budget. This module computes a cause-likelihood primitive,
//! not a complete system phi_s calculation or a consciousness determination.

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    min_fill_log_sum_product, BinaryCausalModel, BinaryLogFactor, DirectedSystemPartition,
    FactorGraphError, ModelError, PartitionApplyError, MAX_DENSE_UNITS,
};

const SPARSE_MODEL_MAGIC: &[u8] = b"SYMTHAEA-IIT-SPARSE-BINARY-CAUSAL-MODEL\0v0.1\0";
const SPARSE_EVALUATION_MAGIC: &[u8] = b"SYMTHAEA-IIT-SPARSE-EVALUATION\0v0.1\0";
const SPARSE_EQ4_REQUEST_MAGIC: &[u8] = b"SYMTHAEA-IIT-SPARSE-EQ4-REQUEST\0v0.1\0";
const SPARSE_EQ4_RECEIPT_MAGIC: &[u8] = b"SYMTHAEA-IIT-SPARSE-EQ4-RECEIPT\0v0.1\0";
pub const SPARSE_IIT4_EQ4_ALGORITHM_ID: &str = "symthaea-iit/sparse-iit4-eq4-minfill/0.1.0";

pub const MAX_SPARSE_UNITS: usize = 1_000_000;
pub const MAX_LOCAL_PARENTS: usize = 20;

/// One local target CPT. Parent variables are ascending; parent 0 is the
/// least-significant bit of the local CPT row index.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMechanismCpt {
    target_unit: usize,
    parent_units: Vec<usize>,
    probability_next_one: Vec<f64>,
}

impl SparseMechanismCpt {
    pub fn new(
        target_unit: usize,
        parent_units: Vec<usize>,
        probability_next_one: Vec<f64>,
    ) -> Result<Self, SparseCausalError> {
        if parent_units.len() > MAX_LOCAL_PARENTS {
            return Err(SparseCausalError::TooManyParents {
                target: target_unit,
                parents: parent_units.len(),
                maximum: MAX_LOCAL_PARENTS,
            });
        }
        for pair in parent_units.windows(2) {
            if pair[0] >= pair[1] {
                return Err(SparseCausalError::ParentScopeNotStrictlyAscending {
                    target: target_unit,
                    left: pair[0],
                    right: pair[1],
                });
            }
        }
        let expected = 1usize
            .checked_shl(parent_units.len() as u32)
            .ok_or(SparseCausalError::ParentTableSizeOverflow {
                target: target_unit,
                parents: parent_units.len(),
            })?;
        if probability_next_one.len() != expected {
            return Err(SparseCausalError::ParentTableLength {
                target: target_unit,
                expected,
                actual: probability_next_one.len(),
            });
        }
        for (index, value) in probability_next_one.iter().copied().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(SparseCausalError::InvalidProbability {
                    target: target_unit,
                    index,
                    value,
                });
            }
        }
        Ok(Self { target_unit, parent_units, probability_next_one })
    }

    pub fn target_unit(&self) -> usize { self.target_unit }
    pub fn parent_units(&self) -> &[usize] { &self.parent_units }
    pub fn probability_table(&self) -> &[f64] { &self.probability_next_one }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum SparseCausalError {
    #[error("sparse model unit count must be in 1..=1000000; got {units}")]
    InvalidUnitCount { units: usize },
    #[error("target unit {target} is outside 0..{unit_count}")]
    TargetOutOfRange { target: usize, unit_count: usize },
    #[error("target unit {target} occurs more than once")]
    DuplicateTarget { target: usize },
    #[error("target unit {target} has no local conditional table")]
    MissingTarget { target: usize },
    #[error("parent {parent} for target {target} is outside 0..{unit_count}")]
    ParentOutOfRange { target: usize, parent: usize, unit_count: usize },
    #[error("parent scope for target {target} is not strictly ascending: {left} before {right}")]
    ParentScopeNotStrictlyAscending { target: usize, left: usize, right: usize },
    #[error("target {target} has {parents} parents; maximum supported is {maximum}")]
    TooManyParents { target: usize, parents: usize, maximum: usize },
    #[error("local table size overflow for target {target} with {parents} parents")]
    ParentTableSizeOverflow { target: usize, parents: usize },
    #[error("target {target} table length mismatch: expected {expected}, got {actual}")]
    ParentTableLength { target: usize, expected: usize, actual: usize },
    #[error("target {target} probability at local index {index} must be finite and in [0,1]; got {value}")]
    InvalidProbability { target: usize, index: usize, value: f64 },
    #[error("state length mismatch: expected {expected}, got {actual}")]
    StateLength { expected: usize, actual: usize },
    #[error("candidate system must contain at least one unit")]
    EmptyCandidateSystem,
    #[error("candidate unit {unit} is outside 0..{unit_count}")]
    CandidateUnitOutOfRange { unit: usize, unit_count: usize },
    #[error("candidate system contains duplicate unit {unit}")]
    DuplicateCandidateUnit { unit: usize },
    #[error("candidate-system posterior is undefined for the observed whole-system state")]
    UndefinedBackgroundPosterior,
    #[error("sparse Eq4 report entry {index} is not finite and nonnegative: {value}")]
    InvalidReportValue { index: usize, value: f64 },
    #[error("cannot convert {units} sparse units to dense; maximum is {maximum}")]
    DenseConversionLimit { units: usize, maximum: usize },
    #[error("dense model target {target} is not a function of its declared parent scope at full state {state}")]
    DenseModelNotLocal { target: usize, state: usize },
    #[error(transparent)]
    PartitionApply(#[from] PartitionApplyError),
    #[error(transparent)]
    DenseModel(#[from] ModelError),
    #[error(transparent)]
    FactorGraph(#[from] FactorGraphError),
}

/// Sparse local-CPT representation of a factored binary causal model.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseBinaryCausalModel {
    n_units: usize,
    mechanisms: Vec<SparseMechanismCpt>,
}

/// Eq. 4 likelihood output and exact variable-elimination cost telemetry.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseEq4Report {
    /// Identity of model, state, candidate, algorithm, and resource budget.
    pub request_sha256: String,
    pub max_factor_entries: usize,
    /// Unnormalized forward cause likelihood. First candidate unit is the low bit.
    pub forward_likelihood: Vec<f64>,
    pub posterior_induced_width: usize,
    pub maximum_target_induced_width: usize,
    pub peak_factor_entries: usize,
    pub factor_multiplications: usize,
}

impl SparseEq4Report {
    /// Canonical receipt binding the request, resource budget, result vector, and work counters.
    pub fn receipt_sha256(&self) -> Result<String, SparseCausalError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SPARSE_EQ4_RECEIPT_MAGIC);
        sparse_receipt_string(&mut bytes, SPARSE_IIT4_EQ4_ALGORITHM_ID);
        sparse_receipt_string(&mut bytes, &self.request_sha256);
        sparse_receipt_u64(&mut bytes, self.max_factor_entries as u64);
        sparse_receipt_u64(&mut bytes, self.forward_likelihood.len() as u64);
        for (index, value) in self.forward_likelihood.iter().copied().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(SparseCausalError::InvalidReportValue { index, value });
            }
            let canonical = if value == 0.0 { 0.0 } else { value };
            bytes.extend_from_slice(&canonical.to_bits().to_le_bytes());
        }
        sparse_receipt_u64(&mut bytes, self.posterior_induced_width as u64);
        sparse_receipt_u64(&mut bytes, self.maximum_target_induced_width as u64);
        sparse_receipt_u64(&mut bytes, self.peak_factor_entries as u64);
        sparse_receipt_u64(&mut bytes, self.factor_multiplications as u64);
        Ok(sparse_hex_digest(&bytes))
    }
}

fn sparse_receipt_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn sparse_receipt_string(bytes: &mut Vec<u8>, value: &str) {
    sparse_receipt_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

impl SparseBinaryCausalModel {
    /// Construct a sparse model with exactly one local CPT per target unit.
    pub fn new(
        n_units: usize,
        mut mechanisms: Vec<SparseMechanismCpt>,
    ) -> Result<Self, SparseCausalError> {
        if !(1..=MAX_SPARSE_UNITS).contains(&n_units) {
            return Err(SparseCausalError::InvalidUnitCount { units: n_units });
        }
        mechanisms.sort_unstable_by_key(|mechanism| mechanism.target_unit);
        for (expected_target, mechanism) in mechanisms.iter().enumerate() {
            if mechanism.target_unit >= n_units {
                return Err(SparseCausalError::TargetOutOfRange {
                    target: mechanism.target_unit,
                    unit_count: n_units,
                });
            }
            if mechanism.target_unit < expected_target {
                return Err(SparseCausalError::DuplicateTarget { target: mechanism.target_unit });
            }
            if mechanism.target_unit > expected_target {
                return Err(SparseCausalError::MissingTarget { target: expected_target });
            }
            for parent in mechanism.parent_units.iter().copied() {
                if parent >= n_units {
                    return Err(SparseCausalError::ParentOutOfRange {
                        target: mechanism.target_unit,
                        parent,
                        unit_count: n_units,
                    });
                }
            }
        }
        if mechanisms.len() < n_units {
            return Err(SparseCausalError::MissingTarget { target: mechanisms.len() });
        }
        if mechanisms.len() > n_units {
            return Err(SparseCausalError::DuplicateTarget {
                target: mechanisms[n_units].target_unit,
            });
        }
        Ok(Self { n_units, mechanisms })
    }

    /// Convert a dense model only if each TPM column is exactly a function of
    /// its declared parents. Extra declared edges are retained in the sparse scope.
    pub fn from_dense(model: &BinaryCausalModel) -> Result<Self, SparseCausalError> {
        let n_units = model.n_units();
        let tpm = model.transition_probabilities();
        let connectivity = model.connectivity();
        let mut mechanisms = Vec::with_capacity(n_units);

        for target in 0..n_units {
            let parents = (0..n_units)
                .filter(|source| connectivity[*source * n_units + target])
                .collect::<Vec<_>>();
            if parents.len() > MAX_LOCAL_PARENTS {
                return Err(SparseCausalError::TooManyParents {
                    target,
                    parents: parents.len(),
                    maximum: MAX_LOCAL_PARENTS,
                });
            }
            let table_len = 1usize
                .checked_shl(parents.len() as u32)
                .ok_or(SparseCausalError::ParentTableSizeOverflow {
                    target,
                    parents: parents.len(),
                })?;
            let mut probabilities = vec![None; table_len];
            for full_state in 0..model.n_states() {
                let local_state = parents.iter().enumerate().fold(
                    0usize,
                    |index, (slot, parent)| {
                        if ((full_state >> parent) & 1) != 0 {
                            index | (1usize << slot)
                        } else { index }
                    },
                );
                let probability = tpm[full_state * n_units + target];
                match probabilities[local_state] {
                    Some(existing) if existing != probability => {
                        return Err(SparseCausalError::DenseModelNotLocal {
                            target,
                            state: full_state,
                        });
                    }
                    Some(_) => {}
                    None => probabilities[local_state] = Some(probability),
                }
            }
            let probability_table = probabilities
                .into_iter()
                .enumerate()
                .map(|(index, value)| {
                    value.ok_or(SparseCausalError::ParentTableLength {
                        target,
                        expected: table_len,
                        actual: index,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            mechanisms.push(SparseMechanismCpt::new(target, parents, probability_table)?);
        }
        Self::new(n_units, mechanisms)
    }

    pub fn n_units(&self) -> usize { self.n_units }

    pub fn mechanism(&self, target_unit: usize) -> Option<&SparseMechanismCpt> {
        self.mechanisms.get(target_unit)
    }

    /// Convert to the dense reference representation when its exponential
    /// storage bound permits it.
    pub fn to_dense(&self) -> Result<BinaryCausalModel, SparseCausalError> {
        if self.n_units > MAX_DENSE_UNITS {
            return Err(SparseCausalError::DenseConversionLimit {
                units: self.n_units,
                maximum: MAX_DENSE_UNITS,
            });
        }
        let state_count = 1usize << self.n_units;
        let mut transition_probabilities = vec![0.0; state_count * self.n_units];
        let mut connectivity = vec![false; self.n_units * self.n_units];
        for mechanism in &self.mechanisms {
            for parent in mechanism.parent_units.iter().copied() {
                connectivity[parent * self.n_units + mechanism.target_unit] = true;
            }
        }
        for state_index in 0..state_count {
            let state = (0..self.n_units)
                .map(|unit| ((state_index >> unit) & 1) != 0)
                .collect::<Vec<_>>();
            for target in 0..self.n_units {
                transition_probabilities[state_index * self.n_units + target] =
                    self.probability_next_one(&state, target)?;
            }
        }
        Ok(BinaryCausalModel::new(
            self.n_units,
            transition_probabilities,
            connectivity,
        )?)
    }

    /// Apply a directed system partition by maximum-entropy marginalizing every
    /// severed internal parent bit from each target's local CPT.
    ///
    /// Parent variables outside the candidate system are untouched. The
    /// transformation is local to each target CPT and exactly matches the dense
    /// TPM operation: average the target probability over every severed source
    /// bit while retaining all uncut parent assignments.
    pub fn apply_partition(
        &self,
        partition: &DirectedSystemPartition,
    ) -> Result<Self, SparseCausalError> {
        let candidate = partition.candidate_system();
        let candidate_count = candidate.len();
        let expected_cut_entries = candidate_count
            .checked_mul(candidate_count)
            .ok_or(SparseCausalError::ParentTableSizeOverflow {
                target: candidate.first().copied().unwrap_or(0),
                parents: candidate_count.saturating_mul(2),
            })?;
        if partition.cut_matrix().len() != expected_cut_entries {
            return Err(PartitionApplyError::CutMatrixShapeMismatch {
                expected: expected_cut_entries,
                actual: partition.cut_matrix().len(),
            }.into());
        }

        let mut candidate_slot = vec![None; self.n_units];
        for (slot, unit) in candidate.iter().copied().enumerate() {
            if unit >= self.n_units {
                return Err(PartitionApplyError::CandidateUnitOutOfRange {
                    unit,
                    unit_count: self.n_units,
                }.into());
            }
            if candidate_slot[unit].is_some() {
                return Err(SparseCausalError::DuplicateCandidateUnit { unit });
            }
            candidate_slot[unit] = Some(slot);
        }

        let mut transformed = Vec::with_capacity(self.n_units);
        for mechanism in &self.mechanisms {
            let target_slot = candidate_slot[mechanism.target_unit];
            let cut_parent_flags = mechanism
                .parent_units
                .iter()
                .map(|parent| {
                    match (candidate_slot[*parent], target_slot) {
                        (Some(source_slot), Some(target_slot)) => {
                            partition.cut_matrix()[source_slot * candidate_count + target_slot]
                        }
                        _ => false,
                    }
                })
                .collect::<Vec<_>>();
            let cut_count = cut_parent_flags.iter().filter(|cut| **cut).count();
            if cut_count == 0 {
                transformed.push(mechanism.clone());
                continue;
            }

            let retained_parents = mechanism
                .parent_units
                .iter()
                .copied()
                .zip(cut_parent_flags.iter().copied())
                .filter_map(|(parent, cut)| if cut { None } else { Some(parent) })
                .collect::<Vec<_>>();
            let retained_entries = 1usize
                .checked_shl(retained_parents.len() as u32)
                .ok_or(SparseCausalError::ParentTableSizeOverflow {
                    target: mechanism.target_unit,
                    parents: retained_parents.len(),
                })?;
            let cut_assignments = 1usize
                .checked_shl(cut_count as u32)
                .ok_or(SparseCausalError::ParentTableSizeOverflow {
                    target: mechanism.target_unit,
                    parents: cut_count,
                })?;
            let cut_weight = 1.0 / cut_assignments as f64;
            let mut probability_table = Vec::with_capacity(retained_entries);
            for retained_assignment in 0..retained_entries {
                let mut average = 0.0;
                for cut_assignment in 0..cut_assignments {
                    let mut original_assignment = 0usize;
                    let mut retained_slot = 0usize;
                    let mut cut_slot = 0usize;
                    for (original_slot, is_cut) in cut_parent_flags.iter().copied().enumerate() {
                        let bit = if is_cut {
                            let bit = ((cut_assignment >> cut_slot) & 1) != 0;
                            cut_slot += 1;
                            bit
                        } else {
                            let bit = ((retained_assignment >> retained_slot) & 1) != 0;
                            retained_slot += 1;
                            bit
                        };
                        if bit {
                            original_assignment |= 1usize << original_slot;
                        }
                    }
                    average += cut_weight * mechanism.probability_next_one[original_assignment];
                }
                probability_table.push(average);
            }
            transformed.push(SparseMechanismCpt::new(
                mechanism.target_unit,
                retained_parents,
                probability_table,
            )?);
        }
        Self::new(self.n_units, transformed)
    }

    /// Versioned SHA-256 identity in a namespace distinct from dense TPM digests.
    pub fn model_sha256(&self) -> String {
        sparse_hex_digest(&self.canonical_model_bytes())
    }

    pub fn evaluation_sha256(&self, state: &[bool]) -> Result<String, SparseCausalError> {
        self.check_state(state)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SPARSE_EVALUATION_MAGIC);
        bytes.extend_from_slice(&self.canonical_model_bytes());
        bytes.extend(state.iter().map(|on| u8::from(*on)));
        Ok(sparse_hex_digest(&bytes))
    }

    pub fn probability_next_one(
        &self,
        current_state: &[bool],
        target_unit: usize,
    ) -> Result<f64, SparseCausalError> {
        self.check_state(current_state)?;
        let mechanism = self.mechanisms.get(target_unit).ok_or(
            SparseCausalError::TargetOutOfRange {
                target: target_unit,
                unit_count: self.n_units,
            },
        )?;
        let local_state = mechanism.parent_units.iter().enumerate().fold(
            0usize,
            |index, (slot, parent)| {
                if current_state[*parent] { index | (1usize << slot) } else { index }
            },
        );
        Ok(mechanism.probability_next_one[local_state])
    }

    pub fn transition_probability(
        &self,
        current_state: &[bool],
        next_state: &[bool],
    ) -> Result<f64, SparseCausalError> {
        self.check_state(current_state)?;
        self.check_state(next_state)?;
        let mut probability = 1.0;
        for (target, next_on) in next_state.iter().copied().enumerate() {
            let p_on = self.probability_next_one(current_state, target)?;
            probability *= if next_on { p_on } else { 1.0 - p_on };
        }
        Ok(probability)
    }

    /// Exact sparse Eq. 4 cause-likelihood contraction.
    ///
    /// A posterior over all past variables is conditioned on the observed
    /// whole-system next state. For each candidate target, a duplicate copy of
    /// candidate-past variables is attached to that target's observation factor;
    /// the original system and external-background variables are eliminated.
    /// Complexity depends on induced width rather than directly on background count.
    pub fn forward_cause_likelihood_iit4(
        &self,
        candidate_system: &[usize],
        current_state: &[bool],
        max_factor_entries: usize,
    ) -> Result<SparseEq4Report, SparseCausalError> {
        self.check_state(current_state)?;
        if candidate_system.is_empty() {
            return Err(SparseCausalError::EmptyCandidateSystem);
        }
        let mut candidate_slot = vec![None; self.n_units];
        for (slot, unit) in candidate_system.iter().copied().enumerate() {
            if unit >= self.n_units {
                return Err(SparseCausalError::CandidateUnitOutOfRange {
                    unit,
                    unit_count: self.n_units,
                });
            }
            if candidate_slot[unit].is_some() {
                return Err(SparseCausalError::DuplicateCandidateUnit { unit });
            }
            candidate_slot[unit] = Some(slot);
        }

        let candidate_state_count = 1usize
            .checked_shl(candidate_system.len() as u32)
            .ok_or(SparseCausalError::TooManyParents {
                target: candidate_system[0],
                parents: candidate_system.len(),
                maximum: usize::BITS as usize - 1,
            })?;
        if candidate_state_count > max_factor_entries {
            return Err(FactorGraphError::IntermediateTooLarge {
                entries: candidate_state_count,
                limit: max_factor_entries,
            }.into());
        }

        let background = (0..self.n_units)
            .filter(|unit| candidate_slot[*unit].is_none())
            .collect::<Vec<_>>();
        let mut background_slot = vec![None; self.n_units];
        for (slot, unit) in background.iter().copied().enumerate() {
            background_slot[unit] = Some(slot);
        }

        // X: candidate past; B: external past; Y: independent candidate-past copy.
        let x_map = (0..self.n_units)
            .map(|unit| candidate_slot[unit].unwrap_or_else(|| {
                candidate_system.len() + background_slot[unit].unwrap()
            }))
            .collect::<Vec<_>>();
        let posterior_variable_count = self.n_units;
        let y_base = posterior_variable_count;
        let y_map = (0..self.n_units)
            .map(|unit| candidate_slot[unit].map(|slot| y_base + slot).unwrap_or_else(|| {
                candidate_system.len() + background_slot[unit].unwrap()
            }))
            .collect::<Vec<_>>();
        let eliminate_posterior = (0..posterior_variable_count).collect::<Vec<_>>();

        let posterior_factors = (0..self.n_units)
            .map(|target| self.observation_log_factor(
                target, current_state, &x_map, max_factor_entries,
            ))
            .collect::<Result<Vec<_>, _>>()?;
        let normalizer = min_fill_log_sum_product(
            &posterior_factors, &eliminate_posterior, max_factor_entries,
        )?;
        if !normalizer.factor.variables().is_empty() {
            return Err(SparseCausalError::UndefinedBackgroundPosterior);
        }
        let posterior_log_normalizer = normalizer.factor.log_values()[0];
        if !posterior_log_normalizer.is_finite() {
            return Err(SparseCausalError::UndefinedBackgroundPosterior);
        }

        let mut forward = vec![1.0; candidate_state_count];
        let mut maximum_target_induced_width = 0usize;
        let mut peak_factor_entries = normalizer.peak_factor_entries;
        let mut factor_multiplications = normalizer.factor_multiplications;

        for target in candidate_system.iter().copied() {
            let target_factor = self.observation_log_factor(
                target, current_state, &y_map, max_factor_entries,
            )?;
            let mut factors = posterior_factors.clone();
            factors.push(target_factor);
            let report = min_fill_log_sum_product(
                &factors, &eliminate_posterior, max_factor_entries,
            )?;
            if report.factor.variables().iter().any(|variable| {
                *variable < y_base || *variable >= y_base + candidate_system.len()
            }) {
                return Err(SparseCausalError::UndefinedBackgroundPosterior);
            }

            let mut cause_factor = vec![0.0; candidate_state_count];
            for (candidate_state, value) in cause_factor.iter_mut().enumerate() {
                let local_index = report.factor.variables().iter().enumerate().fold(
                    0usize,
                    |index, (factor_slot, variable)| {
                        let candidate_slot = variable - y_base;
                        if ((candidate_state >> candidate_slot) & 1) != 0 {
                            index | (1usize << factor_slot)
                        } else { index }
                    },
                );
                *value = (report.factor.log_values()[local_index] - posterior_log_normalizer).exp();
            }
            if cause_factor.iter().any(|value| {
                !value.is_finite() || *value < 0.0 || *value > 1.0 + 1e-12
            }) {
                return Err(SparseCausalError::UndefinedBackgroundPosterior);
            }
            for (joint, factor_value) in forward.iter_mut().zip(cause_factor) {
                *joint *= factor_value;
            }
            maximum_target_induced_width = maximum_target_induced_width.max(report.induced_width);
            peak_factor_entries = peak_factor_entries.max(report.peak_factor_entries);
            factor_multiplications += report.factor_multiplications;
        }

        if forward.iter().any(|value| !value.is_finite()) {
            return Err(SparseCausalError::UndefinedBackgroundPosterior);
        }
        let request_sha256 = self.eq4_request_sha256(
            candidate_system,
            current_state,
            max_factor_entries,
        );
        Ok(SparseEq4Report {
            request_sha256,
            max_factor_entries,
            forward_likelihood: forward,
            posterior_induced_width: normalizer.induced_width,
            maximum_target_induced_width,
            peak_factor_entries,
            factor_multiplications,
        })
    }

    fn eq4_request_sha256(
        &self,
        candidate_system: &[usize],
        current_state: &[bool],
        max_factor_entries: usize,
    ) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SPARSE_EQ4_REQUEST_MAGIC);
        sparse_receipt_string(&mut bytes, SPARSE_IIT4_EQ4_ALGORITHM_ID);
        bytes.extend_from_slice(self.model_sha256().as_bytes());
        sparse_receipt_u64(&mut bytes, current_state.len() as u64);
        bytes.extend(current_state.iter().map(|on| u8::from(*on)));
        sparse_receipt_u64(&mut bytes, candidate_system.len() as u64);
        for unit in candidate_system {
            sparse_receipt_u64(&mut bytes, *unit as u64);
        }
        sparse_receipt_u64(&mut bytes, max_factor_entries as u64);
        sparse_hex_digest(&bytes)
    }

    fn observation_log_factor(
        &self,
        target_unit: usize,
        current_state: &[bool],
        variable_map: &[usize],
        max_factor_entries: usize,
    ) -> Result<BinaryLogFactor, SparseCausalError> {
        let mechanism = &self.mechanisms[target_unit];
        let mut mapped_parents = mechanism.parent_units.iter().copied().enumerate()
            .map(|(parent_slot, parent)| (variable_map[parent], parent_slot))
            .collect::<Vec<_>>();
        mapped_parents.sort_unstable_by_key(|(variable, _)| *variable);
        let entries = 1usize.checked_shl(mapped_parents.len() as u32)
            .ok_or(SparseCausalError::ParentTableSizeOverflow {
                target: target_unit, parents: mapped_parents.len(),
            })?;
        if entries > max_factor_entries {
            return Err(FactorGraphError::IntermediateTooLarge {
                entries, limit: max_factor_entries,
            }.into());
        }
        let variables = mapped_parents.iter().map(|(variable, _)| *variable).collect::<Vec<_>>();
        let mut log_values = Vec::with_capacity(entries);
        for assignment in 0..entries {
            let parent_assignment = mapped_parents.iter().enumerate().fold(
                0usize,
                |index, (factor_slot, (_, original_parent_slot))| {
                    if ((assignment >> factor_slot) & 1) != 0 {
                        index | (1usize << original_parent_slot)
                    } else { index }
                },
            );
            let p_on = mechanism.probability_next_one[parent_assignment];
            let likelihood = if current_state[target_unit] { p_on } else { 1.0 - p_on };
            log_values.push(if likelihood == 0.0 { f64::NEG_INFINITY } else { likelihood.ln() });
        }
        Ok(BinaryLogFactor::new(variables, log_values)?)
    }

    fn check_state(&self, state: &[bool]) -> Result<(), SparseCausalError> {
        if state.len() != self.n_units {
            return Err(SparseCausalError::StateLength {
                expected: self.n_units, actual: state.len(),
            });
        }
        Ok(())
    }

    fn canonical_model_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SPARSE_MODEL_MAGIC);
        bytes.extend_from_slice(&(self.n_units as u64).to_le_bytes());
        for mechanism in &self.mechanisms {
            bytes.extend_from_slice(&(mechanism.target_unit as u64).to_le_bytes());
            bytes.extend_from_slice(&(mechanism.parent_units.len() as u64).to_le_bytes());
            for parent in &mechanism.parent_units {
                bytes.extend_from_slice(&(*parent as u64).to_le_bytes());
            }
            for probability in &mechanism.probability_next_one {
                let canonical = if *probability == 0.0 { 0.0 } else { *probability };
                bytes.extend_from_slice(&canonical.to_bits().to_le_bytes());
            }
        }
        bytes
    }
}

fn sparse_hex_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_ring_model(n_units: usize) -> BinaryCausalModel {
        let mut connectivity = vec![false; n_units * n_units];
        let mut tpm = vec![0.0; (1usize << n_units) * n_units];
        for target in 0..n_units {
            let previous = (target + n_units - 1) % n_units;
            connectivity[target * n_units + target] = true;
            connectivity[previous * n_units + target] = true;
        }
        for state in 0..(1usize << n_units) {
            for target in 0..n_units {
                let previous = (target + n_units - 1) % n_units;
                let self_on = ((state >> target) & 1) as f64;
                let previous_on = ((state >> previous) & 1) as f64;
                tpm[state * n_units + target] = 0.15 + 0.25 * self_on + 0.30 * previous_on;
            }
        }
        BinaryCausalModel::new(n_units, tpm, connectivity).unwrap()
    }

    fn sparse_ring_model(n_units: usize) -> SparseBinaryCausalModel {
        let mechanisms = (0..n_units).map(|target| {
            let previous = (target + n_units - 1) % n_units;
            let (parents, probabilities) = if previous < target {
                (vec![previous, target], vec![0.15, 0.45, 0.40, 0.70])
            } else {
                (vec![target, previous], vec![0.15, 0.40, 0.45, 0.70])
            };
            SparseMechanismCpt::new(target, parents, probabilities).unwrap()
        }).collect();
        SparseBinaryCausalModel::new(n_units, mechanisms).unwrap()
    }

    #[test]
    fn converts_dense_models_and_reproduces_every_local_transition() {
        let dense = dense_ring_model(5);
        let sparse = SparseBinaryCausalModel::from_dense(&dense).unwrap();
        for state_index in 0..dense.n_states() {
            let current = (0..5).map(|unit| ((state_index >> unit) & 1) != 0)
                .collect::<Vec<_>>();
            for target in 0..5 {
                assert_eq!(
                    sparse.probability_next_one(&current, target).unwrap(),
                    dense.probability_next_one(&current, target).unwrap()
                );
            }
        }
    }

    fn brute_force_eq4(
        model: &SparseBinaryCausalModel,
        candidate: &[usize],
        current: &[bool],
    ) -> Vec<f64> {
        // Independent full-joint oracle: enumerate every past state, accumulate
        // P(observed whole-system state | past) by external-background state,
        // normalize that background posterior, then evaluate each candidate
        // target's observation likelihood under every candidate-past state.
        let n = model.n_units();
        let background = (0..n)
            .filter(|unit| !candidate.contains(unit))
            .collect::<Vec<_>>();
        let mut background_mass = vec![0.0; 1usize << background.len()];
        for past_index in 0..(1usize << n) {
            let past = (0..n)
                .map(|unit| ((past_index >> unit) & 1) != 0)
                .collect::<Vec<_>>();
            let mut likelihood = 1.0;
            for target in 0..n {
                let p_on = model.probability_next_one(&past, target).unwrap();
                likelihood *= if current[target] { p_on } else { 1.0 - p_on };
            }
            let background_index = background.iter().enumerate().fold(
                0usize,
                |index, (slot, unit)| {
                    if past[*unit] { index | (1usize << slot) } else { index }
                },
            );
            background_mass[background_index] += likelihood;
        }
        let normalizer = background_mass.iter().sum::<f64>();
        assert!(normalizer.is_finite() && normalizer > 0.0);
        for mass in &mut background_mass {
            *mass /= normalizer;
        }

        let mut forward = vec![1.0; 1usize << candidate.len()];
        for (candidate_state, joint) in forward.iter_mut().enumerate() {
            for target in candidate.iter().copied() {
                let mut cause_factor = 0.0;
                for (background_index, posterior_mass) in
                    background_mass.iter().copied().enumerate()
                {
                    let mut past = current.to_vec();
                    for (slot, unit) in candidate.iter().copied().enumerate() {
                        past[unit] = ((candidate_state >> slot) & 1) != 0;
                    }
                    for (slot, unit) in background.iter().copied().enumerate() {
                        past[unit] = ((background_index >> slot) & 1) != 0;
                    }
                    let p_on = model.probability_next_one(&past, target).unwrap();
                    let likelihood = if current[target] { p_on } else { 1.0 - p_on };
                    cause_factor += posterior_mass * likelihood;
                }
                *joint *= cause_factor;
            }
        }
        forward
    }

    #[test]
    fn sparse_partition_transform_matches_dense_maximum_entropy_cut() {
        let dense = dense_ring_model(5);
        let sparse = SparseBinaryCausalModel::from_dense(&dense).unwrap();
        let candidate = [0, 1, 3];
        let partitions = enumerate_iit4_system_partitions(&candidate, 5, true, 100)
            .unwrap();
        assert!(partitions.iter().any(DirectedSystemPartition::is_total_cut));
        for partition in &partitions {
            let dense_cut = dense.apply_partition(partition).unwrap();
            let sparse_cut = sparse.apply_partition(partition).unwrap().to_dense().unwrap();
            assert_eq!(
                sparse_cut,
                dense_cut,
                "sparse/dense cut mismatch: total={}, parts={:?}, directions={:?}",
                partition.is_total_cut(),
                partition.parts(),
                partition.directions()
            );
        }
    }

    #[test]
    fn sparse_eq4_matches_dense_oracle_for_full_and_embedded_systems() {
        let dense = dense_ring_model(5);
        let sparse = SparseBinaryCausalModel::from_dense(&dense).unwrap();
        let state = [true, false, true, false, true];
        for candidate in [&[0, 1][..], &[0, 2, 4][..], &[0, 1, 2, 3, 4][..]] {
            let expected = dense.forward_cause_likelihood_iit4(candidate, candidate, &state, candidate)
                .unwrap();
            let actual = sparse.forward_cause_likelihood_iit4(candidate, &state, 1 << 16)
                .unwrap();
            assert_eq!(expected.len(), actual.forward_likelihood.len());
            for (index, (expected, actual)) in expected.iter()
                .zip(&actual.forward_likelihood).enumerate()
            {
                assert!(
                    (expected - actual).abs() <= expected.abs().max(1.0) * 1e-12,
                    "candidate={candidate:?}, index={index}: dense={expected:.17}, sparse={actual:.17}"
                );
            }
        }
    }

    #[test]
    fn sparse_eq4_matches_independent_full_joint_posterior_oracle() {
        let sparse = sparse_ring_model(5);
        let state = [true, false, true, false, true];
        for candidate in [&[0, 1][..], &[0, 2, 4][..], &[0, 1, 2, 3, 4][..]] {
            let expected = brute_force_eq4(&sparse, candidate, &state);
            let actual = sparse
                .forward_cause_likelihood_iit4(candidate, &state, 1 << 16)
                .unwrap();
            assert_eq!(expected.len(), actual.forward_likelihood.len());
            for (index, (expected, actual)) in expected
                .iter()
                .zip(&actual.forward_likelihood)
                .enumerate()
            {
                assert!(
                    (expected - actual).abs() <= expected.abs().max(1.0) * 1e-12,
                    "candidate={candidate:?}, index={index}: oracle={expected:.17}, sparse={actual:.17}"
                );
            }
        }
    }

    #[test]
    fn sparse_eq4_scales_to_large_ring_without_dense_global_tpm() {
        for n_units in [64, 256, 512] {
            let sparse = sparse_ring_model(n_units);
            let state = (0..n_units).map(|unit| (unit * 7 + 3) % 5 < 2)
                .collect::<Vec<_>>();
            let report = sparse.forward_cause_likelihood_iit4(&[0, 1, 2, 3], &state, 1 << 16)
                .unwrap();
            assert_eq!(report.forward_likelihood.len(), 16);
            assert!(report.forward_likelihood.iter().all(|value| value.is_finite()));
            assert!(report.peak_factor_entries <= 1 << 16);
            assert!(report.maximum_target_induced_width <= 8);
            assert!(report.posterior_induced_width <= 8);
        }
    }

    #[test]
    fn sparse_model_digest_is_stable_and_separate_from_dense_identity() {
        let sparse = sparse_ring_model(5);
        let repeated = sparse_ring_model(5);
        assert_eq!(sparse.model_sha256(), repeated.model_sha256());
        assert_eq!(
            sparse.evaluation_sha256(&[true, false, true, false, true]).unwrap(),
            repeated.evaluation_sha256(&[true, false, true, false, true]).unwrap()
        );
        assert_eq!(sparse.model_sha256().len(), 64);
    }

    #[test]
    fn sparse_eq4_receipt_binds_request_budget_and_result_values() {
        let sparse = sparse_ring_model(5);
        let state = [true, false, true, false, true];
        let report = sparse.forward_cause_likelihood_iit4(&[0, 1], &state, 1 << 16).unwrap();
        let repeated = sparse.forward_cause_likelihood_iit4(&[0, 1], &state, 1 << 16).unwrap();
        assert_eq!(report.receipt_sha256().unwrap(), repeated.receipt_sha256().unwrap());

        let other_budget = sparse.forward_cause_likelihood_iit4(&[0, 1], &state, 1 << 17).unwrap();
        assert_ne!(report.receipt_sha256().unwrap(), other_budget.receipt_sha256().unwrap());

        let mut tampered = report.clone();
        tampered.forward_likelihood[0] += 0.001;
        assert_ne!(report.receipt_sha256().unwrap(), tampered.receipt_sha256().unwrap());

        let mut non_finite = report;
        non_finite.forward_likelihood[0] = f64::NAN;
        assert!(matches!(
            non_finite.receipt_sha256(),
            Err(SparseCausalError::InvalidReportValue { index: 0, value }) if value.is_nan()
        ));
    }

    #[test]
    fn sparse_model_rejects_invalid_local_tables_and_duplicate_targets() {
        assert!(matches!(
            SparseMechanismCpt::new(0, vec![1, 0], vec![0.1; 4]),
            Err(SparseCausalError::ParentScopeNotStrictlyAscending { .. })
        ));
        assert!(matches!(
            SparseMechanismCpt::new(0, vec![1], vec![0.1]),
            Err(SparseCausalError::ParentTableLength { expected: 2, actual: 1, .. })
        ));
        assert!(matches!(
            SparseMechanismCpt::new(0, vec![], vec![f64::NAN]),
            Err(SparseCausalError::InvalidProbability { .. })
        ));
        let duplicate = vec![
            SparseMechanismCpt::new(0, vec![], vec![0.5]).unwrap(),
            SparseMechanismCpt::new(0, vec![], vec![0.7]).unwrap(),
        ];
        assert!(matches!(
            SparseBinaryCausalModel::new(2, duplicate),
            Err(SparseCausalError::DuplicateTarget { target: 0 })
        ));
    }

    #[test]
    fn sparse_eq4_fails_before_building_a_factor_larger_than_budget() {
        let sparse = sparse_ring_model(6);
        let state = [true, false, true, false, true, false];
        assert!(matches!(
            sparse.forward_cause_likelihood_iit4(&[0, 1], &state, 2),
            Err(SparseCausalError::FactorGraph(
                FactorGraphError::IntermediateTooLarge { entries: 4, limit: 2 }
            ))
        ));
    }
}
