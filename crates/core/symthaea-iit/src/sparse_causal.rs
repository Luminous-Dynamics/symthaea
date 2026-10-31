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
    enumerate_iit4_system_partitions_with_stats, min_fill_log_sum_product,
    BinaryCausalModel, BinaryLogFactor, CausalDirection, DirectedSystemPartition,
    FactorGraphError, FormalismProfile, Iit4IntrinsicSpecificationResult,
    Iit4SystemIntrinsicSpecification, Iit4SystemPhiAnalysis,
    Iit4SystemPhiPartitionSummary, Iit4SystemPhiSearchConfig,
    Iit4SystemPhiStatePairResult, Iit4SystemPhiStatus,
    ModelError, PartitionApplyError, SystemPhiDirectionScore, SystemPhiError,
    SystemPhiPartitionScore, MAX_DENSE_UNITS,
};

const SPARSE_MODEL_MAGIC: &[u8] = b"SYMTHAEA-IIT-SPARSE-BINARY-CAUSAL-MODEL\0v0.1\0";
const SPARSE_EVALUATION_MAGIC: &[u8] = b"SYMTHAEA-IIT-SPARSE-EVALUATION\0v0.1\0";
const SPARSE_EQ4_REQUEST_MAGIC: &[u8] = b"SYMTHAEA-IIT-SPARSE-EQ4-REQUEST\0v0.1\0";
const SPARSE_EQ4_RECEIPT_MAGIC: &[u8] = b"SYMTHAEA-IIT-SPARSE-EQ4-RECEIPT\0v0.3\0";
const SPARSE_PARTITION_SCORE_REQUEST_MAGIC: &[u8] = b"SYMTHAEA-IIT-SPARSE-PARTITION-SCORE-REQUEST\0v0.1\0";
const SPARSE_SYSTEM_SEARCH_REQUEST_MAGIC: &[u8] = b"SYMTHAEA-IIT-SPARSE-SYSTEM-SEARCH-REQUEST\0v0.1\0";
pub const SPARSE_IIT4_2023_SYSTEM_SEARCH_ALGORITHM_ID: &str =
    "symthaea-iit/sparse-iit4-2023-system-search/0.3.0";
const SPARSE_PARTITION_SCORE_RECEIPT_MAGIC: &[u8] = b"SYMTHAEA-IIT-SPARSE-PARTITION-SCORE-RECEIPT\0v0.2\0";
pub const SPARSE_IIT4_EQ4_ALGORITHM_ID: &str = "symthaea-iit/sparse-iit4-eq4-minfill/0.3.0";
pub const SPARSE_IIT4_PARTITION_SCORE_ALGORITHM_ID: &str =
    "symthaea-iit/sparse-iit4-partition-score/0.2.0";

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
    #[error("{subset} contains duplicate unit {unit}")]
    DuplicateSubsetUnit { subset: &'static str, unit: usize },
    #[error("{subset} unit {unit} is not part of the candidate system")]
    SubsetOutsideCandidateSystem { subset: &'static str, unit: usize },
    #[error("candidate-system posterior is undefined for the observed whole-system state")]
    UndefinedBackgroundPosterior,
    #[error("sparse Eq4 report entry {index} is not finite and nonnegative: {value}")]
    InvalidReportValue { index: usize, value: f64 },
    #[error("partition score report field {field} is not finite: {value}")]
    InvalidScoreReportValue { field: &'static str, value: f64 },
    #[error("cannot convert {units} sparse units to dense; maximum is {maximum}")]
    DenseConversionLimit { units: usize, maximum: usize },
    #[error("partition cut-matrix size overflows for {candidate_count} candidate units")]
    PartitionCutMatrixSizeOverflow { candidate_count: usize },
    #[error("specified {direction:?} state has {actual} bits, expected {expected}")]
    SpecifiedStateLength {
        direction: CausalDirection,
        expected: usize,
        actual: usize,
    },
    #[error("system partition cuts no connections and has no EdgeCut normalization")]
    ZeroSeveredConnections,
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
    /// True when this report reused the original model/state posterior prepared by
    /// a surrounding multi-partition analysis instead of contracting it again.
    pub posterior_contraction_reused: bool,
    /// Factor multiplications attributable to preparing the original posterior.
    /// They are included in `factor_multiplications` when this report prepares
    /// its own posterior, and reported separately (without double-counting) when
    /// the prepared posterior is reused.
    pub posterior_factor_multiplications: usize,
    /// Largest preflight bucket table for the observed-state posterior contraction.
    pub posterior_predicted_peak_factor_entries: usize,
    /// Largest preflight bucket table across target-wise likelihood contractions.
    pub maximum_target_predicted_peak_factor_entries: usize,
    /// Maximum predicted factor-table size over all contractions in this request.
    pub predicted_peak_factor_entries: usize,
    /// Largest table actually observed over all contractions in this request.
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
        bytes.push(u8::from(self.posterior_contraction_reused));
        sparse_receipt_u64(&mut bytes, self.posterior_factor_multiplications as u64);
        sparse_receipt_u64(&mut bytes, self.posterior_predicted_peak_factor_entries as u64);
        sparse_receipt_u64(&mut bytes, self.maximum_target_predicted_peak_factor_entries as u64);
        sparse_receipt_u64(&mut bytes, self.predicted_peak_factor_entries as u64);
        sparse_receipt_u64(&mut bytes, self.peak_factor_entries as u64);
        sparse_receipt_u64(&mut bytes, self.factor_multiplications as u64);
        Ok(sparse_hex_digest(&bytes))
    }
}


/// Reusable Eq. 4 posterior over the original, uncut substrate and observation.
/// Candidate-past variables are duplicated in the target query; external variables
/// remain shared. A context is scoped to one model, candidate ordering, state and
/// factor-entry budget.
#[derive(Debug, Clone)]
struct SparseEq4PosteriorContext {
    model_sha256: String,
    candidate_system: Vec<usize>,
    current_state: Vec<bool>,
    max_factor_entries: usize,
    y_base: usize,
    y_map: Vec<usize>,
    posterior_factors: Vec<BinaryLogFactor>,
    eliminate_posterior: Vec<usize>,
    candidate_state_count: usize,
    posterior_log_normalizer: f64,
    posterior_induced_width: usize,
    posterior_predicted_peak_factor_entries: usize,
    posterior_peak_factor_entries: usize,
    posterior_factor_multiplications: usize,
}

/// Values independent of a particular partition or specified cause/effect state.
#[derive(Debug, Clone)]
struct SparseIit4ScoringBaseline {
    posterior: SparseEq4PosteriorContext,
    cause_forward: SparseEq4Report,
    cause_forward_receipt_sha256: String,
    cause_selectivity: Vec<f64>,
    effect_selectivity: Vec<f64>,
}

/// Partition-dependent repertoire outputs computed once and reused for every
/// tied intrinsic cause/effect specified-state pair.
#[derive(Debug, Clone)]
struct SparseIit4PartitionTerms {
    partitioned_cause: SparseEq4Report,
    partitioned_cause_receipt_sha256: String,
    effect_partitioned: Vec<f64>,
    severed_connections: usize,
}

/// Proof-carrying sparse per-partition score report.
///
/// This records one fixed state pair and one directed partition. It is not a
/// complete system phi_s search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SparsePartitionScoreReport {
    pub request_sha256: String,
    pub max_factor_entries: usize,
    pub uncut_cause_receipt_sha256: String,
    pub partitioned_cause_receipt_sha256: String,
    pub score: SystemPhiPartitionScore,
}

impl SparsePartitionScoreReport {
    /// Integrity receipt binding both Eq. 4 contractions and the complete score.
    pub fn receipt_sha256(&self) -> Result<String, SparseCausalError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SPARSE_PARTITION_SCORE_RECEIPT_MAGIC);
        sparse_receipt_string(&mut bytes, SPARSE_IIT4_PARTITION_SCORE_ALGORITHM_ID);
        sparse_receipt_string(&mut bytes, &self.request_sha256);
        sparse_receipt_u64(&mut bytes, self.max_factor_entries as u64);
        sparse_receipt_string(&mut bytes, &self.uncut_cause_receipt_sha256);
        sparse_receipt_string(&mut bytes, &self.partitioned_cause_receipt_sha256);
        sparse_score_f64(&mut bytes, "cause.selectivity", self.score.cause.selectivity)?;
        sparse_score_f64(&mut bytes, "cause.forward", self.score.cause.forward_probability)?;
        sparse_score_f64(
            &mut bytes,
            "cause.partitioned_forward",
            self.score.cause.partitioned_forward_probability,
        )?;
        sparse_score_f64(&mut bytes, "cause.signed_phi", self.score.cause.signed_phi)?;
        sparse_score_f64(&mut bytes, "effect.selectivity", self.score.effect.selectivity)?;
        sparse_score_f64(&mut bytes, "effect.forward", self.score.effect.forward_probability)?;
        sparse_score_f64(
            &mut bytes,
            "effect.partitioned_forward",
            self.score.effect.partitioned_forward_probability,
        )?;
        sparse_score_f64(&mut bytes, "effect.signed_phi", self.score.effect.signed_phi)?;
        sparse_score_f64(&mut bytes, "signed_phi", self.score.signed_phi)?;
        sparse_score_f64(&mut bytes, "phi", self.score.phi)?;
        sparse_score_f64(
            &mut bytes,
            "normalization_factor",
            self.score.normalization_factor,
        )?;
        sparse_score_f64(
            &mut bytes,
            "signed_normalized_phi",
            self.score.signed_normalized_phi,
        )?;
        sparse_score_f64(
            &mut bytes,
            "normalized_phi",
            self.score.normalized_phi,
        )?;
        sparse_receipt_u64(&mut bytes, self.score.severed_connections as u64);
        Ok(sparse_hex_digest(&bytes))
    }
}

fn sparse_score_f64(
    bytes: &mut Vec<u8>,
    field: &'static str,
    value: f64,
) -> Result<(), SparseCausalError> {
    if !value.is_finite() {
        return Err(SparseCausalError::InvalidScoreReportValue { field, value });
    }
    let canonical = if value == 0.0 { 0.0 } else { value };
    bytes.extend_from_slice(&canonical.to_bits().to_le_bytes());
    Ok(())
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
            .ok_or(SparseCausalError::PartitionCutMatrixSizeOverflow { candidate_count })?;
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

    /// Score one IIT 4.0 (2023) system partition using sparse local CPTs.
    ///
    /// This uses Eq. 4 cause inference with the uncut substrate posterior,
    /// Eq. 3 effect conditioning, the sparse maximum-entropy partition transform,
    /// and the dense reference score convention (minimum cause/effect, positive
    /// part, and inverse severed-edge normalization).
    pub fn score_iit4_system_partition(
        &self,
        partition: &DirectedSystemPartition,
        current_state: &[bool],
        system_state: &crate::Iit4SystemState,
        max_factor_entries: usize,
    ) -> Result<SparsePartitionScoreReport, SparseCausalError> {
        let baseline = self.prepare_iit4_scoring_baseline(
            partition.candidate_system(),
            current_state,
            max_factor_entries,
        )?;
        let terms = self.prepare_iit4_partition_terms(partition, &baseline)?;
        self.score_iit4_system_partition_from_terms(partition, system_state, &baseline, &terms)
    }

    fn prepare_iit4_scoring_baseline(
        &self,
        candidate_system: &[usize],
        current_state: &[bool],
        max_factor_entries: usize,
    ) -> Result<SparseIit4ScoringBaseline, SparseCausalError> {
        let posterior = self.prepare_iit4_eq4_posterior(
            candidate_system,
            current_state,
            max_factor_entries,
        )?;
        let cause_forward = self.forward_cause_likelihood_iit4_with_posterior(
            self,
            &posterior,
            false,
        )?;
        let cause_forward_receipt_sha256 = cause_forward.receipt_sha256()?;
        let cause_mass = cause_forward.forward_likelihood.iter().sum::<f64>();
        if !cause_mass.is_finite() || cause_mass <= 0.0 {
            return Err(SparseCausalError::UndefinedBackgroundPosterior);
        }
        let cause_selectivity = cause_forward
            .forward_likelihood
            .iter()
            .map(|value| value / cause_mass)
            .collect::<Vec<_>>();
        let effect_selectivity = self.effect_repertoire_iit4(
            candidate_system,
            candidate_system,
            current_state,
            candidate_system,
            max_factor_entries,
        )?;
        Ok(SparseIit4ScoringBaseline {
            posterior,
            cause_forward,
            cause_forward_receipt_sha256,
            cause_selectivity,
            effect_selectivity,
        })
    }

    fn prepare_iit4_partition_terms(
        &self,
        partition: &DirectedSystemPartition,
        baseline: &SparseIit4ScoringBaseline,
    ) -> Result<SparseIit4PartitionTerms, SparseCausalError> {
        let severed_connections = partition.cut_matrix().iter().filter(|cut| **cut).count();
        if severed_connections == 0 {
            return Err(SparseCausalError::ZeroSeveredConnections);
        }
        let cut_model = self.apply_partition(partition)?;
        let partitioned_cause = self.forward_cause_likelihood_iit4_with_posterior(
            &cut_model,
            &baseline.posterior,
            true,
        )?;
        let partitioned_cause_receipt_sha256 = partitioned_cause.receipt_sha256()?;
        let candidate = partition.candidate_system();
        let effect_partitioned = cut_model.effect_repertoire_iit4(
            candidate,
            candidate,
            &baseline.posterior.current_state,
            candidate,
            baseline.posterior.max_factor_entries,
        )?;
        Ok(SparseIit4PartitionTerms {
            partitioned_cause,
            partitioned_cause_receipt_sha256,
            effect_partitioned,
            severed_connections,
        })
    }

    fn score_iit4_system_partition_from_terms(
        &self,
        partition: &DirectedSystemPartition,
        system_state: &crate::Iit4SystemState,
        baseline: &SparseIit4ScoringBaseline,
        terms: &SparseIit4PartitionTerms,
    ) -> Result<SparsePartitionScoreReport, SparseCausalError> {
        let candidate = partition.candidate_system();
        if system_state.cause_specified_state.len() != candidate.len() {
            return Err(SparseCausalError::SpecifiedStateLength {
                direction: CausalDirection::Cause,
                expected: candidate.len(),
                actual: system_state.cause_specified_state.len(),
            });
        }
        if system_state.effect_specified_state.len() != candidate.len() {
            return Err(SparseCausalError::SpecifiedStateLength {
                direction: CausalDirection::Effect,
                expected: candidate.len(),
                actual: system_state.effect_specified_state.len(),
            });
        }
        let cause_state_index = system_state.cause_specified_state
            .iter()
            .enumerate()
            .fold(0usize, |index, (unit, on)| {
                if *on { index | (1usize << unit) } else { index }
            });
        let effect_state_index = system_state.effect_specified_state
            .iter()
            .enumerate()
            .fold(0usize, |index, (unit, on)| {
                if *on { index | (1usize << unit) } else { index }
            });

        let cause = sparse_direction_score(
            CausalDirection::Cause,
            baseline.cause_selectivity[cause_state_index],
            baseline.cause_forward.forward_likelihood[cause_state_index],
            terms.partitioned_cause.forward_likelihood[cause_state_index],
        );
        let effect = sparse_direction_score(
            CausalDirection::Effect,
            baseline.effect_selectivity[effect_state_index],
            baseline.effect_selectivity[effect_state_index],
            terms.effect_partitioned[effect_state_index],
        );
        let signed_phi = cause.signed_phi.min(effect.signed_phi);
        let normalization_factor = 1.0 / terms.severed_connections as f64;
        let signed_normalized_phi = signed_phi * normalization_factor;

        let score = SystemPhiPartitionScore {
            cause,
            effect,
            signed_phi,
            phi: signed_phi.max(0.0),
            normalization_factor,
            signed_normalized_phi,
            normalized_phi: signed_normalized_phi.max(0.0),
            severed_connections: terms.severed_connections,
        };
        let request_sha256 = self.partition_score_request_sha256(
            partition,
            &baseline.posterior.current_state,
            system_state,
            baseline.posterior.max_factor_entries,
        );
        Ok(SparsePartitionScoreReport {
            request_sha256,
            max_factor_entries: baseline.posterior.max_factor_entries,
            uncut_cause_receipt_sha256: baseline.cause_forward_receipt_sha256.clone(),
            partitioned_cause_receipt_sha256: terms.partitioned_cause_receipt_sha256.clone(),
            score,
        })
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

    /// Exact sparse IIT 4.0 Eq. 3 effect repertoire.
    ///
    /// External parent inputs are clamped to their observed values in
    /// current_state. Candidate-system units outside the mechanism are averaged
    /// uniformly, and target purview units are factorized under the same
    /// conditional-independence semantics as the dense TPM implementation.
    /// The first listed candidate/purview unit is the least-significant state bit.
    pub fn effect_repertoire_iit4(
        &self,
        candidate_system: &[usize],
        mechanism: &[usize],
        current_state: &[bool],
        purview: &[usize],
        max_repertoire_entries: usize,
    ) -> Result<Vec<f64>, SparseCausalError> {
        self.check_state(current_state)?;
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

        let mut mechanism_slots = Vec::with_capacity(mechanism.len());
        let mut seen_mechanism = vec![false; self.n_units];
        for unit in mechanism.iter().copied() {
            if unit >= self.n_units || candidate_slot[unit].is_none() {
                return Err(SparseCausalError::SubsetOutsideCandidateSystem {
                    subset: "mechanism",
                    unit,
                });
            }
            if seen_mechanism[unit] {
                return Err(SparseCausalError::DuplicateSubsetUnit {
                    subset: "mechanism",
                    unit,
                });
            }
            seen_mechanism[unit] = true;
            mechanism_slots.push(candidate_slot[unit].unwrap());
        }
        let mut seen_purview = vec![false; self.n_units];
        for unit in purview.iter().copied() {
            if unit >= self.n_units || candidate_slot[unit].is_none() {
                return Err(SparseCausalError::SubsetOutsideCandidateSystem {
                    subset: "purview",
                    unit,
                });
            }
            if seen_purview[unit] {
                return Err(SparseCausalError::DuplicateSubsetUnit {
                    subset: "purview",
                    unit,
                });
            }
            seen_purview[unit] = true;
        }

        let candidate_state_count = 1usize
            .checked_shl(candidate_system.len() as u32)
            .ok_or(SparseCausalError::ParentTableSizeOverflow {
                target: candidate_system.first().copied().unwrap_or(0),
                parents: candidate_system.len(),
            })?;
        let purview_state_count = 1usize
            .checked_shl(purview.len() as u32)
            .ok_or(SparseCausalError::ParentTableSizeOverflow {
                target: purview.first().copied().unwrap_or(0),
                parents: purview.len(),
            })?;
        if candidate_state_count > max_repertoire_entries {
            return Err(FactorGraphError::IntermediateTooLarge {
                entries: candidate_state_count,
                limit: max_repertoire_entries,
            }.into());
        }
        if purview_state_count > max_repertoire_entries {
            return Err(FactorGraphError::IntermediateTooLarge {
                entries: purview_state_count,
                limit: max_repertoire_entries,
            }.into());
        }
        if purview.is_empty() {
            return Ok(vec![1.0]);
        }

        let unconstrained_count = 1usize
            .checked_shl((candidate_system.len() - mechanism.len()) as u32)
            .ok_or(SparseCausalError::ParentTableSizeOverflow {
                target: candidate_system.first().copied().unwrap_or(0),
                parents: candidate_system.len() - mechanism.len(),
            })?;
        let mut per_purview_unit = vec![0.0; purview.len()];
        for candidate_state_index in 0..candidate_state_count {
            let mechanism_matches = mechanism_slots
                .iter()
                .zip(mechanism)
                .all(|(slot, unit)| {
                    (((candidate_state_index >> *slot) & 1) != 0) == current_state[*unit]
                });
            if !mechanism_matches {
                continue;
            }
            let mut past_state = current_state.to_vec();
            for (slot, unit) in candidate_system.iter().copied().enumerate() {
                past_state[unit] = ((candidate_state_index >> slot) & 1) != 0;
            }
            for (purview_slot, target) in purview.iter().copied().enumerate() {
                per_purview_unit[purview_slot] += self.probability_next_one(&past_state, target)?;
            }
        }
        for probability in &mut per_purview_unit {
            *probability /= unconstrained_count as f64;
        }

        let mut repertoire = vec![1.0; purview_state_count];
        for (future_state, probability) in repertoire.iter_mut().enumerate() {
            for (slot, p_on) in per_purview_unit.iter().copied().enumerate() {
                let target_on = ((future_state >> slot) & 1) != 0;
                *probability *= if target_on { p_on } else { 1.0 - p_on };
            }
        }
        Ok(repertoire)
    }

    /// Compute the IIT 4.0 directional intrinsic specification using sparse CPTs.
    ///
    /// The cause reference is the uniform mean of the forward likelihood vector.
    /// The effect reference is the average Eq. 3 repertoire over all candidate
    /// mechanism states. State selection, tie tolerance, and runner-up semantics
    /// match the dense reference implementation.
    pub fn intrinsic_specification_iit4(
        &self,
        candidate_system: &[usize],
        current_state: &[bool],
        direction: CausalDirection,
        decimal_precision: u32,
        max_factor_entries: usize,
    ) -> Result<Iit4IntrinsicSpecificationResult, SystemPhiError> {
        self.check_state(current_state)?;
        if candidate_system.is_empty() {
            return Err(SystemPhiError::Model(ModelError::EmptyCandidateSystem));
        }
        if decimal_precision > 15 {
            return Err(SystemPhiError::InvalidNumericalPrecision {
                precision: decimal_precision,
            });
        }
        validate_sparse_candidate_system(self, candidate_system)?;
        if candidate_system.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(SystemPhiError::CandidateSystemNotCanonical);
        }

        let state_count = 1usize
            .checked_shl(candidate_system.len() as u32)
            .ok_or_else(|| {
                SystemPhiError::SparseCausal(SparseCausalError::ParentTableSizeOverflow {
                    target: candidate_system[0],
                    parents: candidate_system.len(),
                })
            })?;
        if state_count > max_factor_entries {
            return Err(SystemPhiError::SparseCausal(
                SparseCausalError::FactorGraph(FactorGraphError::IntermediateTooLarge {
                    entries: state_count,
                    limit: max_factor_entries,
                }),
            ));
        }

        let (selectivity, forward, unconstrained_forward) = match direction {
            CausalDirection::Cause => {
                let forward = self
                    .forward_cause_likelihood_iit4(
                        candidate_system,
                        current_state,
                        max_factor_entries,
                    )
                    .map_err(SystemPhiError::SparseCausal)?
                    .forward_likelihood;
                let mass = forward.iter().sum::<f64>();
                if !mass.is_finite() || mass <= 0.0 {
                    return Err(SystemPhiError::SparseCausal(
                        SparseCausalError::UndefinedBackgroundPosterior,
                    ));
                }
                let selectivity = forward.iter().map(|value| value / mass).collect::<Vec<_>>();
                let mean = mass / state_count as f64;
                (selectivity, forward, vec![mean; state_count])
            }
            CausalDirection::Effect => {
                let selectivity = self
                    .effect_repertoire_iit4(
                        candidate_system,
                        candidate_system,
                        current_state,
                        candidate_system,
                        max_factor_entries,
                    )
                    .map_err(SystemPhiError::SparseCausal)?;
                let forward = selectivity.clone();
                let mut unconstrained = vec![0.0; state_count];
                for mechanism_state_index in 0..state_count {
                    let mut mechanism_state = current_state.to_vec();
                    for (slot, unit) in candidate_system.iter().copied().enumerate() {
                        mechanism_state[unit] = ((mechanism_state_index >> slot) & 1) != 0;
                    }
                    let repertoire = self
                        .effect_repertoire_iit4(
                            candidate_system,
                            candidate_system,
                            &mechanism_state,
                            candidate_system,
                            max_factor_entries,
                        )
                        .map_err(SystemPhiError::SparseCausal)?;
                    for (sum, probability) in unconstrained.iter_mut().zip(repertoire) {
                        *sum += probability / state_count as f64;
                    }
                }
                (selectivity, forward, unconstrained)
            }
        };

        let mut information = Vec::with_capacity(state_count);
        for state_index in 0..state_count {
            let forward_probability = forward[state_index];
            let unconstrained_probability = unconstrained_forward[state_index];
            let pointwise = if forward_probability == 0.0 || unconstrained_probability == 0.0 {
                0.0
            } else {
                let ratio = forward_probability / unconstrained_probability;
                if !ratio.is_finite() {
                    return Err(SystemPhiError::NonFiniteIntrinsicSpecification { state_index });
                }
                if ratio == 0.0 {
                    0.0
                } else {
                    selectivity[state_index] * ratio.log2()
                }
            };
            if !pointwise.is_finite() {
                return Err(SystemPhiError::NonFiniteIntrinsicSpecification { state_index });
            }
            information.push(pointwise);
        }

        let mut winner_index = 0usize;
        for index in 1..state_count {
            if information[index] > information[winner_index] {
                winner_index = index;
            }
        }
        let maximum = information[winner_index];
        let epsilon = 10.0_f64.powi(-(decimal_precision as i32));
        let specified_states = information
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| crate::pyphi_eq(*value, maximum, epsilon))
            .map(|(index, _)| {
                (0..candidate_system.len())
                    .map(|slot| ((index >> slot) & 1) != 0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut runner_up_index: Option<usize> = None;
        for index in 0..state_count {
            if index == winner_index {
                continue;
            }
            if runner_up_index
                .map(|best| information[index] > information[best])
                .unwrap_or(true)
            {
                runner_up_index = Some(index);
            }
        }
        let runner_up_state = runner_up_index.map(|index| {
            (0..candidate_system.len())
                .map(|slot| ((index >> slot) & 1) != 0)
                .collect::<Vec<_>>()
        });
        let runner_up_intrinsic_specification =
            runner_up_index.map(|index| information[index]);
        let selection_margin = runner_up_intrinsic_specification
            .map(|runner_up| (maximum - runner_up).max(0.0));

        Ok(Iit4IntrinsicSpecificationResult {
            direction,
            specified_states,
            intrinsic_specification: maximum,
            runner_up_state,
            runner_up_intrinsic_specification,
            selection_margin,
            unconstrained_forward_repertoire: unconstrained_forward,
            decimal_precision,
        })
    }

    /// Select cause and effect intrinsic state families independently.
    pub fn system_intrinsic_specification_iit4(
        &self,
        candidate_system: &[usize],
        current_state: &[bool],
        decimal_precision: u32,
        max_factor_entries: usize,
    ) -> Result<Iit4SystemIntrinsicSpecification, SystemPhiError> {
        let cause = self.intrinsic_specification_iit4(
            candidate_system,
            current_state,
            CausalDirection::Cause,
            decimal_precision,
            max_factor_entries,
        )?;
        let effect = self.intrinsic_specification_iit4(
            candidate_system,
            current_state,
            CausalDirection::Effect,
            decimal_precision,
            max_factor_entries,
        )?;
        Ok(Iit4SystemIntrinsicSpecification { cause, effect })
    }

    /// Bounded exhaustive sparse IIT 4.0 (2023) Integration-level system search.
    ///
    /// The search uses the same intrinsic-state tie families, directed partition
    /// quotient, MIP ranking, and composition-tie ceiling as the dense reference.
    /// The explicit factor-entry limit bounds local/intermediate factors and
    /// candidate-state vectors; system config independently bounds partition
    /// enumeration, state-pair count, and total state-pair x partition scores.
    pub fn analyze_iit4_2023_system_phi_s(
        &self,
        candidate_system: &[usize],
        current_state: &[bool],
        config: Iit4SystemPhiSearchConfig,
        max_factor_entries: usize,
    ) -> Result<Iit4SystemPhiAnalysis, SystemPhiError> {
        self.check_state(current_state)?;
        if candidate_system.is_empty() {
            return Err(SystemPhiError::Model(ModelError::EmptyCandidateSystem));
        }
        validate_sparse_candidate_system(self, candidate_system)?;
        if candidate_system.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(SystemPhiError::CandidateSystemNotCanonical);
        }
        if config.decimal_precision > 15 {
            return Err(SystemPhiError::InvalidNumericalPrecision {
                precision: config.decimal_precision,
            });
        }

        let request_sha256 = self.system_search_request_sha256(
            candidate_system,
            current_state,
            config,
            max_factor_entries,
        );
        let empty_summary = Iit4SystemPhiPartitionSummary::default();
        let certified_upper_bound =
            crate::certified_iit4_2023_system_phi_upper_bound(
                candidate_system.len(),
                config.include_total_cut,
            );
        let null_result = |status, phi_s| Iit4SystemPhiAnalysis {
            formalism: FormalismProfile::Iit4_0_2023,
            candidate_system: candidate_system.to_vec(),
            request_sha256: request_sha256.clone(),
            algorithm_id: SPARSE_IIT4_2023_SYSTEM_SEARCH_ALGORITHM_ID,
            search_config: config,
            certified_upper_bound,
            status,
            phi_s,
            selected_state_pair_index: None,
            tied_state_pair_indices: Vec::new(),
            state_pair_results: Vec::new(),
            partition_summary: empty_summary,
            partition_score_evaluations: 0,
            complete: true,
        };

        if !self.candidate_is_strongly_connected(candidate_system) {
            return Ok(null_result(Iit4SystemPhiStatus::NullDisconnected, Some(0.0)));
        }

        let intrinsic = self.system_intrinsic_specification_iit4(
            candidate_system,
            current_state,
            config.decimal_precision,
            max_factor_entries,
        )?;
        let epsilon = 10.0_f64.powi(-(config.decimal_precision as i32));
        if !crate::pyphi_is_positive(intrinsic.cause.intrinsic_specification, epsilon) {
            return Ok(null_result(Iit4SystemPhiStatus::NullNoCause, Some(0.0)));
        }
        if !crate::pyphi_is_positive(intrinsic.effect.intrinsic_specification, epsilon) {
            return Ok(null_result(Iit4SystemPhiStatus::NullNoEffect, Some(0.0)));
        }

        let state_pair_count = intrinsic
            .cause
            .specified_states
            .len()
            .checked_mul(intrinsic.effect.specified_states.len())
            .ok_or(SystemPhiError::StatePairCounterOverflow)?;
        if state_pair_count > config.max_state_pair_evaluations {
            return Err(SystemPhiError::StatePairBudgetExceeded {
                candidate_pairs: state_pair_count,
                max_state_pairs: config.max_state_pair_evaluations,
            });
        }

        let mut enumeration = enumerate_iit4_system_partitions_with_stats(
            candidate_system,
            self.n_units,
            config.include_total_cut,
            config.max_partition_combinations,
        )?;
        let partition_summary = Iit4SystemPhiPartitionSummary {
            raw_direction_labelings_equivalent: enumeration.raw_direction_assignments_equivalent,
            canonical_direction_assignments_evaluated: enumeration.canonical_direction_assignments_evaluated,
            redundant_direction_assignments_pruned: enumeration.redundant_direction_assignments_pruned,
            unique_non_total_cuts: enumeration.unique_non_total_cuts,
            total_cut_included: enumeration.total_cut_included,
        };
        let partitions = std::mem::take(&mut enumeration.partitions);
        let partition_score_evaluations = partitions
            .len()
            .checked_mul(state_pair_count)
            .ok_or(SystemPhiError::StatePairCounterOverflow)?;
        if partition_score_evaluations > config.max_partition_score_evaluations {
            return Err(SystemPhiError::PartitionScoreBudgetExceeded {
                candidate_evaluations: partition_score_evaluations,
                max_evaluations: config.max_partition_score_evaluations,
            });
        }

        let state_pairs = intrinsic.state_pairs();
        let baseline = self.prepare_iit4_scoring_baseline(
            candidate_system,
            current_state,
            max_factor_entries,
        )?;
        let mut scores_by_state_pair = (0..state_pair_count)
            .map(|_| Vec::with_capacity(partitions.len()))
            .collect::<Vec<_>>();
        for partition in &partitions {
            let terms = self.prepare_iit4_partition_terms(partition, &baseline)?;
            for (state_pair_index, system_state) in state_pairs.iter().enumerate() {
                let report = self.score_iit4_system_partition_from_terms(
                    partition,
                    system_state,
                    &baseline,
                    &terms,
                )?;
                // Validate receipt serialization before moving the score out.
                let _receipt_sha256 = report.receipt_sha256()?;
                let score = report.score;
                if !score.signed_phi.is_finite()
                    || !score.phi.is_finite()
                    || !score.normalized_phi.is_finite()
                {
                    return Err(SystemPhiError::NonFinitePartitionScore.into());
                }
                scores_by_state_pair[state_pair_index].push(score);
            }
        }

        let mut state_pair_results = Vec::with_capacity(state_pair_count);
        for (state_pair_index, system_state) in state_pairs.into_iter().enumerate() {
            let scores = std::mem::take(&mut scores_by_state_pair[state_pair_index]);
            if scores.is_empty() {
                return Err(SystemPhiError::ZeroSeveredConnections.into());
            }
            let winner_index = crate::select_pyphi_sia_partition(&partitions, &scores, epsilon);
            let winner_score = scores[winner_index].clone();
            let competing_normalized = scores
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != winner_index)
                .map(|(_, score)| score.normalized_phi)
                .min_by(f64::total_cmp);
            let partition_margin = competing_normalized
                .map(|runner_up| (runner_up - winner_score.normalized_phi).max(0.0));
            state_pair_results.push(Iit4SystemPhiStatePairResult {
                system_state,
                winning_partition: partitions[winner_index].clone(),
                score: winner_score,
                partition_margin,
            });
        }

        let maximum_phi = state_pair_results
            .iter()
            .map(|result| result.score.phi)
            .max_by(f64::total_cmp)
            .ok_or(SystemPhiError::StatePairCounterOverflow)?;
        let tied_state_pair_indices = state_pair_results
            .iter()
            .enumerate()
            .filter(|(_, result)| crate::pyphi_eq(result.score.phi, maximum_phi, epsilon))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();

        let (status, phi_s, selected_state_pair_index) =
            if tied_state_pair_indices.len() == 1 {
                (
                    Iit4SystemPhiStatus::ExactUniqueStatePair,
                    Some(maximum_phi),
                    Some(tied_state_pair_indices[0]),
                )
            } else if tied_state_pair_indices
                .iter()
                .all(|index| state_pair_results[*index].score.phi == 0.0)
            {
                (
                    Iit4SystemPhiStatus::UnresolvedCanonicalTie,
                    Some(0.0),
                    None,
                )
            } else {
                (
                    Iit4SystemPhiStatus::UnresolvedCompositionTie,
                    None,
                    None,
                )
            };

        Ok(Iit4SystemPhiAnalysis {
            formalism: FormalismProfile::Iit4_0_2023,
            candidate_system: candidate_system.to_vec(),
            request_sha256,
            algorithm_id: SPARSE_IIT4_2023_SYSTEM_SEARCH_ALGORITHM_ID,
            search_config: config,
            certified_upper_bound,
            status,
            phi_s,
            selected_state_pair_index,
            tied_state_pair_indices,
            state_pair_results,
            partition_summary,
            partition_score_evaluations,
            complete: true,
        })
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
        self.forward_cause_likelihood_iit4_with_target_model(
            self,
            candidate_system,
            current_state,
            max_factor_entries,
        )
    }

    /// Partitioned Eq. 4 likelihood using the original substrate posterior.
    ///
    /// The observation posterior is always computed from self (the uncut
    /// substrate). Only candidate-target conditional tables are taken from the
    /// maximum-entropy cut model. This is the proper Eq. 4 cut semantics; using
    /// the cut model to recompute the external-background posterior is not
    /// equivalent.
    pub fn partitioned_forward_cause_likelihood_iit4(
        &self,
        partition: &DirectedSystemPartition,
        current_state: &[bool],
        max_factor_entries: usize,
    ) -> Result<SparseEq4Report, SparseCausalError> {
        let cut_model = self.apply_partition(partition)?;
        self.forward_cause_likelihood_iit4_with_target_model(
            &cut_model,
            partition.candidate_system(),
            current_state,
            max_factor_entries,
        )
    }

    fn prepare_iit4_eq4_posterior(
        &self,
        candidate_system: &[usize],
        current_state: &[bool],
        max_factor_entries: usize,
    ) -> Result<SparseEq4PosteriorContext, SparseCausalError> {
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

        Ok(SparseEq4PosteriorContext {
            model_sha256: self.model_sha256(),
            candidate_system: candidate_system.to_vec(),
            current_state: current_state.to_vec(),
            max_factor_entries,
            y_base,
            y_map,
            posterior_factors,
            eliminate_posterior,
            candidate_state_count,
            posterior_log_normalizer,
            posterior_induced_width: normalizer.induced_width,
            posterior_predicted_peak_factor_entries: normalizer.predicted_peak_factor_entries,
            posterior_peak_factor_entries: normalizer.peak_factor_entries,
            posterior_factor_multiplications: normalizer.factor_multiplications,
        })
    }

    fn forward_cause_likelihood_iit4_with_posterior(
        &self,
        target_model: &Self,
        posterior: &SparseEq4PosteriorContext,
        posterior_contraction_reused: bool,
    ) -> Result<SparseEq4Report, SparseCausalError> {
        if self.model_sha256() != posterior.model_sha256 {
            return Err(SparseCausalError::UndefinedBackgroundPosterior);
        }
        if target_model.n_units != self.n_units {
            return Err(SparseCausalError::InvalidUnitCount {
                units: target_model.n_units,
            });
        }

        let mut forward = vec![1.0; posterior.candidate_state_count];
        let mut maximum_target_induced_width = 0usize;
        let mut maximum_target_predicted_peak_factor_entries = 0usize;
        let mut predicted_peak_factor_entries =
            posterior.posterior_predicted_peak_factor_entries;
        let mut peak_factor_entries = posterior.posterior_peak_factor_entries;
        let mut factor_multiplications = if posterior_contraction_reused {
            0
        } else {
            posterior.posterior_factor_multiplications
        };

        for target in posterior.candidate_system.iter().copied() {
            let target_factor = target_model.observation_log_factor(
                target,
                &posterior.current_state,
                &posterior.y_map,
                posterior.max_factor_entries,
            )?;
            let mut factors = posterior.posterior_factors.clone();
            factors.push(target_factor);
            let report = min_fill_log_sum_product(
                &factors,
                &posterior.eliminate_posterior,
                posterior.max_factor_entries,
            )?;
            if report.factor.variables().iter().any(|variable| {
                *variable < posterior.y_base
                    || *variable >= posterior.y_base + posterior.candidate_system.len()
            }) {
                return Err(SparseCausalError::UndefinedBackgroundPosterior);
            }

            let mut cause_factor = vec![0.0; posterior.candidate_state_count];
            for (candidate_state, value) in cause_factor.iter_mut().enumerate() {
                let local_index = report.factor.variables().iter().enumerate().fold(
                    0usize,
                    |index, (factor_slot, variable)| {
                        let candidate_slot = variable - posterior.y_base;
                        if ((candidate_state >> candidate_slot) & 1) != 0 {
                            index | (1usize << factor_slot)
                        } else { index }
                    },
                );
                *value = (report.factor.log_values()[local_index]
                    - posterior.posterior_log_normalizer).exp();
            }
            if cause_factor.iter().any(|value| {
                !value.is_finite() || *value < 0.0 || *value > 1.0 + 1e-12
            }) {
                return Err(SparseCausalError::UndefinedBackgroundPosterior);
            }
            for (joint, factor_value) in forward.iter_mut().zip(cause_factor) {
                *joint *= factor_value;
            }
            maximum_target_induced_width =
                maximum_target_induced_width.max(report.induced_width);
            maximum_target_predicted_peak_factor_entries =
                maximum_target_predicted_peak_factor_entries
                    .max(report.predicted_peak_factor_entries);
            predicted_peak_factor_entries =
                predicted_peak_factor_entries.max(report.predicted_peak_factor_entries);
            peak_factor_entries = peak_factor_entries.max(report.peak_factor_entries);
            factor_multiplications += report.factor_multiplications;
        }

        if forward.iter().any(|value| !value.is_finite()) {
            return Err(SparseCausalError::UndefinedBackgroundPosterior);
        }
        let request_sha256 = self.eq4_request_sha256(
            target_model,
            &posterior.candidate_system,
            &posterior.current_state,
            posterior.max_factor_entries,
        );
        Ok(SparseEq4Report {
            request_sha256,
            max_factor_entries: posterior.max_factor_entries,
            forward_likelihood: forward,
            posterior_induced_width: posterior.posterior_induced_width,
            maximum_target_induced_width,
            posterior_contraction_reused,
            posterior_factor_multiplications: posterior.posterior_factor_multiplications,
            posterior_predicted_peak_factor_entries:
                posterior.posterior_predicted_peak_factor_entries,
            maximum_target_predicted_peak_factor_entries,
            predicted_peak_factor_entries,
            peak_factor_entries,
            factor_multiplications,
        })
    }

    fn forward_cause_likelihood_iit4_with_target_model(
        &self,
        target_model: &Self,
        candidate_system: &[usize],
        current_state: &[bool],
        max_factor_entries: usize,
    ) -> Result<SparseEq4Report, SparseCausalError> {
        let posterior = self.prepare_iit4_eq4_posterior(
            candidate_system,
            current_state,
            max_factor_entries,
        )?;
        self.forward_cause_likelihood_iit4_with_posterior(
            target_model,
            &posterior,
            false,
        )
    }

    fn partition_score_request_sha256(
        &self,
        partition: &DirectedSystemPartition,
        current_state: &[bool],
        system_state: &crate::Iit4SystemState,
        max_factor_entries: usize,
    ) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SPARSE_PARTITION_SCORE_REQUEST_MAGIC);
        sparse_receipt_string(&mut bytes, SPARSE_IIT4_PARTITION_SCORE_ALGORITHM_ID);
        sparse_receipt_string(&mut bytes, &self.model_sha256());
        sparse_receipt_u64(&mut bytes, current_state.len() as u64);
        bytes.extend(current_state.iter().map(|on| u8::from(*on)));
        sparse_receipt_u64(&mut bytes, partition.candidate_system().len() as u64);
        for unit in partition.candidate_system() {
            sparse_receipt_u64(&mut bytes, *unit as u64);
        }
        sparse_receipt_u64(&mut bytes, partition.cut_matrix().len() as u64);
        bytes.extend(partition.cut_matrix().iter().map(|cut| u8::from(*cut)));
        sparse_receipt_u64(&mut bytes, system_state.cause_specified_state.len() as u64);
        bytes.extend(system_state.cause_specified_state.iter().map(|on| u8::from(*on)));
        sparse_receipt_u64(&mut bytes, system_state.effect_specified_state.len() as u64);
        bytes.extend(system_state.effect_specified_state.iter().map(|on| u8::from(*on)));
        sparse_receipt_u64(&mut bytes, max_factor_entries as u64);
        sparse_hex_digest(&bytes)
    }

    fn eq4_request_sha256(
        &self,
        target_model: &Self,
        candidate_system: &[usize],
        current_state: &[bool],
        max_factor_entries: usize,
    ) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SPARSE_EQ4_REQUEST_MAGIC);
        sparse_receipt_string(&mut bytes, SPARSE_IIT4_EQ4_ALGORITHM_ID);
        bytes.extend_from_slice(self.model_sha256().as_bytes());
        bytes.extend_from_slice(target_model.model_sha256().as_bytes());
        sparse_receipt_u64(&mut bytes, current_state.len() as u64);
        bytes.extend(current_state.iter().map(|on| u8::from(*on)));
        sparse_receipt_u64(&mut bytes, candidate_system.len() as u64);
        for unit in candidate_system {
            sparse_receipt_u64(&mut bytes, *unit as u64);
        }
        sparse_receipt_u64(&mut bytes, max_factor_entries as u64);
        sparse_hex_digest(&bytes)
    }

    fn candidate_is_strongly_connected(&self, candidate_system: &[usize]) -> bool {
        if candidate_system.is_empty() {
            return false;
        }
        let reaches_all = |reverse: bool| {
            let mut visited = vec![false; candidate_system.len()];
            let mut stack = vec![0usize];
            visited[0] = true;
            while let Some(source_local) = stack.pop() {
                let source = candidate_system[source_local];
                for (target_local, target) in candidate_system.iter().copied().enumerate() {
                    if visited[target_local] {
                        continue;
                    }
                    let edge = if reverse {
                        self.mechanisms[source].parent_units.contains(&target)
                    } else {
                        self.mechanisms[target].parent_units.contains(&source)
                    };
                    if edge {
                        visited[target_local] = true;
                        stack.push(target_local);
                    }
                }
            }
            visited.into_iter().all(|seen| seen)
        };
        reaches_all(false) && reaches_all(true)
    }

    fn system_search_request_sha256(
        &self,
        candidate_system: &[usize],
        current_state: &[bool],
        config: Iit4SystemPhiSearchConfig,
        max_factor_entries: usize,
    ) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SPARSE_SYSTEM_SEARCH_REQUEST_MAGIC);
        sparse_receipt_string(&mut bytes, SPARSE_IIT4_2023_SYSTEM_SEARCH_ALGORITHM_ID);
        sparse_receipt_string(&mut bytes, FormalismProfile::Iit4_0_2023.id());
        sparse_receipt_string(&mut bytes, crate::AnalysisQuantity::Iit4SystemPhiS.id());
        sparse_receipt_string(&mut bytes, &self.model_sha256());
        sparse_receipt_u64(&mut bytes, candidate_system.len() as u64);
        for unit in candidate_system {
            sparse_receipt_u64(&mut bytes, *unit as u64);
        }
        sparse_receipt_u64(&mut bytes, current_state.len() as u64);
        bytes.extend(current_state.iter().map(|on| u8::from(*on)));
        sparse_receipt_u64(&mut bytes, config.decimal_precision as u64);
        sparse_receipt_u64(&mut bytes, config.max_partition_combinations as u64);
        sparse_receipt_u64(&mut bytes, config.max_state_pair_evaluations as u64);
        sparse_receipt_u64(&mut bytes, config.max_partition_score_evaluations as u64);
        bytes.push(u8::from(config.include_total_cut));
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

fn validate_sparse_candidate_system(
    model: &SparseBinaryCausalModel,
    candidate_system: &[usize],
) -> Result<(), SparseCausalError> {
    if candidate_system.is_empty() {
        return Err(SparseCausalError::EmptyCandidateSystem);
    }
    for (position, unit) in candidate_system.iter().copied().enumerate() {
        if unit >= model.n_units {
            return Err(SparseCausalError::CandidateUnitOutOfRange {
                unit,
                unit_count: model.n_units,
            });
        }
        if position > 0 && candidate_system[position - 1] == unit {
            return Err(SparseCausalError::DuplicateCandidateUnit { unit });
        }
    }
    Ok(())
}

fn sparse_direction_score(
    direction: CausalDirection,
    selectivity: f64,
    forward_probability: f64,
    partitioned_forward_probability: f64,
) -> SystemPhiDirectionScore {
    let signed_phi = if forward_probability == 0.0 || partitioned_forward_probability == 0.0 {
        0.0
    } else {
        selectivity * (forward_probability / partitioned_forward_probability).log2()
    };
    SystemPhiDirectionScore {
        direction,
        selectivity,
        forward_probability,
        partitioned_forward_probability,
        signed_phi,
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
    use crate::enumerate_iit4_system_partitions;

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
            assert_eq!(sparse_cut.connectivity(), dense_cut.connectivity());
            for (index, (sparse_value, dense_value)) in sparse_cut
                .transition_probabilities()
                .iter()
                .zip(dense_cut.transition_probabilities())
                .enumerate()
            {
                assert!(
                    (sparse_value - dense_value).abs() <= 1e-12,
                    "sparse/dense cut mismatch: total={}, parts={:?}, directions={:?}, cell={index}, sparse={sparse_value:.17}, dense={dense_value:.17}",
                    partition.is_total_cut(),
                    partition.parts(),
                    partition.directions()
                );
            }

            let current_state = [true, false, true, false, true];
            let dense_partitioned = dense
                .partitioned_forward_cause_likelihood_iit4(partition, &current_state)
                .unwrap();
            let sparse_partitioned = sparse
                .partitioned_forward_cause_likelihood_iit4(
                    partition,
                    &current_state,
                    1 << 16,
                )
                .unwrap();
            for (index, (dense_value, sparse_value)) in dense_partitioned
                .iter()
                .zip(&sparse_partitioned.forward_likelihood)
                .enumerate()
            {
                assert!(
                    (dense_value - sparse_value).abs() <= 1e-12,
                    "post-cut Eq4 mismatch: total={}, index={index}, dense={dense_value:.17}, sparse={sparse_value:.17}",
                    partition.is_total_cut()
                );
            }
            let current_state = [true, false, true, false, true];
            let dense_cause = dense_cut
                .forward_cause_likelihood_iit4(
                    &candidate,
                    &candidate,
                    &current_state,
                    &candidate,
                )
                .unwrap();
            let sparse_cause = sparse_cut
                .forward_cause_likelihood_iit4(
                    &candidate,
                    &candidate,
                    &current_state,
                    &candidate,
                )
                .unwrap();
            for (index, (dense_value, sparse_value)) in
                dense_cause.iter().zip(&sparse_cause).enumerate()
            {
                assert!(
                    (dense_value - sparse_value).abs() <= 1e-12,
                    "post-cut Eq4 mismatch: total={}, index={index}, dense={dense_value:.17}, sparse={sparse_value:.17}",
                    partition.is_total_cut()
                );
            }
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
    fn sparse_partition_score_matches_dense_reference_for_embedded_systems() {
        let dense = dense_ring_model(5);
        let sparse = SparseBinaryCausalModel::from_dense(&dense).unwrap();
        let candidate = [0, 1, 3];
        let current_state = [true, false, true, false, true];
        let system_state = crate::Iit4SystemState {
            cause_specified_state: vec![true, false, true],
            effect_specified_state: vec![false, true, false],
        };
        let partitions = enumerate_iit4_system_partitions(&candidate, 5, true, 100)
            .unwrap();
        for partition in &partitions {
            let dense_score = dense
                .score_iit4_system_partition(&partition, &current_state, &system_state)
                .unwrap();
            let sparse_score = sparse
                .score_iit4_system_partition(
                    &partition,
                    &current_state,
                    &system_state,
                    1 << 16,
                )
                .unwrap();
            let sparse_score_value = &sparse_score.score;
            assert_eq!(dense_score.severed_connections, sparse_score_value.severed_connections);
            assert!((dense_score.cause.signed_phi - sparse_score_value.cause.signed_phi).abs() <= 1e-12);
            assert!((dense_score.effect.signed_phi - sparse_score_value.effect.signed_phi).abs() <= 1e-12);
            assert!((dense_score.signed_phi - sparse_score_value.signed_phi).abs() <= 1e-12);
            assert!((dense_score.normalized_phi - sparse_score_value.normalized_phi).abs() <= 1e-12);
            let receipt = sparse_score.receipt_sha256().unwrap();
            assert_eq!(
                receipt,
                sparse.score_iit4_system_partition(
                    &partition, &current_state, &system_state, 1 << 16
                ).unwrap().receipt_sha256().unwrap()
            );
            let mut alternate_state_pair = system_state.clone();
            alternate_state_pair.cause_specified_state[0] =
                !alternate_state_pair.cause_specified_state[0];
            let alternate = sparse.score_iit4_system_partition(
                &partition,
                &current_state,
                &alternate_state_pair,
                1 << 16,
            ).unwrap();
            assert_ne!(receipt, alternate.receipt_sha256().unwrap());
        }
    }

    #[test]
    fn sparse_system_search_reuses_partition_terms_across_state_pairs_with_dense_parity() {
        let dense = dense_ring_model(3);
        let sparse = SparseBinaryCausalModel::from_dense(&dense).unwrap();
        let candidate = [0, 1, 2];
        let current_state = [true, false, true];
        let config = crate::Iit4SystemPhiSearchConfig {
            max_partition_combinations: 100,
            max_state_pair_evaluations: 64,
            max_partition_score_evaluations: 10_000,
            ..crate::Iit4SystemPhiSearchConfig::default()
        };

        let dense_result = dense
            .analyze_iit4_2023_system_phi_s(&candidate, &current_state, config)
            .unwrap();
        let sparse_result = sparse
            .analyze_iit4_2023_system_phi_s(&candidate, &current_state, config, 1 << 16)
            .unwrap();
        assert_eq!(sparse_result.formalism, dense_result.formalism);
        assert_eq!(
            dense_result.algorithm_id,
            crate::IIT4_2023_SYSTEM_SEARCH_ALGORITHM_ID
        );
        assert_eq!(
            sparse_result.algorithm_id,
            SPARSE_IIT4_2023_SYSTEM_SEARCH_ALGORITHM_ID
        );
        assert_ne!(dense_result.algorithm_id, sparse_result.algorithm_id);
        let mut relabeled_receipt = sparse_result.clone();
        relabeled_receipt.algorithm_id = crate::IIT4_2023_SYSTEM_SEARCH_ALGORITHM_ID;
        assert_ne!(
            sparse_result.receipt_sha256().unwrap(),
            relabeled_receipt.receipt_sha256().unwrap()
        );
        assert_eq!(sparse_result.certified_upper_bound, dense_result.certified_upper_bound);
        assert_eq!(sparse_result.certified_upper_bound, Some(candidate.len() * (candidate.len() - 1)));
        assert_eq!(sparse_result.receipt_sha256().unwrap(), sparse
            .analyze_iit4_2023_system_phi_s(&candidate, &current_state, config, 1 << 16)
            .unwrap()
            .receipt_sha256()
            .unwrap());
        assert_eq!(sparse_result.status, dense_result.status);
        assert_eq!(sparse_result.selected_state_pair_index, dense_result.selected_state_pair_index);
        assert_eq!(sparse_result.tied_state_pair_indices, dense_result.tied_state_pair_indices);
        assert_eq!(sparse_result.partition_score_evaluations, dense_result.partition_score_evaluations);
        assert_eq!(sparse_result.state_pair_results.len(), dense_result.state_pair_results.len());
        match (sparse_result.phi_s, dense_result.phi_s) {
            (Some(left), Some(right)) => assert!((left - right).abs() <= 1e-12),
            (None, None) => {}
            other => panic!("dense/sparse phi_s presence differs: {other:?}"),
        }
        for (sparse_pair, dense_pair) in sparse_result
            .state_pair_results
            .iter()
            .zip(&dense_result.state_pair_results)
        {
            assert_eq!(sparse_pair.system_state, dense_pair.system_state);
            assert_eq!(
                sparse_pair.winning_partition.cut_matrix(),
                dense_pair.winning_partition.cut_matrix()
            );
            assert_eq!(sparse_pair.score.severed_connections, dense_pair.score.severed_connections);
            assert!((sparse_pair.score.signed_phi - dense_pair.score.signed_phi).abs() <= 1e-12);
            assert!((sparse_pair.score.normalized_phi - dense_pair.score.normalized_phi).abs() <= 1e-12);
            assert!((sparse_pair.score.cause.signed_phi - dense_pair.score.cause.signed_phi).abs() <= 1e-12);
            assert!((sparse_pair.score.effect.signed_phi - dense_pair.score.effect.signed_phi).abs() <= 1e-12);
        }

        // The base observation posterior is contracted once. Each partition query
        // reuses that exact posterior and each partition's full repertoire terms
        // are then independent of which tied specified-state pair is scored.
        let baseline = sparse
            .prepare_iit4_scoring_baseline(&candidate, &current_state, 1 << 16)
            .unwrap();
        assert!(!baseline.cause_forward.posterior_contraction_reused);
        let partitions = enumerate_iit4_system_partitions(&candidate, 3, false, 100).unwrap();
        assert!(partitions.len() > 1);
        let terms = sparse.prepare_iit4_partition_terms(&partitions[0], &baseline).unwrap();
        assert!(terms.partitioned_cause.posterior_contraction_reused);
        assert_eq!(
            terms.partitioned_cause.posterior_factor_multiplications,
            baseline.cause_forward.posterior_factor_multiplications
        );

        let intrinsic = sparse
            .system_intrinsic_specification_iit4(&candidate, &current_state, 6, 1 << 16)
            .unwrap();
        let pairs = intrinsic.state_pairs();
        for partition in partitions.iter().take(3) {
            let cached_terms = sparse.prepare_iit4_partition_terms(partition, &baseline).unwrap();
            for system_state in pairs.iter().take(3) {
                let cached = sparse
                    .score_iit4_system_partition_from_terms(
                        partition,
                        system_state,
                        &baseline,
                        &cached_terms,
                    )
                    .unwrap();
                let independently_prepared = sparse
                    .score_iit4_system_partition(
                        partition,
                        &current_state,
                        system_state,
                        1 << 16,
                    )
                    .unwrap();
                assert!((cached.score.signed_phi - independently_prepared.score.signed_phi).abs() <= 1e-12);
                assert!((cached.score.normalized_phi - independently_prepared.score.normalized_phi).abs() <= 1e-12);
                assert_eq!(
                    cached.score.severed_connections,
                    independently_prepared.score.severed_connections
                );
                assert_eq!(
                    cached.uncut_cause_receipt_sha256,
                    baseline.cause_forward_receipt_sha256
                );
                assert_eq!(
                    cached.partitioned_cause_receipt_sha256,
                    cached_terms.partitioned_cause_receipt_sha256
                );
                assert_eq!(
                    cached.receipt_sha256().unwrap(),
                    independently_prepared.receipt_sha256().unwrap()
                );
            }
        }
    }

    #[test]
    fn sparse_effect_repertoire_matches_dense_eq3_for_embedded_subsystems() {
        let dense = dense_ring_model(5);
        let sparse = SparseBinaryCausalModel::from_dense(&dense).unwrap();
        let state = [true, false, true, false, true];
        for (candidate, mechanism, purview) in [
            (&[0, 1][..], &[0, 1][..], &[0, 1][..]),
            (&[0, 2, 4][..], &[0][..], &[2, 4][..]),
            (&[0, 1, 2, 3, 4][..], &[1, 3][..], &[0, 2, 4][..]),
        ] {
            let expected = dense.effect_repertoire_iit4(
                candidate, mechanism, &state, purview,
            ).unwrap();
            let actual = sparse.effect_repertoire_iit4(
                candidate, mechanism, &state, purview, 1 << 16,
            ).unwrap();
            assert_eq!(expected.len(), actual.len());
            for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                assert!(
                    (expected - actual).abs() <= 1e-12,
                    "candidate={candidate:?}, mechanism={mechanism:?}, purview={purview:?}, index={index}"
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
        assert_eq!(
            report.predicted_peak_factor_entries,
            report.peak_factor_entries,
            "preflight prediction must match the observed factor-table peak"
        );
        assert!(
            report.posterior_predicted_peak_factor_entries <= report.predicted_peak_factor_entries
                && report.maximum_target_predicted_peak_factor_entries
                    <= report.predicted_peak_factor_entries
        );

        let mut changed_prediction = report.clone();
        changed_prediction.predicted_peak_factor_entries += 1;
        assert_ne!(
            report.receipt_sha256().unwrap(),
            changed_prediction.receipt_sha256().unwrap()
        );

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
