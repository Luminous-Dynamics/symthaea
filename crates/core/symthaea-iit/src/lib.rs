// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A small, explicit causal-model substrate for a future Rust-native IIT engine.
//!
//! # Scope
//!
//! Version 0.1 defines and validates a dense, binary, factored transition-probability
//! model. It deliberately does not calculate IIT phi, system phi_s, structure Phi,
//! a cause-effect structure, or consciousness. These are separate algorithms and
//! must not be inferred from a valid TPM.
//!
//! # Model contract
//!
//! * A state is a bool slice of length n_units.
//! * Unit 0 is the least-significant bit in the state index.
//! * The TPM is row-major [past_state_index][next_unit].
//! * Each TPM cell is P(next_unit = 1 | entire past state).
//! * The factored transition probability assumes next units are conditionally
//!   independent given the current state, as in a state-by-node TPM.
//! * Connectivity is an explicit, directed matrix [source_unit][target_unit].
//!   It is supplied explicitly. Construction rejects a graph that omits a
//!   dependency present in the TPM, while allowing declared extra edges; this
//!   matches PyPhi's conservative connectivity-validation convention.
//! * The model digest covers the exact dense TPM and connectivity matrix, not an
//!   evaluation state or an IIT formalism. The evaluation digest binds model bytes
//!   to one state. Neither digest proves that the model is scientifically correct.
//!
//! This crate is intentionally small before implementing repertoire construction,
//! partitions, minimum-information searches, or formal IIT quantities. Those stages
//! need formalism-specific definitions and independent golden fixtures.

use sha2::{Digest, Sha256};
use thiserror::Error;

/// Maximum dense binary units accepted by this in-memory v0.1 representation.
///
/// This is a representation guard, not an IIT tractability claim. A dense binary
/// TPM stores 2^n * n probabilities, so storage grows exponentially with n.
pub const MAX_DENSE_UNITS: usize = 16;

const MODEL_MAGIC: &[u8] = b"SYMTHAEA-IIT-BINARY-TPM\0v0.1\0";
const EVALUATION_MAGIC: &[u8] = b"SYMTHAEA-IIT-EVALUATION\0v0.1\0";
const ANALYSIS_REQUEST_MAGIC: &[u8] = b"SYMTHAEA-IIT-ANALYSIS-REQUEST\0v0.1\0";

/// Errors caused by invalid model inputs or invalid state requests.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ModelError {
    #[error("unit count must be in 1..=16; got {units}")]
    InvalidUnitCount { units: usize },

    #[error("TPM length mismatch: expected {expected} state-by-unit values, got {actual}")]
    TransitionTableLength { expected: usize, actual: usize },

    #[error("connectivity length mismatch: expected {expected} directed entries, got {actual}")]
    ConnectivityLength { expected: usize, actual: usize },

    #[error("TPM value at flat index {index} must be finite and in [0, 1]; got {value}")]
    InvalidProbability { index: usize, value: f64 },

    #[error("state length mismatch: expected {expected} bits, got {actual}")]
    StateLength { expected: usize, actual: usize },

    #[error("unit index {index} is outside 0..{unit_count}")]
    UnitIndexOutOfRange { index: usize, unit_count: usize },

    #[error("quantity {quantity} is not supported by formalism {formalism}")]
    IncompatibleQuantity { quantity: &'static str, formalism: &'static str },

    #[error("candidate system must contain at least one unit")]
    EmptyCandidateSystem,

    #[error("candidate system repeats unit {unit}")]
    DuplicateCandidateUnit { unit: usize },

    #[error("connectivity omits causal influence {source_unit}->{target_unit} between states {state_without_source} and {state_with_source}")]
    MissingCausalEdge {
        source_unit: usize,
        target_unit: usize,
        state_without_source: usize,
        state_with_source: usize,
    },
}


/// Direction assigned to a block in an IIT 4.x directed set partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitionDirection {
    /// Sever connections from the complement into this block.
    Cause,
    /// Sever connections from this block into the complement.
    Effect,
    /// Sever connections in both directions across this block boundary.
    Bidirectional,
}

/// One directed system-partition candidate in candidate-system-local coordinates.
///
/// cut_matrix[source * n + target] == true means source -> target is severed.
/// parts stores the same blocks in global model unit indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectedSystemPartition {
    candidate_system: Vec<usize>,
    parts: Vec<Vec<usize>>,
    directions: Vec<PartitionDirection>,
    cut_matrix: Vec<bool>,
    total_cut: bool,
}

impl DirectedSystemPartition {
    pub fn candidate_system(&self) -> &[usize] { &self.candidate_system }
    pub fn parts(&self) -> &[Vec<usize>] { &self.parts }
    pub fn directions(&self) -> &[PartitionDirection] { &self.directions }
    pub fn cut_matrix(&self) -> &[bool] { &self.cut_matrix }
    pub fn is_total_cut(&self) -> bool { self.total_cut }
}

/// Errors while applying a directed partition to a causal model.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PartitionApplyError {
    #[error(transparent)]
    Model(#[from] ModelError),

    #[error("partition candidate system contains unit {unit}, outside model range 0..{unit_count}")]
    CandidateUnitOutOfRange { unit: usize, unit_count: usize },

    #[error("partition cut matrix length mismatch: expected {expected}, got {actual}")]
    CutMatrixShapeMismatch { expected: usize, actual: usize },
}

/// Exact counters for one completed directed-partition enumeration.
///
/// Counts distinguish combinatorial work from unique severance matrices, allowing
/// fairer cost-frontier comparisons than reporting only output-list length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionEnumerationReport {
    /// All unique cuts, including TotalCut when requested/required.
    pub partitions: Vec<DirectedSystemPartition>,
    /// Number of block-direction assignments fully materialized and evaluated.
    pub block_direction_assignments_evaluated: usize,
    /// Number of unique non-total cut matrices retained.
    pub unique_non_total_cuts: usize,
    /// Number of assignments whose matrix was already seen and was discarded.
    pub duplicate_cut_matrices_suppressed: usize,
    /// Whether the returned partition list includes the special TotalCut.
    pub total_cut_included: bool,
}

/// Failures from directed system-partition enumeration.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PartitionEnumerationError {
    #[error("candidate system must contain at least one unit")]
    EmptyCandidateSystem,
    #[error("candidate-system unit {unit} is outside 0..{unit_count}")]
    UnitOutOfRange { unit: usize, unit_count: usize },
    #[error("candidate system contains duplicate unit {unit}")]
    DuplicateUnit { unit: usize },
    #[error("directed partition enumeration exceeded its budget of {max_combinations} block-direction assignments")]
    BudgetExceeded { max_combinations: usize },
}

/// Enumerate the IIT 4.x DIRECTED_SET_PARTITION cut family.
///
/// For every non-trivial set partition, each block independently receives a
/// cause, effect, or bidirectional direction. Cause cuts complement -> block;
/// Effect cuts block -> complement; Bidirectional cuts both. Outputs are
/// deduplicated by the local cut matrix, as PyPhi's unique_everseen does for
/// its directed-set-partition candidates.
///
/// TotalCut severs every connection, including self-loops. It is always included
/// for a one-unit system and optional for larger systems. Ordinary directional
/// partitions preserve self-loops.
///
/// max_combinations limits block-direction assignments visited, not just unique
/// output cuts. Exceeding the budget returns an error and no partial list, so an
/// incomplete list cannot be mistaken for exact enumeration.
///
/// This is a candidate generator, not an IIT phi_s calculator.
pub fn enumerate_iit4_system_partitions(
    candidate_system: &[usize],
    unit_count: usize,
    include_total_cut: bool,
    max_combinations: usize,
) -> Result<Vec<DirectedSystemPartition>, PartitionEnumerationError> {
    Ok(enumerate_iit4_system_partitions_with_stats(
        candidate_system, unit_count, include_total_cut, max_combinations,
    )?.partitions)
}

/// Enumerate directed set partitions and return exact counters for the completed run.
///
/// The counter contract is stable for this enumerator version: total-cut insertion is
/// not counted as a block-direction assignment; every fully materialized directional
/// assignment increments the work counter; a duplicate matrix increments the
/// duplicate counter and is not returned twice.
pub fn enumerate_iit4_system_partitions_with_stats(
    candidate_system: &[usize],
    unit_count: usize,
    include_total_cut: bool,
    max_combinations: usize,
) -> Result<PartitionEnumerationReport, PartitionEnumerationError> {
    if candidate_system.is_empty() {
        return Err(PartitionEnumerationError::EmptyCandidateSystem);
    }

    let mut canonical_system = candidate_system.to_vec();
    canonical_system.sort_unstable();
    for (position, unit) in canonical_system.iter().copied().enumerate() {
        if unit >= unit_count {
            return Err(PartitionEnumerationError::UnitOutOfRange { unit, unit_count });
        }
        if position > 0 && canonical_system[position - 1] == unit {
            return Err(PartitionEnumerationError::DuplicateUnit { unit });
        }
    }

    let n = canonical_system.len();
    let mut output = Vec::new();
    let mut seen_cuts: std::collections::HashSet<Vec<bool>> =
        std::collections::HashSet::new();

    if n == 1 || include_total_cut {
        let matrix = vec![true; n * n];
        seen_cuts.insert(matrix.clone());
        output.push(DirectedSystemPartition {
            candidate_system: canonical_system.clone(),
            parts: vec![canonical_system.clone()],
            directions: vec![PartitionDirection::Bidirectional],
            cut_matrix: matrix,
            total_cut: true,
        });
    }

    if n == 1 {
        return Ok(PartitionEnumerationReport {
            partitions: output,
            block_direction_assignments_evaluated: 0,
            unique_non_total_cuts: 0,
            duplicate_cut_matrices_suppressed: 0,
            total_cut_included: true,
        });
    }

    let mut blocks = Vec::<Vec<usize>>::new();
    let mut visited_combinations = 0usize;
    enumerate_set_partitions_recursive(
        0,
        n,
        &canonical_system,
        &mut blocks,
        &mut output,
        &mut seen_cuts,
        &mut visited_combinations,
        max_combinations,
    )?;
    let unique_non_total_cuts = output.iter().filter(|partition| !partition.is_total_cut()).count();
    let total_cut_included = output.iter().any(DirectedSystemPartition::is_total_cut);
    Ok(PartitionEnumerationReport {
        partitions: output,
        block_direction_assignments_evaluated: visited_combinations,
        unique_non_total_cuts,
        duplicate_cut_matrices_suppressed: visited_combinations - unique_non_total_cuts,
        total_cut_included,
    })
}

/// Restricted-growth recursion enumerates each unordered set partition once.
fn enumerate_set_partitions_recursive(
    position: usize,
    n: usize,
    candidate_system: &[usize],
    blocks: &mut Vec<Vec<usize>>,
    output: &mut Vec<DirectedSystemPartition>,
    seen_cuts: &mut std::collections::HashSet<Vec<bool>>,
    visited_combinations: &mut usize,
    max_combinations: usize,
) -> Result<(), PartitionEnumerationError> {
    if position == n {
        if blocks.len() <= 1 {
            return Ok(()); // PyPhi's set-partition path omits the trivial block.
        }
        let mut directions = Vec::with_capacity(blocks.len());
        return enumerate_block_directions_recursive(
            0, candidate_system, blocks, &mut directions, output, seen_cuts,
            visited_combinations, max_combinations,
        );
    }

    // Assign the next unit to each existing block, then to a new final block.
    for block_index in 0..=blocks.len() {
        if block_index == blocks.len() {
            blocks.push(vec![position]);
            enumerate_set_partitions_recursive(
                position + 1, n, candidate_system, blocks, output, seen_cuts,
                visited_combinations, max_combinations,
            )?;
            blocks.pop();
        } else {
            blocks[block_index].push(position);
            enumerate_set_partitions_recursive(
                position + 1, n, candidate_system, blocks, output, seen_cuts,
                visited_combinations, max_combinations,
            )?;
            blocks[block_index].pop();
        }
    }
    Ok(())
}

fn enumerate_block_directions_recursive(
    block_index: usize,
    candidate_system: &[usize],
    blocks: &[Vec<usize>],
    directions: &mut Vec<PartitionDirection>,
    output: &mut Vec<DirectedSystemPartition>,
    seen_cuts: &mut std::collections::HashSet<Vec<bool>>,
    visited_combinations: &mut usize,
    max_combinations: usize,
) -> Result<(), PartitionEnumerationError> {
    if block_index == blocks.len() {
        if *visited_combinations >= max_combinations {
            return Err(PartitionEnumerationError::BudgetExceeded { max_combinations });
        }
        *visited_combinations += 1;

        let n = candidate_system.len();
        let mut cut_matrix = vec![false; n * n];
        for (block, direction) in blocks.iter().zip(directions.iter().copied()) {
            for source in 0..n {
                for target in 0..n {
                    let source_inside = block.contains(&source);
                    let target_inside = block.contains(&target);
                    let cut = match direction {
                        PartitionDirection::Cause => !source_inside && target_inside,
                        PartitionDirection::Effect => source_inside && !target_inside,
                        PartitionDirection::Bidirectional => source_inside != target_inside,
                    };
                    if cut {
                        cut_matrix[source * n + target] = true;
                    }
                }
            }
        }

        if seen_cuts.insert(cut_matrix.clone()) {
            let parts = blocks.iter().map(|block| {
                block.iter().map(|local| candidate_system[*local]).collect()
            }).collect();
            output.push(DirectedSystemPartition {
                candidate_system: candidate_system.to_vec(),
                parts,
                directions: directions.clone(),
                cut_matrix,
                total_cut: false,
            });
        }
        return Ok(());
    }

    for direction in [
        PartitionDirection::Cause,
        PartitionDirection::Effect,
        PartitionDirection::Bidirectional,
    ] {
        directions.push(direction);
        enumerate_block_directions_recursive(
            block_index + 1, candidate_system, blocks, directions, output, seen_cuts,
            visited_combinations, max_combinations,
        )?;
        directions.pop();
    }
    Ok(())
}


/// Errors in probability-repertoire construction.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RepertoireError {
    #[error(transparent)]
    Model(#[from] ModelError),

    #[error("subset {subset} contains duplicate unit {unit}")]
    DuplicateSubsetUnit { subset: &'static str, unit: usize },

    #[error("cause repertoire has zero or non-finite likelihood mass")]
    UndefinedCauseRepertoire,

    #[error("GID distribution lengths differ: forward={forward}, partitioned={partitioned}, selectivity={selectivity}")]
    DistributionLengthMismatch { forward: usize, partitioned: usize, selectivity: usize },

    #[error("GID input distribution {name} has invalid probability at index {index}: {value}")]
    InvalidDistributionProbability { name: &'static str, index: usize, value: f64 },

    #[error("GID input distribution {name} sums to {sum}, outside normalization tolerance")]
    DistributionNotNormalized { name: &'static str, sum: f64 },

    #[error("GID state index {index} outside distribution of length {length}")]
    DistributionStateOutOfRange { index: usize, length: usize },
}

/// Evaluate one point of PyPhi's IIT 4.0 generalized intrinsic difference (GID).
///
/// Formula at selected state i:
/// selectivity[i] * log2(forward[i] / partitioned[i]).
///
/// If either forward or partitioned probability is exactly zero, the pointwise
/// ratio is undefined and PyPhi defines that term to be zero. This function
/// validates the forward/partitioned entries as finite probabilities and the
/// selectivity input as a normalized distribution; it does not infer a formal
/// IIT quantity, choose a partition, or compute system phi_s.
pub fn generalized_intrinsic_difference_at_state(
    forward_repertoire: &[f64],
    partitioned_forward_repertoire: &[f64],
    selectivity_repertoire: &[f64],
    state_index: usize,
) -> Result<f64, RepertoireError> {
    if forward_repertoire.len() != partitioned_forward_repertoire.len()
        || forward_repertoire.len() != selectivity_repertoire.len()
    {
        return Err(RepertoireError::DistributionLengthMismatch {
            forward: forward_repertoire.len(),
            partitioned: partitioned_forward_repertoire.len(),
            selectivity: selectivity_repertoire.len(),
        });
    }
    let length = forward_repertoire.len();
    if state_index >= length {
        return Err(RepertoireError::DistributionStateOutOfRange {
            index: state_index,
            length,
        });
    }

    // Cause-side forward repertoires are likelihood arrays, not normalized
    // distributions; only selectivity is required to sum to one.
    validate_probability_values(forward_repertoire, "forward")?;
    validate_probability_values(partitioned_forward_repertoire, "partitioned")?;
    validate_probability_distribution(selectivity_repertoire, "selectivity")?;

    let forward = forward_repertoire[state_index];
    let partitioned = partitioned_forward_repertoire[state_index];
    if forward == 0.0 || partitioned == 0.0 {
        return Ok(0.0);
    }
    Ok(selectivity_repertoire[state_index] * (forward / partitioned).log2())
}

fn validate_probability_values(
    distribution: &[f64],
    name: &'static str,
) -> Result<(), RepertoireError> {
    for (index, value) in distribution.iter().copied().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(RepertoireError::InvalidDistributionProbability {
                name,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn validate_probability_distribution(
    distribution: &[f64],
    name: &'static str,
) -> Result<(), RepertoireError> {
    validate_probability_values(distribution, name)?;
    let sum: f64 = distribution.iter().sum();
    if (sum - 1.0).abs() > 1e-10 {
        return Err(RepertoireError::DistributionNotNormalized { name, sum });
    }
    Ok(())
}

/// Dense, binary, factored transition-probability model.
///
/// transition_probabilities[state_index * n_units + unit] stores the probability
/// that unit is ON at the next step, given the complete current state indexed by
/// state_index. This format does not encode an IIT formalism or a value of Phi.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryCausalModel {
    n_units: usize,
    transition_probabilities: Vec<f64>,
    /// Row-major directed adjacency: connectivity[source * n_units + target].
    connectivity: Vec<bool>,
}

impl BinaryCausalModel {
    /// Construct a model after validating all shape and probability invariants.
    pub fn new(
        n_units: usize,
        transition_probabilities: Vec<f64>,
        connectivity: Vec<bool>,
    ) -> Result<Self, ModelError> {
        if !(1..=MAX_DENSE_UNITS).contains(&n_units) {
            return Err(ModelError::InvalidUnitCount { units: n_units });
        }

        // n_units is bounded above, so this shift and multiplication are safe.
        let state_count = 1usize << n_units;
        let expected_tpm_len = state_count * n_units;
        if transition_probabilities.len() != expected_tpm_len {
            return Err(ModelError::TransitionTableLength {
                expected: expected_tpm_len,
                actual: transition_probabilities.len(),
            });
        }

        let expected_connectivity_len = n_units * n_units;
        if connectivity.len() != expected_connectivity_len {
            return Err(ModelError::ConnectivityLength {
                expected: expected_connectivity_len,
                actual: connectivity.len(),
            });
        }

        for (index, value) in transition_probabilities.iter().copied().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(ModelError::InvalidProbability { index, value });
            }
        }

        // A declared connectivity matrix may include extra edges, but it must not
        // omit an influence present in the TPM. Use exact inequality as a
        // conservative rule: even a small nonzero numeric effect requires an edge.
        // State pairs differ only in the candidate source unit.
        for source_unit in 0..n_units {
            for target_unit in 0..n_units {
                if connectivity[source_unit * n_units + target_unit] {
                    continue;
                }
                for state_without_source in 0..state_count {
                    if ((state_without_source >> source_unit) & 1) != 0 {
                        continue;
                    }
                    let state_with_source = state_without_source | (1usize << source_unit);
                    let p_without = transition_probabilities
                        [state_without_source * n_units + target_unit];
                    let p_with = transition_probabilities
                        [state_with_source * n_units + target_unit];
                    if p_without != p_with {
                        return Err(ModelError::MissingCausalEdge {
                            source_unit,
                            target_unit,
                            state_without_source,
                            state_with_source,
                        });
                    }
                }
            }
        }

        Ok(Self {
            n_units,
            transition_probabilities,
            connectivity,
        })
    }

    /// Number of binary causal units in the model.
    pub fn n_units(&self) -> usize {
        self.n_units
    }

    /// Number of possible current states (2^n_units).
    pub fn n_states(&self) -> usize {
        1usize << self.n_units
    }

    /// Read the state-by-unit TPM in its documented row-major order.
    pub fn transition_probabilities(&self) -> &[f64] {
        &self.transition_probabilities
    }

    /// Read the declared directed connectivity matrix in source-major order.
    pub fn connectivity(&self) -> &[bool] {
        &self.connectivity
    }

    /// Whether the declared directed edge source -> target is present.
    pub fn has_edge(&self, source: usize, target: usize) -> Result<bool, ModelError> {
        self.check_unit(source)?;
        self.check_unit(target)?;
        Ok(self.connectivity[source * self.n_units + target])
    }

    /// State index with unit 0 as the least-significant bit.
    pub fn state_index(&self, state: &[bool]) -> Result<usize, ModelError> {
        self.check_state(state)?;
        Ok(state.iter().enumerate().fold(0usize, |index, (unit, on)| {
            if *on { index | (1usize << unit) } else { index }
        }))
    }

    /// Probability that target_unit is ON next, conditional on current_state.
    pub fn probability_next_one(
        &self,
        current_state: &[bool],
        target_unit: usize,
    ) -> Result<f64, ModelError> {
        self.check_unit(target_unit)?;
        let state_index = self.state_index(current_state)?;
        Ok(self.transition_probabilities[state_index * self.n_units + target_unit])
    }

    /// Probability of a complete next state under the factored TPM.
    ///
    /// This is the product of per-unit conditional probabilities. It is valid for
    /// a factored TPM with conditional independence of next units given the current
    /// state; it is not a general joint-TPM representation.
    pub fn transition_probability(
        &self,
        current_state: &[bool],
        next_state: &[bool],
    ) -> Result<f64, ModelError> {
        let current_index = self.state_index(current_state)?;
        self.check_state(next_state)?;
        let row_start = current_index * self.n_units;
        let probability = next_state.iter().enumerate().map(|(unit, next_on)| {
            let p_on = self.transition_probabilities[row_start + unit];
            if *next_on { p_on } else { 1.0 - p_on }
        }).product();
        Ok(probability)
    }


    /// Compute an effect repertoire using a product of per-purview-unit marginals.
    ///
    /// For each target unit, fix mechanism values and marginalize the non-mechanism
    /// units over a uniform maximum-entropy background. The full purview distribution
    /// is the product of these per-unit distributions, following PyPhi's factored
    /// repertoire convention. This deliberately does not preserve correlations induced
    /// solely by mixing over the same unconstrained hidden state.
    ///
    /// The result indexes future purview states with the first listed purview unit as
    /// the least-significant bit. This primitive does not compute mechanism phi.
    pub fn effect_repertoire(
        &self,
        mechanism: &[usize],
        current_state: &[bool],
        purview: &[usize],
    ) -> Result<Vec<f64>, RepertoireError> {
        self.check_state(current_state)?;
        self.validate_subset(mechanism, "mechanism")?;
        self.validate_subset(purview, "purview")?;

        let unconstrained_count = 1usize << (self.n_units - mechanism.len());
        let background_weight = 1.0 / unconstrained_count as f64;
        let mut per_unit_p_on = Vec::with_capacity(purview.len());

        for target_unit in purview.iter().copied() {
            let mut p_on = 0.0;
            for past_index in 0..self.n_states() {
                let mechanism_matches = mechanism.iter().all(|unit| {
                    (((past_index >> *unit) & 1) != 0) == current_state[*unit]
                });
                if mechanism_matches {
                    p_on += background_weight
                        * self.transition_probabilities[past_index * self.n_units + target_unit];
                }
            }
            per_unit_p_on.push(p_on);
        }

        let purview_state_count = 1usize << purview.len();
        let mut repertoire = vec![1.0; purview_state_count];
        for future_state in 0..purview_state_count {
            for (slot, p_on) in per_unit_p_on.iter().copied().enumerate() {
                let target_on = ((future_state >> slot) & 1) != 0;
                repertoire[future_state] *= if target_on { p_on } else { 1.0 - p_on };
            }
        }
        Ok(repertoire)
    }

    /// Compute the forward cause likelihood over past purview states.
    ///
    /// For each purview state, average the probability of the observed mechanism
    /// state over all unconstrained background units under a uniform prior. Each
    /// mechanism unit's likelihood factor is marginalized separately, matching the
    /// factored TPM repertoire convention; those factors are then multiplied.
    ///
    /// The result is a likelihood vector and is deliberately **not normalized**
    /// over past purview states. For system-level GID, this is the forward cause
    /// repertoire; its selectivity weighting is supplied separately by the
    /// normalized cause repertoire. The first purview unit is the least-significant
    /// bit in the output index.
    pub fn forward_cause_likelihood(
        &self,
        mechanism: &[usize],
        current_state: &[bool],
        purview: &[usize],
    ) -> Result<Vec<f64>, RepertoireError> {
        self.check_state(current_state)?;
        self.validate_subset(mechanism, "mechanism")?;
        self.validate_subset(purview, "purview")?;

        let purview_state_count = 1usize << purview.len();
        let background_count = 1usize << (self.n_units - purview.len());
        let mut per_mechanism_unit = vec![vec![0.0; purview_state_count]; mechanism.len()];

        for past_index in 0..self.n_states() {
            let purview_state = purview.iter().copied().enumerate().fold(
                0usize,
                |state_index, (slot, unit)| {
                    if ((past_index >> unit) & 1) != 0 {
                        state_index | (1usize << slot)
                    } else {
                        state_index
                    }
                },
            );
            for (mechanism_slot, target_unit) in mechanism.iter().copied().enumerate() {
                let p_on = self.transition_probabilities[past_index * self.n_units + target_unit];
                let likelihood = if current_state[target_unit] { p_on } else { 1.0 - p_on };
                per_mechanism_unit[mechanism_slot][purview_state] += likelihood;
            }
        }

        let mut forward = vec![1.0; purview_state_count];
        for factor in &mut per_mechanism_unit {
            for likelihood in factor.iter_mut() {
                *likelihood /= background_count as f64;
            }
            for (state_likelihood, factor_likelihood) in forward.iter_mut().zip(factor.iter()) {
                *state_likelihood *= factor_likelihood;
            }
        }
        Ok(forward)
    }

    /// Compute a cause repertoire as a normalized product of single-unit cause repertoires.
    ///
    /// Each mechanism unit contributes a distribution over past purview states:
    /// likelihoods are obtained from the unit's TPM column under a uniform prior
    /// over full past states, marginalized to the requested purview, then normalized.
    /// The per-mechanism-unit distributions are combined by pointwise multiplication
    /// and normalized again. If a mechanism unit's observed state has zero likelihood
    /// under every past state, this returns an error rather than NaN.
    ///
    /// This uses the factored multi-unit cause-repertoire convention described in
    /// PyPhi's repertoire documentation; it is a primitive, not a full IIT calculation.
    pub fn cause_repertoire(
        &self,
        mechanism: &[usize],
        current_state: &[bool],
        purview: &[usize],
    ) -> Result<Vec<f64>, RepertoireError> {
        self.check_state(current_state)?;
        self.validate_subset(mechanism, "mechanism")?;
        self.validate_subset(purview, "purview")?;

        let purview_state_count = 1usize << purview.len();
        if mechanism.is_empty() {
            return Ok(vec![1.0 / purview_state_count as f64; purview_state_count]);
        }

        let mut repertoire = vec![1.0; purview_state_count];
        for mechanism_unit in mechanism.iter().copied() {
            let mut single_unit = vec![0.0; purview_state_count];
            for past_index in 0..self.n_states() {
                let purview_state = purview.iter().copied().enumerate().fold(
                    0usize,
                    |state_index, (slot, unit)| {
                        if ((past_index >> unit) & 1) != 0 {
                            state_index | (1usize << slot)
                        } else {
                            state_index
                        }
                    },
                );
                let p_on = self.transition_probabilities
                    [past_index * self.n_units + mechanism_unit];
                let likelihood = if current_state[mechanism_unit] {
                    p_on
                } else {
                    1.0 - p_on
                };
                single_unit[purview_state] += likelihood;
            }

            let single_mass: f64 = single_unit.iter().sum();
            if !single_mass.is_finite() || single_mass <= 0.0 {
                return Err(RepertoireError::UndefinedCauseRepertoire);
            }
            for probability in &mut single_unit {
                *probability /= single_mass;
            }
            for (joint, single) in repertoire.iter_mut().zip(single_unit) {
                *joint *= single;
            }
        }

        let mass: f64 = repertoire.iter().sum();
        if !mass.is_finite() || mass <= 0.0 {
            return Err(RepertoireError::UndefinedCauseRepertoire);
        }
        for probability in &mut repertoire {
            *probability /= mass;
        }
        Ok(repertoire)
    }

    fn validate_subset(&self, subset: &[usize], name: &'static str) -> Result<(), RepertoireError> {
        let mut seen = vec![false; self.n_units];
        for unit in subset.iter().copied() {
            self.check_unit(unit)?;
            if seen[unit] {
                return Err(RepertoireError::DuplicateSubsetUnit { subset: name, unit });
            }
            seen[unit] = true;
        }
        Ok(())
    }


    /// Apply the severance matrix using maximum-entropy interventions on cut inputs.
    ///
    /// For each severed source -> target link, the target's TPM column is averaged
    /// over the two values of the source bit, keeping all other bits fixed. Applying
    /// multiple cuts in sequence is equivalent to averaging over all combinations
    /// of severed input bits because each operation is linear averaging. This yields
    /// a factored TPM for a system with those causal influences severed.
    ///
    /// Only links whose source and target both belong to the partition candidate
    /// system are cut; links to units outside that candidate are left intact so
    /// those external units can be marginalized by repertoire operations.
    ///
    /// This transformation follows the maximum-entropy intervention interpretation
    /// and is still only a TPM transform. It does not compute any IIT quantity.
    pub fn apply_partition(
        &self,
        partition: &DirectedSystemPartition,
    ) -> Result<Self, PartitionApplyError> {
        let candidate = partition.candidate_system();
        let n_candidate = candidate.len();
        let expected_cut_len = n_candidate * n_candidate;
        if partition.cut_matrix().len() != expected_cut_len {
            return Err(PartitionApplyError::CutMatrixShapeMismatch {
                expected: expected_cut_len,
                actual: partition.cut_matrix().len(),
            });
        }
        for unit in candidate.iter().copied() {
            if unit >= self.n_units {
                return Err(PartitionApplyError::CandidateUnitOutOfRange {
                    unit,
                    unit_count: self.n_units,
                });
            }
        }

        let mut tpm = self.transition_probabilities.clone();
        let mut connectivity = self.connectivity.clone();

        for local_source in 0..n_candidate {
            let source = candidate[local_source];
            for local_target in 0..n_candidate {
                if !partition.cut_matrix()[local_source * n_candidate + local_target] {
                    continue;
                }
                let target = candidate[local_target];
                for state_without_source in 0..self.n_states() {
                    if ((state_without_source >> source) & 1) != 0 {
                        continue;
                    }
                    let state_with_source = state_without_source | (1usize << source);
                    let without_offset = state_without_source * self.n_units + target;
                    let with_offset = state_with_source * self.n_units + target;
                    let average = (tpm[without_offset] + tpm[with_offset]) * 0.5;
                    tpm[without_offset] = average;
                    tpm[with_offset] = average;
                }
                connectivity[source * self.n_units + target] = false;
            }
        }

        Self::new(self.n_units, tpm, connectivity).map_err(PartitionApplyError::from)
    }

    /// SHA-256 of the v0.1 canonical binary model encoding.
    ///
    /// Encoding: magic/version bytes; n_units as little-endian u32; each TPM
    /// probability as little-endian IEEE-754 f64 bits in row-major order (negative
    /// zero is normalized to positive zero); then each connectivity bool as 0/1.
    /// This encoding is independent of Rust struct layout and debug formatting.
    pub fn model_sha256(&self) -> String {
        hex_digest(&self.canonical_bytes())
    }

    /// SHA-256 binding the canonical model and one evaluation state.
    pub fn evaluation_sha256(&self, state: &[bool]) -> Result<String, ModelError> {
        self.check_state(state)?;
        let canonical_model = self.canonical_bytes();
        let mut bytes = Vec::with_capacity(
            EVALUATION_MAGIC.len() + canonical_model.len() + state.len(),
        );
        bytes.extend_from_slice(EVALUATION_MAGIC);
        bytes.extend_from_slice(&canonical_model);
        bytes.extend(state.iter().map(|on| u8::from(*on)));
        Ok(hex_digest(&bytes))
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            MODEL_MAGIC.len() + 4 + self.transition_probabilities.len() * 8
                + self.connectivity.len(),
        );
        bytes.extend_from_slice(MODEL_MAGIC);
        bytes.extend_from_slice(&(self.n_units as u32).to_le_bytes());
        for probability in &self.transition_probabilities {
            let canonical = if *probability == 0.0 { 0.0 } else { *probability };
            bytes.extend_from_slice(&canonical.to_bits().to_le_bytes());
        }
        bytes.extend(self.connectivity.iter().map(|edge| u8::from(*edge)));
        bytes
    }

    fn check_state(&self, state: &[bool]) -> Result<(), ModelError> {
        if state.len() != self.n_units {
            return Err(ModelError::StateLength {
                expected: self.n_units,
                actual: state.len(),
            });
        }
        Ok(())
    }

    fn check_unit(&self, index: usize) -> Result<(), ModelError> {
        if index >= self.n_units {
            return Err(ModelError::UnitIndexOutOfRange {
                index,
                unit_count: self.n_units,
            });
        }
        Ok(())
    }
}


/// Explicit IIT formalism profile. Profiles are not interchangeable defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormalismProfile {
    /// Original IIT 3.0 formalism (PyPhi preset: IIT_3_0).
    Iit3_0,
    /// IIT 4.0 as published in 2023, without the later intrinsic-information
    /// requirement in the system-level phi_s measure.
    Iit4_0_2023,
    /// IIT 4.0 with the intrinsic-information requirement described in 2026.
    Iit4_0_2026,
}

impl FormalismProfile {
    /// Stable external identifier intended for manifests and request digests.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Iit3_0 => "IIT_3_0",
            Self::Iit4_0_2023 => "IIT_4_0_2023",
            Self::Iit4_0_2026 => "IIT_4_0_2026",
        }
    }
}

/// Requested result identity. Distinct variants prevent conflating IIT 3.0 Phi,
/// IIT 4.0 system phi_s, mechanism small phi, and structure-level big Phi.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalysisQuantity {
    Iit3SystemPhi,
    Iit4SystemPhiS,
    Iit4CausePhiC,
    Iit4EffectPhiE,
    Iit4CauseEffectStructure,
    Iit4StructureBigPhi,
}

impl AnalysisQuantity {
    /// Stable external identifier intended for manifests and request digests.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Iit3SystemPhi => "iit3_system_phi",
            Self::Iit4SystemPhiS => "iit4_system_phi_s",
            Self::Iit4CausePhiC => "iit4_cause_phi_c",
            Self::Iit4EffectPhiE => "iit4_effect_phi_e",
            Self::Iit4CauseEffectStructure => "iit4_cause_effect_structure",
            Self::Iit4StructureBigPhi => "iit4_structure_big_phi",
        }
    }

    fn compatible_with(self, formalism: FormalismProfile) -> bool {
        match (self, formalism) {
            (Self::Iit3SystemPhi, FormalismProfile::Iit3_0) => true,
            (
                Self::Iit4SystemPhiS
                | Self::Iit4CausePhiC
                | Self::Iit4EffectPhiE
                | Self::Iit4CauseEffectStructure
                | Self::Iit4StructureBigPhi,
                FormalismProfile::Iit4_0_2023 | FormalismProfile::Iit4_0_2026,
            ) => true,
            _ => false,
        }
    }
}

/// Complete identity of one requested calculation; no calculation algorithm is
/// implied by creating or hashing this record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisRequest {
    state: Vec<bool>,
    candidate_system: Vec<usize>,
    formalism: FormalismProfile,
    quantity: AnalysisQuantity,
}

impl AnalysisRequest {
    /// Create a request only when state shape and quantity/formalism are compatible.
    pub fn new(
        model: &BinaryCausalModel,
        state: Vec<bool>,
        mut candidate_system: Vec<usize>,
        formalism: FormalismProfile,
        quantity: AnalysisQuantity,
    ) -> Result<Self, ModelError> {
        model.check_state(&state)?;
        if candidate_system.is_empty() {
            return Err(ModelError::EmptyCandidateSystem);
        }
        let mut seen = vec![false; model.n_units];
        for unit in candidate_system.iter().copied() {
            model.check_unit(unit)?;
            if seen[unit] {
                return Err(ModelError::DuplicateCandidateUnit { unit });
            }
            seen[unit] = true;
        }
        // A subsystem is a set, not an ordered sequence. Canonicalize before
        // hashing so equivalent permutations produce the same request identity.
        candidate_system.sort_unstable();
        if !quantity.compatible_with(formalism) {
            return Err(ModelError::IncompatibleQuantity {
                quantity: quantity.id(),
                formalism: formalism.id(),
            });
        }
        Ok(Self { state, candidate_system, formalism, quantity })
    }

    pub fn state(&self) -> &[bool] {
        &self.state
    }

    pub fn candidate_system(&self) -> &[usize] {
        &self.candidate_system
    }

    pub fn formalism(&self) -> FormalismProfile {
        self.formalism
    }

    pub fn quantity(&self) -> AnalysisQuantity {
        self.quantity
    }

    /// SHA-256 binding model bytes, evaluation state, formalism, and quantity.
    ///
    /// A qualified result receipt should also bind build identity, algorithm
    /// version, numerical policy, and partition-search configuration.
    pub fn request_sha256(&self, model: &BinaryCausalModel) -> String {
        let mut bytes = Vec::with_capacity(
            ANALYSIS_REQUEST_MAGIC.len()
                + model.canonical_bytes().len()
                + self.state.len()
                + 4
                + self.candidate_system.len() * 4
                + self.formalism.id().len()
                + self.quantity.id().len()
                + 2,
        );
        bytes.extend_from_slice(ANALYSIS_REQUEST_MAGIC);
        bytes.extend_from_slice(&model.canonical_bytes());
        bytes.extend(self.state.iter().map(|on| u8::from(*on)));
        bytes.extend_from_slice(&(self.candidate_system.len() as u32).to_le_bytes());
        for unit in &self.candidate_system {
            bytes.extend_from_slice(&(*unit as u32).to_le_bytes());
        }
        bytes.push(0);
        bytes.extend_from_slice(self.formalism.id().as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.quantity.id().as_bytes());
        hex_digest(&bytes)
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        // Writing to a String is infallible.
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_unit_model() -> BinaryCausalModel {
        // State indices 0..3 map to [false,false], [true,false],
        // [false,true], [true,true]; unit 0 is the least-significant bit.
        // Each row contains P(A_next=1), P(B_next=1).
        BinaryCausalModel::new(
            2,
            vec![
                0.1, 0.2,
                0.9, 0.2,
                0.1, 0.8,
                0.9, 0.8,
            ],
            vec![
                true, true,
                false, true,
            ],
        ).expect("valid fixture")
    }

    #[test]
    fn validates_dimensions_and_probabilities() {
        assert!(matches!(
            BinaryCausalModel::new(0, vec![], vec![]),
            Err(ModelError::InvalidUnitCount { .. })
        ));
        assert!(matches!(
            BinaryCausalModel::new(2, vec![0.5; 7], vec![false; 4]),
            Err(ModelError::TransitionTableLength { expected: 8, actual: 7 })
        ));
        assert!(matches!(
            BinaryCausalModel::new(2, vec![0.5; 8], vec![false; 3]),
            Err(ModelError::ConnectivityLength { expected: 4, actual: 3 })
        ));
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
            let mut tpm = vec![0.5; 8];
            tpm[3] = invalid;
            assert!(matches!(
                BinaryCausalModel::new(2, tpm, vec![false; 4]),
                Err(ModelError::InvalidProbability { index: 3, .. })
            ));
        }
    }

    #[test]
    fn uses_documented_little_endian_state_index() {
        let model = two_unit_model();
        assert_eq!(model.state_index(&[false, false]).unwrap(), 0);
        assert_eq!(model.state_index(&[true, false]).unwrap(), 1);
        assert_eq!(model.state_index(&[false, true]).unwrap(), 2);
        assert_eq!(model.state_index(&[true, true]).unwrap(), 3);
        assert_eq!(model.probability_next_one(&[false, true], 0).unwrap(), 0.1);
    }

    #[test]
    fn computes_factored_transition_probabilities() {
        let model = two_unit_model();
        assert!((model.transition_probability(&[true, false], &[true, true]).unwrap() - 0.18).abs() < 1e-12);
        assert!((model.transition_probability(&[true, false], &[false, true]).unwrap() - 0.02).abs() < 1e-12);
        assert!((model.transition_probability(&[false, false], &[false, false]).unwrap() - 0.72).abs() < 1e-12);
    }

    #[test]
    fn preserves_directed_connectivity_semantics() {
        let model = two_unit_model();
        assert!(model.has_edge(0, 1).unwrap());
        assert!(!model.has_edge(1, 0).unwrap());
        assert!(matches!(
            model.has_edge(2, 0),
            Err(ModelError::UnitIndexOutOfRange { index: 2, unit_count: 2 })
        ));
    }

    #[test]
    fn digests_are_deterministic_and_cover_model_state_and_connectivity() {
        let model = two_unit_model();
        let same = two_unit_model();
        assert_eq!(model.model_sha256(), same.model_sha256());
        assert_eq!(
            model.evaluation_sha256(&[true, false]).unwrap(),
            same.evaluation_sha256(&[true, false]).unwrap()
        );
        assert_ne!(
            model.evaluation_sha256(&[true, false]).unwrap(),
            model.evaluation_sha256(&[false, true]).unwrap()
        );

        let changed_tpm = BinaryCausalModel::new(
            2,
            vec![0.11, 0.2, 0.9, 0.2, 0.1, 0.8, 0.9, 0.8],
            vec![true, true, false, true],
        ).unwrap();
        assert_ne!(model.model_sha256(), changed_tpm.model_sha256());

        let changed_connectivity = BinaryCausalModel::new(
            2,
            model.transition_probabilities().to_vec(),
            vec![true, false, false, true],
        ).unwrap();
        assert_ne!(model.model_sha256(), changed_connectivity.model_sha256());
    }

    #[test]
    fn negative_zero_is_canonicalized_for_model_identity() {
        let positive_zero = BinaryCausalModel::new(1, vec![0.0, 1.0], vec![true]).unwrap();
        let negative_zero = BinaryCausalModel::new(1, vec![-0.0, 1.0], vec![true]).unwrap();
        assert_eq!(positive_zero.model_sha256(), negative_zero.model_sha256());
    }

    #[test]
    fn rejects_state_length_mismatch() {
        let model = two_unit_model();
        assert!(matches!(
            model.transition_probability(&[true], &[false, true]),
            Err(ModelError::StateLength { expected: 2, actual: 1 })
        ));
        assert!(matches!(
            model.evaluation_sha256(&[false]),
            Err(ModelError::StateLength { expected: 2, actual: 1 })
        ));
    }
    #[test]
    fn formalism_and_quantity_are_typed_and_compatible() {
        let model = two_unit_model();
        assert!(AnalysisRequest::new(
            &model,
            vec![true, false],
            vec![0, 1],
            FormalismProfile::Iit4_0_2023,
            AnalysisQuantity::Iit4SystemPhiS
        ).is_ok());
        assert!(matches!(
            AnalysisRequest::new(
                &model,
                vec![true, false],
                vec![0, 1],
                FormalismProfile::Iit3_0,
                AnalysisQuantity::Iit4StructureBigPhi
            ),
            Err(ModelError::IncompatibleQuantity { .. })
        ));
        assert!(matches!(
            AnalysisRequest::new(
                &model,
                vec![true],
                vec![0],
                FormalismProfile::Iit4_0_2026,
                AnalysisQuantity::Iit4SystemPhiS
            ),
            Err(ModelError::StateLength { .. })
        ));
        assert_eq!(FormalismProfile::Iit4_0_2023.id(), "IIT_4_0_2023");
        assert_eq!(FormalismProfile::Iit4_0_2026.id(), "IIT_4_0_2026");
        assert_ne!(
            AnalysisQuantity::Iit4SystemPhiS.id(),
            AnalysisQuantity::Iit4StructureBigPhi.id()
        );
        assert_eq!(AnalysisQuantity::Iit4CausePhiC.id(), "iit4_cause_phi_c");
        assert_eq!(AnalysisQuantity::Iit4EffectPhiE.id(), "iit4_effect_phi_e");
        assert!(matches!(
            AnalysisRequest::new(
                &model,
                vec![true, false],
                vec![],
                FormalismProfile::Iit4_0_2023,
                AnalysisQuantity::Iit4SystemPhiS
            ),
            Err(ModelError::EmptyCandidateSystem)
        ));
        assert!(matches!(
            AnalysisRequest::new(
                &model,
                vec![true, false],
                vec![0, 0],
                FormalismProfile::Iit4_0_2023,
                AnalysisQuantity::Iit4SystemPhiS
            ),
            Err(ModelError::DuplicateCandidateUnit { unit: 0 })
        ));
    }

    #[test]
    fn analysis_request_digest_binds_formalism_and_quantity() {
        let model = two_unit_model();
        let req_2023 = AnalysisRequest::new(
            &model,
            vec![true, false],
            vec![0, 1],
            FormalismProfile::Iit4_0_2023,
            AnalysisQuantity::Iit4SystemPhiS,
        ).unwrap();
        let req_2026 = AnalysisRequest::new(
            &model,
            vec![true, false],
            vec![0, 1],
            FormalismProfile::Iit4_0_2026,
            AnalysisQuantity::Iit4SystemPhiS,
        ).unwrap();
        let req_structure = AnalysisRequest::new(
            &model,
            vec![true, false],
            vec![0, 1],
            FormalismProfile::Iit4_0_2023,
            AnalysisQuantity::Iit4StructureBigPhi,
        ).unwrap();
        assert_ne!(req_2023.request_sha256(&model), req_2026.request_sha256(&model));
        assert_ne!(req_2023.request_sha256(&model), req_structure.request_sha256(&model));
        let req_reversed_subset = AnalysisRequest::new(
            &model,
            vec![true, false],
            vec![1, 0],
            FormalismProfile::Iit4_0_2023,
            AnalysisQuantity::Iit4SystemPhiS,
        ).unwrap();
        assert_eq!(
            req_2023.request_sha256(&model),
            req_reversed_subset.request_sha256(&model)
        );
        let req_subset_a = AnalysisRequest::new(
            &model,
            vec![true, false],
            vec![0],
            FormalismProfile::Iit4_0_2023,
            AnalysisQuantity::Iit4SystemPhiS,
        ).unwrap();
        assert_ne!(req_2023.request_sha256(&model), req_subset_a.request_sha256(&model));
    }

    #[test]
    fn identity_model_has_point_mass_cause_and_effect_repertoires() {
        let model = BinaryCausalModel::new(1, vec![0.0, 1.0], vec![true]).unwrap();
        let effect = model.effect_repertoire(&[0], &[true], &[0]).unwrap();
        let cause = model.cause_repertoire(&[0], &[true], &[0]).unwrap();
        assert_eq!(effect, vec![0.0, 1.0]);
        assert_eq!(cause, vec![0.0, 1.0]);

        let effect_off = model.effect_repertoire(&[0], &[false], &[0]).unwrap();
        let cause_off = model.cause_repertoire(&[0], &[false], &[0]).unwrap();
        assert_eq!(effect_off, vec![1.0, 0.0]);
        assert_eq!(cause_off, vec![1.0, 0.0]);
    }

    #[test]
    fn effect_repertoire_averages_each_target_over_unconstrained_units() {
        // Unit B's next-state probability is 0.3 for every past state, independently of A.
        let model = BinaryCausalModel::new(
            2,
            vec![0.1, 0.3, 0.9, 0.3, 0.1, 0.3, 0.9, 0.3],
            vec![true, false, false, false],
        ).unwrap();
        let repertoire = model.effect_repertoire(&[0], &[true, false], &[1]).unwrap();
        assert!((repertoire[0] - 0.7).abs() < 1e-12);
        assert!((repertoire[1] - 0.3).abs() < 1e-12);
        assert!((repertoire.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn effect_repertoire_factorizes_per_unit_instead_of_mixing_hidden_state_correlations() {
        // Given A=ON, B is unconstrained. Both next-unit marginals average to 0.5,
        // so the declared factored repertoire is uniform, not a 50/50 mixture of 00 and 11.
        let model = BinaryCausalModel::new(
            2,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![false, false, true, true],
        ).unwrap();
        let repertoire = model.effect_repertoire(&[0], &[true, false], &[0, 1]).unwrap();
        for probability in repertoire {
            assert!((probability - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn cause_repertoire_is_uniform_when_past_purview_is_irrelevant() {
        // A's next probability depends on past A but not past B; conditioning on current A
        // therefore leaves the B purview at its uniform prior.
        let model = BinaryCausalModel::new(
            2,
            vec![0.1, 0.3, 0.9, 0.3, 0.1, 0.3, 0.9, 0.3],
            vec![true, false, false, false],
        ).unwrap();
        let repertoire = model.cause_repertoire(&[0], &[true, false], &[1]).unwrap();
        assert!((repertoire[0] - 0.5).abs() < 1e-12);
        assert!((repertoire[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn repertoire_rejects_duplicate_subset_units() {
        let model = two_unit_model();
        assert!(matches!(
            model.effect_repertoire(&[0, 0], &[true, false], &[1]),
            Err(RepertoireError::DuplicateSubsetUnit { subset: "mechanism", unit: 0 })
        ));
        assert!(matches!(
            model.cause_repertoire(&[0], &[true, false], &[1, 1]),
            Err(RepertoireError::DuplicateSubsetUnit { subset: "purview", unit: 1 })
        ));
    }

    #[test]
    fn cause_repertoire_fails_closed_for_zero_likelihood() {
        let model = BinaryCausalModel::new(1, vec![0.0, 0.0], vec![true]).unwrap();
        assert!(matches!(
            model.cause_repertoire(&[0], &[true], &[0]),
            Err(RepertoireError::UndefinedCauseRepertoire)
        ));
    }

    #[test]
    fn rejects_connectivity_that_omits_a_tpm_influence() {
        // Unit 1 changes target 1's transition probability, but the declared
        // graph omits edge 1 -> 1. PyPhi likewise rejects such under-specified connectivity.
        assert!(matches!(
            BinaryCausalModel::new(
                2,
                vec![0.1, 0.2, 0.9, 0.2, 0.1, 0.8, 0.9, 0.8],
                vec![true, false, true, false]
            ),
            Err(ModelError::MissingCausalEdge {
                source_unit: 1,
                target_unit: 1,
                ..
            })
        ));
    }

    #[test]
    fn matches_published_pyphi_state_by_node_tpm_ordering() {
        // The canonical two-node table in PyPhi's conventions documentation:
        // states are 00, 10, 01, 11 because unit 0 is the least-significant bit.
        let model = BinaryCausalModel::new(
            2,
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            vec![true; 4],
        ).unwrap();
        assert_eq!(model.probability_next_one(&[false, false], 0).unwrap(), 0.1);
        assert_eq!(model.probability_next_one(&[true, false], 0).unwrap(), 0.3);
        assert_eq!(model.probability_next_one(&[false, true], 0).unwrap(), 0.5);
        assert_eq!(model.probability_next_one(&[true, true], 1).unwrap(), 0.8);

        // Frozen cross-language digest vector for the PyPhi conventions example.
        assert_eq!(
            model.model_sha256(),
            "8e8000cfacf6631f5d229449662b260e07419d76b566d6a78214fe1f4b408894"
        );
        assert_eq!(
            model.evaluation_sha256(&[true, false]).unwrap(),
            "143ea0c7ae2676acb89adbd02f93d40a71ee3e6098bade2421a4a2105176eab8"
        );
        let request = AnalysisRequest::new(
            &model,
            vec![true, false],
            vec![0, 1],
            FormalismProfile::Iit4_0_2023,
            AnalysisQuantity::Iit4SystemPhiS,
        ).unwrap();
        assert_eq!(
            request.request_sha256(&model),
            "1b1d3a2f76139abfd2e5bf89da48376b6fac11420382844be813e6e7ac87f2fd"
        );
    }

    #[test]
    fn rejects_dense_models_above_representation_limit_before_allocation() {
        assert!(matches!(
            BinaryCausalModel::new(MAX_DENSE_UNITS + 1, vec![], vec![]),
            Err(ModelError::InvalidUnitCount { .. })
        ));
    }

    #[test]
    fn canonical_digest_golden_vectors_are_stable_for_cross_language_replay() {
        let model = two_unit_model();
        assert_eq!(
            model.model_sha256(),
            "728eb74b44c4aec7896db360919b6efcb0f06e4b7eb487b2974c486c42a80b62"
        );
        assert_eq!(
            model.evaluation_sha256(&[true, false]).unwrap(),
            "6188b51af953ac133cd3c4af0d628ceba73692ff36c2f37f6e05bdb1531214e2"
        );
        let request = AnalysisRequest::new(
            &model,
            vec![true, false],
            vec![0, 1],
            FormalismProfile::Iit4_0_2023,
            AnalysisQuantity::Iit4SystemPhiS,
        ).unwrap();
        assert_eq!(
            request.request_sha256(&model),
            "8b697dce9b5ad3786d4e15f7d5db39c563bce54ea3676976c07efea3f4698db0"
        );
    }

    #[test]
    fn unit_relabeling_preserves_repertoire_values() {
        let original = two_unit_model();
        let mut permuted_tpm = vec![0.0; 8];
        for new_state_index in 0..4 {
            let new_unit_0_on = (new_state_index & 1) != 0;
            let new_unit_1_on = (new_state_index & 2) != 0;
            // New unit 0 is old unit 1 and new unit 1 is old unit 0.
            let old_state_index = usize::from(new_unit_1_on)
                | (usize::from(new_unit_0_on) << 1);
            for new_target in 0..2 {
                let old_target = 1 - new_target;
                permuted_tpm[new_state_index * 2 + new_target] =
                    original.transition_probabilities[old_state_index * 2 + old_target];
            }
        }

        let mut permuted_connectivity = vec![false; 4];
        for new_source in 0..2 {
            for new_target in 0..2 {
                let old_source = 1 - new_source;
                let old_target = 1 - new_target;
                permuted_connectivity[new_source * 2 + new_target] =
                    original.connectivity[old_source * 2 + old_target];
            }
        }
        let permuted = BinaryCausalModel::new(
            2,
            permuted_tpm,
            permuted_connectivity,
        ).unwrap();

        let original_effect = original
            .effect_repertoire(&[0], &[true, false], &[1])
            .unwrap();
        let permuted_effect = permuted
            .effect_repertoire(&[1], &[false, true], &[0])
            .unwrap();
        let original_cause = original
            .cause_repertoire(&[0], &[true, false], &[1])
            .unwrap();
        let permuted_cause = permuted
            .cause_repertoire(&[1], &[false, true], &[0])
            .unwrap();

        for (left, right) in original_effect.iter().zip(&permuted_effect) {
            assert!((left - right).abs() < 1e-12);
        }
        for (left, right) in original_cause.iter().zip(&permuted_cause) {
            assert!((left - right).abs() < 1e-12);
        }
    }

    #[test]
    fn directed_set_partition_for_two_units_matches_three_unique_cross_edge_cuts() {
        let partitions = enumerate_iit4_system_partitions(&[0, 1], 2, false, 100).unwrap();
        assert_eq!(partitions.len(), 3);
        assert!(partitions.iter().all(|p| !p.is_total_cut()));
        let mut matrices = partitions.iter().map(|p| p.cut_matrix().to_vec()).collect::<Vec<_>>();
        matrices.sort();
        matrices.dedup();
        assert_eq!(matrices.len(), 3);
        for matrix in matrices {
            assert!(!matrix[0]);
            assert!(!matrix[3]); // ordinary cuts preserve self-loops
        }
    }

    #[test]
    fn total_cut_includes_self_loops_and_is_optional_for_larger_systems() {
        let without = enumerate_iit4_system_partitions(&[0, 1], 2, false, 100).unwrap();
        let with = enumerate_iit4_system_partitions(&[0, 1], 2, true, 100).unwrap();
        assert_eq!(with.len(), without.len() + 1);
        assert!(with[0].is_total_cut());
        assert_eq!(with[0].cut_matrix(), &[true, true, true, true]);

        let singleton = enumerate_iit4_system_partitions(&[0], 1, false, 0).unwrap();
        assert_eq!(singleton.len(), 1);
        assert!(singleton[0].is_total_cut());
    }

    #[test]
    fn partition_enumeration_is_invariant_to_candidate_input_order() {
        let ordered = enumerate_iit4_system_partitions(&[0, 1, 2], 3, false, 10_000).unwrap();
        let reversed = enumerate_iit4_system_partitions(&[2, 1, 0], 3, false, 10_000).unwrap();
        assert_eq!(ordered, reversed);
    }

    #[test]
    fn partition_enumeration_rejects_invalid_candidates_and_budget_exhaustion() {
        assert!(matches!(
            enumerate_iit4_system_partitions(&[], 3, false, 100),
            Err(PartitionEnumerationError::EmptyCandidateSystem)
        ));
        assert!(matches!(
            enumerate_iit4_system_partitions(&[0, 0], 2, false, 100),
            Err(PartitionEnumerationError::DuplicateUnit { unit: 0 })
        ));
        assert!(matches!(
            enumerate_iit4_system_partitions(&[2], 2, false, 100),
            Err(PartitionEnumerationError::UnitOutOfRange { unit: 2, unit_count: 2 })
        ));
        assert!(matches!(
            enumerate_iit4_system_partitions(&[0, 1, 2], 3, false, 1),
            Err(PartitionEnumerationError::BudgetExceeded { max_combinations: 1 })
        ));
    }

    #[test]
    fn directed_partition_cuts_follow_cause_effect_direction() {
        let partitions = enumerate_iit4_system_partitions(&[0, 1], 2, false, 100).unwrap();
        let cuts = partitions.iter().map(|p| p.cut_matrix()).collect::<Vec<_>>();
        assert!(cuts.iter().any(|m| *m == &[false, true, false, false][..]));
        assert!(cuts.iter().any(|m| *m == &[false, false, true, false][..]));
        assert!(cuts.iter().any(|m| *m == &[false, true, true, false][..]));
    }

    #[test]
    fn three_unit_directed_set_partition_count_matches_bell_partition_expansion() {
        // Bell(3) has four unordered set partitions; removing the trivial block
        // leaves 3 two-block partitions and one three-block partition:
        // 3 * 3^2 + 1 * 3^3 = 54 block-direction assignments.
        let without_total = enumerate_iit4_system_partitions(&[0, 1, 2], 3, false, 54).unwrap();
        assert_eq!(without_total.len(), 22); // Deduplicated matrices, not raw assignments.
        assert!(without_total.iter().all(|p| !p.is_total_cut()));
        let with_total = enumerate_iit4_system_partitions(&[0, 1, 2], 3, true, 54).unwrap();
        assert_eq!(with_total.len(), 23);
        assert!(with_total[0].is_total_cut());
    }

    #[test]
    fn generalized_intrinsic_difference_matches_published_gid_formula_with_unnormalized_likelihoods() {
        // Forward cause repertoires are likelihoods and need not sum to one.
        let gid = generalized_intrinsic_difference_at_state(
            &[0.1, 0.8],
            &[0.2, 0.4],
            &[1.0 / 3.0, 2.0 / 3.0],
            1,
        ).unwrap();
        assert!((gid - 2.0 / 3.0).abs() < 1e-12); // (2/3) * log2(0.8 / 0.4)
    }

    #[test]
    fn generalized_intrinsic_difference_preserves_negative_values() {
        let gid = generalized_intrinsic_difference_at_state(
            &[0.1, 0.9],
            &[0.2, 0.8],
            &[0.2, 0.8],
            0,
        ).unwrap();
        assert!((gid + 0.2).abs() < 1e-12); // 0.2 * log2(0.1 / 0.2)
    }

    #[test]
    fn generalized_intrinsic_difference_returns_zero_for_undefined_zero_ratios() {
        assert_eq!(
            generalized_intrinsic_difference_at_state(
                &[0.0, 1.0],
                &[0.0, 1.0],
                &[0.4, 0.6],
                0,
            ).unwrap(),
            0.0
        );
        assert_eq!(
            generalized_intrinsic_difference_at_state(
                &[0.5, 0.5],
                &[0.0, 1.0],
                &[0.5, 0.5],
                0,
            ).unwrap(),
            0.0
        );
    }

    #[test]
    fn generalized_intrinsic_difference_rejects_invalid_distributions_and_state() {
        assert!(matches!(
            generalized_intrinsic_difference_at_state(&[0.5], &[0.5, 0.5], &[1.0], 0),
            Err(RepertoireError::DistributionLengthMismatch { .. })
        ));
        assert!(matches!(
            generalized_intrinsic_difference_at_state(&[0.5, 0.5], &[0.5, 0.5], &[0.5, 0.5], 2),
            Err(RepertoireError::DistributionStateOutOfRange { index: 2, length: 2 })
        ));
        assert!(matches!(
            generalized_intrinsic_difference_at_state(&[0.5, 0.5], &[0.5, 0.5], &[0.6, 0.6], 0),
            Err(RepertoireError::DistributionNotNormalized { name: "selectivity", .. })
        ));
        assert!(matches!(
            generalized_intrinsic_difference_at_state(&[f64::NAN, 1.0], &[0.0, 1.0], &[0.0, 1.0], 0),
            Err(RepertoireError::InvalidDistributionProbability { name: "forward", index: 0, .. })
        ));
    }

    #[test]
    fn four_and_five_unit_partition_counts_pin_combinatorial_growth() {
        // Raw assignments are sum_k S(n,k) * (3^k), excluding the one-block
        // partition: n=4 gives 306, n=5 gives 1,863. Deduplication reduces
        // these to 150 and 1,061 distinct directed cut matrices respectively.
        let four = enumerate_iit4_system_partitions(&[0, 1, 2, 3], 4, false, 306).unwrap();
        assert_eq!(four.len(), 150);

        let five = enumerate_iit4_system_partitions(&[0, 1, 2, 3, 4], 5, false, 1_863).unwrap();
        assert_eq!(five.len(), 1_061);

        assert!(matches!(
            enumerate_iit4_system_partitions(&[0, 1, 2, 3, 4], 5, false, 1_862),
            Err(PartitionEnumerationError::BudgetExceeded { max_combinations: 1_862 })
        ));
    }

    #[test]
    fn applying_partition_averages_only_the_cut_causal_input() {
        // A drives both its own next state and B's next state. B does not
        // influence A or itself in this TPM; its declared self-edge is extra.
        let model = BinaryCausalModel::new(
            2,
            vec![
                0.1, 0.2,
                0.9, 0.8,
                0.1, 0.2,
                0.9, 0.8,
            ],
            vec![
                true, true,
                false, true,
            ],
        ).unwrap();
        let partitions = enumerate_iit4_system_partitions(&[0, 1], 2, false, 100).unwrap();
        let a_to_b = partitions.iter().find(|p| {
            p.cut_matrix() == &[false, true, false, false][..]
        }).unwrap();

        let cut = model.apply_partition(a_to_b).unwrap();
        assert!(!cut.has_edge(0, 1).unwrap());
        assert!((cut.probability_next_one(&[false, false], 1).unwrap() - 0.5).abs() < 1e-12);
        assert!((cut.probability_next_one(&[true, false], 1).unwrap() - 0.5).abs() < 1e-12);
        assert!((cut.probability_next_one(&[false, true], 1).unwrap() - 0.5).abs() < 1e-12);
        assert!((cut.probability_next_one(&[true, true], 1).unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(cut.probability_next_one(&[false, false], 0).unwrap(), 0.1);
        assert_eq!(cut.probability_next_one(&[true, false], 0).unwrap(), 0.9);
    }

    #[test]
    fn total_cut_averages_all_candidate_inputs_including_self_loops() {
        let model = two_unit_model();
        let total = enumerate_iit4_system_partitions(&[0, 1], 2, true, 100).unwrap()
            .into_iter().find(|p| p.is_total_cut()).unwrap();
        let cut = model.apply_partition(&total).unwrap();
        for current in [&[false, false][..], &[true, false][..], &[false, true][..], &[true, true][..]] {
            assert!((cut.probability_next_one(current, 0).unwrap() - 0.5).abs() < 1e-12);
            assert!((cut.probability_next_one(current, 1).unwrap() - 0.5).abs() < 1e-12);
        }
        assert!(cut.connectivity().iter().all(|edge| !edge));
    }

    #[test]
    fn partition_application_keeps_external_unit_edges_unmodified() {
        let model = BinaryCausalModel::new(
            3,
            vec![
                0.1, 0.2, 0.3,  0.9, 0.2, 0.3,
                0.1, 0.8, 0.3,  0.9, 0.8, 0.3,
                0.1, 0.2, 0.7,  0.9, 0.2, 0.7,
                0.1, 0.8, 0.7,  0.9, 0.8, 0.7,
            ],
            vec![
                true, true, false,
                false, true, false,
                false, false, true,
            ],
        ).unwrap();
        let total_a = enumerate_iit4_system_partitions(&[0], 3, false, 0).unwrap()
            .into_iter().next().unwrap();
        let cut = model.apply_partition(&total_a).unwrap();
        assert!(!cut.has_edge(0, 0).unwrap());
        assert!(cut.has_edge(0, 1).unwrap());
        assert!((cut.probability_next_one(&[false, false, false], 1).unwrap() - 0.2).abs() < 1e-12);
        assert!((cut.probability_next_one(&[true, false, false], 1).unwrap() - 0.2).abs() < 1e-12);
        assert_eq!(cut.probability_next_one(&[false, false, false], 2).unwrap(), 0.3);
    }

    #[test]
    fn forward_cause_likelihood_is_unnormalized_and_matches_complete_tpm_rows() {
        let model = two_unit_model();
        let forward = model.forward_cause_likelihood(
            &[0, 1],
            &[true, false],
            &[0, 1],
        ).unwrap();
        let expected = [0.08, 0.72, 0.02, 0.18];
        for (actual, expected) in forward.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        // Its role is a likelihood vector; its entries are not generally normalized.
        assert!((forward.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn forward_cause_likelihood_marginalizes_hidden_background_units() {
        let model = two_unit_model();
        let forward = model.forward_cause_likelihood(
            &[0],
            &[true, false],
            &[1],
        ).unwrap();
        assert!((forward[0] - 0.5).abs() < 1e-12);
        assert!((forward[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn empty_mechanism_has_unit_forward_cause_likelihood() {
        let model = two_unit_model();
        assert_eq!(
            model.forward_cause_likelihood(&[], &[true, false], &[0, 1]).unwrap(),
            vec![1.0; 4]
        );
    }

    #[test]
    fn partition_enumeration_report_distinguishes_work_from_unique_cuts() {
        let report = enumerate_iit4_system_partitions_with_stats(&[0, 1, 2], 3, false, 54).unwrap();
        assert_eq!(report.block_direction_assignments_evaluated, 54);
        assert_eq!(report.unique_non_total_cuts, 22);
        assert_eq!(report.duplicate_cut_matrices_suppressed, 32);
        assert!(!report.total_cut_included);
        assert_eq!(report.partitions.len(), 22);

        let with_total = enumerate_iit4_system_partitions_with_stats(&[0, 1, 2], 3, true, 54).unwrap();
        assert_eq!(with_total.block_direction_assignments_evaluated, 54);
        assert_eq!(with_total.unique_non_total_cuts, 22);
        assert_eq!(with_total.duplicate_cut_matrices_suppressed, 32);
        assert!(with_total.total_cut_included);
        assert_eq!(with_total.partitions.len(), 23);
    }

    #[test]
    fn partition_enumeration_report_counts_large_duplicate_pressure() {
        let report = enumerate_iit4_system_partitions_with_stats(&[0, 1, 2, 3, 4], 5, false, 1_863).unwrap();
        assert_eq!(report.block_direction_assignments_evaluated, 1_863);
        assert_eq!(report.unique_non_total_cuts, 1_061);
        assert_eq!(report.duplicate_cut_matrices_suppressed, 802);
    }

}
