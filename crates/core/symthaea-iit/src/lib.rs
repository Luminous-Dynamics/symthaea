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

/// IIT 4.0 causal direction for a fixed system-level partition evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalDirection {
    /// Bayesian-inverted cause likelihood over candidate past states.
    Cause,
    /// Forward effect repertoire over candidate future states.
    Effect,
}

/// Both cause and effect specified-state tie families for one candidate system.
#[derive(Debug, Clone, PartialEq)]
pub struct Iit4SystemIntrinsicSpecification {
    pub cause: Iit4IntrinsicSpecificationResult,
    pub effect: Iit4IntrinsicSpecificationResult,
}

impl Iit4SystemIntrinsicSpecification {
    /// Cartesian product of every tied cause/effect state, preserving stable order.
    ///
    /// IIT 4.0's SIA tie cascade must score each pair; choosing one state
    /// independently from either direction can change the selected MIP.
    pub fn state_pairs(&self) -> Vec<Iit4SystemState> {
        let mut pairs = Vec::with_capacity(
            self.cause.specified_states.len() * self.effect.specified_states.len(),
        );
        for cause_state in &self.cause.specified_states {
            for effect_state in &self.effect.specified_states {
                pairs.push(Iit4SystemState {
                    cause_specified_state: cause_state.clone(),
                    effect_specified_state: effect_state.clone(),
                });
            }
        }
        pairs
    }
}

/// Result of maximizing one direction's IIT 4.0 system intrinsic specification.
#[derive(Debug, Clone, PartialEq)]
pub struct Iit4IntrinsicSpecificationResult {
    pub direction: CausalDirection,
    /// Every maximum state within the requested PyPhi-compatible tolerance.
    /// The first state is the stable enumeration-order winner.
    pub specified_states: Vec<Vec<bool>>,
    pub intrinsic_specification: f64,
    /// Highest-scoring state other than the first winner (may itself be tied).
    pub runner_up_state: Option<Vec<bool>>,
    pub runner_up_intrinsic_specification: Option<f64>,
    /// Maximum minus runner-up; zero if another state ties the winner.
    pub selection_margin: Option<f64>,
    /// Unconstrained forward reference repertoire used in the GID vector.
    pub unconstrained_forward_repertoire: Vec<f64>,
    /// Decimal precision used for relative/absolute tie equality.
    pub decimal_precision: u32,
}

/// IIT 4.0 system intrinsic states: cause and effect specifications are distinct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Iit4SystemState {
    /// Candidate past state selected by cause intrinsic specification.
    pub cause_specified_state: Vec<bool>,
    /// Candidate future state selected by effect intrinsic specification.
    pub effect_specified_state: Vec<bool>,
}

/// One direction's fixed-state system-level GID contribution for a cut.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemPhiDirectionScore {
    pub direction: CausalDirection,
    /// Unpartitioned cause/effect repertoire mass at the specified state.
    pub selectivity: f64,
    /// Unpartitioned forward probability (a likelihood on the cause side).
    pub forward_probability: f64,
    /// Forward probability after applying the partition.
    pub partitioned_forward_probability: f64,
    /// Signed GID contribution: selectivity * log2(forward / partitioned).
    pub signed_phi: f64,
}

/// IIT 4.0 system-partition score for an explicitly supplied state and cut.
///
/// This is the per-partition quantity used by the IIT 4.0 (2023) SIA search;
/// it is not an SIA or phi_s result until the system's intrinsic specified
/// states are selected and all eligible partitions are searched.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemPhiPartitionScore {
    pub cause: SystemPhiDirectionScore,
    pub effect: SystemPhiDirectionScore,
    /// Signed minimum of cause and effect contributions before |.|+ clamping.
    pub signed_phi: f64,
    /// Nonnegative phi value after the IIT |.|+ convention.
    pub phi: f64,
    /// 1 / number of severed connections (the EdgeCut normalization).
    pub normalization_factor: f64,
    pub signed_normalized_phi: f64,
    pub normalized_phi: f64,
    pub severed_connections: usize,
}

/// Bounded, explicit policy for an exact IIT 4.0 (2023) system search.
///
/// The search is exhaustive within these declared limits. Exceeding either
/// limit returns an error; no partial result is marked exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Iit4SystemPhiSearchConfig {
    pub decimal_precision: u32,
    pub max_partition_combinations: usize,
    pub max_state_pair_evaluations: usize,
    pub include_total_cut: bool,
}

impl Default for Iit4SystemPhiSearchConfig {
    fn default() -> Self {
        Self {
            decimal_precision: 6,
            max_partition_combinations: 100_000,
            max_state_pair_evaluations: 1_024,
            include_total_cut: false,
        }
    }
}

/// Outcome of the IIT 4.0 (2023) Integration-level system search.
///
/// Composition-level state-pair tie resolution requires the cause-effect
/// structure and its big-Phi value. This crate does not yet compute that
/// structure, so it reports such ties rather than choosing an arbitrary pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Iit4SystemPhiStatus {
    ExactUniqueStatePair,
    UnresolvedCompositionTie,
    UnresolvedCanonicalTie,
    NullDisconnected,
    NullNoCause,
    NullNoEffect,
    NullSingleUnit,
}

/// Selected minimum-information partition for one fixed system-state pair.
#[derive(Debug, Clone, PartialEq)]
pub struct Iit4SystemPhiStatePairResult {
    pub system_state: Iit4SystemState,
    pub winning_partition: DirectedSystemPartition,
    pub score: SystemPhiPartitionScore,
    /// Gap in clamped normalized Phi to the nearest competing partition.
    /// Zero records a tie under the default PyPhi SIA partition ranking.
    pub partition_margin: Option<f64>,
}

/// Summary of the exact directed partition enumeration used by a system search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Iit4SystemPhiPartitionSummary {
    pub raw_direction_labelings_equivalent: usize,
    pub canonical_direction_assignments_evaluated: usize,
    pub redundant_direction_assignments_pruned: usize,
    pub unique_non_total_cuts: usize,
    pub total_cut_included: bool,
}

/// Auditable IIT 4.0 (2023) system-level search result.
///
/// phi_s is populated only when the state-pair cascade has a unique winner,
/// or when every unresolved zero-Phi tie necessarily has the same exact value
/// zero. A nonzero Composition-level tie leaves phi_s unset because the
/// winning specified-state pair cannot be known without big Phi.
#[derive(Debug, Clone, PartialEq)]
pub struct Iit4SystemPhiAnalysis {
    pub formalism: FormalismProfile,
    pub candidate_system: Vec<usize>,
    pub request_sha256: String,
    pub status: Iit4SystemPhiStatus,
    pub phi_s: Option<f64>,
    pub selected_state_pair_index: Option<usize>,
    pub tied_state_pair_indices: Vec<usize>,
    /// One MIP result for each member of the Cartesian cause/effect tie family.
    pub state_pair_results: Vec<Iit4SystemPhiStatePairResult>,
    pub partition_summary: Iit4SystemPhiPartitionSummary,
    pub partition_score_evaluations: usize,
    /// True means the declared exact search completed; null/short-circuit
    /// results are complete under the documented PyPhi guard conditions.
    pub complete: bool,
}

/// Errors while evaluating a fixed-state IIT 4.x system partition.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SystemPhiError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Repertoire(#[from] RepertoireError),
    #[error(transparent)]
    Partition(#[from] PartitionApplyError),
    #[error("{direction:?} specified system state has {actual} bits, expected {expected}")]
    SpecifiedStateLength {
        direction: CausalDirection,
        expected: usize,
        actual: usize,
    },
    #[error("system partition cuts no connections and therefore has no EdgeCut normalization")]
    ZeroSeveredConnections,
    #[error("candidate system must be in strictly ascending unit-index order")]
    CandidateSystemNotCanonical,
    #[error("proper-subsystem IIT 4.0 scoring is not qualified: Eq. 3/Eq. 4 external-background marginals are not implemented")]
    UnsupportedSubsystemBackground,
    #[error("numeric decimal precision {precision} is outside the supported range 0..=15")]
    InvalidNumericalPrecision { precision: u32 },
    #[error("intrinsic-specification GID became non-finite at state index {state_index}")]
    NonFiniteIntrinsicSpecification { state_index: usize },
    #[error("system partition enumeration failed: {0}")]
    PartitionEnumeration(#[from] PartitionEnumerationError),
    #[error("system partition enumeration exceeded its budget of {max_combinations} canonical representatives")]
    PartitionBudgetExceeded { max_combinations: usize },
    #[error("partition GID score became non-finite")]
    NonFinitePartitionScore,
    #[error("specified-state pair count {candidate_pairs} exceeds the configured budget of {max_state_pairs}")]
    StatePairBudgetExceeded {
        candidate_pairs: usize,
        max_state_pairs: usize,
    },
    #[error("specified-state pair count overflow")]
    StatePairCounterOverflow,
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
    /// Number of canonical direction representatives actually materialized.
    pub canonical_direction_assignments_evaluated: usize,
    /// Number of raw 3^k direction labelings represented by the search.
    pub raw_direction_assignments_equivalent: usize,
    /// Redundant raw labelings avoided by quotienting equal cut matrices.
    pub redundant_direction_assignments_pruned: usize,
    /// Number of unique non-total cut matrices emitted; equals the canonical count.
    pub unique_non_total_cuts: usize,
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
    #[error("directed partition enumeration exceeded its budget of {max_combinations} canonical direction representatives")]
    BudgetExceeded { max_combinations: usize },
    #[error("directed partition work counter overflow")]
    CounterOverflow,
}

/// Enumerate the IIT 4.x DIRECTED_SET_PARTITION cut family.
///
/// For every non-trivial set partition, emit one canonical representative for
/// each unique cut matrix under the Cause/Effect/Bidirectional block-direction rule.
/// The exact quotient avoids generating direction labelings that produce an
/// already represented severance matrix, matching PyPhi's matrix-based EdgeCut
/// equality semantics.
///
/// TotalCut severs every connection, including self-loops. It is always included
/// for a one-unit system and optional for larger systems. Ordinary directional
/// partitions preserve self-loops.
///
/// max_combinations limits canonical direction representatives evaluated. The raw
/// equivalent labelings and the number pruned are reported separately. Exceeding
/// the budget returns an error and no partial list, so an incomplete list cannot
/// be mistaken for exact enumeration.
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
/// Raw-equivalent counts cover the full 3^k labeling space. Canonical counts cover
/// only representatives materialized by the exact matrix-equivalence quotient.
/// Redundant counts are raw minus canonical; all emitted non-total matrices are
/// unique by construction. TotalCut insertion is not counted in these counters.
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

    if n == 1 || include_total_cut {
        output.push(DirectedSystemPartition {
            candidate_system: canonical_system.clone(),
            parts: vec![canonical_system.clone()],
            directions: vec![PartitionDirection::Bidirectional],
            cut_matrix: vec![true; n * n],
            total_cut: true,
        });
    }

    if n == 1 {
        return Ok(PartitionEnumerationReport {
            partitions: output,
            canonical_direction_assignments_evaluated: 0,
            raw_direction_assignments_equivalent: 0,
            redundant_direction_assignments_pruned: 0,
            unique_non_total_cuts: 0,
            total_cut_included: true,
        });
    }

    let mut blocks = Vec::<Vec<usize>>::new();
    let mut canonical_assignments_evaluated = 0usize;
    let mut raw_assignments_equivalent = 0usize;
    let mut redundant_assignments_pruned = 0usize;
    {
        let mut context = PartitionEnumerationContext {
            candidate_system: &canonical_system,
            output: &mut output,
            canonical_assignments_evaluated: &mut canonical_assignments_evaluated,
            raw_assignments_equivalent: &mut raw_assignments_equivalent,
            redundant_assignments_pruned: &mut redundant_assignments_pruned,
            max_combinations,
        };
        enumerate_set_partitions_recursive(0, n, &mut blocks, &mut context)?;
    }
    let unique_non_total_cuts = output.iter().filter(|partition| !partition.is_total_cut()).count();
    let total_cut_included = output.iter().any(DirectedSystemPartition::is_total_cut);
    debug_assert_eq!(canonical_assignments_evaluated, unique_non_total_cuts);
    debug_assert_eq!(
        raw_assignments_equivalent,
        canonical_assignments_evaluated + redundant_assignments_pruned
    );
    Ok(PartitionEnumerationReport {
        partitions: output,
        canonical_direction_assignments_evaluated: canonical_assignments_evaluated,
        raw_direction_assignments_equivalent: raw_assignments_equivalent,
        redundant_direction_assignments_pruned: redundant_assignments_pruned,
        unique_non_total_cuts,
        total_cut_included,
    })
}

/// Mutable state shared by set-partition recursion and canonical cut emission.
struct PartitionEnumerationContext<'a> {
    candidate_system: &'a [usize],
    output: &'a mut Vec<DirectedSystemPartition>,
    canonical_assignments_evaluated: &'a mut usize,
    raw_assignments_equivalent: &'a mut usize,
    redundant_assignments_pruned: &'a mut usize,
    max_combinations: usize,
}

/// Restricted-growth recursion enumerates each unordered set partition once.
fn enumerate_set_partitions_recursive(
    position: usize,
    n: usize,
    blocks: &mut Vec<Vec<usize>>,
    context: &mut PartitionEnumerationContext<'_>,
) -> Result<(), PartitionEnumerationError> {
    if position == n {
        if blocks.len() <= 1 {
            return Ok(()); // PyPhi's set-partition path omits the trivial block.
        }
        let mut block_by_local_unit = vec![0usize; n];
        for (block_index, block) in blocks.iter().enumerate() {
            for local_unit in block {
                block_by_local_unit[*local_unit] = block_index;
            }
        }
        return enumerate_canonical_block_directions(blocks, &block_by_local_unit, context);
    }

    // Assign the next unit to each existing block, then to a new final block.
    for block_index in 0..=blocks.len() {
        if block_index == blocks.len() {
            blocks.push(vec![position]);
            enumerate_set_partitions_recursive(position + 1, n, blocks, context)?;
            blocks.pop();
        } else {
            blocks[block_index].push(position);
            enumerate_set_partitions_recursive(position + 1, n, blocks, context)?;
            blocks[block_index].pop();
        }
    }
    Ok(())
}

/// Enumerate one canonical direction labeling for every distinct cut matrix.
///
/// An uncut cross-block edge can only run from a Cause block to an Effect block.
/// Consequently a cut matrix is identified by two disjoint, nonempty block subsets
/// C and E (the missing C -> E biclique), or by the all-cut cross-block matrix
/// represented by all Bidirectional blocks. This quotient emits
/// 1 + (3^k - 2*2^k + 1) = 3^k - 2*2^k + 2 candidates for k blocks.
fn enumerate_canonical_block_directions(
    blocks: &[Vec<usize>],
    block_by_local_unit: &[usize],
    context: &mut PartitionEnumerationContext<'_>,
) -> Result<(), PartitionEnumerationError> {
    let block_count = blocks.len();
    let raw_for_partition = 3usize
        .checked_pow(block_count as u32)
        .ok_or(PartitionEnumerationError::CounterOverflow)?;
    let two_to_k = 1usize
        .checked_shl(block_count as u32)
        .ok_or(PartitionEnumerationError::CounterOverflow)?;
    let canonical_for_partition = raw_for_partition
        .checked_sub(
            two_to_k
                .checked_mul(2)
                .ok_or(PartitionEnumerationError::CounterOverflow)?,
        )
        .and_then(|count| count.checked_add(2))
        .ok_or(PartitionEnumerationError::CounterOverflow)?;
    let pruned_for_partition = raw_for_partition - canonical_for_partition;

    let next_canonical = context
        .canonical_assignments_evaluated
        .checked_add(canonical_for_partition)
        .ok_or(PartitionEnumerationError::CounterOverflow)?;
    if next_canonical > context.max_combinations {
        return Err(PartitionEnumerationError::BudgetExceeded {
            max_combinations: context.max_combinations,
        });
    }
    let next_raw = context
        .raw_assignments_equivalent
        .checked_add(raw_for_partition)
        .ok_or(PartitionEnumerationError::CounterOverflow)?;
    let next_pruned = context
        .redundant_assignments_pruned
        .checked_add(pruned_for_partition)
        .ok_or(PartitionEnumerationError::CounterOverflow)?;

    *context.canonical_assignments_evaluated = next_canonical;
    *context.raw_assignments_equivalent = next_raw;
    *context.redundant_assignments_pruned = next_pruned;

    let all_blocks = two_to_k - 1;
    let mut directions = vec![PartitionDirection::Bidirectional; block_count];

    // Canonical representative for the matrix that cuts every cross-block edge.
    emit_canonical_partition(blocks, block_by_local_unit, &directions, 0, 0, context);

    // Every other unique cut matrix is identified by nonempty disjoint subsets:
    // C: source blocks whose outgoing edge to E remains uncut;
    // E: target blocks whose incoming edge from C remains uncut.
    for cause_mask in 1..two_to_k {
        let eligible_effect_blocks = all_blocks ^ cause_mask;
        let mut effect_mask = eligible_effect_blocks;
        while effect_mask != 0 {
            for (block_index, direction) in directions.iter_mut().enumerate() {
                *direction = if (cause_mask & (1usize << block_index)) != 0 {
                    PartitionDirection::Cause
                } else if (effect_mask & (1usize << block_index)) != 0 {
                    PartitionDirection::Effect
                } else {
                    PartitionDirection::Bidirectional
                };
            }
            emit_canonical_partition(
                blocks,
                block_by_local_unit,
                &directions,
                cause_mask,
                effect_mask,
                context,
            );
            effect_mask = (effect_mask - 1) & eligible_effect_blocks;
        }
    }

    Ok(())
}

/// Build a cut matrix directly from its canonical Cause/Effect block masks.
fn emit_canonical_partition(
    blocks: &[Vec<usize>],
    block_by_local_unit: &[usize],
    directions: &[PartitionDirection],
    cause_mask: usize,
    effect_mask: usize,
    context: &mut PartitionEnumerationContext<'_>,
) {
    let n = context.candidate_system.len();
    let mut cut_matrix = vec![false; n * n];
    for source in 0..n {
        let source_block = block_by_local_unit[source];
        for target in 0..n {
            let target_block = block_by_local_unit[target];
            let is_unchanged_cause_to_effect_edge =
                (cause_mask & (1usize << source_block)) != 0
                    && (effect_mask & (1usize << target_block)) != 0;
            if source_block != target_block && !is_unchanged_cause_to_effect_edge {
                cut_matrix[source * n + target] = true;
            }
        }
    }

    let parts = blocks
        .iter()
        .map(|block| {
            block
                .iter()
                .map(|local| context.candidate_system[*local])
                .collect()
        })
        .collect();
    context.output.push(DirectedSystemPartition {
        candidate_system: context.candidate_system.to_vec(),
        parts,
        directions: directions.to_vec(),
        cut_matrix,
        total_cut: false,
    });
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
/// Compute the full pointwise generalized intrinsic difference (GID) vector.
///
/// For every state i, the value is selectivity[i] * log2(forward[i] /
/// partitioned[i]). If either ratio term is zero, its contribution is zero.
/// Forward and partitioned inputs may be likelihood vectors and need not be
/// normalized. Selectivity must be a normalized probability distribution.
///
/// This returns the vector of pointwise values, including negative entries; it
/// is not itself a system-level phi_s value or a minimum-information-partition
/// calculation.
pub fn generalized_intrinsic_difference(
    forward_repertoire: &[f64],
    partitioned_forward_repertoire: &[f64],
    selectivity_repertoire: &[f64],
) -> Result<Vec<f64>, RepertoireError> {
    if forward_repertoire.len() != partitioned_forward_repertoire.len()
        || forward_repertoire.len() != selectivity_repertoire.len()
    {
        return Err(RepertoireError::DistributionLengthMismatch {
            forward: forward_repertoire.len(),
            partitioned: partitioned_forward_repertoire.len(),
            selectivity: selectivity_repertoire.len(),
        });
    }

    validate_probability_values(forward_repertoire, "forward")?;
    validate_probability_values(partitioned_forward_repertoire, "partitioned")?;
    validate_probability_distribution(selectivity_repertoire, "selectivity")?;

    Ok(forward_repertoire
        .iter()
        .copied()
        .zip(partitioned_forward_repertoire.iter().copied())
        .zip(selectivity_repertoire.iter().copied())
        .map(|((forward, partitioned), selectivity)| {
            if forward == 0.0 || partitioned == 0.0 {
                0.0
            } else {
                selectivity * (forward / partitioned).log2()
            }
        })
        .collect())
}

/// Evaluate PyPhi's pointwise GID at one specified state.
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
    if state_index >= forward_repertoire.len() {
        return Err(RepertoireError::DistributionStateOutOfRange {
            index: state_index,
            length: forward_repertoire.len(),
        });
    }

    let gid = generalized_intrinsic_difference(
        forward_repertoire,
        partitioned_forward_repertoire,
        selectivity_repertoire,
    )?;
    Ok(gid[state_index])
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
        for (future_state, probability) in repertoire.iter_mut().enumerate() {
            for (slot, p_on) in per_unit_p_on.iter().copied().enumerate() {
                let target_on = ((future_state >> slot) & 1) != 0;
                *probability *= if target_on { p_on } else { 1.0 - p_on };
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

        for (local_source, source) in candidate.iter().copied().enumerate() {
            for (local_target, target) in candidate.iter().copied().enumerate() {
                if !partition.cut_matrix()[local_source * n_candidate + local_target] {
                    continue;
                }
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

    /// Compute both cause and effect intrinsic specified-state tie families.
    pub fn system_intrinsic_specification_iit4(
        &self,
        candidate_system: &[usize],
        current_state: &[bool],
        decimal_precision: u32,
    ) -> Result<Iit4SystemIntrinsicSpecification, SystemPhiError> {
        let cause = self.intrinsic_specification_iit4(
            candidate_system,
            current_state,
            CausalDirection::Cause,
            decimal_precision,
        )?;
        let effect = self.intrinsic_specification_iit4(
            candidate_system,
            current_state,
            CausalDirection::Effect,
            decimal_precision,
        )?;
        Ok(Iit4SystemIntrinsicSpecification { cause, effect })
    }

    /// Select the system's intrinsically specified past/future state in one direction.
    ///
    /// The current qualified path requires the full substrate as the candidate
    /// system. Proper-subsystem Eq. 3/Eq. 4 external-background marginalization
    /// is not yet implemented and is rejected explicitly.
    ///
    /// This implements PyPhi IIT 4.0's composite intrinsic-specification path:
    /// pointwise GID(forward, unconstrained-forward) weighted by the current
    /// cause/effect selectivity repertoire, maximized over all candidate states.
    /// Cause uses the unnormalized forward cause likelihood and a constant
    /// mean-likelihood baseline; effect uses the effect repertoire averaged over
    /// all candidate mechanism states as its unconstrained baseline.
    ///
    /// Ties use symmetric relative/absolute equality with epsilon 10^-precision,
    /// matching PyPhi's numerics.eq/eq_mask policy. Every tied state is returned
    /// in deterministic little-endian state order; callers must preserve that tie
    /// family for the later cause/effect-state-pair SIA cascade.
    pub fn intrinsic_specification_iit4(
        &self,
        candidate_system: &[usize],
        current_state: &[bool],
        direction: CausalDirection,
        decimal_precision: u32,
    ) -> Result<Iit4IntrinsicSpecificationResult, SystemPhiError> {
        self.check_state(current_state)?;
        if decimal_precision > 15 {
            return Err(SystemPhiError::InvalidNumericalPrecision {
                precision: decimal_precision,
            });
        }
        self.validate_subset(candidate_system, "candidate_system")?;
        if candidate_system.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(SystemPhiError::CandidateSystemNotCanonical);
        }
        if !candidate_system.iter().copied().eq(0..self.n_units) {
            return Err(SystemPhiError::UnsupportedSubsystemBackground);
        }

        let state_count = 1usize << candidate_system.len();
        let (selectivity, forward, unconstrained_forward) = match direction {
            CausalDirection::Cause => {
                let selectivity =
                    self.cause_repertoire(candidate_system, current_state, candidate_system)?;
                let forward =
                    self.forward_cause_likelihood(candidate_system, current_state, candidate_system)?;
                let mean = forward.iter().sum::<f64>() / state_count as f64;
                (selectivity, forward, vec![mean; state_count])
            }
            CausalDirection::Effect => {
                let selectivity =
                    self.effect_repertoire(candidate_system, current_state, candidate_system)?;
                let forward = selectivity.clone();
                let mut unconstrained = vec![0.0; state_count];
                for mechanism_state_index in 0..state_count {
                    let mut mechanism_state = current_state.to_vec();
                    for (slot, unit) in candidate_system.iter().copied().enumerate() {
                        mechanism_state[unit] = ((mechanism_state_index >> slot) & 1) != 0;
                    }
                    let repertoire = self.effect_repertoire(
                        candidate_system,
                        &mechanism_state,
                        candidate_system,
                    )?;
                    for (sum, probability) in unconstrained.iter_mut().zip(repertoire) {
                        *sum += probability / state_count as f64;
                    }
                }
                (selectivity, forward, unconstrained)
            }
        };

        let mut information = Vec::with_capacity(state_count);
        for state_index in 0..state_count {
            let f = forward[state_index];
            let u = unconstrained_forward[state_index];
            let pointwise = if f == 0.0 || u == 0.0 {
                0.0
            } else {
                let ratio = f / u;
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
        let tied_states = information
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| {
                (value - maximum).abs()
                    <= (epsilon * value.abs().max(maximum.abs())).max(epsilon)
            })
            .map(|(index, _)| {
                candidate_system
                    .iter()
                    .enumerate()
                    .map(|(slot, _)| ((index >> slot) & 1) != 0)
                    .collect::<Vec<bool>>()
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
            candidate_system
                .iter()
                .enumerate()
                .map(|(slot, _)| ((index >> slot) & 1) != 0)
                .collect::<Vec<_>>()
        });
        let runner_up_intrinsic_specification =
            runner_up_index.map(|index| information[index]);
        let selection_margin = runner_up_intrinsic_specification
            .map(|runner_up| (maximum - runner_up).max(0.0));

        Ok(Iit4IntrinsicSpecificationResult {
            direction,
            specified_states: tied_states,
            intrinsic_specification: maximum,
            runner_up_state,
            runner_up_intrinsic_specification,
            selection_margin,
            unconstrained_forward_repertoire: unconstrained_forward,
            decimal_precision,
        })
    }

    /// Evaluate both IIT 4.0 system-level directional GID contributions for a cut.
    ///
    /// This matches PyPhi's fixed-state partition-evaluation path: effect uses
    /// the forward effect probability, cause uses the unnormalized forward cause
    /// likelihood, both weighted by the corresponding unpartitioned repertoire
    /// probability at the specified state. The system score is min(cause, effect),
    /// followed by the positive-part convention and EdgeCut normalization.
    ///
    /// The candidate system is taken from the supplied DirectedSystemPartition;
    /// the current state is in model-global unit order. The qualified implementation
    /// currently requires the candidate to be the full substrate; proper-subsystem
    /// Eq. 3/Eq. 4 external-background marginals are not yet implemented. Cause and
    /// effect use separate specified states, as required by IIT 4.0. This function deliberately
    /// does not choose either state, resolve state ties, filter non-disconnecting
    /// cuts, resolve partition ties, or search for the MIP.
    pub fn score_iit4_system_partition(
        &self,
        partition: &DirectedSystemPartition,
        current_state: &[bool],
        system_state: &Iit4SystemState,
    ) -> Result<SystemPhiPartitionScore, SystemPhiError> {
        self.check_state(current_state)?;
        let candidate = partition.candidate_system();
        if system_state.cause_specified_state.len() != candidate.len() {
            return Err(SystemPhiError::SpecifiedStateLength {
                direction: CausalDirection::Cause,
                expected: candidate.len(),
                actual: system_state.cause_specified_state.len(),
            });
        }
        if system_state.effect_specified_state.len() != candidate.len() {
            return Err(SystemPhiError::SpecifiedStateLength {
                direction: CausalDirection::Effect,
                expected: candidate.len(),
                actual: system_state.effect_specified_state.len(),
            });
        }
        self.validate_subset(candidate, "candidate_system")?;
        if !candidate.iter().copied().eq(0..self.n_units) {
            return Err(SystemPhiError::UnsupportedSubsystemBackground);
        }
        let cut_edges = partition.cut_matrix().iter().filter(|cut| **cut).count();
        if cut_edges == 0 {
            return Err(SystemPhiError::ZeroSeveredConnections);
        }
        let cause_state_index = system_state
            .cause_specified_state
            .iter()
            .enumerate()
            .fold(0usize, |index, (unit, on)| {
                if *on { index | (1usize << unit) } else { index }
            });
        let effect_state_index = system_state
            .effect_specified_state
            .iter()
            .enumerate()
            .fold(0usize, |index, (unit, on)| {
                if *on { index | (1usize << unit) } else { index }
            });

        let cause_selectivity = self.cause_repertoire(candidate, current_state, candidate)?;
        let cause_forward = self.forward_cause_likelihood(candidate, current_state, candidate)?;
        let effect_selectivity = self.effect_repertoire(candidate, current_state, candidate)?;
        // In IIT 4.0 the forward effect repertoire is the effect repertoire.
        let effect_forward = effect_selectivity.clone();

        let cut_model = self.apply_partition(partition)?;
        let cause_partitioned =
            cut_model.forward_cause_likelihood(candidate, current_state, candidate)?;
        let effect_partitioned = cut_model.effect_repertoire(candidate, current_state, candidate)?;

        let cause = system_phi_direction_score(
            CausalDirection::Cause,
            cause_selectivity[cause_state_index],
            cause_forward[cause_state_index],
            cause_partitioned[cause_state_index],
        );
        let effect = system_phi_direction_score(
            CausalDirection::Effect,
            effect_selectivity[effect_state_index],
            effect_forward[effect_state_index],
            effect_partitioned[effect_state_index],
        );
        let signed_phi = cause.signed_phi.min(effect.signed_phi);
        let normalization_factor = 1.0 / cut_edges as f64;
        let signed_normalized_phi = signed_phi * normalization_factor;
        let phi = signed_phi.max(0.0);
        let normalized_phi = signed_normalized_phi.max(0.0);

        Ok(SystemPhiPartitionScore {
            cause,
            effect,
            signed_phi,
            phi,
            normalization_factor,
            signed_normalized_phi,
            normalized_phi,
            severed_connections: cut_edges,
        })
    }

    /// Run the exhaustive IIT 4.0 (2023) system-level phi_s search.
    ///
    /// For each tied cause/effect intrinsic-specification state pair, this:
    /// 1. evaluates every member of PyPhi's DIRECTED_SET_PARTITION family;
    /// 2. selects the MIP by minimum normalized phi, then maximum nonnegative
    ///    phi, then PyPhi's lexicographic induced-cut order;
    /// 3. resolves the system-state pair by maximum phi_s.
    ///
    /// If the Integration-level state-pair maximum ties at nonzero phi, PyPhi
    /// escalates to Composition and compares big Phi. Since this crate does
    /// not yet compute the cause-effect structure, the result is explicitly
    /// marked unresolved and phi_s remains None. No arbitrary representative
    /// is chosen.
    ///
    /// Only IIT 4.0 (2023) is implemented here. The 2026 intrinsic-information
    /// cap is intentionally not applied or implied by this API.
    pub fn analyze_iit4_2023_system_phi_s(
        &self,
        candidate_system: &[usize],
        current_state: &[bool],
        config: Iit4SystemPhiSearchConfig,
    ) -> Result<Iit4SystemPhiAnalysis, SystemPhiError> {
        self.check_state(current_state)?;
        self.validate_subset(candidate_system, "candidate_system")?;
        if candidate_system.is_empty() {
            return Err(SystemPhiError::Model(ModelError::EmptyCandidateSystem));
        }
        if candidate_system.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(SystemPhiError::CandidateSystemNotCanonical);
        }
        if config.decimal_precision > 15 {
            return Err(SystemPhiError::InvalidNumericalPrecision {
                precision: config.decimal_precision,
            });
        }

        let request = AnalysisRequest::new(
            self,
            current_state.to_vec(),
            candidate_system.to_vec(),
            FormalismProfile::Iit4_0_2023,
            AnalysisQuantity::Iit4SystemPhiS,
        )?;
        let request_sha256 = request.request_sha256(self);

        let empty_summary = Iit4SystemPhiPartitionSummary::default();
        let null_result = |status, phi_s| Iit4SystemPhiAnalysis {
            formalism: FormalismProfile::Iit4_0_2023,
            candidate_system: candidate_system.to_vec(),
            request_sha256: request_sha256.clone(),
            status,
            phi_s,
            selected_state_pair_index: None,
            tied_state_pair_indices: Vec::new(),
            state_pair_results: Vec::new(),
            partition_summary: empty_summary,
            partition_score_evaluations: 0,
            complete: true,
        };

        // PyPhi's default IIT 4.0 convention treats a single micro-unit system
        // as zero-Phi, regardless of whether that unit has a self-loop.
        if candidate_system.len() == 1 {
            return Ok(null_result(Iit4SystemPhiStatus::NullSingleUnit, Some(0.0)));
        }
        if !self.candidate_is_strongly_connected(candidate_system) {
            return Ok(null_result(Iit4SystemPhiStatus::NullDisconnected, Some(0.0)));
        }

        let intrinsic = self.system_intrinsic_specification_iit4(
            candidate_system,
            current_state,
            config.decimal_precision,
        )?;
        let epsilon = 10.0_f64.powi(-(config.decimal_precision as i32));
        if !pyphi_is_positive(intrinsic.cause.intrinsic_specification, epsilon) {
            return Ok(null_result(Iit4SystemPhiStatus::NullNoCause, Some(0.0)));
        }
        if !pyphi_is_positive(intrinsic.effect.intrinsic_specification, epsilon) {
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
        // Move the canonical partition list out of the stats report so only
        // one candidate vector remains live during the scoring pass.
        let partitions = std::mem::take(&mut enumeration.partitions);
        let partition_score_evaluations = partitions
            .len()
            .checked_mul(state_pair_count)
            .ok_or(SystemPhiError::StatePairCounterOverflow)?;

        let state_pairs = intrinsic.state_pairs();
        let mut state_pair_results = Vec::with_capacity(state_pair_count);
        for system_state in state_pairs {
            let mut scores = Vec::with_capacity(partitions.len());
            for partition in &partitions {
                let score = self.score_iit4_system_partition(
                    partition,
                    current_state,
                    &system_state,
                )?;
                if !score.signed_phi.is_finite()
                    || !score.phi.is_finite()
                    || !score.normalized_phi.is_finite()
                {
                    return Err(SystemPhiError::NonFinitePartitionScore);
                }
                scores.push(score);
            }

            if scores.is_empty() {
                return Err(SystemPhiError::ZeroSeveredConnections);
            }
            let winner_index = select_pyphi_sia_partition(&partitions, &scores, epsilon);
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
            .filter(|(_, result)| pyphi_eq(result.score.phi, maximum_phi, epsilon))
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
                // PyPhi resolves a reducible all-zero state-pair tie by a
                // canonical state representative. Retain the tie family;
                // only the scalar zero is invariant without that choice.
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

    /// Whether the declared candidate-subsystem connectivity is strongly connected.
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
                        self.connectivity[target * self.n_units + source]
                    } else {
                        self.connectivity[source * self.n_units + target]
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
        matches!(
            (self, formalism),
            (Self::Iit3SystemPhi, FormalismProfile::Iit3_0)
                | (
                    Self::Iit4SystemPhiS
                        | Self::Iit4CausePhiC
                        | Self::Iit4EffectPhiE
                        | Self::Iit4CauseEffectStructure
                        | Self::Iit4StructureBigPhi,
                    FormalismProfile::Iit4_0_2023 | FormalismProfile::Iit4_0_2026,
                )
        )
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

/// PyPhi numerics.eq: symmetric relative tolerance with an absolute floor.
fn pyphi_eq(left: f64, right: f64, epsilon: f64) -> bool {
    if !left.is_finite() || !right.is_finite() {
        return left == right;
    }
    let tolerance = (epsilon * left.abs().max(right.abs())).max(epsilon);
    (left - right).abs() <= tolerance
}

/// PyPhi numerics.is_positive: positive and not tolerance-equal to zero.
fn pyphi_is_positive(value: f64, epsilon: f64) -> bool {
    value > 0.0 && !pyphi_eq(value, 0.0, epsilon)
}

/// Default PyPhi IIT 4.0 sia_tie_resolution:
/// min NORMALIZED_PHI, then min NEGATIVE_PHI (max PHI), then PARTITION_LEX.
fn select_pyphi_sia_partition(
    partitions: &[DirectedSystemPartition],
    scores: &[SystemPhiPartitionScore],
    epsilon: f64,
) -> usize {
    debug_assert_eq!(partitions.len(), scores.len());
    let minimum_normalized = scores
        .iter()
        .map(|score| score.normalized_phi)
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let normalized_ties = scores
        .iter()
        .enumerate()
        .filter(|(_, score)| pyphi_eq(score.normalized_phi, minimum_normalized, epsilon))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();

    let maximum_phi = normalized_ties
        .iter()
        .map(|index| scores[*index].phi)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    normalized_ties
        .into_iter()
        .filter(|index| pyphi_eq(scores[*index].phi, maximum_phi, epsilon))
        // PyPhi PARTITION_LEX compares row-major induced cut-matrix bytes.
        // For one fixed ascending candidate subsystem, the local matrix has
        // the same ordering as its zero-embedded substrate matrix.
        .min_by(|left, right| {
            partitions[*left]
                .cut_matrix()
                .cmp(partitions[*right].cut_matrix())
        })
        .unwrap_or(0)
}

fn system_phi_direction_score(
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

    // Deliberately simple reference implementation used only by tests. It
    // materializes each block-direction assignment and applies the original
    // block-membership definition, independent of the optimized block-label path.
    fn reference_directed_cut_matrices(unit_count: usize) -> std::collections::HashSet<Vec<bool>> {
        fn enumerate_directions(
            direction_index: usize,
            unit_count: usize,
            blocks: &[Vec<usize>],
            directions: &mut Vec<PartitionDirection>,
            output: &mut std::collections::HashSet<Vec<bool>>,
        ) {
            if direction_index == blocks.len() {
                let mut matrix = vec![false; unit_count * unit_count];
                for (block, direction) in blocks.iter().zip(directions.iter().copied()) {
                    for source in 0..unit_count {
                        for target in 0..unit_count {
                            let source_inside = block.contains(&source);
                            let target_inside = block.contains(&target);
                            let cut = match direction {
                                PartitionDirection::Cause => !source_inside && target_inside,
                                PartitionDirection::Effect => source_inside && !target_inside,
                                PartitionDirection::Bidirectional => source_inside != target_inside,
                            };
                            if cut {
                                matrix[source * unit_count + target] = true;
                            }
                        }
                    }
                }
                output.insert(matrix);
                return;
            }

            for direction in [
                PartitionDirection::Cause,
                PartitionDirection::Effect,
                PartitionDirection::Bidirectional,
            ] {
                directions.push(direction);
                enumerate_directions(
                    direction_index + 1,
                    unit_count,
                    blocks,
                    directions,
                    output,
                );
                directions.pop();
            }
        }

        fn enumerate_set_partitions(
            position: usize,
            unit_count: usize,
            blocks: &mut Vec<Vec<usize>>,
            output: &mut std::collections::HashSet<Vec<bool>>,
        ) {
            if position == unit_count {
                if blocks.len() > 1 {
                    let mut directions = Vec::with_capacity(blocks.len());
                    enumerate_directions(
                        0,
                        unit_count,
                        blocks,
                        &mut directions,
                        output,
                    );
                }
                return;
            }

            for block_index in 0..=blocks.len() {
                if block_index == blocks.len() {
                    blocks.push(vec![position]);
                    enumerate_set_partitions(position + 1, unit_count, blocks, output);
                    blocks.pop();
                } else {
                    blocks[block_index].push(position);
                    enumerate_set_partitions(position + 1, unit_count, blocks, output);
                    blocks[block_index].pop();
                }
            }
        }

        let mut output = std::collections::HashSet::new();
        let mut blocks = Vec::new();
        enumerate_set_partitions(0, unit_count, &mut blocks, &mut output);
        output
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
            // Change A's self-transition across both values of B, preserving
            // the declared absence of B -> A while changing model identity.
            vec![0.11, 0.2, 0.9, 0.2, 0.11, 0.8, 0.9, 0.8],
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
    fn system_intrinsic_specification_selects_cause_and_effect_states() {
        let model = BinaryCausalModel::new(
            2,
            vec![
                0.1, 0.2,
                0.1, 0.8,
                0.9, 0.2,
                0.9, 0.8,
            ],
            vec![false, true, true, false],
        ).unwrap();
        let selected = model.system_intrinsic_specification_iit4(
            &[0, 1],
            &[true, false],
            6,
        ).unwrap();

        assert_eq!(selected.cause.specified_states, vec![vec![false, true]]);
        assert_eq!(selected.effect.specified_states, vec![vec![false, true]]);
        let expected = 0.72 * (0.72_f64 / 0.25).log2();
        assert!((selected.cause.intrinsic_specification - expected).abs() < 1e-12);
        assert!((selected.effect.intrinsic_specification - expected).abs() < 1e-12);
        // The runner-up is state 10 (likelihood 0.02), not state 11
        // (likelihood 0.18): both have negative GID, but 10 is less negative.
        let runner_up_information = 0.02 * (0.02_f64 / 0.25).log2();
        assert_eq!(selected.cause.runner_up_state, Some(vec![true, false]));
        assert!((selected.cause.runner_up_intrinsic_specification.unwrap()
            - runner_up_information).abs() < 1e-12);
        assert!((selected.cause.selection_margin.unwrap()
            - (expected - runner_up_information)).abs() < 1e-12);
        assert_eq!(selected.state_pairs(), vec![Iit4SystemState {
            cause_specified_state: vec![false, true],
            effect_specified_state: vec![false, true],
        }]);
    }

    #[test]
    fn intrinsic_specification_preserves_all_tied_states() {
        let model = BinaryCausalModel::new(1, vec![0.5, 0.5], vec![false]).unwrap();
        let selected = model.system_intrinsic_specification_iit4(&[0], &[false], 6).unwrap();
        assert_eq!(selected.cause.specified_states, vec![vec![false], vec![true]]);
        assert_eq!(selected.effect.specified_states, vec![vec![false], vec![true]]);
        assert_eq!(selected.cause.intrinsic_specification, 0.0);
        assert_eq!(selected.effect.intrinsic_specification, 0.0);
        assert_eq!(selected.cause.selection_margin, Some(0.0));
        assert_eq!(selected.effect.selection_margin, Some(0.0));
        assert_eq!(selected.state_pairs().len(), 4);
    }

    #[test]
    fn intrinsic_specification_rejects_noncanonical_candidates_and_precision() {
        let model = two_unit_model();
        assert!(matches!(
            model.intrinsic_specification_iit4(&[1, 0], &[false, false], CausalDirection::Cause, 6),
            Err(SystemPhiError::CandidateSystemNotCanonical)
        ));
        assert!(matches!(
            model.intrinsic_specification_iit4(&[0, 1], &[false, false], CausalDirection::Cause, 16),
            Err(SystemPhiError::InvalidNumericalPrecision { precision: 16 })
        ));
    }

    #[test]
    fn fixed_state_system_partition_matches_iit4_2023_directional_gid() {
        // Two mutually coupled binary units. At current state 00, the cause
        // and effect selectivity of specified state 00 is 0.72. Cutting both
        // cross-unit links makes both partitioned forward probabilities 0.25.
        let model = BinaryCausalModel::new(
            2,
            vec![
                0.1, 0.2, // past 00
                0.1, 0.8, // past 10
                0.9, 0.2, // past 01
                0.9, 0.8, // past 11
            ],
            vec![
                false, true,
                true, false,
            ],
        ).unwrap();
        let partitions = enumerate_iit4_system_partitions(&[0, 1], 2, false, 3).unwrap();
        let all_cross_edges_cut = partitions
            .iter()
            .find(|partition| partition.cut_matrix() == [false, true, true, false])
            .expect("all-cross-edge cut is part of the n=2 family");

        let score = model.score_iit4_system_partition(
            all_cross_edges_cut,
            &[false, false],
            &Iit4SystemState {
                cause_specified_state: vec![false, false],
                effect_specified_state: vec![false, false],
            },
        ).unwrap();

        let expected = 0.72 * (0.72_f64 / 0.25).log2();
        assert!((score.cause.selectivity - 0.72).abs() < 1e-12);
        assert!((score.cause.forward_probability - 0.72).abs() < 1e-12);
        assert!((score.cause.partitioned_forward_probability - 0.25).abs() < 1e-12);
        assert!((score.cause.signed_phi - expected).abs() < 1e-12);
        assert!((score.effect.selectivity - 0.72).abs() < 1e-12);
        assert!((score.effect.forward_probability - 0.72).abs() < 1e-12);
        assert!((score.effect.partitioned_forward_probability - 0.25).abs() < 1e-12);
        assert!((score.effect.signed_phi - expected).abs() < 1e-12);
        assert!((score.signed_phi - expected).abs() < 1e-12);
        assert!((score.phi - expected).abs() < 1e-12);
        assert_eq!(score.severed_connections, 2);
        assert_eq!(score.normalization_factor, 0.5);
        assert!((score.normalized_phi - expected * 0.5).abs() < 1e-12);
    }

    #[test]
    fn fixed_state_system_partition_uses_distinct_cause_and_effect_states() {
        let model = BinaryCausalModel::new(
            2,
            vec![
                0.1, 0.2,
                0.1, 0.8,
                0.9, 0.2,
                0.9, 0.8,
            ],
            vec![false, true, true, false],
        ).unwrap();
        let partition = enumerate_iit4_system_partitions(&[0, 1], 2, false, 3)
            .unwrap()
            .into_iter()
            .find(|partition| partition.cut_matrix() == [false, true, true, false])
            .unwrap();

        let score = model.score_iit4_system_partition(
            &partition,
            &[false, false],
            &Iit4SystemState {
                cause_specified_state: vec![false, false],
                effect_specified_state: vec![true, true],
            },
        ).unwrap();

        let expected_cause = 0.72 * (0.72_f64 / 0.25).log2();
        let expected_effect = 0.02 * (0.02_f64 / 0.25).log2();
        assert!((score.cause.signed_phi - expected_cause).abs() < 1e-12);
        assert!((score.effect.signed_phi - expected_effect).abs() < 1e-12);
        assert!((score.signed_phi - expected_effect).abs() < 1e-12);
        assert_eq!(score.phi, 0.0);
    }

    #[test]
    fn fixed_state_system_partition_preserves_preventative_signed_phi() {
        let model = BinaryCausalModel::new(
            2,
            vec![
                0.1, 0.2,
                0.1, 0.8,
                0.9, 0.2,
                0.9, 0.8,
            ],
            vec![false, true, true, false],
        ).unwrap();
        let partition = enumerate_iit4_system_partitions(&[0, 1], 2, false, 3)
            .unwrap()
            .into_iter()
            .find(|partition| partition.cut_matrix() == [false, true, true, false])
            .unwrap();

        let score = model.score_iit4_system_partition(
            &partition,
            &[false, false],
            &Iit4SystemState {
                cause_specified_state: vec![true, true],
                effect_specified_state: vec![true, true],
            },
        ).unwrap();

        assert!(score.signed_phi < 0.0);
        assert_eq!(score.phi, 0.0);
        assert!(score.signed_normalized_phi < 0.0);
        assert_eq!(score.normalized_phi, 0.0);
    }

    #[test]
    fn fixed_state_system_partition_rejects_wrong_specified_state_length() {
        let model = two_unit_model();
        let partition = enumerate_iit4_system_partitions(&[0, 1], 2, false, 3)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert!(matches!(
            model.score_iit4_system_partition(
                &partition,
                &[false, false],
                &Iit4SystemState {
                    cause_specified_state: vec![false],
                    effect_specified_state: vec![false, false],
                },
            ),
            Err(SystemPhiError::SpecifiedStateLength {
                direction: CausalDirection::Cause,
                expected: 2,
                actual: 1,
            })
        ));
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
        // A and B's next states both depend on the unconstrained hidden unit C.
        // Their individual marginals are 0.5. The factored repertoire therefore
        // has mass 0.25 on each pair, unlike a joint mixture with mass only on 00/11.
        let model = BinaryCausalModel::new(
            3,
            vec![
                0.0, 0.0, 0.5,  0.0, 0.0, 0.5,
                0.0, 0.0, 0.5,  0.0, 0.0, 0.5,
                1.0, 1.0, 0.5,  1.0, 1.0, 0.5,
                1.0, 1.0, 0.5,  1.0, 1.0, 0.5,
            ],
            vec![
                false, false, false,
                false, false, false,
                true, true, true,
            ],
        ).unwrap();
        let repertoire = model.effect_repertoire(&[], &[false, false, false], &[0, 1]).unwrap();
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
        // The raw direction space has 3 * 3^2 + 1 * 3^3 = 54 labelings.
        // Quotienting equal cut matrices produces exactly 22 representatives.
        let without_total = enumerate_iit4_system_partitions(&[0, 1, 2], 3, false, 22).unwrap();
        assert_eq!(without_total.len(), 22);
        assert!(without_total.iter().all(|p| !p.is_total_cut()));
        let with_total = enumerate_iit4_system_partitions(&[0, 1, 2], 3, true, 22).unwrap();
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
    fn four_through_six_unit_partition_counts_pin_combinatorial_growth() {
        // Raw 3^k labelings are quotiented to one representative per cut matrix.
        let four = enumerate_iit4_system_partitions(&[0, 1, 2, 3], 4, false, 150).unwrap();
        assert_eq!(four.len(), 150);

        let five = enumerate_iit4_system_partitions(&[0, 1, 2, 3, 4], 5, false, 1_061).unwrap();
        assert_eq!(five.len(), 1_061);

        let six = enumerate_iit4_system_partitions(
            &[0, 1, 2, 3, 4, 5],
            6,
            false,
            7_896,
        ).unwrap();
        assert_eq!(six.len(), 7_896);

        assert!(matches!(
            enumerate_iit4_system_partitions(&[0, 1, 2, 3, 4], 5, false, 1_060),
            Err(PartitionEnumerationError::BudgetExceeded { max_combinations: 1_060 })
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
        // For this one-unit TPM the likelihoods for observing ON are [0.2, 0.4],
        // which are not a normalized distribution over previous states.
        let model = BinaryCausalModel::new(1, vec![0.2, 0.4], vec![true]).unwrap();
        let forward = model.forward_cause_likelihood(&[0], &[true], &[0]).unwrap();
        assert!((forward[0] - 0.2).abs() < 1e-12);
        assert!((forward[1] - 0.4).abs() < 1e-12);
        assert!((forward.iter().sum::<f64>() - 0.6).abs() < 1e-12);
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
    fn partition_enumeration_report_separates_raw_and_canonical_work() {
        let report = enumerate_iit4_system_partitions_with_stats(&[0, 1, 2], 3, false, 22).unwrap();
        assert_eq!(report.canonical_direction_assignments_evaluated, 22);
        assert_eq!(report.raw_direction_assignments_equivalent, 54);
        assert_eq!(report.redundant_direction_assignments_pruned, 32);
        assert_eq!(report.unique_non_total_cuts, 22);
        assert!(!report.total_cut_included);
        assert_eq!(report.partitions.len(), 22);

        let with_total = enumerate_iit4_system_partitions_with_stats(&[0, 1, 2], 3, true, 22).unwrap();
        assert_eq!(with_total.canonical_direction_assignments_evaluated, 22);
        assert_eq!(with_total.raw_direction_assignments_equivalent, 54);
        assert_eq!(with_total.redundant_direction_assignments_pruned, 32);
        assert_eq!(with_total.unique_non_total_cuts, 22);
        assert!(with_total.total_cut_included);
        assert_eq!(with_total.partitions.len(), 23);
    }

    #[test]
    fn partition_enumeration_report_counts_pruned_direction_equivalents() {
        let report = enumerate_iit4_system_partitions_with_stats(&[0, 1, 2, 3, 4], 5, false, 1_061).unwrap();
        assert_eq!(report.canonical_direction_assignments_evaluated, 1_061);
        assert_eq!(report.raw_direction_assignments_equivalent, 1_863);
        assert_eq!(report.redundant_direction_assignments_pruned, 802);
        assert_eq!(report.unique_non_total_cuts, 1_061);
    }

    #[test]
    fn generalized_intrinsic_difference_vector_matches_pointwise_values() {
        let gid = generalized_intrinsic_difference(
            &[0.1, 0.4, 0.0, 0.8],
            &[0.2, 0.2, 0.5, 0.0],
            &[0.25, 0.5, 0.1, 0.15],
        ).unwrap();
        let expected = [
            0.25 * (0.1_f64 / 0.2).log2(),
            0.5 * (0.4_f64 / 0.2).log2(),
            0.0,
            0.0,
        ];
        for (actual, expected) in gid.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        let point = generalized_intrinsic_difference_at_state(
            &[0.1, 0.4, 0.0, 0.8],
            &[0.2, 0.2, 0.5, 0.0],
            &[0.25, 0.5, 0.1, 0.15],
            1,
        ).unwrap();
        assert!((point - gid[1]).abs() < 1e-12);
    }

    #[test]
    fn generalized_intrinsic_difference_vector_preserves_positive_and_negative_entries() {
        let gid = generalized_intrinsic_difference(
            &[0.1, 0.8],
            &[0.2, 0.4],
            &[0.5, 0.5],
        ).unwrap();
        assert!(gid[0] < 0.0);
        assert!(gid[1] > 0.0);
    }

    #[test]
    fn optimized_partition_cut_matrices_match_the_block_definition() {
        for units in 2..=6 {
            let candidate: Vec<usize> = (0..units).collect();
            let report = enumerate_iit4_system_partitions_with_stats(
                &candidate,
                units,
                false,
                match units {
                    2 => 3,
                    3 => 22,
                    4 => 150,
                    5 => 1_061,
                    6 => 7_896,
                    _ => unreachable!(),
                },
            ).unwrap();

            for partition in report.partitions {
                let mut expected = vec![false; units * units];
                for (global_block, direction) in partition.parts().iter()
                    .zip(partition.directions().iter().copied())
                {
                    let local_block: Vec<usize> = global_block.iter().map(|unit| {
                        partition.candidate_system().iter().position(|candidate_unit| candidate_unit == unit)
                            .expect("partition unit belongs to candidate system")
                    }).collect();
                    for source in 0..units {
                        for target in 0..units {
                            let source_inside = local_block.contains(&source);
                            let target_inside = local_block.contains(&target);
                            let cut = match direction {
                                PartitionDirection::Cause => !source_inside && target_inside,
                                PartitionDirection::Effect => source_inside && !target_inside,
                                PartitionDirection::Bidirectional => source_inside != target_inside,
                            };
                            if cut {
                                expected[source * units + target] = true;
                            }
                        }
                    }
                }
                assert_eq!(
                    partition.cut_matrix(),
                    expected.as_slice(),
                    "optimized matrix must equal block semantics for n={units}, parts={:?}, directions={:?}",
                    partition.parts(),
                    partition.directions(),
                );
            }
        }
    }

    #[test]
    fn optimized_enumerator_matches_complete_reference_matrix_set_for_two_through_six_units() {
        for units in 2..=6 {
            let candidate: Vec<usize> = (0..units).collect();
            let budget = match units {
                2 => 3,
                3 => 22,
                4 => 150,
                5 => 1_061,
                6 => 7_896,
                _ => unreachable!(),
            };
            let expected = reference_directed_cut_matrices(units);
            let actual: std::collections::HashSet<Vec<bool>> =
                enumerate_iit4_system_partitions(&candidate, units, false, budget)
                    .unwrap()
                    .into_iter()
                    .map(|partition| partition.cut_matrix().to_vec())
                    .collect();
            assert_eq!(
                actual,
                expected,
                "optimized and reference cut-matrix sets diverged for {units} units"
            );
        }
    }
}
