//! DNA sequence channel metrics and soft-erasure scheduling reference.
//!
//! This crate does not synthesize DNA, sequence molecules, train a probability
//! model, or implement any published codec. It evaluates supplied sequence
//! observations and accepts already-calibrated per-base error probabilities.
//! Synthetic fixtures are deterministic sensitivity tests, never physical rates.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DnaError {
    InvalidBase(u8),
    InvalidGcRange,
    InvalidThreshold,
    NonFiniteProbability { position: usize },
    ProbabilityOutOfRange { position: usize },
    DimensionOverflow,
}

impl fmt::Display for DnaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBase(base) => write!(f, "invalid nucleotide byte: {base}"),
            Self::InvalidGcRange => write!(f, "GC range must satisfy 0 <= min <= max <= 1"),
            Self::InvalidThreshold => write!(f, "minimum error probability must be in [0, 1]"),
            Self::NonFiniteProbability { position } => {
                write!(f, "non-finite error probability at position {position}")
            }
            Self::ProbabilityOutOfRange { position } => {
                write!(f, "error probability outside [0, 1] at position {position}")
            }
            Self::DimensionOverflow => write!(f, "sequence alignment matrix dimensions overflow"),
        }
    }
}

impl std::error::Error for DnaError {}

fn validate_sequence(sequence: &[u8]) -> Result<(), DnaError> {
    for &base in sequence {
        if !matches!(base, b'A' | b'C' | b'G' | b'T' | b'N') {
            return Err(DnaError::InvalidBase(base));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Start,
    Match,
    Substitution,
    Insertion,
    Deletion,
}

impl Op {
    fn priority(self) -> u8 {
        match self {
            Self::Match => 0,
            Self::Substitution => 1,
            Self::Deletion => 2,
            Self::Insertion => 3,
            Self::Start => 4,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Cell {
    cost: usize,
    op: Op,
}

impl Default for Cell {
    fn default() -> Self {
        Self { cost: 0, op: Op::Start }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EditCounts {
    pub distance: usize,
    pub matches: usize,
    pub substitutions: usize,
    pub insertions: usize,
    pub deletions: usize,
}

/// Compute one deterministic minimum-edit global alignment.
///
/// Ties are resolved in this order: match, substitution, deletion, insertion.
/// Repetitive sequence context can admit several equally short alignments; the
/// deterministic tie rule makes the reported decomposition reproducible, but
/// does not claim biological causality for an individual edit.
pub fn edit_counts(reference: &[u8], observed: &[u8]) -> Result<EditCounts, DnaError> {
    validate_sequence(reference)?;
    validate_sequence(observed)?;

    let rows = reference.len().checked_add(1).ok_or(DnaError::DimensionOverflow)?;
    let cols = observed.len().checked_add(1).ok_or(DnaError::DimensionOverflow)?;
    let cells = rows.checked_mul(cols).ok_or(DnaError::DimensionOverflow)?;
    let mut dp = vec![Cell::default(); cells];

    for i in 1..rows {
        dp[i * cols] = Cell { cost: i, op: Op::Deletion };
    }
    for j in 1..cols {
        dp[j] = Cell { cost: j, op: Op::Insertion };
    }

    for i in 1..rows {
        for j in 1..cols {
            let same = reference[i - 1] == observed[j - 1];
            let diagonal = dp[(i - 1) * cols + (j - 1)];
            let diagonal_op = if same { Op::Match } else { Op::Substitution };
            let diagonal_cost = diagonal.cost + if same { 0 } else { 1 };
            let deletion_cost = dp[(i - 1) * cols + j].cost + 1;
            let insertion_cost = dp[i * cols + (j - 1)].cost + 1;

            let mut best = Cell { cost: diagonal_cost, op: diagonal_op };
            let deletion = Cell { cost: deletion_cost, op: Op::Deletion };
            if (deletion.cost, deletion.op.priority()) < (best.cost, best.op.priority()) {
                best = deletion;
            }
            let insertion = Cell { cost: insertion_cost, op: Op::Insertion };
            if (insertion.cost, insertion.op.priority()) < (best.cost, best.op.priority()) {
                best = insertion;
            }
            dp[i * cols + j] = best;
        }
    }

    let mut counts = EditCounts::default();
    let mut i = reference.len();
    let mut j = observed.len();

    while i > 0 || j > 0 {
        let op = dp[i * cols + j].op;
        match op {
            Op::Match => {
                counts.matches += 1;
                i -= 1;
                j -= 1;
            }
            Op::Substitution => {
                counts.substitutions += 1;
                i -= 1;
                j -= 1;
            }
            Op::Deletion => {
                counts.deletions += 1;
                i -= 1;
            }
            Op::Insertion => {
                counts.insertions += 1;
                j -= 1;
            }
            Op::Start => return Err(DnaError::DimensionOverflow),
        }
    }

    counts.distance = counts.substitutions + counts.insertions + counts.deletions;
    Ok(counts)
}

pub fn gc_fraction(sequence: &[u8]) -> Result<f64, DnaError> {
    validate_sequence(sequence)?;
    if sequence.is_empty() {
        return Ok(0.0);
    }
    let gc = sequence.iter().filter(|&&base| base == b'G' || base == b'C').count();
    Ok(gc as f64 / sequence.len() as f64)
}

pub fn maximum_homopolymer_run(sequence: &[u8]) -> Result<usize, DnaError> {
    validate_sequence(sequence)?;
    let mut maximum = 0usize;
    let mut current = 0usize;
    let mut previous = None;
    for &base in sequence {
        if Some(base) == previous {
            current += 1;
        } else {
            current = 1;
            previous = Some(base);
        }
        maximum = maximum.max(current);
    }
    Ok(maximum)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftErasureCandidate {
    pub position: usize,
    /// Calibrated posterior probability that this base is wrong.
    pub error_probability: f64,
}

/// Rank calibrated error probabilities from most to least uncertain.
///
/// This routine does not estimate or calibrate probabilities. An adapter may call
/// it only when the source model's calibration cohort, sequencing platform and
/// data split are recorded. Equal probabilities are ordered by base position.
pub fn rank_soft_erasures(
    error_probabilities: &[f64],
    minimum_error_probability: f64,
) -> Result<Vec<SoftErasureCandidate>, DnaError> {
    if !minimum_error_probability.is_finite()
        || !(0.0..=1.0).contains(&minimum_error_probability)
    {
        return Err(DnaError::InvalidThreshold);
    }

    let mut candidates = Vec::new();
    for (position, &probability) in error_probabilities.iter().enumerate() {
        if !probability.is_finite() {
            return Err(DnaError::NonFiniteProbability { position });
        }
        if !(0.0..=1.0).contains(&probability) {
            return Err(DnaError::ProbabilityOutOfRange { position });
        }
        if probability >= minimum_error_probability {
            candidates.push(SoftErasureCandidate {
                position,
                error_probability: probability,
            });
        }
    }

    candidates.sort_by(|left, right| {
        right.error_probability
            .total_cmp(&left.error_probability)
            .then_with(|| left.position.cmp(&right.position))
    });
    Ok(candidates)
}

/// Generate a deterministic prefix of progressively larger erasure sets.
///
/// The caller supplies a calibrated candidate ranking and an explicit attempt
/// limit. The decoder must independently verify each recovered payload. These
/// are candidate sets, not a guarantee that any given codec can decode them.
pub fn progressive_erasure_sets(
    ranked: &[SoftErasureCandidate],
    maximum_attempts: usize,
) -> Vec<Vec<usize>> {
    let attempt_count = ranked.len().min(maximum_attempts);
    (1..=attempt_count)
        .map(|count| ranked[..count].iter().map(|candidate| candidate.position).collect())
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadBatchSummary {
    pub distinct_read_records: usize,
    pub read_observations: usize,
    pub additional_read_copies: usize,
    pub dropped_oligo: bool,
}

/// Count distinct read records and observed read multiplicity without treating
/// repeated reads as independent evidence. Counts do not identify distinct
/// molecules unless the upstream library provides validated UMI/molecule lineage.
pub fn summarize_read_batch(read_counts: &[usize], dropped_oligo: bool) -> ReadBatchSummary {
    let read_observations = read_counts.iter().copied().sum::<usize>();
    ReadBatchSummary {
        distinct_read_records: read_counts.len(),
        read_observations,
        additional_read_copies: read_observations.saturating_sub(read_counts.len()),
        dropped_oligo,
    }
}

/// Byte-exact success criterion. Archive acceptance must additionally verify
/// an independently preserved and authenticated manifest/digest.
pub fn exact_payload_match(expected: &[u8], decoded: &[u8]) -> bool {
    expected == decoded
}

pub fn sequence_constraints(
    sequence: &[u8],
    minimum_gc: f64,
    maximum_gc: f64,
    maximum_homopolymer: usize,
) -> Result<bool, DnaError> {
    if !minimum_gc.is_finite()
        || !maximum_gc.is_finite()
        || minimum_gc < 0.0
        || maximum_gc > 1.0
        || minimum_gc > maximum_gc
    {
        return Err(DnaError::InvalidGcRange);
    }
    if sequence.is_empty() || maximum_homopolymer == 0 {
        return Ok(false);
    }
    let gc = gc_fraction(sequence)?;
    let homopolymer = maximum_homopolymer_run(sequence)?;
    Ok(gc >= minimum_gc && gc <= maximum_gc && homopolymer <= maximum_homopolymer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_and_single_edit_channels_are_counted() {
        let reference = b"ACGTACGT";
        assert_eq!(edit_counts(reference, reference).unwrap().distance, 0);

        let substitution = edit_counts(reference, b"ACGTTCGT").unwrap();
        assert_eq!(substitution.substitutions, 1);
        assert_eq!(substitution.insertions + substitution.deletions, 0);

        let insertion = edit_counts(reference, b"ACGTTACGT").unwrap();
        assert_eq!(insertion.insertions, 1);
        assert_eq!(insertion.substitutions + insertion.deletions, 0);

        let deletion = edit_counts(reference, b"ACGACGT").unwrap();
        assert_eq!(deletion.deletions, 1);
        assert_eq!(deletion.substitutions + deletion.insertions, 0);
    }

    #[test]
    fn mixed_indels_have_reproducible_minimum_distance() {
        let counts = edit_counts(b"ACGTACGT", b"ATGTCGT").unwrap();
        assert_eq!(counts.distance, counts.substitutions + counts.insertions + counts.deletions);
        assert_eq!(counts.distance, 2);
    }

    #[test]
    fn sequence_constraints_are_separate_from_error_correction() {
        assert!(sequence_constraints(b"ACGTACGT", 0.5, 0.5, 2).unwrap());
        assert!(!sequence_constraints(b"AAAAAA", 0.0, 1.0, 3).unwrap());
        assert!((gc_fraction(b"ACGT").unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(maximum_homopolymer_run(b"AAACGT").unwrap(), 3);
    }

    #[test]
    fn soft_erasures_are_ranked_and_expanded_deterministically() {
        let ranked = rank_soft_erasures(&[0.01, 0.6, 0.6, 0.2], 0.2).unwrap();
        assert_eq!(
            ranked.iter().map(|candidate| candidate.position).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            progressive_erasure_sets(&ranked, 2),
            vec![vec![1], vec![1, 2]]
        );
    }

    #[test]
    fn uncalibrated_probability_inputs_fail_on_invalid_values() {
        assert!(matches!(
            rank_soft_erasures(&[f64::NAN], 0.1),
            Err(DnaError::NonFiniteProbability { position: 0 })
        ));
        assert!(matches!(
            rank_soft_erasures(&[1.1], 0.1),
            Err(DnaError::ProbabilityOutOfRange { position: 0 })
        ));
    }

    #[test]
    fn read_multiplicity_is_not_molecule_count() {
        let summary = summarize_read_batch(&[1, 4, 2], false);
        assert_eq!(summary.distinct_read_records, 3);
        assert_eq!(summary.read_observations, 7);
        assert_eq!(summary.additional_read_copies, 4);
    }

    #[test]
    fn exact_payload_comparison_rejects_wrong_reconstruction() {
        assert!(exact_payload_match(b"archive", b"archive"));
        assert!(!exact_payload_match(b"archive", b"arch1ve"));
    }

    #[test]
    fn homopolymer_ties_are_deterministic() {
        let counts = edit_counts(b"AAAA", b"AAA").unwrap();
        assert_eq!(counts.distance, 1);
        assert_eq!(counts.deletions + counts.insertions, 1);
    }
}
