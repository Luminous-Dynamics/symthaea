// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Point-of-Interest Error Rate (PIER) for explicitly tagged speech tokens.
//!
//! PIER applies a WER-style edit alignment, then scores only operations at
//! reference positions marked as points of interest (POIs). For code-switching,
//! callers typically tag embedded-language words. This module deliberately
//! does not infer language identity, normalize orthography, tokenize text, or
//! decide which spans are important: those are separately versioned inputs.
//!
//! The metric follows Ugan et al. (ICASSP 2025):
//! https://arxiv.org/abs/2501.09512
//!
//! Insertions are attributed to their source/reference insertion position.
//! When the final POI is the final reference token, trailing insertions are
//! included, matching the published boundary rule.
//!
//! Robustness correction relative to the current published Python reference
//! implementation: all-POI utterances are scored; they are not silently skipped
//! merely because they contain no non-POI tokens. A reference with zero POIs
//! returns a typed error (non-estimable) rather than a misleading zero score.
//!
//! The alignment reuses Symthaea's deterministic Levenshtein implementation.
//! Exact tie-by-tie parity with RapidFuzz's editops is not claimed until a
//! differential fixture set has been executed.

use crate::eval::{EditOp, levenshtein_align};
use std::fmt;

/// A PIER result for one reference/hypothesis pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PierResult {
    /// Number of tokens in the reference.
    pub reference_tokens: usize,
    /// Number of tokens in the hypothesis.
    pub hypothesis_tokens: usize,
    /// Number of reference tokens tagged as points of interest.
    pub poi_reference_tokens: usize,
    /// POI reference tokens aligned correctly.
    pub hits: usize,
    /// POI reference tokens aligned as substitutions.
    pub substitutions: usize,
    /// POI reference tokens aligned as deletions.
    pub deletions: usize,
    /// Insertions attributed to POI source positions, including the
    /// published trailing-insertion rule where applicable.
    pub insertions: usize,
}

impl PierResult {
    /// Number of reference POI tokens contributing to the denominator.
    ///
    /// Insertions do not increase the denominator, matching the published
    /// PIER definition: hits + substitutions + deletions.
    pub fn denominator(&self) -> usize {
        self.hits + self.substitutions + self.deletions
    }

    /// Total scored edit operations.
    pub fn errors(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }

    /// PIER as a percentage in [0, +infinity).
    ///
    /// Values above 100% are possible because insertions are included in the
    /// numerator but are not reference POI tokens in the denominator.
    pub fn pier_percent(&self) -> f64 {
        let denominator = self.denominator();
        if denominator == 0 {
            // A valid result cannot have an empty POI denominator. Keep this
            // defensive branch so manually constructed/default values cannot
            // produce NaN or infinity.
            return 0.0;
        }
        (self.errors() as f64 / denominator as f64) * 100.0
    }
}

/// Corpus-level PIER aggregation.
///
/// Both micro PIER (pooled edit counts) and macro PIER (arithmetic mean of
/// estimable utterance-level PIER values) are reported. No-POI utterances are
/// counted as non-estimable and excluded from both means; malformed inputs
/// still return a typed error. This prevents silent denominator changes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PierCorpusResult {
    /// Number of utterance triples supplied.
    pub utterances: usize,
    /// Utterances with at least one POI reference token.
    pub estimable_utterances: usize,
    /// Utterances whose reference contains no POI tokens.
    pub non_estimable_utterances: usize,
    /// Pooled reference/hypothesis and edit counts.
    pub totals: PierResult,
    /// Pooled PIER percentage; absent when no utterance is estimable.
    pub micro_pier_percent: Option<f64>,
    /// Unweighted mean of per-utterance PIER percentages.
    pub macro_pier_percent: Option<f64>,
}

/// Invalid corpus-level PIER input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PierCorpusError {
    /// The three corpus arrays must contain the same number of utterances.
    UtteranceCountMismatch {
        /// Number of references.
        references: usize,
        /// Number of hypotheses.
        hypotheses: usize,
        /// Number of POI-tag vectors.
        tag_vectors: usize,
    },
    /// A malformed utterance was found at the indicated index.
    InvalidUtterance {
        /// Zero-based utterance index.
        utterance_index: usize,
        /// Underlying invalid utterance reason.
        source: PierError,
    },
}

impl fmt::Display for PierCorpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UtteranceCountMismatch {
                references,
                hypotheses,
                tag_vectors,
            } => write!(
                f,
                "corpus arrays differ in length: {references} references, {hypotheses} hypotheses, {tag_vectors} tag vectors"
            ),
            Self::InvalidUtterance {
                utterance_index,
                source,
            } => write!(f, "invalid PIER input at utterance {utterance_index}: {source}"),
        }
    }
}

impl std::error::Error for PierCorpusError {}

/// Aggregate PIER across a corpus without hiding the averaging convention.
///
/// No-POI utterances are counted and excluded from PIER denominators. Empty
/// references and tag cardinality errors are not silently skipped; they return
/// a typed corpus error. An empty corpus returns a valid result with both
/// aggregate rates absent and zero counts.
pub fn aggregate_point_of_interest_error_rates<T: PartialEq + Clone>(
    references: &[Vec<T>],
    hypotheses: &[Vec<T>],
    poi_tags: &[Vec<bool>],
) -> Result<PierCorpusResult, PierCorpusError> {
    if references.len() != hypotheses.len() || references.len() != poi_tags.len() {
        return Err(PierCorpusError::UtteranceCountMismatch {
            references: references.len(),
            hypotheses: hypotheses.len(),
            tag_vectors: poi_tags.len(),
        });
    }

    let mut result = PierCorpusResult {
        utterances: references.len(),
        ..PierCorpusResult::default()
    };
    let mut macro_total = 0.0f64;

    for (index, ((reference, hypothesis), tags)) in references
        .iter()
        .zip(hypotheses.iter())
        .zip(poi_tags.iter())
        .enumerate()
    {
        match point_of_interest_error_rate(reference, hypothesis, tags) {
            Ok(utterance) => {
                result.estimable_utterances += 1;
                macro_total += utterance.pier_percent();
                result.totals.reference_tokens += utterance.reference_tokens;
                result.totals.hypothesis_tokens += utterance.hypothesis_tokens;
                result.totals.poi_reference_tokens += utterance.poi_reference_tokens;
                result.totals.hits += utterance.hits;
                result.totals.substitutions += utterance.substitutions;
                result.totals.deletions += utterance.deletions;
                result.totals.insertions += utterance.insertions;
            }
            Err(PierError::NoPointsOfInterest) => {
                result.non_estimable_utterances += 1;
            }
            Err(source) => {
                return Err(PierCorpusError::InvalidUtterance {
                    utterance_index: index,
                    source,
                });
            }
        }
    }

    if result.estimable_utterances > 0 {
        result.micro_pier_percent = Some(result.totals.pier_percent());
        result.macro_pier_percent =
            Some(macro_total / result.estimable_utterances as f64);
    }

    Ok(result)
}

/// Invalid PIER input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PierError {
    /// PIER requires at least one reference token.
    EmptyReference,
    /// There must be exactly one POI tag per reference token.
    TagLengthMismatch {
        /// Number of reference tokens supplied.
        reference_tokens: usize,
        /// Number of tags supplied.
        tag_count: usize,
    },
    /// No reference token was marked as a POI; PIER is not estimable.
    NoPointsOfInterest,
}

impl fmt::Display for PierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyReference => write!(f, "PIER requires a non-empty reference"),
            Self::TagLengthMismatch {
                reference_tokens,
                tag_count,
            } => write!(
                f,
                "PIER tag count ({tag_count}) must equal reference token count ({reference_tokens})"
            ),
            Self::NoPointsOfInterest => {
                write!(f, "PIER is not estimable: reference contains no points of interest")
            }
        }
    }
}

impl std::error::Error for PierError {}

/// Compute Point-of-Interest Error Rate from token sequences and reference tags.
///
/// poi_tags[i] == true marks reference token i as a point of interest.
/// Callers should apply any language-specific tokenization and normalization
/// before calling this function, and must preserve the same token coordinate
/// system for the POI tags. This function compares tokens exactly; it never
/// strips diacritics, folds case, transliterates or normalizes Unicode.
///
/// # Errors
///
/// Returns EmptyReference for an empty reference,
/// TagLengthMismatch when tag and reference lengths differ, and
/// NoPointsOfInterest when no reference token is tagged.
///
/// An empty hypothesis is valid and is scored as deletions. All-POI references
/// are also valid.
pub fn point_of_interest_error_rate<T: PartialEq + Clone>(
    reference: &[T],
    hypothesis: &[T],
    poi_tags: &[bool],
) -> Result<PierResult, PierError> {
    if reference.is_empty() {
        return Err(PierError::EmptyReference);
    }

    if poi_tags.len() != reference.len() {
        return Err(PierError::TagLengthMismatch {
            reference_tokens: reference.len(),
            tag_count: poi_tags.len(),
        });
    }

    let poi_reference_tokens = poi_tags.iter().filter(|&&is_poi| is_poi).count();
    if poi_reference_tokens == 0 {
        return Err(PierError::NoPointsOfInterest);
    }

    let alignment = levenshtein_align(reference, hypothesis);
    let final_poi_is_final_reference_token = poi_tags.last().copied() == Some(true);

    let mut result = PierResult {
        reference_tokens: reference.len(),
        hypothesis_tokens: hypothesis.len(),
        poi_reference_tokens,
        ..PierResult::default()
    };

    // The alignment is ordered left-to-right. source_position is the number
    // of reference tokens consumed so far; insertions use this boundary index.
    let mut source_position = 0usize;

    for operation in alignment.operations {
        match operation {
            EditOp::Correct => {
                if poi_tags[source_position] {
                    result.hits += 1;
                }
                source_position += 1;
            }
            EditOp::Substitution => {
                if poi_tags[source_position] {
                    result.substitutions += 1;
                }
                source_position += 1;
            }
            EditOp::Deletion => {
                if poi_tags[source_position] {
                    result.deletions += 1;
                }
                source_position += 1;
            }
            EditOp::Insertion => {
                let insertion_is_at_poi = poi_tags
                    .get(source_position)
                    .copied()
                    .unwrap_or(false);
                let trailing_insertion_after_final_poi =
                    final_poi_is_final_reference_token && source_position >= reference.len();

                if insertion_is_at_poi || trailing_insertion_after_final_poi {
                    result.insertions += 1;
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{PierError, point_of_interest_error_rate};

    fn score(reference: &[&str], hypothesis: &[&str], tags: &[bool]) -> super::PierResult {
        point_of_interest_error_rate(reference, hypothesis, tags)
            .expect("fixture should have a valid POI reference")
    }

    #[test]
    fn perfect_recognition_has_zero_pier() {
        let result = score(
            &["ngiyabona", "hello", "mngani"],
            &["ngiyabona", "hello", "mngani"],
            &[false, true, false],
        );
        assert_eq!(result.hits, 1);
        assert_eq!(result.errors(), 0);
        assert_eq!(result.pier_percent(), 0.0);
    }

    #[test]
    fn non_poi_errors_do_not_change_poi_score() {
        let baseline = score(
            &["matrix_a", "hello", "matrix_b"],
            &["matrix_a", "hello", "matrix_b"],
            &[false, true, false],
        );
        let changed_matrix = score(
            &["matrix_a", "hello", "matrix_b"],
            &["matrix_x", "hello", "matrix_y"],
            &[false, true, false],
        );
        assert_eq!(baseline.pier_percent(), 0.0);
        assert_eq!(changed_matrix.pier_percent(), 0.0);
    }

    #[test]
    fn embedded_word_substitution_counts_as_poi_error() {
        let result = score(
            &["matrix_a", "hello", "matrix_b"],
            &["matrix_a", "hallo", "matrix_b"],
            &[false, true, false],
        );
        assert_eq!(result.substitutions, 1);
        assert_eq!(result.denominator(), 1);
        assert_eq!(result.pier_percent(), 100.0);
    }

    #[test]
    fn embedded_word_deletion_counts_even_when_matrix_words_survive() {
        let result = score(
            &["matrix_a", "hello", "matrix_b"],
            &["matrix_a", "matrix_b"],
            &[false, true, false],
        );
        assert_eq!(result.deletions, 1);
        assert_eq!(result.hits, 0);
        assert_eq!(result.pier_percent(), 100.0);
    }

    #[test]
    fn insertion_at_poi_source_position_counts() {
        let result = score(
            &["matrix_a", "hello", "matrix_b"],
            &["matrix_a", "extra", "hello", "matrix_b"],
            &[false, true, false],
        );
        assert_eq!(result.insertions, 1);
        assert_eq!(result.pier_percent(), 100.0);
    }

    #[test]
    fn insertion_outside_poi_source_positions_is_not_counted() {
        let result = score(
            &["matrix_a", "hello", "matrix_b"],
            &["extra", "matrix_a", "hello", "matrix_b"],
            &[false, true, false],
        );
        assert_eq!(result.insertions, 0);
        assert_eq!(result.pier_percent(), 0.0);
    }

    #[test]
    fn trailing_insertion_counts_when_final_reference_token_is_poi() {
        let result = score(
            &["matrix_a", "hello"],
            &["matrix_a", "hello", "trailing"],
            &[false, true],
        );
        assert_eq!(result.insertions, 1);
        assert_eq!(result.pier_percent(), 100.0);
    }

    #[test]
    fn trailing_insertion_does_not_count_when_final_reference_token_is_not_poi() {
        let result = score(
            &["hello", "matrix_b"],
            &["hello", "matrix_b", "trailing"],
            &[true, false],
        );
        assert_eq!(result.insertions, 0);
        assert_eq!(result.pier_percent(), 0.0);
    }

    #[test]
    fn all_poi_reference_is_scored_instead_of_skipped() {
        let result = score(
            &["one", "two"],
            &["one", "too"],
            &[true, true],
        );
        assert_eq!(result.hits, 1);
        assert_eq!(result.substitutions, 1);
        assert_eq!(result.pier_percent(), 50.0);
    }

    #[test]
    fn empty_hypothesis_is_valid_and_charges_poi_deletions() {
        let result = score(&["matrix", "poi"], &[], &[false, true]);
        assert_eq!(result.deletions, 1);
        assert_eq!(result.pier_percent(), 100.0);
    }

    #[test]
    fn diacritics_are_not_silently_normalized() {
        let result = score(&["á"], &["a"], &[true]);
        assert_eq!(result.substitutions, 1);
        assert_eq!(result.pier_percent(), 100.0);
    }

    #[test]
    fn empty_reference_is_rejected() {
        let error = point_of_interest_error_rate::<&str>(&[], &[], &[]);
        assert_eq!(error, Err(PierError::EmptyReference));
    }

    #[test]
    fn pier_can_exceed_one_hundred_percent_due_to_insertions() {
        let result = score(&["hello"], &["extra_a", "extra_b", "hello"], &[true]);
        assert_eq!(result.hits, 1);
        assert_eq!(result.insertions, 2);
        assert_eq!(result.denominator(), 1);
        assert_eq!(result.pier_percent(), 200.0);
    }

    #[test]
    fn zero_poi_is_explicitly_non_estimable() {
        let error = point_of_interest_error_rate(
            &["matrix_a", "matrix_b"],
            &["matrix_a", "matrix_b"],
            &[false, false],
        );
        assert_eq!(error, Err(PierError::NoPointsOfInterest));
    }

    #[test]
    fn tag_length_mismatch_is_rejected() {
        let error = point_of_interest_error_rate(
            &["a", "b"],
            &["a", "b"],
            &[true],
        );
        assert_eq!(
            error,
            Err(PierError::TagLengthMismatch {
                reference_tokens: 2,
                tag_count: 1,
            })
        );
    }

    #[test]
    fn corpus_pier_reports_micro_and_macro_separately() {
        let result = super::aggregate_point_of_interest_error_rates(
            &[
                vec!["a", "poi1"],
                vec!["b", "poi2", "poi3"],
            ],
            &[
                vec!["a", "wrong"],
                vec!["b", "poi2", "poi3"],
            ],
            &[
                vec![false, true],
                vec![false, true, true],
            ],
        ).expect("valid corpus");
        assert_eq!(result.utterances, 2);
        assert_eq!(result.estimable_utterances, 2);
        assert_eq!(result.non_estimable_utterances, 0);
        assert_eq!(result.totals.hits, 2);
        assert_eq!(result.totals.substitutions, 1);
        assert_eq!(result.micro_pier_percent, Some(100.0 / 3.0));
        assert_eq!(result.macro_pier_percent, Some(50.0));
    }

    #[test]
    fn corpus_pier_counts_non_estimable_utterances_without_scoring_them_as_zero() {
        let result = super::aggregate_point_of_interest_error_rates(
            &[
                vec!["a", "poi"],
                vec!["b", "c"],
            ],
            &[
                vec!["a", "poi"],
                vec!["b", "c"],
            ],
            &[
                vec![false, true],
                vec![false, false],
            ],
        ).expect("valid corpus");
        assert_eq!(result.estimable_utterances, 1);
        assert_eq!(result.non_estimable_utterances, 1);
        assert_eq!(result.micro_pier_percent, Some(0.0));
        assert_eq!(result.macro_pier_percent, Some(0.0));
    }

    #[test]
    fn empty_corpus_has_absent_aggregate_rates() {
        let result = super::aggregate_point_of_interest_error_rates::<&str>(
            &[],
            &[],
            &[],
        ).expect("empty corpus is structurally valid");
        assert_eq!(result.utterances, 0);
        assert_eq!(result.micro_pier_percent, None);
        assert_eq!(result.macro_pier_percent, None);
    }

    #[test]
    fn corpus_rejects_mismatched_utterance_counts() {
        let error = super::aggregate_point_of_interest_error_rates(
            &[vec!["a"]],
            &[],
            &[vec![true]],
        );
        assert_eq!(
            error,
            Err(super::PierCorpusError::UtteranceCountMismatch {
                references: 1,
                hypotheses: 0,
                tag_vectors: 1,
            })
        );
    }

    #[test]
    fn corpus_rejects_malformed_utterance_with_index() {
        let error = super::aggregate_point_of_interest_error_rates(
            &[vec!["a"], vec!["b", "c"]],
            &[vec!["a"], vec!["b", "c"]],
            &[vec![true], vec![true]],
        );
        assert_eq!(
            error,
            Err(super::PierCorpusError::InvalidUtterance {
                utterance_index: 1,
                source: PierError::TagLengthMismatch {
                    reference_tokens: 2,
                    tag_count: 1,
                },
            })
        );
    }

    #[test]
    fn repeated_token_alignment_is_deterministic() {
        let reference = ["a", "x", "a", "x"];
        let hypothesis = ["a", "a", "x"];
        let tags = [false, true, false, true];
        let first = score(&reference, &hypothesis, &tags);
        let second = score(&reference, &hypothesis, &tags);
        assert_eq!(first, second);
    }
}
