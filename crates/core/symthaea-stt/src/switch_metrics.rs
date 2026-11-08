// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Switch-localized speech recognition metrics.
//!
//! Implements language-specific masked error rate, SETER (Switch-Entry Token
//! Error Rate), and SPER@k (Switch-Point Error Rate within a token window).
//! Definitions follow Okocha & Grant (2026):
//! https://arxiv.org/abs/2609.11786
//!
//! Language/variety tags are caller-supplied and reference-token aligned.
//! This module does not infer language identity from scripts, words, or audio.
//! Tokenization and normalization are also caller-owned.
//!
//! For masked token metrics, a reference token is erroneous when the edit
//! alignment marks it as a substitution or deletion. Insertions contribute to
//! conventional WER but are not assigned to reference positions for SETER or
//! SPER@k. This intentionally differs from PIER insertion attribution.

use crate::eval::{EditOp, levenshtein_align};
use std::fmt;

/// A masked error-rate result over a selected set of reference tokens.
///
/// error_rate is a fraction in [0, 1]. It is None when the selected set is
/// empty, such as an utterance with no language switches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskedErrorRate {
    /// Number of selected reference tokens.
    pub selected_tokens: usize,
    /// Selected reference tokens aligned as substitutions or deletions.
    pub errors: usize,
    /// Errors divided by selected tokens; absent when not estimable.
    pub error_rate: Option<f64>,
}

impl MaskedErrorRate {
    fn from_counts(selected_tokens: usize, errors: usize) -> Self {
        Self {
            selected_tokens,
            errors,
            error_rate: if selected_tokens == 0 {
                None
            } else {
                Some(errors as f64 / selected_tokens as f64)
            },
        }
    }
}

/// Error rate for one explicitly supplied language/variety tag.
#[derive(Debug, Clone, PartialEq)]
pub struct LanguageErrorRate<L> {
    /// Caller-supplied language or variety tag.
    pub language: L,
    /// Number of reference tokens carrying this tag.
    pub selected_tokens: usize,
    /// Substitutions or deletions among those tokens.
    pub errors: usize,
    /// Errors divided by selected tokens.
    pub error_rate: f64,
}

/// Switch-window metric for one requested radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwitchWindowErrorRate {
    /// Number of tokens on either side of each switch-entry token.
    /// Radius 0 is exactly the SETER position set.
    pub radius: usize,
    /// Masked error-rate result for the union of all switch windows.
    pub score: MaskedErrorRate,
}

/// Combined switch-aware evaluation result.
#[derive(Debug, Clone, PartialEq)]
pub struct SwitchAwareResult<L> {
    /// Number of reference tokens.
    pub reference_tokens: usize,
    /// Number of hypothesis tokens.
    pub hypothesis_tokens: usize,
    /// Reference-token substitutions.
    pub substitutions: usize,
    /// Reference-token deletions.
    pub deletions: usize,
    /// Hypothesis insertions.
    pub insertions: usize,
    /// Conventional WER as a fraction, including insertions.
    pub wer: f64,
    /// Error rate over all reference tokens, by supplied language/variety.
    /// Groups appear in order of first occurrence in the reference.
    pub per_language: Vec<LanguageErrorRate<L>>,
    /// Switch-entry token error rate. None when there are no language changes.
    pub seter: MaskedErrorRate,
    /// Switch-point error rates for the requested radii.
    pub sper: Vec<SwitchWindowErrorRate>,
}

/// Invalid switch-aware metric input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwitchMetricError {
    /// A non-empty reference is required.
    EmptyReference,
    /// Exactly one language tag is required per reference token.
    LanguageTagLengthMismatch {
        /// Number of reference tokens.
        reference_tokens: usize,
        /// Number of language tags.
        tag_count: usize,
    },
}

impl fmt::Display for SwitchMetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyReference => write!(f, "switch-aware metrics require a non-empty reference"),
            Self::LanguageTagLengthMismatch {
                reference_tokens,
                tag_count,
            } => write!(
                f,
                "language tag count ({tag_count}) must equal reference token count ({reference_tokens})"
            ),
        }
    }
}

impl std::error::Error for SwitchMetricError {}

/// Compute per-language error, SETER, and SPER@k from explicit reference tags.
///
/// A switch entry is the first reference token whose tag differs from the
/// preceding reference token. SPER windows are unions, so overlapping windows
/// never double-count a reference position.
///
/// window_radii is returned in caller-supplied order. Include 0, 1, and 3 to
/// obtain SETER, SPER@1, and SPER@3 respectively.
///
/// Tokens are compared exactly. This function does not case-fold, strip
/// diacritics, normalize Unicode, transliterate, or tokenize text. Apply any
/// desired transformation upstream, version it, and construct language tags
/// against the exact resulting reference-token sequence.
///
/// # Errors
///
/// Returns EmptyReference for an empty reference and
/// LanguageTagLengthMismatch for tag cardinality errors.
pub fn switch_aware_error_rates<T, L>(
    reference: &[T],
    hypothesis: &[T],
    language_tags: &[L],
    window_radii: &[usize],
) -> Result<SwitchAwareResult<L>, SwitchMetricError>
where
    T: PartialEq + Clone,
    L: PartialEq + Clone,
{
    if reference.is_empty() {
        return Err(SwitchMetricError::EmptyReference);
    }

    if language_tags.len() != reference.len() {
        return Err(SwitchMetricError::LanguageTagLengthMismatch {
            reference_tokens: reference.len(),
            tag_count: language_tags.len(),
        });
    }

    let alignment = levenshtein_align(reference, hypothesis);
    let mut token_errors = vec![false; reference.len()];
    let mut source_position = 0usize;
    let mut substitutions = 0usize;
    let mut deletions = 0usize;
    let mut insertions = 0usize;

    for operation in alignment.operations {
        match operation {
            EditOp::Correct => {
                source_position += 1;
            }
            EditOp::Substitution => {
                token_errors[source_position] = true;
                substitutions += 1;
                source_position += 1;
            }
            EditOp::Deletion => {
                token_errors[source_position] = true;
                deletions += 1;
                source_position += 1;
            }
            EditOp::Insertion => {
                insertions += 1;
            }
        }
    }

    let mut per_language: Vec<LanguageErrorRate<L>> = Vec::new();
    for (position, language) in language_tags.iter().enumerate() {
        if let Some(group) = per_language
            .iter_mut()
            .find(|group| &group.language == language)
        {
            group.selected_tokens += 1;
            group.errors += usize::from(token_errors[position]);
        } else {
            per_language.push(LanguageErrorRate {
                language: language.clone(),
                selected_tokens: 1,
                errors: usize::from(token_errors[position]),
                error_rate: 0.0,
            });
        }
    }
    for group in &mut per_language {
        group.error_rate = group.errors as f64 / group.selected_tokens as f64;
    }

    let mut switch_entries = Vec::new();
    for position in 1..language_tags.len() {
        if language_tags[position] != language_tags[position - 1] {
            switch_entries.push(position);
        }
    }

    let seter_errors = switch_entries
        .iter()
        .filter(|&&position| token_errors[position])
        .count();
    let seter = MaskedErrorRate::from_counts(switch_entries.len(), seter_errors);

    // Compute distance to the nearest switch-entry token in two linear passes.
    // usize::MAX marks the no-switch case and must never match a window radius.
    let mut distance_to_switch = vec![usize::MAX; reference.len()];
    for &position in &switch_entries {
        distance_to_switch[position] = 0;
    }

    let mut previous_switch: Option<usize> = None;
    for position in 0..reference.len() {
        if distance_to_switch[position] == 0 {
            previous_switch = Some(position);
        }
        if let Some(switch_position) = previous_switch {
            let distance = position - switch_position;
            distance_to_switch[position] = distance_to_switch[position].min(distance);
        }
    }

    previous_switch = None;
    for position in (0..reference.len()).rev() {
        if distance_to_switch[position] == 0 {
            previous_switch = Some(position);
        }
        if let Some(switch_position) = previous_switch {
            let distance = switch_position - position;
            distance_to_switch[position] = distance_to_switch[position].min(distance);
        }
    }

    let mut sper = Vec::with_capacity(window_radii.len());
    for &radius in window_radii {
        let mut selected_tokens = 0usize;
        let mut errors = 0usize;
        for position in 0..reference.len() {
            if distance_to_switch[position] != usize::MAX
                && distance_to_switch[position] <= radius
            {
                selected_tokens += 1;
                errors += usize::from(token_errors[position]);
            }
        }
        sper.push(SwitchWindowErrorRate {
            radius,
            score: MaskedErrorRate::from_counts(selected_tokens, errors),
        });
    }

    let reference_tokens = reference.len();
    let wer = (substitutions + deletions + insertions) as f64 / reference_tokens as f64;

    Ok(SwitchAwareResult {
        reference_tokens,
        hypothesis_tokens: hypothesis.len(),
        substitutions,
        deletions,
        insertions,
        wer,
        per_language,
        seter,
        sper,
    })
}

#[cfg(test)]
mod tests {
    use super::{SwitchMetricError, switch_aware_error_rates};

    #[test]
    fn perfect_transcript_has_zero_rates() {
        let result = switch_aware_error_rates(
            &["hello", "sawubona", "friend"],
            &["hello", "sawubona", "friend"],
            &["en", "zu", "en"],
            &[0, 1, 3],
        ).expect("valid fixture");
        assert_eq!(result.wer, 0.0);
        assert_eq!(result.seter.error_rate, Some(0.0));
        assert_eq!(result.sper[0].score.error_rate, Some(0.0));
    }

    #[test]
    fn seter_scores_only_first_token_of_new_language_run() {
        let result = switch_aware_error_rates(
            &["a", "b", "c", "d"],
            &["a", "b", "x", "d"],
            &["en", "en", "zu", "zu"],
            &[0, 1],
        ).expect("valid fixture");
        assert_eq!(result.seter.selected_tokens, 1);
        assert_eq!(result.seter.errors, 1);
        assert_eq!(result.seter.error_rate, Some(1.0));
    }

    #[test]
    fn switch_entry_deletion_is_an_error() {
        let result = switch_aware_error_rates(
            &["a", "b", "c"],
            &["a", "c"],
            &["en", "zu", "zu"],
            &[0],
        ).expect("valid fixture");
        assert_eq!(result.seter.selected_tokens, 1);
        assert_eq!(result.seter.errors, 1);
    }

    #[test]
    fn sper_zero_equals_seter() {
        let result = switch_aware_error_rates(
            &["a", "b", "c", "d"],
            &["a", "x", "c", "d"],
            &["en", "zu", "zu", "en"],
            &[0, 1, 3],
        ).expect("valid fixture");
        assert_eq!(result.sper[0].radius, 0);
        assert_eq!(result.sper[0].score, result.seter);
    }

    #[test]
    fn sper_one_includes_one_token_before_and_after_switch_entry() {
        let result = switch_aware_error_rates(
            &["a", "b", "c", "d", "e"],
            &["a", "x", "c", "d", "e"],
            &["en", "en", "zu", "zu", "zu"],
            &[1],
        ).expect("valid fixture");
        assert_eq!(result.sper[0].score.selected_tokens, 3);
        assert_eq!(result.sper[0].score.errors, 1);
        assert!((result.sper[0].score.error_rate.unwrap() - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn overlapping_switch_windows_use_a_union() {
        let result = switch_aware_error_rates(
            &["a", "b", "c", "d", "e"],
            &["a", "b", "x", "d", "e"],
            &["en", "zu", "en", "zu", "en"],
            &[1],
        ).expect("valid fixture");
        assert_eq!(result.seter.selected_tokens, 3);
        assert_eq!(result.sper[0].score.selected_tokens, 5);
        assert_eq!(result.sper[0].score.errors, 1);
    }

    #[test]
    fn multiple_transitions_are_counted_separately() {
        let result = switch_aware_error_rates(
            &["a", "b", "c", "d", "e"],
            &["a", "b", "c", "d", "e"],
            &["en", "en", "zu", "zu", "af"],
            &[0],
        ).expect("valid fixture");
        assert_eq!(result.seter.selected_tokens, 2);
        assert_eq!(result.seter.errors, 0);
        assert_eq!(result.seter.error_rate, Some(0.0));
    }

    #[test]
    fn no_switch_is_non_estimable_not_zero() {
        let result = switch_aware_error_rates(
            &["a", "b", "c"],
            &["a", "b", "c"],
            &["zu", "zu", "zu"],
            &[0, 1, 3],
        ).expect("valid fixture");
        assert_eq!(result.seter.error_rate, None);
        assert_eq!(result.sper[0].score.error_rate, None);
        assert_eq!(result.sper[1].score.selected_tokens, 0);
    }

    #[test]
    fn language_specific_rates_preserve_each_tag() {
        let result = switch_aware_error_rates(
            &["a", "b", "c", "d"],
            &["x", "b", "y", "d"],
            &["en", "en", "zu", "zu"],
            &[],
        ).expect("valid fixture");
        assert_eq!(result.per_language.len(), 2);
        assert_eq!(result.per_language[0].language, "en");
        assert_eq!(result.per_language[0].errors, 1);
        assert_eq!(result.per_language[0].error_rate, 0.5);
        assert_eq!(result.per_language[1].language, "zu");
        assert_eq!(result.per_language[1].errors, 1);
        assert_eq!(result.per_language[1].error_rate, 0.5);
    }

    #[test]
    fn insertions_affect_wer_but_not_reference_masked_rates() {
        let result = switch_aware_error_rates(
            &["a", "b"],
            &["a", "extra", "b"],
            &["en", "zu"],
            &[0],
        ).expect("valid fixture");
        assert_eq!(result.insertions, 1);
        assert_eq!(result.wer, 0.5);
        assert_eq!(result.seter.errors, 0);
        assert_eq!(result.seter.error_rate, Some(0.0));
    }

    #[test]
    fn radius_three_can_cover_the_entire_short_utterance() {
        let result = switch_aware_error_rates(
            &["a", "b", "c", "d", "e"],
            &["a", "b", "x", "d", "e"],
            &["en", "en", "zu", "zu", "zu"],
            &[3],
        ).expect("valid fixture");
        assert_eq!(result.sper[0].score.selected_tokens, 5);
        assert_eq!(result.sper[0].score.errors, 1);
    }

    #[test]
    fn one_token_utterance_has_no_switch_entries() {
        let result = switch_aware_error_rates(
            &["word"],
            &["wrong"],
            &["en"],
            &[0, 1],
        ).expect("valid fixture");
        assert_eq!(result.seter.error_rate, None);
        assert_eq!(result.per_language[0].error_rate, 1.0);
    }

    #[test]
    fn diacritics_are_not_silently_normalized() {
        let result = switch_aware_error_rates(
            &["ẹ"],
            &["e"],
            &["yo"],
            &[],
        ).expect("valid fixture");
        assert_eq!(result.substitutions, 1);
        assert_eq!(result.per_language[0].error_rate, 1.0);
    }

    #[test]
    fn tag_cardinality_mismatch_is_rejected() {
        let error = switch_aware_error_rates(
            &["a", "b"],
            &["a", "b"],
            &["en"],
            &[0],
        );
        assert_eq!(
            error,
            Err(SwitchMetricError::LanguageTagLengthMismatch {
                reference_tokens: 2,
                tag_count: 1,
            })
        );
    }

    #[test]
    fn empty_reference_is_rejected() {
        let error = switch_aware_error_rates::<&str, &str>(&[], &[], &[], &[0]);
        assert_eq!(error, Err(SwitchMetricError::EmptyReference));
    }

    #[test]
    fn repeated_token_alignment_is_deterministic() {
        let first = switch_aware_error_rates(
            &["a", "x", "a", "x"],
            &["a", "a", "x"],
            &["en", "zu", "en", "zu"],
            &[0, 1, 3],
        ).expect("valid fixture");
        let second = switch_aware_error_rates(
            &["a", "x", "a", "x"],
            &["a", "a", "x"],
            &["en", "zu", "en", "zu"],
            &[0, 1, 3],
        ).expect("valid fixture");
        assert_eq!(first, second);
    }
}
