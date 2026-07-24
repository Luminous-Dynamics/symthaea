//! Dependency-light metrics used by provider conformance and pilot evaluations.

use crate::CalibrationPoint;
use std::collections::BTreeMap;

/// Case-fold, strip punctuation, and collapse whitespace so WER/CER measure
/// content differences rather than casing/punctuation conventions that
/// differ between a reference transcript and a real ASR provider's output
/// (e.g. FLEURS references are lowercase/unpunctuated; Whisper output is
/// not) — matches standard ASR scoring practice (Whisper's own normalizer
/// does the same before computing its published WER numbers).
pub fn normalize_for_scoring(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut last_was_space = true; // Suppress leading whitespace.
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            normalized.extend(ch.to_lowercase());
            last_was_space = false;
        } else if ch.is_whitespace() {
            if !last_was_space {
                normalized.push(' ');
            }
            last_was_space = true;
        }
        // Punctuation and other symbols are dropped entirely, not replaced
        // with a space, so "it's" and "its" normalize identically to how a
        // human transcriber would treat minor punctuation variants.
    }
    normalized.truncate(normalized.trim_end().len());
    normalized
}

pub fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let reference = normalize_for_scoring(reference);
    let hypothesis = normalize_for_scoring(hypothesis);
    let reference: Vec<_> = reference.split_whitespace().collect();
    let hypothesis: Vec<_> = hypothesis.split_whitespace().collect();
    normalized_edit_distance(&reference, &hypothesis)
}

pub fn character_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let reference = normalize_for_scoring(reference);
    let hypothesis = normalize_for_scoring(hypothesis);
    let reference: Vec<_> = reference.chars().collect();
    let hypothesis: Vec<_> = hypothesis.chars().collect();
    normalized_edit_distance(&reference, &hypothesis)
}

fn normalized_edit_distance<T: Eq>(reference: &[T], hypothesis: &[T]) -> f64 {
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    for (row, expected) in reference.iter().enumerate() {
        let mut current = vec![row + 1; hypothesis.len() + 1];
        for (column, actual) in hypothesis.iter().enumerate() {
            current[column + 1] = (current[column] + 1)
                .min(previous[column + 1] + 1)
                .min(previous[column] + usize::from(expected != actual));
        }
        previous = current;
    }
    previous[hypothesis.len()] as f64 / reference.len() as f64
}

/// Character n-gram F-score, averaged over orders 1 through `maximum_order`.
pub fn chrf(reference: &str, hypothesis: &str, maximum_order: usize) -> f64 {
    if maximum_order == 0 {
        return 0.0;
    }
    (1..=maximum_order)
        .map(|order| {
            let expected = ngrams(reference, order);
            let actual = ngrams(hypothesis, order);
            let overlap: usize = expected
                .iter()
                .map(|(gram, count)| count.min(actual.get(gram).unwrap_or(&0)))
                .sum();
            let precision = overlap as f64 / actual.values().sum::<usize>().max(1) as f64;
            let recall = overlap as f64 / expected.values().sum::<usize>().max(1) as f64;
            if precision + recall == 0.0 {
                0.0
            } else {
                2.0 * precision * recall / (precision + recall)
            }
        })
        .sum::<f64>()
        / maximum_order as f64
}

fn ngrams(text: &str, order: usize) -> BTreeMap<Vec<char>, usize> {
    let characters: Vec<_> = text.chars().collect();
    let mut result = BTreeMap::new();
    if characters.len() < order {
        return result;
    }
    for window in characters.windows(order) {
        *result.entry(window.to_vec()).or_insert(0) += 1;
    }
    result
}

pub fn expected_calibration_error(points: &[CalibrationPoint]) -> Option<f64> {
    let total: u64 = points.iter().map(|point| point.sample_count).sum();
    if total == 0
        || points.iter().any(|point| {
            !point.predicted_confidence.is_finite()
                || !point.observed_frequency.is_finite()
                || !(0.0..=1.0).contains(&point.predicted_confidence)
                || !(0.0..=1.0).contains(&point.observed_frequency)
        })
    {
        return None;
    }
    Some(
        points
            .iter()
            .map(|point| {
                (point.predicted_confidence - point.observed_frequency).abs() as f64
                    * point.sample_count as f64
                    / total as f64
            })
            .sum(),
    )
}

pub fn exact_preservation(required: &[String], output: &str) -> f64 {
    if required.is_empty() {
        return 1.0;
    }
    required
        .iter()
        .filter(|value| output.contains(value.as_str()))
        .count() as f64
        / required.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_metrics_are_zero_for_identity() {
        assert_eq!(word_error_rate("sawubona mhlaba", "sawubona mhlaba"), 0.0);
        assert_eq!(character_error_rate("こんにちは", "こんにちは"), 0.0);
        assert_eq!(chrf("hello", "hello", 3), 1.0);
    }

    #[test]
    fn wer_is_insensitive_to_case_and_punctuation() {
        // A lowercase/unpunctuated reference (FLEURS style) against a
        // properly-cased, punctuated hypothesis (real ASR output style)
        // should score as an exact match once normalized.
        let reference = "however due to the slow channels styles could lag";
        let hypothesis = "However, due to the slow channels, styles could lag.";
        assert_eq!(word_error_rate(reference, hypothesis), 0.0);
        assert_eq!(character_error_rate(reference, hypothesis), 0.0);
    }

    #[test]
    fn normalize_for_scoring_collapses_whitespace_and_drops_punctuation() {
        assert_eq!(normalize_for_scoring("  Hello,   World!  "), "hello world");
        assert_eq!(normalize_for_scoring("it's"), "its");
    }

    #[test]
    fn calibration_rejects_invalid_probabilities() {
        assert!(
            expected_calibration_error(&[CalibrationPoint {
                predicted_confidence: 1.2,
                observed_frequency: 1.0,
                sample_count: 2,
            }])
            .is_none()
        );
    }

    #[test]
    fn wer_on_empty_reference_is_zero_for_empty_hypothesis() {
        assert_eq!(word_error_rate("", ""), 0.0);
    }

    #[test]
    fn wer_on_empty_reference_is_one_for_non_empty_hypothesis() {
        assert_eq!(word_error_rate("", "spurious word"), 1.0);
    }

    #[test]
    fn cer_on_empty_hypothesis_is_one() {
        // Every reference character is deleted.
        assert_eq!(character_error_rate("hello", ""), 1.0);
    }

    #[test]
    fn chrf_is_zero_for_order_zero() {
        assert_eq!(chrf("hello", "hello", 0), 0.0);
    }

    #[test]
    fn chrf_is_one_for_identical_strings_any_order() {
        for order in 1..=6 {
            assert_eq!(
                chrf("Sawubona", "Sawubona", order),
                1.0,
                "chrf order {order} should be 1.0 for identical strings"
            );
        }
    }

    #[test]
    fn chrf_is_zero_for_completely_different_strings() {
        // No character overlap at all.
        let score = chrf("aaa", "bbb", 3);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn exact_preservation_empty_required_returns_one() {
        assert_eq!(exact_preservation(&[], "any output"), 1.0);
    }

    #[test]
    fn exact_preservation_partial_match() {
        let required = vec!["alpha".into(), "beta".into(), "gamma".into()];
        let score = exact_preservation(&required, "alpha and gamma are present");
        assert!((score - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn exact_preservation_all_missing_returns_zero() {
        let required = vec!["delta".into(), "epsilon".into()];
        assert_eq!(exact_preservation(&required, "completely unrelated"), 0.0);
    }

    #[test]
    fn calibration_is_zero_for_perfectly_calibrated_single_bin() {
        assert_eq!(
            expected_calibration_error(&[CalibrationPoint {
                predicted_confidence: 0.8,
                observed_frequency: 0.8,
                sample_count: 100,
            }]),
            Some(0.0)
        );
    }

    #[test]
    fn calibration_returns_none_for_zero_samples() {
        assert_eq!(
            expected_calibration_error(&[CalibrationPoint {
                predicted_confidence: 0.5,
                observed_frequency: 0.5,
                sample_count: 0,
            }]),
            None
        );
    }
}
