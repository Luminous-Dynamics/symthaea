// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evaluation Metrics Module
//!
//! Provides metrics for evaluating speech recognition accuracy:
//! - Phoneme Error Rate (PER)
//! - Word Error Rate (WER)
//! - Character Error Rate (CER)
//! - Confusion matrices
//!
//! All error rates are computed using the Levenshtein edit distance.

use std::collections::HashMap;

/// Edit operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditOp {
    /// Correct match
    Correct,
    /// Substitution error
    Substitution,
    /// Insertion error (extra token in hypothesis)
    Insertion,
    /// Deletion error (missing token in hypothesis)
    Deletion,
}

/// Alignment result between reference and hypothesis
#[derive(Debug, Clone)]
pub struct Alignment<T: Clone> {
    /// Reference tokens
    pub reference: Vec<Option<T>>,
    /// Hypothesis tokens
    pub hypothesis: Vec<Option<T>>,
    /// Edit operations
    pub operations: Vec<EditOp>,
}

impl<T: Clone> Alignment<T> {
    /// Count correct matches
    pub fn correct(&self) -> usize {
        self.operations
            .iter()
            .filter(|&&op| op == EditOp::Correct)
            .count()
    }

    /// Count substitutions
    pub fn substitutions(&self) -> usize {
        self.operations
            .iter()
            .filter(|&&op| op == EditOp::Substitution)
            .count()
    }

    /// Count insertions
    pub fn insertions(&self) -> usize {
        self.operations
            .iter()
            .filter(|&&op| op == EditOp::Insertion)
            .count()
    }

    /// Count deletions
    pub fn deletions(&self) -> usize {
        self.operations
            .iter()
            .filter(|&&op| op == EditOp::Deletion)
            .count()
    }

    /// Total errors
    pub fn errors(&self) -> usize {
        self.substitutions() + self.insertions() + self.deletions()
    }
}

/// Evaluation result
#[derive(Debug, Clone, Default)]
pub struct EvalResult {
    /// Total reference tokens
    pub reference_length: usize,
    /// Total hypothesis tokens
    pub hypothesis_length: usize,
    /// Number of correct tokens
    pub correct: usize,
    /// Number of substitutions
    pub substitutions: usize,
    /// Number of insertions
    pub insertions: usize,
    /// Number of deletions
    pub deletions: usize,
}

impl EvalResult {
    /// Compute error rate
    pub fn error_rate(&self) -> f32 {
        if self.reference_length == 0 {
            return if self.hypothesis_length == 0 {
                0.0
            } else {
                1.0
            };
        }
        (self.substitutions + self.insertions + self.deletions) as f32
            / self.reference_length as f32
    }

    /// Compute accuracy
    pub fn accuracy(&self) -> f32 {
        1.0 - self.error_rate()
    }

    /// Total errors
    pub fn total_errors(&self) -> usize {
        self.substitutions + self.insertions + self.deletions
    }

    /// Merge with another result
    pub fn merge(&mut self, other: &EvalResult) {
        self.reference_length += other.reference_length;
        self.hypothesis_length += other.hypothesis_length;
        self.correct += other.correct;
        self.substitutions += other.substitutions;
        self.insertions += other.insertions;
        self.deletions += other.deletions;
    }
}

/// Compute Levenshtein edit distance and alignment
pub fn levenshtein_align<T: PartialEq + Clone>(reference: &[T], hypothesis: &[T]) -> Alignment<T> {
    let n = reference.len();
    let m = hypothesis.len();

    // DP table for edit distance
    let mut dp = vec![vec![0usize; m + 1]; n + 1];

    // Initialize base cases
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }

    // Fill DP table
    for i in 1..=n {
        for j in 1..=m {
            let cost = if reference[i - 1] == hypothesis[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1) // deletion
                .min(dp[i][j - 1] + 1) // insertion
                .min(dp[i - 1][j - 1] + cost); // substitution/match
        }
    }

    // Backtrack to get alignment
    let mut ref_aligned = Vec::new();
    let mut hyp_aligned = Vec::new();
    let mut ops = Vec::new();

    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && reference[i - 1] == hypothesis[j - 1] {
            // Match
            ref_aligned.push(Some(reference[i - 1].clone()));
            hyp_aligned.push(Some(hypothesis[j - 1].clone()));
            ops.push(EditOp::Correct);
            i -= 1;
            j -= 1;
        } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
            // Substitution
            ref_aligned.push(Some(reference[i - 1].clone()));
            hyp_aligned.push(Some(hypothesis[j - 1].clone()));
            ops.push(EditOp::Substitution);
            i -= 1;
            j -= 1;
        } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
            // Insertion
            ref_aligned.push(None);
            hyp_aligned.push(Some(hypothesis[j - 1].clone()));
            ops.push(EditOp::Insertion);
            j -= 1;
        } else {
            // Deletion
            ref_aligned.push(Some(reference[i - 1].clone()));
            hyp_aligned.push(None);
            ops.push(EditOp::Deletion);
            i -= 1;
        }
    }

    // Reverse to get correct order
    ref_aligned.reverse();
    hyp_aligned.reverse();
    ops.reverse();

    Alignment {
        reference: ref_aligned,
        hypothesis: hyp_aligned,
        operations: ops,
    }
}

/// Compute evaluation result from alignment
pub fn eval_from_alignment<T: Clone>(alignment: &Alignment<T>) -> EvalResult {
    EvalResult {
        reference_length: alignment.reference.iter().filter(|t| t.is_some()).count(),
        hypothesis_length: alignment.hypothesis.iter().filter(|t| t.is_some()).count(),
        correct: alignment.correct(),
        substitutions: alignment.substitutions(),
        insertions: alignment.insertions(),
        deletions: alignment.deletions(),
    }
}

// ============================================================================
// PHONEME ERROR RATE
// ============================================================================

/// Compute Phoneme Error Rate
pub fn phoneme_error_rate(reference: &[&str], hypothesis: &[&str]) -> f32 {
    let alignment = levenshtein_align(reference, hypothesis);
    let result = eval_from_alignment(&alignment);
    result.error_rate()
}

/// Compute PER with detailed results
pub fn phoneme_error_rate_detailed<'a>(
    reference: &[&'a str],
    hypothesis: &[&'a str],
) -> (f32, EvalResult, Alignment<&'a str>) {
    let alignment = levenshtein_align(reference, hypothesis);
    let result = eval_from_alignment(&alignment);
    let per = result.error_rate();
    (per, result, alignment)
}

/// Evaluate multiple utterances and return aggregate PER
pub fn evaluate_phonemes(references: &[Vec<String>], hypotheses: &[Vec<String>]) -> EvalResult {
    assert_eq!(
        references.len(),
        hypotheses.len(),
        "Number of references and hypotheses must match"
    );

    let mut total = EvalResult::default();

    for (reference, hypothesis) in references.iter().zip(hypotheses.iter()) {
        let ref_strs: Vec<&str> = reference.iter().map(|s| s.as_str()).collect();
        let hyp_strs: Vec<&str> = hypothesis.iter().map(|s| s.as_str()).collect();

        let alignment = levenshtein_align(&ref_strs, &hyp_strs);
        let result = eval_from_alignment(&alignment);
        total.merge(&result);
    }

    total
}

// ============================================================================
// WORD ERROR RATE
// ============================================================================

/// Compute Word Error Rate
pub fn word_error_rate(reference: &str, hypothesis: &str) -> f32 {
    let ref_words: Vec<&str> = reference.split_whitespace().collect();
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();

    let alignment = levenshtein_align(&ref_words, &hyp_words);
    let result = eval_from_alignment(&alignment);
    result.error_rate()
}

/// Compute WER with detailed results
pub fn word_error_rate_detailed<'a>(
    reference: &'a str,
    hypothesis: &'a str,
) -> (f32, EvalResult, Alignment<&'a str>) {
    let ref_words: Vec<&str> = reference.split_whitespace().collect();
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();

    let alignment = levenshtein_align(&ref_words, &hyp_words);
    let result = eval_from_alignment(&alignment);
    let wer = result.error_rate();
    (wer, result, alignment)
}

/// Evaluate multiple utterances and return aggregate WER
pub fn evaluate_words(references: &[&str], hypotheses: &[&str]) -> EvalResult {
    assert_eq!(
        references.len(),
        hypotheses.len(),
        "Number of references and hypotheses must match"
    );

    let mut total = EvalResult::default();

    for (reference, hypothesis) in references.iter().zip(hypotheses.iter()) {
        let ref_words: Vec<&str> = reference.split_whitespace().collect();
        let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();

        let alignment = levenshtein_align(&ref_words, &hyp_words);
        let result = eval_from_alignment(&alignment);
        total.merge(&result);
    }

    total
}

// ============================================================================
// CHARACTER ERROR RATE
// ============================================================================

/// Compute Character Error Rate
pub fn character_error_rate(reference: &str, hypothesis: &str) -> f32 {
    let ref_chars: Vec<char> = reference.chars().collect();
    let hyp_chars: Vec<char> = hypothesis.chars().collect();

    let alignment = levenshtein_align(&ref_chars, &hyp_chars);
    let result = eval_from_alignment(&alignment);
    result.error_rate()
}

// ============================================================================
// CONFUSION MATRIX
// ============================================================================

/// Confusion matrix for token substitutions
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// Counts: (reference, hypothesis) -> count
    confusions: HashMap<(String, String), usize>,
    /// Total per reference token
    reference_counts: HashMap<String, usize>,
}

impl ConfusionMatrix {
    /// Create a new confusion matrix
    pub fn new() -> Self {
        Self {
            confusions: HashMap::new(),
            reference_counts: HashMap::new(),
        }
    }

    /// Add a substitution
    pub fn add_substitution(&mut self, reference: &str, hypothesis: &str) {
        *self
            .confusions
            .entry((reference.to_string(), hypothesis.to_string()))
            .or_insert(0) += 1;
        *self
            .reference_counts
            .entry(reference.to_string())
            .or_insert(0) += 1;
    }

    /// Add a correct match
    pub fn add_correct(&mut self, token: &str) {
        *self
            .confusions
            .entry((token.to_string(), token.to_string()))
            .or_insert(0) += 1;
        *self.reference_counts.entry(token.to_string()).or_insert(0) += 1;
    }

    /// Get count for a specific confusion
    pub fn get(&self, reference: &str, hypothesis: &str) -> usize {
        self.confusions
            .get(&(reference.to_string(), hypothesis.to_string()))
            .cloned()
            .unwrap_or(0)
    }

    /// Get most common confusions
    pub fn top_confusions(&self, n: usize) -> Vec<((String, String), usize)> {
        let mut confusions: Vec<_> = self
            .confusions
            .iter()
            .filter(|((r, h), _)| r != h) // Exclude correct matches
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        confusions.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        confusions.into_iter().take(n).collect()
    }

    /// Get accuracy per token
    pub fn token_accuracy(&self, token: &str) -> f32 {
        let total = self.reference_counts.get(token).cloned().unwrap_or(0);
        if total == 0 {
            return 0.0;
        }
        let correct = self.get(token, token);
        correct as f32 / total as f32
    }

    /// Build from alignments
    pub fn from_alignments(alignments: &[Alignment<String>]) -> Self {
        let mut matrix = Self::new();

        for alignment in alignments {
            for (i, op) in alignment.operations.iter().enumerate() {
                match op {
                    EditOp::Correct => {
                        if let Some(ref token) = alignment.reference[i] {
                            matrix.add_correct(token);
                        }
                    }
                    EditOp::Substitution => {
                        if let (Some(r), Some(h)) =
                            (&alignment.reference[i], &alignment.hypothesis[i])
                        {
                            matrix.add_substitution(r, h);
                        }
                    }
                    _ => {}
                }
            }
        }

        matrix
    }
}

impl Default for ConfusionMatrix {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// EVALUATION REPORT
// ============================================================================

/// Comprehensive evaluation report
#[derive(Debug, Clone)]
pub struct EvaluationReport {
    /// Number of utterances
    pub num_utterances: usize,
    /// Phoneme-level results
    pub phoneme_results: Option<EvalResult>,
    /// Word-level results
    pub word_results: Option<EvalResult>,
    /// Character-level results
    pub char_results: Option<EvalResult>,
    /// Confusion matrix
    pub confusion_matrix: Option<ConfusionMatrix>,
}

impl std::fmt::Display for EvaluationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Evaluation Report ({} utterances)", self.num_utterances)?;
        writeln!(f, "{}", "=".repeat(50))?;

        if let Some(ref per) = self.phoneme_results {
            writeln!(f)?;
            writeln!(f, "Phoneme Error Rate (PER)")?;
            write!(f, "{}", "-".repeat(30))?;
            writeln!(f, "\n  Reference phonemes: {}", per.reference_length)?;
            writeln!(f, "  Hypothesis phonemes: {}", per.hypothesis_length)?;
            writeln!(f, "  Correct: {}", per.correct)?;
            writeln!(f, "  Substitutions: {}", per.substitutions)?;
            writeln!(f, "  Insertions: {}", per.insertions)?;
            writeln!(f, "  Deletions: {}", per.deletions)?;
            writeln!(f, "  PER: {:.2}%", per.error_rate() * 100.0)?;
            writeln!(f, "  Accuracy: {:.2}%", per.accuracy() * 100.0)?;
        }

        if let Some(ref wer) = self.word_results {
            writeln!(f)?;
            writeln!(f, "Word Error Rate (WER)")?;
            write!(f, "{}", "-".repeat(30))?;
            writeln!(f, "\n  Reference words: {}", wer.reference_length)?;
            writeln!(f, "  Hypothesis words: {}", wer.hypothesis_length)?;
            writeln!(f, "  Correct: {}", wer.correct)?;
            writeln!(f, "  Substitutions: {}", wer.substitutions)?;
            writeln!(f, "  Insertions: {}", wer.insertions)?;
            writeln!(f, "  Deletions: {}", wer.deletions)?;
            writeln!(f, "  WER: {:.2}%", wer.error_rate() * 100.0)?;
            writeln!(f, "  Accuracy: {:.2}%", wer.accuracy() * 100.0)?;
        }

        if let Some(ref cer) = self.char_results {
            writeln!(f)?;
            writeln!(f, "Character Error Rate (CER)")?;
            write!(f, "{}", "-".repeat(30))?;
            writeln!(f, "\n  CER: {:.2}%", cer.error_rate() * 100.0)?;
        }

        if let Some(ref cm) = self.confusion_matrix {
            writeln!(f)?;
            writeln!(f, "Top Confusions")?;
            write!(f, "{}", "-".repeat(30))?;
            for ((r, h), count) in cm.top_confusions(10) {
                write!(f, "\n  {r} -> {h}: {count}")?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_identical() {
        let reference = vec!["A", "B", "C"];
        let hypothesis = vec!["A", "B", "C"];

        let alignment = levenshtein_align(&reference, &hypothesis);
        assert_eq!(alignment.correct(), 3);
        assert_eq!(alignment.errors(), 0);
    }

    #[test]
    fn test_levenshtein_substitution() {
        let reference = vec!["A", "B", "C"];
        let hypothesis = vec!["A", "X", "C"];

        let alignment = levenshtein_align(&reference, &hypothesis);
        assert_eq!(alignment.correct(), 2);
        assert_eq!(alignment.substitutions(), 1);
    }

    #[test]
    fn test_levenshtein_insertion() {
        let reference = vec!["A", "C"];
        let hypothesis = vec!["A", "B", "C"];

        let alignment = levenshtein_align(&reference, &hypothesis);
        assert_eq!(alignment.correct(), 2);
        assert_eq!(alignment.insertions(), 1);
    }

    #[test]
    fn test_levenshtein_deletion() {
        let reference = vec!["A", "B", "C"];
        let hypothesis = vec!["A", "C"];

        let alignment = levenshtein_align(&reference, &hypothesis);
        assert_eq!(alignment.correct(), 2);
        assert_eq!(alignment.deletions(), 1);
    }

    #[test]
    fn test_phoneme_error_rate() {
        let reference = vec!["AH", "B", "AW", "T"];
        let hypothesis = vec!["AH", "B", "OW", "T"];

        let per = phoneme_error_rate(&reference, &hypothesis);
        assert!((per - 0.25).abs() < 0.01); // 1 error / 4 tokens = 25%
    }

    #[test]
    fn test_word_error_rate() {
        let reference = "the cat sat on the mat";
        let hypothesis = "the cat sat in the mat";

        let wer = word_error_rate(reference, hypothesis);
        assert!((wer - 1.0 / 6.0).abs() < 0.01); // 1 substitution / 6 words
    }

    #[test]
    fn test_confusion_matrix() {
        let mut cm = ConfusionMatrix::new();

        cm.add_correct("AA");
        cm.add_correct("AA");
        cm.add_substitution("AA", "AH");
        cm.add_substitution("IH", "IY");

        assert_eq!(cm.get("AA", "AA"), 2);
        assert_eq!(cm.get("AA", "AH"), 1);
        assert!((cm.token_accuracy("AA") - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_eval_result_merge() {
        let mut r1 = EvalResult {
            reference_length: 10,
            hypothesis_length: 9,
            correct: 8,
            substitutions: 1,
            insertions: 0,
            deletions: 1,
        };

        let r2 = EvalResult {
            reference_length: 5,
            hypothesis_length: 6,
            correct: 4,
            substitutions: 0,
            insertions: 1,
            deletions: 1,
        };

        r1.merge(&r2);

        assert_eq!(r1.reference_length, 15);
        assert_eq!(r1.correct, 12);
        assert_eq!(r1.total_errors(), 4);
    }
}
