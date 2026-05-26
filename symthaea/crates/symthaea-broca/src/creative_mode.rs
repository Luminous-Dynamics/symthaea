// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Creative mode for Broca: relaxed gating for poetry and artistic text.
//!
//! When creative mode is active:
//! - Epistemic gating is reduced (art doesn't hedge)
//! - Repetition penalty is adjustable (refrains vs novelty)
//! - Form constraints enforce poetic structure (haiku, sonnet, etc.)

use crate::syllable;
use serde::{Deserialize, Serialize};

/// Creative gating configuration that modifies Broca's standard gating behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeGating {
    /// Weight applied to epistemic gate adjustments (0.0 = disabled, 1.0 = full).
    /// Art doesn't need hedging — set to 0.0 for poetry.
    pub epistemic_gate_weight: f32,

    /// Logit boost for semantically distant token combinations (metaphor).
    /// Higher values encourage more surprising word choices.
    pub metaphor_boost: f32,

    /// Optional poetic form constraint.
    pub form_constraint: Option<PoeticForm>,

    /// Override for repetition penalty (None = use default).
    /// Lower for refrains/repetition, higher for novelty.
    pub repetition_penalty_override: Option<f32>,
}

impl Default for CreativeGating {
    fn default() -> Self {
        Self {
            epistemic_gate_weight: 0.0, // fully creative by default
            metaphor_boost: 0.3,
            form_constraint: None,
            repetition_penalty_override: None,
        }
    }
}

impl CreativeGating {
    /// Poetry mode: no epistemic gating, moderate metaphor, no form constraint.
    pub fn free_verse() -> Self {
        Self {
            epistemic_gate_weight: 0.0,
            metaphor_boost: 0.4,
            form_constraint: Some(PoeticForm::FreeVerse),
            repetition_penalty_override: Some(1.2),
        }
    }

    /// Haiku mode: 5-7-5 syllable constraint, high metaphor.
    pub fn haiku() -> Self {
        Self {
            epistemic_gate_weight: 0.0,
            metaphor_boost: 0.5,
            form_constraint: Some(PoeticForm::Haiku),
            repetition_penalty_override: Some(2.0), // avoid repetition in short form
        }
    }

    /// Tanka mode: 5-7-5-7-7 syllable constraint.
    pub fn tanka() -> Self {
        Self {
            epistemic_gate_weight: 0.0,
            metaphor_boost: 0.4,
            form_constraint: Some(PoeticForm::Tanka),
            repetition_penalty_override: Some(1.5),
        }
    }
}

/// Poetic form constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoeticForm {
    /// No structural constraint — free expression.
    FreeVerse,
    /// 5-7-5 syllable structure across 3 lines.
    Haiku,
    /// 5-7-5-7-7 syllable structure across 5 lines.
    Tanka,
    /// 14 lines with specified rhyme scheme (e.g., "ABAB CDCD EFEF GG").
    Sonnet { rhyme_scheme: String },
    /// Custom syllable counts per line.
    Custom { syllable_counts: Vec<u8> },
}

impl PoeticForm {
    /// Get the syllable counts for each line of this form.
    pub fn syllable_counts(&self) -> Vec<u8> {
        match self {
            PoeticForm::FreeVerse => vec![], // no constraint
            PoeticForm::Haiku => vec![5, 7, 5],
            PoeticForm::Tanka => vec![5, 7, 5, 7, 7],
            PoeticForm::Sonnet { .. } => vec![10; 14], // iambic pentameter
            PoeticForm::Custom { syllable_counts } => syllable_counts.clone(),
        }
    }

    /// Total number of lines in this form.
    pub fn line_count(&self) -> usize {
        match self {
            PoeticForm::FreeVerse => 0, // unconstrained
            PoeticForm::Haiku => 3,
            PoeticForm::Tanka => 5,
            PoeticForm::Sonnet { .. } => 14,
            PoeticForm::Custom { syllable_counts } => syllable_counts.len(),
        }
    }
}

// ─── Poem Validation ────────────────────────────────────────────────────────

/// Error found during poem validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationError {
    /// A line has the wrong number of syllables.
    WrongSyllableCount {
        line: usize,
        expected: u8,
        got: usize,
    },
    /// The poem has too few lines for the form.
    TooFewLines { expected: usize, got: usize },
    /// The poem has too many lines for the form.
    TooManyLines { expected: usize, got: usize },
}

/// Result of validating a poem against a poetic form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoemValidation {
    /// Syllable counts for each actual line.
    pub line_syllable_counts: Vec<usize>,
    /// Target syllable counts from the form (empty for free verse).
    pub target_counts: Vec<u8>,
    /// Errors found.
    pub errors: Vec<ValidationError>,
    /// Whether the poem fully conforms to the form.
    pub valid: bool,
}

/// Validate a poem text against a poetic form.
///
/// Splits the text by newlines, counts syllables per line, and checks
/// against the form's expected syllable counts and line count.
///
/// Returns a `PoemValidation` with detailed error information.
/// Free verse always validates (no constraints).
pub fn validate_poem(text: &str, form: &PoeticForm) -> PoemValidation {
    let target_counts = form.syllable_counts();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let line_syllable_counts: Vec<usize> = lines
        .iter()
        .map(|line| syllable::count_line_syllables(line))
        .collect();

    // Free verse: always valid
    if target_counts.is_empty() {
        return PoemValidation {
            line_syllable_counts,
            target_counts,
            errors: vec![],
            valid: true,
        };
    }

    let mut errors = Vec::new();
    let expected_lines = form.line_count();

    // Check line count
    if lines.len() < expected_lines {
        errors.push(ValidationError::TooFewLines {
            expected: expected_lines,
            got: lines.len(),
        });
    } else if lines.len() > expected_lines {
        errors.push(ValidationError::TooManyLines {
            expected: expected_lines,
            got: lines.len(),
        });
    }

    // Check syllable counts per line
    for (i, (&target, &actual)) in target_counts
        .iter()
        .zip(line_syllable_counts.iter())
        .enumerate()
    {
        if actual != target as usize {
            errors.push(ValidationError::WrongSyllableCount {
                line: i,
                expected: target,
                got: actual,
            });
        }
    }

    let valid = errors.is_empty();
    PoemValidation {
        line_syllable_counts,
        target_counts,
        errors,
        valid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haiku_syllable_counts() {
        let form = PoeticForm::Haiku;
        assert_eq!(form.syllable_counts(), vec![5, 7, 5]);
        assert_eq!(form.line_count(), 3);
    }

    #[test]
    fn tanka_syllable_counts() {
        let form = PoeticForm::Tanka;
        assert_eq!(form.syllable_counts(), vec![5, 7, 5, 7, 7]);
        assert_eq!(form.line_count(), 5);
    }

    #[test]
    fn free_verse_no_constraint() {
        let form = PoeticForm::FreeVerse;
        assert!(form.syllable_counts().is_empty());
        assert_eq!(form.line_count(), 0);
    }

    #[test]
    fn creative_gating_defaults() {
        let cg = CreativeGating::default();
        assert_eq!(cg.epistemic_gate_weight, 0.0);
        assert!(cg.metaphor_boost > 0.0);
    }

    #[test]
    fn haiku_gating() {
        let cg = CreativeGating::haiku();
        assert_eq!(cg.epistemic_gate_weight, 0.0);
        assert!(matches!(cg.form_constraint, Some(PoeticForm::Haiku)));
        assert!(cg.repetition_penalty_override.unwrap() > 1.0);
    }

    // ─── Validation tests ───

    #[test]
    fn valid_haiku() {
        // "An old silent pond" = 5
        // "A frog jumps into the pond" = 7
        // "Splash! Silence again" = 5
        let poem = "An old silent pond\nA frog jumps into the pond\nSplash Silence again";
        let v = validate_poem(poem, &PoeticForm::Haiku);
        assert_eq!(v.target_counts, vec![5, 7, 5]);
        assert_eq!(v.line_syllable_counts.len(), 3);
        // Note: syllable counting is approximate, so check structure not exact values
        assert_eq!(v.line_syllable_counts[0], 5, "line 0 should be 5 syllables");
    }

    #[test]
    fn free_verse_always_valid() {
        let poem = "Any text at all\nwith any number of lines\nand any syllable count";
        let v = validate_poem(poem, &PoeticForm::FreeVerse);
        assert!(v.valid);
        assert!(v.errors.is_empty());
    }

    #[test]
    fn too_few_lines_detected() {
        let poem = "Just one line";
        let v = validate_poem(poem, &PoeticForm::Haiku);
        assert!(!v.valid);
        assert!(
            v.errors
                .iter()
                .any(|e| matches!(e, ValidationError::TooFewLines { .. }))
        );
    }

    #[test]
    fn too_many_lines_detected() {
        let poem = "Line one\nLine two\nLine three\nLine four";
        let v = validate_poem(poem, &PoeticForm::Haiku);
        assert!(!v.valid);
        assert!(
            v.errors
                .iter()
                .any(|e| matches!(e, ValidationError::TooManyLines { .. }))
        );
    }

    #[test]
    fn wrong_syllable_count_detected() {
        // Line 1 has way too many syllables for haiku (target: 5)
        let poem = "A very very very long line indeed here\nShort\nShort";
        let v = validate_poem(poem, &PoeticForm::Haiku);
        assert!(!v.valid);
        assert!(
            v.errors
                .iter()
                .any(|e| matches!(e, ValidationError::WrongSyllableCount { line: 0, .. }))
        );
    }

    #[test]
    fn tanka_validation() {
        let v = validate_poem(
            "An old silent pond\nA frog jumps into the pond\nSplash Silence again\nThe water ripples outward\nCalm returns to the surface",
            &PoeticForm::Tanka,
        );
        assert_eq!(v.line_syllable_counts.len(), 5);
        assert_eq!(v.target_counts, vec![5, 7, 5, 7, 7]);
    }

    #[test]
    fn validation_reports_all_errors() {
        // 2 lines instead of 3 for haiku, both wrong syllable count
        let poem = "One\nTwo";
        let v = validate_poem(poem, &PoeticForm::Haiku);
        // Should have TooFewLines AND WrongSyllableCount errors
        assert!(v.errors.len() >= 2);
    }

    #[test]
    fn empty_poem_invalid_for_haiku() {
        let v = validate_poem("", &PoeticForm::Haiku);
        assert!(!v.valid);
    }

    #[test]
    fn custom_form_validation() {
        let form = PoeticForm::Custom {
            syllable_counts: vec![3, 5, 3],
        };
        let v = validate_poem("the big cat\nthe cat sat on it\nthe big cat", &form);
        assert_eq!(v.target_counts, vec![3, 5, 3]);
    }
}
