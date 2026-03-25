// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Creative mode for Broca: relaxed gating for poetry and artistic text.
//!
//! When creative mode is active:
//! - Epistemic gating is reduced (art doesn't hedge)
//! - Repetition penalty is adjustable (refrains vs novelty)
//! - Form constraints enforce poetic structure (haiku, sonnet, etc.)

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
}
