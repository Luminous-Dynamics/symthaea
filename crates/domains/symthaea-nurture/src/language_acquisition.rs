// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Language acquisition and infant-directed speech (IDS).
//!
//! Models the progression of language development from cooing through complex
//! sentences, and the characteristics of infant-directed speech (motherese)
//! that caregivers naturally use.
//!
//! IDS parameters are designed for integration with the vocal-tract crate:
//! higher F0, exaggerated prosody, slower rate, simplified vocabulary.
//!
//! Science: Kuhl (2004), Fernald (1985), Tomasello (2003).

use serde::{Deserialize, Serialize};

use crate::types::DevelopmentalAge;

/// Infant-Directed Speech (IDS) parameters for vocal-tract integration.
///
/// IDS (motherese/parentese) has characteristic acoustic properties:
/// - Higher fundamental frequency (F0)
/// - Wider pitch range (exaggerated prosody)
/// - Slower speaking rate
/// - Simplified vocabulary
///
/// These parameters decrease toward adult values as the infant matures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfantDirectedSpeech {
    /// F0 multiplier relative to adult speech (IDS is ~1.5x higher).
    pub f0_multiplier: f64,
    /// F0 range multiplier (IDS has ~2x the pitch range).
    pub f0_range_multiplier: f64,
    /// Speaking rate relative to adult (IDS is ~0.6-0.7x slower).
    pub speaking_rate: f64,
    /// Target vocabulary size (simplified for infant).
    pub vocabulary_level: u32,
}

impl InfantDirectedSpeech {
    /// Create IDS parameters appropriate for the given age.
    ///
    /// Parameters gradually approach adult speech values as the infant ages:
    /// - 0-12 months: maximum exaggeration
    /// - 12-24 months: moderately exaggerated
    /// - 24+ months: approaching normal speech
    pub fn for_age(age: &DevelopmentalAge) -> Self {
        let months = age.months;
        Self {
            f0_multiplier: if months < 12.0 {
                1.5
            } else if months < 24.0 {
                1.3
            } else {
                1.1
            },
            f0_range_multiplier: if months < 12.0 {
                2.0
            } else if months < 24.0 {
                1.5
            } else {
                1.2
            },
            speaking_rate: if months < 12.0 {
                0.6
            } else if months < 24.0 {
                0.75
            } else {
                0.9
            },
            vocabulary_level: (months * 5.0).min(2000.0) as u32,
        }
    }

    /// Whether the speech parameters are still in IDS mode (significantly different from adult).
    pub fn is_ids_active(&self) -> bool {
        self.f0_multiplier > 1.15
    }
}

/// A single language development milestone with normative age.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageMilestone {
    /// Age at which milestone typically occurs (months).
    pub age_months: f64,
    /// Description of the language milestone (e.g., "Babbling", "Two-word phrases").
    pub milestone: String,
}

/// Return the standard sequence of language development milestones.
///
/// Based on CDC developmental milestones and normative data.
pub fn language_milestones() -> Vec<LanguageMilestone> {
    vec![
        LanguageMilestone {
            age_months: 2.0,
            milestone: "Cooing".into(),
        },
        LanguageMilestone {
            age_months: 4.0,
            milestone: "Vocal play".into(),
        },
        LanguageMilestone {
            age_months: 6.0,
            milestone: "Babbling".into(),
        },
        LanguageMilestone {
            age_months: 8.0,
            milestone: "Canonical babbling".into(),
        },
        LanguageMilestone {
            age_months: 10.0,
            milestone: "First words".into(),
        },
        LanguageMilestone {
            age_months: 12.0,
            milestone: "Pointing with vocalization".into(),
        },
        LanguageMilestone {
            age_months: 15.0,
            milestone: "10+ word vocabulary".into(),
        },
        LanguageMilestone {
            age_months: 18.0,
            milestone: "Two-word phrases".into(),
        },
        LanguageMilestone {
            age_months: 24.0,
            milestone: "50+ word vocabulary".into(),
        },
        LanguageMilestone {
            age_months: 30.0,
            milestone: "Multi-word sentences".into(),
        },
        LanguageMilestone {
            age_months: 36.0,
            milestone: "Simple sentences".into(),
        },
        LanguageMilestone {
            age_months: 42.0,
            milestone: "Questions and negation".into(),
        },
        LanguageMilestone {
            age_months: 48.0,
            milestone: "Complex sentences".into(),
        },
        LanguageMilestone {
            age_months: 60.0,
            milestone: "Narrative ability".into(),
        },
    ]
}

/// Get language milestones expected by a given age.
pub fn milestones_expected_by(age: &DevelopmentalAge) -> Vec<&'static str> {
    let months = age.months;
    let mut expected = Vec::new();
    if months >= 2.0 {
        expected.push("Cooing");
    }
    if months >= 4.0 {
        expected.push("Vocal play");
    }
    if months >= 6.0 {
        expected.push("Babbling");
    }
    if months >= 8.0 {
        expected.push("Canonical babbling");
    }
    if months >= 10.0 {
        expected.push("First words");
    }
    if months >= 12.0 {
        expected.push("Pointing with vocalization");
    }
    if months >= 15.0 {
        expected.push("10+ word vocabulary");
    }
    if months >= 18.0 {
        expected.push("Two-word phrases");
    }
    if months >= 24.0 {
        expected.push("50+ word vocabulary");
    }
    if months >= 30.0 {
        expected.push("Multi-word sentences");
    }
    if months >= 36.0 {
        expected.push("Simple sentences");
    }
    if months >= 42.0 {
        expected.push("Questions and negation");
    }
    if months >= 48.0 {
        expected.push("Complex sentences");
    }
    if months >= 60.0 {
        expected.push("Narrative ability");
    }
    expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ids_newborn_exaggerated() {
        let ids = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(3.0));
        assert!((ids.f0_multiplier - 1.5).abs() < 1e-10);
        assert!((ids.f0_range_multiplier - 2.0).abs() < 1e-10);
        assert!((ids.speaking_rate - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_ids_toddler_moderate() {
        let ids = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(18.0));
        assert!((ids.f0_multiplier - 1.3).abs() < 1e-10);
        assert!((ids.speaking_rate - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_ids_approaches_adult_by_24_months() {
        let ids = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(30.0));
        assert!((ids.f0_multiplier - 1.1).abs() < 1e-10);
        assert!((ids.speaking_rate - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_ids_f0_decreases_with_age() {
        let ids3 = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(3.0));
        let ids18 = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(18.0));
        let ids30 = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(30.0));

        assert!(ids3.f0_multiplier > ids18.f0_multiplier);
        assert!(ids18.f0_multiplier > ids30.f0_multiplier);
    }

    #[test]
    fn test_ids_speaking_rate_increases_with_age() {
        let ids3 = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(3.0));
        let ids18 = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(18.0));
        let ids30 = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(30.0));

        assert!(ids3.speaking_rate < ids18.speaking_rate);
        assert!(ids18.speaking_rate < ids30.speaking_rate);
    }

    #[test]
    fn test_vocabulary_increases_with_age() {
        let ids6 = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(6.0));
        let ids24 = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(24.0));
        let ids48 = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(48.0));

        assert!(ids24.vocabulary_level > ids6.vocabulary_level);
        assert!(ids48.vocabulary_level > ids24.vocabulary_level);
    }

    #[test]
    fn test_vocabulary_capped_at_2000() {
        let ids = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(600.0));
        assert_eq!(ids.vocabulary_level, 2000);
    }

    #[test]
    fn test_is_ids_active() {
        let young = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(6.0));
        assert!(young.is_ids_active());

        // At 30 months, f0_multiplier = 1.1, which is < 1.15
        let older = InfantDirectedSpeech::for_age(&DevelopmentalAge::new(30.0));
        assert!(!older.is_ids_active());
    }

    #[test]
    fn test_language_milestones_sequence() {
        let milestones = language_milestones();
        assert!(milestones.len() >= 10, "Should have 10+ milestones");

        // Should be ordered by age
        for i in 1..milestones.len() {
            assert!(
                milestones[i].age_months >= milestones[i - 1].age_months,
                "Milestones should be ordered: {} >= {}",
                milestones[i].age_months,
                milestones[i - 1].age_months
            );
        }
    }

    #[test]
    fn test_milestones_expected_by_age() {
        let at_12 = milestones_expected_by(&DevelopmentalAge::new(12.0));
        assert!(at_12.contains(&"Cooing"));
        assert!(at_12.contains(&"Babbling"));
        assert!(at_12.contains(&"First words"));
        assert!(!at_12.contains(&"Two-word phrases"));

        let at_36 = milestones_expected_by(&DevelopmentalAge::new(36.0));
        assert!(at_36.contains(&"Simple sentences"));
        assert!(!at_36.contains(&"Complex sentences"));
    }

    #[test]
    fn test_milestones_expected_empty_at_birth() {
        let at_0 = milestones_expected_by(&DevelopmentalAge::new(0.0));
        assert!(at_0.is_empty());
    }
}
