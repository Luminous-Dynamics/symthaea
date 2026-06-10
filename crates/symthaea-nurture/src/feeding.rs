// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Feeding program and nutritional stages.
//!
//! Models the progression of infant feeding from exclusive milk through
//! introduction of solids to adult diet, following WHO guidelines.
//!
//! Science: WHO (2003) "Global Strategy for Infant and Young Child Feeding".

use serde::{Deserialize, Serialize};

use crate::types::DevelopmentalAge;

/// Feeding stage progression following WHO complementary feeding guidelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedingStage {
    /// 0-6 months: exclusive breast milk or formula.
    ExclusiveMilk,
    /// 6-12 months: introduction of complementary foods.
    IntroductionSolids,
    /// 12-24 months: mixed feeding (milk + solids).
    MixedFeeding,
    /// 24+ months: transition to adult diet.
    AdultDiet,
}

/// Feeding program state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedingProgram {
    /// Current feeding stage.
    pub stage: FeedingStage,
    /// Known food allergies/sensitivities.
    pub allergies_detected: Vec<String>,
    /// Daily caloric target (kcal).
    pub caloric_target: f64,
    /// Number of feedings per day.
    pub feedings_per_day: u32,
    /// Whether the infant is breastfed (affects bonding/oxytocin).
    pub breastfed: bool,
}

impl FeedingProgram {
    /// Create a feeding program appropriate for the given age.
    pub fn for_age(age: &DevelopmentalAge) -> Self {
        let months = age.months;
        let (stage, caloric_target, feedings) = if months < 6.0 {
            (FeedingStage::ExclusiveMilk, 500.0 + months * 15.0, 8)
        } else if months < 12.0 {
            (
                FeedingStage::IntroductionSolids,
                700.0 + (months - 6.0) * 20.0,
                6,
            )
        } else if months < 24.0 {
            (
                FeedingStage::MixedFeeding,
                900.0 + (months - 12.0) * 10.0,
                5,
            )
        } else {
            (FeedingStage::AdultDiet, 1100.0 + (months - 24.0) * 5.0, 4)
        };

        Self {
            stage,
            allergies_detected: Vec::new(),
            caloric_target,
            feedings_per_day: feedings,
            breastfed: months < 12.0, // Default assumption
        }
    }

    /// Advance the feeding program to match a new age.
    pub fn advance(&mut self, age: &DevelopmentalAge) {
        let new = Self::for_age(age);
        self.stage = new.stage;
        self.caloric_target = new.caloric_target;
        self.feedings_per_day = new.feedings_per_day;
        // Preserve allergies and breastfed status
    }

    /// Record a food allergy or sensitivity.
    pub fn record_allergy(&mut self, food: String) {
        if !self.allergies_detected.contains(&food) {
            self.allergies_detected.push(food);
        }
    }

    /// Check if a food has a known allergy.
    pub fn has_allergy(&self, food: &str) -> bool {
        self.allergies_detected.iter().any(|a| a == food)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newborn_exclusive_milk() {
        let program = FeedingProgram::for_age(&DevelopmentalAge::new(1.0));
        assert_eq!(program.stage, FeedingStage::ExclusiveMilk);
        assert!(program.breastfed);
    }

    #[test]
    fn test_six_months_introduction_solids() {
        let program = FeedingProgram::for_age(&DevelopmentalAge::new(7.0));
        assert_eq!(program.stage, FeedingStage::IntroductionSolids);
    }

    #[test]
    fn test_twelve_months_mixed_feeding() {
        let program = FeedingProgram::for_age(&DevelopmentalAge::new(14.0));
        assert_eq!(program.stage, FeedingStage::MixedFeeding);
    }

    #[test]
    fn test_two_years_adult_diet() {
        let program = FeedingProgram::for_age(&DevelopmentalAge::new(30.0));
        assert_eq!(program.stage, FeedingStage::AdultDiet);
    }

    #[test]
    fn test_caloric_target_increases_with_age() {
        let p1 = FeedingProgram::for_age(&DevelopmentalAge::new(1.0));
        let p6 = FeedingProgram::for_age(&DevelopmentalAge::new(7.0));
        let p12 = FeedingProgram::for_age(&DevelopmentalAge::new(14.0));
        let p24 = FeedingProgram::for_age(&DevelopmentalAge::new(30.0));

        assert!(p6.caloric_target > p1.caloric_target);
        assert!(p12.caloric_target > p6.caloric_target);
        assert!(p24.caloric_target > p12.caloric_target);
    }

    #[test]
    fn test_feedings_decrease_with_age() {
        let p1 = FeedingProgram::for_age(&DevelopmentalAge::new(1.0));
        let p12 = FeedingProgram::for_age(&DevelopmentalAge::new(14.0));
        let p24 = FeedingProgram::for_age(&DevelopmentalAge::new(30.0));

        assert!(p1.feedings_per_day > p12.feedings_per_day);
        assert!(p12.feedings_per_day > p24.feedings_per_day);
    }

    #[test]
    fn test_advance_updates_stage() {
        let mut program = FeedingProgram::for_age(&DevelopmentalAge::new(1.0));
        assert_eq!(program.stage, FeedingStage::ExclusiveMilk);

        program.advance(&DevelopmentalAge::new(8.0));
        assert_eq!(program.stage, FeedingStage::IntroductionSolids);

        program.advance(&DevelopmentalAge::new(18.0));
        assert_eq!(program.stage, FeedingStage::MixedFeeding);
    }

    #[test]
    fn test_allergy_tracking() {
        let mut program = FeedingProgram::for_age(&DevelopmentalAge::new(8.0));
        program.record_allergy("peanut".to_string());
        program.record_allergy("egg".to_string());
        program.record_allergy("peanut".to_string()); // Duplicate

        assert!(program.has_allergy("peanut"));
        assert!(program.has_allergy("egg"));
        assert!(!program.has_allergy("milk"));
        assert_eq!(program.allergies_detected.len(), 2); // No duplicate
    }

    #[test]
    fn test_advance_preserves_allergies() {
        let mut program = FeedingProgram::for_age(&DevelopmentalAge::new(8.0));
        program.record_allergy("dairy".to_string());

        program.advance(&DevelopmentalAge::new(18.0));
        assert!(
            program.has_allergy("dairy"),
            "Allergies should persist across advances"
        );
    }
}
