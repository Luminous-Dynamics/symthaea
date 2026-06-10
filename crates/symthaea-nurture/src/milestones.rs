// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WHO developmental milestone database.
//!
//! Provides a comprehensive database of 30+ developmental milestones across
//! 5 categories (gross motor, fine motor, language, social, cognitive) with
//! normative ranges (5th-95th percentile).
//!
//! Science: WHO Multicentre Growth Reference Study Group (2006),
//! CDC Developmental Milestones (2022).

use serde::{Deserialize, Serialize};

use crate::types::{DevelopmentalAge, MilestoneCategory};

/// A developmental milestone with normative age range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    /// Milestone category.
    pub category: MilestoneCategory,
    /// Short name.
    pub name: String,
    /// Expected age in months (50th percentile).
    pub expected_months: f64,
    /// Early achievers (5th percentile).
    pub range_low_months: f64,
    /// Late achievers (95th percentile).
    pub range_high_months: f64,
}

impl Milestone {
    fn new(category: MilestoneCategory, name: &str, expected: f64, low: f64, high: f64) -> Self {
        Self {
            category,
            name: name.to_string(),
            expected_months: expected,
            range_low_months: low,
            range_high_months: high,
        }
    }
}

/// Developmental milestone database.
pub struct DevelopmentalMilestoneDb;

impl DevelopmentalMilestoneDb {
    /// Return all milestones in the database (30+ entries across 5 categories).
    pub fn all_milestones() -> Vec<Milestone> {
        vec![
            // === Gross Motor ===
            Milestone::new(MilestoneCategory::GrossMotor, "Head lift", 2.0, 1.0, 3.0),
            Milestone::new(MilestoneCategory::GrossMotor, "Roll over", 4.0, 3.0, 6.0),
            Milestone::new(
                MilestoneCategory::GrossMotor,
                "Sitting unsupported",
                6.0,
                5.0,
                8.0,
            ),
            Milestone::new(MilestoneCategory::GrossMotor, "Crawling", 9.0, 6.0, 11.0),
            Milestone::new(
                MilestoneCategory::GrossMotor,
                "Pull to stand",
                10.0,
                8.0,
                12.0,
            ),
            Milestone::new(
                MilestoneCategory::GrossMotor,
                "First steps",
                12.0,
                9.0,
                16.0,
            ),
            Milestone::new(
                MilestoneCategory::GrossMotor,
                "Walking well",
                15.0,
                12.0,
                18.0,
            ),
            Milestone::new(MilestoneCategory::GrossMotor, "Running", 24.0, 18.0, 30.0),
            // === Fine Motor ===
            Milestone::new(MilestoneCategory::FineMotor, "Reaching", 4.0, 3.0, 5.0),
            Milestone::new(MilestoneCategory::FineMotor, "Palmar grasp", 5.0, 4.0, 6.0),
            Milestone::new(
                MilestoneCategory::FineMotor,
                "Transfer objects",
                7.0,
                5.0,
                9.0,
            ),
            Milestone::new(MilestoneCategory::FineMotor, "Pincer grasp", 9.0, 7.0, 12.0),
            Milestone::new(
                MilestoneCategory::FineMotor,
                "Stacking blocks",
                15.0,
                12.0,
                18.0,
            ),
            Milestone::new(MilestoneCategory::FineMotor, "Scribbling", 18.0, 14.0, 22.0),
            Milestone::new(
                MilestoneCategory::FineMotor,
                "Drawing circle",
                36.0,
                30.0,
                42.0,
            ),
            // === Language ===
            Milestone::new(MilestoneCategory::Language, "Cooing", 2.0, 1.0, 4.0),
            Milestone::new(MilestoneCategory::Language, "Babbling", 6.0, 4.0, 8.0),
            Milestone::new(MilestoneCategory::Language, "First words", 10.0, 8.0, 14.0),
            Milestone::new(
                MilestoneCategory::Language,
                "Two-word phrases",
                18.0,
                14.0,
                24.0,
            ),
            Milestone::new(MilestoneCategory::Language, "50+ words", 24.0, 18.0, 30.0),
            Milestone::new(
                MilestoneCategory::Language,
                "Simple sentences",
                36.0,
                30.0,
                42.0,
            ),
            Milestone::new(
                MilestoneCategory::Language,
                "Complex sentences",
                48.0,
                36.0,
                60.0,
            ),
            // === Social ===
            Milestone::new(MilestoneCategory::Social, "Social smile", 2.0, 1.0, 3.0),
            Milestone::new(
                MilestoneCategory::Social,
                "Stranger anxiety",
                8.0,
                6.0,
                10.0,
            ),
            Milestone::new(
                MilestoneCategory::Social,
                "Separation anxiety",
                9.0,
                6.0,
                12.0,
            ),
            Milestone::new(MilestoneCategory::Social, "Parallel play", 24.0, 18.0, 30.0),
            Milestone::new(
                MilestoneCategory::Social,
                "Cooperative play",
                36.0,
                30.0,
                48.0,
            ),
            Milestone::new(
                MilestoneCategory::Social,
                "Theory of mind",
                48.0,
                36.0,
                60.0,
            ),
            // === Cognitive ===
            Milestone::new(
                MilestoneCategory::Cognitive,
                "Object permanence",
                8.0,
                6.0,
                10.0,
            ),
            Milestone::new(
                MilestoneCategory::Cognitive,
                "Means-end reasoning",
                10.0,
                8.0,
                12.0,
            ),
            Milestone::new(
                MilestoneCategory::Cognitive,
                "Symbolic play",
                18.0,
                14.0,
                24.0,
            ),
            Milestone::new(
                MilestoneCategory::Cognitive,
                "Sorting by category",
                24.0,
                20.0,
                30.0,
            ),
            Milestone::new(
                MilestoneCategory::Cognitive,
                "Counting to 10",
                36.0,
                30.0,
                48.0,
            ),
            Milestone::new(
                MilestoneCategory::Cognitive,
                "Conservation",
                60.0,
                48.0,
                72.0,
            ),
        ]
    }

    /// Return milestones expected by a given age (expected_months <= age).
    pub fn milestones_for_age(age: &DevelopmentalAge) -> Vec<Milestone> {
        Self::all_milestones()
            .into_iter()
            .filter(|m| m.expected_months <= age.months)
            .collect()
    }

    /// Return milestones expected in a given age range.
    pub fn milestones_in_range(min_months: f64, max_months: f64) -> Vec<Milestone> {
        Self::all_milestones()
            .into_iter()
            .filter(|m| m.expected_months >= min_months && m.expected_months <= max_months)
            .collect()
    }

    /// Check if a milestone is delayed (achieved later than 95th percentile).
    pub fn is_delayed(milestone_name: &str, actual_months: f64) -> bool {
        Self::all_milestones()
            .iter()
            .find(|m| m.name == milestone_name)
            .map(|m| actual_months > m.range_high_months)
            .unwrap_or(false)
    }

    /// Check if a milestone is advanced (achieved earlier than 5th percentile).
    pub fn is_advanced(milestone_name: &str, actual_months: f64) -> bool {
        Self::all_milestones()
            .iter()
            .find(|m| m.name == milestone_name)
            .map(|m| actual_months < m.range_low_months)
            .unwrap_or(false)
    }

    /// Get milestones for a specific category.
    pub fn milestones_for_category(category: MilestoneCategory) -> Vec<Milestone> {
        Self::all_milestones()
            .into_iter()
            .filter(|m| m.category == category)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_milestones_count() {
        let all = DevelopmentalMilestoneDb::all_milestones();
        assert!(
            all.len() >= 30,
            "Should have 30+ milestones, got {}",
            all.len()
        );
    }

    #[test]
    fn test_all_categories_represented() {
        let all = DevelopmentalMilestoneDb::all_milestones();
        let categories: Vec<MilestoneCategory> = vec![
            MilestoneCategory::GrossMotor,
            MilestoneCategory::FineMotor,
            MilestoneCategory::Language,
            MilestoneCategory::Social,
            MilestoneCategory::Cognitive,
        ];

        for cat in categories {
            let count = all.iter().filter(|m| m.category == cat).count();
            assert!(
                count >= 5,
                "{:?} should have 5+ milestones, got {}",
                cat,
                count
            );
        }
    }

    #[test]
    fn test_range_valid() {
        for m in DevelopmentalMilestoneDb::all_milestones() {
            assert!(
                m.range_low_months <= m.expected_months,
                "{}: low ({}) should be <= expected ({})",
                m.name,
                m.range_low_months,
                m.expected_months
            );
            assert!(
                m.expected_months <= m.range_high_months,
                "{}: expected ({}) should be <= high ({})",
                m.name,
                m.expected_months,
                m.range_high_months
            );
        }
    }

    #[test]
    fn test_milestones_for_age_12() {
        let at_12 = DevelopmentalMilestoneDb::milestones_for_age(&DevelopmentalAge::new(12.0));
        let names: Vec<&str> = at_12.iter().map(|m| m.name.as_str()).collect();

        assert!(names.contains(&"First steps"));
        assert!(names.contains(&"Social smile"));
        assert!(names.contains(&"Babbling"));
        assert!(names.contains(&"First words"));
        assert!(!names.contains(&"Running"));
        assert!(!names.contains(&"Simple sentences"));
    }

    #[test]
    fn test_milestones_for_age_0() {
        let at_0 = DevelopmentalMilestoneDb::milestones_for_age(&DevelopmentalAge::new(0.0));
        assert!(at_0.is_empty());
    }

    #[test]
    fn test_is_delayed() {
        // First steps expected at 12mo, range_high = 16mo
        assert!(!DevelopmentalMilestoneDb::is_delayed("First steps", 14.0));
        assert!(DevelopmentalMilestoneDb::is_delayed("First steps", 17.0));
    }

    #[test]
    fn test_is_advanced() {
        // First steps range_low = 9mo
        assert!(DevelopmentalMilestoneDb::is_advanced("First steps", 8.0));
        assert!(!DevelopmentalMilestoneDb::is_advanced("First steps", 10.0));
    }

    #[test]
    fn test_milestones_for_category() {
        let gross =
            DevelopmentalMilestoneDb::milestones_for_category(MilestoneCategory::GrossMotor);
        assert!(gross.len() >= 6);
        for m in &gross {
            assert_eq!(m.category, MilestoneCategory::GrossMotor);
        }
    }

    #[test]
    fn test_milestones_in_range() {
        let range = DevelopmentalMilestoneDb::milestones_in_range(6.0, 12.0);
        assert!(!range.is_empty());
        for m in &range {
            assert!(
                m.expected_months >= 6.0 && m.expected_months <= 12.0,
                "{} at {}mo is outside 6-12mo range",
                m.name,
                m.expected_months
            );
        }
    }

    #[test]
    fn test_unknown_milestone_not_delayed() {
        assert!(!DevelopmentalMilestoneDb::is_delayed("Flying", 100.0));
    }

    #[test]
    fn test_social_milestones_progression() {
        let social = DevelopmentalMilestoneDb::milestones_for_category(MilestoneCategory::Social);
        // Social smile should come before theory of mind
        let smile_age = social
            .iter()
            .find(|m| m.name == "Social smile")
            .unwrap()
            .expected_months;
        let tom_age = social
            .iter()
            .find(|m| m.name == "Theory of mind")
            .unwrap()
            .expected_months;
        assert!(smile_age < tom_age);
    }

    #[test]
    fn test_cognitive_milestones_progression() {
        let cognitive =
            DevelopmentalMilestoneDb::milestones_for_category(MilestoneCategory::Cognitive);
        // Object permanence before conservation
        let op_age = cognitive
            .iter()
            .find(|m| m.name == "Object permanence")
            .unwrap()
            .expected_months;
        let cons_age = cognitive
            .iter()
            .find(|m| m.name == "Conservation")
            .unwrap()
            .expected_months;
        assert!(op_age < cons_age);
    }
}
