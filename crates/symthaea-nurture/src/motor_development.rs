// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Motor development milestones.
//!
//! Tracks both gross motor (whole body movement) and fine motor (hand/finger
//! dexterity) milestone progression. Based on WHO Motor Development Study (2006).
//!
//! Science: WHO Multicentre Growth Reference Study Group (2006).

use serde::{Deserialize, Serialize};

use crate::types::{DevelopmentalAge, MilestoneCategory};

/// A motor development milestone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorMilestone {
    /// Expected age in months.
    pub age_months: f64,
    /// Motor milestone category.
    pub category: MilestoneCategory,
    /// Short name of the milestone.
    pub name: String,
    /// Description of the motor achievement.
    pub description: String,
}

/// Return the standard gross motor milestone sequence.
pub fn gross_motor_milestones() -> Vec<MotorMilestone> {
    vec![
        MotorMilestone {
            age_months: 1.0,
            category: MilestoneCategory::GrossMotor,
            name: "Head lift prone".into(),
            description: "Lifts head briefly when on tummy".into(),
        },
        MotorMilestone {
            age_months: 2.0,
            category: MilestoneCategory::GrossMotor,
            name: "Head lift sustained".into(),
            description: "Lifts head 45 degrees during tummy time".into(),
        },
        MotorMilestone {
            age_months: 3.0,
            category: MilestoneCategory::GrossMotor,
            name: "Head control".into(),
            description: "Holds head steady when upright".into(),
        },
        MotorMilestone {
            age_months: 4.0,
            category: MilestoneCategory::GrossMotor,
            name: "Roll over".into(),
            description: "Rolls from tummy to back".into(),
        },
        MotorMilestone {
            age_months: 5.0,
            category: MilestoneCategory::GrossMotor,
            name: "Roll both ways".into(),
            description: "Rolls both tummy-to-back and back-to-tummy".into(),
        },
        MotorMilestone {
            age_months: 6.0,
            category: MilestoneCategory::GrossMotor,
            name: "Sitting".into(),
            description: "Sits without support".into(),
        },
        MotorMilestone {
            age_months: 7.0,
            category: MilestoneCategory::GrossMotor,
            name: "Supported standing".into(),
            description: "Stands while holding onto support".into(),
        },
        MotorMilestone {
            age_months: 9.0,
            category: MilestoneCategory::GrossMotor,
            name: "Crawling".into(),
            description: "Crawls on hands and knees".into(),
        },
        MotorMilestone {
            age_months: 10.0,
            category: MilestoneCategory::GrossMotor,
            name: "Pull to stand".into(),
            description: "Pulls self to standing position".into(),
        },
        MotorMilestone {
            age_months: 11.0,
            category: MilestoneCategory::GrossMotor,
            name: "Cruising".into(),
            description: "Walks while holding furniture".into(),
        },
        MotorMilestone {
            age_months: 12.0,
            category: MilestoneCategory::GrossMotor,
            name: "First steps".into(),
            description: "Takes first independent steps".into(),
        },
        MotorMilestone {
            age_months: 15.0,
            category: MilestoneCategory::GrossMotor,
            name: "Walking steady".into(),
            description: "Walks without falling frequently".into(),
        },
        MotorMilestone {
            age_months: 18.0,
            category: MilestoneCategory::GrossMotor,
            name: "Walking well".into(),
            description: "Walks well independently".into(),
        },
        MotorMilestone {
            age_months: 24.0,
            category: MilestoneCategory::GrossMotor,
            name: "Running".into(),
            description: "Runs with coordination".into(),
        },
        MotorMilestone {
            age_months: 30.0,
            category: MilestoneCategory::GrossMotor,
            name: "Jumping".into(),
            description: "Jumps with both feet off ground".into(),
        },
        MotorMilestone {
            age_months: 36.0,
            category: MilestoneCategory::GrossMotor,
            name: "Balance".into(),
            description: "Stands on one foot briefly".into(),
        },
    ]
}

/// Return the standard fine motor milestone sequence.
pub fn fine_motor_milestones() -> Vec<MotorMilestone> {
    vec![
        MotorMilestone {
            age_months: 1.0,
            category: MilestoneCategory::FineMotor,
            name: "Grasp reflex".into(),
            description: "Reflexive grasp when object placed in hand".into(),
        },
        MotorMilestone {
            age_months: 3.0,
            category: MilestoneCategory::FineMotor,
            name: "Hands open".into(),
            description: "Hands frequently open, not fisted".into(),
        },
        MotorMilestone {
            age_months: 4.0,
            category: MilestoneCategory::FineMotor,
            name: "Reaching".into(),
            description: "Reaches for objects".into(),
        },
        MotorMilestone {
            age_months: 5.0,
            category: MilestoneCategory::FineMotor,
            name: "Palmar grasp".into(),
            description: "Grasps objects with whole hand".into(),
        },
        MotorMilestone {
            age_months: 7.0,
            category: MilestoneCategory::FineMotor,
            name: "Transfer".into(),
            description: "Transfers objects from hand to hand".into(),
        },
        MotorMilestone {
            age_months: 9.0,
            category: MilestoneCategory::FineMotor,
            name: "Pincer grasp emerging".into(),
            description: "Picks up small objects with thumb and finger".into(),
        },
        MotorMilestone {
            age_months: 12.0,
            category: MilestoneCategory::FineMotor,
            name: "Pincer grasp mature".into(),
            description: "Neat pincer grasp (thumb + index fingertip)".into(),
        },
        MotorMilestone {
            age_months: 15.0,
            category: MilestoneCategory::FineMotor,
            name: "Stacking".into(),
            description: "Stacks 2 blocks".into(),
        },
        MotorMilestone {
            age_months: 18.0,
            category: MilestoneCategory::FineMotor,
            name: "Scribbling".into(),
            description: "Scribbles with crayon".into(),
        },
        MotorMilestone {
            age_months: 24.0,
            category: MilestoneCategory::FineMotor,
            name: "Tower building".into(),
            description: "Stacks 6+ blocks into tower".into(),
        },
        MotorMilestone {
            age_months: 36.0,
            category: MilestoneCategory::FineMotor,
            name: "Drawing".into(),
            description: "Copies circle, draws person (head + legs)".into(),
        },
    ]
}

/// Get motor milestones expected by a given age.
pub fn motor_milestones_expected_by(age: &DevelopmentalAge) -> Vec<String> {
    let months = age.months;
    let mut expected = Vec::new();

    for m in gross_motor_milestones()
        .iter()
        .chain(fine_motor_milestones().iter())
    {
        if months >= m.age_months {
            expected.push(m.name.clone());
        }
    }

    expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gross_motor_milestones_ordered() {
        let milestones = gross_motor_milestones();
        for i in 1..milestones.len() {
            assert!(
                milestones[i].age_months >= milestones[i - 1].age_months,
                "Gross motor milestones should be ordered"
            );
        }
    }

    #[test]
    fn test_fine_motor_milestones_ordered() {
        let milestones = fine_motor_milestones();
        for i in 1..milestones.len() {
            assert!(
                milestones[i].age_months >= milestones[i - 1].age_months,
                "Fine motor milestones should be ordered"
            );
        }
    }

    #[test]
    fn test_gross_motor_count() {
        let milestones = gross_motor_milestones();
        assert!(
            milestones.len() >= 10,
            "Should have 10+ gross motor milestones"
        );
    }

    #[test]
    fn test_fine_motor_count() {
        let milestones = fine_motor_milestones();
        assert!(
            milestones.len() >= 8,
            "Should have 8+ fine motor milestones"
        );
    }

    #[test]
    fn test_all_gross_motor_category() {
        for m in gross_motor_milestones() {
            assert_eq!(m.category, MilestoneCategory::GrossMotor);
        }
    }

    #[test]
    fn test_all_fine_motor_category() {
        for m in fine_motor_milestones() {
            assert_eq!(m.category, MilestoneCategory::FineMotor);
        }
    }

    #[test]
    fn test_expected_at_12_months() {
        let expected = motor_milestones_expected_by(&DevelopmentalAge::new(12.0));
        assert!(expected.contains(&"First steps".to_string()));
        assert!(expected.contains(&"Crawling".to_string()));
        assert!(expected.contains(&"Pincer grasp mature".to_string()));
        assert!(!expected.contains(&"Running".to_string()));
    }

    #[test]
    fn test_expected_at_birth() {
        let expected = motor_milestones_expected_by(&DevelopmentalAge::new(0.0));
        assert!(expected.is_empty());
    }

    #[test]
    fn test_expected_at_36_months() {
        let expected = motor_milestones_expected_by(&DevelopmentalAge::new(36.0));
        assert!(expected.contains(&"Running".to_string()));
        assert!(expected.contains(&"Balance".to_string()));
        assert!(expected.contains(&"Drawing".to_string()));
    }

    #[test]
    fn test_walking_before_running() {
        let gross = gross_motor_milestones();
        let walk_idx = gross.iter().position(|m| m.name == "First steps").unwrap();
        let run_idx = gross.iter().position(|m| m.name == "Running").unwrap();
        assert!(walk_idx < run_idx, "Walking should come before running");
    }
}
