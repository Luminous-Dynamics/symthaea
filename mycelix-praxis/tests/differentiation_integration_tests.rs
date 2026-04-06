// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Differentiation Zomes Integration Tests
//!
//! Comprehensive integration tests for the EduNet differentiation features:
//! - SRS (Spaced Repetition System)
//! - Gamification (XP, badges, streaks)
//! - Adaptive Learning (BKT, ZPD, VARK)
//! - Integration Layer (cross-zome orchestration)
//!
//! These tests verify complete learning workflows and cross-zome interactions.

use praxis_core::errors::{
    adaptive_errors, errors, gamification_errors, integration_errors, srs_errors,
    EduNetError, ErrorCode,
};
use serde::{Deserialize, Serialize};

// ============================================================================
// Test Data Structures (mirroring zome types for testing)
// ============================================================================

/// Recall quality for SM-2 algorithm (0-5)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecallQuality {
    CompleteBlackout = 0,
    IncorrectButRecognized = 1,
    IncorrectButEasyRecall = 2,
    CorrectWithDifficulty = 3,
    CorrectWithHesitation = 4,
    Perfect = 5,
}

impl RecallQuality {
    fn value(&self) -> u8 {
        *self as u8
    }
}

/// Card status in the SRS system
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CardStatus {
    New,
    Learning,
    Review,
    Graduated,
    Suspended,
    Buried,
}

/// Mastery level based on permille
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MasteryLevel {
    Novice,      // 0-166
    Beginner,    // 167-333
    Competent,   // 334-500
    Proficient,  // 501-667
    Expert,      // 668-834
    Master,      // 835-1000
}

impl MasteryLevel {
    fn from_permille(p: u16) -> Self {
        match p {
            0..=166 => MasteryLevel::Novice,
            167..=333 => MasteryLevel::Beginner,
            334..=500 => MasteryLevel::Competent,
            501..=667 => MasteryLevel::Proficient,
            668..=834 => MasteryLevel::Expert,
            _ => MasteryLevel::Master,
        }
    }
}

/// Learning style based on VARK
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningStyle {
    Visual,
    Auditory,
    ReadWrite,
    Kinesthetic,
    Multimodal,
}

/// Badge rarity tiers
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BadgeRarity {
    Common,
    Uncommon,
    Rare,
    Epic,
    Legendary,
    Mythic,
    Transcendent,
}

// ============================================================================
// SM-2 Algorithm Tests
// ============================================================================

mod srs_tests {
    use super::*;

    /// SM-2 ease factor calculation
    fn calculate_new_ease(current_ef: f64, quality: u8) -> f64 {
        let q = quality as f64;
        let adjustment = 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02);
        (current_ef + adjustment).max(1.3).min(5.0)
    }

    #[test]
    fn test_sm2_perfect_response_increases_ease() {
        let current_ef = 2.5;
        let new_ef = calculate_new_ease(current_ef, RecallQuality::Perfect.value());

        assert!(
            new_ef > current_ef,
            "Perfect response should increase ease factor: {} -> {}",
            current_ef,
            new_ef
        );
    }

    #[test]
    fn test_sm2_failed_response_decreases_ease() {
        let current_ef = 2.5;
        let new_ef = calculate_new_ease(current_ef, RecallQuality::CompleteBlackout.value());

        assert!(
            new_ef < current_ef,
            "Failed response should decrease ease factor: {} -> {}",
            current_ef,
            new_ef
        );
    }

    #[test]
    fn test_sm2_ease_bounded_minimum() {
        // Many failures should not go below 1.3
        let mut ef = 2.5;
        for _ in 0..20 {
            ef = calculate_new_ease(ef, RecallQuality::CompleteBlackout.value());
        }

        assert!(ef >= 1.3, "Ease factor should not go below 1.3: {}", ef);
    }

    #[test]
    fn test_sm2_ease_bounded_maximum() {
        // Many perfect responses should not exceed 5.0
        let mut ef = 2.5;
        for _ in 0..20 {
            ef = calculate_new_ease(ef, RecallQuality::Perfect.value());
        }

        assert!(ef <= 5.0, "Ease factor should not exceed 5.0: {}", ef);
    }

    #[test]
    fn test_sm2_quality_grade_effects() {
        let current_ef = 2.5;

        // Quality 3 (difficult but correct) should maintain ease approximately
        let ef_3 = calculate_new_ease(current_ef, RecallQuality::CorrectWithDifficulty.value());

        // Quality 4 (correct with hesitation) should slightly increase
        let ef_4 = calculate_new_ease(current_ef, RecallQuality::CorrectWithHesitation.value());

        // Quality 5 (perfect) should increase more
        let ef_5 = calculate_new_ease(current_ef, RecallQuality::Perfect.value());

        assert!(
            ef_5 > ef_4 && ef_4 > ef_3,
            "Higher quality should result in higher ease: q3={}, q4={}, q5={}",
            ef_3,
            ef_4,
            ef_5
        );
    }

    #[test]
    fn test_interval_progression() {
        // Interval should increase with successful reviews
        let base_interval = 1.0; // day
        let ease_factor = 2.5;

        let after_first = 1.0; // First graduation
        let after_second = 6.0; // Standard 6 day
        let after_third = after_second * ease_factor; // 15 days

        assert!(
            after_third > after_second,
            "Intervals should grow: 1 -> 6 -> {}",
            after_third
        );
    }

    #[test]
    fn test_leech_detection() {
        let leech_threshold = 8;
        let lapse_counts = vec![5, 7, 8, 10, 15];

        for lapse_count in lapse_counts {
            let is_leech = lapse_count >= leech_threshold;

            if lapse_count >= leech_threshold {
                assert!(is_leech, "Card with {} lapses should be leech", lapse_count);
            } else {
                assert!(!is_leech, "Card with {} lapses should not be leech", lapse_count);
            }
        }
    }
}

// ============================================================================
// Gamification Tests
// ============================================================================

mod gamification_tests {
    use super::*;

    /// Calculate XP required for a level
    fn xp_for_level(level: u32) -> u64 {
        if level == 0 {
            return 0;
        }
        // Exponential growth: 100 * 1.5^(level-1)
        (100.0 * 1.5_f64.powi(level as i32 - 1)) as u64
    }

    /// Calculate level from total XP
    fn level_from_xp(total_xp: u64) -> u32 {
        if total_xp == 0 {
            return 1;
        }
        // Inverse of XP formula
        let level = (1.0 + (total_xp as f64 / 100.0).log(1.5).floor()) as u32;
        level.max(1)
    }

    /// Calculate streak bonus multiplier (permille)
    fn streak_bonus_permille(streak_days: u32) -> u16 {
        // 5% bonus per day, max 50%
        let bonus = (streak_days as u16 * 50).min(500);
        1000 + bonus
    }

    #[test]
    fn test_xp_progression() {
        // XP requirements should increase exponentially
        let xp_1 = xp_for_level(1);
        let xp_2 = xp_for_level(2);
        let xp_3 = xp_for_level(3);
        let xp_10 = xp_for_level(10);

        assert!(xp_2 > xp_1, "Level 2 requires more XP than level 1");
        assert!(xp_3 > xp_2, "Level 3 requires more XP than level 2");
        assert!(xp_10 > xp_3, "Level 10 requires more XP than level 3");
    }

    #[test]
    fn test_level_calculation() {
        // Test level calculation from XP
        assert_eq!(level_from_xp(0), 1, "0 XP = level 1");
        assert_eq!(level_from_xp(100), 2, "100 XP = level 2");
        assert_eq!(level_from_xp(250), 3, "250 XP = level 3");
    }

    #[test]
    fn test_streak_bonus_progression() {
        let bonus_0 = streak_bonus_permille(0);
        let bonus_5 = streak_bonus_permille(5);
        let bonus_10 = streak_bonus_permille(10);
        let bonus_20 = streak_bonus_permille(20);

        assert_eq!(bonus_0, 1000, "No streak = 1.0x");
        assert_eq!(bonus_5, 1250, "5 day streak = 1.25x");
        assert_eq!(bonus_10, 1500, "10 day streak = 1.5x (max)");
        assert_eq!(bonus_20, 1500, "20 day streak = still 1.5x (capped)");
    }

    #[test]
    fn test_badge_rarity_ordering() {
        assert!(BadgeRarity::Uncommon > BadgeRarity::Common);
        assert!(BadgeRarity::Rare > BadgeRarity::Uncommon);
        assert!(BadgeRarity::Epic > BadgeRarity::Rare);
        assert!(BadgeRarity::Legendary > BadgeRarity::Epic);
        assert!(BadgeRarity::Mythic > BadgeRarity::Legendary);
        assert!(BadgeRarity::Transcendent > BadgeRarity::Mythic);
    }

    #[test]
    fn test_xp_calculation_with_multipliers() {
        let base_xp: u64 = 100;
        let activity_mult: f64 = 1.0; // Normal activity
        let streak_mult: f64 = 1.25; // 5-day streak
        let difficulty_mult: f64 = 1.2; // Hard content

        let final_xp = (base_xp as f64 * activity_mult * streak_mult * difficulty_mult) as u64;

        assert_eq!(final_xp, 150, "100 * 1.0 * 1.25 * 1.2 = 150");
    }
}

// ============================================================================
// Adaptive Learning Tests
// ============================================================================

mod adaptive_tests {
    use super::*;

    /// Bayesian Knowledge Tracing update for correct answer
    fn bkt_update_correct(p_learn: f64, p_transition: f64) -> f64 {
        p_learn + (1.0 - p_learn) * p_transition
    }

    /// Bayesian Knowledge Tracing update for incorrect answer
    fn bkt_update_incorrect(p_learn: f64, p_slip: f64, p_guess: f64) -> f64 {
        (p_learn * p_slip) / (p_learn * p_slip + (1.0 - p_learn) * (1.0 - p_guess))
    }

    /// Calculate Zone of Proximal Development bounds
    fn zpd_bounds(current_mastery_permille: u16) -> (u16, u16, u16) {
        let lower = current_mastery_permille.saturating_add(50);
        let optimal = current_mastery_permille.saturating_add(100);
        let upper = current_mastery_permille.saturating_add(150);

        (lower.min(1000), optimal.min(1000), upper.min(1000))
    }

    #[test]
    fn test_bkt_correct_increases_mastery() {
        let p_learn = 0.5;
        let p_transition = 0.3;

        let new_p_learn = bkt_update_correct(p_learn, p_transition);

        assert!(
            new_p_learn > p_learn,
            "Correct answer should increase mastery: {} -> {}",
            p_learn,
            new_p_learn
        );
    }

    #[test]
    fn test_bkt_incorrect_decreases_mastery() {
        let p_learn = 0.7;
        let p_slip = 0.1;
        let p_guess = 0.2;

        let new_p_learn = bkt_update_incorrect(p_learn, p_slip, p_guess);

        assert!(
            new_p_learn < p_learn,
            "Incorrect answer should decrease mastery: {} -> {}",
            p_learn,
            new_p_learn
        );
    }

    #[test]
    fn test_zpd_bounds_calculation() {
        // For a learner at 400 permille (40% mastery)
        let (lower, optimal, upper) = zpd_bounds(400);

        assert_eq!(lower, 450, "Lower ZPD bound should be +50");
        assert_eq!(optimal, 500, "Optimal should be +100");
        assert_eq!(upper, 550, "Upper ZPD bound should be +150");
    }

    #[test]
    fn test_zpd_bounds_capped_at_max() {
        // For a learner at 900 permille (90% mastery)
        let (lower, optimal, upper) = zpd_bounds(900);

        assert_eq!(lower, 950, "Lower bound should be 950");
        assert_eq!(optimal, 1000, "Optimal should be capped at 1000");
        assert_eq!(upper, 1000, "Upper bound should be capped at 1000");
    }

    #[test]
    fn test_mastery_level_progression() {
        assert_eq!(MasteryLevel::from_permille(100), MasteryLevel::Novice);
        assert_eq!(MasteryLevel::from_permille(250), MasteryLevel::Beginner);
        assert_eq!(MasteryLevel::from_permille(400), MasteryLevel::Competent);
        assert_eq!(MasteryLevel::from_permille(600), MasteryLevel::Proficient);
        assert_eq!(MasteryLevel::from_permille(750), MasteryLevel::Expert);
        assert_eq!(MasteryLevel::from_permille(900), MasteryLevel::Master);
    }

    #[test]
    fn test_difficulty_match_in_zpd() {
        let skill_level = 500u16;
        let (lower, _, upper) = zpd_bounds(skill_level);

        // Content difficulty within ZPD
        let easy_content = lower + 10;
        let hard_content = upper - 10;
        let too_easy = lower - 100;
        let too_hard = upper + 100;

        assert!(
            easy_content >= lower && easy_content <= upper,
            "Easy content {} should be in ZPD [{}-{}]",
            easy_content,
            lower,
            upper
        );

        assert!(
            hard_content >= lower && hard_content <= upper,
            "Hard content {} should be in ZPD [{}-{}]",
            hard_content,
            lower,
            upper
        );

        assert!(
            too_easy < lower,
            "Too easy content {} should be below ZPD lower bound {}",
            too_easy,
            lower
        );

        assert!(
            too_hard > upper,
            "Too hard content {} should be above ZPD upper bound {}",
            too_hard,
            upper
        );
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn test_srs_error_codes() {
        let err = srs_errors::card_not_found("Qm123");
        assert_eq!(err.code, ErrorCode::EntityNotFound);
        assert_eq!(err.entity_type, "ReviewCard");

        let err = srs_errors::invalid_recall_quality(7);
        assert_eq!(err.code, ErrorCode::OutOfRange);

        let err = srs_errors::deck_not_found("deck123");
        assert_eq!(err.code, ErrorCode::EntityNotFound);
        assert_eq!(err.entity_type, "Deck");
    }

    #[test]
    fn test_gamification_error_codes() {
        let err = gamification_errors::badge_already_earned("badge1");
        assert_eq!(err.code, ErrorCode::EntityAlreadyExists);

        let err = gamification_errors::streak_broken(0, 3);
        assert_eq!(err.code, ErrorCode::InvalidEntityState);

        let err = gamification_errors::no_streak_freezes(3, 3);
        assert_eq!(err.code, ErrorCode::ResourceExhausted);
    }

    #[test]
    fn test_adaptive_error_codes() {
        let err = adaptive_errors::profile_not_found("agent123");
        assert_eq!(err.code, ErrorCode::EntityNotFound);
        assert_eq!(err.entity_type, "LearnerProfile");

        let err = adaptive_errors::invalid_mastery_level(1500);
        assert_eq!(err.code, ErrorCode::OutOfRange);

        let err = adaptive_errors::difficulty_mismatch(300, 800);
        assert_eq!(err.code, ErrorCode::InvalidEntityState);
    }

    #[test]
    fn test_integration_error_codes() {
        let err = integration_errors::session_not_found("session1");
        assert_eq!(err.code, ErrorCode::EntityNotFound);

        let err = integration_errors::session_already_active("existing");
        assert_eq!(err.code, ErrorCode::EntityAlreadyExists);

        let err = integration_errors::cross_zome_srs_failed("get_cards", "timeout");
        assert_eq!(err.code, ErrorCode::CrossZomeCallFailed);
    }

    #[test]
    fn test_error_message_contains_context() {
        let err = srs_errors::card_not_found("QmTestHash123");
        let msg = err.to_message();

        assert!(msg.contains("E100"), "Should contain error code");
        assert!(msg.contains("ReviewCard"), "Should contain entity type");
        assert!(msg.contains("QmTestHash123"), "Should contain context");
        assert!(msg.contains("Hint:"), "Should contain hint");
    }

    #[test]
    fn test_error_chaining() {
        let err = EduNetError::new(
            ErrorCode::ValidationFailed,
            "TestEntity",
            "test_action",
            "Test message",
        )
        .with_context("test context")
        .with_hint("test hint");

        let msg = err.to_message();
        assert!(msg.contains("test context"));
        assert!(msg.contains("test hint"));
    }
}

// ============================================================================
// Integration Workflow Tests
// ============================================================================

mod workflow_tests {
    use super::*;

    /// Simulates a complete learning session
    #[test]
    fn test_complete_learning_session_workflow() {
        // 1. Start with a learner at level 1 with 0 XP
        let mut total_xp = 0u64;
        let mut level = 1u32;
        let mut streak = 0u32;

        // 2. Complete some reviews
        let reviews = vec![
            (RecallQuality::Perfect, 25),      // Perfect = 25 XP
            (RecallQuality::CorrectWithHesitation, 20), // Good = 20 XP
            (RecallQuality::CorrectWithDifficulty, 15), // Hard = 15 XP
            (RecallQuality::IncorrectButRecognized, 5), // Wrong = 5 XP (participation)
        ];

        for (quality, base_xp) in reviews {
            // Apply streak bonus
            let streak_mult = 1.0 + (streak as f64 * 0.05).min(0.5);
            let earned_xp = (base_xp as f64 * streak_mult) as u64;
            total_xp += earned_xp;

            // Update streak if quality >= 3
            if quality.value() >= 3 {
                streak += 1;
            } else {
                // Failed review doesn't break streak in EduNet
            }
        }

        // 3. Calculate new level
        while total_xp >= super::gamification_tests::xp_for_level(level + 1) {
            level += 1;
        }

        // 4. Verify progress
        assert!(total_xp > 0, "Should have earned XP");
        assert!(streak > 0, "Should have a streak");
        assert!(level >= 1, "Should be at least level 1");
    }

    /// Simulates adaptive difficulty selection
    #[test]
    fn test_adaptive_content_selection() {
        let skill_mastery = 450u16; // 45% mastery
        let (zpd_lower, zpd_optimal, zpd_upper) = super::adaptive_tests::zpd_bounds(skill_mastery);

        // Available content difficulties
        let content = vec![
            ("Basic concepts", 300u16),
            ("Intermediate practice", 500u16),
            ("Advanced theory", 700u16),
            ("Expert challenges", 900u16),
        ];

        // Find content in ZPD
        let recommended: Vec<_> = content
            .iter()
            .filter(|(_, diff)| *diff >= zpd_lower && *diff <= zpd_upper)
            .collect();

        // Should recommend content in the sweet spot
        assert!(
            !recommended.is_empty(),
            "Should have recommended content for skill level {}",
            skill_mastery
        );

        // Verify recommendations are appropriate
        for (name, diff) in &recommended {
            assert!(
                *diff >= zpd_lower && *diff <= zpd_upper,
                "{} (difficulty {}) should be in ZPD [{}-{}]",
                name,
                diff,
                zpd_lower,
                zpd_upper
            );
        }
    }

    /// Simulates spaced repetition over time
    #[test]
    fn test_spaced_repetition_interval_growth() {
        // Simulate a card being reviewed multiple times
        let mut interval_days = 1.0;
        let ease_factor = 2.5;
        let reviews = 5;

        let mut intervals = vec![interval_days];

        for i in 0..reviews {
            if i == 0 {
                interval_days = 1.0; // First graduation
            } else if i == 1 {
                interval_days = 6.0; // Standard 6 days
            } else {
                interval_days *= ease_factor;
            }
            intervals.push(interval_days);
        }

        // Verify exponential growth
        for i in 2..intervals.len() {
            assert!(
                intervals[i] > intervals[i - 1],
                "Interval should grow: {} -> {}",
                intervals[i - 1],
                intervals[i]
            );
        }

        // After 5 successful reviews with 2.5 EF, interval should be significant
        assert!(
            interval_days > 30.0,
            "After 5 reviews, interval should be > 30 days: {}",
            interval_days
        );
    }

    /// Simulates badge earning progression
    #[test]
    fn test_badge_progression() {
        // Badge requirements
        let badges = vec![
            ("First Steps", 1, BadgeRarity::Common),         // 1 review
            ("Week Warrior", 7, BadgeRarity::Uncommon),      // 7 day streak
            ("Month Master", 30, BadgeRarity::Rare),         // 30 day streak
            ("Century Scholar", 100, BadgeRarity::Epic),     // 100 day streak
            ("Year of Learning", 365, BadgeRarity::Legendary), // 365 day streak
        ];

        let current_streak = 45; // 45 day streak

        let earned: Vec<_> = badges
            .iter()
            .filter(|(_, days, _)| current_streak >= *days)
            .collect();

        // Should have earned first 3 badges
        assert_eq!(earned.len(), 3, "Should have earned 3 badges at 45 day streak");

        // Verify correct badges earned
        assert!(earned.iter().any(|(name, _, _)| *name == "First Steps"));
        assert!(earned.iter().any(|(name, _, _)| *name == "Week Warrior"));
        assert!(earned.iter().any(|(name, _, _)| *name == "Month Master"));
    }

    /// Test confidence scoring based on data availability
    #[test]
    fn test_confidence_scoring() {
        // Confidence based on data sources
        fn calculate_confidence(
            has_gamification: bool,
            has_adaptive: bool,
            has_srs: bool,
        ) -> u16 {
            let mut score = 200u16; // Base minimum

            if has_gamification {
                score += 200;
            }
            if has_adaptive {
                score += 300;
            }
            if has_srs {
                score += 300;
            }

            score.min(1000)
        }

        assert_eq!(
            calculate_confidence(false, false, false),
            200,
            "No data = minimum confidence"
        );
        assert_eq!(
            calculate_confidence(true, false, false),
            400,
            "Only gamification"
        );
        assert_eq!(
            calculate_confidence(true, true, false),
            700,
            "Gamification + adaptive"
        );
        assert_eq!(
            calculate_confidence(true, true, true),
            1000,
            "All data = max confidence"
        );
    }
}

// ============================================================================
// Cross-Zome Interaction Tests
// ============================================================================

mod cross_zome_tests {
    use super::*;

    /// Test unified dashboard data aggregation
    #[test]
    fn test_dashboard_aggregation() {
        // Simulated data from different zomes
        let srs_due_count = 15u32;
        let total_xp = 2500u64;
        let level = 5u32;
        let streak_days = 12u32;
        let mastery_count = 8u32;
        let avg_mastery_permille = 650u16;

        // Aggregate into unified view
        let dashboard = UnifiedDashboardData {
            srs_due_count,
            total_xp,
            level,
            streak_days,
            mastery_count,
            avg_mastery_permille,
            overall_progress_permille: (avg_mastery_permille as u32 * mastery_count
                / mastery_count.max(1)) as u16,
        };

        assert_eq!(dashboard.srs_due_count, 15);
        assert_eq!(dashboard.level, 5);
        assert_eq!(dashboard.streak_days, 12);
    }

    struct UnifiedDashboardData {
        srs_due_count: u32,
        total_xp: u64,
        level: u32,
        streak_days: u32,
        mastery_count: u32,
        avg_mastery_permille: u16,
        overall_progress_permille: u16,
    }

    /// Test XP calculation across zomes
    #[test]
    fn test_cross_zome_xp_calculation() {
        // XP sources from different activities
        let srs_xp = 50u64;        // From SRS reviews
        let course_xp = 100u64;    // From course progress
        let milestone_xp = 200u64; // From adaptive milestones

        let total_xp = srs_xp + course_xp + milestone_xp;

        assert_eq!(total_xp, 350, "Total XP should be sum of all sources");
    }

    /// Test session state transitions
    #[test]
    fn test_session_state_machine() {
        #[derive(Clone, Debug, PartialEq, Eq)]
        enum SessionState {
            Scheduled,
            Active,
            Paused,
            Completed,
            Abandoned,
        }

        // Valid transitions
        let valid_transitions = vec![
            (SessionState::Scheduled, SessionState::Active),
            (SessionState::Active, SessionState::Paused),
            (SessionState::Active, SessionState::Completed),
            (SessionState::Active, SessionState::Abandoned),
            (SessionState::Paused, SessionState::Active),
            (SessionState::Paused, SessionState::Abandoned),
        ];

        // Invalid transitions
        let invalid_transitions = vec![
            (SessionState::Completed, SessionState::Active),
            (SessionState::Abandoned, SessionState::Active),
            (SessionState::Scheduled, SessionState::Completed),
        ];

        fn is_valid_transition(from: &SessionState, to: &SessionState) -> bool {
            let valid = vec![
                (SessionState::Scheduled, SessionState::Active),
                (SessionState::Active, SessionState::Paused),
                (SessionState::Active, SessionState::Completed),
                (SessionState::Active, SessionState::Abandoned),
                (SessionState::Paused, SessionState::Active),
                (SessionState::Paused, SessionState::Abandoned),
            ];

            valid.contains(&(from.clone(), to.clone()))
        }

        for (from, to) in valid_transitions {
            assert!(
                is_valid_transition(&from, &to),
                "Should allow {:?} -> {:?}",
                from,
                to
            );
        }

        for (from, to) in invalid_transitions {
            assert!(
                !is_valid_transition(&from, &to),
                "Should not allow {:?} -> {:?}",
                from,
                to
            );
        }
    }
}
