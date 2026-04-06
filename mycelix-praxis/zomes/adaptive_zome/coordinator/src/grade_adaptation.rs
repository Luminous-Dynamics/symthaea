// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Grade-level adaptive tuning for PreK-12 education.
//!
//! Adjusts mastery thresholds, ZPD ranges, and pacing based on developmental stage.
//! Grade ordinals: 0=PreK, 1=K, 2=Grade1, ..., 13=Grade12.

/// Mastery threshold by grade level (permille).
/// Younger students need higher mastery before advancing (more repetition).
pub fn mastery_threshold_for_grade(grade_ordinal: u8) -> u16 {
    match grade_ordinal {
        0..=1 => 900,   // PreK-K: 90% mastery required
        2..=4 => 850,   // Grade 1-3: 85%
        5..=7 => 800,   // Grade 4-6: 80%
        8..=10 => 800,  // Grade 7-9: 80%
        11..=13 => 750, // Grade 10-12: 75%
        _ => 800,        // Default
    }
}

/// Zone of Proximal Development range (min, max offset in permille).
/// Younger students get a narrower ZPD (gentler difficulty progression).
pub fn zpd_range_for_grade(grade_ordinal: u8) -> (u16, u16) {
    match grade_ordinal {
        0..=1 => (50, 100),    // PreK-K: very gentle
        2..=4 => (75, 150),    // Grade 1-3: gentle
        5..=7 => (100, 200),   // Grade 4-6: moderate
        8..=10 => (100, 250),  // Grade 7-9: challenging
        11..=13 => (100, 300), // Grade 10-12: demanding
        _ => (100, 200),
    }
}

/// Recommended session length in minutes by grade.
pub fn session_length_for_grade(grade_ordinal: u8) -> u32 {
    match grade_ordinal {
        0..=1 => 15,   // PreK-K: 15 min max
        2..=4 => 25,   // Grade 1-3: 25 min
        5..=7 => 35,   // Grade 4-6: 35 min
        8..=10 => 45,  // Grade 7-9: 45 min
        11..=13 => 55, // Grade 10-12: 55 min
        _ => 45,
    }
}

/// New cards per day in SRS by grade.
pub fn new_cards_per_day_for_grade(grade_ordinal: u8) -> u32 {
    match grade_ordinal {
        0..=1 => 5,    // PreK-K
        2..=4 => 10,   // Grade 1-3
        5..=7 => 15,   // Grade 4-6
        8..=10 => 20,  // Grade 7-9
        11..=13 => 25, // Grade 10-12
        _ => 15,
    }
}

/// Break frequency (minutes between breaks) by grade.
pub fn break_interval_for_grade(grade_ordinal: u8) -> u32 {
    match grade_ordinal {
        0..=1 => 10,   // PreK-K: break every 10 min
        2..=4 => 20,   // Grade 1-3
        5..=7 => 30,   // Grade 4-6
        8..=10 => 40,  // Grade 7-9
        11..=13 => 50, // Grade 10-12
        _ => 30,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mastery_threshold_prek() {
        assert_eq!(mastery_threshold_for_grade(0), 900);
    }

    #[test]
    fn test_mastery_threshold_kindergarten() {
        assert_eq!(mastery_threshold_for_grade(1), 900);
    }

    #[test]
    fn test_mastery_threshold_grade3() {
        assert_eq!(mastery_threshold_for_grade(4), 850);
    }

    #[test]
    fn test_mastery_threshold_grade6() {
        assert_eq!(mastery_threshold_for_grade(7), 800);
    }

    #[test]
    fn test_mastery_threshold_grade9() {
        assert_eq!(mastery_threshold_for_grade(10), 800);
    }

    #[test]
    fn test_mastery_threshold_grade12() {
        assert_eq!(mastery_threshold_for_grade(13), 750);
    }

    #[test]
    fn test_mastery_threshold_out_of_range() {
        assert_eq!(mastery_threshold_for_grade(100), 800);
    }

    #[test]
    fn test_zpd_younger_narrower() {
        let (_, max_k) = zpd_range_for_grade(1);
        let (_, max_12) = zpd_range_for_grade(13);
        assert!(max_k < max_12, "Younger students should have narrower ZPD");
    }

    #[test]
    fn test_zpd_prek() {
        assert_eq!(zpd_range_for_grade(0), (50, 100));
    }

    #[test]
    fn test_zpd_middle_school() {
        assert_eq!(zpd_range_for_grade(8), (100, 250));
    }

    #[test]
    fn test_session_length_increases_with_grade() {
        assert!(session_length_for_grade(0) < session_length_for_grade(13));
    }

    #[test]
    fn test_session_length_prek() {
        assert_eq!(session_length_for_grade(0), 15);
    }

    #[test]
    fn test_session_length_high_school() {
        assert_eq!(session_length_for_grade(13), 55);
    }

    #[test]
    fn test_new_cards_increases_with_grade() {
        assert!(new_cards_per_day_for_grade(0) < new_cards_per_day_for_grade(13));
    }

    #[test]
    fn test_new_cards_prek() {
        assert_eq!(new_cards_per_day_for_grade(0), 5);
    }

    #[test]
    fn test_break_interval_increases_with_grade() {
        assert!(break_interval_for_grade(0) < break_interval_for_grade(13));
    }

    #[test]
    fn test_break_interval_prek() {
        assert_eq!(break_interval_for_grade(0), 10);
    }

    #[test]
    fn test_break_interval_out_of_range() {
        assert_eq!(break_interval_for_grade(255), 30);
    }

    #[test]
    fn test_monotonic_session_length() {
        // Session lengths should be non-decreasing across grade bands
        let grades = [0u8, 2, 5, 8, 11];
        for window in grades.windows(2) {
            assert!(
                session_length_for_grade(window[0]) <= session_length_for_grade(window[1]),
                "Session length should not decrease from grade {} to {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_monotonic_new_cards() {
        let grades = [0u8, 2, 5, 8, 11];
        for window in grades.windows(2) {
            assert!(
                new_cards_per_day_for_grade(window[0]) <= new_cards_per_day_for_grade(window[1]),
                "New cards should not decrease from grade {} to {}",
                window[0],
                window[1]
            );
        }
    }
}
