// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Circadian entrainment and sleep patterns.
//!
//! Models age-appropriate sleep schedules including total sleep hours,
//! nap frequency, nighttime sleep, and circadian rhythm development.
//! Integrates with the adenosine-based sleep pressure system in
//! NeuromodulatorBath.
//!
//! Science: Iglowstein et al. (2003), Galland et al. (2012),
//! Rivkees (2003) — circadian development.

use serde::{Deserialize, Serialize};

use crate::types::DevelopmentalAge;

/// Age-appropriate sleep schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepSchedule {
    /// Total daily sleep hours (naps + night).
    pub total_hours: f64,
    /// Number of daytime naps.
    pub naps_per_day: u32,
    /// Nighttime sleep hours.
    pub night_hours: f64,
    /// Whether circadian rhythm is established.
    pub circadian_entrained: bool,
    /// Average nap duration in minutes.
    pub avg_nap_minutes: f64,
    /// Whether infant sleeps through the night (6+ hours).
    pub sleeps_through_night: bool,
}

impl SleepSchedule {
    /// Create an age-appropriate sleep schedule.
    ///
    /// Based on Iglowstein et al. (2003) normative data:
    /// - Newborns: 16h total, 4 naps, no circadian
    /// - 3mo: 14h, circadian emerging
    /// - 6mo: 13h, may sleep through night
    /// - 12mo: 12.5h, 2 naps
    /// - 18mo: 12h, 1-2 naps
    /// - 24mo: 11.5h, 1 nap
    /// - 36mo+: 11h, 0-1 naps
    pub fn for_age(age: &DevelopmentalAge) -> Self {
        let months = age.months;

        let total_hours = if months < 3.0 {
            16.0
        } else if months < 6.0 {
            14.0
        } else if months < 12.0 {
            13.0
        } else if months < 24.0 {
            12.0
        } else {
            11.0
        };

        let naps_per_day = if months < 4.0 {
            4
        } else if months < 9.0 {
            3
        } else if months < 18.0 {
            2
        } else if months < 36.0 {
            1
        } else {
            0
        };

        let night_hours = if months < 3.0 {
            8.0
        } else if months < 6.0 {
            10.0
        } else {
            11.0
        };

        let circadian_entrained = months >= 3.0;
        let sleeps_through_night = months >= 6.0;

        // Nap duration: longer naps when fewer per day
        let avg_nap_minutes = if naps_per_day > 0 {
            let nap_hours = total_hours - night_hours;
            (nap_hours / naps_per_day as f64 * 60.0).max(30.0)
        } else {
            0.0
        };

        Self {
            total_hours,
            naps_per_day,
            night_hours,
            circadian_entrained,
            avg_nap_minutes,
            sleeps_through_night,
        }
    }

    /// Nap hours per day.
    pub fn nap_hours(&self) -> f64 {
        self.total_hours - self.night_hours
    }

    /// Awake hours per day.
    pub fn awake_hours(&self) -> f64 {
        24.0 - self.total_hours
    }

    /// Adenosine pressure factor (higher = more sleep pressure).
    /// Returns a value proportional to how sleep-deprived the schedule is relative to needs.
    pub fn sleep_pressure_factor(&self, actual_sleep_hours: f64) -> f64 {
        let deficit = (self.total_hours - actual_sleep_hours).max(0.0);
        (deficit / self.total_hours).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newborn_sleep() {
        let schedule = SleepSchedule::for_age(&DevelopmentalAge::new(1.0));
        assert!((schedule.total_hours - 16.0).abs() < 1e-10);
        assert_eq!(schedule.naps_per_day, 4);
        assert!(!schedule.circadian_entrained);
        assert!(!schedule.sleeps_through_night);
    }

    #[test]
    fn test_three_month_circadian() {
        let schedule = SleepSchedule::for_age(&DevelopmentalAge::new(4.0));
        assert!(schedule.circadian_entrained);
        assert!((schedule.total_hours - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_six_month_sleep() {
        let schedule = SleepSchedule::for_age(&DevelopmentalAge::new(7.0));
        assert!((schedule.total_hours - 13.0).abs() < 1e-10);
        assert!(schedule.sleeps_through_night);
        assert_eq!(schedule.naps_per_day, 3);
    }

    #[test]
    fn test_one_year_sleep() {
        let schedule = SleepSchedule::for_age(&DevelopmentalAge::new(13.0));
        assert!((schedule.total_hours - 12.0).abs() < 1e-10);
        assert_eq!(schedule.naps_per_day, 2);
    }

    #[test]
    fn test_two_year_sleep() {
        let schedule = SleepSchedule::for_age(&DevelopmentalAge::new(25.0));
        assert!((schedule.total_hours - 11.0).abs() < 1e-10);
        assert_eq!(schedule.naps_per_day, 1);
    }

    #[test]
    fn test_three_year_no_naps() {
        let schedule = SleepSchedule::for_age(&DevelopmentalAge::new(40.0));
        assert_eq!(schedule.naps_per_day, 0);
        assert!((schedule.avg_nap_minutes - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_hours_decrease_with_age() {
        let s1 = SleepSchedule::for_age(&DevelopmentalAge::new(1.0));
        let s6 = SleepSchedule::for_age(&DevelopmentalAge::new(7.0));
        let s24 = SleepSchedule::for_age(&DevelopmentalAge::new(25.0));

        assert!(s1.total_hours > s6.total_hours);
        assert!(s6.total_hours > s24.total_hours);
    }

    #[test]
    fn test_naps_decrease_with_age() {
        let s1 = SleepSchedule::for_age(&DevelopmentalAge::new(1.0));
        let s12 = SleepSchedule::for_age(&DevelopmentalAge::new(13.0));
        let s36 = SleepSchedule::for_age(&DevelopmentalAge::new(40.0));

        assert!(s1.naps_per_day > s12.naps_per_day);
        assert!(s12.naps_per_day > s36.naps_per_day);
    }

    #[test]
    fn test_awake_hours_increase() {
        let s1 = SleepSchedule::for_age(&DevelopmentalAge::new(1.0));
        let s24 = SleepSchedule::for_age(&DevelopmentalAge::new(25.0));

        assert!(s24.awake_hours() > s1.awake_hours());
    }

    #[test]
    fn test_sleep_pressure_zero_when_met() {
        let schedule = SleepSchedule::for_age(&DevelopmentalAge::new(12.0));
        let pressure = schedule.sleep_pressure_factor(schedule.total_hours);
        assert!((pressure - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sleep_pressure_high_when_deprived() {
        let schedule = SleepSchedule::for_age(&DevelopmentalAge::new(12.0));
        let pressure = schedule.sleep_pressure_factor(6.0); // Only 6h of needed 13h
        assert!(pressure > 0.4, "Should have high pressure, got {pressure}");
    }

    #[test]
    fn test_nap_duration_positive() {
        for months in [1, 6, 12, 18, 24] {
            let schedule = SleepSchedule::for_age(&DevelopmentalAge::new(months as f64));
            if schedule.naps_per_day > 0 {
                assert!(
                    schedule.avg_nap_minutes > 0.0,
                    "Nap duration should be positive at {}mo",
                    months
                );
            }
        }
    }
}
