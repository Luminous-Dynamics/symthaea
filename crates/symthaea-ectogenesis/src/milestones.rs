// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Developmental milestone tracking with verification.
//!
//! Tracks eight key milestones from neural tube formation (week 4)
//! through full term (week 37+), enforcing chronological ordering
//! and providing on-track/delayed status.

use crate::types::{DevelopmentalMilestone, FetalMetrics, GestationalWeek};

/// Tracks achieved developmental milestones in chronological order.
#[derive(Debug, Clone)]
pub struct MilestoneTracker {
    /// Achieved milestones with the week they were verified.
    pub achieved: Vec<(DevelopmentalMilestone, GestationalWeek)>,
}

impl MilestoneTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self {
            achieved: Vec::new(),
        }
    }

    /// Check whether a milestone should be achieved at the current week
    /// based on fetal metrics. Returns the milestone if newly achieved.
    ///
    /// Milestones are checked in order; a milestone can only be achieved
    /// if all prior milestones have been met.
    pub fn check_milestone(
        &mut self,
        week: GestationalWeek,
        metrics: &FetalMetrics,
    ) -> Option<DevelopmentalMilestone> {
        let next = self.next_expected()?;
        let expected_week = next.expected_week();

        // Only check if we've reached the expected week
        if week < expected_week {
            return None;
        }

        // Verify milestone criteria from metrics
        let achieved = match next {
            DevelopmentalMilestone::NeuralTubeFormation => {
                week.0 >= 4 && metrics.brain_activity_score >= 0.0
            }
            DevelopmentalMilestone::HeartbeatDetected => metrics.heart_rate_bpm > 50.0,
            DevelopmentalMilestone::LimbBudFormation => {
                week.0 >= 8 && metrics.crown_rump_length_mm > 10.0
            }
            DevelopmentalMilestone::OrganogenesisComplete => {
                week.0 >= 12 && metrics.weight_grams > 5.0
            }
            DevelopmentalMilestone::Quickening => metrics.movement_score > 0.1,
            DevelopmentalMilestone::Viability => week.0 >= 24 && metrics.weight_grams > 400.0,
            DevelopmentalMilestone::LungMaturation => metrics.lung_maturity > 0.5,
            DevelopmentalMilestone::FullTerm => week.0 >= 37 && metrics.lung_maturity > 0.7,
        };

        if achieved {
            self.achieved.push((next, week));
            Some(next)
        } else {
            None
        }
    }

    /// Whether the fetus is on track (all expected milestones achieved).
    ///
    /// A fetus is on track if every milestone whose expected week has
    /// passed has actually been achieved.
    pub fn is_on_track(&self, current_week: GestationalWeek) -> bool {
        for milestone in DevelopmentalMilestone::all() {
            if milestone.expected_week() <= current_week {
                let is_achieved = self.achieved.iter().any(|(m, _)| *m == milestone);
                if !is_achieved {
                    return false;
                }
            }
        }
        true
    }

    /// The next milestone that hasn't been achieved yet.
    pub fn next_expected(&self) -> Option<DevelopmentalMilestone> {
        let all = DevelopmentalMilestone::all();
        for milestone in all {
            let already = self.achieved.iter().any(|(m, _)| *m == milestone);
            if !already {
                return Some(milestone);
            }
        }
        None
    }

    /// Whether all eight milestones have been achieved.
    pub fn all_achieved(&self) -> bool {
        self.achieved.len() >= DevelopmentalMilestone::all().len()
    }

    /// Number of milestones achieved so far.
    pub fn count(&self) -> usize {
        self.achieved.len()
    }
}

impl Default for MilestoneTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics_at(week: u32) -> FetalMetrics {
        FetalMetrics::normative(GestationalWeek::new(week))
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = MilestoneTracker::new();
        assert!(!tracker.all_achieved());
        assert_eq!(tracker.count(), 0);
        assert_eq!(
            tracker.next_expected(),
            Some(DevelopmentalMilestone::NeuralTubeFormation)
        );
    }

    #[test]
    fn test_milestone_achieved_at_correct_week() {
        let mut tracker = MilestoneTracker::new();
        let week = GestationalWeek::new(4);
        let metrics = metrics_at(4);
        let result = tracker.check_milestone(week, &metrics);
        assert_eq!(result, Some(DevelopmentalMilestone::NeuralTubeFormation));
        assert_eq!(tracker.count(), 1);
    }

    #[test]
    fn test_milestone_not_achieved_too_early() {
        let mut tracker = MilestoneTracker::new();
        let week = GestationalWeek::new(2);
        let metrics = metrics_at(2);
        let result = tracker.check_milestone(week, &metrics);
        assert_eq!(result, None);
    }

    #[test]
    fn test_milestone_order_enforced() {
        let mut tracker = MilestoneTracker::new();
        // Try to achieve HeartbeatDetected without NeuralTubeFormation
        let week = GestationalWeek::new(6);
        let metrics = metrics_at(6);
        let result = tracker.check_milestone(week, &metrics);
        // Should achieve NeuralTubeFormation first (since week 6 > 4)
        assert_eq!(result, Some(DevelopmentalMilestone::NeuralTubeFormation));
        // Now heartbeat should be next
        let result2 = tracker.check_milestone(week, &metrics);
        assert_eq!(result2, Some(DevelopmentalMilestone::HeartbeatDetected));
    }

    #[test]
    fn test_full_gestation_milestones() {
        let mut tracker = MilestoneTracker::new();
        // Walk through entire gestation
        for w in 0..=40 {
            let week = GestationalWeek::new(w);
            let metrics = metrics_at(w);
            // Keep checking until no more milestones at this week
            while tracker.check_milestone(week, &metrics).is_some() {}
        }
        assert!(
            tracker.all_achieved(),
            "All milestones should be achieved after full gestation. Got {} of 8",
            tracker.count()
        );
    }

    #[test]
    fn test_is_on_track() {
        let mut tracker = MilestoneTracker::new();
        // At week 0, should be on track (no milestones expected yet)
        assert!(tracker.is_on_track(GestationalWeek::new(0)));

        // At week 5, should NOT be on track (NeuralTubeFormation expected at 4)
        assert!(!tracker.is_on_track(GestationalWeek::new(5)));

        // Achieve it
        let metrics = metrics_at(5);
        tracker.check_milestone(GestationalWeek::new(5), &metrics);
        assert!(tracker.is_on_track(GestationalWeek::new(5)));
    }

    #[test]
    fn test_next_expected_after_partial() {
        let mut tracker = MilestoneTracker::new();
        let metrics = metrics_at(8);
        let week = GestationalWeek::new(8);
        // Achieve first three
        tracker.check_milestone(week, &metrics);
        tracker.check_milestone(week, &metrics);
        tracker.check_milestone(week, &metrics);
        assert_eq!(
            tracker.next_expected(),
            Some(DevelopmentalMilestone::OrganogenesisComplete)
        );
    }
}
