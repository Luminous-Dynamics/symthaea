// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Active sequencing agent using Free Energy Principle.
//!
//! The FEP sequencing agent decides how to allocate sequencing resources
//! to minimize uncertainty. It observes quality scores and coverage, then
//! selects actions that most reduce expected free energy (uncertainty).

use serde::{Deserialize, Serialize};

/// Actions available to the sequencing agent.
///
/// Each variant represents a distinct resource allocation decision that minimizes expected free energy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SequencingAction {
    /// Re-sequence a high-uncertainty region.
    RereadRegion,
    /// Add more reads to a low-coverage region.
    IncreaseDepth,
    /// Change sequencing chemistry for damaged regions.
    SwitchChemistry,
    /// Extend read length for better overlap.
    ExtendRead,
    /// Region is irreparable; skip it.
    SkipRegion,
    /// Current sequencing is adequate; do nothing.
    MaintainCourse,
}

/// Active sequencing agent that uses FEP-inspired uncertainty minimization
/// to guide sequencing decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequencingFepAgent {
    /// Available actions.
    pub actions: [SequencingAction; 6],
    /// Uncertainty threshold below which MaintainCourse is chosen.
    pub uncertainty_threshold: f64,
    /// Target coverage depth.
    pub coverage_target: u32,
}

impl Default for SequencingFepAgent {
    fn default() -> Self {
        Self {
            actions: [
                SequencingAction::RereadRegion,
                SequencingAction::IncreaseDepth,
                SequencingAction::SwitchChemistry,
                SequencingAction::ExtendRead,
                SequencingAction::SkipRegion,
                SequencingAction::MaintainCourse,
            ],
            uncertainty_threshold: 0.1,
            coverage_target: 30,
        }
    }
}

impl SequencingFepAgent {
    /// Create a new FEP sequencing agent with custom parameters.
    pub fn new(uncertainty_threshold: f64, coverage_target: u32) -> Self {
        Self {
            uncertainty_threshold,
            coverage_target,
            ..Default::default()
        }
    }

    /// Select the best action to reduce uncertainty.
    ///
    /// The agent examines quality scores and coverage to identify the most
    /// pressing issue, then selects the action that would most reduce
    /// expected free energy.
    pub fn select_action(&self, quality_scores: &[f64], coverage: &[u32]) -> SequencingAction {
        if quality_scores.is_empty() || coverage.is_empty() {
            return SequencingAction::MaintainCourse;
        }

        let uncertainty = Self::compute_uncertainty(quality_scores);

        // If overall uncertainty is low, maintain course
        if uncertainty < self.uncertainty_threshold {
            return SequencingAction::MaintainCourse;
        }

        // Find the most problematic region
        let min_coverage = coverage.iter().copied().min().unwrap_or(0);
        let min_quality = quality_scores.iter().copied().fold(f64::INFINITY, f64::min);

        // Fraction of positions below coverage target
        let low_coverage_fraction = coverage
            .iter()
            .filter(|&&c| c < self.coverage_target)
            .count() as f64
            / coverage.len() as f64;

        // Fraction of positions with low quality
        let low_quality_fraction = quality_scores.iter().filter(|&&q| q < 0.5).count() as f64
            / quality_scores.len() as f64;

        // Decision logic: prioritize the most impactful action
        if min_coverage == 0 {
            // No coverage at all: need more reads
            return SequencingAction::IncreaseDepth;
        }

        if min_quality < 0.2 {
            // Very low quality: re-read the problematic region
            return SequencingAction::RereadRegion;
        }

        if low_coverage_fraction > 0.5 {
            // More than half the assembly has insufficient coverage
            return SequencingAction::IncreaseDepth;
        }

        if low_quality_fraction > 0.3 {
            // Significant portion has low quality (possibly damaged reads)
            return SequencingAction::SwitchChemistry;
        }

        if min_quality < 0.5 && low_coverage_fraction < 0.1 {
            // Good coverage but localized quality issues
            return SequencingAction::RereadRegion;
        }

        // If coverage is good but reads are short, extend them
        let mean_coverage: f64 =
            coverage.iter().map(|&c| c as f64).sum::<f64>() / coverage.len() as f64;
        if mean_coverage >= self.coverage_target as f64
            && uncertainty > self.uncertainty_threshold * 2.0
        {
            return SequencingAction::ExtendRead;
        }

        // Default: try to add more reads
        SequencingAction::IncreaseDepth
    }

    /// Compute overall uncertainty from quality scores.
    ///
    /// Uncertainty = 1 - mean(quality), representing the average information deficit.
    /// This is a simplified proxy for the entropy of the quality distribution.
    pub fn compute_uncertainty(quality_scores: &[f64]) -> f64 {
        if quality_scores.is_empty() {
            return 1.0;
        }

        let mean_quality = quality_scores.iter().sum::<f64>() / quality_scores.len() as f64;
        (1.0 - mean_quality).clamp(0.0, 1.0)
    }

    /// Compute per-region uncertainty for targeted action.
    ///
    /// Returns uncertainty for each position, enabling the agent to
    /// focus sequencing on the most uncertain regions.
    pub fn per_region_uncertainty(
        quality_scores: &[f64],
        coverage: &[u32],
        coverage_target: u32,
    ) -> Vec<f64> {
        quality_scores
            .iter()
            .zip(coverage.iter())
            .map(|(&q, &c)| {
                let quality_uncertainty = 1.0 - q;
                let coverage_factor = if c >= coverage_target {
                    0.0
                } else {
                    1.0 - (c as f64 / coverage_target as f64)
                };
                // Combine both signals
                (quality_uncertainty * 0.6 + coverage_factor * 0.4).clamp(0.0, 1.0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maintain_course_when_quality_high() {
        let agent = SequencingFepAgent::default();
        let quality = vec![0.99; 100];
        let coverage = vec![50; 100];

        let action = agent.select_action(&quality, &coverage);
        assert_eq!(
            action,
            SequencingAction::MaintainCourse,
            "High quality + high coverage should maintain course"
        );
    }

    #[test]
    fn test_increase_depth_when_no_coverage() {
        let agent = SequencingFepAgent::default();
        let quality = vec![0.5; 100];
        let mut coverage = vec![10; 100];
        coverage[50] = 0; // One position with no coverage

        let action = agent.select_action(&quality, &coverage);
        assert_eq!(
            action,
            SequencingAction::IncreaseDepth,
            "Zero coverage should trigger IncreaseDepth"
        );
    }

    #[test]
    fn test_reread_when_low_quality() {
        let agent = SequencingFepAgent::default();
        let mut quality = vec![0.9; 100];
        quality[50] = 0.1; // One very low quality position
        let coverage = vec![30; 100];

        let action = agent.select_action(&quality, &coverage);
        assert_eq!(
            action,
            SequencingAction::RereadRegion,
            "Very low quality should trigger RereadRegion"
        );
    }

    #[test]
    fn test_increase_depth_when_low_coverage() {
        let agent = SequencingFepAgent::default();
        let quality = vec![0.7; 100];
        let coverage = vec![5; 100]; // All below target of 30

        let action = agent.select_action(&quality, &coverage);
        assert_eq!(
            action,
            SequencingAction::IncreaseDepth,
            "Low coverage everywhere should trigger IncreaseDepth"
        );
    }

    #[test]
    fn test_switch_chemistry_when_many_low_quality() {
        let agent = SequencingFepAgent::default();
        // 40% of positions have low quality (> 30% threshold)
        let mut quality = vec![0.9; 100];
        for i in 0..40 {
            quality[i] = 0.3;
        }
        let coverage = vec![30; 100]; // Coverage is fine

        let action = agent.select_action(&quality, &coverage);
        assert_eq!(
            action,
            SequencingAction::SwitchChemistry,
            "Many low-quality positions should trigger SwitchChemistry"
        );
    }

    #[test]
    fn test_compute_uncertainty_high_quality() {
        let quality = vec![0.95; 100];
        let uncertainty = SequencingFepAgent::compute_uncertainty(&quality);
        assert!(
            uncertainty < 0.1,
            "High quality should have low uncertainty: {}",
            uncertainty
        );
    }

    #[test]
    fn test_compute_uncertainty_low_quality() {
        let quality = vec![0.1; 100];
        let uncertainty = SequencingFepAgent::compute_uncertainty(&quality);
        assert!(
            uncertainty > 0.8,
            "Low quality should have high uncertainty: {}",
            uncertainty
        );
    }

    #[test]
    fn test_compute_uncertainty_empty() {
        let uncertainty = SequencingFepAgent::compute_uncertainty(&[]);
        assert_eq!(uncertainty, 1.0);
    }

    #[test]
    fn test_select_action_empty_input() {
        let agent = SequencingFepAgent::default();
        let action = agent.select_action(&[], &[]);
        assert_eq!(action, SequencingAction::MaintainCourse);
    }

    #[test]
    fn test_per_region_uncertainty() {
        let quality = vec![0.9, 0.1, 0.5];
        let coverage = vec![30, 5, 30];

        let uncertainties = SequencingFepAgent::per_region_uncertainty(&quality, &coverage, 30);

        assert_eq!(uncertainties.len(), 3);
        // Position 0: high quality, good coverage -> low uncertainty
        assert!(uncertainties[0] < 0.2);
        // Position 1: low quality, low coverage -> high uncertainty
        assert!(uncertainties[1] > 0.5);
    }

    #[test]
    fn test_all_actions_representable() {
        let agent = SequencingFepAgent::default();
        // Verify all 6 actions are in the action set
        assert_eq!(agent.actions.len(), 6);
        assert!(agent.actions.contains(&SequencingAction::RereadRegion));
        assert!(agent.actions.contains(&SequencingAction::IncreaseDepth));
        assert!(agent.actions.contains(&SequencingAction::SwitchChemistry));
        assert!(agent.actions.contains(&SequencingAction::ExtendRead));
        assert!(agent.actions.contains(&SequencingAction::SkipRegion));
        assert!(agent.actions.contains(&SequencingAction::MaintainCourse));
    }
}
