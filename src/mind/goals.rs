// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Goal management for the Continuous Mind.

use super::ContinuousMind;

impl ContinuousMind {
    /// Process active goals, advancing progress based on consciousness level.
    pub(crate) fn process_goals(&mut self) {
        for goal in self.goals.iter_mut() {
            if !goal.is_active {
                continue;
            }

            // Simulate progress based on consciousness and effort
            let progress_increment = self.state.consciousness_level as f32 * 0.01;
            goal.progress = (goal.progress + progress_increment).min(1.0);

            if goal.progress >= 1.0 {
                goal.is_active = false;
                self.stats.goals_completed += 1;
            }
        }

        // Update active goals list
        self.state.active_goals = self
            .goals
            .iter()
            .filter(|g| g.is_active)
            .map(|g| g.id.clone())
            .collect();
    }
}
