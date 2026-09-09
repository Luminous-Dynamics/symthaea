// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::spatial_goal::HumanoidSpatiallyBoundSkillPermit;

/// Redacted diagnostics for the opaque spatial permit. The implementation uses
/// only public read-only accessors and does not expose construction internals.
impl std::fmt::Debug for HumanoidSpatiallyBoundSkillPermit<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidSpatiallyBoundSkillPermit")
            .field("validation_epoch", &self.semantic().epoch())
            .field("role", &self.role())
            .field("goal_id", &self.goal().goal_id)
            .field("goal_kind", &self.goal().kind)
            .field("hand", &self.goal().hand)
            .field("target_root_m", &self.target_root_m())
            .field("workspace_utilization_sq", &self.workspace_utilization_sq())
            .field("selected_hand_actuation", &self.selected_hand_actuation())
            .finish_non_exhaustive()
    }
}
