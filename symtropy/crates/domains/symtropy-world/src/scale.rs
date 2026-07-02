// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! World scale levels for macro/micro zoom.

/// Scale levels in the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldScale {
    /// Walk-around: 60fps, tile-level physics, individual agents.
    /// Player sees individual NPCs, items, rooms.
    Micro,

    /// City/station view: ~10fps, building-level, population groups.
    /// Player sees districts, infrastructure, aggregate flows.
    Meso,

    /// Civilization: 1 tick/5s = 1 month, world-level, demographics.
    /// Player sees worlds, trade routes, epoch-level trends.
    Macro,
}

impl WorldScale {
    /// Target frames per second for this scale level.
    pub fn target_fps(&self) -> f64 {
        match self {
            Self::Micro => 60.0,
            Self::Meso => 10.0,
            Self::Macro => 2.0,
        }
    }

    /// Whether individual entities are visible at this scale.
    pub fn shows_entities(&self) -> bool {
        matches!(self, Self::Micro)
    }

    /// Whether aggregate statistics are visible at this scale.
    pub fn shows_aggregates(&self) -> bool {
        matches!(self, Self::Meso | Self::Macro)
    }

    /// Whether world-level overlays (trade routes, migration) are visible.
    pub fn shows_world_overlays(&self) -> bool {
        matches!(self, Self::Macro)
    }
}
