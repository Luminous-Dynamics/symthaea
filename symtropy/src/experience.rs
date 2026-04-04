// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experience Registry — dynamic catalog of Symtropy Engine experiences.
//!
//! Each experience (game, tool, visualization) registers itself with a
//! descriptor. The Nexus launcher reads the registry to build the menu.

use bevy::prelude::*;
use crate::resources::GamePhase;

/// Describes a launchable experience.
#[derive(Debug, Clone)]
pub struct ExperienceDescriptor {
    pub id: &'static str,
    pub name: &'static str,
    pub subtitle: &'static str,
    pub icon_color: [f32; 3],
    pub phase: GamePhase,
    pub available: bool,
}

/// Registry of all available experiences.
#[derive(Resource)]
pub struct ExperienceRegistry {
    pub experiences: Vec<ExperienceDescriptor>,
    pub selected: usize,
}

impl Default for ExperienceRegistry {
    fn default() -> Self {
        let mut experiences = vec![
            ExperienceDescriptor {
                id: "the-room",
                name: "The Room That Remembers You",
                subtitle: "Consciousness survival horror",
                icon_color: [0.3, 0.9, 0.8],  // Symtropy cyan
                phase: GamePhase::Loading,
                available: true,
            },
        ];

        // Sol Atlas (feature-gated)
        #[cfg(feature = "atlas")]
        experiences.push(ExperienceDescriptor {
            id: "sol-atlas",
            name: "Sol Atlas",
            subtitle: "Civilizational planetary instrument",
            icon_color: [0.2, 0.6, 1.0],  // Deep blue
            phase: GamePhase::GlobeView,
            available: true,
        });

        Self { experiences, selected: 0 }
    }
}
