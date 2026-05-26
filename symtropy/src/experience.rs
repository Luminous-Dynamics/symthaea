// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experience Registry — dynamic catalog of Symtropy Engine experiences.
//!
//! Each experience (game, tool, visualization) registers itself with a
//! descriptor. The Nexus launcher reads the registry to build the menu.

use crate::resources::GamePhase;
use bevy::prelude::*;

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
        let mut experiences = vec![ExperienceDescriptor {
            id: "the-room",
            name: "The Room That Remembers You",
            subtitle: "Consciousness survival horror",
            icon_color: [0.3, 0.9, 0.8], // Symtropy cyan
            phase: GamePhase::Loading,
            available: true,
        }];

        // Sol Atlas (feature-gated)
        #[cfg(feature = "atlas")]
        experiences.push(ExperienceDescriptor {
            id: "sol-atlas",
            name: "Sol Atlas",
            subtitle: "Civilizational planetary instrument",
            icon_color: [0.2, 0.6, 1.0], // Deep blue
            phase: GamePhase::GlobeView,
            available: true,
        });

        // City-Scale Governance (Phase 11)
        experiences.push(ExperienceDescriptor {
            id: "city-scale",
            name: "City-Scale Governance",
            subtitle: "300,000 citizens + reactive hydrogeology",
            icon_color: [0.9, 0.3, 0.2], // Toxic red/orange
            phase: GamePhase::CityScale,
            available: true,
        });

        // Muse: Thermodynamic Visualizer
        experiences.push(ExperienceDescriptor {
            id: "muse",
            name: "Muse: Thermodynamic Visualizer",
            subtitle: "Audio-reactive 64K geodesic imagination",
            icon_color: [1.0, 0.5, 0.3], // High-entropy orange
            phase: GamePhase::Muse,
            available: true,
        });

        Self {
            experiences,
            selected: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_room() {
        let reg = ExperienceRegistry::default();
        assert!(!reg.experiences.is_empty());
        assert_eq!(reg.experiences[0].id, "the-room");
        assert!(reg.experiences[0].available);
    }

    #[test]
    fn registry_selection_bounds() {
        let reg = ExperienceRegistry::default();
        assert_eq!(reg.selected, 0);
        assert!(reg.selected < reg.experiences.len());
    }

    #[test]
    fn descriptors_have_names() {
        let reg = ExperienceRegistry::default();
        for exp in &reg.experiences {
            assert!(!exp.name.is_empty());
            assert!(!exp.subtitle.is_empty());
            assert!(!exp.id.is_empty());
        }
    }
}
